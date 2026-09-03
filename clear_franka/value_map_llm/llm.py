"""Small provider-agnostic LLM backend for value-map synthesis.

Separate from `voxposer.lmp.LLMBackend` because that one hard-codes a
CALVIN-specific system prompt at module scope (`voxposer.lmp.SYSTEM_PROMPT`),
and the frame convention it states — "x is left to right, y is front to back" —
is wrong for `fr3_link0`. Responses are cached on disk with LangSteer's
`voxposer.llm_cache.DiskCache`, so re-running synthesis with an unchanged scene
and task costs nothing.
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-opus-5"

SYSTEM_PROMPT = (
    "You write Python that builds 3D value maps steering a robot arm in a "
    "tabletop scene. Complete the code after the final query, following the "
    "patterns in the context exactly. Do not write import statements, do not "
    "repeat the query, and do not explain anything outside of code comments.\n"
    "The world frame is the Franka base frame (fr3_link0), metres: +x points "
    "away from the robot base out across the table, +y points to the robot's "
    "left, +z points up. 'In front of' an object means the -x side (the side "
    "facing the robot); 'above' means +z."
)


class LLMBackend:
    """One cached chat completion per call, provider chosen by config."""

    def __init__(
        self,
        provider: str = DEFAULT_PROVIDER,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 8000,
        effort: str = "low",
        cache_dir: str = "cache/value_map_llm",
        load_cache: bool = True,
        system_prompt: str = SYSTEM_PROMPT,
        max_retries: int = 3,
    ) -> None:
        from voxposer.llm_cache import DiskCache

        self._provider = provider
        self._model = model
        self._max_tokens = int(max_tokens)
        self._effort = effort
        self._system = system_prompt
        self._max_retries = int(max_retries)
        self._cache = DiskCache(cache_dir=cache_dir, load_cache=load_cache)
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        if self._provider == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic()
        elif self._provider == "openai":
            import openai

            self._client = openai.OpenAI()
        else:
            raise ValueError(f"Unknown LLM provider: {self._provider}")
        return self._client

    def generate(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        image: Optional[np.ndarray] = None,
    ) -> str:
        stop = list(stop or [])
        image_b64 = _encode_image_b64(image) if image is not None else None
        cache_key = {
            "provider": self._provider,
            "model": self._model,
            "prompt": prompt,
            "system": self._system,
            "max_tokens": self._max_tokens,
            "effort": self._effort,
            "image_sha256": (
                hashlib.sha256(image_b64.encode("ascii")).hexdigest()
                if image_b64 is not None
                else None
            ),
        }
        if cache_key in self._cache:
            logger.debug("LLM cache hit (%s)", self._model)
            return self._cache[cache_key]

        client = self._get_client()
        start = time.time()
        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                if self._provider == "anthropic":
                    text = self._call_anthropic(client, prompt, stop, image_b64)
                else:
                    text = self._call_openai(client, prompt, stop, image_b64)
                break
            except Exception as e:  # noqa: BLE001 — retried, then re-raised
                # Both SDKs put the HTTP status on the exception. A bad key,
                # a revoked key or an unknown model will never succeed on a
                # retry, so surface those immediately instead of sleeping
                # through the backoff.
                status = getattr(e, "status_code", None)
                if status in (400, 401, 403, 404):
                    raise
                # A 429 is normally transient, but an exhausted quota is not —
                # retrying just buries the billing message under a backoff.
                if status == 429 and "insufficient_quota" in str(e):
                    raise
                last_error = e
                wait = 2 ** attempt
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s. Retrying in %ds",
                    attempt + 1, self._max_retries, e, wait,
                )
                time.sleep(wait)
        else:
            raise RuntimeError(
                f"LLM call to {self._provider}/{self._model} failed after "
                f"{self._max_retries} attempts: {last_error}"
            ) from last_error

        logger.info(
            "LLM call (%s/%s) took %.2fs", self._provider, self._model,
            time.time() - start,
        )
        self._cache[cache_key] = text
        return text

    # ------------------------------------------------------------------

    def _call_anthropic(
        self, client, prompt: str, stop: list[str], image_b64: Optional[str] = None
    ) -> str:
        # No `temperature`: sampling parameters are rejected on Opus 5 / Sonnet 5.
        # Adaptive thinking at low effort — these are short, highly patterned
        # emissions, and the in-context examples do most of the work.
        content: object = prompt
        if image_b64 is not None:
            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": prompt},
            ]
        response = client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=self._system,
            thinking={"type": "adaptive"},
            output_config={"effort": self._effort},
            stop_sequences=stop or None,
            messages=[{"role": "user", "content": content}],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"model refused: {response.stop_details}")
        text = "".join(b.text for b in response.content if b.type == "text")
        return _strip_fences(text)

    def _call_openai(
        self, client, prompt: str, stop: list[str], image_b64: Optional[str] = None
    ) -> str:
        user_content: object = prompt
        if image_b64 is not None:
            user_content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
                {"type": "text", "text": prompt},
            ]
        kwargs: dict = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._system},
                {"role": "user", "content": user_content},
            ],
        }
        # The gpt-5 family rejects `max_tokens` and `stop`; LangSteer's
        # voxposer backend hit the same wall (see voxposer/lmp.py::_call_openai).
        if self._model.startswith("gpt-5"):
            kwargs["max_completion_tokens"] = self._max_tokens
        else:
            kwargs["max_tokens"] = self._max_tokens
            if stop:
                kwargs["stop"] = stop
        response = client.chat.completions.create(**kwargs)
        text = _strip_fences(response.choices[0].message.content or "")
        if self._model.startswith("gpt-5"):
            for marker in stop:
                idx = text.find(marker)
                if idx >= 0:
                    text = text[:idx].rstrip()
        return text


def _strip_fences(text: str) -> str:
    return text.replace("```python", "").replace("```", "").strip()


def _encode_image_b64(image: np.ndarray) -> str:
    """(H, W, 3) uint8 RGB -> base64-encoded PNG bytes."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
