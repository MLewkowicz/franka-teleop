"""Prompt assembly + sandboxed execution of LLM-generated value-map code.

The two-level structure is VoxPoser's, trimmed to what hardware needs:

    PlannerLMP        task + scene block            -> [stage dict, ...]
    MapLMP            one affordance/avoidance query -> (M, M, M) voxel grid

Prompts live in `clear_franka/value_map_llm/prompts/franka/`; they are short
because the scene has no simulator state to disambiguate and the deploy sets
its own grasp/place primitives.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).parent / "prompts"
STOP = ["# Query:"]


def load_prompt(name: str, env: str = "franka") -> str:
    # encoding is explicit: the prompts contain non-ASCII punctuation, and a
    # machine whose locale resolves to ASCII would otherwise fail to read them.
    return (PROMPT_DIR / env / f"{name}.txt").read_text(encoding="utf-8").strip()


def _exec_safe(code: str, gvars: dict, lvars: dict) -> None:
    """Run generated code with imports and dunder access banned."""
    from voxposer.lmp import exec_safe

    exec_safe(code, gvars, lvars)


class MapLMP:
    """Turns one natural-language query into an affordance or avoidance grid."""

    def __init__(self, name: str, backend, interface, scene_block: str) -> None:
        self._name = name
        self._backend = backend
        self._interface = interface
        self._scene_block = scene_block
        self._base_prompt = load_prompt(f"{name}_prompt")

    def __call__(self, query: str) -> np.ndarray:
        prompt = (
            f"{self._base_prompt}\n\n{self._scene_block}\n# Query: {query}."
        )
        code = self._backend.generate(prompt, stop=STOP)
        logger.info('[%s] "%s" ->\n%s', self._name, query, code)

        gvars = {
            "np": np,
            "parse_query_obj": self._interface.detect,
            "detect": self._interface.detect,
            "cm2index": self._interface.cm2index,
            "set_voxel_by_radius": self._interface.set_voxel_by_radius,
            "set_voxel_by_box": self._interface.set_voxel_by_box,
            "get_empty_affordance_map": self._interface.get_empty_affordance_map,
            "get_empty_avoidance_map": self._interface.get_empty_avoidance_map,
        }
        lvars: dict = {}
        _exec_safe(code, gvars, lvars)
        if "ret_val" not in lvars:
            raise ValueError(
                f"[{self._name}] generated code did not assign ret_val:\n{code}"
            )
        grid = np.asarray(lvars["ret_val"], dtype=np.float32)
        if grid.ndim != 3:
            raise ValueError(
                f"[{self._name}] expected a 3D voxel grid, got shape {grid.shape}"
            )
        return grid


class PlannerLMP:
    """Turns the task + scene (+ optional workspace photo) into steering stages."""

    def __init__(
        self, backend, scene_block: str, image: Optional[np.ndarray] = None
    ) -> None:
        self._backend = backend
        self._scene_block = scene_block
        self._image = image
        self._base_prompt = load_prompt("planner_prompt")

    def __call__(self, task: str) -> list[dict]:
        prompt = f"{self._base_prompt}\n\n{self._scene_block}\n# Query: {task}."
        code = self._backend.generate(prompt, stop=STOP, image=self._image)
        logger.info('[planner] "%s" ->\n%s', task, code)
        return parse_stages(code)


def parse_stages(code: str) -> list[dict]:
    """Extract and validate the planner's `ret_val = {...}` literal.

    The planner emits ONE stage — the destination the object ends up at — so the
    expected shape is a bare dict. A one-element list is accepted too, since
    that is the obvious thing for a model to reach for and costs nothing to
    tolerate; anything longer is rejected rather than quietly steering the arm
    through targets nobody asked for.

    Returns a one-element list, so callers keep working with a sequence.

    Parsed with `ast.literal_eval` rather than exec: the planner emits pure
    data, so nothing here needs to run.
    """
    text = code.strip()
    marker = "ret_val"
    idx = text.find(marker)
    if idx < 0:
        raise ValueError(f"planner did not emit ret_val:\n{code}")
    body = text[idx + len(marker):].lstrip()
    if not body.startswith("="):
        raise ValueError(f"planner emitted a malformed ret_val:\n{code}")
    try:
        raw = ast.literal_eval(body[1:].strip())
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"planner emitted unparseable stages:\n{code}") from e

    if isinstance(raw, dict):
        entry = raw
    elif isinstance(raw, (list, tuple)):
        if len(raw) != 1:
            raise ValueError(
                f"planner must emit exactly one stage, got {len(raw)}. The "
                "steering takes a single destination; re-run with --no-cache "
                f"after tightening the prompt.\n{code}"
            )
        entry = raw[0]
    else:
        raise ValueError(f"planner returned {type(raw).__name__}, not a stage:\n{code}")

    if not isinstance(entry, dict) or "affordance" not in entry:
        raise ValueError(
            f"the stage must be a dict with an 'affordance' key, got {entry!r}"
        )
    return [{
        "affordance": str(entry["affordance"]),
        "avoidance": (
            None if entry.get("avoidance") in (None, "", "None")
            else str(entry["avoidance"])
        ),
        "arrival_radius_cm": float(entry.get("arrival_radius_cm", 8.0)),
        "label": str(entry.get("label", "place")),
    }]
