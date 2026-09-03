"""SAM 3 text-prompted instance segmentation.

Lifted from `hardware_pipeline_example_offline.py`. The model import is lazy so
this module (and everything that imports `clear_franka.perception`) loads on a
machine without the SAM 3 package or checkpoint.

`bpe_path` is passed explicitly because the sam3 wheel does not ship the text
encoder's vocabulary: its default resolves to `<site-packages>/assets/
bpe_simple_vocab_16e6.txt.gz`, a path that only exists in a source checkout of
the repo, so a pip install raises FileNotFoundError before any weights load.
The file is the standard CLIP BPE vocab; we keep our own copy next to the
checkpoint rather than reaching into a sibling package's install directory.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from clear_franka.perception.types import Detection

logger = logging.getLogger(__name__)

DEFAULT_CONFIDENCE = 0.5


class Sam3Segmenter:
    """Segments one image against a fixed list of text prompts."""

    def __init__(
        self,
        checkpoint: str,
        classes: list[str],
        confidence: float = DEFAULT_CONFIDENCE,
        bpe_path: str | None = None,
        load_from_hf: bool = False,
    ) -> None:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        if bpe_path is not None and not Path(bpe_path).is_file():
            raise FileNotFoundError(
                f"SAM 3 BPE vocabulary not found at {bpe_path}. It is the "
                "standard CLIP vocab (bpe_simple_vocab_16e6.txt.gz) and is not "
                "shipped in the sam3 wheel; set perception.bpe_path to a copy."
            )
        model = build_sam3_image_model(
            checkpoint_path=checkpoint,
            load_from_HF=load_from_hf,
            bpe_path=bpe_path,
        )
        self.processor = Sam3Processor(model, confidence_threshold=confidence)
        self.classes = list(classes)
        logger.info(
            "Sam3Segmenter ready: %d prompt(s) %s, confidence=%.2f",
            len(self.classes), self.classes, confidence,
        )

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        """Run every text prompt against one RGB frame."""
        import torch
        from PIL import Image

        detections: list[Detection] = []
        # The vision backbone runs once in set_image(); each prompt below only
        # re-runs the text encoder and grounding head.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.processor.set_image(Image.fromarray(rgb))

            for label in self.classes:
                out = self.processor.set_text_prompt(prompt=label, state=state)
                # set_text_prompt overwrites `state` in place, so copy out now.
                masks = out["masks"][:, 0].cpu().numpy().astype(bool)
                scores = out["scores"].float().cpu().numpy()
                detections += [
                    Detection(label=label, score=float(s), mask=m)
                    for m, s in zip(masks, scores)
                ]
        return detections
