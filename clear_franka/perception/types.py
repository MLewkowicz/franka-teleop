"""Dataclasses shared by the perception pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    """One SAM 3 instance mask for one text prompt, in image space."""

    label: str          # the text prompt that produced it, e.g. "wine rack"
    score: float        # detection confidence in [0, 1]
    mask: np.ndarray    # (H, W) bool


@dataclass
class SceneBox:
    """An axis-aligned box in the Franka base frame (`fr3_link0`), metres.

    `name` is unique within a scene (the label, suffixed `_2`, `_3`, ... when a
    prompt returns several instances); `label` is the raw text prompt. Both are
    kept because the LLM refers to objects by NAME while the operator thinks in
    prompts.
    """

    name: str
    label: str
    center: np.ndarray  # (3,)
    size: np.ndarray    # (3,) full extents, not half
    score: float = 1.0
    n_points: int = 0

    @property
    def aabb(self) -> np.ndarray:
        """(2, 3) [lo, hi] world-frame corners."""
        half = self.size / 2.0
        return np.stack([self.center - half, self.center + half])

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "center": [float(v) for v in self.center],
            "size": [float(v) for v in self.size],
            "score": float(self.score),
            "n_points": int(self.n_points),
        }

    @classmethod
    def from_json(cls, raw: dict) -> "SceneBox":
        return cls(
            name=str(raw["name"]),
            label=str(raw.get("label", raw["name"])),
            center=np.asarray(raw["center"], dtype=np.float32),
            size=np.asarray(raw["size"], dtype=np.float32),
            score=float(raw.get("score", 1.0)),
            n_points=int(raw.get("n_points", 0)),
        )
