"""Persist synthesized stages so the robot deploy never calls SAM 3 or an LLM.

Synthesis is offline: `synthesize_value_map.py` writes one .npz, the deploy
loads it. That keeps model loading and network calls out of the control path
and makes a run reproducible — the artifact records the task, the scene it was
built from, and the maps that came out of it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from clear_franka.value_map_llm.synthesize import SynthesizedStage

logger = logging.getLogger(__name__)

ARTIFACT_VERSION = 1


def save_stages(
    path: str | Path,
    stages: list[SynthesizedStage],
    *,
    task: str,
    boxes_path: Optional[str] = None,
    avoidance_weight: float = 1.0,
    obstacle_sigma: float = 3.0,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    vm0 = stages[0].value_map
    payload: dict = {
        "meta": np.array(json.dumps({
            "version": ARTIFACT_VERSION,
            "task": task,
            "boxes_path": boxes_path,
            "avoidance_weight": avoidance_weight,
            "obstacle_sigma": obstacle_sigma,
            "map_size": int(vm0.map_size),
            "workspace_bounds_min": vm0.workspace_bounds_min.tolist(),
            "workspace_bounds_max": vm0.workspace_bounds_max.tolist(),
            "stages": [
                {
                    "label": s.label,
                    "affordance_query": s.affordance_query,
                    "avoidance_query": s.avoidance_query,
                    "arrival_radius_m": float(s.arrival_radius_m),
                    "target_world": s.target_world.tolist(),
                    "has_avoidance": s.value_map.avoidance is not None,
                }
                for s in stages
            ],
        })),
    }
    for i, s in enumerate(stages):
        # The smoothed fields are stored, not the sparse seeds: they are what
        # precompute_gradients() consumes, and re-deriving them on load would
        # have to re-run smooth() with matching parameters.
        payload[f"affordance_{i}"] = s.value_map.affordance.astype(np.float32)
        payload[f"raw_affordance_{i}"] = (
            s.value_map._raw_affordance.astype(np.float32)
            if s.value_map._raw_affordance is not None
            else np.zeros_like(s.value_map.affordance, dtype=np.float32)
        )
        if s.value_map.avoidance is not None:
            payload[f"avoidance_{i}"] = s.value_map.avoidance.astype(np.float32)

    np.savez_compressed(path, **payload)
    logger.info("wrote %d stage(s) -> %s", len(stages), path)
    return path


def load_stages(path: str | Path) -> tuple[list[SynthesizedStage], dict]:
    """Rebuild the stages, with cost-map gradients precomputed and ready.

    Returns (stages, meta). `meta` carries the task, the scene the maps were
    built from, and the grid parameters, so callers can log (or refuse) a stale
    artifact.
    """
    from voxposer.value_map import ValueMap

    path = Path(path)
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    if meta.get("version") != ARTIFACT_VERSION:
        raise ValueError(
            f"{path}: artifact version {meta.get('version')} != "
            f"{ARTIFACT_VERSION}; regenerate with synthesize_value_map.py"
        )

    ws_min = np.asarray(meta["workspace_bounds_min"], dtype=np.float32)
    ws_max = np.asarray(meta["workspace_bounds_max"], dtype=np.float32)

    stages: list[SynthesizedStage] = []
    for i, spec in enumerate(meta["stages"]):
        vm = ValueMap(
            affordance=data[f"affordance_{i}"],
            avoidance=(data[f"avoidance_{i}"] if spec["has_avoidance"] else None),
            workspace_bounds_min=ws_min,
            workspace_bounds_max=ws_max,
            map_size=int(meta["map_size"]),
            instruction=f"{meta['task']} :: {spec['label']}",
        )
        # Already smoothed at synthesis time — go straight to the gradients.
        vm._raw_affordance = data[f"raw_affordance_{i}"]
        vm.precompute_gradients(
            avoidance_weight=float(meta.get("avoidance_weight", 1.0))
        )
        stages.append(
            SynthesizedStage(
                label=spec["label"],
                affordance_query=spec["affordance_query"],
                avoidance_query=spec["avoidance_query"],
                arrival_radius_m=float(spec["arrival_radius_m"]),
                value_map=vm,
                target_world=np.asarray(spec["target_world"], dtype=np.float32),
            )
        )

    logger.info(
        "loaded %d stage(s) from %s (task=%r)", len(stages), path, meta["task"]
    )
    return stages, meta
