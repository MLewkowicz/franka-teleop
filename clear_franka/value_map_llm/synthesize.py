"""Task + detected boxes -> a list of steering stages with LangSteer ValueMaps.

This is the seam between perception and steering. Everything above it speaks
natural language and world-frame boxes; everything below it is the same
`voxposer.value_map.ValueMap` the hand-built maps in `clear_franka.value_maps`
produce, so `PositionFieldSteering` consumes either without caring which.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from clear_franka.perception.types import SceneBox
from clear_franka.value_map_llm.codegen import MapLMP, PlannerLMP
from clear_franka.value_map_llm.interface import BoxSceneInterface
from clear_franka.value_map_llm.llm import LLMBackend

logger = logging.getLogger(__name__)

# Margin added around the union of the detected boxes when the workspace
# bounds are derived from the scene. Wide enough that the gripper's approach
# is inside the grid (gradients are clipped at the bounds), tight enough that
# a 100^3 grid still resolves ~1cm per voxel.
DEFAULT_BOUNDS_MARGIN_M = 0.25


@dataclass
class SynthesizedStage:
    """One steering stage: where to go, what to avoid, and when to hand off."""

    label: str
    affordance_query: str
    avoidance_query: Optional[str]
    arrival_radius_m: float
    value_map: Any            # voxposer.value_map.ValueMap
    target_world: np.ndarray  # (3,) fr3_link0 metres


def workspace_bounds_from_boxes(
    boxes: list[SceneBox], margin_m: float = DEFAULT_BOUNDS_MARGIN_M
) -> tuple[np.ndarray, np.ndarray]:
    """Grid bounds covering every detected box plus a margin."""
    if not boxes:
        raise ValueError("cannot derive workspace bounds from an empty scene")
    lo = np.min([b.aabb[0] for b in boxes], axis=0) - margin_m
    hi = np.max([b.aabb[1] for b in boxes], axis=0) + margin_m
    return lo.astype(np.float32), hi.astype(np.float32)


def _seed_centroid_world(vm) -> np.ndarray:
    """World-frame centroid of the affordance seed voxels set by the LLM.

    `ValueMap.smooth()` stashes the pre-EDT binary seed in `_raw_affordance`;
    its centroid is the point the steering treats as the stage target (the
    DistanceScaler ramp and the arrival latch both measure against it).
    """
    from voxposer.calvin_interface import voxel2pc

    seed = vm._raw_affordance
    idx = np.argwhere(seed > 0)
    if len(idx) == 0:
        raise ValueError("affordance map is empty after smoothing")
    return voxel2pc(
        idx.mean(axis=0),
        vm.workspace_bounds_min,
        vm.workspace_bounds_max,
        vm.map_size,
    ).astype(np.float32)


# Probe radius (m) for the near-field check below, and the directions probed.
_NEAR_PROBE_M = 0.05
_PROBE_DIRS = np.array([
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
], dtype=np.float32)


def _validate_pull(vm, target_world: np.ndarray, spec: dict, idx: int) -> None:
    """Fail if the finished field does not actually pull toward its own target.

    A structural "does the avoidance overlap the affordance seed" test is not
    enough: the failure this exists to catch had a seed 8cm clear of the padded
    obstacle box and still shoved the arm away, because a large blurred
    avoidance region near the target dominates the descent around it. So probe
    the assembled cost field close in — within `_NEAR_PROBE_M`, nothing should
    push away from the destination — and report the direction that fails.
    """
    t = np.asarray(target_world, dtype=np.float32)
    probes = t[None, :] + _PROBE_DIRS * _NEAR_PROBE_M
    descent = -vm.gradient_at_world_points(probes)
    to_target = t[None, :] - probes

    dn = np.linalg.norm(descent, axis=1)
    tn = np.linalg.norm(to_target, axis=1)
    cos = np.einsum("ij,ij->i", descent, to_target) / np.maximum(dn * tn, 1e-9)

    bad = np.nonzero(cos < 0)[0]
    if len(bad) == 0:
        # An obstacle beside or under the target does not invert the descent,
        # but it does flatten it: measured on this scene, boxing the plate took
        # the lateral probes from +1.00 to ~+0.1 while every sign stayed
        # positive. Worth surfacing, not worth failing on — a legitimate
        # obstacle near the destination degrades the pull the same way.
        weak = float(cos.mean())
        if weak < 0.5:
            logger.warning(
                "stage %d ('%s'): the pull toward %s is weak near the target "
                "(mean cos %.2f over %d probes). Check the avoidance query "
                "(%s) and the rendered plot before running this.",
                idx, spec["label"], np.round(t, 3), weak, len(cos),
                spec["avoidance"],
            )
        return
    detail = ", ".join(
        f"{np.round(_PROBE_DIRS[b], 0).astype(int).tolist()}: cos={cos[b]:+.2f}"
        for b in bad
    )
    raise ValueError(
        f"stage {idx} ('{spec['label']}'): the value map pushes AWAY from its "
        f"own target {np.round(t, 3)} at {len(bad)} of {len(cos)} probe points "
        f"{_NEAR_PROBE_M * 100:.0f}cm out ({detail}). This is almost always the "
        f"avoidance query — '{spec['avoidance']}' — covering the destination or "
        "the carried object. Avoidance must name a THIRD object that blocks the "
        "path, or be None."
    )


def synthesize_value_maps(
    task: str,
    boxes: list[SceneBox],
    *,
    workspace_bounds_min: Optional[np.ndarray] = None,
    workspace_bounds_max: Optional[np.ndarray] = None,
    bounds_margin_m: float = DEFAULT_BOUNDS_MARGIN_M,
    map_size: int = 100,
    avoidance_weight: float = 1.0,
    obstacle_sigma: float = 3.0,
    ee_pos_world: Optional[np.ndarray] = None,
    workspace_image: Optional[np.ndarray] = None,
    llm: Optional[dict] = None,
) -> list[SynthesizedStage]:
    """Run planner -> per-stage affordance/avoidance LMPs -> ValueMaps.

    Raises rather than degrading: an empty affordance map or an object the
    planner invented means the steering would aim at nothing, and a silent
    fallback there would put the arm somewhere no one chose.
    """
    from voxposer.value_map import ValueMap

    if workspace_bounds_min is None or workspace_bounds_max is None:
        workspace_bounds_min, workspace_bounds_max = workspace_bounds_from_boxes(
            boxes, bounds_margin_m
        )
    ws_min = np.asarray(workspace_bounds_min, dtype=np.float32)
    ws_max = np.asarray(workspace_bounds_max, dtype=np.float32)

    interface = BoxSceneInterface(
        boxes,
        workspace_bounds_min=ws_min,
        workspace_bounds_max=ws_max,
        map_size=map_size,
        ee_pos_world=ee_pos_world,
    )
    scene_block = interface.scene_block()
    logger.info("scene handed to the planner:\n%s", scene_block)

    backend = LLMBackend(**(llm or {}))
    plan = PlannerLMP(backend, scene_block, image=workspace_image)(task)
    logger.info("planner returned %d stage(s) for '%s'", len(plan), task)

    affordance_lmp = MapLMP("get_affordance_map", backend, interface, scene_block)
    avoidance_lmp = MapLMP("get_avoidance_map", backend, interface, scene_block)

    stages: list[SynthesizedStage] = []
    for i, spec in enumerate(plan):
        interface.reset_resolved()
        affordance = affordance_lmp(spec["affordance"])
        affordance_objects = interface.resolved_objects()
        if affordance.max() <= 0:
            raise ValueError(
                f"stage {i} ('{spec['label']}'): the affordance query "
                f"'{spec['affordance']}' produced an empty map — the target "
                "probably fell outside the workspace bounds "
                f"[{np.round(ws_min, 2)}, {np.round(ws_max, 2)}]"
            )

        avoidance = None
        if spec["avoidance"]:
            interface.reset_resolved()
            avoidance = avoidance_lmp(spec["avoidance"])
            shared = affordance_objects & interface.resolved_objects()
            if shared:
                raise ValueError(
                    f"stage {i} ('{spec['label']}'): the avoidance query "
                    f"'{spec['avoidance']}' and the affordance query "
                    f"'{spec['affordance']}' both resolve to {sorted(shared)}. "
                    "Avoidance must name a THIRD object that blocks the path — "
                    "never the destination, never the object in the gripper."
                )
            if avoidance.max() <= 0:
                logger.warning(
                    "stage %d: avoidance query '%s' produced an empty map; "
                    "continuing with affordance only", i, spec["avoidance"],
                )
                avoidance = None
            elif bool((avoidance[affordance > 0]).any()):
                # Checked on the sparse masks: after smooth() the avoidance is
                # blurred across most of the grid and this would false-positive.
                raise ValueError(
                    f"stage {i} ('{spec['label']}'): the avoidance query "
                    f"'{spec['avoidance']}' covers the affordance target "
                    f"'{spec['affordance']}'. Avoidance must name a THIRD "
                    "object that blocks the path, never the destination or the "
                    "object in the gripper."
                )

        vm = ValueMap(
            affordance=affordance,
            avoidance=avoidance,
            workspace_bounds_min=ws_min,
            workspace_bounds_max=ws_max,
            map_size=map_size,
            instruction=f"{task} :: {spec['label']}",
        )
        vm.smooth(obstacle_sigma=obstacle_sigma)
        vm.precompute_gradients(avoidance_weight=avoidance_weight)

        target = _seed_centroid_world(vm)
        _validate_pull(vm, target, spec, i)
        stages.append(
            SynthesizedStage(
                label=spec["label"],
                affordance_query=spec["affordance"],
                avoidance_query=spec["avoidance"],
                arrival_radius_m=spec["arrival_radius_cm"] / 100.0,
                value_map=vm,
                target_world=target,
            )
        )
        logger.info(
            "stage %d '%s': target=%s arrival=%.3fm avoidance=%s",
            i, spec["label"], np.array2string(target, precision=3),
            stages[-1].arrival_radius_m, spec["avoidance"],
        )

    return stages


def synthesize_and_save(
    task: str,
    boxes: list[SceneBox],
    artifact_path: str | Path,
    *,
    workspace_bounds_min: Optional[np.ndarray] = None,
    workspace_bounds_max: Optional[np.ndarray] = None,
    bounds_margin_m: float = DEFAULT_BOUNDS_MARGIN_M,
    map_size: int = 100,
    avoidance_weight: float = 1.0,
    obstacle_sigma: float = 3.0,
    workspace_image: Optional[np.ndarray] = None,
    llm: Optional[dict] = None,
    boxes_path: Optional[str] = None,
) -> tuple[Path, list[SynthesizedStage]]:
    """`synthesize_value_maps` + `save_stages` in one call.

    Returns (artifact_path, stages) -- the stages are handed back too since
    the offline CLI needs them for its diagnostic prints/plots; deploy's
    live-synthesis path just ignores them.

    The shared tail used by both the offline CLI (`synthesize_value_map.py`)
    and deploy's live-synthesis path, so both write artifacts the exact same
    way.
    """
    if workspace_bounds_min is None or workspace_bounds_max is None:
        workspace_bounds_min, workspace_bounds_max = workspace_bounds_from_boxes(
            boxes, bounds_margin_m
        )
    stages = synthesize_value_maps(
        task,
        boxes,
        workspace_bounds_min=workspace_bounds_min,
        workspace_bounds_max=workspace_bounds_max,
        map_size=map_size,
        avoidance_weight=avoidance_weight,
        obstacle_sigma=obstacle_sigma,
        workspace_image=workspace_image,
        llm=llm,
    )
    # Lazy import: artifact.py imports SynthesizedStage from this module, so a
    # module-level import here would be circular.
    from clear_franka.value_map_llm.artifact import save_stages

    artifact = save_stages(
        artifact_path,
        stages,
        task=task,
        boxes_path=boxes_path,
        avoidance_weight=avoidance_weight,
        obstacle_sigma=obstacle_sigma,
    )
    return artifact, stages
