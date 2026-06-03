"""Hardcoded VoxPoser-style value maps over the workspace bounding boxes.

Builds the *place*-stage value map directly from `data/workspace_boxes.json`
(no LLM). The wine rack (box_001) seeds an attractive affordance on its front
(-x, robot-facing) face; the cabinet surface (box_002) and the volume beneath
it (box_003) seed a repulsive avoidance field. Both are handed to LangSteer's
`voxposer.value_map.ValueMap`, which EDT-smooths the affordance into a radiating
field, gaussian-blurs the avoidance, and precomputes the cost-map gradient used
to steer the diffusion denoiser (see clear_franka/box_field_steering.py).

LangSteer (`voxposer.*`) must be importable; callers add it to sys.path first
(deploy_diffuser_actor._wire_langsteer, or the visualize script).

Voxel/world convention matches voxposer.calvin_interface.pc2voxel exactly:
    idx = round((p - ws_min) / (ws_max - ws_min) * (map_size - 1))
so the affordance grid is indexed [ix, iy, iz] with world axes x, y, z.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt

# Default place-stage box roles (confirmed against the scene geometry).
RACK = "box_001"        # wine rack  -> affordance (attractive)
CABINET = "box_002"     # cabinet surface -> avoidance (repulsive)
UNDERNEATH = "box_003"  # volume under the cabinet -> avoidance (repulsive)

# Value-map workspace bounds (absolute world, fr3_link0), chosen to cover all
# three boxes plus margin. Overridable by callers / config.
DEFAULT_WS_MIN = np.array([0.45, -0.50, -0.15], dtype=np.float32)
DEFAULT_WS_MAX = np.array([1.00, 0.80, 0.75], dtype=np.float32)
DEFAULT_MAP_SIZE = 100


def load_boxes(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """Read workspace_boxes.json -> {name: {"center": (3,), "size": (3,)}}."""
    data = json.loads(Path(path).read_text())
    boxes: dict[str, dict[str, np.ndarray]] = {}
    for box in data["boxes"]:
        boxes[box["name"]] = {
            "center": np.asarray(box["center"], dtype=np.float32),
            "size": np.asarray(box["size"], dtype=np.float32),
        }
    return boxes


def _world_to_idx(
    world_xyz: np.ndarray, ws_min: np.ndarray, ws_max: np.ndarray, map_size: int
) -> np.ndarray:
    """World point(s) -> integer voxel index, clipped to the grid (pc2voxel)."""
    world_xyz = np.clip(world_xyz, ws_min, ws_max)
    idx = np.round((world_xyz - ws_min) / (ws_max - ws_min) * (map_size - 1))
    return np.clip(idx.astype(np.int32), 0, map_size - 1)


def _fill_slab(
    mask: np.ndarray,
    lo_world: np.ndarray,
    hi_world: np.ndarray,
    ws_min: np.ndarray,
    ws_max: np.ndarray,
    map_size: int,
) -> None:
    """Set mask True over the voxel index span of an axis-aligned world slab."""
    lo = _world_to_idx(lo_world, ws_min, ws_max, map_size)
    hi = _world_to_idx(hi_world, ws_min, ws_max, map_size)
    mask[
        lo[0]:hi[0] + 1,
        lo[1]:hi[1] + 1,
        lo[2]:hi[2] + 1,
    ] = True


def box_voxel_mask(
    center: np.ndarray,
    size: np.ndarray,
    ws_min: np.ndarray,
    ws_max: np.ndarray,
    map_size: int,
) -> np.ndarray:
    """Axis-aligned box -> (M, M, M) boolean occupancy mask."""
    mask = np.zeros((map_size, map_size, map_size), dtype=bool)
    half = size / 2.0
    _fill_slab(mask, center - half, center + half, ws_min, ws_max, map_size)
    return mask


def front_face_voxels(
    center: np.ndarray,
    size: np.ndarray,
    ws_min: np.ndarray,
    ws_max: np.ndarray,
    map_size: int,
    face_thickness_m: float = 0.04,
    forward_extend_m: float = 0.06,
    x_offset_m: float = 0.0,
    y_offset_m: float = 0.0,
    z_offset_m: float = 0.0,
    y_extent_m: float | None = None,
    z_extent_m: float | None = None,
) -> np.ndarray:
    """Slab on the box's front (-x, robot-facing) face -> (M, M, M) bool mask.

    In x the slab runs from `x_min − forward_extend_m + x_offset_m` to
    `x_min + face_thickness_m + x_offset_m`. `x_offset_m > 0` pushes the whole
    slab deeper into the box (+x direction); useful for avoidance walls that
    should sit *behind* the rack basin.

    In y / z the slab spans the box's full extent by default. Pass
    `y_extent_m` / `z_extent_m` to clip to a narrower window centered on
    `center[1]` / `center[2]`. CRITICAL for affordance seeds: EDT gradients
    point toward the *nearest seed voxel*, not the seed centroid — so a wide
    seed gives weak directional pull in the seed's own axis (a point just
    outside the seed in y barely needs to move in y to reach the nearest seed
    voxel). Clipping the seed to a small window around the basin produces
    strong gradients pointing toward the *center*, which is what we want when
    the basin is a specific point inside a larger rack face.
    """
    mask = np.zeros((map_size, map_size, map_size), dtype=bool)
    half = size / 2.0
    x_min = center[0] - half[0]
    y_half = half[1] if y_extent_m is None else y_extent_m / 2.0
    z_half = half[2] if z_extent_m is None else z_extent_m / 2.0
    lo_world = np.array(
        [x_min - forward_extend_m + x_offset_m,
         center[1] - y_half + y_offset_m,
         center[2] - z_half + z_offset_m],
        dtype=np.float32,
    )
    hi_world = np.array(
        [x_min + face_thickness_m + x_offset_m,
         center[1] + y_half + y_offset_m,
         center[2] + z_half + z_offset_m],
        dtype=np.float32,
    )
    _fill_slab(mask, lo_world, hi_world, ws_min, ws_max, map_size)
    return mask


def _basin_carve_mask(
    ws_min: np.ndarray, ws_max: np.ndarray, map_size: int,
    basin_world: np.ndarray, sigma_m: float,
) -> np.ndarray:
    """Soft Gaussian carve mask centered at the basin: 0 at basin, → 1 far.

    Used to multiply the avoidance field so the rack-approach corridor isn't
    crowded by the obstacle wall: at the basin the avoidance is fully zeroed
    (×0), and the carve fades back to no-effect (×1) at ~2-3σ away. Decouples
    "wall keeps EE out of underneath" from "wall mustn't repel EE at the rack."
    """
    xs = np.linspace(ws_min[0], ws_max[0], map_size, dtype=np.float32)
    ys = np.linspace(ws_min[1], ws_max[1], map_size, dtype=np.float32)
    zs = np.linspace(ws_min[2], ws_max[2], map_size, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    d2 = (X - basin_world[0]) ** 2 + (Y - basin_world[1]) ** 2 + (Z - basin_world[2]) ** 2
    return (1.0 - np.exp(-d2 / (2.0 * sigma_m * sigma_m))).astype(np.float32)


def interior_distance_field(mask: np.ndarray) -> np.ndarray:
    """Interior distance transform of a box mask, normalized to [0, 1].

    Distance from each interior voxel to the nearest boundary: 0 at the faces,
    peaking at the box center. Its gradient points *inward* (toward the center),
    so the descent direction (-gradient) points *outward* through the nearest
    face everywhere inside the box (except the exact centroid). This is what
    makes the avoidance push the EE out of the region — a binary box that is
    merely Gaussian-blurred has a flat interior plateau (zero gradient), letting
    the global affordance leak through and pull upward from below.
    """
    if not mask.any():
        return mask.astype(np.float32)
    d = distance_transform_edt(mask)
    m = d.max()
    return (d / m).astype(np.float32) if m > 0 else d.astype(np.float32)


def front_face_center(
    center: np.ndarray,
    size: np.ndarray,
    *,
    face_thickness_m: float = 0.04,
    forward_extend_m: float = 0.06,
    y_offset_m: float = 0.0,
    z_offset_m: float = 0.0,
) -> np.ndarray:
    """World-frame center of the front-face affordance slab (the basin target).

    The slab spans x ∈ [x_min − forward_extend_m, x_min + face_thickness_m] with
    the box's full y/z extent. The DistanceScaler uses this as the point the
    position guidance should ramp DOWN to as the EE gets close — preserving the
    pull while far away and shrinking it to a floor near the rack so the policy
    can settle into the basin without being shoved past it.
    """
    half = size / 2.0
    x_min = center[0] - half[0]
    return np.array([
        x_min + 0.5 * (face_thickness_m - forward_extend_m),
        center[1] + y_offset_m,
        center[2] + z_offset_m,
    ], dtype=np.float32)


def _extend_back(box: dict[str, np.ndarray], back_extend_m: float) -> dict[str, np.ndarray]:
    """Push a box's +x face further back (away from the robot) by `back_extend_m`.

    Used to bias the obstacle interior-distance field so the *nearest face* from
    every reachable EE position is the −x (robot-facing) face. Without this, the
    back half of a long obstacle pushes the EE *backward* (away from the robot)
    rather than toward the robot's base, which is the wrong direction for
    settling out of an under-shelf pose.
    """
    center = box["center"].copy()
    size = box["size"].copy()
    center[0] += back_extend_m / 2.0
    size[0] += back_extend_m
    return {"center": center, "size": size}


def build_place_value_map(
    boxes: dict[str, dict[str, np.ndarray]],
    *,
    ws_min: np.ndarray = DEFAULT_WS_MIN,
    ws_max: np.ndarray = DEFAULT_WS_MAX,
    map_size: int = DEFAULT_MAP_SIZE,
    avoidance_weight: float = 1.0,
    obstacle_sigma: float = 1.0,
    suppress_affordance_in_obstacles: bool = False,
    face_thickness_m: float = 0.04,
    forward_extend_m: float = 0.06,
    obstacle_back_extend_m: float = 0.0,
    wall_x_offset_m: float = 0.0,
    avoidance_carve_radius_m: float = 0.0,
    affordance_y_extent_m: float | None = None,
    affordance_z_extent_m: float | None = None,
    basin_y_offset_m: float = 0.0,
    basin_z_offset_m: float = 0.0,
    include_underneath_wall: bool = True,
    rack: str = RACK,
    cabinet: str = CABINET,
    underneath: str = UNDERNEATH,
    instruction: str = "place: invert glass onto wine rack",
):
    """Assemble the place-stage ValueMap from the workspace boxes.

    Affordance = front face of the wine rack, EDT-smoothed by ValueMap.smooth()
    into a field that radiates outward (draws the EE up toward the rack once it
    is clear of the obstacles). Avoidance = cabinet surface plus the volume
    beneath it, built as a per-box *interior* distance field (peaked at each
    box center) so its gradient pushes the EE *out* of those regions. Returns a
    smoothed `ValueMap` with gradients precomputed (ready for
    gradient_field_tensor / gradient_at_world_points).
    """
    from voxposer.value_map import ValueMap

    ws_min = np.asarray(ws_min, dtype=np.float32)
    ws_max = np.asarray(ws_max, dtype=np.float32)

    affordance = front_face_voxels(
        boxes[rack]["center"], boxes[rack]["size"],
        ws_min, ws_max, map_size,
        face_thickness_m=face_thickness_m,
        forward_extend_m=forward_extend_m,
        y_offset_m=basin_y_offset_m,
        z_offset_m=basin_z_offset_m,
        y_extent_m=affordance_y_extent_m,
        z_extent_m=affordance_z_extent_m,
    ).astype(np.float32)

    # Avoidance = front-face *wall* on each obstacle box (cabinet + underneath),
    # built with the same front-face slab geometry as the rack affordance and
    # Gaussian-smoothed by ValueMap.smooth() below. The resulting field peaks
    # at the wall (slab voxels) and decays exponentially with distance to the
    # nearest slab voxel (decay constant ≈ obstacle_sigma in voxel units), so:
    #   * EE in front of the wall (x < x_face) and at the wall's z range
    #     → strong gradient toward the slab → descent pushes −x toward the
    #     robot base.
    #   * EE rising up to rack height (z > obstacle top) → distance to slab
    #     grows in z → avoidance value & gradient decay → push fades
    #     automatically as the EE aligns vertically with the rack.
    # (The previous interior-EDT approach pushed outward from box centers,
    # which gave the back half of the obstacle a +x push *away* from the
    # robot — wrong direction. The slab wall fixes that.)
    cabinet_wall = front_face_voxels(
        boxes[cabinet]["center"], boxes[cabinet]["size"],
        ws_min, ws_max, map_size,
        face_thickness_m=face_thickness_m, forward_extend_m=forward_extend_m,
        x_offset_m=wall_x_offset_m,
    )
    # The underneath wall sits at x ≈ underneath_x_face − wall_x_offset_m, which
    # lands in front of the rack basin in x and spans the basin's y range. With
    # heavy avoidance_weight it dominates the affordance gradient once the EE
    # crosses to x < wall_x, locking the EE on the wrong side from the basin.
    # Gated off here when the cabinet wall is sufficient anti-cabinet guard.
    avoidance_mask = cabinet_wall
    if include_underneath_wall:
        underneath_wall = front_face_voxels(
            boxes[underneath]["center"], boxes[underneath]["size"],
            ws_min, ws_max, map_size,
            face_thickness_m=face_thickness_m, forward_extend_m=forward_extend_m,
            x_offset_m=wall_x_offset_m,
        )
        avoidance_mask = avoidance_mask | underneath_wall
    avoidance = avoidance_mask.astype(np.float32)
    # Full-volume obstacle masks — used below to suppress the global affordance
    # EDT inside the obstacles (so the rack's pull can't reach the EE *through*
    # the cabinet/underneath body), even though the avoidance gradient itself
    # is now driven only by the front-face wall, not the volume.
    cabinet_mask = box_voxel_mask(
        boxes[cabinet]["center"], boxes[cabinet]["size"], ws_min, ws_max, map_size)
    underneath_mask = box_voxel_mask(
        boxes[underneath]["center"], boxes[underneath]["size"], ws_min, ws_max, map_size)
    # `obstacle_back_extend_m` and `interior_distance_field` are no longer used
    # for the default avoidance shape but kept exported for opt-in experiments.
    _ = obstacle_back_extend_m

    vm = ValueMap(
        affordance=affordance,
        workspace_bounds_min=ws_min,
        workspace_bounds_max=ws_max,
        map_size=map_size,
        instruction=instruction,
        avoidance=avoidance,
    )
    # smooth() EDT-expands the affordance seed (global radiating field, covers
    # the whole grid) and Gaussian-smooths the avoidance walls.
    vm.smooth(obstacle_sigma=obstacle_sigma)

    # Default is no suppression — the global EDT affordance should reach the
    # entire workspace so the EE keeps getting pulled toward the rack at every
    # height. Suppressing inside the obstacles created "dead zones" where any
    # predicted-waypoint that landed in the underneath/cabinet body got zero
    # pull, which manifested as the EE stalling at certain heights mid-climb.
    if suppress_affordance_in_obstacles:
        vm.affordance[cabinet_mask | underneath_mask] = 0.0

    # Carve out a soft sphere of avoidance around the basin so the wall behind
    # it doesn't repel the EE as it approaches the rack from the front. At the
    # basin the carve is ×0 (no avoidance); the carve fades back to ×1 at
    # ~2-3σ of `avoidance_carve_radius_m`. Disabled when radius=0.
    if avoidance_carve_radius_m > 0.0:
        carve = _basin_carve_mask(
            ws_min, ws_max, map_size,
            front_face_center(boxes[rack]["center"], boxes[rack]["size"],
                              face_thickness_m=face_thickness_m,
                              forward_extend_m=forward_extend_m,
                              y_offset_m=basin_y_offset_m,
                              z_offset_m=basin_z_offset_m),
            avoidance_carve_radius_m,
        )
        vm.avoidance = vm.avoidance * carve

    vm.precompute_gradients(avoidance_weight=avoidance_weight)
    return vm


def gradient_field_tensor(vm: Any, device: str = "cuda"):
    """Stack a ValueMap's precomputed cost gradients into a (M, M, M, 3) tensor.

    Mirrors StageManager's gradient_field assembly so PositionTransform.
    lookup_voxel_gradient can index it directly. precompute_gradients() must
    have been called (build_place_value_map does this).
    """
    import torch

    if vm._grad_x is None:
        vm.precompute_gradients()
    grad = np.stack([vm._grad_x, vm._grad_y, vm._grad_z], axis=-1)
    return torch.from_numpy(grad).float().to(device)
