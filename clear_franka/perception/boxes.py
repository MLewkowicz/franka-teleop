"""Mask + depth -> axis-aligned bounding boxes in the Franka base frame.

`mask_to_world_aabb` is the offline prototype's `mask_to_aabb` plus the two
things a hardware pipeline needs: rejection of the invalid-depth pixels that
`depth_to_camera_xyz` maps to the camera origin, and the `T_cam2base` hop into
`fr3_link0`. Everything downstream (the LLM scene block, the value maps, the
viser editor) speaks that world frame only.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from clear_franka.perception.types import Detection, SceneBox

logger = logging.getLogger(__name__)

TRIM_PCT = 2.0        # percentile trimmed off each end before fitting a box
MIN_POINTS = 50       # a mask with fewer valid 3D points is discarded
MIN_DEPTH_M = 0.10    # camera-frame z below this is invalid depth, not geometry
MAX_DEPTH_M = 4.00


def mask_to_world_aabb(
    xyz_cam: np.ndarray,
    mask: np.ndarray,
    T_cam2base: np.ndarray,
    *,
    trim_pct: float = TRIM_PCT,
    min_points: int = MIN_POINTS,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Backproject a 2D mask and fit a trimmed axis-aligned box in base frame.

    Returns (lo, hi, n_points) in `fr3_link0` metres, or None when the mask has
    too little valid depth behind it.
    """
    points = xyz_cam[mask]
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > MIN_DEPTH_M) & (z < MAX_DEPTH_M)
    points = points[valid]
    if len(points) < min_points:
        return None

    hom = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    world = (np.asarray(T_cam2base, dtype=np.float64) @ hom.T).T[:, :3]

    # Percentile trim, not min/max: SAM masks bleed a few pixels onto whatever
    # is behind the object, and those outliers would otherwise stretch the box
    # by tens of centimetres in depth.
    lo = np.percentile(world, trim_pct, axis=0)
    hi = np.percentile(world, 100.0 - trim_pct, axis=0)
    return lo.astype(np.float32), hi.astype(np.float32), int(len(points))


def _unique_name(label: str, taken: set[str]) -> str:
    name = label.strip().lower().replace(" ", "_")
    if name not in taken:
        return name
    i = 2
    while f"{name}_{i}" in taken:
        i += 1
    return f"{name}_{i}"


def detections_to_scene_boxes(
    per_frame: list[tuple[list[Detection], np.ndarray]],
    T_cam2base: np.ndarray,
    *,
    max_per_label: int = 1,
    min_size_m: float = 0.01,
    trim_pct: float = TRIM_PCT,
) -> list[SceneBox]:
    """Fuse per-frame detections into one set of named world-frame boxes.

    `per_frame` is [(detections, xyz_cam), ...] — one entry per captured frame.
    Within a frame, the `max_per_label` highest-scoring instances of each label
    are kept. Across frames, the per-instance boxes are combined by taking the
    ELEMENTWISE MEDIAN of the (lo, hi) corners, which throws out the occasional
    frame whose depth dropped out over part of the object without needing any
    instance-to-instance association beyond score rank.
    """
    # rank -> label -> list of (lo, hi, n_points, score), one entry per frame
    stacks: dict[tuple[str, int], list[tuple]] = defaultdict(list)

    for detections, xyz_cam in per_frame:
        by_label: dict[str, list[Detection]] = defaultdict(list)
        for det in detections:
            by_label[det.label].append(det)
        for label, dets in by_label.items():
            dets = sorted(dets, key=lambda d: d.score, reverse=True)[:max_per_label]
            for rank, det in enumerate(dets):
                fit = mask_to_world_aabb(
                    xyz_cam, det.mask, T_cam2base, trim_pct=trim_pct
                )
                if fit is None:
                    logger.debug(
                        "dropping %s[%d]: too few valid depth points", label, rank
                    )
                    continue
                lo, hi, n = fit
                stacks[(label, rank)].append((lo, hi, n, det.score))

    boxes: list[SceneBox] = []
    taken: set[str] = set()
    for (label, rank), entries in sorted(stacks.items(), key=lambda kv: kv[0]):
        lo = np.median(np.stack([e[0] for e in entries]), axis=0)
        hi = np.median(np.stack([e[1] for e in entries]), axis=0)
        size = np.maximum(hi - lo, min_size_m)
        name = _unique_name(label, taken)
        taken.add(name)
        boxes.append(
            SceneBox(
                name=name,
                label=label,
                center=((lo + hi) / 2.0).astype(np.float32),
                size=size.astype(np.float32),
                score=float(np.mean([e[3] for e in entries])),
                n_points=int(np.median([e[2] for e in entries])),
            )
        )
        logger.info(
            "%-20s score=%.3f center=%s size=%s (%d frame(s), %d pts)",
            name, boxes[-1].score,
            np.array2string(boxes[-1].center, precision=3),
            np.array2string(boxes[-1].size, precision=3),
            len(entries), boxes[-1].n_points,
        )
    return boxes


def save_scene_boxes(path: str | Path, boxes: list[SceneBox]) -> Path:
    """Write the workspace-boxes JSON schema (a superset of the hand-edited one).

    `clear_franka.value_maps.load_boxes` and `WorkspaceBoxStore` read this back
    unchanged — they only look at `name` / `center` / `size`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "frame": "fr3_link0",
        "units": "m",
        "boxes": [b.to_json() for b in boxes],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    logger.info("wrote %d box(es) -> %s", len(boxes), path)
    return path


def load_scene_boxes(path: str | Path) -> list[SceneBox]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [SceneBox.from_json(b) for b in raw["boxes"]]


def frame_to_world_cloud(
    xyz_cam: np.ndarray,
    rgb: np.ndarray,
    T_cam2base: np.ndarray,
    *,
    stride: int = 3,
    max_points: int = 40_000,
) -> tuple[np.ndarray, np.ndarray]:
    """One organized frame -> a decimated coloured cloud in `fr3_link0`.

    Shared by the viser preview and the value-map plots so both show the same
    geometry the boxes were fitted from. `max_points` matters for the Plotly
    path: it renders one rgb string per point, so a full 720p cloud would make
    an unopenable HTML file.
    """
    xyz = xyz_cam[::stride, ::stride].reshape(-1, 3)
    colors = rgb[::stride, ::stride].reshape(-1, 3)

    z = xyz[:, 2]
    valid = np.isfinite(xyz).all(axis=1) & (z > MIN_DEPTH_M) & (z < MAX_DEPTH_M)
    xyz, colors = xyz[valid], colors[valid]

    hom = np.concatenate([xyz, np.ones((len(xyz), 1))], axis=1)
    world = (np.asarray(T_cam2base, dtype=np.float64) @ hom.T).T[:, :3]

    if len(world) > max_points:
        idx = np.random.default_rng(0).choice(len(world), max_points, replace=False)
        world, colors = world[idx], colors[idx]
    return world.astype(np.float32), colors.astype(np.uint8)


def save_scene_cloud(path: str | Path, points: np.ndarray, colors: np.ndarray) -> Path:
    """Cache the cloud so `--source boxes` re-runs can still draw the scene."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, points=points, colors=colors)
    logger.info("wrote %d cloud point(s) -> %s", len(points), path)
    return path


def load_scene_cloud(path: str | Path):
    """(points, colors), or None when no cloud has been captured yet."""
    path = Path(path)
    if not path.is_file():
        return None
    data = np.load(path)
    return data["points"], data["colors"]


def aabb_edges(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """The 12 edges of a box, as (12, 2, 3) line segments (for viser)."""
    corners = np.array(np.meshgrid(*zip(lo, hi), indexing="ij")).reshape(3, -1).T
    return np.array(
        [
            [a, b]
            for i, a in enumerate(corners)
            for b in corners[i + 1:]
            if np.count_nonzero(~np.isclose(a, b)) == 1  # differ on exactly one axis
        ]
    )
