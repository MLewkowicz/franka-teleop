"""Orchestrates a scene capture: source frames -> SAM 3 -> fitted world boxes.

Shared by `synthesize_value_map.py` (offline CLI, builds its own source from
an SVO/live camera) and `deploy_diffuser_actor.py` (online, wraps an
already-open camera in a `ZedLiveSource`) so both go through the exact same
perception body.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from clear_franka.perception.boxes import (
    detections_to_scene_boxes,
    frame_to_world_cloud,
    label_scene_image,
    match_boxes_to_frame_detections,
    save_scene_boxes,
    save_scene_cloud,
    save_scene_image,
)
from clear_franka.perception.sam3 import Sam3Segmenter
from clear_franka.perception.sources import FrameSource, collect_frames
from clear_franka.perception.types import SceneBox


def capture_scene_boxes(
    source: FrameSource,
    classes: list[str],
    T_cam2base: np.ndarray,
    *,
    sam3_checkpoint: str,
    confidence: float = 0.5,
    bpe_path: Optional[str] = None,
    max_per_label: int = 2,
    frames: int = 5,
    skip_frames: int = 2,
    boxes_out: Optional[str] = None,
    cloud_out: Optional[str] = None,
    cloud_points: int = 40000,
    image_out: Optional[str] = None,
) -> tuple[list[SceneBox], tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Segment `classes` in `source` and fit their world-frame boxes.

    Returns (boxes, (points, colors), labeled_image) -- the cloud and the
    labeled snapshot both come from the same frame the boxes were fitted
    from, so anything that looks wrong in a plot is wrong against the exact
    data that produced it.
    """
    if not classes:
        raise ValueError("capture_scene_boxes: classes must be non-empty")
    if not sam3_checkpoint:
        raise ValueError("capture_scene_boxes: sam3_checkpoint is required")

    collected = collect_frames(source, frames, skip=skip_frames)
    segmenter = Sam3Segmenter(
        sam3_checkpoint, classes, confidence=confidence, bpe_path=bpe_path
    )
    per_frame = [(segmenter.detect(f.rgb), f.xyz) for f in collected]

    boxes = detections_to_scene_boxes(
        per_frame, T_cam2base, max_per_label=max_per_label
    )
    if not boxes:
        raise RuntimeError(
            f"no objects detected for {classes} -- lower confidence, check "
            "the camera view, or rename the prompts"
        )
    if boxes_out:
        save_scene_boxes(boxes_out, boxes)

    cloud = frame_to_world_cloud(
        collected[-1].xyz, collected[-1].rgb, T_cam2base, max_points=cloud_points
    )
    if cloud_out:
        save_scene_cloud(cloud_out, *cloud)

    # Label the last frame with each box's own SAM footprint (not a
    # reprojected 3D box) so the planner can tell which physical object in
    # the photo is 'bowl' vs 'bowl_2' -- the box table's world-frame numbers
    # alone give it no way to make that correspondence.
    matches = match_boxes_to_frame_detections(
        boxes, per_frame[-1][0], max_per_label
    )
    labeled_image = label_scene_image(collected[-1].rgb, matches)
    if image_out:
        save_scene_image(image_out, labeled_image)

    return boxes, cloud, labeled_image
