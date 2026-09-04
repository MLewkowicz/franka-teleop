"""Text-prompted 3D perception: ZED frames -> SAM 3 masks -> world-frame boxes.

The pipeline is deliberately split so each half is usable alone:

    source (Frame: rgb + camera-frame XYZ)
        -> Sam3Segmenter.detect(rgb)          -> [Detection]
        -> detections_to_scene_boxes(...)     -> [SceneBox]  (fr3_link0 frame)
        -> save_scene_boxes(path)             -> workspace-boxes JSON

The JSON is a superset of `data/workspace_boxes.json`, so `clear_franka.
value_maps.load_boxes` and the viser `WorkspaceBoxEditor` read it unchanged;
the extra `label` / `score` / `n_points` keys are ignored by both.
"""

from clear_franka.perception.boxes import (
    aabb_edges,
    detections_to_scene_boxes,
    frame_to_world_cloud,
    load_scene_boxes,
    label_scene_image,
    load_scene_cloud,
    load_scene_image,
    mask_to_world_aabb,
    match_boxes_to_frame_detections,
    save_scene_boxes,
    save_scene_cloud,
    save_scene_image,
)
from clear_franka.perception.capture import capture_scene_boxes
from clear_franka.perception.sam3 import Sam3Segmenter
from clear_franka.perception.sources import (
    Frame,
    SvoSource,
    ZedLiveSource,
    collect_frames,
)
from clear_franka.perception.types import Detection, SceneBox

__all__ = [
    "Detection",
    "SceneBox",
    "Frame",
    "Sam3Segmenter",
    "SvoSource",
    "ZedLiveSource",
    "collect_frames",
    "aabb_edges",
    "capture_scene_boxes",
    "detections_to_scene_boxes",
    "frame_to_world_cloud",
    "load_scene_boxes",
    "label_scene_image",
    "load_scene_cloud",
    "load_scene_image",
    "mask_to_world_aabb",
    "match_boxes_to_frame_detections",
    "save_scene_boxes",
    "save_scene_cloud",
    "save_scene_image",
]
