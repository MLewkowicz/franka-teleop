"""The sandbox API that LLM-generated value-map code runs against.

`BoxSceneInterface` is the hardware counterpart of LangSteer's
`voxposer.calvin_interface.CalvinLMPInterface`. Only `detect()` differs: CALVIN
resolves objects out of the simulator's ground-truth `scene_obs`, we resolve
them out of the SAM 3 boxes in `fr3_link0`. Everything else — `cm2index`,
`set_voxel_by_radius`, `set_voxel_by_box`, `get_empty_*_map` and the
world<->voxel convention that `ValueMap.gradient_at_world_points` inverts — is
scene-agnostic, so we DELEGATE to a CalvinLMPInterface configured with the same
grid instead of reimplementing it. That keeps exactly one implementation of the
voxel maths in the codebase.

Coordinate frame (matters for every prompt): `fr3_link0`, metres.
    +x  away from the robot base, out across the table
    +y  to the robot's left
    +z  up
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from clear_franka.perception.types import SceneBox

logger = logging.getLogger(__name__)

EE_ALIAS = {"ee", "gripper", "hand", "end effector", "end_effector", "endeffector"}
TABLE_ALIAS = {"table", "workspace", "surface"}


class ObjectResolutionError(ValueError):
    """Raised when generated code asks for an object that is not in the scene.

    Mirrors `voxposer.calvin_interface.ObjectResolutionError`: a hard failure,
    never a silent fall back to the workspace centre, so a hallucinated object
    name surfaces during synthesis instead of steering the arm at empty space.
    """


class BoxSceneInterface:
    """Exposes the detected scene to LLM-generated affordance/avoidance code."""

    def __init__(
        self,
        boxes: list[SceneBox],
        *,
        workspace_bounds_min: np.ndarray,
        workspace_bounds_max: np.ndarray,
        map_size: int = 100,
        ee_pos_world: Optional[np.ndarray] = None,
    ) -> None:
        from voxposer.calvin_interface import CalvinLMPInterface

        self._boxes = {b.name: b for b in boxes}
        self._map_size = int(map_size)
        self._ws_min = np.asarray(workspace_bounds_min, dtype=np.float32)
        self._ws_max = np.asarray(workspace_bounds_max, dtype=np.float32)
        self._ee_pos_world = (
            None if ee_pos_world is None
            else np.asarray(ee_pos_world, dtype=np.float32)
        )

        # Names resolved by detect() since the last reset_resolved(). Lets the
        # caller ask "which objects did this LMP actually touch?", which is how
        # an avoidance map that names the destination gets caught.
        self._resolved: set[str] = set()

        # Grid helpers only — this instance's own detect() is never called.
        self._grid = CalvinLMPInterface({
            "map_size": self._map_size,
            "workspace_bounds_min": self._ws_min,
            "workspace_bounds_max": self._ws_max,
        })

    # ------------------------------------------------------------------
    # Exposed to generated code
    # ------------------------------------------------------------------

    def detect(self, obj_name: str):
        """Resolve an object name to an Observation with voxel + world geometry."""
        from voxposer.utils import Observation

        key = str(obj_name).strip().lower()
        if key in EE_ALIAS:
            return self._ee_observation()

        box = self._match(key)
        if box is None:
            if key in TABLE_ALIAS:
                return self._workspace_observation(key)
            raise ObjectResolutionError(
                f"'{obj_name}' is not in the scene. Known objects: "
                f"{sorted(self._boxes)}"
            )

        self._resolved.add(box.name)
        lo, hi = box.aabb
        return Observation({
            "name": box.name,
            "position": self._to_voxel(box.center),
            "aabb": np.stack([self._to_voxel(lo), self._to_voxel(hi)]),
            "_position_world": box.center.astype(np.float32),
            "obb_center_world": box.center.astype(np.float32),
            "obb_size": box.size.astype(np.float32),
            # SAM boxes are fitted axis-aligned in the base frame, so the OBB
            # rotation is identity and set_voxel_by_box degenerates to an AABB
            # fill — the field is present so the same helper works either way.
            "obb_rotation": np.eye(3, dtype=np.float32),
        })

    def cm2index(self, cm, direction):
        return self._grid.cm2index(cm, direction)

    def set_voxel_by_radius(self, voxel_map, voxel_xyz, radius_cm=0, value=1):
        return self._grid.set_voxel_by_radius(voxel_map, voxel_xyz, radius_cm, value)

    def set_voxel_by_box(self, voxel_map, obj, value=1, pad_cm=0.0):
        return self._grid.set_voxel_by_box(voxel_map, obj, value=value, pad_cm=pad_cm)

    def get_empty_affordance_map(self):
        return self._grid.get_empty_affordance_map()

    def get_empty_avoidance_map(self):
        return self._grid.get_empty_avoidance_map()

    # ------------------------------------------------------------------
    # Scene description handed to the planner
    # ------------------------------------------------------------------

    def object_names(self) -> list[str]:
        return sorted(self._boxes)

    def reset_resolved(self) -> None:
        self._resolved = set()

    def resolved_objects(self) -> set[str]:
        """Boxes `detect()` returned since the last `reset_resolved()`."""
        return set(self._resolved)

    def scene_block(self) -> str:
        """The world-frame box table injected above the planner's query."""
        lines = [
            "# Scene (frame = fr3_link0, metres; +x away from the robot base, "
            "+y left, +z up)",
            f"#   workspace x [{self._ws_min[0]:.2f}, {self._ws_max[0]:.2f}]  "
            f"y [{self._ws_min[1]:.2f}, {self._ws_max[1]:.2f}]  "
            f"z [{self._ws_min[2]:.2f}, {self._ws_max[2]:.2f}]",
        ]
        for name in sorted(self._boxes):
            b = self._boxes[name]
            lo, hi = b.aabb
            lines.append(
                f"#   {name}: center=[{b.center[0]:.3f}, {b.center[1]:.3f}, "
                f"{b.center[2]:.3f}] size=[{b.size[0]:.3f}, {b.size[1]:.3f}, "
                f"{b.size[2]:.3f}] "
                f"x=[{lo[0]:.3f}, {hi[0]:.3f}] y=[{lo[1]:.3f}, {hi[1]:.3f}] "
                f"z=[{lo[2]:.3f}, {hi[2]:.3f}]"
            )
        if self._ee_pos_world is not None:
            p = self._ee_pos_world
            lines.append(
                f"#   gripper: position=[{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _match(self, key: str) -> Optional[SceneBox]:
        if key in self._boxes:
            return self._boxes[key]
        underscored = key.replace(" ", "_")
        if underscored in self._boxes:
            return self._boxes[underscored]
        # Substring match, longest name first, so 'rack' finds 'wine_rack' but
        # 'cabinet' still prefers an exact 'cabinet' over 'cabinet_shelf'.
        for name in sorted(self._boxes, key=len, reverse=True):
            if underscored in name or name in underscored:
                return self._boxes[name]
        return None

    def _to_voxel(self, world_xyz: np.ndarray) -> np.ndarray:
        from voxposer.calvin_interface import pc2voxel

        return pc2voxel(world_xyz, self._ws_min, self._ws_max, self._map_size)

    def _ee_observation(self):
        from voxposer.utils import Observation

        if self._ee_pos_world is None:
            raise ObjectResolutionError(
                "the gripper position was not supplied to this synthesis run"
            )
        vox = self._to_voxel(self._ee_pos_world)
        return Observation({
            "name": "gripper",
            "position": vox,
            "aabb": np.stack([vox, vox]),
            "_position_world": self._ee_pos_world,
        })

    def _workspace_observation(self, name: str):
        from voxposer.utils import Observation

        center = (self._ws_min + self._ws_max) / 2.0
        return Observation({
            "name": name,
            "position": self._to_voxel(center),
            "aabb": np.stack([
                self._to_voxel(self._ws_min), self._to_voxel(self._ws_max)
            ]),
            "_position_world": center.astype(np.float32),
            "obb_center_world": center.astype(np.float32),
            "obb_size": (self._ws_max - self._ws_min).astype(np.float32),
            "obb_rotation": np.eye(3, dtype=np.float32),
        })
