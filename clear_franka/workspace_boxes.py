"""Interactive viser editor for workspace boxes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class WorkspaceBoxStore:
    """Mutable workspace boxes with JSON save/load support.

    Box centers and dimensions are expressed in the Franka base frame
    (`fr3_link0`) in metres. Boxes are axis-aligned in that frame.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.boxes: list[dict] = []
        self.dirty = False
        self._next_id = 1
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.boxes = []
            self.dirty = False
            self._next_id = 1
            return

        with open(self.path) as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            raw_boxes = raw.get("boxes", [])
        elif isinstance(raw, list):
            raw_boxes = raw
        else:
            raise ValueError(
                f"Expected workspace boxes JSON object/list, got {type(raw).__name__}"
            )

        self.boxes = []
        for i, raw_box in enumerate(raw_boxes, start=1):
            name = str(raw_box.get("name", f"box_{i:03d}"))
            center = np.asarray(
                raw_box.get("center", raw_box.get("position", [0.5, 0.0, 0.2])),
                dtype=float,
            )
            size = np.asarray(
                raw_box.get("size", raw_box.get("dimensions", [0.2, 0.2, 0.2])),
                dtype=float,
            )
            if center.shape != (3,):
                raise ValueError(
                    f"Box {name!r} center must have 3 values, got {center.shape}"
                )
            if size.shape != (3,):
                raise ValueError(
                    f"Box {name!r} size must have 3 values, got {size.shape}"
                )
            self.boxes.append(
                {
                    "name": name,
                    "center": center.astype(float).tolist(),
                    "size": np.maximum(size.astype(float), 1e-4).tolist(),
                }
            )

        self._next_id = len(self.boxes) + 1
        self.dirty = False

    def add_box(
        self,
        center: tuple[float, float, float],
        size: tuple[float, float, float],
    ) -> int:
        used_names = {box["name"] for box in self.boxes}
        while True:
            name = f"box_{self._next_id:03d}"
            self._next_id += 1
            if name not in used_names:
                break
        self.boxes.append(
            {
                "name": name,
                "center": [float(v) for v in center],
                "size": [max(float(v), 1e-4) for v in size],
            }
        )
        self.dirty = True
        return len(self.boxes) - 1

    def update_center(self, index: int, center: np.ndarray) -> None:
        self.boxes[index]["center"] = np.asarray(center, dtype=float).reshape(3).tolist()
        self.dirty = True

    def update_size(self, index: int, size: np.ndarray) -> None:
        self.boxes[index]["size"] = np.maximum(
            np.asarray(size, dtype=float).reshape(3),
            1e-4,
        ).tolist()
        self.dirty = True

    def delete_box(self, index: int) -> None:
        del self.boxes[index]
        self.dirty = True

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"frame": "fr3_link0", "units": "m", "boxes": self.boxes}
        with open(self.path, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        self.dirty = False


class WorkspaceBoxEditor:
    """Interactive viser editor for axis-aligned workspace boxes."""

    def __init__(
        self,
        visualizer: Any,
        json_path: str | Path,
        default_center: tuple[float, float, float] = (0.5, 0.0, 0.2),
        default_size: tuple[float, float, float] = (0.2, 0.2, 0.2),
        position_limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
            (0.0, 1.0),
            (-0.8, 0.8),
            (-0.1, 1.2),
        ),
        size_limits: tuple[float, float] = (0.01, 1.5),
    ):
        self._visualizer = visualizer
        self._server = visualizer.server
        self._store = WorkspaceBoxStore(json_path)
        self._default_center = tuple(float(v) for v in default_center)
        self._default_size = tuple(float(v) for v in default_size)
        self._position_limits = position_limits
        self._size_limits = size_limits

        self._handles = {}
        self._selected_index: int | None = None
        self._transform_control = None
        self._selection_folder = None
        self._size_sliders = []
        self._center_sliders = []
        self._updating_gui = False

        with self._server.gui.add_folder("Workspace Boxes"):
            self._file_text = self._server.gui.add_text(
                "File", initial_value=str(self._store.path), disabled=True
            )
            self._status_text = self._server.gui.add_text(
                "Status", initial_value=self._status(), disabled=True
            )
            add_button = self._server.gui.add_button("Add Box")
            save_button = self._server.gui.add_button("Save")
            self._deselect_button = self._server.gui.add_button(
                "Deselect", disabled=True
            )
            self._delete_button = self._server.gui.add_button(
                "Delete Selected", disabled=True
            )

        @add_button.on_click
        def _(_) -> None:
            self.add_box(select=True)

        @save_button.on_click
        def _(_) -> None:
            self.save()

        @self._deselect_button.on_click
        def _(_) -> None:
            self.deselect()

        @self._delete_button.on_click
        def _(_) -> None:
            self.delete_selected()

        self._rebuild()

    @property
    def boxes(self) -> list[dict]:
        return self._store.boxes

    @property
    def path(self) -> Path:
        return self._store.path

    def add_box(self, select: bool = True) -> int:
        index = self._store.add_box(self._default_center, self._default_size)
        self._status_text.value = self._status()
        self._rebuild()
        if select:
            self.select(index)
        return index

    def save(self) -> None:
        self._store.save()
        self._status_text.value = self._status()

    def delete_selected(self) -> None:
        if self._selected_index is None:
            return
        index = self._selected_index
        self.deselect()
        self._store.delete_box(index)
        self._rebuild()
        self._status_text.value = self._status()

    def select(self, index: int) -> None:
        if index < 0 or index >= len(self._store.boxes):
            return
        if self._selected_index == index:
            return

        self.deselect()
        self._selected_index = index
        self._deselect_button.disabled = False
        self._delete_button.disabled = False
        self._recreate_box(index, selected=True)
        self._add_transform_control(index)
        self._show_selection_gui(index)

    def deselect(self) -> None:
        if self._selected_index is None:
            return
        index = self._selected_index
        self._selected_index = None
        self._deselect_button.disabled = True
        self._delete_button.disabled = True

        if self._transform_control is not None:
            self._transform_control.remove()
            self._transform_control = None
        self._hide_selection_gui()
        if index < len(self._store.boxes):
            self._recreate_box(index, selected=False)

    def _rebuild(self) -> None:
        if self._transform_control is not None:
            self._transform_control.remove()
            self._transform_control = None
        for handle in self._handles.values():
            handle.remove()
        self._handles.clear()
        self._selected_index = None
        self._hide_selection_gui()
        self._deselect_button.disabled = True
        self._delete_button.disabled = True
        for index in range(len(self._store.boxes)):
            self._recreate_box(index, selected=False)

    def _recreate_box(self, index: int, selected: bool) -> None:
        old_handle = self._handles.pop(index, None)
        if old_handle is not None:
            old_handle.remove()

        box = self._store.boxes[index]
        position, wxyz = self._base_pose_to_root(box["center"])
        color = (255, 215, 64) if selected else (80, 170, 255)
        opacity = 0.35 if selected else 0.22
        handle = self._server.scene.add_box(
            name=f"/workspace_boxes/{box['name']}",
            dimensions=np.asarray(box["size"], dtype=float),
            position=position,
            wxyz=wxyz,
            color=color,
            opacity=opacity,
            side="double",
        )
        self._handles[index] = handle

        @handle.on_click
        def _(_) -> None:
            self.select(index)

    def _add_transform_control(self, index: int) -> None:
        if self._transform_control is not None:
            self._transform_control.remove()
        box = self._store.boxes[index]
        position, wxyz = self._base_pose_to_root(box["center"])
        self._transform_control = self._server.scene.add_transform_controls(
            "/workspace_boxes/transform_control",
            position=position,
            wxyz=wxyz,
            scale=0.2,
            disable_rotations=True,
            translation_limits=self._root_translation_limits(),
            depth_test=False,
        )

        @self._transform_control.on_update
        def _(_) -> None:
            self._on_transform_update()

    def _show_selection_gui(self, index: int) -> None:
        self._hide_selection_gui()
        box = self._store.boxes[index]
        center = np.asarray(box["center"], dtype=float)
        size = np.asarray(box["size"], dtype=float)

        self._selection_folder = self._server.gui.add_folder("Selected Box")
        with self._selection_folder:
            self._server.gui.add_text("Name", initial_value=box["name"], disabled=True)
            self._center_sliders = [
                self._server.gui.add_slider(
                    label,
                    min=float(bounds[0]),
                    max=float(bounds[1]),
                    step=0.005,
                    initial_value=float(value),
                )
                for label, bounds, value in zip(
                    ("X", "Y", "Z"), self._position_limits, center
                )
            ]
            self._size_sliders = [
                self._server.gui.add_slider(
                    label,
                    min=float(self._size_limits[0]),
                    max=float(self._size_limits[1]),
                    step=0.005,
                    initial_value=float(value),
                )
                for label, value in zip(("L", "W", "H"), size)
            ]

        for slider in self._center_sliders:
            @slider.on_update
            def _(_) -> None:
                self._on_center_slider_update()

        for slider in self._size_sliders:
            @slider.on_update
            def _(_) -> None:
                self._on_size_slider_update()

    def _hide_selection_gui(self) -> None:
        if self._selection_folder is not None:
            self._selection_folder.remove()
        self._selection_folder = None
        self._size_sliders = []
        self._center_sliders = []

    def _on_transform_update(self) -> None:
        if self._transform_control is None or self._selected_index is None:
            return
        center = self._root_position_to_base(
            np.asarray(self._transform_control.position, dtype=float)
        )
        self._store.update_center(self._selected_index, center)
        self._update_selected_visual()
        self._set_center_sliders(center)
        self._status_text.value = self._status()

    def _on_center_slider_update(self) -> None:
        if self._updating_gui or self._selected_index is None or not self._center_sliders:
            return
        center = np.array([slider.value for slider in self._center_sliders], dtype=float)
        self._store.update_center(self._selected_index, center)
        position, _ = self._base_pose_to_root(center)
        if self._transform_control is not None:
            self._transform_control.position = position
        self._update_selected_visual()
        self._status_text.value = self._status()

    def _on_size_slider_update(self) -> None:
        if self._updating_gui or self._selected_index is None or not self._size_sliders:
            return
        size = np.array([slider.value for slider in self._size_sliders], dtype=float)
        self._store.update_size(self._selected_index, size)
        self._update_selected_visual()
        self._status_text.value = self._status()

    def _update_selected_visual(self) -> None:
        if self._selected_index is None:
            return
        self._recreate_box(self._selected_index, selected=True)

    def _set_center_sliders(self, center: np.ndarray) -> None:
        if not self._center_sliders:
            return
        self._updating_gui = True
        try:
            for slider, value in zip(self._center_sliders, center):
                slider.value = float(value)
        finally:
            self._updating_gui = False

    def _base_pose_to_root(self, center: list[float] | np.ndarray):
        import viser.transforms

        T_base_to_root = self._visualizer.urdf_model.get_transform("fr3_link0")
        center = np.asarray(center, dtype=float).reshape(3)
        position = T_base_to_root[:3, :3] @ center + T_base_to_root[:3, 3]
        wxyz = viser.transforms.SO3.from_matrix(T_base_to_root[:3, :3]).wxyz
        return position, wxyz

    def _root_position_to_base(self, position: np.ndarray) -> np.ndarray:
        T_base_to_root = self._visualizer.urdf_model.get_transform("fr3_link0")
        return T_base_to_root[:3, :3].T @ (
            np.asarray(position, dtype=float).reshape(3) - T_base_to_root[:3, 3]
        )

    def _root_translation_limits(self):
        corners = np.array(
            [
                [x, y, z]
                for x in self._position_limits[0]
                for y in self._position_limits[1]
                for z in self._position_limits[2]
            ],
            dtype=float,
        )
        T_base_to_root = self._visualizer.urdf_model.get_transform("fr3_link0")
        root_corners = (T_base_to_root[:3, :3] @ corners.T).T + T_base_to_root[:3, 3]
        return tuple(
            (float(root_corners[:, i].min()), float(root_corners[:, i].max()))
            for i in range(3)
        )

    def _status(self) -> str:
        prefix = "Unsaved*" if self._store.dirty else "Saved"
        return f"{prefix} ({len(self._store.boxes)} boxes)"
