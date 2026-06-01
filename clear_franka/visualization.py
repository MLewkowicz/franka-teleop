"""Optional viser visualization for the Cortado robot description."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlparse

import numpy as np

from clear_franka.geometry import load_T_cam2base, load_T_cam2gripper
from clear_franka.workspace_boxes import WorkspaceBoxEditor


DEFAULT_ARM_JOINT_NAMES = tuple(f"fr3_joint{i}" for i in range(1, 8))
DEFAULT_GRIPPER_JOINT_NAME = "finger_joint"
DEFAULT_GRIPPER_JOINT_CLOSED = 0.8
_GRIPPER_TCP_VISER_PATH = (
    "/cortado/visual/st_ext_005_0585_1/st_gp_001_0019_2/st_ext_002_0045"
    "/st_ext_005_0405_1/st_ext_001_0225/st_ext_002_0225/st_rb_013_0001"
    "/franka_base_mount/fr3_link0/fr3_link1/fr3_link2/fr3_link3/fr3_link4"
    "/fr3_link5/fr3_link6/fr3_link7/fr3_link8/coupling_link/base_link"
    "/robotiq_arg2f_base_link/robotiq_arg2f_tcp"
)


def _hand_camera_frame_path(name: str) -> str:
    """Resolve a hand-camera frame name under the gripper TCP frame."""
    name = str(name)
    if name == _GRIPPER_TCP_VISER_PATH or name.startswith(f"{_GRIPPER_TCP_VISER_PATH}/"):
        return name
    return f"{_GRIPPER_TCP_VISER_PATH}/{name.strip('/')}"

ROBOT_XACRO_ARGS = {
    "robot_name": "cortado",
    "wrist_camera": "true",
    "flip_gripper": "true",
    "camera_mast": "2",
}

ROBOT_DESCRIPTIONS_CACHE_ROOT = Path(
    os.environ.get("ROBOT_DESCRIPTIONS_CACHE", "~/.cache/robot_descriptions")
).expanduser()

_CORTADO_DESCRIPTION_REPO_URL = "https://github.com/MIT-CLEAR-Lab/cortado_description.git"
_CORTADO_DESCRIPTION_COMMIT = "2e2c0f11199c059b8aef633fb5599e8255c64b30"
_CORTADO_DESCRIPTION_MODULE = "cortado_description"
_ROBOTIQ_2F85_REPO_URL = "https://github.com/nickswalker/robotiq-2f-85.git"
_ROBOTIQ_2F85_COMMIT = "aa93c26c09bf5c78d1e6508bbce47657d809d16a"
_ROBOTIQ_2F85_PACKAGE = "robotiq_2f_85_gripper_visualization"
_ROBOTIQ_2F85_MODULE = "robotiq_2f85_v4_description"
_ZED_DESCRIPTION_REPO_URL = "https://github.com/stereolabs/zed-ros2-description.git"
_ZED_DESCRIPTION_COMMIT = "1aa88b319897311d2f33882a4abbc40da8d75ace"
_ZED_DESCRIPTION_PACKAGE = "zed_description"
_ZED_DESCRIPTION_MODULE = "zed_description"


def _get_manual_description(repo_url: str, commit: str | None, module: str, package: str | None):
    from robot_descriptions._cache import clone_to_directory

    repo_dir = ROBOT_DESCRIPTIONS_CACHE_ROOT / "manual" / module
    clone_to_directory(repo_url, str(repo_dir), commit=commit)
    package_path = repo_dir / package if package is not None else repo_dir.resolve()
    return SimpleNamespace(
        REPOSITORY_PATH=str(repo_dir),
        PACKAGE_PATH=str(package_path),
    )


def _get_cortado_description(repo_url: str, commit: str | None):
    description = _get_manual_description(
        repo_url,
        commit,
        _CORTADO_DESCRIPTION_MODULE,
        None,
    )
    _ensure_cortado_description_compatibility(Path(description.REPOSITORY_PATH))
    return description


def _ensure_cortado_description_compatibility(description_root: Path) -> None:
    """Patch known upstream xacro filename drift in the local cache."""
    common_dir = description_root / "robots" / "common"
    for stem in ("cortado_cart", "fr3_robotiq_2f_85"):
        expected = common_dir / f"{stem}.urdf.xacro"
        actual = common_dir / f"{stem}.xacro"
        if expected.exists() or not actual.exists():
            continue
        expected.symlink_to(actual.name)


def _get_package_roots(description_root: Path) -> dict[str, Path]:
    import xacrodoc
    from robot_descriptions import fr3_description

    robotiq_description = _get_manual_description(
        _ROBOTIQ_2F85_REPO_URL,
        _ROBOTIQ_2F85_COMMIT,
        _ROBOTIQ_2F85_MODULE,
        _ROBOTIQ_2F85_PACKAGE,
    )
    zed_description = _get_manual_description(
        _ZED_DESCRIPTION_REPO_URL,
        _ZED_DESCRIPTION_COMMIT,
        _ZED_DESCRIPTION_MODULE,
        None,
    )
    package_roots = {
        "cortado_description": description_root,
        "franka_description": Path(fr3_description.REPOSITORY_PATH).resolve(),
        _ROBOTIQ_2F85_PACKAGE: Path(robotiq_description.PACKAGE_PATH).resolve(),
        _ZED_DESCRIPTION_PACKAGE: Path(zed_description.PACKAGE_PATH).resolve(),
    }
    xacrodoc.packages.update_package_cache(package_roots)
    return package_roots


def _bake_cortado_urdf(description_root: Path) -> Path:
    from robot_descriptions import _xacro

    xacro_path = description_root / "robots" / "cortado.urdf.xacro"
    description_module = SimpleNamespace(
        __name__="cortado_description_cortado_urdf",
        XACRO_PATH=str(xacro_path.resolve()),
        XACRO_ARGS=ROBOT_XACRO_ARGS,
        PACKAGE_PATH=str(description_root.resolve()),
    )
    return Path(_xacro.get_urdf_path(description_module)).resolve()


def _package_filename_handler(fname: str, package_roots: dict[str, Path]) -> str:
    if fname.startswith("file://"):
        parsed = urlparse(fname)
        return unquote(parsed.path)

    fname = unquote(fname)
    if fname.startswith("package://"):
        package, relpath = fname.removeprefix("package://").split("/", 1)
        if package in package_roots:
            return str(package_roots[package] / relpath)
    return fname


class CortadoViserVisualizer:
    """Load Cortado into viser and update its arm joints from Franka state."""

    def __init__(
        self,
        description_root: str | Path | None = None,
        description_repo_url: str = _CORTADO_DESCRIPTION_REPO_URL,
        description_commit: str | None = _CORTADO_DESCRIPTION_COMMIT,
        host: str = "0.0.0.0",
        port: int = 8080,
        root_node_name: str = "/cortado",
    ):
        import viser
        import yourdfpy
        from viser.extras import ViserUrdf

        self._pointcloud_handles = {}

        if description_root is None:
            description = _get_cortado_description(description_repo_url, description_commit)
            description_root = description.REPOSITORY_PATH

        self.description_root = Path(description_root).expanduser().resolve()
        if not self.description_root.exists():
            raise FileNotFoundError(f"Cortado description root does not exist: {self.description_root}")

        package_roots = _get_package_roots(self.description_root)
        urdf_path = _bake_cortado_urdf(self.description_root)
        self.urdf_model = yourdfpy.URDF.load(
            str(urdf_path),
            filename_handler=lambda fname: _package_filename_handler(fname, package_roots),
            force_mesh=True,
        )

        self.server = viser.ViserServer(host=host, port=port)
        self.server.scene.add_frame(root_node_name, show_axes=False)
        self.urdf = ViserUrdf(
            self.server,
            urdf_or_path=self.urdf_model,
            root_node_name=root_node_name,
            load_meshes=True,
        )
        self.urdf.show_visual = True
        self._root_node_name = root_node_name

        self.actuated_joint_names = list(self.urdf.get_actuated_joint_names())
        self._cfg = np.zeros(len(self.actuated_joint_names), dtype=float)
        self._arm_joint_indices = self._resolve_arm_joint_indices()
        self._gripper_joint_index = self._resolve_optional_joint_index(DEFAULT_GRIPPER_JOINT_NAME)
        self.urdf.update_cfg(self._cfg)
        self._plan_line_handle = None
        self._plan_point_handle = None
        self._interpolated_plan_line_handle = None
        self._plan_frame_handles = []
        self.workspace_box_editor: WorkspaceBoxEditor | None = None

        url_host = "localhost" if host in {"0.0.0.0", "::"} else host
        print(f"  [viser] Cortado URDF loaded from {urdf_path}")
        print(f"  [viser] Open http://{url_host}:{port} in your browser.")

    def _resolve_arm_joint_indices(self) -> list[int]:
        indices = []
        for joint_name in DEFAULT_ARM_JOINT_NAMES:
            try:
                indices.append(self.actuated_joint_names.index(joint_name))
            except ValueError as exc:
                names = ", ".join(self.actuated_joint_names)
                raise ValueError(
                    f"URDF is missing expected arm joint {joint_name!r}. "
                    f"Actuated joints: {names}"
                ) from exc
        return indices

    def _resolve_optional_joint_index(self, joint_name: str) -> int | None:
        try:
            return self.actuated_joint_names.index(joint_name)
        except ValueError:
            return None

    def update(self, joint_pos: np.ndarray | None) -> None:
        if joint_pos is None:
            return

        q = np.asarray(joint_pos, dtype=float).reshape(-1)
        if q.shape[0] < len(self._arm_joint_indices):
            raise ValueError(f"Expected at least 7 joint positions, got shape {q.shape}")

        self._cfg[self._arm_joint_indices] = q[:7]
        self.urdf.update_cfg(self._cfg)

    def update_gripper_width(
        self,
        opening_width_m: float,
        max_width_m: float,
    ) -> None:
        if self._gripper_joint_index is None:
            return
        if max_width_m <= 0.0:
            raise ValueError(f"max_width_m must be positive, got {max_width_m!r}")

        opening_width_m = min(max(float(opening_width_m), 0.0), float(max_width_m))
        closed_fraction = 1.0 - opening_width_m / float(max_width_m)
        self._cfg[self._gripper_joint_index] = closed_fraction * DEFAULT_GRIPPER_JOINT_CLOSED
        self.urdf.update_cfg(self._cfg)

    def enable_workspace_box_editor(
        self,
        json_path: str | Path = "data/workspace_boxes.json",
        **kwargs,
    ) -> WorkspaceBoxEditor:
        self.workspace_box_editor = WorkspaceBoxEditor(self, json_path=json_path, **kwargs)
        print(f"  [viser] Workspace box editor saving to {self.workspace_box_editor.path}")
        return self.workspace_box_editor

    def get_workspace_boxes(self) -> list[dict]:
        if self.workspace_box_editor is None:
            return []
        return self.workspace_box_editor.boxes

    def add_camera_frame(
        self,
        name: str,
        T_cam2base: np.ndarray,
        axes_length: float = 0.08,
        axes_radius: float = 0.003,
    ) -> str:
        import viser.transforms

        T_cam2base = np.asarray(T_cam2base, dtype=float)
        if T_cam2base.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 camera transform, got {T_cam2base.shape}")

        T_cam2root = self.urdf_model.get_transform("fr3_link0") @ T_cam2base

        frame = self.server.scene.add_frame(
            name,
            axes_length=axes_length,
            axes_radius=axes_radius,
        )
        frame.wxyz = viser.transforms.SO3.from_matrix(T_cam2root[:3, :3]).wxyz
        frame.position = T_cam2root[:3, 3]
        return name

    def add_camera_frame_from_extrinsics(self, name: str, extrinsics_path: str | Path) -> str:
        T_cam2base = load_T_cam2base(extrinsics_path)
        return self.add_camera_frame(name, T_cam2base)

    def add_hand_camera_frame(
        self,
        name: str,
        T_cam2gripper: np.ndarray,
        axes_length: float = 0.08,
        axes_radius: float = 0.003,
    ) -> str:
        import viser.transforms

        T_cam2gripper = np.asarray(T_cam2gripper, dtype=float)
        if T_cam2gripper.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 camera transform, got {T_cam2gripper.shape}")

        frame_name = _hand_camera_frame_path(name)
        frame = self.server.scene.add_frame(
            frame_name,
            axes_length=axes_length,
            axes_radius=axes_radius,
        )
        frame.wxyz = viser.transforms.SO3.from_matrix(T_cam2gripper[:3, :3]).wxyz
        frame.position = T_cam2gripper[:3, 3]
        return frame_name

    def add_hand_camera_frame_from_extrinsics(self, name: str, extrinsics_path: str | Path) -> str:
        T_cam2gripper = load_T_cam2gripper(extrinsics_path)
        return self.add_hand_camera_frame(name, T_cam2gripper)

    def update_pointcloud(
        self,
        frame_name: str,
        points: np.ndarray,
        colors: np.ndarray,
        point_size: float = 0.01,
    ) -> None:
        cloud_name = f"{frame_name}/pointcloud"
        points = np.asarray(points, dtype=np.float32)
        colors = np.asarray(colors, dtype=np.uint8)
        handle = self._pointcloud_handles.get(cloud_name)
        if handle is None:
            handle = self.server.scene.add_point_cloud(
                name=cloud_name,
                points=points,
                colors=colors,
                point_size=point_size,
            )
            self._pointcloud_handles[cloud_name] = handle
            return

        handle.points = points
        handle.colors = colors
        handle.point_size = point_size

    def update_eef_frame(
        self,
        O_T_EE: np.ndarray,
        name: str = "/eef_debug",
        axes_length: float = 0.08,
        axes_radius: float = 0.003,
    ) -> None:
        import viser.transforms

        # O_T_EE is in the Franka base (fr3_link0) frame. Transform into the
        # viser world frame the same way add_camera_frame does for extrinsics.
        T = self.urdf_model.get_transform("fr3_link0") @ np.asarray(O_T_EE, dtype=float)
        handle = getattr(self, "_eef_frame_handle", None)
        if handle is None:
            handle = self.server.scene.add_frame(
                name, axes_length=axes_length, axes_radius=axes_radius
            )
            self._eef_frame_handle = handle
        handle.wxyz = viser.transforms.SO3.from_matrix(T[:3, :3]).wxyz
        handle.position = T[:3, 3]

    def clear_plan_waypoints(self) -> None:
        for handle in (
            self._plan_line_handle,
            self._plan_point_handle,
            self._interpolated_plan_line_handle,
        ):
            if handle is not None:
                handle.remove()
        self._plan_line_handle = None
        self._plan_point_handle = None
        self._interpolated_plan_line_handle = None

        for handle in self._plan_frame_handles:
            handle.remove()
        self._plan_frame_handles = []

    def update_interpolated_plan_path(
        self,
        positions: np.ndarray | None,
        name: str = "/diffuser_plan/interpolated",
        color: tuple[int, int, int] = (255, 95, 70),
        line_width: float = 2.0,
    ) -> None:
        if self._interpolated_plan_line_handle is not None:
            self._interpolated_plan_line_handle.remove()
            self._interpolated_plan_line_handle = None

        if positions is None:
            return

        positions = np.asarray(positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"Expected positions shape (N, 3), got {positions.shape}")
        if len(positions) < 2:
            return

        T_base_to_root = self.urdf_model.get_transform("fr3_link0")
        R_base_to_root = T_base_to_root[:3, :3]
        points = (R_base_to_root @ positions.T).T + T_base_to_root[:3, 3]
        segments = np.stack([points[:-1], points[1:]], axis=1).astype(np.float32)
        colors = np.full((len(segments), 2, 3), color, dtype=np.uint8)
        self._interpolated_plan_line_handle = self.server.scene.add_line_segments(
            name=name,
            points=segments,
            colors=colors,
            line_width=line_width,
        )

    def update_plan_waypoints(
        self,
        trajectory: np.ndarray | None,
        active_index: int = 0,
        gripper: np.ndarray | None = None,
        name: str = "/diffuser_plan",
        point_size: float = 0.0009,
        line_width: float = 1.0,
        axes_length: float = 0.008,
        axes_radius: float = 0.0015,
        show_axes: bool = False,
    ) -> None:
        if trajectory is None:
            self.clear_plan_waypoints()
            return

        trajectory = np.asarray(trajectory, dtype=float)
        if trajectory.ndim != 2 or trajectory.shape[1] < 3:
            raise ValueError(f"Expected trajectory shape (N, >=3), got {trajectory.shape}")
        if len(trajectory) == 0:
            self.clear_plan_waypoints()
            return

        active_index = int(np.clip(active_index, 0, len(trajectory) - 1))
        T_base_to_root = self.urdf_model.get_transform("fr3_link0")
        R_base_to_root = T_base_to_root[:3, :3]
        points = (R_base_to_root @ trajectory[:, :3].T).T + T_base_to_root[:3, 3]

        if gripper is None:
            gripper_cmd = np.ones(len(points), dtype=bool)
        else:
            gripper_cmd = np.asarray(gripper, dtype=float).reshape(-1)[:len(points)] > 0.0
            if len(gripper_cmd) == 0:
                gripper_cmd = np.ones(len(points), dtype=bool)
            if len(gripper_cmd) < len(points):
                gripper_cmd = np.pad(
                    gripper_cmd,
                    (0, len(points) - len(gripper_cmd)),
                    mode="edge",
                )
        colors = np.where(
            gripper_cmd[:, None],
            np.array([80, 160, 255], dtype=np.uint8),
            np.array([255, 95, 70], dtype=np.uint8),
        )
        colors[:active_index] = (colors[:active_index].astype(np.float32) * 0.45).astype(np.uint8)
        colors[active_index] = np.minimum(
            colors[active_index].astype(np.uint16)
            + np.array([55, 55, 55], dtype=np.uint16),
            255,
        ).astype(np.uint8)

        if self._plan_point_handle is not None:
            self._plan_point_handle.remove()
        self._plan_point_handle = self.server.scene.add_point_cloud(
            name=f"{name}/waypoints",
            points=points.astype(np.float32),
            colors=colors,
            point_size=point_size,
            point_shape="circle",
        )

        if self._plan_line_handle is not None:
            self._plan_line_handle.remove()
        if len(points) > 1:
            segments = np.stack([points[:-1], points[1:]], axis=1).astype(np.float32)
            segment_colors = np.stack([colors[:-1], colors[1:]], axis=1)
            self._plan_line_handle = self.server.scene.add_line_segments(
                name=f"{name}/segments",
                points=segments,
                colors=segment_colors,
                line_width=line_width,
            )
        else:
            self._plan_line_handle = None

        for handle in self._plan_frame_handles:
            handle.remove()
        self._plan_frame_handles = []
        if not show_axes or trajectory.shape[1] < 6:
            return

        import viser.transforms
        from scipy.spatial.transform import Rotation as R

        rotations = R.from_euler("XYZ", trajectory[:, 3:6]).as_matrix()
        for i, (point, rot_base) in enumerate(zip(points, rotations)):
            T = np.eye(4)
            T[:3, :3] = R_base_to_root @ rot_base
            T[:3, 3] = point
            handle = self.server.scene.add_frame(
                f"{name}/frame_{i:02d}",
                axes_length=axes_length * (1.35 if i == active_index else 1.0),
                axes_radius=axes_radius,
            )
            handle.wxyz = viser.transforms.SO3.from_matrix(T[:3, :3]).wxyz
            handle.position = T[:3, 3]
            self._plan_frame_handles.append(handle)
