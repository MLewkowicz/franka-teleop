from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse, unquote

DEFAULT_ARM_JOINT_NAMES = tuple(f"fr3_joint{i}" for i in range(1, 8))
DEFAULT_GRIPPER_JOINT_NAME = "finger_joint"
DEFAULT_GRIPPER_JOINT_CLOSED = 0.8
DEFAULT_GRIPPER_TCP_LINK_NAME = "robotiq_arg2f_tcp"
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
