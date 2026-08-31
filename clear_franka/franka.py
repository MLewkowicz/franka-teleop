"""Shared Franka / Cortado hardware constants and runtime helpers."""

import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

DEFAULT_LOWER_JOINT_LIMITS = [-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508]
DEFAULT_UPPER_JOINT_LIMITS = [2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508]

DEFAULT_ARM_JOINT_NAMES = tuple(f"fr3_joint{i}" for i in range(1, 8))
DEFAULT_GRIPPER_JOINT_NAME = "finger_joint"
DEFAULT_GRIPPER_JOINT_CLOSED = 0.8

# ---------------------------------------------------------------------------
# Cortado robot description
# ---------------------------------------------------------------------------

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


def _get_manual_description(repo_url: str, commit: str | None, module: str, package: str | None):
    from robot_descriptions._cache import clone_to_directory

    repo_dir = ROBOT_DESCRIPTIONS_CACHE_ROOT / "manual" / module
    clone_to_directory(repo_url, str(repo_dir), commit=commit)
    package_path = repo_dir / package if package is not None else repo_dir.resolve()
    return SimpleNamespace(
        REPOSITORY_PATH=str(repo_dir),
        PACKAGE_PATH=str(package_path),
    )


def _ensure_cortado_description_compatibility(description_root: Path) -> None:
    """Patch known upstream xacro filename drift in the local cache."""
    common_dir = description_root / "robots" / "common"
    for stem in ("cortado_cart", "fr3_robotiq_2f_85"):
        expected = common_dir / f"{stem}.urdf.xacro"
        actual = common_dir / f"{stem}.xacro"
        if expected.exists() or not actual.exists():
            continue
        expected.symlink_to(actual.name)


def get_cortado_description(
    repo_url: str = _CORTADO_DESCRIPTION_REPO_URL,
    commit: str | None = _CORTADO_DESCRIPTION_COMMIT,
):
    description = _get_manual_description(repo_url, commit, _CORTADO_DESCRIPTION_MODULE, None)
    _ensure_cortado_description_compatibility(Path(description.REPOSITORY_PATH))
    return description


def bake_cortado_urdf(description_root: Path) -> Path:
    from robot_descriptions import _xacro

    xacro_path = description_root / "robots" / "cortado.urdf.xacro"
    description_module = SimpleNamespace(
        __name__="cortado_description_cortado_urdf",
        XACRO_PATH=str(xacro_path.resolve()),
        XACRO_ARGS=ROBOT_XACRO_ARGS,
        PACKAGE_PATH=str(description_root.resolve()),
    )
    return Path(_xacro.get_urdf_path(description_module)).resolve()


# ---------------------------------------------------------------------------
# Forward kinematics (Cortado / FR3 + Robotiq 2F-85)
# ---------------------------------------------------------------------------

_FK_EE_LINK = "robotiq_arg2f_tcp"
_FK_BASE_LINK = "fr3_link0"

_fk_cache: tuple | None = None
_fk_f_t_ee_cache: np.ndarray | None = None


def _get_fk_model() -> tuple:
    global _fk_cache
    if _fk_cache is not None:
        return _fk_cache

    import yourdfpy

    desc = get_cortado_description()
    urdf_path = bake_cortado_urdf(Path(desc.REPOSITORY_PATH))
    model = yourdfpy.URDF.load(str(urdf_path), load_meshes=False)

    all_joints = list(model.actuated_joint_names)
    arm_indices = [all_joints.index(j) for j in DEFAULT_ARM_JOINT_NAMES]

    # fr3_link0 position in the Cortado world frame is fixed (column joint = 0).
    cfg0 = np.zeros(len(all_joints))
    model.update_cfg(cfg0)
    T_root_to_base = np.linalg.inv(model.get_transform(_FK_BASE_LINK))

    _fk_cache = (model, arm_indices, T_root_to_base, cfg0)
    return _fk_cache


def fk_f_t_ee() -> np.ndarray:
    """Flange-to-TCP offset (fixed; independent of arm config and gripper state).

    Calibrated once against the Cortado URDF (same model `_get_fk_model` loads)
    by comparing its EE-link pose at a reference configuration against franky's
    analytic flange pose at that same configuration. Cached so `fk_ee_poses` can
    use franky's closed-form FK for every row instead of a full URDF graph
    traversal per row — the traversal is the same one-time cost either way,
    just paid for one configuration instead of N.
    """
    global _fk_f_t_ee_cache
    if _fk_f_t_ee_cache is not None:
        return _fk_f_t_ee_cache

    from franky import Affine
    from franky.kinematics import forward_kinematics

    model, arm_indices, T_root_to_base, cfg0 = _get_fk_model()
    q_ref = np.zeros(len(arm_indices))
    cfg = cfg0.copy()
    cfg[arm_indices] = q_ref
    model.update_cfg(cfg)
    o_t_ee_urdf = T_root_to_base @ model.get_transform(_FK_EE_LINK)

    o_t_flange = forward_kinematics(q_ref)
    f_t_ee = o_t_flange.inverse * Affine(o_t_ee_urdf)
    _fk_f_t_ee_cache = f_t_ee.matrix
    return _fk_f_t_ee_cache


def fk_ee_poses(joint_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute O_T_EE for each row of joint_pos (N, 7).

    Returns ee_pos (N, 3) and ee_rot (N, 3, 3) in the fr3_link0 (robot base)
    frame — the same convention as the O_T_EE recorded during demonstration.

    Uses franky's analytic (no robot connection required) forward kinematics
    per row; the flange-to-TCP offset is calibrated once against the Cortado
    URDF (`fk_f_t_ee`) rather than walking the URDF graph for every row.
    """
    from franky import Affine
    from franky.kinematics import forward_kinematics

    f_t_ee = Affine(fk_f_t_ee())
    N = joint_pos.shape[0]
    ee_pos = np.empty((N, 3), dtype=np.float64)
    ee_rot = np.empty((N, 3, 3), dtype=np.float64)
    for i in range(N):
        T = forward_kinematics(joint_pos[i], f_t_ee=f_t_ee).matrix
        ee_pos[i] = T[:3, 3]
        ee_rot[i] = T[:3, :3]
    return ee_pos, ee_rot


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------


def joint_friction_kwargs(cfg, speed: float = 1.0) -> dict:
    """Build the franky ``friction=`` kwarg for a ``start_*_impedance_tracker`` call.

    franky takes a single ``FrictionCompensationParams`` (coulomb, viscous,
    max_torque, velocity_epsilon) instead of the old flat ``friction_coulomb`` /
    ``friction_viscous`` / ... kwargs. zero_franky accepts it as a plain dict and
    converts it server-side. The same payload applies to both the joint and the
    cartesian tracker. Returns ``{}`` when disabled, so callers can splat it.

    ``velocity_epsilon`` is the joint speed at which the ``tanh`` Coulomb term
    saturates. Joint velocities scale ~linearly with replay ``speed``, so the
    configured ``velocity_epsilon`` (interpreted at speed=1.0) is scaled by
    ``speed`` to keep the compensator equally effective across the speed sweep.
    Without this, a slow replay sits inside the smoothing band and friction is
    under-compensated. ``speed`` defaults to 1.0 (no scaling) for non-replay
    callers.
    """
    friction_cfg = cfg.get("joint_friction", {})
    if not friction_cfg or not friction_cfg.get("enabled", False):
        return {}

    coulomb = [float(v) for v in friction_cfg.coulomb]
    viscous = [float(v) for v in friction_cfg.viscous]
    max_torque_cfg = friction_cfg.get("max_torque")
    max_torque = (
        [float(v) for v in max_torque_cfg]
        if max_torque_cfg is not None
        else [1.0] * len(coulomb)
    )
    velocity_epsilon = friction_cfg.get("velocity_epsilon")
    velocity_epsilon = 0.03 if velocity_epsilon is None else float(velocity_epsilon)
    velocity_epsilon = max(velocity_epsilon * float(speed), 1e-6)  # franky requires > 0
    return {
        "friction": {
            "coulomb": coulomb,
            "viscous": viscous,
            "max_torque": max_torque,
            "velocity_epsilon": velocity_epsilon,
        }
    }


def desk_credentials(
    *,
    hostname: str,
    username: str | None = None,
    password: str | None = None,
) -> tuple[str, str, str]:
    username = username or os.environ.get("FRANKA_DESK_USERNAME")
    password = password or os.environ.get("FRANKA_DESK_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "Set desk.username/desk.password in Hydra config or "
            "FRANKA_DESK_USERNAME/FRANKA_DESK_PASSWORD in the environment."
        )
    return hostname, str(username), str(password)


def wait_for_motion_idle(robot, timeout_s: float = 5.0, poll_s: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        if robot.join_motion(min(poll_s, remaining)):
            return True


def stop_tracker_motion(robot, session, join_timeout: float = 1.0, idle_timeout_s: float = 5.0) -> bool:
    session.stop(join_timeout=join_timeout)
    return wait_for_motion_idle(robot, timeout_s=idle_timeout_s)
