"""Deploy a trained 3D Diffuser Actor checkpoint on the real Franka.

Chunked playback: an inference worker grabs two ZED frames + the current
end-effector pose when the executor asks for a new plan, builds a
`core.types.Observation`, and calls `policy.forward(obs)` to produce an
absolute trajectory. The main loop executes that trajectory to completion
before requesting another one. If execution stalls past a timeout, the executor
abandons the old plan and pauses while the worker generates a replacement. The
gripper command is issued when the executed trajectory step flips relative to
what we last commanded.

Stage transitions (grasp → place → done) are driven by SpaceMouse buttons:
    LEFT  short tap   → toggle ENABLED (closed-loop control on/off)
    RIGHT short tap   → advance stage (grasp → place → exit)

Launch:
    uv run python deploy_diffuser_actor.py \\
        deploy.policy_config=/path/to/policy.yaml \\
        deploy.langsteer_path=$HOME/Documents/michal/LangSteer
"""

from __future__ import annotations

import contextlib
import datetime as _dt
from dataclasses import dataclass
import logging
import os
import sys
import threading
import time
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger("deploy_diffuser_actor")

# Hybrid control modes (see DIFFUSER_ACTOR_DEPLOY_DEBUG.md / plan).
TELEOP = "teleop"
INFERENCE = "inference"


def _teleop_target_from_sample(sample, robot_pos, robot_rot, input_filter,
                               global_frame, so3_exp):
    """SpaceMouse sample -> cartesian target. Mirror of teleop.py:376-398.

    The filter modifiers mutate v/w in place (deadband, softmax, scale,
    smoothing). Returns (target_pos, target_rot, v_world, w_world); the world
    twist is fed forward to the impedance tracker exactly like the teleop tool.
    """
    v = np.asarray(sample.xyz, dtype=float).copy()
    w = np.asarray(sample.rpy, dtype=float).copy()
    input_filter._translation_modifier(v)
    input_filter._rotation_modifier(w)
    if global_frame:
        return robot_pos + v, so3_exp(w) @ robot_rot, v, w
    base = robot_rot
    return robot_pos + base @ v, base @ so3_exp(w), base @ v, base @ w


@dataclass
class InferencePlan:
    sequence: int
    stage_idx: int
    epoch: int
    created_at: float
    obs_started_at: float
    trajectory: np.ndarray
    gripper: np.ndarray
    ee_pos_at_obs: np.ndarray
    ee_euler_at_obs: np.ndarray


class LatestPlanSlot:
    """Single-slot handoff from the inference worker to the executor loop."""

    def __init__(self):
        self._lock = threading.Lock()
        self._plan: InferencePlan | None = None

    def publish(self, plan: InferencePlan) -> None:
        with self._lock:
            self._plan = plan

    def latest_after(self, sequence: int) -> InferencePlan | None:
        with self._lock:
            if self._plan is None or self._plan.sequence <= sequence:
                return None
            return self._plan


class DeployTrace:
    """Sidecar recorder for a diffuser-actor deploy session.

    Captures the two things a policy run produces that the executed-trajectory
    episode (TrajectoryRecorder) does not:
      1. every plan published by the inference worker (the raw policy output),
      2. the Cartesian tracking error — commanded reference vs measured EE pose
         — at each execution tick.
    Written to a sidecar HDF5 next to the episode file: ``episode_<wall>_deploy.h5``.
    Thread-safe: ``record_plan`` is called from the worker thread, ``record_error``
    from the executor loop.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._t0 = 0.0
        self._plans: list[dict] = []
        self._err: list[dict] = []

    def start(self) -> None:
        """Anchor the relative time base (call alongside recorder.start())."""
        self._t0 = time.monotonic()

    def record_plan(self, plan: "InferencePlan") -> None:
        with self._lock:
            self._plans.append({
                "sequence": int(plan.sequence),
                "stage_idx": int(plan.stage_idx),
                "epoch": int(plan.epoch),
                "created_at": float(plan.created_at),
                "obs_started_at": float(plan.obs_started_at),
                "trajectory": np.asarray(plan.trajectory, dtype=np.float64).copy(),
                "gripper": np.asarray(plan.gripper, dtype=np.float64).copy(),
                "ee_pos_at_obs": np.asarray(plan.ee_pos_at_obs, dtype=np.float64).copy(),
                "ee_euler_at_obs": np.asarray(plan.ee_euler_at_obs, dtype=np.float64).copy(),
            })

    def record_error(self, *, t, plan_sequence, active_index,
                     target_pos, target_rot, measured_pos, measured_rot) -> None:
        target_pos = np.asarray(target_pos, dtype=np.float64)
        measured_pos = np.asarray(measured_pos, dtype=np.float64)
        target_rot = np.asarray(target_rot, dtype=np.float64)
        measured_rot = np.asarray(measured_rot, dtype=np.float64)
        pos_err = target_pos - measured_pos
        # Geodesic angle between commanded and measured orientation.
        R_err = target_rot @ measured_rot.T
        cos_angle = (np.trace(R_err) - 1.0) / 2.0
        rot_err_rad = float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        with self._lock:
            self._err.append({
                "t": float(t),
                "plan_sequence": int(plan_sequence),
                "active_index": int(active_index),
                "target_pos": target_pos.copy(),
                "measured_pos": measured_pos.copy(),
                "pos_error": pos_err.copy(),
                "pos_error_norm": float(np.linalg.norm(pos_err)),
                "target_rot": target_rot.copy(),
                "measured_rot": measured_rot.copy(),
                "rot_error_rad": rot_err_rad,
            })

    def save(self, path) -> int:
        """Write the sidecar HDF5. Returns the number of plans saved."""
        import h5py

        with self._lock:
            plans = list(self._plans)
            err = list(self._err)
            t0 = self._t0

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            f.attrs["num_plans"] = len(plans)
            f.attrs["num_error_samples"] = len(err)

            # ---- per-tick Cartesian tracking error ----
            g = f.create_group("tracking_error")
            if err:
                def col(key):
                    return np.stack([e[key] for e in err])
                g.create_dataset("timestamps",
                                 data=np.array([e["t"] - t0 for e in err], dtype=np.float64))
                g.create_dataset("plan_sequence",
                                 data=np.array([e["plan_sequence"] for e in err], dtype=np.int32))
                g.create_dataset("active_index",
                                 data=np.array([e["active_index"] for e in err], dtype=np.int32))
                g.create_dataset("target_pos", data=col("target_pos"))
                g.create_dataset("measured_pos", data=col("measured_pos"))
                g.create_dataset("pos_error", data=col("pos_error"))
                g.create_dataset("pos_error_norm",
                                 data=np.array([e["pos_error_norm"] for e in err], dtype=np.float64))
                g.create_dataset("target_rot", data=col("target_rot"))
                g.create_dataset("measured_rot", data=col("measured_rot"))
                g.create_dataset("rot_error_rad",
                                 data=np.array([e["rot_error_rad"] for e in err], dtype=np.float64))

            # ---- raw policy plans (one subgroup per published plan) ----
            pg = f.create_group("plans")
            for p in plans:
                sub = pg.create_group(f"plan_{p['sequence']:04d}")
                sub.attrs["sequence"] = p["sequence"]
                sub.attrs["stage_idx"] = p["stage_idx"]
                sub.attrs["epoch"] = p["epoch"]
                sub.attrs["created_at_s"] = p["created_at"] - t0
                sub.attrs["obs_started_at_s"] = p["obs_started_at"] - t0
                sub.create_dataset("trajectory", data=p["trajectory"],
                                   compression="gzip", compression_opts=1)
                sub.create_dataset("gripper", data=p["gripper"],
                                   compression="gzip", compression_opts=1)
                sub.create_dataset("ee_pos_at_obs", data=p["ee_pos_at_obs"])
                sub.create_dataset("ee_euler_at_obs", data=p["ee_euler_at_obs"])
        return len(plans)

# ---------------------------------------------------------------------------
# Sys.path wiring — LangSteer is installed on the robot machine and provides
# the policy/model code; we add it to PYTHONPATH before importing the policy.
# ---------------------------------------------------------------------------

def _wire_langsteer(langsteer_path: str) -> None:
    p = Path(langsteer_path).expanduser().resolve()
    if not (p / "policies" / "diffuser_actor.py").is_file():
        raise RuntimeError(
            f"deploy.langsteer_path={p} does not look like a LangSteer checkout "
            "(missing policies/diffuser_actor.py). Set it to your LangSteer "
            "repo root."
        )
    sys.path.insert(0, str(p))


# ---------------------------------------------------------------------------
# Policy instantiation
# ---------------------------------------------------------------------------

def _build_policy(deploy_cfg: DictConfig):
    """Load the policy yaml, build the right variant, load the checkpoint."""
    from policies.diffuser_actor import build_diffuser_actor_policy

    policy_cfg = OmegaConf.load(deploy_cfg.policy_config)

    checkpoint = policy_cfg.get("ckpt_path")

    # The factory reads use_primitive_id / use_object_id from cfg, so make
    # sure the policy yaml has them.
    policy = build_diffuser_actor_policy(policy_cfg)
    policy.load_checkpoint(checkpoint)
    policy.reset()

    # gripper_loc_bounds defines the [-1, 1] -> meters mapping the model
    # unnormalises against. Anything outside these bounds means the model
    # extrapolated beyond the training workspace.
    policy_loc_bounds = None
    bounds_cfg = policy_cfg.get("gripper_loc_bounds", None)
    if bounds_cfg is not None:
        policy_loc_bounds = np.asarray(bounds_cfg, dtype=np.float64).reshape(2, 3)

    return policy, bool(policy_cfg.get("relative", False)), policy_loc_bounds


def _build_steering(deploy_cfg: DictConfig, policy, policy_relative, policy_loc_bounds):
    """Build the optional steering module and wire its schedulers.

    Returns (steering, steer_stage_indices). Disabled -> (None, set()).
    Enabled via a `deploy.steering` block with enabled=true and a target_euler.

    Rotation steering pulls the predicted EE rotation toward a fixed absolute
    orientation (the inverted-wrist place); it works at the trained 25-step
    regime because guidance actively biases the denoiser, not the sampling noise.

    When `steering.position.enabled` is also true, a `CombinedBoxSteering` is
    built instead: it keeps that rotation steering and adds a positional branch
    driven by hardcoded value maps over the workspace boxes (pulls toward the
    wine rack, pushes off the cabinet / the volume beneath it). The position
    branch needs the policy's gripper_loc_bounds + relative flag to map the
    predicted (relative, normalized) positions into the value map's world frame,
    so they are merged into the cfg here.
    """
    steer_cfg = deploy_cfg.get("steering", None)
    if not steer_cfg or not steer_cfg.get("enabled", False):
        return None, set()

    container = OmegaConf.to_container(steer_cfg, resolve=True)
    pos_cfg = steer_cfg.get("position", None)
    if pos_cfg and pos_cfg.get("enabled", False):
        from clear_franka.box_field_steering import CombinedBoxSteering

        container["relative"] = bool(policy_relative)
        container["gripper_loc_bounds"] = (
            policy_loc_bounds.tolist() if policy_loc_bounds is not None else None
        )
        steering = CombinedBoxSteering(container)
    else:
        from steering.target_rotation import TargetRotationSteering

        steering = TargetRotationSteering(container)

    # alpha_bar for the Tweedie x0 estimate comes from the schedulers.
    steering.set_rotation_scheduler(policy._model.rotation_noise_scheduler)
    steering.set_position_scheduler(policy._model.position_noise_scheduler)
    # Which stage indices to steer; default = place stage only (idx 1).
    stages = steer_cfg.get("stage_indices", [1])
    return steering, {int(s) for s in stages}


# ---------------------------------------------------------------------------
# Camera + extrinsics setup
# ---------------------------------------------------------------------------

def _setup_cameras(cfg: DictConfig):
    from clear_franka.camera import make_zed_camera, enabled_camera_names
    from clear_franka.diffuser_actor_io import (
        CameraPreprocessor, load_extrinsics_json,
    )

    names = enabled_camera_names(cfg)
    if "hand" not in names or "third_person" not in names:
        raise RuntimeError(
            f"deploy needs both 'hand' and 'third_person' enabled in cameras.* "
            f"of conf/config.yaml; got: {names}"
        )

    cam_hand = make_zed_camera(cfg, "hand")
    cam_tp = make_zed_camera(cfg, "third_person")

    hand_ext = load_extrinsics_json(cfg.cameras.hand.extrinsics_path)
    tp_ext = load_extrinsics_json(cfg.cameras.third_person.extrinsics_path)
    pre_hand = CameraPreprocessor(hand_ext, mount="hand")
    pre_tp = CameraPreprocessor(tp_ext, mount="third_person")
    return cam_hand, cam_tp, pre_hand, pre_tp


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------

def _build_observation(rgb_tp_200, pcd_tp_200, rgb_hand_200, pcd_hand_200,
                       ee_pos, ee_rot, gripper_command):
    """Pack into the core.types.Observation the DiffuserActor expects.

    Keys match the policy's cfg.cameras = ["front", "wrist"]; values:
        rgb["front"], rgb["wrist"]  — (200, 200, 3) uint8 RGB
        depth["front"], depth["wrist"]  — (200, 200, 3) float32 base-frame XYZ
        ee_pose  — concat(xyz, euler_XYZ, gripper)   shape (7,)
    """
    from core.types import Observation
    from clear_franka.diffuser_actor_io import ee_rot_to_euler_xyz

    euler = ee_rot_to_euler_xyz(ee_rot)  # (3,) pytorch3d "XYZ" intrinsic
    ee_pose = np.concatenate([
        np.asarray(ee_pos, dtype=np.float32),
        euler,
        np.array([float(gripper_command)], dtype=np.float32),
    ])

    return Observation(
        rgb={"front": rgb_tp_200, "wrist": rgb_hand_200},
        depth={"front": pcd_tp_200, "wrist": pcd_hand_200},
        proprio=np.zeros(0, dtype=np.float32),  # unused by DiffuserActor
        ee_pose=ee_pose,
        instruction="",                          # unused in primitive+object mode
    )


def _read_ee_pose_from_state(state: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    O_T_EE = np.asarray(state["O_T_EE"], dtype=np.float64).reshape(4, 4)
    ee_pos = O_T_EE[:3, 3].copy()
    ee_rot = O_T_EE[:3, :3].copy()
    return ee_pos, ee_rot, O_T_EE


def _make_T_gripper_to_base(ee_pos: np.ndarray, ee_rot: np.ndarray) -> np.ndarray:
    T_g2b = np.eye(4)
    T_g2b[:3, :3] = ee_rot
    T_g2b[:3, 3] = ee_pos
    return T_g2b


def _update_visualizer_robot_state(visualizer, state: dict | None) -> None:
    if visualizer is None or state is None:
        return

    joint_pos = np.asarray(state["q"], dtype=float) if "q" in state else None
    visualizer.update(joint_pos)
    if "O_T_EE" in state:
        visualizer.update_eef_frame(np.asarray(state["O_T_EE"], dtype=float).reshape(4, 4))


def _extract_gripper_plan(action, horizon: int) -> np.ndarray:
    trajectory = np.asarray(action.trajectory)
    if trajectory.ndim == 2 and trajectory.shape[1] >= 7:
        return trajectory[:horizon, 6].astype(np.float64)
    return np.full(horizon, float(action.gripper), dtype=np.float64)


def _plan_start_index(
    plan: InferencePlan,
    current_ee_pos: np.ndarray,
    plan_dt: float,
    now: float,
) -> int:
    horizon = len(plan.trajectory)
    if horizon <= 1:
        return 0

    distances = np.linalg.norm(plan.trajectory[:, :3] - current_ee_pos[None, :], axis=1)
    closest_next = int(np.argmin(distances)) + 1
    latency_skip = int(max(0.0, now - plan.obs_started_at) / plan_dt)
    return min(max(1, closest_next, latency_skip), horizon - 1)


def _make_cartesian_trajectory_for_plan(
    plan: InferencePlan,
    start_index: int,
    plan_dt: float,
    euler_to_matrix_fn,
    *,
    max_linear_vel: float,
    max_angular_vel: float,
    min_segment_dt: float,
    current_ee_pos: np.ndarray | None = None,
    current_ee_euler: np.ndarray | None = None,
):
    """Build a retimed Cartesian trajectory from the plan suffix.

    When `current_ee_pos`/`current_ee_euler` are provided, the current EE pose
    is prepended at t=0 so the executor's first commanded target equals the
    current EE — preventing the impedance controller from receiving a sudden
    jump to plan[start_index] (which can be 10-20+ cm away after position
    steering or large diffusion-time skip). The retime then constrains the
    bridge segment (current EE → plan[start_index]) by `max_linear_vel`, so
    the commanded velocity is bounded to a safe rate regardless of how far
    the policy/steering placed the first waypoint.
    """
    from clear_franka.cartesian_trajectory import CartesianTrajectory

    suffix = plan.trajectory[start_index:, :6]
    if current_ee_pos is not None and current_ee_euler is not None:
        current = np.concatenate([
            np.asarray(current_ee_pos, dtype=np.float64).reshape(3),
            np.asarray(current_ee_euler, dtype=np.float64).reshape(3),
        ])
        suffix = np.vstack([current[None, :], suffix])
    if len(suffix) == 1:
        suffix = np.vstack([suffix, suffix])
    times = np.arange(len(suffix), dtype=np.float64) * float(plan_dt)
    trajectory = CartesianTrajectory.from_euler_xyz(
        suffix,
        times,
        euler_to_matrix_fn=euler_to_matrix_fn,
        smooth_orientation=True,
    )
    return trajectory.retime(
        max_linear_vel=max_linear_vel,
        max_angular_vel=max_angular_vel,
        min_segment_dt=min_segment_dt,
    )


def _sample_cartesian_trajectory_positions(
    trajectory,
    dt: float,
    max_samples: int = 500,
) -> np.ndarray:
    duration = float(trajectory.duration)
    if duration <= 0.0:
        position, _rotation = trajectory.interpolate(0.0)
        return position.reshape(1, 3)

    num_samples = max(int(np.ceil(duration / float(dt))) + 1, 2)
    num_samples = min(num_samples, int(max_samples))
    times = np.linspace(0.0, duration, num_samples)
    return np.stack([trajectory.interpolate(t)[0] for t in times], axis=0)


# ---------------------------------------------------------------------------
# Outlier waypoint filter + per-plan diagnostics.
# Reconstructed from the deploy-debug session transcript; see
# DIFFUSER_ACTOR_DEPLOY_DEBUG.md sections 5 (interior median filter),
# 7 (diagnostics) and 11 (endpoint extrapolation).
# ---------------------------------------------------------------------------

# Running ee->wp[0] statistics across plans (the underfit/mean-collapse
# signature is a near-constant ee->wp0 vector plan after plan).
_EE_TO_WP0_STATS: dict = {"n": 0, "sum": np.zeros(3, dtype=np.float64),
                          "sum_sq": np.zeros(3, dtype=np.float64)}


def _filter_outlier_waypoints(
    pos_traj: np.ndarray,
    euler_traj: np.ndarray,
    gripper_plan: np.ndarray,
    *,
    pos_thresh_m: float,
    eul_thresh_rad: float,
    window_radius: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[float]]:
    """Detect and replace isolated outlier waypoints in a policy trajectory.

    Interior waypoints: flag i when its xyz (or geodesic orientation) deviates
    from the component-wise median of the [i-W, i+W] window (excluding i) by
    more than pos_thresh_m / eul_thresh_rad. The median is robust up to W
    outliers per 2W-window, so single spikes, short runs, and an outlier's
    innocent neighbors are distinguished correctly.

    Endpoints (0, N-1): a median window would false-positive on a genuinely
    moving endpoint, so they are flagged by linear extrapolation from their two
    interior neighbors instead (expected wp[0] = 2*wp[1] - wp[2]). This catches
    the safety-critical wp[0] diffusion spike that the executor would start on.

    Replacement: xyz by linear interp between the nearest non-outlier anchors,
    orientation by SLERP; flagged endpoints snap to the nearest good neighbor.
    The gripper bit snaps to anchor consensus only when both anchors agree, so
    sustained openness transitions survive but isolated bit-flips are cleaned.

    Returns patched (pos, euler, gripper), the sorted patched indices, and each
    outlier's xyz deviation (for logging).
    """
    from scipy.spatial.transform import Rotation, Slerp

    pos = pos_traj.copy()
    eul = euler_traj.copy()
    grip = gripper_plan.copy()
    n = pos.shape[0]
    if n < 3:
        return pos, eul, grip, [], []

    is_outlier = np.zeros(n, dtype=bool)
    out_devs: dict[int, float] = {}

    def _rot_dev(i: int, eul_ref: np.ndarray) -> float:
        return float((Rotation.from_euler("XYZ", eul[i]) *
                      Rotation.from_euler("XYZ", eul_ref).inv()).magnitude())

    # Endpoints via linear extrapolation from the two adjacent interior points.
    for end, a, b in ((0, 1, 2), (n - 1, n - 2, n - 3)):
        pos_pred = 2.0 * pos[a] - pos[b]
        eul_pred = 2.0 * eul[a] - eul[b]
        pos_dev = float(np.linalg.norm(pos[end] - pos_pred))
        if pos_dev > pos_thresh_m or _rot_dev(end, eul_pred) > eul_thresh_rad:
            is_outlier[end] = True
            out_devs[end] = pos_dev

    # Interior via robust median-of-window.
    for i in range(1, n - 1):
        lo = max(0, i - window_radius)
        hi = min(n, i + window_radius + 1)
        window = [k for k in range(lo, hi) if k != i]
        if len(window) < 2:
            continue
        pos_med = np.median(pos[window], axis=0)
        pos_dev = float(np.linalg.norm(pos[i] - pos_med))
        if pos_dev > pos_thresh_m or _rot_dev(i, np.median(eul[window], axis=0)) > eul_thresh_rad:
            is_outlier[i] = True
            out_devs[i] = pos_dev

    patched = sorted(out_devs.keys())
    if not patched:
        return pos, eul, grip, patched, []

    def _anchors(i: int) -> tuple[int, int]:
        left = i - 1
        while left >= 0 and is_outlier[left]:
            left -= 1
        right = i + 1
        while right < n and is_outlier[right]:
            right += 1
        return left, right

    for i in patched:
        left, right = _anchors(i)
        if left < 0 and right >= n:
            continue
        if left < 0:                       # leading endpoint: snap to first good
            pos[i], eul[i], grip[i] = pos[right], eul[right], grip[right]
            continue
        if right >= n:                     # trailing endpoint: snap to last good
            pos[i], eul[i], grip[i] = pos[left], eul[left], grip[left]
            continue
        t = (i - left) / (right - left)
        pos[i] = (1.0 - t) * pos[left] + t * pos[right]
        rots = Rotation.from_euler("XYZ", np.stack([eul[left], eul[right]]))
        eul[i] = Slerp([0.0, 1.0], rots)(t).as_euler("XYZ")
        if grip[left] == grip[right]:
            grip[i] = grip[left]

    return pos, eul, grip, patched, [out_devs[i] for i in patched]


def _format_waypoint_table(pos_traj: np.ndarray, gripper_bin: np.ndarray,
                           highlight_idx: int | None = None) -> str:
    """Compact (horizon)-row table of xyz + gripper, with optional spotlight."""
    lines = ["    idx |    x       y       z   | gripper"]
    for i in range(pos_traj.shape[0]):
        marker = " <--" if i == highlight_idx else ""
        lines.append(
            f"    {i:>3} | {pos_traj[i, 0]:>7.3f} {pos_traj[i, 1]:>7.3f} "
            f"{pos_traj[i, 2]:>7.3f} |   {int(gripper_bin[i])}{marker}"
        )
    return "\n".join(lines)


def _log_plan_diagnostics(
    plan: InferencePlan,
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    policy_loc_bounds: np.ndarray | None,
    policy_relative: bool,
    far_jump_thresh_m: float = 0.10,
    big_step_thresh_m: float = 0.05,
) -> None:
    """Print per-plan stats useful for diagnosing 'far jump' predictions.

    Logs:
      - EE pose at the moment the observation was captured
      - First/middle/last waypoints (absolute base-frame xyz, meters)
      - Δ from EE-at-obs to first waypoint (the 'far jump' indicator —
        retime() only slows execution, it does NOT shorten this jump)
      - Intra-plan span ||wp[-1] - wp[0]|| (stationary => underfit/collapse)
      - Per-step linear & angular deltas (max, mean, total path length)
      - Out-of-bounds counts vs cfg.deploy.workspace_lo/hi (safety clip
        applied at execute time) and vs policy.gripper_loc_bounds (the
        model's own unnormalise range — outside this means extrapolation)
      - Gripper plan as a 0/1 string + transition count
      - Running mean/std of ee->wp0 and a relative-mode underfit check
    """
    trajectory = plan.trajectory
    pos_traj = trajectory[:, :3]
    euler_traj = trajectory[:, 3:6]
    ee = plan.ee_pos_at_obs
    horizon = pos_traj.shape[0]

    delta_first = pos_traj[0] - ee
    delta_first_norm = float(np.linalg.norm(delta_first))

    if horizon >= 2:
        step_dp = np.diff(pos_traj, axis=0)
        step_lin = np.linalg.norm(step_dp, axis=1)
        step_eul = np.linalg.norm(np.diff(euler_traj, axis=0), axis=1)
    else:
        step_lin = np.zeros(0, dtype=np.float64)
        step_eul = np.zeros(0, dtype=np.float64)

    max_lin = float(step_lin.max()) if step_lin.size else 0.0
    mean_lin = float(step_lin.mean()) if step_lin.size else 0.0
    total_lin = float(step_lin.sum()) if step_lin.size else 0.0
    max_eul = float(step_eul.max()) if step_eul.size else 0.0
    mean_eul = float(step_eul.mean()) if step_eul.size else 0.0

    oob_ws = ((pos_traj < workspace_lo) | (pos_traj > workspace_hi)).any(axis=1)
    oob_ws_count = int(oob_ws.sum())
    # In relative mode, policy_loc_bounds describes the range of DELTAS the
    # model unnormalises into. In absolute mode, it describes absolute base-
    # frame positions. Compare the right quantity.
    if policy_loc_bounds is not None:
        if policy_relative:
            check_vals = pos_traj - ee
            bounds_label = "policy-delta-bounds"
        else:
            check_vals = pos_traj
            bounds_label = "policy-abs-bounds"
        oob_policy = (
            (check_vals < policy_loc_bounds[0]) | (check_vals > policy_loc_bounds[1])
        ).any(axis=1)
        oob_policy_count = int(oob_policy.sum())
        oob_policy_str = f"{oob_policy_count}/{horizon} [{bounds_label}]"
    else:
        oob_policy_str = "n/a"

    gripper_bin = (plan.gripper > 0.0).astype(np.int8)
    transitions = int(np.abs(np.diff(gripper_bin)).sum()) if gripper_bin.size > 1 else 0

    # Intra-plan span: ||wp[-1] - wp[0]||. Underfit / mean-collapsed policies
    # produce near-stationary trajectories where this stays a few mm even when
    # the task demands many cm of motion.
    span_vec = pos_traj[-1] - pos_traj[0]
    span_norm = float(np.linalg.norm(span_vec))

    # Normalized-space midpoint check for relative mode. If the model is
    # unconditioned/underfit it outputs values near normalized [0, 0, 0]; with
    # asymmetric gripper_loc_bounds that unnormalizes to a non-zero meter
    # offset. Comparing predicted wp[0] (relative delta) against the bounds'
    # midpoint tells us how close the model is to that "do-nothing" output.
    midpoint_xyz = None
    pred_rel = None
    pred_norm = None
    if policy_loc_bounds is not None and policy_relative:
        lo, hi = policy_loc_bounds[0], policy_loc_bounds[1]
        midpoint_xyz = 0.5 * (hi + lo)
        pred_rel = pos_traj[0] - ee
        span = hi - lo
        span_safe = np.where(span > 0, span, 1.0)
        pred_norm = 2.0 * (pred_rel - lo) / span_safe - 1.0

    # Running statistics across plans so we can see if ee->wp0 is constant
    # (underfit signature) or actually varies plan-to-plan.
    _EE_TO_WP0_STATS["n"] += 1
    _EE_TO_WP0_STATS["sum"] += delta_first
    _EE_TO_WP0_STATS["sum_sq"] += delta_first ** 2
    n = _EE_TO_WP0_STATS["n"]
    mean_d = _EE_TO_WP0_STATS["sum"] / n
    var_d = _EE_TO_WP0_STATS["sum_sq"] / n - mean_d ** 2
    std_d = np.sqrt(np.maximum(var_d, 0.0))

    middle_idx = horizon // 2
    np.set_printoptions(suppress=True)
    logger.info(
        "  plan %d diagnostics:\n"
        "    ee_at_obs xyz=%s euler=%s\n"
        "    waypoint[0]   xyz=%s euler=%s\n"
        "    waypoint[%d]  xyz=%s euler=%s\n"
        "    waypoint[%d]  xyz=%s euler=%s\n"
        "    ee->wp0 Δxyz=%s |Δ|=%.3fm\n"
        "    intra-plan span (wp[-1]-wp[0])=%s |span|=%.4fm\n"
        "    step Δxyz  max=%.3fm mean=%.3fm total=%.3fm\n"
        "    step Δeul  max=%.3frad mean=%.3frad\n"
        "    out-of-workspace=%d/%d  out-of-policy-bounds=%s\n"
        "    gripper=%s transitions=%d\n"
        "    running ee->wp0 mean=%s std=%s (n=%d)",
        plan.sequence,
        np.array2string(ee, precision=3),
        np.array2string(plan.ee_euler_at_obs, precision=3),
        np.array2string(pos_traj[0], precision=3),
        np.array2string(euler_traj[0], precision=3),
        middle_idx,
        np.array2string(pos_traj[middle_idx], precision=3),
        np.array2string(euler_traj[middle_idx], precision=3),
        horizon - 1,
        np.array2string(pos_traj[-1], precision=3),
        np.array2string(euler_traj[-1], precision=3),
        np.array2string(delta_first, precision=3),
        delta_first_norm,
        np.array2string(span_vec, precision=4),
        span_norm,
        max_lin, mean_lin, total_lin,
        max_eul, mean_eul,
        oob_ws_count, horizon, oob_policy_str,
        "".join(str(int(g)) for g in gripper_bin), transitions,
        np.array2string(mean_d, precision=4),
        np.array2string(std_d, precision=4),
        n,
    )

    if midpoint_xyz is not None:
        # Distance from the model's predicted relative wp[0] to the
        # "do-nothing" midpoint. Small => model output ≈ unconditioned mean.
        dist_to_midpoint = float(np.linalg.norm(pred_rel - midpoint_xyz))
        logger.info(
            "    underfit check: bounds_midpoint_xyz=%s  pred_rel_xyz=%s\n"
            "                    pred_normalized=%s  |pred_norm|=%.3f"
            "  |pred_rel - midpoint|=%.4fm",
            np.array2string(midpoint_xyz, precision=4),
            np.array2string(pred_rel, precision=4),
            np.array2string(pred_norm, precision=3),
            float(np.linalg.norm(pred_norm)),
            dist_to_midpoint,
        )

    if delta_first_norm > far_jump_thresh_m:
        logger.warning(
            "  plan %d FAR-JUMP: first waypoint is %.3fm from current EE "
            "(threshold %.3fm); retime will slow but not shorten this jump",
            plan.sequence, delta_first_norm, far_jump_thresh_m,
        )
    if step_lin.size and max_lin > big_step_thresh_m:
        worst_idx = int(np.argmax(step_lin))
        logger.warning(
            "  plan %d BIG-STEP: %.3fm between waypoint %d->%d (threshold %.3fm)\n"
            "  full waypoint table (xyz + gripper):\n%s",
            plan.sequence, max_lin, worst_idx, worst_idx + 1, big_step_thresh_m,
            _format_waypoint_table(pos_traj, gripper_bin, highlight_idx=worst_idx + 1),
        )
        # Diffusion couples openness with position; flag co-occurrence so we
        # can tell true cross-modal artifacts from raw position noise.
        if transitions > 0:
            transition_indices = (np.where(np.abs(np.diff(gripper_bin)) > 0)[0] + 1).tolist()
            if any(abs(ti - (worst_idx + 1)) <= 1 for ti in transition_indices):
                logger.warning(
                    "  plan %d BIG-STEP CO-OCCURS WITH GRIPPER TRANSITION at "
                    "indices=%s — likely cross-modal denoising artifact",
                    plan.sequence, transition_indices,
                )
    if oob_ws_count > 0:
        oob_indices = np.where(oob_ws)[0].tolist()
        logger.warning(
            "  plan %d OOB-WORKSPACE: %d/%d waypoints outside cfg.deploy.workspace; "
            "indices=%s (will be clipped at execute time)",
            plan.sequence, oob_ws_count, horizon, oob_indices,
        )


def _start_inference_worker(
    *,
    policy,
    policy_lock: threading.Lock,
    robot,
    cam_hand,
    cam_tp,
    pre_hand,
    pre_tp,
    latest_plan: LatestPlanSlot,
    enabled_event: threading.Event,
    request_event: threading.Event,
    stop_event: threading.Event,
    stage_state: dict[str, int],
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    policy_loc_bounds: np.ndarray | None,
    policy_relative: bool,
    outlier_filter_enabled: bool,
    outlier_pos_thresh_m: float,
    outlier_eul_thresh_rad: float,
    steering=None,
    steer_stage_indices: set[int] | None = None,
    deploy_trace: "DeployTrace | None" = None,
) -> threading.Thread:
    steer_stage_indices = steer_stage_indices or set()
    def worker() -> None:
        sequence = 0
        while not stop_event.is_set():
            if not enabled_event.wait(0.05):
                continue
            if not request_event.wait(0.05):
                continue
            request_event.clear()
            if not enabled_event.is_set():
                continue

            try:
                obs_started_at = time.monotonic()
                state = robot.wait_for_state(timeout=1.0)
                ee_pos, ee_rot, _O_T_EE = _read_ee_pose_from_state(state)
                T_g2b = _make_T_gripper_to_base(ee_pos, ee_rot)

                hand_frame = cam_hand.grab_frame()
                tp_frame = cam_tp.grab_frame()
                if hand_frame is None or tp_frame is None:
                    logger.warning("Camera grab failed — skipping inference")
                    request_event.set()
                    continue
                rgb_hand_full, depth_hand_full = hand_frame
                rgb_tp_full, depth_tp_full = tp_frame

                rgb_hand_200, pcd_hand_200 = pre_hand.process(
                    rgb_hand_full, depth_hand_full, T_g2b
                )
                rgb_tp_200, pcd_tp_200 = pre_tp.process(
                    rgb_tp_full, depth_tp_full
                )

                from clear_franka.diffuser_actor_io import ee_rot_to_euler_xyz
                ee_euler_at_obs = ee_rot_to_euler_xyz(ee_rot)

                obs = _build_observation(
                    rgb_tp_200, pcd_tp_200,
                    rgb_hand_200, pcd_hand_200,
                    ee_pos, ee_rot, stage_state["gripper_cmd"],
                )
                forward_started_at = time.monotonic()
                with policy_lock:
                    stage_idx = stage_state["idx"]
                    epoch = stage_state["epoch"]
                    active_steering = (
                        steering
                        if (steering is not None and stage_idx in steer_stage_indices)
                        else None
                    )
                    action = policy.forward(obs, steering=active_steering)
                forward_s = time.monotonic() - forward_started_at

                # DiffuserActorBasePolicy.forward() already converts relative
                # model outputs into absolute poses before returning Action.
                trajectory = np.asarray(action.trajectory, dtype=np.float64).copy()
                horizon = trajectory.shape[0]
                if not enabled_event.is_set():
                    continue

                gripper_plan_raw = _extract_gripper_plan(action, horizon)
                gripper_plan = gripper_plan_raw

                # Filter isolated outlier waypoints (single-index diffusion
                # spikes or gripper-coupled teleports, including the safety-
                # critical wp[0]/wp[N-1]). The clean trajectory is what the
                # executor sees; diagnostics below report on the patch.
                if outlier_filter_enabled and trajectory.shape[1] >= 6:
                    pos_filt, eul_filt, grip_filt, patched_idx, patched_pos_devs = (
                        _filter_outlier_waypoints(
                            trajectory[:, :3], trajectory[:, 3:6],
                            gripper_plan_raw,
                            pos_thresh_m=outlier_pos_thresh_m,
                            eul_thresh_rad=outlier_eul_thresh_rad,
                        )
                    )
                    if patched_idx:
                        trajectory[:, :3] = pos_filt
                        trajectory[:, 3:6] = eul_filt
                        if trajectory.shape[1] >= 7:
                            trajectory[:, 6] = grip_filt
                        gripper_plan = grip_filt
                        logger.warning(
                            "  plan %d OUTLIER-FILTER: patched %d waypoint(s) at indices=%s "
                            "(pos devs=%s)",
                            sequence,
                            len(patched_idx),
                            patched_idx,
                            ["%.3f" % d for d in patched_pos_devs],
                        )

                plan = InferencePlan(
                    sequence=sequence,
                    stage_idx=stage_idx,
                    epoch=epoch,
                    created_at=time.monotonic(),
                    obs_started_at=obs_started_at,
                    trajectory=trajectory,
                    gripper=gripper_plan,
                    ee_pos_at_obs=ee_pos.astype(np.float64).copy(),
                    ee_euler_at_obs=np.asarray(ee_euler_at_obs, dtype=np.float64).copy(),
                )
                latest_plan.publish(plan)
                if deploy_trace is not None:
                    deploy_trace.record_plan(plan)
                logger.info(
                    "  published plan %d horizon=%d forward=%.2fs total=%.2fs first=%s last=%s gripper=%s",
                    plan.sequence,
                    horizon,
                    forward_s,
                    plan.created_at - obs_started_at,
                    np.array2string(plan.trajectory[0, :3], precision=3),
                    np.array2string(plan.trajectory[-1, :3], precision=3),
                    np.array2string((plan.gripper > 0.0).astype(np.float64), precision=0),
                )
                _log_plan_diagnostics(
                    plan, workspace_lo, workspace_hi,
                    policy_loc_bounds, policy_relative,
                )
                sequence += 1
            except Exception as e:
                print(e)
                logger.exception("Inference worker failed")
                stop_event.set()
                break

    thread = threading.Thread(target=worker, name="diffuser-inference", daemon=True)
    thread.start()
    return thread


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    # Hydra chdirs into a per-run output dir; drop a plain text log there so
    # post-mortem analysis of far-jump plans is just a file read away.
    log_path = (Path.cwd() / f"deploy_diffuser_actor_"
                f"{_dt.datetime.now():%Y%m%d_%H%M%S}.log")
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s: %(message)s"
    ))
    logging.getLogger().addHandler(file_handler)
    logger.info("Writing log to %s", log_path)

    _wire_langsteer(cfg.deploy.langsteer_path)

    # Late imports — zero_franky and the policy depend on host-side venvs.
    from zero_franky import Robot, setup_zero_franky
    from franky import Affine, JointMotion, JointState, Twist, PostureTask, ManipulabilityTask
    from clear_franka.utils import LoopRatePrinter
    from clear_franka.diffuser_actor_io import euler_xyz_to_matrix
    from clear_franka.geometry import pack_Rp
    from clear_franka.robotiq_net_proxy import RobotiqGripperProxy
    from clear_franka.recorder import TrajectoryRecorder
    from clear_franka.visualization import CortadoViserVisualizer
    from threed_mouse import ThreeDMouse
    from threed_mouse.geometry import so3_exp
    from threed_mouse.threedmousefilter import ThreeDMouseFilter

    from clear_franka.franka import DEFAULT_LOWER_JOINT_LIMITS, DEFAULT_UPPER_JOINT_LIMITS

    # ----- policy -----
    policy, policy_relative, policy_loc_bounds = _build_policy(cfg.deploy)
    if policy_loc_bounds is not None:
        logger.info(
            "Policy gripper_loc_bounds (used for unnormalize_pos): lo=%s hi=%s",
            np.array2string(policy_loc_bounds[0], precision=3),
            np.array2string(policy_loc_bounds[1], precision=3),
        )
    workspace_lo_np = np.asarray(cfg.deploy.workspace_lo, dtype=np.float64)
    workspace_hi_np = np.asarray(cfg.deploy.workspace_hi, dtype=np.float64)
    logger.info(
        "Safety-clip workspace (cfg.deploy): lo=%s hi=%s",
        np.array2string(workspace_lo_np, precision=3),
        np.array2string(workspace_hi_np, precision=3),
    )

    # ----- optional steering (rotational toward the inverted place, and
    # optionally positional toward the wine rack via hardcoded value maps) -----
    steering, steer_stage_indices = _build_steering(
        cfg.deploy, policy, policy_relative, policy_loc_bounds
    )
    if steering is not None:
        kind = type(steering).__name__
        logger.info(
            f"Steering ENABLED ({kind}) on stages {sorted(steer_stage_indices)}"
        )

    # Stage 0: grasp, Stage 1: place — primitive ids in the trained vocab.
    stages = [
        {"primitive": 0, "object": 0, "label": "grasp glass"},
        {"primitive": 1, "object": 0, "label": "place glass"},
    ]
    stage_idx = 0
    policy.set_primitive(stages[stage_idx]["primitive"])
    policy.set_object(stages[stage_idx]["object"])
    logger.info(f"[stage 0] {stages[stage_idx]['label']}")

    # ----- cameras -----
    # make_zed_camera() returns an already-opened ZedCamera (it calls
    # zed.open() inside __init__). We use synchronous grab_frame() per tick,
    # so we deliberately do NOT call .run() — that would start a background
    # capture thread and grab_frame() warns it must not run concurrently with
    # it (see camera.py:237 docstring).
    cam_hand, cam_tp, pre_hand, pre_tp = _setup_cameras(cfg)

    # ----- gripper (init pattern mirrors teleop.py:184-200) -----
    gc = cfg.gripper
    gripper = None
    if gc.get("enabled", False):
        gripper = RobotiqGripperProxy(
            server_host=gc.host,
            server_port=int(gc.port),
            com_port=gc.com_port,
            device_id=int(gc.device_id),
            connection_type=gc.connection_type,
            tcp_host=gc.tcp_host,
            tcp_port=int(gc.tcp_port),
            auto_activate=True,
        )
        # Default open at start (matches training: episodes begin with gripper open).
        gripper.move_width(gc.open_width_m, wait=False)

    # ----- visualization -----
    vc = cfg.get("visualization", {})
    visualizer = CortadoViserVisualizer(
        host=vc.get("host", "0.0.0.0"),
        port=int(vc.get("port", 8080)),
    )
    visualizer.update_gripper_width(
        cfg.gripper.open_width_m,
        max_width_m=cfg.gripper.max_width_m,
    )

    # ----- robot -----
    setup_zero_franky(cfg.zero_franky.ip, cfg.zero_franky.port,
                      pub_port=cfg.zero_franky.pub_port)
    robot = Robot(cfg.robot.ip)
    robot.recover_from_errors()
    reset_joint_config = np.asarray(cfg.teleop.reset_joint_config, dtype=float)
    logger.info(f"Resetting to start config {reset_joint_config}")
    robot.move(JointMotion(JointState(reset_joint_config),
                            relative_dynamics_factor=0.1),
               asynchronous=False)

    mouse = None
    try:
        mouse = ThreeDMouse(control_rate=cfg.teleop.spacemouse.control_rate)
        # SpaceMouse axis remap — match the standalone teleop tool (teleop.py:56-65),
        # otherwise the teleop branch's jog axes differ from what users expect.
        mouse._frame_rotation_linear = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1],
        ], dtype=float)
        mouse._frame_rotation_angular = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, -1],
        ], dtype=float)
        mouse.run()
        logger.info(
            "Hybrid control — start in TELEOP. Tap LEFT=grasp / RIGHT=place to run "
            "inference; long-press=take over (teleop); chord(L+R)=toggle gripper."
        )
    except Exception as e:
        logger.warning(f"No SpaceMouse ({e}); buttons disabled — Ctrl-C to stop.")

    rate = LoopRatePrinter()
    plan_hz = float(cfg.deploy.get("control_hz", 10.0))
    execution_hz = float(cfg.deploy.get("execution_hz", 100.0))
    plan_dt = 1.0 / plan_hz
    execution_dt = 1.0 / execution_hz
    policy_lock = threading.Lock()
    latest_plan = LatestPlanSlot()
    enabled_event = threading.Event()
    request_event = threading.Event()
    stop_event = threading.Event()
    stage_state: dict = {"idx": stage_idx, "epoch": 0, "gripper_cmd": 1.0}

    # ----- teleop branch control objects (hybrid mode) -----
    sc = cfg.teleop.spacemouse
    input_filter = ThreeDMouseFilter(
        smoothing_factor=sc.smoothing_factor,
        softmax_temp=sc.softmax_temp,
        translation_modifier=cfg.teleop.linear_scale,
        rotation_modifer=cfg.teleop.angular_scale,   # note: library's param typo
        translation_deadband=sc.translation_deadband,
        rotation_deadband=sc.rotation_deadband,
        translation_enabled=True,
        rotation_enabled=True,
    )
    global_frame = bool(cfg.teleop.global_frame)
    long_press_s = float(cfg.teleop.get("long_press_s", 0.8))
    teleop_workspace_clip = bool(cfg.deploy.get("teleop_workspace_clip", True))
    mode = TELEOP   # start in teleop so the user positions the arm first

    plan_hz = float(cfg.deploy.get("control_hz", 10.0))
    execution_hz = float(cfg.deploy.get("execution_hz", 100.0))
    plan_dt = 1.0 / plan_hz
    execution_dt = 1.0 / execution_hz
    policy_lock = threading.Lock()
    latest_plan = LatestPlanSlot()
    enabled_event = threading.Event()
    request_event = threading.Event()
    stop_event = threading.Event()
    stage_state: dict = {"idx": stage_idx, "epoch": 0, "gripper_cmd": 1.0}

    # ----- optional trajectory recorder -----
    # Logs the executed session to data_dir/episode_*.h5 in the exact format the
    # teleop/replay tools write, so a deploy run can be re-run with
    #   uv run python main.py mode=replay replay.episode=<file>
    # Cameras are deliberately NOT recorded: the inference worker grabs ZED
    # frames synchronously (no .run()), which would conflict with the recorder's
    # background camera capture. The joint trajectory is all replay needs.
    recorder = None
    deploy_trace = None
    if bool(cfg.deploy.get("record_trajectory", False)):
        recorder = TrajectoryRecorder(
            save_dir=cfg.data_dir,
            cameras={},
            metadata={
                "control_mode": "diffuser_actor_deploy",
                "policy_config": str(cfg.deploy.policy_config),
                "gripper_enabled": gripper is not None,
            },
        )
        # Companion trace: raw policy plans + per-tick Cartesian tracking error,
        # written to episode_<wall>_deploy.h5 next to the executed trajectory.
        deploy_trace = DeployTrace()

    with contextlib.ExitStack() as stack:
        stack.enter_context(cam_hand)
        stack.enter_context(cam_tp)
        if gripper is not None:
            stack.enter_context(gripper)
        if mouse is not None:
            stack.callback(mouse.close)
        stack.callback(rate.newline)
        # Null-space target — when set together with nullspace_stiffness > 0,
        # the Cartesian impedance controller biases the arm toward this joint
        # config while resolving the 1-DOF redundancy. Lets you keep the EE at
        # a target Cartesian pose while choosing how the elbow/joint-0 sit, which
        # matters for poses near the rack (the arm has to swing joint 0 back to
        # reach the rack y comfortably). Defaults to teleop.reset_joint_config
        # when not set; passing null/None disables the target entirely.
        _ns_target_cfg = cfg.deploy.get("nullspace_target", None)
        if _ns_target_cfg is None:
            _ns_target = np.asarray(cfg.teleop.reset_joint_config, dtype=np.float64)
        elif str(_ns_target_cfg).lower() == "none":
            _ns_target = None
        else:
            _ns_target = np.asarray(_ns_target_cfg, dtype=np.float64)
        logger.info(
            "Impedance session: nullspace_stiffness=%s nullspace_target=%s",
            float(cfg.deploy.nullspace_stiffness),
            "<none>" if _ns_target is None else np.array2string(_ns_target, precision=3),
        )
        # Joint-limit soft-stop. Franky's impedance controller has a built-in
        # repulsive field that pushes joints away from their limits when they
        # come within `activation_distance` (rad). Defaults
        # (stiffness=4/damping=1/max_torque=5/activation=0.1) are too weak when
        # nullspace_stiffness > ~4 and the target is close to a limit — the
        # nullspace pull simply overwhelms the limit repulsion and the arm
        # crosses into Franka's hardware reflex. Strong defaults below:
        #   activation 0.20 rad = 11° of "buffer zone" before limit
        #   stiffness 40 N·m/rad → ~4 N·m at half-buffer; beats nullspace pull
        #   damping 5 → smooths velocity approach
        #   max_torque 30 → enough headroom even under dynamic overshoot
        # `joint_limit_buffer_rad` further narrows the limits we hand to franky
        # so the soft-stop kicks in *before* the physical hardware limit.
        jl_act = float(cfg.deploy.get("joint_limit_activation_distance", 0.20))
        jl_stf = float(cfg.deploy.get("joint_limit_stiffness", 40.0))
        jl_dmp = float(cfg.deploy.get("joint_limit_damping", 5.0))
        jl_tmx = float(cfg.deploy.get("joint_limit_max_torque", 30.0))
        jl_buf = float(cfg.deploy.get("joint_limit_buffer_rad", 0.05))
        _lower_lim = (np.asarray(DEFAULT_LOWER_JOINT_LIMITS, dtype=np.float64) + jl_buf).tolist()
        _upper_lim = (np.asarray(DEFAULT_UPPER_JOINT_LIMITS, dtype=np.float64) - jl_buf).tolist()
        logger.info(
            "Joint-limit soft-stop: activation=%.2frad stiffness=%.1f damping=%.1f "
            "max_torque=%.1f buffer=%.2frad",
            jl_act, jl_stf, jl_dmp, jl_tmx, jl_buf,
        )
        tracker = stack.enter_context(robot.start_cartesian_impedance_session(
            period=0.001,
            translational_stiffness=cfg.deploy.translational_stiffness,
            rotational_stiffness=cfg.deploy.rotational_stiffness,
            nullspace_tasks=[
                PostureTask(_ns_target, stiffness=cfg.deploy.nullspace_stiffness),
                ManipulabilityTask(gain=5.0, max_torque=1.0),
            ],
            lower_joint_limits=_lower_lim,
            upper_joint_limits=_upper_lim,
            joint_limit_activation_distance=jl_act,
            joint_limit_stiffness=jl_stf,
            joint_limit_damping=jl_dmp,
            joint_limit_max_torque=jl_tmx,
        ))
        robot.start_state_stream(timeout_ms=250)
        stack.callback(robot.stop_state_stream)
        if recorder is not None:
            # __exit__ calls close()→stop()→_save_episode, so the h5 is written
            # on any exit path (Ctrl-C, completion, fault).
            stack.enter_context(recorder)
            recorder.start()
            logger.info("Recording trajectory to %s/episode_%s.h5",
                        cfg.data_dir, recorder._start_wall)
            if deploy_trace is not None:
                deploy_trace.start()
                _trace_path = Path(cfg.data_dir) / f"episode_{recorder._start_wall}_deploy.h5"
                stack.callback(
                    lambda: logger.info(
                        "Saved deploy trace (%d plans) to %s",
                        deploy_trace.save(_trace_path), _trace_path,
                    )
                )
        inference_thread = _start_inference_worker(
            policy=policy,
            policy_lock=policy_lock,
            robot=robot,
            cam_hand=cam_hand,
            cam_tp=cam_tp,
            pre_hand=pre_hand,
            pre_tp=pre_tp,
            latest_plan=latest_plan,
            enabled_event=enabled_event,
            request_event=request_event,
            stop_event=stop_event,
            stage_state=stage_state,
            workspace_lo=workspace_lo_np,
            workspace_hi=workspace_hi_np,
            policy_loc_bounds=policy_loc_bounds,
            policy_relative=policy_relative,
            outlier_filter_enabled=bool(
                cfg.deploy.get("outlier_filter", {}).get("enabled", True)
            ),
            outlier_pos_thresh_m=float(
                cfg.deploy.get("outlier_filter", {}).get("pos_thresh_m", 0.05)
            ),
            outlier_eul_thresh_rad=float(
                cfg.deploy.get("outlier_filter", {}).get("eul_thresh_rad", 0.30)
            ),
            steering=steering,
            steer_stage_indices=steer_stage_indices,
            deploy_trace=deploy_trace,
        )
        stack.callback(lambda: (stop_event.set(), enabled_event.set(), inference_thread.join(timeout=1.0)))

        prev_left = 0
        prev_right = 0
        prev_chord = 0
        left_press_time: float | None = None
        right_press_time: float | None = None
        left_used_in_chord = False
        right_used_in_chord = False
        suppress_left_until_release = False
        suppress_right_until_release = False
        sample = None
        active_plan: InferencePlan | None = None
        active_cartesian_trajectory = None
        active_plan_index_offset = 0
        active_index = 0
        active_plan_started_at = 0.0
        consumed_sequence = -1
        waiting_for_plan = False
        # Set when the time-parameterized target reaches the final waypoint; while
        # non-None we HOLD that reference and let the impedance controller converge
        # before requesting the next plan (prevents tracking error accumulating
        # across plan boundaries). Reset to None on adoption.
        settle_started_at: float | None = None
        next_tick = time.monotonic()
        last_viz_update = 0.0
        viz_dt = 1.0 / 5.0
        plan_timeout_s = 3.0
        plan_completion_tolerance_m = 0.015
        plan_hard_skip_m = 0.12
        plan_max_linear_vel_m_s = float(cfg.deploy.get("max_linear_vel_m_s", 0.03))
        plan_max_angular_vel_rad_s = float(cfg.deploy.get("max_angular_vel_rad_s", 0.25))
        plan_timeout_grace_s = 2.0
        # ----- settle phase (let the tracker catch up before the next plan) -----
        # After the target has been streamed to the final waypoint, hold it and
        # wait up to plan_settle_timeout_s for the measured EE to converge within
        # plan_settle_tolerance_m, with a guaranteed minimum dwell of
        # plan_settle_min_s. Then request the next plan.
        plan_settle_tolerance_m = float(
            cfg.deploy.get("plan_settle_tolerance_m", plan_completion_tolerance_m)
        )
        plan_settle_timeout_s = float(cfg.deploy.get("plan_settle_timeout_s", 2.0))
        plan_settle_min_s = float(cfg.deploy.get("plan_settle_min_s", 0.0))

        # ----- mode transitions (swappable gesture->action mapping layer) -----
        def _enter_inference(prim: int, label: str) -> None:
            nonlocal mode, stage_idx, active_plan, active_cartesian_trajectory
            nonlocal active_plan_index_offset, active_plan_started_at, waiting_for_plan
            mode = INFERENCE
            active_plan = None
            active_cartesian_trajectory = None
            active_plan_index_offset = 0
            active_plan_started_at = 0.0
            waiting_for_plan = True
            visualizer.clear_plan_waypoints()
            with policy_lock:
                policy.set_primitive(prim)
                policy.set_object(0)
                policy.reset()
                stage_state["idx"] = prim
                stage_state["epoch"] += 1
            stage_idx = prim          # keep executor adopt guard (plan.stage_idx==stage_idx)
            enabled_event.set()
            request_event.set()
            logger.info(f"[INFERENCE] {label} (primitive {prim}) from current pose")

        def _enter_teleop() -> None:
            nonlocal mode, active_plan, active_cartesian_trajectory
            nonlocal active_plan_index_offset, active_plan_started_at, waiting_for_plan
            if mode == TELEOP:
                return
            mode = TELEOP
            enabled_event.clear()
            request_event.clear()
            stage_state["epoch"] += 1          # orphan any in-flight plan (adopt guard)
            active_plan = None
            active_cartesian_trajectory = None
            active_plan_index_offset = 0
            active_plan_started_at = 0.0
            waiting_for_plan = False
            visualizer.clear_plan_waypoints()
            # No pose seeding needed: the teleop branch reads the measured pose each
            # tick and adds (near-zero) deltas, so the arm does not jump on takeover.
            logger.info("[TELEOP] take over — jog; tap L=grasp R=place, chord=gripper")

        def _toggle_gripper_teleop() -> None:
            if gripper is None:
                return
            is_open = stage_state["gripper_cmd"] >= 0.5
            width = cfg.gripper.close_width_m if is_open else cfg.gripper.open_width_m
            gripper.move_width(
                width,
                speed=int(cfg.gripper.get("speed", 255)),
                force=int(cfg.gripper.get("force", 255)),
                wait=False,
                max_width_m=cfg.gripper.max_width_m,
            )
            # Propagate to the obs gripper channel read by the worker at conditioning.
            stage_state["gripper_cmd"] = 0.0 if is_open else 1.0
            visualizer.update_gripper_width(width, max_width_m=cfg.gripper.max_width_m)
            logger.info(f"[gripper] {'CLOSE' if is_open else 'OPEN'}")

        def _record_tick(enabled: bool, buttons: int = 0) -> None:
            """Log one measured timestep. Mirrors replay.py / teleop.py so the
            resulting episode_*.h5 is byte-format compatible with mode=replay.
            `gripper_open` is the last commanded state (stage_state)."""
            if recorder is None:
                return
            teleop_state = robot.get_last_teleop_state()
            if teleop_state is None:
                return
            measured_pose = np.asarray(teleop_state["O_T_EE"], dtype=float).reshape(4, 4)
            recorder.step(
                ee_pos=measured_pose[:3, 3],
                ee_rot=measured_pose[:3, :3],
                cmd_linear_vel=np.zeros(3),
                cmd_angular_vel=np.zeros(3),
                buttons=buttons,
                enabled=enabled,
                joint_pos=np.asarray(teleop_state["q"], dtype=float),
                joint_vel=np.asarray(teleop_state["dq"], dtype=float),
                gripper_open=(stage_state["gripper_cmd"] if gripper is not None else None),
                robot_abs_time=float(teleop_state["abs_time"]),
            )

        while not stop_event.is_set():
            rate.start_tick()
            # ---------- button polling ----------
            if mouse is not None:
                sample = mouse.get_controller_state()
                if sample is not None:
                    buttons = np.asarray(sample.buttons, dtype=int)
                    left = int(buttons[0]) if len(buttons) > 0 else 0
                    right = int(buttons[1]) if len(buttons) > 1 else 0

                    # Ignore a button until release after it took part in a chord.
                    if suppress_left_until_release:
                        if left:
                            left = 0
                        else:
                            suppress_left_until_release = False
                    if suppress_right_until_release:
                        if right:
                            right = 0
                        else:
                            suppress_right_until_release = False
                    chord = bool(left and right)

                    # Press-start capture (leading edges).
                    if left and not prev_left:
                        left_press_time = time.monotonic()
                        left_used_in_chord = False
                    if right and not prev_right:
                        right_press_time = time.monotonic()
                        right_used_in_chord = False
                    # If the second button joins while the first is held, the first
                    # is part of a chord, not a solo tap/long-press.
                    if right and not prev_right and left:
                        left_used_in_chord = True
                    if left and not prev_left and right:
                        right_used_in_chord = True

                    # Long-press (fires while held) -> take over (TELEOP).
                    if (left and left_press_time is not None and not left_used_in_chord
                            and time.monotonic() - left_press_time >= long_press_s):
                        _enter_teleop()
                        suppress_left_until_release = True
                        left_press_time = None
                    if (right and right_press_time is not None and not right_used_in_chord
                            and time.monotonic() - right_press_time >= long_press_s):
                        _enter_teleop()
                        suppress_right_until_release = True
                        right_press_time = None

                    # Chord rising edge -> toggle gripper (TELEOP only).
                    if chord and not prev_chord:
                        left_used_in_chord = True
                        right_used_in_chord = True
                        if mode == TELEOP:
                            _toggle_gripper_teleop()
                        else:
                            logger.info("  (chord ignored — gripper toggles in TELEOP only)")

                    # Tap on release -> condition + enter INFERENCE.
                    if (not left) and prev_left:
                        if not left_used_in_chord and left_press_time is not None:
                            _enter_inference(0, "grasp")
                        left_press_time = None
                        left_used_in_chord = False
                    if (not right) and prev_right:
                        if not right_used_in_chord and right_press_time is not None:
                            _enter_inference(1, "place")
                        right_press_time = None
                        right_used_in_chord = False

                    prev_left = left
                    prev_right = right
                    prev_chord = chord

            now = time.monotonic()
            if now - last_viz_update >= viz_dt:
                _update_visualizer_robot_state(visualizer, robot.latest_state)
                last_viz_update = now

            # ---------- TELEOP branch: SpaceMouse drives the arm directly ----------
            if mode == TELEOP:
                state = robot.latest_state
                if state is None:
                    state = rate.time_call("state_wait", robot.wait_for_state, 1.0)
                ee_pos, ee_rot, _O_T_EE = _read_ee_pose_from_state(state)
                if mouse is not None and sample is not None:
                    target_pos, target_rot, v_world, w_world = _teleop_target_from_sample(
                        sample, ee_pos, ee_rot, input_filter, global_frame, so3_exp
                    )
                    if teleop_workspace_clip:
                        target_pos = np.clip(target_pos, workspace_lo_np, workspace_hi_np)
                    try:
                        tracker.set_cartesian_reference(
                            Affine(pack_Rp(target_rot, target_pos)),
                            Twist(v_world, w_world),
                        )
                    except Exception as exc:
                        logger.warning(f"[teleop] set_cartesian_reference failed: {exc}")
                _record_tick(enabled=False)
                rate.finish_tick()
                next_tick += execution_dt
                sleep_time = next_tick - time.monotonic()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_tick = time.monotonic()
                continue

            if active_plan is None and not waiting_for_plan:
                request_event.set()
                waiting_for_plan = True

            # ---------- adopt latest completed plan ----------
            current_epoch = stage_state["epoch"]
            plan = latest_plan.latest_after(consumed_sequence)
            if plan is not None and active_plan is None:
                consumed_sequence = plan.sequence
                if plan.stage_idx == stage_idx and plan.epoch == current_epoch:
                    state = robot.latest_state
                    if state is None:
                        state = rate.time_call("state_wait", robot.wait_for_state, 1.0)
                    ee_pos, ee_rot, _O_T_EE = _read_ee_pose_from_state(state)
                    min_dist = float(np.min(
                        np.linalg.norm(plan.trajectory[:, :3] - ee_pos[None, :], axis=1)
                    ))
                    if min_dist > plan_hard_skip_m:
                        visualizer.update_plan_waypoints(plan.trajectory, 0)
                        logger.info(
                            "  skipping plan %d: nearest waypoint %.1f cm from EE; "
                            "ee=%s first=%s last=%s",
                            plan.sequence,
                            min_dist * 100.0,
                            np.array2string(ee_pos, precision=3),
                            np.array2string(plan.trajectory[0, :3], precision=3),
                            np.array2string(plan.trajectory[-1, :3], precision=3),
                        )
                        request_event.set()
                    else:
                        active_plan = plan
                        waiting_for_plan = False
                        settle_started_at = None
                        adopted_at = time.monotonic()
                        active_index = _plan_start_index(
                            plan,
                            ee_pos,
                            plan_dt,
                            adopted_at,
                        )
                        active_plan_index_offset = active_index
                        # Compute current EE euler so the trajectory can prepend
                        # the live pose at t=0 (smooth handoff — bridges from
                        # current EE → plan[start_index] at max_linear_vel
                        # instead of stepping the impedance target by a sudden
                        # 20+ cm and tripping the joint torque reflex).
                        from clear_franka.diffuser_actor_io import ee_rot_to_euler_xyz
                        ee_euler = ee_rot_to_euler_xyz(ee_rot)
                        active_cartesian_trajectory = _make_cartesian_trajectory_for_plan(
                            plan,
                            active_index,
                            plan_dt,
                            euler_xyz_to_matrix,
                            max_linear_vel=plan_max_linear_vel_m_s,
                            max_angular_vel=plan_max_angular_vel_rad_s,
                            min_segment_dt=execution_dt,
                            current_ee_pos=ee_pos,
                            current_ee_euler=ee_euler,
                        )
                        active_plan_started_at = adopted_at
                        visualizer.update_plan_waypoints(plan.trajectory, active_index)
                        visualizer.update_interpolated_plan_path(
                            _sample_cartesian_trajectory_positions(
                                active_cartesian_trajectory,
                                execution_dt,
                            )
                        )
                        logger.info(
                            "  adopted plan %d idx=%d/%d age=%.0fms infer=%.0fms",
                            plan.sequence,
                            active_index,
                            len(plan.trajectory),
                            (time.monotonic() - plan.created_at) * 1000.0,
                            (plan.created_at - plan.obs_started_at) * 1000.0,
                        )
                else:
                    request_event.set()

            # ---------- execute the active plan suffix ----------
            if active_plan is not None and active_index < len(active_plan.trajectory):
                state = robot.latest_state
                if state is None:
                    state = rate.time_call("state_wait", robot.wait_for_state, 1.0)
                ee_pos, _ee_rot, _O_T_EE = _read_ee_pose_from_state(state)
                assert active_cartesian_trajectory is not None

                elapsed = time.monotonic() - active_plan_started_at
                local_index = active_cartesian_trajectory.waypoint_index_at(elapsed)
                active_index = min(
                    active_plan_index_offset + local_index,
                    len(active_plan.trajectory) - 1,
                )
                target_xyz, target_rot = active_cartesian_trajectory.interpolate(elapsed)
                visualizer.update_plan_waypoints(active_plan.trajectory, active_index)

                # Optional safety clip — keep targets inside the recorded workspace.
                lo = np.array(cfg.deploy.workspace_lo, dtype=np.float64)
                hi = np.array(cfg.deploy.workspace_hi, dtype=np.float64)
                target_xyz = np.clip(target_xyz, lo, hi)

                tracker.set_cartesian_reference(Affine(pack_Rp(target_rot, target_xyz)))

                if deploy_trace is not None:
                    deploy_trace.record_error(
                        t=time.monotonic(),
                        plan_sequence=active_plan.sequence,
                        active_index=active_index,
                        target_pos=target_xyz,
                        target_rot=target_rot,
                        measured_pos=ee_pos,
                        measured_rot=_ee_rot,
                    )

                cmd_state = 1.0 if active_plan.gripper[active_index] >= 0.5 else 0.0
                if gripper is not None and cmd_state != stage_state["gripper_cmd"]:
                    width = (cfg.gripper.open_width_m if cmd_state == 1.0
                             else cfg.gripper.close_width_m)
                    logger.info(f"  gripper → {'OPEN' if cmd_state == 1.0 else 'CLOSE'}")
                    gripper.move_width(width, wait=False)
                    visualizer.update_gripper_width(width, max_width_m=cfg.gripper.max_width_m)
                    stage_state["gripper_cmd"] = cmd_state

                final_dist = float(np.linalg.norm(
                    active_plan.trajectory[-1, :3] - ee_pos
                ))
                active_plan_timeout_s = max(
                    plan_timeout_s,
                    active_cartesian_trajectory.duration + plan_timeout_grace_s,
                )

                # The time-parameterized target has reached the final waypoint
                # (interpolate() clamps there for elapsed > duration, so the same
                # reference keeps being commanded each tick — we're holding it).
                streamed_to_end = active_index >= len(active_plan.trajectory) - 1

                if streamed_to_end and settle_started_at is None:
                    settle_started_at = time.monotonic()
                    logger.info(
                        "  plan %d streamed to end; settling (err=%.1fmm, "
                        "tol=%.1fmm, timeout=%.1fs)",
                        active_plan.sequence, final_dist * 1000.0,
                        plan_settle_tolerance_m * 1000.0, plan_settle_timeout_s,
                    )

                settle_elapsed = (
                    time.monotonic() - settle_started_at
                    if settle_started_at is not None else 0.0
                )
                settled = (
                    streamed_to_end
                    and final_dist <= plan_settle_tolerance_m
                    and settle_elapsed >= plan_settle_min_s
                )
                settle_expired = (
                    settle_started_at is not None
                    and settle_elapsed > plan_settle_timeout_s
                )

                if settled or settle_expired:
                    if settled:
                        logger.info(
                            "  completed plan %d (settled %.0fms, err=%.1fmm)",
                            active_plan.sequence, settle_elapsed * 1000.0,
                            final_dist * 1000.0,
                        )
                    else:
                        logger.info(
                            "  plan %d settle timeout (%.1fs, err=%.1fmm); advancing",
                            active_plan.sequence, plan_settle_timeout_s,
                            final_dist * 1000.0,
                        )
                    visualizer.clear_plan_waypoints()
                    active_plan = None
                    active_cartesian_trajectory = None
                    active_plan_index_offset = 0
                    active_plan_started_at = 0.0
                    settle_started_at = None
                    request_event.set()
                    waiting_for_plan = True
                elif (
                    not streamed_to_end
                    and time.monotonic() - active_plan_started_at > active_plan_timeout_s
                ):
                    # Stalled mid-trajectory (target not yet at the final waypoint);
                    # abandon and replan. Once streaming has finished the settle
                    # phase above governs instead, so this won't fire during the hold.
                    logger.info(
                        "  plan %d timed out mid-stream after %.1fs at idx=%d/%d; replacing",
                        active_plan.sequence,
                        active_plan_timeout_s,
                        active_index,
                        len(active_plan.trajectory),
                    )
                    visualizer.clear_plan_waypoints()
                    active_plan = None
                    active_cartesian_trajectory = None
                    active_plan_index_offset = 0
                    active_plan_started_at = 0.0
                    settle_started_at = None
                    request_event.set()
                    waiting_for_plan = True

            elif active_plan is not None:
                visualizer.clear_plan_waypoints()
                active_plan = None
                active_cartesian_trajectory = None
                active_plan_index_offset = 0

            _record_tick(enabled=active_plan is not None)
            rate.finish_tick()
            next_tick += execution_dt
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_tick = time.monotonic()

    return 0


if __name__ == "__main__":
    main()
