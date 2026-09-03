"""Deploy a trained 3D Diffuser Actor checkpoint on the real Franka.

Chunked playback: an inference worker grabs two ZED frames + the current
end-effector pose when the executor asks for a new plan, builds a
`core.types.Observation`, and calls `policy.forward(obs)` to produce an
absolute trajectory. The main loop executes that trajectory to completion
before requesting another one. If execution stalls past a timeout, the executor
abandons the old plan and pauses while the worker generates a replacement. The
gripper command is issued when the executed trajectory step flips relative to
what we last commanded.

With deploy.auto_start_inference=True, inference starts in the grasp stage as
soon as the script comes up (no manual tap needed). The grasp->place transition
is then automatic too: once the policy commands the gripper closed, we poll
Robotiq object_detection() for closure confirmation (or time out) and advance
to place on our own (deploy.gate_place_on_grasp, default True).

SpaceMouse buttons remain available as manual overrides:
    LEFT  short tap   → (re)enter INFERENCE at the grasp stage
    RIGHT short tap   → (re)enter INFERENCE at the place stage
    long-press either → take over in TELEOP
    chord (L+R)       → toggle gripper (TELEOP, or place stage if manual-gripper)

Launch:
    uv run python deploy_diffuser_actor.py \\
        deploy.policy_config=/path/to/policy.yaml \\
        deploy.langsteer_path=$HOME/Documents/michal/LangSteer
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import select
import signal
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


def _ignore_terminal_stop_signals() -> None:
    """Keep SIGTTIN/SIGTTOU from freezing the whole rollout.

    The kill-key listener reads the controlling terminal from a background
    thread for the life of the run. If the process is not in the terminal's
    foreground process group — it got backgrounded, or the shell reclaimed the
    terminal — that read raises SIGTTIN, and its default action stops EVERY
    thread in the process group. A stopped process cannot service SIGINT, so
    Ctrl-C then appears to do nothing at all and the run looks hung (observed:
    State: T (stopped), 120 threads parked in do_signal_stop).

    Ignoring the two signals makes a background terminal read fail with EIO
    instead, which the listener treats as "no keyboard right now".
    """
    for sig in (signal.SIGTTIN, signal.SIGTTOU):
        try:
            signal.signal(sig, signal.SIG_IGN)
        except (ValueError, OSError, AttributeError):
            # Not on the main thread, or the platform lacks it.
            pass


def _stdin_is_foreground() -> bool:
    """True when reading stdin will not hit SIGTTIN/EIO."""
    try:
        return os.tcgetpgrp(sys.stdin.fileno()) == os.getpgrp()
    except (OSError, ValueError, AttributeError):
        # No controlling terminal (piped or redirected) — a read is fine.
        return True


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

    # Optional per-policy home: the joint config the arm moves to before inference.
    # Lives in the policy yaml so it travels with the trained policy. None if absent.
    policy_home = policy_cfg.get("home_config", None)

    return policy, bool(policy_cfg.get("relative", False)), policy_loc_bounds, policy_home


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

    # Resolve place_mode → effective rotation target/direction on the container
    # BEFORE building either steering class. The cabinet (mode A, upright) block
    # overrides the rack (mode B, inverted) defaults that live at the top level.
    # Doing it here — not just inside CombinedBoxSteering — means the rotation
    # steers to the right placement even when position steering is disabled and
    # only the plain TargetRotationSteering is built (it reads the top-level
    # target_euler and knows nothing about place_mode).
    if str(container.get("place_mode", "rack")).lower() == "cabinet":
        cab = container.get("cabinet", {}) or {}
        if cab.get("target_euler") is not None:
            container["target_euler"] = cab["target_euler"]
        container["rot_reverse_direction"] = bool(cab.get("rot_reverse_direction", False))
        if cab.get("guidance_strength") is not None:
            container["guidance_strength"] = cab["guidance_strength"]

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


def _make_joint_trajectory_for_plan(
    cartesian_trajectory,
    *,
    ik,
    q_seed: np.ndarray,
    dt: float,
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    max_distance: float,
    max_waypoints: int = 40,
):
    """Convert an adopted plan's Cartesian path into a joint trajectory via IK.

    This is the `tracker: ik` counterpart to streaming Cartesian references, and
    the per-plan equivalent of what `replay.tracker=ik` does per episode. It
    works because the executor adopts one plan at a time and runs it to
    completion, so the plan's Cartesian waypoints are known up front.

    The workspace clip is applied to each sample BEFORE the solve: in Cartesian
    mode the clip happens on the way to `set_target`, but here the solve has to
    be for the pose that will actually be commanded, or the joint waypoints would
    correspond to poses that were then discarded.

    `q_seed` should be the measured joint configuration. The retimed trajectory
    already starts at the live EE pose (the bridge segment), so the arm is
    effectively pre-positioned at row 0 — the seeding contract
    `solve_trajectory_prefix` expects.

    Sampling is capped at `max_waypoints` because every sample costs an IK solve;
    the joint spline interpolates between them at the streaming rate, exactly as
    replay's joint playback interpolates the solved episode.

    Returns `(trajectory, reason)`. `trajectory` is a
    `clear_franka.joint_trajectory.Trajectory` over the solved prefix, or None
    when fewer than two waypoints solved (nothing interpolable — the caller
    should skip the plan). `reason` is None only on a complete solve.
    """
    from clear_franka.joint_trajectory import Trajectory

    duration = float(cartesian_trajectory.duration)
    if duration <= 0.0:
        return None, "plan trajectory has zero duration"

    num_samples = max(int(np.ceil(duration / float(dt))) + 1, 2)
    num_samples = min(num_samples, int(max_waypoints))
    times = np.linspace(0.0, duration, num_samples)

    samples = [cartesian_trajectory.interpolate(t) for t in times]
    positions = np.clip(
        np.stack([sample[0] for sample in samples], axis=0), workspace_lo, workspace_hi
    )
    rotations = np.stack([sample[1] for sample in samples], axis=0)

    joint_pos, reason = ik.solve_trajectory_prefix(
        positions,
        rotations,
        q_seed=q_seed,
        timestamps=times,
        max_distance=max_distance,
    )
    if len(joint_pos) < 2:
        return None, reason or "IK produced no executable prefix"
    return Trajectory(joint_pos, times[:len(joint_pos)]), reason


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

                # Read the newest frame published by each camera's background
                # capture loop (cam.run() + enable_frame_stream() in main). We do
                # NOT call grab_frame() here: the loop is running, and a second
                # concurrent grab() is unsafe. The loop also records the 30fps
                # video, so one grab feeds both obs and the recording.
                hand_frame = cam_hand.get_latest_frame()
                tp_frame = cam_tp.get_latest_frame()
                if hand_frame is None or tp_frame is None:
                    logger.warning("No camera frame yet — waiting")
                    time.sleep(0.05)
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
    from zero_franky.robotiq import RobotiqGripperProxy
    from scipy.spatial.transform import Rotation
    from clear_franka.recorder import TrajectoryRecorder
    from clear_franka.visualization import CortadoViserVisualizer
    from threed_mouse import ThreeDMouse
    from threed_mouse.geometry import so3_exp
    from threed_mouse.threedmousefilter import ThreeDMouseFilter

    from clear_franka.franka import DEFAULT_LOWER_JOINT_LIMITS, DEFAULT_UPPER_JOINT_LIMITS

    # ----- policy -----
    policy, policy_relative, policy_loc_bounds, policy_home = _build_policy(cfg.deploy)
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
    # Stages where the gripper is MANUAL-ONLY: the policy plan's gripper channel
    # is ignored and only the SpaceMouse chord (both buttons) opens/closes it.
    # Default = the place stage (1), so the glass is held until the user commands
    # the release into the rack slot — the policy never auto-opens at the basin.
    manual_gripper_stages = {
        int(s) for s in cfg.deploy.get("manual_gripper_stages", [1])
    }
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
    # zed.open() inside __init__). Below (inside the ExitStack) we call .run() +
    # enable_frame_stream() so the background loop publishes the newest frame for
    # the worker (get_latest_frame) AND records 30fps video off the same grab —
    # so we no longer use synchronous grab_frame() (which can't run concurrently
    # with the background loop).
    cam_hand, cam_tp, pre_hand, pre_tp = _setup_cameras(cfg)

    # ----- gripper (init pattern mirrors teleop.py:184-200) -----
    gc = cfg.gripper
    gripper = None
    if gc.get("enabled", False):
        gripper = RobotiqGripperProxy(
            server_host=gc.host,
            server_port=int(gc.port),
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
    setup_zero_franky(cfg.zero_franky.ip, cfg.zero_franky.port)
    robot = Robot(cfg.robot.ip)
    robot.recover_from_errors()
    # Home config the arm moves to before inference (and the default nullspace
    # posture). Priority: deploy.home_config (per-run override) > policy yaml
    # home_config (tied to the trained policy) > teleop.reset_joint_config.
    _deploy_home = cfg.deploy.get("home_config", None)
    if _deploy_home is not None and str(_deploy_home).lower() != "none":
        reset_joint_config = np.asarray(_deploy_home, dtype=float)
        logger.info("Home config from deploy.home_config")
    elif policy_home is not None:
        reset_joint_config = np.asarray(policy_home, dtype=float)
        logger.info("Home config from policy yaml: %s", cfg.deploy.policy_config)
    else:
        reset_joint_config = np.asarray(cfg.teleop.reset_joint_config, dtype=float)
    if reset_joint_config.shape != (7,):
        raise ValueError(
            f"home config must have 7 joint angles, got {reset_joint_config.shape[0]}: "
            f"{reset_joint_config.tolist()}"
        )
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
            "Hybrid control — Tap LEFT=grasp / RIGHT=place to run inference; "
            "long-press=take over (teleop); chord(L+R)=toggle gripper."
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
    # Logs the executed session to data_dir/*.mcap in the exact format the
    # teleop/replay tools write, so a deploy run can be re-run with
    #   uv run python main.py mode=replay replay.episode=<file>
    # Cameras ARE recorded now: the worker reads frames from each camera's
    # background loop (cam.run() + enable_frame_stream above), so the recorder
    # can drive start_recording/stop_recording on both cameras for 30fps video
    # off that same background grab — no conflicting synchronous grab.
    recorder = None
    deploy_trace = None
    if bool(cfg.deploy.get("record_trajectory", False)):
        recorder = TrajectoryRecorder(
            save_dir=cfg.data_dir,
            cameras={"hand": cam_hand, "third_person": cam_tp},
            svo_compression=str(cfg.recorder.get("svo_compression", "H264")),
            camera_format=str(cfg.recorder.get("camera_format", "svo")),
            metadata={
                "control_mode": "diffuser_actor_deploy",
                "policy_config": str(cfg.deploy.policy_config),
                "gripper_enabled": gripper is not None,
            },
        )
        # Companion trace: raw policy plans + per-tick Cartesian tracking error,
        # written to episode_<wall>_deploy.h5 next to the executed trajectory.
        deploy_trace = DeployTrace()

    # Execution mode. "cartesian" streams Cartesian references straight to the
    # Cartesian impedance tracker. "ik" solves each adopted plan's Cartesian
    # waypoints into joint waypoints and streams those to the joint impedance
    # tracker — the same conversion replay.tracker=ik does, applied per plan
    # (the executor adopts one plan at a time and runs it to completion, so a
    # plan's Cartesian path is fully known when it is adopted).
    tracker_mode = str(cfg.deploy.get("tracker", "cartesian")).lower()
    if tracker_mode not in {"cartesian", "ik"}:
        raise ValueError(
            f"deploy.tracker must be 'cartesian' or 'ik', got {tracker_mode!r}"
        )
    # plan_ik is built after the state stream comes up (below): the solver has to
    # work in the SAME tool frame the plans and the tracker use, which is the
    # controller's O_T_EE, and that frame is only knowable from a live state.
    plan_ik = None
    ik_max_distance = float(cfg.deploy.get("ik_max_distance", 0.2))
    ik_max_consecutive_failures = int(cfg.deploy.get("ik_max_consecutive_failures", 3))

    with contextlib.ExitStack() as stack:
        stack.enter_context(cam_hand)
        stack.enter_context(cam_tp)
        # Run the background capture loops and have them publish the newest frame
        # for the inference worker (get_latest_frame). This replaces the worker's
        # synchronous grab_frame() and lets the SAME background grab feed both the
        # observation and the 30fps video recording (no concurrent grab()).
        for _cam in (cam_hand, cam_tp):
            _cam.enable_frame_stream()
            _cam.run()
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
            _ns_target = reset_joint_config.astype(np.float64)
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
        # tracker: ik streams JOINT references (plan Cartesian waypoints are solved
        # into joint waypoints per adoption), so it needs the joint impedance
        # tracker. The posture and manipulability tasks exist only to resolve the
        # 7->6 DOF redundancy for a Cartesian reference; under IK the solver
        # resolves it from the measured q, so they do not apply.
        if tracker_mode == "ik":
            ik_stiffness = [float(v) for v in cfg.deploy.get(
                "joint_stiffness", [320.0, 320.0, 320.0, 320.0, 120.0, 120.0, 30.0]
            )]
            logger.info("Tracker: ik (joint impedance, stiffness=%s)", ik_stiffness)
            tracker = stack.enter_context(robot.start_joint_impedance_tracker(
                period=0.001,
                stiffness=ik_stiffness,
                lower_joint_limits=_lower_lim,
                upper_joint_limits=_upper_lim,
                joint_limit_activation_distance=jl_act,
                joint_limit_stiffness=jl_stf,
                joint_limit_damping=jl_dmp,
                joint_limit_max_torque=jl_tmx,
            ))
        else:
            logger.info("Tracker: cartesian (Cartesian impedance)")
            tracker = stack.enter_context(robot.start_cartesian_impedance_tracker(
                period=0.001,
                translational_stiffness=cfg.deploy.translational_stiffness,
                rotational_stiffness=cfg.deploy.rotational_stiffness,
                posture_task=(
                    PostureTask(_ns_target, stiffness=cfg.deploy.nullspace_stiffness)
                    if _ns_target is not None else None
                ),
                manipulability_task=ManipulabilityTask(gain=5.0, max_torque=1.0),
                lower_joint_limits=_lower_lim,
                upper_joint_limits=_upper_lim,
                joint_limit_activation_distance=jl_act,
                joint_limit_stiffness=jl_stf,
                joint_limit_damping=jl_dmp,
                joint_limit_max_torque=jl_tmx,
            ))
        robot.start_state_stream(timeout_ms=250)
        stack.callback(robot.stop_state_stream)

        if tracker_mode == "ik":
            # Every Cartesian pose in deploy — the policy's observation, the plan
            # waypoints, and the Cartesian tracker's targets — is the robot's
            # O_T_EE, i.e. the CONTROLLER's flange-to-TCP frame. CartesianIK
            # defaults to the URDF's offset instead, which is a different frame
            # unless the Desk EE config happens to match it, so solving with the
            # default would put a fixed pose offset on every commanded waypoint.
            # Recover the controller's offset from one live state the same way
            # replay's ik_frame_from_episode recovers it from a recording:
            # flange^-1 * tool.
            from franky import Affine
            from franky.kinematics import forward_kinematics

            from clear_franka.ik import CartesianIK

            _ik_state = robot.wait_for_state(timeout=5.0)
            _ik_q = np.asarray(_ik_state["q"], dtype=np.float64)
            _ik_O_T_EE = np.asarray(_ik_state["O_T_EE"], dtype=np.float64).reshape(4, 4)
            _ik_f_t_ee = forward_kinematics(_ik_q).inverse * Affine(_ik_O_T_EE)
            plan_ik = CartesianIK(
                f_t_ee=_ik_f_t_ee,
                joint_limit_margin=float(cfg.deploy.get("ik_joint_limit_margin", 0.02)),
                position_tolerance=float(cfg.deploy.get("ik_position_tolerance", 1e-3)),
                rotation_tolerance=float(cfg.deploy.get("ik_rotation_tolerance", 1e-2)),
            )
            logger.info(
                "IK execution: f_t_ee from live state (translation=%s), "
                "max_distance=%.3frad tolerances=%.1fmm/%.2fdeg "
                "max_consecutive_failures=%d",
                np.array2string(np.asarray(_ik_f_t_ee.matrix)[:3, 3], precision=4),
                ik_max_distance,
                float(cfg.deploy.get("ik_position_tolerance", 1e-3)) * 1000.0,
                np.degrees(float(cfg.deploy.get("ik_rotation_tolerance", 1e-2))),
                ik_max_consecutive_failures,
            )
            # Sanity: FK of the measured q in this frame must land on the measured
            # pose. If it doesn't, the frame is wrong and every solve would be off.
            _ik_check = np.linalg.norm(plan_ik.forward(_ik_q)[:3, 3] - _ik_O_T_EE[:3, 3])
            if _ik_check > 1e-3:
                raise RuntimeError(
                    f"IK tool frame disagrees with the measured pose by "
                    f"{_ik_check * 1000:.2f} mm — refusing to run deploy.tracker=ik "
                    f"with a frame that would offset every commanded waypoint."
                )
        # Set True once the kill-key save/discard prompt has finalized the
        # recording, so the ExitStack callbacks below don't re-save (or recreate
        # a just-discarded) trace. Normal exits (Ctrl-C / completion) leave this
        # False and save everything as before.
        _recording_finalized = {"done": False}
        _trace_path = None
        if recorder is not None:
            # __exit__ calls close()→stop(), so the episode MCAP is written
            # on any exit path (Ctrl-C, completion, fault).
            stack.enter_context(recorder)
            recorder.start()
            logger.info("Recording trajectory to %s", recorder.episode_path)
            if deploy_trace is not None:
                deploy_trace.start()
                _trace_path = Path(cfg.data_dir) / f"{recorder.episode_base}_deploy.h5"

                def _save_trace_on_exit():
                    if _recording_finalized["done"]:
                        return
                    logger.info(
                        "Saved deploy trace (%d plans) to %s",
                        deploy_trace.save(_trace_path), _trace_path,
                    )
                stack.callback(_save_trace_on_exit)
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

        # ----- keyboard kill key -----
        # A background reader watches stdin; pressing [k] or [q] (or bare Enter)
        # then return sets kill_event. The main loop then halts inference, holds
        # the pose, and prompts to save or discard the recording. Separate from
        # the SpaceMouse gestures (which keep doing stage/teleop/gripper).
        kill_event = threading.Event()

        def _keyboard_listener():
            # Polled rather than `for line in sys.stdin`, which blocks
            # indefinitely: it only noticed stop_event after a line arrived, and
            # it held a blocking read on the terminal the whole time — the read
            # that gets the process stopped by SIGTTIN when backgrounded.
            while not stop_event.is_set():
                if not _stdin_is_foreground():
                    time.sleep(0.25)
                    continue
                try:
                    ready, _, _ = select.select([sys.stdin], [], [], 0.25)
                    if not ready:
                        continue
                    line = sys.stdin.readline()
                    if not line:
                        return                      # EOF: stdin closed
                    if line.strip().lower() in ("k", "q", ""):
                        kill_event.set()
                        return
                except (OSError, ValueError):
                    # EIO from a background read, or stdin closed under us.
                    time.sleep(0.25)
                except Exception:
                    return

        if recorder is not None:
            _ignore_terminal_stop_signals()
            threading.Thread(
                target=_keyboard_listener, name="kill-key", daemon=True
            ).start()
            logger.info("[KILL KEY] press 'k' (or Enter) then return to stop the "
                        "rollout and choose save/discard.")

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
        # tracker: ik only — the joint trajectory solved from
        # active_cartesian_trajectory at adoption, streamed instead of Cartesian
        # references. Consecutive adoptions that yield no executable prefix are
        # counted so a wedged arm cannot churn through plans without moving.
        active_joint_trajectory = None
        ik_consecutive_failures = 0
        active_plan_index_offset = 0
        active_index = 0
        active_plan_started_at = 0.0
        consumed_sequence = -1
        waiting_for_plan = False
        # Catchup hold countdown (cap). -1 = not holding; set to plan_catchup_ticks
        # when a plan finishes (streamed to its final waypoint) OR is abandoned
        # mid-stream (timeout), decremented each tick while we re-command the
        # frozen reference. The next plan is requested once the EE converges to
        # that reference (pos+rot within tolerance) or the counter drops below 0,
        # whichever first. Reset to -1 on adoption. catchup_target_* = frozen ref.
        catchup_remaining = -1
        catchup_target_pos: np.ndarray | None = None
        catchup_target_rot: np.ndarray | None = None
        # tracker: ik holds the last commanded joint configuration; the pos/rot
        # pair above is still what convergence is judged on, since that is what
        # the next observation depends on.
        catchup_joint_target: np.ndarray | None = None
        next_tick = time.monotonic()
        last_viz_update = 0.0
        last_teleop_ik_warn = 0.0
        viz_dt = 1.0 / 5.0
        plan_timeout_s = 3.0
        plan_completion_tolerance_m = 0.015
        plan_hard_skip_m = 0.12
        plan_max_linear_vel_m_s = float(cfg.deploy.get("max_linear_vel_m_s", 0.03))
        plan_max_angular_vel_rad_s = float(cfg.deploy.get("max_angular_vel_rad_s", 0.25))
        plan_timeout_grace_s = 2.0
        # Catchup: after a plan finishes (or stalls), hold the frozen reference
        # and let the impedance controller converge before requesting the next
        # plan, so the next observation is captured from a settled pose. The hold
        # exits as soon as BOTH the position and rotation tracking errors fall
        # within tolerance, or after catchup_ticks execution ticks (the cap),
        # whichever comes first. catchup_ticks=0 disables (single re-command tick).
        plan_catchup_ticks = int(cfg.deploy.get("catchup_ticks", 0))
        catchup_pos_tol_m = float(
            cfg.deploy.get("catchup_pos_tol_m", plan_completion_tolerance_m)
        )
        catchup_rot_tol_rad = float(cfg.deploy.get("catchup_rot_tol_rad", 0.05))

        # ----- grasp->place gate (auto-advance on gripper closure confirmation) -----
        # Once the policy commands the gripper CLOSE during the grasp stage, poll
        # Robotiq object_detection() for closure confirmation and auto-advance to
        # place — or advance anyway after a timeout so a hardware hiccup can't
        # stall the rollout. A manual RIGHT tap still works and supersedes this
        # (cleared in _enter_inference/_enter_teleop) so it can't double-fire.
        gate_place_on_grasp = bool(cfg.deploy.get("gate_place_on_grasp", True))
        grasp_gate_poll_dt = 1.0 / float(cfg.deploy.get("grasp_gate_poll_hz", 15.0))
        grasp_gate_timeout_s = float(cfg.deploy.get("grasp_gate_timeout_s", 3.0))
        grasp_close_pending = False
        grasp_close_commanded_at = 0.0
        last_grasp_gate_poll = 0.0

        # ----- mode transitions (swappable gesture->action mapping layer) -----
        def _enter_inference(prim: int, label: str) -> None:
            nonlocal mode, stage_idx, active_plan, active_cartesian_trajectory
            nonlocal active_plan_index_offset, active_plan_started_at, waiting_for_plan
            nonlocal active_joint_trajectory, ik_consecutive_failures
            nonlocal grasp_close_pending
            grasp_close_pending = False   # any manual/auto transition supersedes a pending gate
            mode = INFERENCE
            active_plan = None
            active_cartesian_trajectory = None
            active_joint_trajectory = None
            ik_consecutive_failures = 0
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
            nonlocal active_joint_trajectory, ik_consecutive_failures
            nonlocal grasp_close_pending
            if mode == TELEOP:
                return
            grasp_close_pending = False
            mode = TELEOP
            enabled_event.clear()
            request_event.clear()
            stage_state["epoch"] += 1          # orphan any in-flight plan (adopt guard)
            active_plan = None
            active_cartesian_trajectory = None
            active_joint_trajectory = None
            ik_consecutive_failures = 0
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
            resulting *.mcap is byte-format compatible with mode=replay.
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

        def finalize_recording(save: bool) -> None:
            """Stop + save the recording, then keep or delete it. Called from the
            kill-key handler. Collects every artifact (episode MCAP, each camera
            video, deploy trace); on discard, unlinks them. Idempotent."""
            if _recording_finalized["done"]:
                return
            _recording_finalized["done"] = True
            enabled_event.clear()
            paths: list[Path] = []
            if recorder is not None:
                recorder.stop()  # writes episode MCAP + closes camera videos
                if recorder.last_saved_path is not None:
                    # artifact_paths() covers the episode MCAP plus one sidecar
                    # per camera actually attached, with the right extension.
                    paths.extend(recorder.artifact_paths())
            if deploy_trace is not None and _trace_path is not None:
                n = deploy_trace.save(_trace_path)
                logger.info("Saved deploy trace (%d plans) to %s", n, _trace_path)
                paths.append(Path(_trace_path))
            if save:
                kept = [p.name for p in paths if p.exists()]
                logger.info("[ROLLOUT] SAVED %d file(s): %s", len(kept), ", ".join(kept))
            else:
                deleted = 0
                for p in paths:
                    try:
                        if p.exists():
                            p.unlink()
                            deleted += 1
                    except OSError as e:
                        logger.warning("Could not delete %s: %s", p, e)
                logger.info("[ROLLOUT] DISCARDED %d file(s)", deleted)

        if bool(cfg.deploy.get("auto_start_inference", False)):
            logger.info(
                "auto_start_inference=True — skipping TELEOP wait, entering "
                "INFERENCE (grasp) now"
            )
            _enter_inference(0, "grasp")

        while not stop_event.is_set():
            rate.start_tick()
            # ---------- kill key: halt, hold pose, prompt save/discard ----------
            if kill_event.is_set():
                logger.info("[KILL] rollout stopped — holding pose.")
                enabled_event.clear()
                try:
                    ans = input(
                        "\n[ROLLOUT] save or discard? [s]ave / [d]iscard: "
                    ).strip().lower()
                except EOFError:
                    ans = "s"
                finalize_recording(save=not ans.startswith("d"))
                stop_event.set()
                break
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

                    # Chord rising edge -> toggle gripper. Allowed in TELEOP, and
                    # in INFERENCE during a manual-gripper stage (place) so the user
                    # commands the glass release at the basin while the policy runs.
                    if chord and not prev_chord:
                        left_used_in_chord = True
                        right_used_in_chord = True
                        if mode == TELEOP or stage_idx in manual_gripper_stages:
                            _toggle_gripper_teleop()
                        else:
                            logger.info("  (chord ignored — gripper toggles in TELEOP "
                                        "or the place stage only)")

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

            # ---------- grasp->place gate ----------
            # Poll for gripper-closure confirmation (Robotiq object_detection) and
            # auto-advance once it arrives, or after a timeout so a hardware hiccup
            # can't stall the rollout. Runs before the active-plan block below so a
            # transition here can safely null out active_plan for this tick (same
            # timing as the RIGHT-tap handler above). RIGHT tap still works as a
            # manual override — it clears grasp_close_pending in _enter_inference
            # so this can't double-fire afterward.
            if grasp_close_pending and gripper is not None:
                now_ = time.monotonic()
                if now_ - last_grasp_gate_poll >= grasp_gate_poll_dt:
                    last_grasp_gate_poll = now_
                    try:
                        obj = gripper.object_detection(refresh_status=True)
                    except Exception as e:
                        logger.warning(f"[grasp-gate] object_detection() failed: {e}")
                        obj = None
                    if obj in (2, 3):
                        logger.info(
                            f"[grasp-gate] gripper closed (object_detection={obj}"
                            f"{' — object detected' if obj == 2 else ' — no contact, missed grasp'});"
                            " auto-advancing to place"
                        )
                        grasp_close_pending = False
                        _enter_inference(1, "place")
                    elif now_ - grasp_close_commanded_at > grasp_gate_timeout_s:
                        logger.warning(
                            f"[grasp-gate] timed out after {grasp_gate_timeout_s}s waiting "
                            f"for closure confirmation (last object_detection={obj}); "
                            "advancing anyway"
                        )
                        grasp_close_pending = False
                        _enter_inference(1, "place")

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
                        if tracker_mode == "ik":
                            # Single-pose solve seeded from the measured q (~0.02 ms).
                            # An unreachable jog is simply not commanded, so the arm
                            # holds its last target instead of lurching to a
                            # nearest-branch configuration.
                            solution = plan_ik.solve(
                                pack_Rp(target_rot, target_pos),
                                np.asarray(state["q"], dtype=np.float64),
                            )
                            if solution.reached:
                                tracker.set_target(solution.joint_pos)
                            elif now - last_teleop_ik_warn >= 1.0:
                                logger.warning(
                                    "[teleop] IK cannot reach the jogged pose "
                                    "(%.1fmm/%.1fdeg off); holding",
                                    solution.position_error * 1000.0,
                                    np.degrees(solution.rotation_error),
                                )
                                last_teleop_ik_warn = now
                        else:
                            tracker.set_target(
                                Affine(pack_Rp(target_rot, target_pos)),
                                Twist(v_world, w_world),
                            )
                    except Exception as exc:
                        logger.warning(f"[teleop] set_target failed: {exc}")
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
                        # tracker: ik — solve the plan's Cartesian path into joint
                        # waypoints now, while it is fully known. A partial solve
                        # is executed as far as it goes and replanned from there;
                        # only a solve with nothing executable drops the plan.
                        if tracker_mode == "ik":
                            active_joint_trajectory, ik_reason = _make_joint_trajectory_for_plan(
                                active_cartesian_trajectory,
                                ik=plan_ik,
                                q_seed=np.asarray(state["q"], dtype=np.float64),
                                dt=execution_dt,
                                workspace_lo=workspace_lo_np,
                                workspace_hi=workspace_hi_np,
                                max_distance=ik_max_distance,
                            )
                            if active_joint_trajectory is None:
                                ik_consecutive_failures += 1
                                logger.warning(
                                    "  plan %d: IK produced no executable prefix "
                                    "(%d/%d in a row) — %s",
                                    plan.sequence,
                                    ik_consecutive_failures,
                                    ik_max_consecutive_failures,
                                    ik_reason,
                                )
                                # Same response as a hard skip: hold nothing, keep
                                # the arm where it is, ask for another plan. Leaving
                                # active_plan None makes the catchup and execute
                                # branches below no-ops for this tick.
                                active_plan = None
                                active_cartesian_trajectory = None
                                active_plan_index_offset = 0
                                waiting_for_plan = True
                                visualizer.clear_plan_waypoints()
                                if ik_consecutive_failures >= ik_max_consecutive_failures:
                                    logger.error(
                                        "  IK failed on %d consecutive plans — the arm "
                                        "cannot follow from this configuration. Holding "
                                        "pose and handing off to the operator.",
                                        ik_consecutive_failures,
                                    )
                                    kill_event.set()
                                else:
                                    request_event.set()
                            else:
                                ik_consecutive_failures = 0
                                if ik_reason is not None:
                                    logger.info(
                                        "  plan %d: executing IK prefix (%.2fs of %.2fs) — %s",
                                        plan.sequence,
                                        float(active_joint_trajectory.waypts_time[-1]),
                                        float(active_cartesian_trajectory.duration),
                                        ik_reason,
                                    )

                    if active_plan is not None:
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

            # ---------- catchup hold: let the controller converge before replan ----------
            # Both terminal conditions (streamed-to-end and mid-stream timeout)
            # funnel here: we freeze the reference at the pose we were last
            # tracking and re-command it until the EE catches up (position AND
            # rotation error within tolerance) or the plan_catchup_ticks cap is
            # hit, then request the next plan — so the next observation is captured
            # from a settled pose and tracking error doesn't accumulate across plan
            # boundaries. With plan_catchup_ticks=0 this is a single tick (≈ the
            # original immediate-replan behavior).
            if active_plan is not None and catchup_remaining >= 0:
                state = robot.latest_state
                if state is None:
                    state = rate.time_call("state_wait", robot.wait_for_state, 1.0)
                ee_pos, _ee_rot, _O_T_EE = _read_ee_pose_from_state(state)
                if tracker_mode == "ik":
                    # Freeze the joint configuration; convergence is still judged
                    # on the EE pose below, since that is what the next
                    # observation depends on.
                    if catchup_joint_target is not None:
                        tracker.set_target(catchup_joint_target)
                else:
                    tracker.set_cartesian_reference(
                        Affine(pack_Rp(catchup_target_rot, catchup_target_pos))
                    )
                if deploy_trace is not None:
                    deploy_trace.record_error(
                        t=time.monotonic(),
                        plan_sequence=active_plan.sequence,
                        active_index=active_index,
                        target_pos=catchup_target_pos,
                        target_rot=catchup_target_rot,
                        measured_pos=ee_pos,
                        measured_rot=_ee_rot,
                    )
                # Caught up when BOTH position and rotation errors are within
                # tolerance; otherwise keep holding until the catchup_ticks cap.
                pos_err = float(np.linalg.norm(catchup_target_pos - ee_pos))
                R_err = catchup_target_rot @ _ee_rot.T
                rot_err = float(np.arccos(
                    np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
                ))
                converged = pos_err <= catchup_pos_tol_m and rot_err <= catchup_rot_tol_rad
                catchup_remaining -= 1
                if converged or catchup_remaining < 0:
                    logger.info(
                        "  plan %d catchup %s (pos=%.1fmm rot=%.1f°); requesting next plan",
                        active_plan.sequence,
                        "converged" if converged else "cap reached",
                        pos_err * 1000.0, np.degrees(rot_err),
                    )
                    visualizer.clear_plan_waypoints()
                    active_plan = None
                    active_cartesian_trajectory = None
                    active_joint_trajectory = None
                    active_plan_index_offset = 0
                    active_plan_started_at = 0.0
                    request_event.set()
                    waiting_for_plan = True

            # ---------- execute the active plan suffix ----------
            elif active_plan is not None and active_index < len(active_plan.trajectory):
                state = robot.latest_state
                if state is None:
                    state = rate.time_call("state_wait", robot.wait_for_state, 1.0)
                ee_pos, _ee_rot, _O_T_EE = _read_ee_pose_from_state(state)
                assert active_cartesian_trajectory is not None

                elapsed = time.monotonic() - active_plan_started_at
                # tracker: ik may hold only a PREFIX of the plan (a partial solve),
                # so the plan is done when the joint trajectory runs out, not when
                # the Cartesian one would. Clamping elapsed keeps the commanded
                # target, the trace and the catchup hold all consistent with the
                # last pose actually reachable.
                ik_prefix_done = False
                if tracker_mode == "ik":
                    assert active_joint_trajectory is not None
                    ik_joint_end = float(active_joint_trajectory.waypts_time[-1])
                    ik_prefix_done = elapsed >= ik_joint_end
                    elapsed = min(elapsed, ik_joint_end)
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

                if tracker_mode == "ik":
                    # The joint waypoints were solved from the clipped Cartesian
                    # samples, so this commands the same pose target_xyz reports.
                    tracker.set_target(
                        active_joint_trajectory.interpolate(elapsed).reshape(7)
                    )
                else:
                    tracker.set_target(Affine(pack_Rp(target_rot, target_xyz)))

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

                # Plan-driven gripper — skipped in manual-only stages (place), where
                # the gripper holds its current state and only the SpaceMouse chord
                # opens it, so the user controls the release into the rack slot.
                if active_plan.stage_idx not in manual_gripper_stages:
                    cmd_state = 1.0 if active_plan.gripper[active_index] >= 0.5 else 0.0
                    if gripper is not None and cmd_state != stage_state["gripper_cmd"]:
                        width = (cfg.gripper.open_width_m if cmd_state == 1.0
                                 else cfg.gripper.close_width_m)
                        logger.info(f"  gripper → {'OPEN' if cmd_state == 1.0 else 'CLOSE'}")
                        gripper.move_width(width, wait=False)
                        visualizer.update_gripper_width(width, max_width_m=cfg.gripper.max_width_m)
                        stage_state["gripper_cmd"] = cmd_state
                        if (gate_place_on_grasp and cmd_state == 0.0
                                and active_plan.stage_idx == 0):
                            grasp_close_pending = True
                            grasp_close_commanded_at = time.monotonic()
                            last_grasp_gate_poll = 0.0

                final_dist = float(np.linalg.norm(
                    active_plan.trajectory[-1, :3] - ee_pos
                ))
                active_plan_timeout_s = max(
                    plan_timeout_s,
                    active_cartesian_trajectory.duration + plan_timeout_grace_s,
                )
                streamed_to_end = active_index >= len(active_plan.trajectory) - 1
                timed_out = (
                    time.monotonic() - active_plan_started_at > active_plan_timeout_s
                )

                # Either terminal condition freezes the current reference and
                # hands off to the catchup hold above, which counts down and then
                # requests the replacement plan. streamed_to_end is reached at the
                # trajectory's scheduled duration (well before active_plan_timeout_s),
                # so in normal operation we always finish via the streamed-to-end
                # path; the timeout is the safety net for a stalled loop.
                if streamed_to_end or timed_out or ik_prefix_done:
                    catchup_target_pos = np.asarray(target_xyz, dtype=np.float64).copy()
                    catchup_target_rot = np.asarray(target_rot, dtype=np.float64).copy()
                    if tracker_mode == "ik":
                        catchup_joint_target = np.asarray(
                            active_joint_trajectory.interpolate(elapsed), dtype=np.float64
                        ).reshape(7)
                    catchup_remaining = plan_catchup_ticks
                    if ik_prefix_done and not streamed_to_end:
                        logger.info(
                            "  plan %d: IK prefix exhausted at idx=%d/%d; "
                            "catchup hold %d ticks then replan",
                            active_plan.sequence, active_index,
                            len(active_plan.trajectory), plan_catchup_ticks,
                        )
                    if streamed_to_end:
                        logger.info(
                            "  plan %d streamed to end at idx=%d/%d (err=%.1fmm); "
                            "catchup hold %d ticks",
                            active_plan.sequence, active_index,
                            len(active_plan.trajectory), final_dist * 1000.0,
                            plan_catchup_ticks,
                        )
                    else:
                        logger.info(
                            "  plan %d timed out after %.1fs at idx=%d/%d; "
                            "catchup hold %d ticks then replace",
                            active_plan.sequence, active_plan_timeout_s,
                            active_index, len(active_plan.trajectory),
                            plan_catchup_ticks,
                        )

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
