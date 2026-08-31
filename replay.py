"""Replay a recorded trajectory using joint impedance control."""

import time
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from clear_franka.episode_io import find_latest_episode, load_episode

from zero_franky import Robot
from franky import Affine, JointMotion, JointState, ManipulabilityTask, PostureTask, Twist
from franky.kinematics import (
    IKOptions,
    RedundancyParameter,
    forward_kinematics,
    inverse_kinematics,
)

from clear_franka.franka import (
    DEFAULT_LOWER_JOINT_LIMITS,
    DEFAULT_UPPER_JOINT_LIMITS,
    fk_f_t_ee,
    joint_friction_kwargs,
)
from clear_franka.cartesian_trajectory import CartesianTrajectory
from clear_franka.geometry import pack_Rp
from clear_franka.joint_trajectory import Trajectory
from clear_franka.preprocess import preprocess_episode_arrays


def prompt_reverse_reset() -> bool:
    try:
        answer = input("  Play trajectory in reverse to reset robot? [Y/n] ").strip().lower()
    except EOFError:
        print("  Reverse reset skipped (no input available).")
        return False
    return answer in ("", "y", "yes")


def _preprocess_kwargs(pre_cfg: DictConfig) -> dict:
    params = OmegaConf.to_container(pre_cfg, resolve=True)
    trim_cfg = params.get("trim", {})
    retime_cfg = params.get("retime", {})
    smooth_cfg = params.get("smooth", {})
    return {
        "trim_enabled": bool(trim_cfg.get("enabled", True)),
        "trim_time_window": float(trim_cfg.get("time_window", 0.3)),
        "trim_threshold": float(trim_cfg.get("threshold", 0.01)),
        "retime_enabled": bool(retime_cfg.get("enabled", False)),
        "retime_sample_uniform": bool(retime_cfg.get("sample_uniform", False)),
        "retime_path_tol": retime_cfg.get("path_tol", None),
        "retime_max_joint_vel": retime_cfg.get("max_joint_vel", None),
        "retime_max_joint_accel": retime_cfg.get("max_joint_accel", None),
        "smooth_enabled": bool(smooth_cfg.get("enabled", True)),
        "smooth_max_joint_vel": smooth_cfg["max_joint_vel"],
        "smooth_max_joint_accel": smooth_cfg["max_joint_accel"],
        "smooth_max_joint_jerk": smooth_cfg["max_joint_jerk"],
        "smooth_dt": float(smooth_cfg.get("dt", 0.001)),
        "gripper_dwell_s": float(params.get("gripper_dwell_s", 0.0)),
    }


def play_joint_trajectory(
    *,
    robot: Robot,
    rc: DictConfig,
    gc,
    stiffness: np.ndarray,
    timestamps: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    has_joint_vel: bool,
    gripper,
    gripper_open_data,
    recorder=None,
    gripper_open_for_record=None,
):
    n_steps = len(timestamps)
    trajectory = Trajectory(joint_pos, timestamps)

    with robot.start_joint_impedance_tracker(
        period=rc.period,
        stiffness=stiffness,
        lower_joint_limits=DEFAULT_LOWER_JOINT_LIMITS,
        upper_joint_limits=DEFAULT_UPPER_JOINT_LIMITS,
        **joint_friction_kwargs(rc, speed=float(rc.speed)),
    ) as session:
        step = 0
        replay_start = None
        last_gripper_open = None

        while True:
            if session.tick() is None:
                raise RuntimeError("Impedance tracker stopped before replay completed.")
            if replay_start is None:
                replay_start = time.monotonic()

            elapsed = (time.monotonic() - replay_start) * rc.speed

            while step < n_steps - 1 and timestamps[step + 1] <= elapsed:
                step += 1

            if elapsed >= timestamps[-1]:
                print("  Replay complete.")
                session.set_target(joint_pos[-1])
                break

            q = trajectory.interpolate(elapsed).reshape(7)
            if has_joint_vel:
                dq = np.asarray(trajectory._spline.derivative()(elapsed), dtype=float).reshape(7)
                dq = dq * rc.speed
                session.set_target(q, dq=dq)
            else:
                session.set_target(q)

            if gripper is not None and np.isfinite(gripper_open_data[step]):
                current_gripper_open = bool(round(float(gripper_open_data[step])))
                if current_gripper_open != last_gripper_open:
                    target_width = (
                        gc.get("open_width_m", 0.085)
                        if current_gripper_open
                        else gc.get("close_width_m", 0.0)
                    )
                    try:
                        gripper.move_width(
                            target_width,
                            speed=int(gc.get("speed", 255)),
                            force=int(gc.get("force", 255)),
                            wait=False,
                            max_width_m=gc.get("max_width_m", 0.085),
                        )
                        last_gripper_open = current_gripper_open
                        gripper_open_for_record = current_gripper_open
                    except Exception as e:
                        print(f"\n  [gripper] move failed: {e}")

            if recorder is not None:
                teleop_state = session.state
                if teleop_state is None:
                    teleop_state = robot.wait_for_state(timeout=5.0)
                measured_pose = np.asarray(teleop_state["O_T_EE"], dtype=float).reshape(4, 4)
                recorder.step(
                    ee_pos=measured_pose[:3, 3],
                    ee_rot=measured_pose[:3, :3],
                    cmd_linear_vel=np.zeros(3),
                    cmd_angular_vel=np.zeros(3),
                    buttons=0,
                    enabled=True,
                    joint_pos=np.asarray(teleop_state["q"], dtype=float),
                    joint_vel=np.asarray(teleop_state["dq"], dtype=float),
                    gripper_open=gripper_open_for_record,
                    robot_abs_time=float(teleop_state["abs_time"]),
                )


    return gripper_open_for_record


def play_cartesian_trajectory(
    *,
    robot: Robot,
    cfg: DictConfig,
    rc: DictConfig,
    gc,
    timestamps: np.ndarray,
    ee_pos: np.ndarray,
    ee_rot: np.ndarray,
    joint_pos: np.ndarray,
    gripper,
    gripper_open_data,
    recorder=None,
    gripper_open_for_record=None,
):
    n_steps = len(timestamps)
    trajectory = CartesianTrajectory(ee_pos, ee_rot, timestamps)
    nullspace_target_cfg = rc.get("nullspace_target", None)
    if nullspace_target_cfg is None:
        nullspace_target = np.asarray(joint_pos[0], dtype=float)
    elif str(nullspace_target_cfg).lower() == "none":
        nullspace_target = None
    else:
        nullspace_target = np.asarray(nullspace_target_cfg, dtype=float)

    with robot.start_cartesian_impedance_session(
        period=rc.period,
        translational_stiffness=float(
            rc.get("translational_stiffness", cfg.teleop.translational_stiffness)
        ),
        rotational_stiffness=float(
            rc.get("rotational_stiffness", cfg.teleop.rotational_stiffness)
        ),
        nullspace_tasks=[
            PostureTask(
                nullspace_target,
                stiffness=float(
                    rc.get("nullspace_stiffness", cfg.teleop.nullspace_stiffness)
                ),
            ),
            ManipulabilityTask(gain=5.0, max_torque=1.0),
        ],
        lower_joint_limits=DEFAULT_LOWER_JOINT_LIMITS,
        upper_joint_limits=DEFAULT_UPPER_JOINT_LIMITS,
    ) as session:
        step = 0
        replay_start = None
        last_gripper_open = None

        while True:
            if replay_start is None:
                replay_start = time.monotonic()

            elapsed = (time.monotonic() - replay_start) * rc.speed

            while step < n_steps - 1 and timestamps[step + 1] <= elapsed:
                step += 1

            if elapsed >= timestamps[-1]:
                print("  Replay complete.")
                session.set_cartesian_reference(Affine(pack_Rp(ee_rot[-1], ee_pos[-1])))
                break

            target_pos, target_rot = trajectory.interpolate(elapsed)
            linear_vel, angular_vel = trajectory.velocity(elapsed)
            session.set_cartesian_reference(
                Affine(pack_Rp(target_rot, target_pos)),
                Twist(linear_vel * rc.speed, angular_vel * rc.speed),
            )

            if gripper is not None and np.isfinite(gripper_open_data[step]):
                current_gripper_open = bool(round(float(gripper_open_data[step])))
                if current_gripper_open != last_gripper_open:
                    target_width = (
                        gc.get("open_width_m", 0.085)
                        if current_gripper_open
                        else gc.get("close_width_m", 0.0)
                    )
                    try:
                        gripper.move_width(
                            target_width,
                            speed=int(gc.get("speed", 255)),
                            force=int(gc.get("force", 255)),
                            wait=False,
                            max_width_m=gc.get("max_width_m", 0.085),
                        )
                        last_gripper_open = current_gripper_open
                        gripper_open_for_record = current_gripper_open
                    except Exception as e:
                        print(f"\n  [gripper] move failed: {e}")

            if recorder is not None:
                teleop_state = robot.get_last_teleop_state()
                measured_pose = np.asarray(teleop_state["O_T_EE"], dtype=float).reshape(4, 4)
                recorder.step(
                    ee_pos=measured_pose[:3, 3],
                    ee_rot=measured_pose[:3, :3],
                    cmd_linear_vel=np.zeros(3),
                    cmd_angular_vel=np.zeros(3),
                    buttons=0,
                    enabled=True,
                    joint_pos=np.asarray(teleop_state["q"], dtype=float),
                    joint_vel=np.asarray(teleop_state["dq"], dtype=float),
                    gripper_open=gripper_open_for_record,
                    robot_abs_time=float(teleop_state["abs_time"]),
                )

            time.sleep(rc.period)

    return gripper_open_for_record


def episode_ik_frame(episode: dict) -> tuple[np.ndarray | None, Affine | None]:
    """The seed configuration and tool frame an episode implies, if it has joints.

    IK needs neither — `solve_ik_trajectory` falls back for both — but an episode
    that does carry joint samples pins them down exactly, so it is worth reading
    the first sample for. The flange pose `forward_kinematics` derives from that
    configuration and the tool pose recorded at the same instant differ by
    precisely the flange-to-TCP offset the poses are expressed in, whichever that
    is: the controller's F_T_EE for a raw recording, or the URDF's for one whose
    Cartesian fields preprocessing recomputed by forward kinematics.

    Returns (None, None) when the episode has no usable joint samples.
    """
    joint_pos = episode.get("joint_pos")
    ee_pos, ee_rot = episode.get("ee_pos"), episode.get("ee_rot")
    if joint_pos is None or ee_pos is None or ee_rot is None:
        return None, None
    if not (
        np.all(np.isfinite(joint_pos[0]))
        and np.all(np.isfinite(ee_pos[0]))
        and np.all(np.isfinite(ee_rot[0]))
    ):
        return None, None
    q_seed = np.asarray(joint_pos[0], dtype=float)
    return q_seed, forward_kinematics(q_seed).inverse * Affine(pack_Rp(ee_rot[0], ee_pos[0]))


def _roomiest_configuration(
    pose: Affine,
    *,
    f_t_ee: Affine,
    options: IKOptions,
    lower: np.ndarray,
    upper: np.ndarray,
    samples: int = 360,
) -> np.ndarray | None:
    """A start configuration for `pose`, chosen for room rather than closeness.
    """
    best, best_clearance = None, -np.inf
    for swivel in np.linspace(-np.pi, np.pi, samples, endpoint=False):
        for q in inverse_kinematics(
            pose,
            float(swivel),
            parameter=RedundancyParameter.Swivel,
            f_t_ee=f_t_ee,
            options=options,
        ):
            q = np.asarray(q, dtype=float)
            clearance = float(np.min(np.minimum(q - lower, upper - q)))
            if clearance > best_clearance:
                best, best_clearance = q, clearance
    return best


def _pose_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The twist (3 translation, 3 rotation) taking pose `current` onto `target`."""
    error = np.empty(6, dtype=float)
    error[:3] = target[:3, 3] - current[:3, 3]
    delta = target[:3, :3] @ current[:3, :3].T
    axis = np.array([
        delta[2, 1] - delta[1, 2],
        delta[0, 2] - delta[2, 0],
        delta[1, 0] - delta[0, 1],
    ]) / 2.0
    sin = np.linalg.norm(axis)
    error[3:] = axis if sin < 1e-9 else axis * (np.arctan2(sin, (np.trace(delta) - 1.0) / 2.0) / sin)
    return error


def _jacobian(q: np.ndarray, f_t_ee: Affine, step: float = 1e-7) -> np.ndarray:
    """The 6x7 tool Jacobian at `q`, by finite differences on forward kinematics.
    """
    base = forward_kinematics(q, f_t_ee=f_t_ee).matrix
    jacobian = np.empty((6, 7), dtype=float)
    for j in range(7):
        shifted = q.copy()
        shifted[j] += step
        jacobian[:, j] = _pose_error(base, forward_kinematics(shifted, f_t_ee=f_t_ee).matrix) / step
    return jacobian


def _limit_repulsion(
    q: np.ndarray, lower: np.ndarray, upper: np.ndarray, activation: float, gain: float
) -> np.ndarray:
    """A nudge away from any joint limit `q` has come within `activation` of.
    """
    push = np.zeros(7, dtype=float)
    if activation <= 0.0 or gain <= 0.0:
        return push
    to_lower, to_upper = q - lower, upper - q
    near = to_lower < activation
    push[near] += gain * (activation - to_lower[near]) / activation
    near = to_upper < activation
    push[near] -= gain * (activation - to_upper[near]) / activation
    return push


def _converge_to_pose(
    q_start: np.ndarray,
    target: np.ndarray,
    *,
    f_t_ee: Affine,
    lower: np.ndarray,
    upper: np.ndarray,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Levenberg-Marquardt onto `target`, from `q_start`, staying within limits.

    Returns the configuration and its residual twist. It iterates until the pose
    is hit to machine precision or no amount of damping buys further progress, and
    leaves judging the residual to the caller: near a limit or a singularity the
    last fraction of a millimetre may simply be unreachable, and that is worth
    accepting rather than failing over.
    """
    q = q_start.copy()
    error = _pose_error(forward_kinematics(q, f_t_ee=f_t_ee).matrix, target)
    damping = 1e-4
    for _ in range(max_iterations):
        if np.linalg.norm(error) < 1e-9:
            break
        jacobian = _jacobian(q, f_t_ee)
        jjt = jacobian @ jacobian.T
        for _attempt in range(10):
            step = jacobian.T @ np.linalg.solve(jjt + damping**2 * np.eye(6), error)
            candidate = np.clip(q + step, lower, upper)
            residual = _pose_error(forward_kinematics(candidate, f_t_ee=f_t_ee).matrix, target)
            if np.linalg.norm(residual) < np.linalg.norm(error):
                q, error = candidate, residual
                damping = max(damping * 0.5, 1e-6)
                break
            damping *= 4.0
        else:
            break  # no damping value made progress; this is as close as it gets
    return q, error


def solve_ik_trajectory(
    *,
    timestamps: np.ndarray,
    ee_pos: np.ndarray,
    ee_rot: np.ndarray,
    q_seed: np.ndarray | None = None,
    f_t_ee: np.ndarray | Affine | None = None,
    joint_limit_margin: float = 0.02,
    max_distance: float | None = 0.2,
    position_tolerance: float = 1e-3,
    rotation_tolerance: float = 1e-2,
    limit_activation: float = 0.15,
    limit_gain: float = 0.02,
    max_iterations: int = 40,
) -> np.ndarray:
    """Convert a Cartesian trajectory into a joint trajectory, waypoint by waypoint.

    Needs only the tool poses: `q_seed` (the posture to start from) and `f_t_ee`
    (the flange-to-TCP offset the poses are expressed in) both fall back to
    sensible values when omitted, so a trajectory that was never recorded on
    joints — a policy rollout, a synthetic path — converts the same as one that
    was. `episode_ik_frame` reads both off an episode that does carry joints.

    Each waypoint is solved numerically (damped least squares, Levenberg-Marquardt
    damping) starting from the previous solution.
    """
    n_steps = len(ee_pos)
    margin_lower = np.asarray(DEFAULT_LOWER_JOINT_LIMITS, dtype=float) + joint_limit_margin
    margin_upper = np.asarray(DEFAULT_UPPER_JOINT_LIMITS, dtype=float) - joint_limit_margin

    if f_t_ee is None:
        # No robot to ask: the URDF's flange-to-TCP offset, which is what
        # `clear_franka.franka.fk_ee_poses` computes recorded tool poses with.
        f_t_ee = Affine(fk_f_t_ee())
    elif not isinstance(f_t_ee, Affine):
        f_t_ee = Affine(np.asarray(f_t_ee, dtype=float).reshape(4, 4))

    poses = pack_Rp(ee_rot, ee_pos).reshape(n_steps, 4, 4)

    if q_seed is None:
        q_prev = _roomiest_configuration(
            Affine(poses[0]),
            f_t_ee=f_t_ee,
            options=IKOptions(joint_limits=(margin_lower, margin_upper)),
            lower=margin_lower,
            upper=margin_upper,
        )
        if q_prev is None:
            raise RuntimeError(
                f"IK failed at the first waypoint (t={timestamps[0]:.3f}s): the pose is "
                f"out of reach in every configuration within the joint limits."
            )
    else:
        q_prev = np.clip(np.asarray(q_seed, dtype=float), margin_lower, margin_upper)

    joint_pos_ik = np.empty((n_steps, 7), dtype=float)
    worst_position = worst_rotation = 0.0

    for i in range(n_steps):
        # Bias the start away from any limit the arm has drifted up against.
        start = np.clip(
            q_prev + _limit_repulsion(q_prev, margin_lower, margin_upper, limit_activation, limit_gain),
            margin_lower,
            margin_upper,
        )
        q_i, residual = _converge_to_pose(
            start,
            poses[i],
            f_t_ee=f_t_ee,
            lower=margin_lower,
            upper=margin_upper,
            max_iterations=max_iterations,
        )
        position_error = float(np.linalg.norm(residual[:3]))
        rotation_error = float(np.linalg.norm(residual[3:]))
        moved = float(np.max(np.abs(q_i - q_prev)))

        if position_error > position_tolerance or rotation_error > rotation_tolerance:
            raise RuntimeError(
                f"IK failed at step {i}/{n_steps} (t={timestamps[i]:.3f}s): closest reachable "
                f"configuration still misses the pose by {position_error * 1000:.2f} mm / "
                f"{np.degrees(rotation_error):.2f}deg, outside the "
                f"{position_tolerance * 1000:.2f} mm / {np.degrees(rotation_tolerance):.2f}deg "
                f"tolerance. The arm cannot follow the path further from here — check for a "
                f"joint against its limit (joint_limit_margin={joint_limit_margin} rad)."
            )
        # Not at the first waypoint: `max_distance` bounds motion between waypoints
        # the arm will execute, and nothing has been executed yet. The seed only
        # chooses a starting posture — the caller pre-positions to `[0]` — so it is
        # free to sit far from the trajectory, or to come from a config file.
        if i > 0 and max_distance is not None and moved > max_distance:
            raise RuntimeError(
                f"IK failed at step {i}/{n_steps} (t={timestamps[i]:.3f}s): reaching it means "
                f"moving j{int(np.argmax(np.abs(q_i - q_prev))) + 1} by {moved:.3f} rad in one "
                f"waypoint, more than max_distance={max_distance} rad. The trajectory either "
                f"crosses a singularity here or is sampled too coarsely to follow smoothly."
            )

        joint_pos_ik[i] = q_i
        q_prev = q_i
        worst_position = max(worst_position, position_error)
        worst_rotation = max(worst_rotation, rotation_error)

    if worst_position > 1e-6 or worst_rotation > 1e-6:
        print(f"  IK worst-case pose residual: {worst_position * 1000:.3f} mm / "
              f"{np.degrees(worst_rotation):.3f}deg.")
    return joint_pos_ik


def play_with_recovery(*, play_fn=play_joint_trajectory, **kwargs):
    robot = kwargs["robot"]
    while True:
        robot.recover_from_errors()
        try:
            return play_fn(**kwargs)
        except RuntimeError as e:
            print(f"\n  Controller faulted: {e}")
            print("  Recovering and retrying...")


def run_replay(cfg: DictConfig):
    rc = cfg.replay
    gc = cfg.get("gripper", {})

    if rc.episode is not None:
        episode_path = Path(rc.episode)
    else:
        episode_path = find_latest_episode(cfg.data_dir)

    print(f"Loading episode: {episode_path}")
    episode = load_episode(episode_path)
    if bool(rc.get("preprocess", True)):
        pre_cfg = cfg.get("preprocess", None)
        if pre_cfg is None:
            raise RuntimeError("replay.preprocess=true requires the top-level preprocess config")
        print("  Preprocessing episode before replay...")
        processed = preprocess_episode_arrays(episode, **_preprocess_kwargs(pre_cfg))
        if processed is None:
            raise RuntimeError("Replay preprocessing failed; see preprocess logs above")
        episode.update(processed)

    timestamps = episode["timestamps"]
    joint_pos = episode["joint_pos"]
    joint_vel = episode["joint_vel"]
    tracker_mode = str(rc.get("tracker", "joint")).lower()
    if tracker_mode not in {"joint", "cartesian", "ik"}:
        raise ValueError(f"replay.tracker must be 'joint', 'cartesian', or 'ik', got {tracker_mode!r}")
    n_steps = len(timestamps)
    duration = timestamps[-1]

    if tracker_mode != "ik" and np.any(np.isnan(joint_pos)):
        raise ValueError(
            "Episode has NaN joint_pos samples — was joint state captured during recording?"
        )
    has_joint_vel = not np.any(np.isnan(joint_vel))
    ee_pos = episode.get("ee_pos")
    ee_rot = episode.get("ee_rot")
    if tracker_mode in {"cartesian", "ik"}:
        if ee_pos is None or ee_rot is None:
            raise ValueError(f"{tracker_mode} replay requires ee_pos and ee_rot datasets in the episode")
        if np.any(np.isnan(ee_pos)) or np.any(np.isnan(ee_rot)):
            raise ValueError(f"{tracker_mode} replay requires finite ee_pos and ee_rot samples")

    gripper_open_data = episode.get("gripper_open")
    has_gripper_data = (
        gripper_open_data is not None and not np.all(np.isnan(gripper_open_data))
    )

    stiffness = np.asarray(rc.joint_stiffness, dtype=float)
    if stiffness.shape != (7,):
        raise ValueError(f"replay.joint_stiffness must have 7 entries, got shape {stiffness.shape}")

    print(f"  {n_steps} steps, {duration:.1f}s duration")
    print(f"  Replay speed: {rc.speed}x")
    print(f"  Tracker: {tracker_mode}")
    if tracker_mode == "joint":
        print(f"  Joint stiffness: {stiffness.tolist()}")
        print(f"  Joint velocity feedforward: {'enabled' if has_joint_vel else 'disabled (NaN in recording)'}")
    elif tracker_mode == "ik":
        print(f"  Joint stiffness: {stiffness.tolist()}")
        print(f"  IK: Cartesian waypoints converted to a joint trajectory before playback")
    else:
        print(
            "  Cartesian stiffness: "
            f"translation={float(rc.get('translational_stiffness', cfg.teleop.translational_stiffness))}, "
            f"rotation={float(rc.get('rotational_stiffness', cfg.teleop.rotational_stiffness))}"
        )
    if has_gripper_data:
        print(f"  Gripper replay: {'enabled' if gc.get('enabled', False) else 'disabled (gripper.enabled=false in config)'}")
    else:
        print(f"  Gripper replay: disabled (no gripper data in episode)")
    print(f"  Recording: {'enabled' if rc.get('record', False) else 'disabled'}")
    print(f"  Press Ctrl-C to abort.")

    robot = Robot(cfg.robot.ip)
    robot.recover_from_errors()

    gripper = None
    if has_gripper_data and gc.get("enabled", False):
        try:
            from zero_franky.robotiq import RobotiqGripperProxy

            gripper = RobotiqGripperProxy(
                server_host=gc.host,
                server_port=int(gc.port),
                auto_activate=True,
            )
        except Exception as e:
            print(f"  [gripper] Failed to initialize: {e}")
            gripper = None

    joint_pos_ik = None
    if tracker_mode == "ik":
        q_seed, f_t_ee = episode_ik_frame(episode)
        if q_seed is None:
            reset_joint_config = cfg.teleop.get("reset_joint_config", None)
            if reset_joint_config is not None:
                q_seed = np.asarray(reset_joint_config, dtype=float)
        print(f"  Solving IK for {n_steps} waypoints...")
        joint_pos_ik = solve_ik_trajectory(
            timestamps=timestamps,
            ee_pos=ee_pos,
            ee_rot=ee_rot,
            q_seed=q_seed,
            f_t_ee=f_t_ee,
            joint_limit_margin=float(rc.get("ik_joint_limit_margin", 0.02)),
            max_distance=rc.get("ik_max_distance", 0.2),
            position_tolerance=float(rc.get("ik_position_tolerance", 1e-3)),
            rotation_tolerance=float(rc.get("ik_rotation_tolerance", 1e-2)),
        )
        print("  IK solve complete.")

    start_joint_pos = joint_pos[0] if joint_pos_ik is None else joint_pos_ik[0]
    print(f"  Pre-positioning to start configuration...")
    robot.move(JointMotion(
        JointState(start_joint_pos),
        relative_dynamics_factor=0.1,
    ))

    gripper_open_for_record = None
    if gripper is not None:
        initial_gripper_open = bool(round(float(gripper_open_data[0])))
        initial_width = (
            gc.get("open_width_m", 0.085) if initial_gripper_open else gc.get("close_width_m", 0.0)
        )
        gripper.move_width(
            initial_width,
            speed=int(gc.get("speed", 255)),
            force=int(gc.get("force", 255)),
            wait=True,
            max_width_m=gc.get("max_width_m", 0.085),
        )
        gripper_open_for_record = initial_gripper_open

    time.sleep(0.5)

    cameras = {}
    recorder = None
    if rc.get("record", False):
        from clear_franka.camera import enabled_camera_names, make_zed_camera
        for name in enabled_camera_names(cfg):
            cameras[name] = make_zed_camera(cfg, name)
            cameras[name].run()

        vc = cfg.get("visualization", {})
        extrinsics_metadata = {}
        for cam_name in cameras:
            ext_path = vc.get("pointclouds", {}).get(cam_name, {}).get(
                "extrinsics_path", f"./data/extrinsics_{cam_name}.json"
            )
            try:
                with open(ext_path) as f:
                    extrinsics_metadata[f"extrinsics_{cam_name}"] = f.read()
            except OSError:
                pass

        from clear_franka.recorder import TrajectoryRecorder
        recorder = TrajectoryRecorder(
            save_dir=cfg.data_dir,
            cameras=cameras,
            metadata={
                "replay_episode": str(episode_path),
                "replay_speed": float(rc.speed),
                "replay_tracker": tracker_mode,
                "gripper_enabled": gripper is not None,
                **extrinsics_metadata,
            },
        )
        recorder.start()

    try:
        play_fn = play_cartesian_trajectory if tracker_mode == "cartesian" else play_joint_trajectory
        play_joint_pos = joint_pos if joint_pos_ik is None else joint_pos_ik
        play_joint_vel = joint_vel if joint_pos_ik is None else np.zeros_like(joint_pos_ik)
        play_has_joint_vel = has_joint_vel if joint_pos_ik is None else True
        common_kwargs = dict(
            robot=robot,
            rc=rc,
            gc=gc,
            timestamps=timestamps,
            gripper=gripper,
            gripper_open_data=gripper_open_data,
            recorder=recorder,
            gripper_open_for_record=gripper_open_for_record,
        )
        if tracker_mode == "cartesian":
            common_kwargs.update(
                cfg=cfg,
                ee_pos=ee_pos,
                ee_rot=ee_rot,
                joint_pos=joint_pos,
            )
        else:
            common_kwargs.update(
                stiffness=stiffness,
                joint_pos=play_joint_pos,
                joint_vel=play_joint_vel,
                has_joint_vel=play_has_joint_vel,
            )
        gripper_open_for_record = play_with_recovery(play_fn=play_fn, **common_kwargs)
        if recorder is not None:
            recorder.close()
            recorder = None

        if prompt_reverse_reset():
            reverse_timestamps = timestamps[-1] - timestamps[::-1]
            reverse_joint_pos = play_joint_pos[::-1]
            reverse_joint_vel = (
                -play_joint_vel[::-1] if play_has_joint_vel else play_joint_vel[::-1]
            )
            reverse_ee_pos = ee_pos[::-1] if ee_pos is not None else None
            reverse_ee_rot = ee_rot[::-1] if ee_rot is not None else None
            reverse_gripper_open_data = (
                gripper_open_data[::-1] if gripper_open_data is not None else gripper_open_data
            )
            reverse_kwargs = dict(
                robot=robot,
                rc=rc,
                gc=gc,
                timestamps=reverse_timestamps,
                gripper=gripper,
                gripper_open_data=reverse_gripper_open_data,
                gripper_open_for_record=gripper_open_for_record,
            )
            if tracker_mode == "cartesian":
                reverse_kwargs.update(
                    cfg=cfg,
                    ee_pos=reverse_ee_pos,
                    ee_rot=reverse_ee_rot,
                    joint_pos=reverse_joint_pos,
                )
            else:
                reverse_kwargs.update(
                    stiffness=stiffness,
                    joint_pos=reverse_joint_pos,
                    joint_vel=reverse_joint_vel,
                    has_joint_vel=play_has_joint_vel,
                )
            play_with_recovery(play_fn=play_fn, **reverse_kwargs)
            print("  Reverse reset complete.")
        else:
            print("  Reverse reset skipped.")

    except KeyboardInterrupt:
        print("\n  Replay aborted.")
    finally:
        robot.stop_state_stream()
        if recorder is not None:
            recorder.close()
        if gripper is not None:
            gripper.disconnect()
        for cam in cameras.values():
            cam.close()
