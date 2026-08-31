"""Replay a recorded trajectory using joint impedance control."""

import time
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

from clear_franka.recorder import TRAJECTORY_TOPIC
from omegaconf import DictConfig

from zero_franky import Robot
from franky import Affine, JointMotion, JointState, ManipulabilityTask, PostureTask, Twist

from clear_franka.franka import (
    DEFAULT_LOWER_JOINT_LIMITS,
    DEFAULT_UPPER_JOINT_LIMITS,
    joint_friction_kwargs,
)
from clear_franka.cartesian_trajectory import CartesianTrajectory
from clear_franka.franka import DEFAULT_LOWER_JOINT_LIMITS, DEFAULT_UPPER_JOINT_LIMITS
from clear_franka.geometry import pack_Rp
from clear_franka.joint_trajectory import Trajectory
from clear_franka.preprocess import preprocess_episode_arrays


def find_latest_episode(data_dir: str) -> Path:
    data_path = Path(data_dir)
    episodes = sorted(data_path.glob("episode_*.mcap"))
    if not episodes:
        raise FileNotFoundError(f"No episodes found in {data_dir}")
    return episodes[-1]


def load_episode(path: Path) -> dict:
    samples = []
    attrs = {}
    with open(path, "rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for metadata in reader.iter_metadata():
            attrs.update(metadata.metadata)
        for _, _, _, sample in reader.iter_decoded_messages(topics=[TRAJECTORY_TOPIC]):
            samples.append(sample)
    if not samples:
        raise ValueError(f"No {TRAJECTORY_TOPIC} samples found in {path}")

    def optional(sample, field):
        return getattr(sample, field) if sample.HasField(field) else np.nan

    def rotation_matrix(q):
        x, y, z, w = q.x, q.y, q.z, q.w
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])

    def joint_vector(values):
        return list(values) if values else [np.nan] * 7

    data = {
        "timestamps": np.asarray([sample.episode_time_ns / 1e9 for sample in samples]),
        "robot_abs_time": np.asarray([optional(sample, "robot_time_s") for sample in samples]),
        "joint_pos": np.asarray([joint_vector(sample.joints.position_rad) for sample in samples]),
        "joint_vel": np.asarray([joint_vector(sample.joints.velocity_rad_s) for sample in samples]),
        "ee_pos": np.asarray([
            [sample.end_effector_pose.position_m.x,
             sample.end_effector_pose.position_m.y,
             sample.end_effector_pose.position_m.z]
            if sample.HasField("end_effector_pose") else [np.nan] * 3
            for sample in samples
        ]),
        "ee_rot": np.asarray([
            rotation_matrix(sample.end_effector_pose.orientation)
            if sample.HasField("end_effector_pose") else np.full((3, 3), np.nan)
            for sample in samples
        ]),
        "cmd_linear_vel": np.asarray([
            [sample.control.commanded_twist.linear_m_s.x,
             sample.control.commanded_twist.linear_m_s.y,
             sample.control.commanded_twist.linear_m_s.z]
            for sample in samples
        ]),
        "cmd_angular_vel": np.asarray([
            [sample.control.commanded_twist.angular_rad_s.x,
             sample.control.commanded_twist.angular_rad_s.y,
             sample.control.commanded_twist.angular_rad_s.z]
            for sample in samples
        ]),
        "buttons": np.asarray([sample.control.buttons for sample in samples]),
        "enabled": np.asarray([sample.control.enabled for sample in samples]),
        "gripper_open": np.asarray([
            float(sample.gripper.commanded_open)
            if sample.HasField("gripper") and sample.gripper.HasField("commanded_open") else np.nan
            for sample in samples
        ]),
    }
    data["attrs"] = attrs
    return data


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
    if tracker_mode not in {"joint", "cartesian"}:
        raise ValueError(f"replay.tracker must be 'joint' or 'cartesian', got {tracker_mode!r}")
    n_steps = len(timestamps)
    duration = timestamps[-1]

    if np.any(np.isnan(joint_pos)):
        raise ValueError(
            "Episode has NaN joint_pos samples — was joint state captured during recording?"
        )
    has_joint_vel = not np.any(np.isnan(joint_vel))
    ee_pos = episode.get("ee_pos")
    ee_rot = episode.get("ee_rot")
    if tracker_mode == "cartesian":
        if ee_pos is None or ee_rot is None:
            raise ValueError("Cartesian replay requires ee_pos and ee_rot datasets in the episode")
        if np.any(np.isnan(ee_pos)) or np.any(np.isnan(ee_rot)):
            raise ValueError("Cartesian replay requires finite ee_pos and ee_rot samples")

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

    print(f"  Pre-positioning to start configuration...")
    robot.move(JointMotion(
        JointState(joint_pos[0]),
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
                joint_pos=joint_pos,
                joint_vel=joint_vel,
                has_joint_vel=has_joint_vel,
            )
        gripper_open_for_record = play_with_recovery(play_fn=play_fn, **common_kwargs)
        if recorder is not None:
            recorder.close()
            recorder = None

        if prompt_reverse_reset():
            reverse_timestamps = timestamps[-1] - timestamps[::-1]
            reverse_joint_pos = joint_pos[::-1]
            reverse_joint_vel = -joint_vel[::-1] if has_joint_vel else joint_vel[::-1]
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
                    has_joint_vel=has_joint_vel,
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
