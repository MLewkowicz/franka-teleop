"""Replay a recorded trajectory using joint impedance control."""

import time
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

from clear_franka.recorder import TRAJECTORY_TOPIC

from clear_franka.config import ConfigDict, load_app_config
from zero_franky import Robot
from franky import JointMotion, JointState

from clear_franka.franka import DEFAULT_LOWER_JOINT_LIMITS, DEFAULT_UPPER_JOINT_LIMITS


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


def play_joint_trajectory(
    *,
    robot: Robot,
    rc: ConfigDict,
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
    complete_message: str = "Replay complete.",
):
    n_steps = len(timestamps)

    with robot.start_joint_impedance_tracker(
        period=rc.period,
        stiffness=stiffness,
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

            if step >= n_steps - 1:
                print(f"  {complete_message}")
                session.set_target(joint_pos[-1])
                break

            q = joint_pos[step]
            if has_joint_vel:
                dq = joint_vel[step] * rc.speed
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


def play_with_recovery(**kwargs):
    robot = kwargs["robot"]
    while True:
        robot.recover_from_errors()
        try:
            return play_joint_trajectory(**kwargs)
        except RuntimeError as e:
            print(f"\n  Controller faulted: {e}")
            print("  Recovering and retrying...")


def run_replay(cfg: ConfigDict):
    rc = cfg.replay
    gc = cfg.get("gripper", {})

    if rc.episode is not None:
        episode_path = Path(rc.episode)
    else:
        episode_path = find_latest_episode(cfg.data_dir)

    print(f"Loading episode: {episode_path}")
    episode = load_episode(episode_path)

    timestamps = episode["timestamps"]
    joint_pos = episode["joint_pos"]
    joint_vel = episode["joint_vel"]
    n_steps = len(timestamps)
    duration = timestamps[-1]

    if np.any(np.isnan(joint_pos)):
        raise ValueError(
            "Episode has NaN joint_pos samples — was joint state captured during recording?"
        )
    has_joint_vel = not np.any(np.isnan(joint_vel))

    gripper_open_data = episode.get("gripper_open")
    has_gripper_data = (
        gripper_open_data is not None and not np.all(np.isnan(gripper_open_data))
    )

    stiffness = np.asarray(rc.joint_stiffness, dtype=float)
    if stiffness.shape != (7,):
        raise ValueError(f"replay.joint_stiffness must have 7 entries, got shape {stiffness.shape}")

    print(f"  {n_steps} steps, {duration:.1f}s duration")
    print(f"  Replay speed: {rc.speed}x")
    print(f"  Joint stiffness: {stiffness.tolist()}")
    print(f"  Joint velocity feedforward: {'enabled' if has_joint_vel else 'disabled (NaN in recording)'}")
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
                "gripper_enabled": gripper is not None,
                **extrinsics_metadata,
            },
        )
        recorder.start()

    try:
        gripper_open_for_record = play_with_recovery(
            robot=robot,
            rc=rc,
            gc=gc,
            stiffness=stiffness,
            timestamps=timestamps,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            has_joint_vel=has_joint_vel,
            gripper=gripper,
            gripper_open_data=gripper_open_data,
            recorder=recorder,
            gripper_open_for_record=gripper_open_for_record,
        )
        if recorder is not None:
            recorder.close()
            recorder = None

        if prompt_reverse_reset():
            reverse_timestamps = timestamps[-1] - timestamps[::-1]
            reverse_joint_pos = joint_pos[::-1]
            reverse_joint_vel = -joint_vel[::-1] if has_joint_vel else joint_vel[::-1]
            reverse_gripper_open_data = (
                gripper_open_data[::-1] if gripper_open_data is not None else gripper_open_data
            )
            play_with_recovery(
                robot=robot,
                rc=rc,
                gc=gc,
                stiffness=stiffness,
                timestamps=reverse_timestamps,
                joint_pos=reverse_joint_pos,
                joint_vel=reverse_joint_vel,
                has_joint_vel=has_joint_vel,
                gripper=gripper,
                gripper_open_data=reverse_gripper_open_data,
                gripper_open_for_record=gripper_open_for_record,
                complete_message="Reverse reset complete.",
            )
        else:
            print("  Reverse reset skipped.")

    except KeyboardInterrupt:
        print("\n  Replay aborted.")
    finally:
        if recorder is not None:
            recorder.close()
        if gripper is not None:
            gripper.disconnect()
        for cam in cameras.values():
            cam.close()


if __name__ == "__main__":
    from zero_franky import setup_zero_franky

    cfg = load_app_config(__file__)
    setup_zero_franky(cfg.zero_franky.ip, cfg.zero_franky.port, pub_port=cfg.zero_franky.pub_port)
    run_replay(cfg)
