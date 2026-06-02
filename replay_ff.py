"""Replay a recorded trajectory using joint impedance control."""

import time
from pathlib import Path

import h5py
import numpy as np
from omegaconf import DictConfig

from zero_franky import Robot
from franky import JointMotion, JointState

from clear_franka.franka import DEFAULT_LOWER_JOINT_LIMITS, DEFAULT_UPPER_JOINT_LIMITS


def find_latest_episode(data_dir: str) -> Path:
    data_path = Path(data_dir)
    episodes = sorted(data_path.glob("episode_*.h5"))
    if not episodes:
        raise FileNotFoundError(f"No episodes found in {data_dir}")
    return episodes[-1]


def load_episode(path: Path) -> dict:
    data = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                data[key] = f[key][:]
        data["attrs"] = dict(f.attrs)
    return data


def prompt_reverse_reset() -> bool:
    try:
        answer = input("  Play trajectory in reverse to reset robot? [Y/n] ").strip().lower()
    except EOFError:
        print("  Reverse reset skipped (no input available).")
        return False
    return answer in ("", "y", "yes")


def make_inverse_dynamics_replay_policy(
    *,
    timestamps: np.ndarray,
    joint_pos: np.ndarray,
    speed: float,
):
    """Build a cloudpickle-friendly policy that runs on the zero_franky server."""
    timestamps = np.asarray(timestamps, dtype=float)
    joint_pos = np.asarray(joint_pos, dtype=float)

    def policy(context):
        import numpy as _np
        from scipy.interpolate import make_interp_spline

        model = context.robot.model
        k = 3 if len(timestamps) >= 4 else 1
        kwargs = {"bc_type": "clamped"} if k == 3 else {}
        position_spline = make_interp_spline(timestamps, joint_pos, k=k, **kwargs)
        velocity_spline = position_spline.derivative(1)
        acceleration_spline = position_spline.derivative(2) if k >= 2 else None

        def step(context):
            elapsed = context.elapsed * speed
            if elapsed >= timestamps[-1]:
                return {
                    "position": joint_pos[-1].tolist(),
                    "velocity": [0.0] * 7,
                    "torque_feedforward": [0.0] * 7,
                }

            q = _np.asarray(position_spline(elapsed), dtype=float).reshape(7)
            dq = _np.asarray(velocity_spline(elapsed), dtype=float).reshape(7) * speed
            if acceleration_spline is None:
                ddq = _np.zeros(7, dtype=float)
            else:
                ddq = _np.asarray(acceleration_spline(elapsed), dtype=float).reshape(7)
                ddq = ddq * speed * speed

            state = context.robot.state
            mass = _np.asarray(model.mass(state), dtype=float)
            tau = _np.ravel(mass @ ddq)

            return {
                "position": q.tolist(),
                "velocity": dq.tolist(),
                "torque_feedforward": tau.tolist(),
            }

        return step

    return policy


def play_joint_trajectory(
    *,
    robot: Robot,
    rc: DictConfig,
    gc,
    stiffness: np.ndarray,
    timestamps: np.ndarray,
    joint_pos: np.ndarray,
    gripper,
    gripper_open_data,
    recorder=None,
    gripper_open_for_record=None,
):
    n_steps = len(timestamps)

    motion_kwargs = {
        "stiffness": stiffness.tolist(),
        "compensate_coriolis": True,
        "lower_joint_limits": DEFAULT_LOWER_JOINT_LIMITS,
        "upper_joint_limits": DEFAULT_UPPER_JOINT_LIMITS,
    }
    policy = make_inverse_dynamics_replay_policy(
        timestamps=timestamps,
        joint_pos=joint_pos,
        speed=float(rc.speed),
    )

    session_kwargs = {
        "period": rc.period,
        "policy_transport": "cloudpickle",
        **motion_kwargs,
    }

    with robot.start_joint_impedance_session(policy, **session_kwargs) as session:
        step = 0
        replay_start = None
        last_status_check = time.monotonic()
        last_gripper_open = None

        while True:
            if replay_start is None:
                replay_start = time.monotonic()

            elapsed = (time.monotonic() - replay_start) * rc.speed

            while step < n_steps - 1 and timestamps[step + 1] <= elapsed:
                step += 1

            now = time.monotonic()
            if now - last_status_check >= 0.5:
                status = session.status()
                if status.get("error"):
                    raise RuntimeError(f"Joint replay policy failed: {status['error']}")
                if not status.get("running", False):
                    raise RuntimeError(f"Joint replay policy stopped: {status}")
                last_status_check = now

            if elapsed >= timestamps[-1]:
                print("  Replay complete.")
                session.set_joint_reference(joint_pos[-1].tolist())
                break

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
            if "Joint replay policy " in str(e):
                raise
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

    timestamps = episode["timestamps"]
    joint_pos = episode["joint_pos"]
    joint_vel = episode["joint_vel"]
    n_steps = len(timestamps)
    duration = timestamps[-1]
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("Episode timestamps must be strictly increasing")

    if np.any(np.isnan(joint_pos)):
        raise ValueError(
            "Episode has NaN joint_pos samples — was joint state captured during recording?"
        )

    gripper_open_data = episode.get("gripper_open")
    has_gripper_data = (
        gripper_open_data is not None and not np.all(np.isnan(gripper_open_data))
    )

    stiffness = np.asarray(rc.joint_stiffness, dtype=float)
    print(f"  {n_steps} steps, {duration:.1f}s duration")
    print(f"  Replay speed: {rc.speed}x")
    print(f"  Joint stiffness: {stiffness.tolist()}")
    print("  Joint velocity feedforward: spline derivative")
    print("  Inverse dynamics feedforward: spline acceleration")
    print("    Coriolis compensation: franky motion")
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
            from clear_franka.robotiq_net_proxy import RobotiqGripperProxy

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
            ext_path = cfg.get("cameras", {}).get(cam_name, {}).get(
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
                gripper=gripper,
                gripper_open_data=reverse_gripper_open_data,
                gripper_open_for_record=gripper_open_for_record,
            )
            print("  Reverse reset complete.")
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
