"""Replay a recorded trajectory using joint impedance control."""

import time
from pathlib import Path

import h5py
import numpy as np
from omegaconf import DictConfig

from net_franky.franky import (
    ControlException,
    JointImpedanceTracker,
    JointMotion,
    JointState,
    Robot,
)

DEFAULT_LOWER_JOINT_LIMITS = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
DEFAULT_UPPER_JOINT_LIMITS = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]


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
            from clear_franka.robotiq_net_proxy import RobotiqGripperProxy

            gripper = RobotiqGripperProxy(
                server_host=gc.get("host", cfg.net_franky.ip),
                server_port=int(gc.get("port", cfg.net_franky.port)),
                com_port=gc.get("com_port", "auto"),
                device_id=int(gc.get("device_id", 9)),
                connection_type=gc.get("connection_type", "RTU"),
                tcp_host=gc.get("tcp_host", "127.0.0.1"),
                tcp_port=int(gc.get("tcp_port", 54321)),
                auto_activate=bool(gc.get("activate_on_start", True)),
            )
            if gc.get("activate_on_start", True):
                print("  Robotiq gripper activated.")
        except Exception as e:
            print(f"  [gripper] Failed to initialize: {e}")
            gripper = None

    print(f"  Pre-positioning to start configuration...")
    robot.move(JointMotion(
        JointState(joint_pos[0]),
        relative_dynamics_factor=float(rc.preposition_dynamics_factor),
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

    time.sleep(float(rc.preposition_settle_s))

    recorder = None
    if rc.get("record", False):
        from clear_franka.recorder import TrajectoryRecorder
        recorder = TrajectoryRecorder(
            save_dir=cfg.data_dir,
            metadata={
                "replay_episode": str(episode_path),
                "replay_speed": float(rc.speed),
                "gripper_enabled": gripper is not None,
            },
        )
        recorder.start()

    try:
        while True:
            robot.recover_from_errors()

            try:
                with JointImpedanceTracker(
                    robot,
                    stiffness=stiffness,
                    lower_joint_limits=DEFAULT_LOWER_JOINT_LIMITS,
                    upper_joint_limits=DEFAULT_UPPER_JOINT_LIMITS,
                    period=rc.period,
                ) as tracker:
                    step = 0
                    replay_start = None
                    last_gripper_open = None

                    while tracker.tick():
                        if replay_start is None:
                            replay_start = time.monotonic()

                        elapsed = (time.monotonic() - replay_start) * rc.speed

                        while step < n_steps - 1 and timestamps[step + 1] <= elapsed:
                            step += 1

                        if step >= n_steps - 1:
                            print("  Replay complete.")
                            tracker.set_target(joint_pos[-1])
                            break

                        q = joint_pos[step]
                        if has_joint_vel:
                            dq = joint_vel[step] * rc.speed
                            tracker.set_target(q, dq=dq)
                        else:
                            tracker.set_target(q)

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

                break

            except ControlException as e:
                print(f"\n  Controller faulted: {e}")
                print("  Recovering and retrying...")

    except KeyboardInterrupt:
        print("\n  Replay aborted.")
    finally:
        if recorder is not None:
            recorder.close()
        if gripper is not None:
            gripper.disconnect()
