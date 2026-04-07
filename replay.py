"""Replay a recorded trajectory using Cartesian impedance control."""

import time
from pathlib import Path

import h5py
import numpy as np
from omegaconf import DictConfig

from net_franky.franky import (
    Affine,
    CartesianImpedanceTracker,
    ControlException,
    Robot,
    Twist,
)
from threed_mouse.geometry import pack_Rp


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
            data[key] = f[key][:]
        data["attrs"] = dict(f.attrs)
    return data


def run_replay(cfg: DictConfig):
    rc = cfg.replay

    if rc.episode is not None:
        episode_path = Path(rc.episode)
    else:
        episode_path = find_latest_episode(cfg.data_dir)

    print(f"Loading episode: {episode_path}")
    episode = load_episode(episode_path)

    timestamps = episode["timestamps"]
    ee_pos = episode["ee_pos"]
    ee_rot = episode["ee_rot"]
    cmd_linear_vel = episode["cmd_linear_vel"]
    cmd_angular_vel = episode["cmd_angular_vel"]
    enabled = episode["enabled"]
    n_steps = len(timestamps)
    duration = timestamps[-1]

    print(f"  {n_steps} steps, {duration:.1f}s duration")
    print(f"  Replay speed: {rc.speed}x")
    print(f"  Press Ctrl-C to abort.")

    robot = Robot(cfg.robot.ip)
    robot.recover_from_errors()

    try:
        while True:
            robot.recover_from_errors()

            try:
                with CartesianImpedanceTracker(
                    robot,
                    translational_stiffness=rc.translational_stiffness,
                    rotational_stiffness=rc.rotational_stiffness,
                    nullspace_stiffness=rc.nullspace_stiffness,
                    lower_joint_limits=[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
                    upper_joint_limits=[2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
                    period=rc.period,
                ) as tracker:
                    step = 0
                    replay_start = None

                    while tracker.tick():
                        if replay_start is None:
                            replay_start = time.monotonic()

                        elapsed = (time.monotonic() - replay_start) * rc.speed

                        # Advance to the step matching current elapsed time.
                        while step < n_steps - 1 and timestamps[step + 1] <= elapsed:
                            step += 1

                        if step >= n_steps - 1:
                            print("  Replay complete.")
                            # Hold final pose briefly then exit.
                            current = tracker.current_pose.end_effector_pose
                            tracker.set_target(current)
                            break

                        if enabled[step]:
                            pos = ee_pos[step]
                            rot = ee_rot[step]
                            v = cmd_linear_vel[step]
                            w = cmd_angular_vel[step]

                            pose = Affine(pack_Rp(rot, pos))
                            tracker.set_target(pose, Twist(v * rc.speed, w * rc.speed))
                        else:
                            current = tracker.current_pose.end_effector_pose
                            tracker.set_target(current)

                break  # Clean exit after replay completes.

            except ControlException as e:
                print(f"\n  Controller faulted: {e}")
                print("  Recovering and retrying...")

    except KeyboardInterrupt:
        print("\n  Replay aborted.")
