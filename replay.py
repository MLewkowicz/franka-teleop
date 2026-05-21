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
    joint_pos = episode["joint_pos"]
    joint_vel = episode["joint_vel"]
    n_steps = len(timestamps)
    duration = timestamps[-1]

    if np.any(np.isnan(joint_pos)):
        raise ValueError(
            "Episode has NaN joint_pos samples — was joint state captured during recording?"
        )
    has_joint_vel = not np.any(np.isnan(joint_vel))

    stiffness = np.asarray(rc.joint_stiffness, dtype=float)
    if stiffness.shape != (7,):
        raise ValueError(f"replay.joint_stiffness must have 7 entries, got shape {stiffness.shape}")

    print(f"  {n_steps} steps, {duration:.1f}s duration")
    print(f"  Replay speed: {rc.speed}x")
    print(f"  Joint stiffness: {stiffness.tolist()}")
    print(f"  Joint velocity feedforward: {'enabled' if has_joint_vel else 'disabled (NaN in recording)'}")
    print(f"  Press Ctrl-C to abort.")

    robot = Robot(cfg.robot.ip)
    robot.recover_from_errors()

    print(f"  Pre-positioning to start configuration...")
    robot.move(JointMotion(
        JointState(joint_pos[0]),
        relative_dynamics_factor=float(rc.preposition_dynamics_factor),
    ))
    time.sleep(float(rc.preposition_settle_s))

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

                break

            except ControlException as e:
                print(f"\n  Controller faulted: {e}")
                print("  Recovering and retrying...")

    except KeyboardInterrupt:
        print("\n  Replay aborted.")
