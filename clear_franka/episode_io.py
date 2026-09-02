"""Read recorded episode trajectories (MCAP) shared by replay and visualization."""

from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

from clear_franka.recorder import TRAJECTORY_TOPIC


def find_latest_episode(data_dir: str) -> Path:
    """Most recently written episode in `data_dir`.

    Matches every ``*.mcap``, not just the timestamped ``episode_*`` names:
    demos recorded with a mode_title are named ``{mode_title}_{N}.mcap``.
    Ordering is by mtime because those names don't sort chronologically.
    """
    data_path = Path(data_dir)
    episodes = sorted(data_path.glob("*.mcap"), key=lambda p: p.stat().st_mtime)
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
