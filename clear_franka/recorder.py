"""Streaming MCAP trajectory recorder with native ZED SVO2 sidecars."""

import json
import queue
import re
import threading
import time
from pathlib import Path

import numpy as np
from mcap.well_known import MessageEncoding
from mcap.writer import CompressionType, Writer
from mcap_protobuf.schema import register_schema

from clear_franka.proto.trajectory_pb2 import (
    ControlInput,
    GripperState,
    JointState,
    Pose,
    Quaternion,
    TrajectorySample,
    Twist,
    Vector3,
)


EPISODE_SCHEMA_VERSION = "1.0"
TRAJECTORY_TOPIC = "/franka/trajectory"


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "camera"


def _values(value):
    return [] if value is None else np.asarray(value, dtype=float).tolist()


def _finite_float(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _vector3(value) -> Vector3:
    values = np.asarray(value, dtype=float)
    return Vector3(x=values[0], y=values[1], z=values[2])


def _quaternion(rotation) -> Quaternion:
    """Convert a 3x3 rotation matrix to an xyzw unit quaternion."""
    m = np.asarray(rotation, dtype=float)
    trace = np.trace(m)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        x, y, z, w = (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        x, y, z, w = 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        x, y, z, w = (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        x, y, z, w = (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s
    norm = np.linalg.norm([x, y, z, w])
    return Quaternion(x=x / norm, y=y / norm, z=z / norm, w=w / norm)


class TrajectoryRecorder:
    """Record one episode to MCAP without doing file I/O in the control loop."""

    def __init__(self, save_dir="./data", metadata=None, cameras=None,
                 svo_compression="H264"):
        self._save_dir = Path(save_dir)
        self._metadata = metadata or {}
        if cameras is None:
            cameras = {}
        elif not isinstance(cameras, dict):
            cameras = {getattr(cam, "camera_id", f"camera_{i}"): cam for i, cam in enumerate(cameras)}
        self._cameras = dict(cameras)
        self._svo_compression = svo_compression
        self._recording = False
        self._count = 0
        self._start_monotonic_ns = 0
        self._start_wall_ns = 0
        self._start_name = ""
        self._path = None
        self._queue = None
        self._thread = None
        self._writer_error = None

    @property
    def recording(self) -> bool:
        return self._recording

    def toggle(self):
        self.stop() if self._recording else self.start()

    def start(self):
        if self._recording:
            return
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._count = 0
        self._start_monotonic_ns = time.monotonic_ns()
        self._start_wall_ns = time.time_ns()
        self._start_name = time.strftime("%Y%m%d_%H%M%S")
        self._path = self._save_dir / f"episode_{self._start_name}.mcap"
        self._queue = queue.SimpleQueue()
        self._writer_error = None
        self._thread = threading.Thread(target=self._write_loop, daemon=False)
        self._thread.start()

        try:
            for name, camera in self._cameras.items():
                path = self._save_dir / f"episode_{self._start_name}_{_safe_name(name)}_video.svo2"
                camera.start_recording(str(path), svo_compression=self._svo_compression)
        except Exception:
            self._queue.put(None)
            self._thread.join()
            raise

        self._recording = True
        print("  [recorder] RECORDING started")

    def stop(self):
        if not self._recording:
            return
        self._recording = False
        camera_summaries = {}
        for name, camera in self._cameras.items():
            camera_summaries[name] = camera.stop_recording()

        duration_ns = max(0, time.monotonic_ns() - self._start_monotonic_ns)
        self._queue.put(("finish", duration_ns, camera_summaries))
        self._queue.put(None)
        self._thread.join()
        if self._writer_error is not None:
            raise RuntimeError(f"MCAP writer failed: {self._writer_error}") from self._writer_error
        print(f"  [recorder] SAVED {self._count} steps to {self._path}")

    def step(self, ee_pos, ee_rot, cmd_linear_vel, cmd_angular_vel,
             buttons, enabled, joint_pos=None, joint_vel=None, gripper_open=None,
             robot_abs_time=None):
        if not self._recording:
            return
        elapsed_ns = time.monotonic_ns() - self._start_monotonic_ns
        pose = None if ee_pos is None or ee_rot is None else Pose(
            position_m=_vector3(ee_pos),
            orientation=_quaternion(ee_rot),
            parent_frame="base",
            child_frame="fr3_hand_tcp",
        )
        sample = TrajectorySample(
            sequence=self._count,
            episode_time_ns=elapsed_ns,
            joints=JointState(
                position_rad=_values(joint_pos),
                velocity_rad_s=_values(joint_vel),
            ),
            control=ControlInput(
                commanded_twist=Twist(
                    linear_m_s=_vector3(cmd_linear_vel),
                    angular_rad_s=_vector3(cmd_angular_vel),
                    frame="base",
                ),
                buttons=int(buttons),
                enabled=bool(enabled),
            ),
        )
        if pose is not None:
            sample.end_effector_pose.CopyFrom(pose)
        robot_time = _finite_float(robot_abs_time)
        if robot_time is not None:
            sample.robot_time_s = robot_time
        gripper_command = _finite_float(gripper_open)
        if gripper_command is not None:
            sample.gripper.CopyFrom(GripperState(commanded_open=bool(round(gripper_command))))
        self._queue.put(("sample", elapsed_ns, sample))
        self._count += 1

    def _episode_metadata(self):
        data = {
            "schema_version": EPISODE_SCHEMA_VERSION,
            "episode_id": self._start_name,
            "start_wall_time_ns": str(self._start_wall_ns),
            "start_monotonic_time_ns": str(self._start_monotonic_ns),
            "trajectory_topic": TRAJECTORY_TOPIC,
            "joint_names": json.dumps([f"fr3_joint{i}" for i in range(1, 8)]),
            "base_frame": "base",
            "end_effector_frame": "fr3_hand_tcp",
        }
        for key, value in self._metadata.items():
            data[str(key)] = value if isinstance(value, str) else json.dumps(value)
        for name, camera in self._cameras.items():
            prefix = f"camera.{_safe_name(name)}"
            data[f"{prefix}.video_file"] = f"episode_{self._start_name}_{_safe_name(name)}_video.svo2"
            data[f"{prefix}.id"] = str(getattr(camera, "camera_id", name))
            data[f"{prefix}.serial_number"] = str(getattr(camera, "serial_number", "") or "")
            data[f"{prefix}.resolution"] = str(camera.resolution)
            data[f"{prefix}.fps"] = str(camera.fps)
        return data

    def _write_loop(self):
        try:
            with open(self._path, "wb") as stream:
                writer = Writer(stream, compression=CompressionType.ZSTD)
                writer.start(profile="protobuf", library="franka-teleop")
                schema_id = register_schema(writer, TrajectorySample)
                channel_id = writer.register_channel(
                    topic=TRAJECTORY_TOPIC,
                    message_encoding=MessageEncoding.Protobuf,
                    schema_id=schema_id,
                )
                writer.add_metadata("episode", self._episode_metadata())
                while True:
                    item = self._queue.get()
                    if item is None:
                        break
                    kind = item[0]
                    if kind == "sample":
                        _, elapsed_ns, sample = item
                        timestamp_ns = self._start_wall_ns + elapsed_ns
                        writer.add_message(
                            channel_id=channel_id,
                            log_time=timestamp_ns,
                            publish_time=timestamp_ns,
                            sequence=sample.sequence,
                            data=sample.SerializeToString(),
                        )
                    else:
                        _, duration_ns, camera_summaries = item
                        camera_metadata = {}
                        for name, summary in camera_summaries.items():
                            for key, value in summary.items():
                                camera_metadata[f"camera.{_safe_name(name)}.{key}"] = str(value)
                        writer.add_metadata("episode_summary", {
                            "num_steps": str(self._count),
                            "duration_ns": str(duration_ns),
                            **camera_metadata,
                        })
                writer.finish()
        except BaseException as error:
            self._writer_error = error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self):
        if self._recording:
            self.stop()
