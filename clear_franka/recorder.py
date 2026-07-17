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
    RolloutStep,
    TrajectorySample,
    Twist,
    Vector3,
    Wrench,
)


EPISODE_SCHEMA_VERSION = "1.0"
TRAJECTORY_TOPIC = "/franka/trajectory"
ROLLOUT_TOPIC = "/franka/rollout"


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


def _wrench(value, frame: str) -> Wrench | None:
    """Build a Wrench from a 6-vector [Fx, Fy, Fz, Tx, Ty, Tz], or None if unavailable."""
    if value is None:
        return None
    values = np.asarray(value, dtype=float)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        return None
    return Wrench(force_n=_vector3(values[:3]), torque_nm=_vector3(values[3:]), frame=frame)


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
                 svo_compression="H264", camera_format="svo"):
        self._save_dir = Path(save_dir)
        self._metadata = metadata or {}
        if cameras is None:
            cameras = {}
        elif not isinstance(cameras, dict):
            cameras = {getattr(cam, "camera_id", f"camera_{i}"): cam for i, cam in enumerate(cameras)}
        self._cameras = dict(cameras)
        self._svo_compression = svo_compression
        self._camera_format = camera_format
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

    def _video_filename(self, camera_name: str) -> str:
        extension = "svo2" if self._camera_format == "svo" else "mp4"
        return f"episode_{self._start_name}_{_safe_name(camera_name)}_video.{extension}"

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
                path = self._save_dir / self._video_filename(name)
                camera.start_recording(str(path), svo_compression=self._svo_compression, format=self._camera_format)
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
             buttons, enabled, joint_pos=None, joint_vel=None, joint_effort=None,
             gripper_open=None, measured_width_m=None, motor_current_ma=None,
             object_detection=None, measured_age_s=None,
             ext_wrench=None, measured_linear_vel=None, measured_angular_vel=None,
             robot_abs_time=None):
        """Record one robot observation+action sample. Returns its sequence number
        (for joining a later log_rollout_step() call to it), or None if not recording."""
        if not self._recording:
            return None
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
                effort_nm=_values(joint_effort),
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
        wrench = _wrench(ext_wrench, "stiffness")
        if wrench is not None:
            sample.external_wrench.CopyFrom(wrench)
        if measured_linear_vel is not None and measured_angular_vel is not None:
            sample.measured_twist.CopyFrom(Twist(
                linear_m_s=_vector3(measured_linear_vel),
                angular_rad_s=_vector3(measured_angular_vel),
                frame="base",
            ))
        robot_time = _finite_float(robot_abs_time)
        if robot_time is not None:
            sample.robot_time_s = robot_time
        gripper_command = _finite_float(gripper_open)
        measured_width = _finite_float(measured_width_m)
        motor_current = _finite_float(motor_current_ma)
        measured_age = _finite_float(measured_age_s)
        if any(v is not None for v in (gripper_command, measured_width, motor_current, object_detection)):
            gripper_state = GripperState()
            if gripper_command is not None:
                gripper_state.commanded_open = bool(round(gripper_command))
            if measured_width is not None:
                gripper_state.width_m = measured_width
            if motor_current is not None:
                gripper_state.motor_current_ma = motor_current
            if object_detection is not None:
                gripper_state.object_detection = int(object_detection)
            if measured_age is not None:
                gripper_state.measured_age_s = measured_age
            sample.gripper.CopyFrom(gripper_state)
        self._queue.put(("sample", elapsed_ns, sample))
        self._count += 1
        return sample.sequence

    def log_rollout_step(self, sequence, reward, terminal=False, termination_reason="",
                          is_intervention=False, policy_linear_vel=None, policy_angular_vel=None,
                          policy_buttons=0, policy_enabled=False):
        """Attach RL bookkeeping to the TrajectorySample with the given `sequence`
        (as returned by step()). Only ever called from rollout.py, never teleop.py."""
        if not self._recording:
            return
        elapsed_ns = time.monotonic_ns() - self._start_monotonic_ns
        rollout_step = RolloutStep(
            sequence=int(sequence),
            reward=float(reward),
            terminal=bool(terminal),
            termination_reason=str(termination_reason),
            is_intervention=bool(is_intervention),
        )
        if policy_linear_vel is not None and policy_angular_vel is not None:
            rollout_step.policy_action.CopyFrom(ControlInput(
                commanded_twist=Twist(
                    linear_m_s=_vector3(policy_linear_vel),
                    angular_rad_s=_vector3(policy_angular_vel),
                    frame="base",
                ),
                buttons=int(policy_buttons),
                enabled=bool(policy_enabled),
            ))
        self._queue.put(("rollout", elapsed_ns, rollout_step))

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
            data[f"{prefix}.video_file"] = self._video_filename(name)
            data[f"{prefix}.video_format"] = self._camera_format
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
                rollout_schema_id = register_schema(writer, RolloutStep)
                rollout_channel_id = writer.register_channel(
                    topic=ROLLOUT_TOPIC,
                    message_encoding=MessageEncoding.Protobuf,
                    schema_id=rollout_schema_id,
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
                    elif kind == "rollout":
                        _, elapsed_ns, rollout_step = item
                        timestamp_ns = self._start_wall_ns + elapsed_ns
                        writer.add_message(
                            channel_id=rollout_channel_id,
                            log_time=timestamp_ns,
                            publish_time=timestamp_ns,
                            sequence=rollout_step.sequence,
                            data=rollout_step.SerializeToString(),
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
