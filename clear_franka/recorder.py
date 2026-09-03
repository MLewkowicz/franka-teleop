"""Streaming MCAP trajectory recorder with native ZED SVO2 sidecars."""

import atexit
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

    def __init__(self, save_dir="./data", metadata=None, cameras=None, svo_compression="H264",
                 camera_format="svo", episode_name=None):
        """
        Args:
            save_dir: Directory to save episode files.
            metadata: Optional dict of extra metadata stored in the MCAP
                "episode" metadata record.
            cameras: Optional dict/list of named ZedCamera instances. If provided,
                every camera is recorded with the same trajectory clock epoch.
            svo_compression: SVO compression mode (H264, H265, LOSSLESS, etc.).
                Only used when camera_format="svo".
            camera_format: "svo" for native stereo SVO2 sidecars, "rgb" for
                H.265 mp4 sidecars of the left view only.
            episode_name: Optional base name for saved files. When set, episodes
                are named ``{episode_name}_{N}.mcap`` where N is the next free
                index found by scanning ``save_dir`` (per start()). When None,
                falls back to the timestamped ``episode_{YYYYmmdd_HHMMSS}.mcap``.
        """
        self._save_dir = Path(save_dir)
        self._metadata = metadata or {}
        if cameras is None:
            cameras = {}
        elif not isinstance(cameras, dict):
            cameras = {getattr(cam, "camera_id", f"camera_{i}"): cam for i, cam in enumerate(cameras)}
        self._cameras = dict(cameras)
        self._svo_compression = svo_compression
        self._camera_format = camera_format
        self._episode_name = episode_name
        # Resolved per-episode at start(): the filename stem and its index.
        self._episode_base = ""
        self._episode_index = None
        self._recording = False
        self._count = 0
        self._start_monotonic_ns = 0
        self._start_wall_ns = 0
        self._start_name = ""
        self._path = None
        # Path of the most recently completed episode, set by stop(). None until
        # an episode has been written.
        self.last_saved_path = None
        self._queue = None
        self._thread = None
        self._writer_error = None

    @property
    def recording(self) -> bool:
        return self._recording

    def toggle(self):
        self.stop() if self._recording else self.start()

    def set_metadata(self, **entries):
        self._metadata.update(entries)

    def _video_filename(self, camera_name: str) -> str:
        extension = "svo2" if self._camera_format == "svo" else "mp4"
        return f"{self._episode_base}_{_safe_name(camera_name)}_video.{extension}"

    @staticmethod
    def _next_episode_index(save_dir: Path, safe_name: str) -> int:
        """Lowest free N such that ``{safe_name}_{N}.mcap`` does not exist in save_dir."""
        pattern = re.compile(rf"^{re.escape(safe_name)}_(\d+)\.mcap$")
        max_idx = -1
        if save_dir.is_dir():
            for entry in save_dir.iterdir():
                m = pattern.match(entry.name)
                if m:
                    max_idx = max(max_idx, int(m.group(1)))
        return max_idx + 1

    def start(self):
        if self._recording:
            return
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._count = 0
        self._start_monotonic_ns = time.monotonic_ns()
        self._start_wall_ns = time.time_ns()
        self._start_name = time.strftime("%Y%m%d_%H%M%S")

        # Resolve this episode's filename stem BEFORE anything derives a path
        # from it (_path below, and _video_filename for the camera sidecars).
        # With episode_name set, scan the save dir for the next free
        # {episode_name}_{N} index so demos/replays get stable,
        # human-referenceable names; otherwise use the timestamp.
        if self._episode_name:
            safe = _safe_name(self._episode_name)
            self._episode_index = self._next_episode_index(self._save_dir, safe)
            self._episode_base = f"{safe}_{self._episode_index}"
        else:
            self._episode_index = None
            self._episode_base = f"episode_{self._start_name}"

        self._path = self._save_dir / f"{self._episode_base}.mcap"
        self.last_saved_path = None
        self._queue = queue.SimpleQueue()
        self._writer_error = None
        self._thread = threading.Thread(
            target=self._write_loop, name="mcap-writer", daemon=True
        )
        self._thread.start()
        atexit.register(self._atexit_shutdown)

        try:
            for name, camera in self._cameras.items():
                path = self._save_dir / self._video_filename(name)
                camera.start_recording(
                    str(path),
                    svo_compression=self._svo_compression,
                    format=self._camera_format,
                )
        except BaseException:
            # The writer blocks on the queue, so it has to be shut down on every
            # failure path — otherwise it lingers holding a half-written file.
            self._shutdown_writer()
            raise

        self._recording = True
        print(f"  [recorder] RECORDING started -> {self._path.name}")

    def _shutdown_writer(self) -> None:
        """Send the sentinel and join the writer thread. Idempotent.

        Safe to re-enter after an interrupted stop(): a Ctrl-C landing in the
        middle of teardown must not leave the writer running on a half-written
        file, so the join is retried rather than abandoned on interrupt.
        """
        thread = self._thread
        if thread is None:
            return
        self._thread = None
        if self._queue is not None:
            self._queue.put(None)
        deadline = time.monotonic() + 30.0
        warned = False
        while thread.is_alive() and time.monotonic() < deadline:
            try:
                thread.join(timeout=1.0)
            except KeyboardInterrupt:
                # An impatient second Ctrl-C: keep waiting rather than orphan a
                # thread that is mid-write.
                if not warned:
                    print("  [recorder] finishing the episode file; please wait...")
                    warned = True
        if thread.is_alive():
            print(f"  [recorder] WARNING: writer did not finish {self._path} within 30s")

    def _atexit_shutdown(self) -> None:
        """Last-chance flush for a recorder that was never stopped.

        Runs before daemon threads are torn down, so the episode still gets a
        valid footer instead of being truncated mid-write.
        """
        if self._recording:
            try:
                self.stop()
            except BaseException as error:  # never raise out of atexit
                print(f"  [recorder] error finishing episode at exit: {error}")
        else:
            self._shutdown_writer()

    def stop(self):
        if not self._recording:
            # A previous stop() may have been interrupted before the writer was
            # shut down. Finish that job instead of returning and leaving it.
            self._shutdown_writer()
            return
        self._recording = False
        try:
            camera_summaries = {}
            for name, camera in self._cameras.items():
                try:
                    camera_summaries[name] = camera.stop_recording()
                except Exception as error:
                    # One camera failing to finalize must not cost us the
                    # trajectory or strand the writer thread.
                    print(f"  [recorder] camera {name} stop_recording failed: {error}")
                    camera_summaries[name] = {"frame_count": 0, "error": str(error)}

            duration_ns = max(0, time.monotonic_ns() - self._start_monotonic_ns)
            self._queue.put(("finish", duration_ns, camera_summaries))
        finally:
            # Always join, even if camera teardown raised or was interrupted.
            self._shutdown_writer()
            atexit.unregister(self._atexit_shutdown)
        if self._writer_error is not None:
            raise RuntimeError(f"MCAP writer failed: {self._writer_error}") from self._writer_error
        self.last_saved_path = self._path
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

    @property
    def episode_base(self) -> str:
        """Filename stem of the current/most recent episode (no extension)."""
        return self._episode_base

    @property
    def episode_path(self):
        """Path of the current/most recent episode MCAP, or None before start()."""
        return self._path

    def artifact_paths(self) -> list[Path]:
        """Every file the current/most recent episode writes: the MCAP plus one
        sidecar per camera. Lets callers keep or discard an episode as a unit
        without re-deriving names and extensions."""
        if self._path is None:
            return []
        return [self._path] + [
            self._save_dir / self._video_filename(name) for name in self._cameras
        ]

    def _episode_metadata(self):
        data = {
            "schema_version": EPISODE_SCHEMA_VERSION,
            "episode_id": self._episode_base,
            "episode_base": self._episode_base,
            "start_wall_time_ns": str(self._start_wall_ns),
            "start_monotonic_time_ns": str(self._start_monotonic_ns),
            "trajectory_topic": TRAJECTORY_TOPIC,
            "joint_names": json.dumps([f"fr3_joint{i}" for i in range(1, 8)]),
            "base_frame": "base",
            "end_effector_frame": "fr3_hand_tcp",
        }
        if self._episode_name:
            data["episode_name"] = _safe_name(self._episode_name)
        if self._episode_index is not None:
            data["episode_index"] = str(self._episode_index)
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
        else:
            # Settles a stop() that was interrupted before it finished.
            self._shutdown_writer()
            atexit.unregister(self._atexit_shutdown)
