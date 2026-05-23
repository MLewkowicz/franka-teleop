"""Trajectory recorder for Franka teleoperation.

Records timestamped episodes to HDF5 files for future replay.
Designed for minimal overhead in a 1kHz control loop.
"""

import re
import time
from pathlib import Path

import h5py
import numpy as np


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "camera"


class TrajectoryRecorder:

    def __init__(self, save_dir="./data", capacity=120_000, metadata=None, cameras=None,
                 record_svo=False, svo_compression="H264"):
        """
        Args:
            save_dir: Directory to save episode files.
            capacity: Pre-allocated buffer size in timesteps (120k = 2 min at 1kHz).
            metadata: Optional dict of extra metadata stored in HDF5 attrs.
            cameras: Optional dict/list of named ZedCamera instances. If provided,
                every camera is recorded with the same trajectory clock epoch.
            record_svo: If True, cameras record native ZED SVO2 files instead of HDF5.
            svo_compression: SVO compression mode (H264, H265, LOSSLESS, etc.).
                             Only used when record_svo=True.
        """
        self._save_dir = Path(save_dir)
        self._capacity = capacity
        self._metadata = metadata or {}
        if cameras is None:
            cameras = {}
        elif not isinstance(cameras, dict):
            cameras = {getattr(cam, "camera_id", f"camera_{i}"): cam for i, cam in enumerate(cameras)}
        self._cameras = dict(cameras)
        self._record_svo = record_svo
        self._svo_compression = svo_compression
        self._recording = False
        self._count = 0
        self._start_time = 0.0
        self._start_wall = ""
        self._alloc_buffers()

    def _alloc_buffers(self):
        n = self._capacity
        self._buffers = {
            "timestamps": np.empty(n, dtype=np.float64),
            "robot_abs_time": np.empty(n, dtype=np.float64),
            "ee_pos": np.empty((n, 3), dtype=np.float64),
            "ee_rot": np.empty((n, 3, 3), dtype=np.float64),
            "joint_pos": np.empty((n, 7), dtype=np.float64),
            "joint_vel": np.empty((n, 7), dtype=np.float64),
            "cmd_linear_vel": np.empty((n, 3), dtype=np.float64),
            "cmd_angular_vel": np.empty((n, 3), dtype=np.float64),
            "buttons": np.empty(n, dtype=np.int32),
            "enabled": np.empty(n, dtype=bool),
            "gripper_open": np.empty(n, dtype=np.float64),
        }

    def _grow_buffers(self):
        old = self._capacity
        self._capacity *= 2
        for key, buf in self._buffers.items():
            new_shape = list(buf.shape)
            new_shape[0] = self._capacity
            new_buf = np.empty(new_shape, dtype=buf.dtype)
            new_buf[:old] = buf[:old]
            self._buffers[key] = new_buf

    @property
    def recording(self) -> bool:
        return self._recording

    def toggle(self):
        if self._recording:
            self.stop()
        else:
            self.start()

    def start(self):
        if self._recording:
            return
        self._count = 0
        self._start_time = time.monotonic()
        self._start_wall = time.strftime("%Y%m%d_%H%M%S")
        self._recording = True

        self._save_dir.mkdir(parents=True, exist_ok=True)
        video_ext = "svo2" if self._record_svo else "hdf5"
        for name, camera in self._cameras.items():
            video_path = str(
                self._save_dir / f"episode_{self._start_wall}_{_safe_name(name)}_video.{video_ext}"
            )
            camera.start_recording(video_path, self._start_time,
                                   svo=self._record_svo, svo_compression=self._svo_compression)

        print("  [recorder] RECORDING started")

    def stop(self):
        if not self._recording:
            return
        self._recording = False

        camera_ts = {}
        for name, camera in self._cameras.items():
            timestamps, _ = camera.stop_recording()
            if timestamps is not None and len(timestamps) > 0:
                camera_ts[name] = timestamps

        if self._count > 0:
            self._save_episode(camera_ts)
            print(f"  [recorder] SAVED {self._count} steps")
        else:
            print("  [recorder] No data recorded, skipping save")

    def step(self, ee_pos, ee_rot, cmd_linear_vel, cmd_angular_vel,
             buttons, enabled, joint_pos=None, joint_vel=None, gripper_open=None,
             robot_abs_time=None):
        """Record one timestep. Fast path: single branch check when not recording."""
        if not self._recording:
            return

        i = self._count
        if i >= self._capacity:
            self._grow_buffers()

        self._buffers["timestamps"][i] = time.monotonic() - self._start_time
        self._buffers["robot_abs_time"][i] = np.nan if robot_abs_time is None else robot_abs_time
        self._buffers["ee_pos"][i] = np.nan if ee_pos is None else ee_pos
        self._buffers["ee_rot"][i] = np.nan if ee_rot is None else ee_rot
        self._buffers["cmd_linear_vel"][i] = cmd_linear_vel
        self._buffers["cmd_angular_vel"][i] = cmd_angular_vel
        self._buffers["buttons"][i] = buttons
        self._buffers["enabled"][i] = enabled
        self._buffers["gripper_open"][i] = np.nan if gripper_open is None else float(gripper_open)

        if joint_pos is not None:
            self._buffers["joint_pos"][i] = joint_pos
        else:
            self._buffers["joint_pos"][i] = np.nan

        if joint_vel is not None:
            self._buffers["joint_vel"][i] = joint_vel
        else:
            self._buffers["joint_vel"][i] = np.nan

        self._count += 1

    def _save_episode(self, camera_ts=None):
        self._save_dir.mkdir(parents=True, exist_ok=True)
        n = self._count
        fname = self._save_dir / f"episode_{self._start_wall}.h5"
        video_ext = "svo2" if self._record_svo else "hdf5"

        with h5py.File(fname, "w") as f:
            for key, buf in self._buffers.items():
                f.create_dataset(key, data=buf[:n], compression="gzip",
                                 compression_opts=1)

            # Write per-camera metadata for all attached cameras (not just those
            # that returned timestamps — SVO recordings embed timestamps internally).
            cameras_to_record = set(self._cameras.keys())
            if isinstance(camera_ts, dict):
                cameras_to_record |= set(camera_ts.keys())

            if cameras_to_record:
                if isinstance(camera_ts, dict) and camera_ts:
                    group = f.create_group("camera_timestamps")
                    for name, timestamps in camera_ts.items():
                        safe = _safe_name(name)
                        group.create_dataset(
                            safe,
                            data=timestamps,
                            compression="gzip",
                            compression_opts=1,
                        )

                for name in cameras_to_record:
                    camera = self._cameras.get(name)
                    if camera is None:
                        continue
                    safe = _safe_name(name)
                    f.attrs[f"{safe}_camera_video_file"] = (
                        f"episode_{self._start_wall}_{safe}_video.{video_ext}"
                    )
                    f.attrs[f"{safe}_camera_id"] = getattr(camera, "camera_id", name)
                    serial_number = getattr(camera, "serial_number", None)
                    f.attrs[f"{safe}_camera_serial_number"] = "" if serial_number is None else str(serial_number)
                    f.attrs[f"{safe}_camera_resolution"] = camera.resolution
                    f.attrs[f"{safe}_camera_fps"] = camera.fps

            for k, v in self._metadata.items():
                f.attrs[k] = v
            f.attrs["start_time"] = self._start_wall
            f.attrs["num_steps"] = n
            f.attrs["duration_s"] = float(self._buffers["timestamps"][n - 1])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self):
        if self._recording:
            self.stop()
