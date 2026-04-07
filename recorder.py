"""Trajectory recorder for Franka teleoperation.

Records timestamped episodes to HDF5 files for future replay.
Designed for minimal overhead in a 1kHz control loop.
"""

import time
from pathlib import Path

import h5py
import numpy as np


class TrajectoryRecorder:

    def __init__(self, save_dir="./data", capacity=120_000, metadata=None):
        """
        Args:
            save_dir: Directory to save episode files.
            capacity: Pre-allocated buffer size in timesteps (120k = 2 min at 1kHz).
            metadata: Optional dict of extra metadata stored in HDF5 attrs.
        """
        self._save_dir = Path(save_dir)
        self._capacity = capacity
        self._metadata = metadata or {}
        self._recording = False
        self._count = 0
        self._start_time = 0.0
        self._start_wall = ""
        self._alloc_buffers()

    def _alloc_buffers(self):
        n = self._capacity
        self._buffers = {
            "timestamps": np.empty(n, dtype=np.float64),
            "ee_pos": np.empty((n, 3), dtype=np.float64),
            "ee_rot": np.empty((n, 3, 3), dtype=np.float64),
            "joint_pos": np.empty((n, 7), dtype=np.float64),
            "joint_vel": np.empty((n, 7), dtype=np.float64),
            "cmd_linear_vel": np.empty((n, 3), dtype=np.float64),
            "cmd_angular_vel": np.empty((n, 3), dtype=np.float64),
            "buttons": np.empty(n, dtype=np.int32),
            "enabled": np.empty(n, dtype=bool),
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
        print("  [recorder] RECORDING started")

    def stop(self):
        if not self._recording:
            return
        self._recording = False
        if self._count > 0:
            self._save_episode()
            print(f"  [recorder] SAVED {self._count} steps")
        else:
            print("  [recorder] No data recorded, skipping save")

    def step(self, ee_pos, ee_rot, cmd_linear_vel, cmd_angular_vel,
             buttons, enabled, joint_pos=None, joint_vel=None):
        """Record one timestep. Fast path: single branch check when not recording."""
        if not self._recording:
            return

        i = self._count
        if i >= self._capacity:
            self._grow_buffers()

        self._buffers["timestamps"][i] = time.monotonic() - self._start_time
        self._buffers["ee_pos"][i] = ee_pos
        self._buffers["ee_rot"][i] = ee_rot
        self._buffers["cmd_linear_vel"][i] = cmd_linear_vel
        self._buffers["cmd_angular_vel"][i] = cmd_angular_vel
        self._buffers["buttons"][i] = buttons
        self._buffers["enabled"][i] = enabled

        if joint_pos is not None:
            self._buffers["joint_pos"][i] = joint_pos
        else:
            self._buffers["joint_pos"][i] = np.nan

        if joint_vel is not None:
            self._buffers["joint_vel"][i] = joint_vel
        else:
            self._buffers["joint_vel"][i] = np.nan

        self._count += 1

    def _save_episode(self):
        self._save_dir.mkdir(parents=True, exist_ok=True)
        n = self._count
        fname = self._save_dir / f"episode_{self._start_wall}.h5"

        with h5py.File(fname, "w") as f:
            for key, buf in self._buffers.items():
                f.create_dataset(key, data=buf[:n], compression="gzip",
                                 compression_opts=1)
            for k, v in self._metadata.items():
                f.attrs[k] = v
            f.attrs["start_time"] = self._start_wall
            f.attrs["num_steps"] = n
            f.attrs["duration_s"] = float(self._buffers["timestamps"][n - 1])

    def close(self):
        if self._recording:
            self.stop()
