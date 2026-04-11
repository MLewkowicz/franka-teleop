"""ZED 2i camera capture and recording for teleoperation.

Runs camera capture in a background daemon thread. Records RGB + depth
frames to a separate HDF5 video file, synchronized with trajectory
recording via a shared monotonic clock epoch.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

logger = logging.getLogger(__name__)


class ZedCamera:
    """Threaded ZED 2i camera capture with HDF5 video recording."""

    def __init__(self, resolution="HD720", fps=30, depth_mode="PERFORMANCE"):
        """
        Args:
            resolution: ZED resolution string (HD2K, HD1080, HD720, VGA).
            fps: Target framerate.
            depth_mode: NONE, PERFORMANCE, QUALITY, ULTRA, or NEURAL.
        """
        import pyzed.sl as sl

        self._sl = sl
        self._zed = sl.Camera()

        init_params = sl.InitParameters()
        init_params.camera_resolution = getattr(sl.RESOLUTION, resolution)
        init_params.camera_fps = fps
        init_params.depth_mode = getattr(sl.DEPTH_MODE, depth_mode)
        init_params.coordinate_units = sl.UNIT.METER

        status = self._zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {status}")

        self._resolution_str = resolution
        self._fps = fps
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Recording state (protected by _rec_lock)
        self._rec_lock = threading.Lock()
        self._recording = False
        self._video_file: Optional[h5py.File] = None
        self._start_time = 0.0
        self._frame_count = 0

        # Pre-allocate retrieval buffers
        self._rgb_mat = sl.Mat()
        self._depth_mat = sl.Mat()

        # Resolve image dimensions from camera config
        cam_info = self._zed.get_camera_information()
        self._img_h = cam_info.camera_configuration.resolution.height
        self._img_w = cam_info.camera_configuration.resolution.width

        logger.info(
            "ZED 2i opened: %s %dx%d @ %dfps, depth=%s",
            resolution, self._img_w, self._img_h, fps, depth_mode,
        )

    # ------------------------------------------------------------------
    # Recording control
    # ------------------------------------------------------------------

    def start_recording(self, video_path: str, start_time: float):
        """Begin writing RGB + depth frames to an HDF5 video file.

        Args:
            video_path: Output path for the video HDF5 file.
            start_time: The time.monotonic() epoch shared with the
                        trajectory recorder, for synchronized timestamps.
        """
        with self._rec_lock:
            if self._recording:
                return

            h, w = self._img_h, self._img_w
            f = h5py.File(video_path, "w")
            f.create_dataset(
                "rgb", shape=(0, h, w, 3), maxshape=(None, h, w, 3),
                dtype=np.uint8, chunks=(1, h, w, 3), compression="lzf",
            )
            f.create_dataset(
                "depth", shape=(0, h, w), maxshape=(None, h, w),
                dtype=np.float32, chunks=(1, h, w), compression="lzf",
            )
            f.create_dataset(
                "timestamps", shape=(0,), maxshape=(None,),
                dtype=np.float64, chunks=(256,),
            )

            self._video_file = f
            self._start_time = start_time
            self._frame_count = 0
            self._recording = True
            logger.info("  [camera] Recording started: %s", video_path)

    def stop_recording(self):
        """Stop recording and close the video file.

        Returns:
            (camera_timestamps, frame_count) or (None, 0) if not recording.
        """
        with self._rec_lock:
            if not self._recording:
                return None, 0
            self._recording = False

            f = self._video_file
            self._video_file = None

        # Read back timestamps (file access outside lock is fine since
        # the capture loop won't touch it after _recording is False).
        n = self._frame_count
        if n > 0:
            timestamps = f["timestamps"][:n]
        else:
            timestamps = None
        f.close()
        logger.info("  [camera] Recording stopped, %d frames", n)
        return timestamps, n

    # ------------------------------------------------------------------
    # Thread lifecycle
    # ------------------------------------------------------------------

    def run(self):
        """Start the background capture thread."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        sl = self._sl
        runtime = sl.RuntimeParameters()

        while not self._stop_event.is_set():
            err = self._zed.grab(runtime)
            if err != sl.ERROR_CODE.SUCCESS:
                continue

            with self._rec_lock:
                if not self._recording:
                    continue

                ts = time.monotonic() - self._start_time

                # Retrieve left RGB (BGRA) and depth
                self._zed.retrieve_image(self._rgb_mat, sl.VIEW.LEFT)
                self._zed.retrieve_measure(self._depth_mat, sl.MEASURE.DEPTH)

                rgb = self._rgb_mat.get_data()[:, :, :3]    # BGRA → BGR
                rgb = rgb[:, :, ::-1].copy()                 # BGR → RGB
                depth = self._depth_mat.get_data().copy()

                # Append to HDF5 datasets
                i = self._frame_count
                f = self._video_file

                f["rgb"].resize(i + 1, axis=0)
                f["rgb"][i] = rgb

                f["depth"].resize(i + 1, axis=0)
                f["depth"][i] = depth

                f["timestamps"].resize(i + 1, axis=0)
                f["timestamps"][i] = ts

                self._frame_count = i + 1

    def close(self):
        """Stop capture thread and release camera."""
        if self._recording:
            self.stop_recording()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._zed.close()
        logger.info("ZED camera closed.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def resolution(self) -> str:
        return self._resolution_str

    @property
    def fps(self) -> int:
        return self._fps
