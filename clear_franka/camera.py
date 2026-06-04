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


def get_camera_config(cfg, name: str = "third_person") -> dict:
    named = cfg.get("cameras", {}).get(name, {})
    return {
        "name": named.get("name", name),
        "enabled": bool(named.get("enabled", False)),
        "resolution": named.get("resolution", "HD720"),
        "fps": int(named.get("fps", 30)),
        "depth_mode": named.get("depth_mode", "PERFORMANCE"),
        "serial_number": named.get("serial_number"),
    }


def make_zed_camera(cfg, name: str = "third_person"):
    camera_cfg = get_camera_config(cfg, name)
    return ZedCamera(
        resolution=camera_cfg["resolution"],
        fps=camera_cfg["fps"],
        depth_mode=camera_cfg["depth_mode"],
        serial_number=camera_cfg["serial_number"],
        camera_id=camera_cfg["name"],
    )


def enabled_camera_names(cfg, include: tuple[str, ...] = ()) -> list[str]:
    names = set(include) | set(cfg.get("cameras", {}).keys())
    return [
        name for name in ("third_person", "hand")
        if name in names and (name in include or get_camera_config(cfg, name)["enabled"])
    ] + [
        name for name in sorted(names)
        if name not in {"third_person", "hand"}
        and (name in include or get_camera_config(cfg, name)["enabled"])
    ]


class ZedCamera:
    """Threaded ZED camera capture with HDF5 video recording."""

    def __init__(self, resolution="HD720", fps=30, depth_mode="PERFORMANCE",
                 serial_number=None, camera_id=None):
        """
        Args:
            resolution: ZED resolution string (HD2K, HD1080, HD720, VGA).
            fps: Target framerate.
            depth_mode: NONE, PERFORMANCE, QUALITY, ULTRA, or NEURAL.
            serial_number: Optional ZED serial number for multi-camera rigs.
            camera_id: Logical name stored in metadata/logs.
        """
        import pyzed.sl as sl

        self._sl = sl
        self._zed = sl.Camera()

        init_params = sl.InitParameters()
        init_params.camera_resolution = getattr(sl.RESOLUTION, resolution)
        init_params.camera_fps = fps
        init_params.depth_mode = getattr(sl.DEPTH_MODE, depth_mode)
        init_params.coordinate_units = sl.UNIT.METER
        if serial_number is not None:
            init_params.set_from_serial_number(int(serial_number))

        status = self._zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            suffix = f" serial={serial_number}" if serial_number is not None else ""
            raise RuntimeError(f"Failed to open ZED camera{suffix}: {status}")

        self._camera_id = camera_id or (
            f"zed_{serial_number}" if serial_number is not None else "zed"
        )
        self._serial_number = serial_number
        self._resolution_str = resolution
        self._fps = fps
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Recording state (protected by _rec_lock)
        self._rec_lock = threading.Lock()
        self._recording = False
        self._svo_recording = False
        self._video_file: Optional[h5py.File] = None
        self._start_time = 0.0
        self._frame_count = 0

        # Pre-allocate retrieval buffers
        self._rgb_mat = sl.Mat()
        self._depth_mat = sl.Mat()
        self._pointcloud_mat = sl.Mat()

        # Latest full RGB+depth frame from the background loop (protected by
        # _frame_lock). Off by default; enable with enable_frame_stream() so the
        # loop also publishes the newest frame for get_latest_frame() consumers
        # (e.g. the diffuser-actor deploy worker, which must NOT call grab_frame()
        # while the background loop is running — concurrent grab() is unsafe).
        self._frame_lock = threading.Lock()
        self._stream_latest = False
        self._latest_frame = None  # (rgb, depth, monotonic_ts) or None

        # Latest point cloud state (protected by _pc_lock)
        self._pc_lock = threading.Lock()
        self._pointcloud_enabled = False
        self._pointcloud_period = 0.2
        self._pointcloud_stride = 4
        self._pointcloud_max_points = 100_000
        self._pointcloud_max_distance_m = 3.0
        self._last_pointcloud_time = 0.0
        self._latest_pointcloud = None

        # Resolve image dimensions from camera config
        cam_info = self._zed.get_camera_information()
        self._img_h = cam_info.camera_configuration.resolution.height
        self._img_w = cam_info.camera_configuration.resolution.width

        logger.info(
            "ZED 2i opened: id=%s serial=%s %s %dx%d @ %dfps, depth=%s",
            self._camera_id, serial_number, resolution, self._img_w, self._img_h, fps, depth_mode,
        )

    # ------------------------------------------------------------------
    # Recording control
    # ------------------------------------------------------------------

    def start_recording(self, video_path: str, start_time: float,
                        svo: bool = False, svo_compression: str = "H264"):
        """Begin recording frames to disk.

        Args:
            video_path: Output path. HDF5 (.hdf5) or SVO2 (.svo2) depending on ``svo``.
            start_time: The time.monotonic() epoch shared with the trajectory
                        recorder, for synchronized timestamps.
            svo: If True, record a native ZED SVO2 file instead of HDF5.
            svo_compression: SVO compression mode name (H264, H265, LOSSLESS,
                             H264_LOSSLESS, H265_LOSSLESS). Ignored when svo=False.
        """
        with self._rec_lock:
            if self._recording:
                return

            if svo:
                sl = self._sl
                recording_params = sl.RecordingParameters()
                recording_params.video_filename = video_path
                recording_params.compression_mode = getattr(
                    sl.SVO_COMPRESSION_MODE, svo_compression
                )
                err = self._zed.enable_recording(recording_params)
                if err != sl.ERROR_CODE.SUCCESS:
                    raise RuntimeError(f"Failed to start SVO recording: {err}")
                self._svo_recording = True
                self._video_file = None
            else:
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
                self._svo_recording = False
                self._video_file = f

            self._start_time = start_time
            self._frame_count = 0
            self._recording = True
            logger.info("  [camera] Recording started: %s", video_path)

    def stop_recording(self):
        """Stop recording and close the video file.

        Returns:
            (camera_timestamps, frame_count) or (None, 0) if not recording.
            camera_timestamps is None for SVO recordings (timestamps are
            embedded in the SVO file).
        """
        with self._rec_lock:
            if not self._recording:
                return None, 0
            self._recording = False
            svo = self._svo_recording
            self._svo_recording = False

            f = self._video_file
            self._video_file = None

        n = self._frame_count
        if svo:
            self._zed.disable_recording()
            logger.info("  [camera] SVO recording stopped, %d frames", n)
            return None, n

        # Read back timestamps (file access outside lock is fine since
        # the capture loop won't touch it after _recording is False).
        if n > 0:
            timestamps = f["timestamps"][:n]
        else:
            timestamps = None
        f.close()
        logger.info("  [camera] Recording stopped, %d frames", n)
        return timestamps, n

    # ------------------------------------------------------------------
    # Synchronous access (used by calibration; safe before run() is called)
    # ------------------------------------------------------------------

    def get_intrinsics(self):
        """Return (K, dist) for the LEFT camera using factory ZED calibration.

        dist follows OpenCV convention: [k1, k2, p1, p2, k3].
        """
        cal = self._zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
        K = np.array(
            [[cal.fx, 0.0, cal.cx],
             [0.0, cal.fy, cal.cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist = np.array(cal.disto[:5], dtype=np.float64)
        return K, dist

    def enable_frame_stream(self, enabled: bool = True):
        """Have the background loop publish the newest (rgb, depth) for
        get_latest_frame(). Lets a consumer read frames WITHOUT calling
        grab_frame() (which would be a second, unsafe concurrent grab()).
        Call before/after run(); the loop picks it up on the next iteration.
        """
        self._stream_latest = bool(enabled)

    def get_latest_frame(self):
        """Return the newest (rgb, depth) published by the background loop, or
        None if streaming isn't enabled yet / no frame captured. Copies out
        under the lock so the caller owns the arrays.
        """
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            rgb, depth, _ts = self._latest_frame
            return rgb, depth

    def grab_frame(self):
        """Synchronously grab one (rgb, depth) frame. Returns None on failure.

        Does not require run() to have been called. Must not be invoked
        concurrently with the background capture thread — intended for
        offline/calibration use where run() is never started.
        """
        sl = self._sl
        runtime = sl.RuntimeParameters()
        if self._zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            return None
        self._zed.retrieve_image(self._rgb_mat, sl.VIEW.LEFT)
        self._zed.retrieve_measure(self._depth_mat, sl.MEASURE.DEPTH)
        rgb = self._rgb_mat.get_data()[:, :, :3][:, :, ::-1].copy()
        depth = self._depth_mat.get_data().copy()
        return rgb, depth

    def start_pointcloud_stream(
        self,
        update_hz: float = 5.0,
        stride: int = 4,
        max_points: int = 100_000,
        max_distance_m: float = 3.0,
    ) -> None:
        """Enable background capture of decimated colored point clouds."""
        with self._pc_lock:
            self._pointcloud_enabled = True
            self._pointcloud_period = 0.0 if update_hz <= 0 else 1.0 / update_hz
            self._pointcloud_stride = max(1, int(stride))
            self._pointcloud_max_points = max(1, int(max_points))
            self._pointcloud_max_distance_m = float(max_distance_m)

    def stop_pointcloud_stream(self) -> None:
        with self._pc_lock:
            self._pointcloud_enabled = False
            self._latest_pointcloud = None

    def get_latest_pointcloud(self):
        """Return the latest (points, colors, timestamp) tuple, or None."""
        with self._pc_lock:
            if self._latest_pointcloud is None:
                return None
            points, colors, timestamp = self._latest_pointcloud
            return points.copy(), colors.copy(), timestamp

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

        consecutive_failures = 0
        last_fail_log = 0.0
        while not self._stop_event.is_set():
            err = self._zed.grab(runtime)
            if err != sl.ERROR_CODE.SUCCESS:
                consecutive_failures += 1
                now_f = time.monotonic()
                # Throttle to one line / 2s so a transient drop doesn't spam,
                # but surface the actual ERROR_CODE (CORRUPTED_FRAME vs
                # CAMERA_NOT_DETECTED vs NO_NEW_FRAMES) for diagnosis.
                if now_f - last_fail_log >= 2.0:
                    print(
                        f"  [camera:{self._camera_id}] grab failed: {err} "
                        f"({consecutive_failures} consecutive)"
                    )
                    last_fail_log = now_f
                # Back off once failures persist so we don't busy-spin grab().
                if consecutive_failures > 3:
                    time.sleep(0.05)
                continue
            consecutive_failures = 0

            now = time.monotonic()
            should_capture_pointcloud = self._should_capture_pointcloud(now)

            with self._rec_lock:
                recording = self._recording
                svo = self._svo_recording
            stream = self._stream_latest
            if not recording and not should_capture_pointcloud and not stream:
                continue

            # Retrieve RGB+depth once if anyone needs it (HDF5 recording OR the
            # latest-frame stream). SVO recording is written by the SDK on grab()
            # and needs no retrieve unless we're also streaming.
            rgb = depth = None
            if (recording and not svo) or stream:
                self._zed.retrieve_image(self._rgb_mat, sl.VIEW.LEFT)
                self._zed.retrieve_measure(self._depth_mat, sl.MEASURE.DEPTH)
                rgb = self._rgb_mat.get_data()[:, :, :3][:, :, ::-1].copy()  # BGRA→RGB
                depth = self._depth_mat.get_data().copy()
                if stream:
                    with self._frame_lock:
                        self._latest_frame = (rgb, depth, now)

            if recording:
                with self._rec_lock:
                    if self._recording:  # re-check; stop_recording may have raced
                        if self._svo_recording:
                            self._frame_count += 1
                        elif rgb is not None:
                            i = self._frame_count
                            f = self._video_file
                            f["rgb"].resize(i + 1, axis=0)
                            f["rgb"][i] = rgb
                            f["depth"].resize(i + 1, axis=0)
                            f["depth"][i] = depth
                            f["timestamps"].resize(i + 1, axis=0)
                            f["timestamps"][i] = now - self._start_time
                            self._frame_count = i + 1

            if should_capture_pointcloud:
                self._retrieve_pointcloud(now)

    def _should_capture_pointcloud(self, now: float) -> bool:
        with self._pc_lock:
            if not self._pointcloud_enabled:
                return False
            if now - self._last_pointcloud_time < self._pointcloud_period:
                return False
            self._last_pointcloud_time = now
            return True

    def _retrieve_pointcloud(self, timestamp: float) -> None:
        sl = self._sl
        self._zed.retrieve_measure(self._pointcloud_mat, sl.MEASURE.XYZRGBA)
        xyzrgba = self._pointcloud_mat.get_data()

        with self._pc_lock:
            stride = self._pointcloud_stride
            max_points = self._pointcloud_max_points
            max_distance_m = self._pointcloud_max_distance_m

        if stride > 1:
            xyzrgba = xyzrgba[::stride, ::stride]

        flat = xyzrgba.reshape(-1, 4)
        points = flat[:, :3]
        colors_packed = flat[:, 3]

        finite = np.isfinite(points).all(axis=1)
        if max_distance_m > 0.0:
            finite &= np.linalg.norm(points, axis=1) <= max_distance_m
        points = points[finite].astype(np.float32, copy=False)
        colors_packed = colors_packed[finite].astype(np.float32, copy=False)

        if points.shape[0] > max_points:
            step = int(np.ceil(points.shape[0] / max_points))
            points = points[::step]
            colors_packed = colors_packed[::step]

        colors = np.ascontiguousarray(colors_packed).view(np.uint8).reshape(-1, 4)[:, :3].copy()
        points = points.copy()

        with self._pc_lock:
            self._latest_pointcloud = (points, colors, timestamp)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

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

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def serial_number(self):
        return self._serial_number
