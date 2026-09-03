"""ZED 2i camera capture and native SVO2 recording for teleoperation."""

import logging
import threading
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class _FfmpegFrameWriter:
    """Pipes raw BGR frames to a system `ffmpeg` subprocess for H.26x encoding.

    cv2.VideoWriter can't do this: the FFmpeg bundled in the opencv-python wheel
    omits libx264/libx265 (excluded from the prebuilt wheel), so h264/h265 fourccs
    silently fail to open. The system ffmpeg (`apt install ffmpeg` on Ubuntu) has
    both, so we drive it directly instead.
    """

    def __init__(self, path: str, width: int, height: int, fps: int,
                 codec: str = "libx265", preset: str = "veryfast", crf: int = 28):
        import shutil
        import subprocess

        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is None:
            raise RuntimeError(
                "ffmpeg not found on PATH; install it for RGB video recording (e.g. `sudo apt install ffmpeg`)"
            )
        cmd = [
            ffmpeg_bin, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-",
            "-an", "-vcodec", codec, "-pix_fmt", "yuv420p",
            "-preset", preset, "-crf", str(crf),
            path,
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame_bgr) -> None:
        self._proc.stdin.write(frame_bgr.tobytes())

    def release(self) -> None:
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        self._proc.wait(timeout=30)


def get_camera_config(cfg, name: str = "third_person") -> dict:
    named = cfg.get("cameras", {}).get(name, {})
    return {
        "name": named.get("name", name),
        "enabled": bool(named.get("enabled", False)),
        "resolution": named.get("resolution", "HD720"),
        "fps": int(named.get("fps", 30)),
        "depth_mode": named.get("depth_mode", "NEURAL"),
        "serial_number": named.get("serial_number"),
    }


def make_zed_camera(cfg, name: str = "third_person", depth_mode: str | None = None):
    """Build a camera from config. `depth_mode` overrides the configured mode.

    Depth costs real time inside `grab()` — NEURAL runs a network per frame per
    camera — so callers that only need color should pass "NONE".
    """
    camera_cfg = get_camera_config(cfg, name)
    return ZedCamera(
        resolution=camera_cfg["resolution"],
        fps=camera_cfg["fps"],
        depth_mode=depth_mode or camera_cfg["depth_mode"],
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
    """Threaded ZED camera capture with native SVO2 recording."""

    def __init__(self, resolution="HD720", fps=30, depth_mode="NEURAL",
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
        self._record_format = "svo"
        self._video_writer = None
        self._frame_count = 0
        self._first_clock_anchor = None
        self._last_clock_anchor = None
        # enable_recording()/disable_recording() are native SDK calls and, like
        # grab(), are not safe to make from a different thread than whichever
        # one is currently mid-grab() — the ZED SDK's Camera object is not
        # thread-safe. start/stop_recording() (called from the recorder thread)
        # hand the actual toggle to _capture_loop via these fields and block on
        # the event instead of calling self._zed directly. Concretely calling
        # disable_recording() from another thread while grab() is in flight can
        # wedge the SDK forever (observed: Ctrl-C didn't even work).
        self._pending_recording_params = None  # sl.RecordingParameters or None
        self._pending_disable_recording = False
        self._recording_toggle_error = None  # sl.ERROR_CODE or None
        self._recording_toggle_done = threading.Event()

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

        self._rgb_lock = threading.Lock()
        self._rgb_stream_enabled = False
        self._rgb_period = 0.0
        self._last_rgb_time = 0.0
        self._latest_rgb = None

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

    def start_recording(self, video_path: str, svo_compression: str = "H264", format: str = "svo"):
        """Begin recording camera video.

        Args:
            video_path: Output path (.svo2 for format="svo"; a video container, e.g. .mp4,
                for format="rgb").
            svo_compression: H264, H265, LOSSLESS, H264_LOSSLESS, or H265_LOSSLESS.
                Only used for format="svo".
            format: "svo" records the native stereo pair (depth reconstructable later,
                larger files). "rgb" records only the left view as an H.265 color video
                (via a piped system ffmpeg) — no depth, smaller files.
        """
        with self._rec_lock:
            if self._recording:
                return

            if format == "svo":
                sl = self._sl
                recording_params = sl.RecordingParameters()
                recording_params.video_filename = video_path
                recording_params.compression_mode = getattr(
                    sl.SVO_COMPRESSION_MODE, svo_compression
                )
                self._recording_toggle_done.clear()
                self._recording_toggle_error = None
                self._pending_recording_params = recording_params
                self._svo_recording = True
            elif format == "rgb":
                self._video_writer = _FfmpegFrameWriter(video_path, self._img_w, self._img_h, self._fps)
            else:
                raise ValueError(f"Unknown camera recording format: {format!r}")

            self._record_format = format
            self._frame_count = 0
            self._first_clock_anchor = None
            self._last_clock_anchor = None
            self._recording = True

        if format == "svo":
            # enable_recording() is actually issued by _capture_loop, on the
            # same thread as grab() — see the comment by _pending_recording_params.
            if not self._recording_toggle_done.wait(timeout=5.0):
                with self._rec_lock:
                    self._recording = False
                    self._svo_recording = False
                raise RuntimeError(
                    "Timed out starting SVO recording (capture loop not running?)"
                )
            if self._recording_toggle_error is not None:
                error = self._recording_toggle_error
                self._recording_toggle_error = None
                with self._rec_lock:
                    self._recording = False
                    self._svo_recording = False
                raise RuntimeError(f"Failed to start SVO recording: {error}")

        logger.info("  [camera] Recording started (%s): %s", format, video_path)

    def stop_recording(self):
        """Stop recording and close the video file.

        Returns:
            Frame count and first/last ZED-image-to-host-monotonic clock anchors.
        """
        with self._rec_lock:
            if not self._recording:
                return {"frame_count": 0}
            self._recording = False
            record_format = self._record_format
            self._svo_recording = False
            if record_format == "svo":
                self._recording_toggle_done.clear()
                self._pending_disable_recording = True

        n = self._frame_count
        if record_format == "svo":
            # disable_recording() is actually issued by _capture_loop — see the
            # comment by _pending_recording_params for why this can't just call
            # self._zed.disable_recording() directly from here. A stalled
            # capture loop leaves the SVO file open rather than hanging forever.
            if not self._recording_toggle_done.wait(timeout=5.0):
                logger.warning(
                    "  [camera] Timed out stopping SVO recording (capture loop stalled?)"
                )
        elif self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
        logger.info("  [camera] Recording stopped (%s), %d frames", record_format, n)
        result = {"frame_count": n}
        for label, anchor in (("first", self._first_clock_anchor), ("last", self._last_clock_anchor)):
            if anchor is not None:
                result[f"{label}_zed_image_time_ns"] = anchor[0]
                result[f"{label}_host_monotonic_time_ns"] = anchor[1]
        return result

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
        None if streaming isn't enabled yet / no frame captured.

        The arrays are handed out without copying: the loop allocates fresh ones
        each iteration and never mutates a published frame, so the caller can
        read them safely (but must not write to them).
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

    def start_rgb_stream(self, update_hz: float = 0.0) -> None:
        """Enable background capture of the latest left-view RGB frame.

        Unlike `grab_frame`, this is safe to use alongside `run()` and recording:
        the capture thread retrieves the left view once per grab and shares it.

        Args:
            update_hz: 0 or less stores every grabbed frame, which is the default
                and what live inference wants. Only set a rate when deliberately
                sampling below the camera fps — rate-limiting at exactly the fps
                lets grab jitter drop alternate frames, halving the effective rate.
        """
        with self._rgb_lock:
            self._rgb_stream_enabled = True
            self._rgb_period = 0.0 if update_hz <= 0 else 1.0 / update_hz

    def stop_rgb_stream(self) -> None:
        with self._rgb_lock:
            self._rgb_stream_enabled = False
            self._latest_rgb = None

    def get_latest_rgb(self):
        """Return the latest (rgb, timestamp) tuple, or None.

        `rgb` is HxWx3 uint8 in RGB order; `timestamp` is `time.monotonic()` at
        capture, so callers can check staleness.
        """
        with self._rgb_lock:
            if self._latest_rgb is None:
                return None
            rgb, timestamp = self._latest_rgb
            return rgb.copy(), timestamp

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
            # Service a pending enable_recording()/disable_recording() request
            # before grab() — never concurrently with it. See the comment by
            # _pending_recording_params in __init__.
            with self._rec_lock:
                if self._pending_recording_params is not None:
                    params = self._pending_recording_params
                    self._pending_recording_params = None
                    err_toggle = self._zed.enable_recording(params)
                    if err_toggle != sl.ERROR_CODE.SUCCESS:
                        self._recording_toggle_error = err_toggle
                    self._recording_toggle_done.set()
                elif self._pending_disable_recording:
                    self._pending_disable_recording = False
                    self._zed.disable_recording()
                    self._recording_toggle_done.set()

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
            should_capture_rgb = self._should_capture_rgb(now)
            stream_frame = self._stream_latest

            with self._rec_lock:
                recording = self._recording
                if (
                    not recording
                    and not should_capture_pointcloud
                    and not should_capture_rgb
                    and not stream_frame
                ):
                    continue

                # Retrieve the left view at most once and fan it out to whoever
                # wants it. get_data() is BGRA: ffmpeg is fed bgr24, while the
                # streams store RGB to match grab_frame's convention.
                rgb_recording = recording and self._record_format == "rgb"
                if rgb_recording or should_capture_rgb or stream_frame:
                    self._zed.retrieve_image(self._rgb_mat, sl.VIEW.LEFT)
                    bgra = self._rgb_mat.get_data()
                    if rgb_recording:
                        self._video_writer.write(np.ascontiguousarray(bgra[:, :, :3]))
                    if should_capture_rgb or stream_frame:
                        rgb = np.ascontiguousarray(bgra[:, :, :3][:, :, ::-1])
                        if should_capture_rgb:
                            with self._rgb_lock:
                                self._latest_rgb = (rgb, now)
                        if stream_frame:
                            # Depth is retrieved only for the frame stream —
                            # rgb recording and the rgb stream don't need it,
                            # and retrieve_measure is not free.
                            self._zed.retrieve_measure(self._depth_mat, sl.MEASURE.DEPTH)
                            depth = self._depth_mat.get_data().copy()
                            with self._frame_lock:
                                self._latest_frame = (rgb, depth, now)

                if recording:
                    # SVO frames are written automatically by the SDK on grab();
                    # the rgb format was already handled above.
                    self._frame_count += 1
                    image_time_ns = self._zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
                    anchor = (int(image_time_ns), time.monotonic_ns())
                    if self._first_clock_anchor is None:
                        self._first_clock_anchor = anchor
                    self._last_clock_anchor = anchor

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

    def _should_capture_rgb(self, now: float) -> bool:
        with self._rgb_lock:
            if not self._rgb_stream_enabled:
                return False
            if now - self._last_rgb_time < self._rgb_period:
                return False
            self._last_rgb_time = now
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
