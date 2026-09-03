"""Capture-loop tests for the live RGB stream, driven by a fake ZED SDK.

The ZED SDK isn't importable without hardware, so these exercise
`ZedCamera._capture_loop` directly against fakes. The behaviour that matters and
is easy to regress: the left view is retrieved at most once per grab and shared
between the recorder and the stream, and each consumer gets its own channel order.
"""

import threading
from types import SimpleNamespace

import numpy as np

from clear_franka.camera import ZedCamera


# Distinct per-channel values so channel order is unambiguous in assertions.
B, G, R = 10, 20, 30


class FakeMat:
    def __init__(self, height=4, width=6):
        bgra = np.zeros((height, width, 4), dtype=np.uint8)
        bgra[..., 0] = B
        bgra[..., 1] = G
        bgra[..., 2] = R
        bgra[..., 3] = 255
        self._bgra = bgra

    def get_data(self):
        return self._bgra


DEPTH_VALUE = 1.25


class FakeDepthMat:
    def __init__(self, height=4, width=6):
        self._depth = np.full((height, width), DEPTH_VALUE, dtype=np.float32)

    def get_data(self):
        return self._depth


class FakeZed:
    """Runs the loop for `frames` grabs, then trips the stop event."""

    def __init__(self, stop_event, frames):
        self._stop_event = stop_event
        self._remaining = frames
        self.retrieve_image_calls = 0
        self.retrieve_measure_calls = 0
        self.retrieve_depth_calls = 0

    def grab(self, runtime):
        self._remaining -= 1
        if self._remaining <= 0:
            self._stop_event.set()
        return FakeSl.ERROR_CODE.SUCCESS

    def retrieve_image(self, mat, view):
        assert view is FakeSl.VIEW.LEFT
        self.retrieve_image_calls += 1

    def retrieve_measure(self, mat, measure):
        self.retrieve_measure_calls += 1
        if measure is FakeSl.MEASURE.DEPTH:
            self.retrieve_depth_calls += 1

    def get_timestamp(self, ref):
        return SimpleNamespace(get_nanoseconds=lambda: 1_000)


class FakeSl:
    ERROR_CODE = SimpleNamespace(SUCCESS=object())
    VIEW = SimpleNamespace(LEFT=object())
    MEASURE = SimpleNamespace(XYZRGBA=object(), DEPTH=object())
    TIME_REFERENCE = SimpleNamespace(IMAGE=object())

    @staticmethod
    def RuntimeParameters():
        return object()


class FakeWriter:
    def __init__(self):
        self.frames = []

    def write(self, frame):
        self.frames.append(frame.copy())


def make_camera(frames=3):
    """Build a ZedCamera with only the fields `_capture_loop` touches."""
    cam = ZedCamera.__new__(ZedCamera)
    cam._stop_event = threading.Event()
    cam._sl = FakeSl
    cam._zed = FakeZed(cam._stop_event, frames)
    cam._rgb_mat = FakeMat()
    cam._depth_mat = FakeDepthMat()
    cam._pointcloud_mat = FakeMat()

    cam._rec_lock = threading.Lock()
    cam._recording = False
    cam._record_format = "svo"
    cam._video_writer = None
    cam._frame_count = 0
    cam._first_clock_anchor = None
    cam._last_clock_anchor = None

    cam._pc_lock = threading.Lock()
    cam._pointcloud_enabled = False
    cam._pointcloud_period = 0.2
    cam._last_pointcloud_time = 0.0

    cam._rgb_lock = threading.Lock()
    cam._rgb_stream_enabled = False
    cam._rgb_period = 0.0
    cam._last_rgb_time = 0.0
    cam._latest_rgb = None

    cam._frame_lock = threading.Lock()
    cam._stream_latest = False
    cam._latest_frame = None
    return cam


def test_idle_loop_retrieves_nothing():
    cam = make_camera(frames=3)
    cam._capture_loop()
    assert cam._zed.retrieve_image_calls == 0
    assert cam._zed.retrieve_depth_calls == 0
    assert cam.get_latest_rgb() is None
    assert cam.get_latest_frame() is None


def test_stream_yields_rgb_frames():
    cam = make_camera(frames=3)
    cam.start_rgb_stream()
    cam._capture_loop()

    latest = cam.get_latest_rgb()
    assert latest is not None
    rgb, timestamp = latest
    assert rgb.shape == (4, 6, 3)
    assert rgb.dtype == np.uint8
    # Stored RGB, reversed from the SDK's BGRA.
    assert tuple(rgb[0, 0]) == (R, G, B)
    assert timestamp > 0


def test_stream_shares_one_retrieval_with_rgb_recording():
    cam = make_camera(frames=3)
    writer = FakeWriter()
    cam._recording = True
    cam._record_format = "rgb"
    cam._video_writer = writer
    cam.start_rgb_stream()
    cam._capture_loop()

    # The key regression: one retrieve per grab, not one per consumer.
    assert cam._zed.retrieve_image_calls == 3
    assert cam._frame_count == 3
    assert len(writer.frames) == 3
    # ffmpeg is fed bgr24 while the stream holds RGB.
    assert tuple(writer.frames[0][0, 0]) == (B, G, R)
    assert tuple(cam.get_latest_rgb()[0][0, 0]) == (R, G, B)


def test_svo_recording_retrieves_only_for_the_stream():
    cam = make_camera(frames=3)
    cam._recording = True
    cam._record_format = "svo"
    cam._capture_loop()
    # SVO frames are written by the SDK on grab(); nothing to retrieve.
    assert cam._zed.retrieve_image_calls == 0
    assert cam._frame_count == 3

    cam = make_camera(frames=3)
    cam._recording = True
    cam._record_format = "svo"
    cam.start_rgb_stream()
    cam._capture_loop()
    assert cam._zed.retrieve_image_calls == 3
    assert cam.get_latest_rgb() is not None


def test_update_hz_rate_limits_capture():
    cam = make_camera(frames=4)
    cam.start_rgb_stream(update_hz=0.001)  # ~1000 s period: only the first frame
    cam._capture_loop()
    assert cam._zed.retrieve_image_calls == 1


def test_stop_rgb_stream_clears_latest():
    cam = make_camera(frames=2)
    cam.start_rgb_stream()
    cam._capture_loop()
    assert cam.get_latest_rgb() is not None

    cam.stop_rgb_stream()
    assert cam.get_latest_rgb() is None


# --- frame stream (rgb + depth) -------------------------------------------
# enable_frame_stream()/get_latest_frame() is what the diffuser-actor deploy
# worker reads instead of grab_frame(). A merge once left the accessors in place
# while dropping the capture-loop publish, so get_latest_frame() always returned
# None and the worker never saw a frame; these pin the publish down.

def test_frame_stream_publishes_rgb_and_depth():
    cam = make_camera(frames=3)
    assert cam.get_latest_frame() is None
    cam.enable_frame_stream()
    cam._capture_loop()

    latest = cam.get_latest_frame()
    assert latest is not None, "capture loop never published a frame"
    rgb, depth = latest
    # RGB order, matching grab_frame's convention.
    assert tuple(rgb[0, 0]) == (R, G, B)
    assert depth.shape == rgb.shape[:2]
    assert np.allclose(depth, DEPTH_VALUE)


def test_frame_stream_runs_without_recording_or_rgb_stream():
    """The stream alone must keep the loop retrieving — the early-continue has
    to account for it, not just recording/pointcloud/rgb-stream."""
    cam = make_camera(frames=3)
    cam.enable_frame_stream()
    cam._capture_loop()
    assert cam._zed.retrieve_image_calls > 0
    assert cam._zed.retrieve_depth_calls > 0


def test_frame_stream_shares_one_retrieval_with_rgb_stream():
    cam = make_camera(frames=4)
    cam.enable_frame_stream()
    cam.start_rgb_stream()
    cam._capture_loop()
    grabs = cam._zed.retrieve_depth_calls
    # One left-view retrieval per grab, shared by both streams.
    assert cam._zed.retrieve_image_calls == grabs
    assert cam.get_latest_rgb() is not None
    assert cam.get_latest_frame() is not None


def test_svo_recording_with_frame_stream_retrieves_depth_once_per_grab():
    cam = make_camera(frames=4)
    cam._recording = True
    cam._record_format = "svo"
    cam.enable_frame_stream()
    cam._capture_loop()
    assert cam._zed.retrieve_image_calls == cam._frame_count
    assert cam._zed.retrieve_depth_calls == cam._frame_count


def test_rgb_recording_alone_does_not_retrieve_depth():
    """Depth is only for the frame stream; retrieve_measure is not free."""
    cam = make_camera(frames=4)
    cam._recording = True
    cam._record_format = "rgb"
    cam._video_writer = FakeWriter()
    cam._capture_loop()
    assert cam._zed.retrieve_image_calls > 0
    assert cam._zed.retrieve_depth_calls == 0
    assert cam.get_latest_frame() is None


def test_disabling_frame_stream_stops_publishing():
    cam = make_camera(frames=3)
    cam.enable_frame_stream()
    cam._capture_loop()
    assert cam.get_latest_frame() is not None

    cam2 = make_camera(frames=3)
    cam2.enable_frame_stream()
    cam2.enable_frame_stream(False)
    cam2._capture_loop()
    assert cam2.get_latest_frame() is None
    assert cam2._zed.retrieve_image_calls == 0
