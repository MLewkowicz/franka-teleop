"""Frame sources: live ZED, or an SVO recording, both in the CAMERA optical frame.

Both sources yield camera-frame XYZ using the ZED's default coordinate system
(x right, y down, z forward), because that is the frame the hand-eye extrinsics
in `data/extrinsics_*.json` were solved in — `T_cam2base @ xyz_cam` then lands
in `fr3_link0`. The offline prototype opened the SVO with
`COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP` for direct viser display; doing that here
would silently rotate every box by the ZED->viser change of basis, so we keep
the default and let `T_cam2base` do the work.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, Protocol

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    rgb: np.ndarray  # (H, W, 3) uint8
    xyz: np.ndarray  # (H, W, 3) float32, metres, CAMERA optical frame


class FrameSource(Protocol):
    def frames(self) -> Iterator[Frame]: ...
    def close(self) -> None: ...


class ZedLiveSource:
    """Wraps an already-open `clear_franka.camera.ZedCamera`.

    Uses `grab_frame()` (synchronous) by default — synthesis runs offline, with
    no background capture loop competing for `grab()`. Pass `use_stream=True`
    when the camera's background loop is already running (e.g. inside deploy),
    in which case frames come from `get_latest_frame()`.
    """

    def __init__(self, camera, use_stream: bool = False) -> None:
        self._cam = camera
        self._use_stream = use_stream
        self._K, _dist = camera.get_intrinsics()

    @property
    def intrinsics(self) -> np.ndarray:
        return self._K

    def frames(self) -> Iterator[Frame]:
        from clear_franka.diffuser_actor_io import depth_to_camera_xyz

        while True:
            got = (
                self._cam.get_latest_frame()
                if self._use_stream
                else self._cam.grab_frame()
            )
            if got is None:
                continue
            rgb, depth = got
            yield Frame(
                rgb=np.ascontiguousarray(rgb),
                xyz=depth_to_camera_xyz(depth, self._K).astype(np.float32),
            )

    def close(self) -> None:  # the caller owns the camera's lifetime
        pass


class SvoSource:
    """Plays an SVO recording once (or looping), yielding RGB + camera XYZ."""

    def __init__(
        self,
        path: str,
        depth_mode: str = "NEURAL",
        loop: bool = False,
    ) -> None:
        import pyzed.sl as sl

        self._sl = sl
        init = sl.InitParameters()
        init.set_from_svo_file(str(path))
        init.coordinate_units = sl.UNIT.METER
        init.depth_mode = getattr(sl.DEPTH_MODE, depth_mode)

        self.cam = sl.Camera()
        status = self.cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open SVO {path}: {status}")
        self._image = sl.Mat()
        self._cloud = sl.Mat()
        self._loop = loop

    def frames(self) -> Iterator[Frame]:
        sl = self._sl
        while True:
            if self.cam.grab() != sl.ERROR_CODE.SUCCESS:
                if not self._loop:
                    return
                self.cam.set_svo_position(0)
                continue

            self.cam.retrieve_image(self._image, sl.VIEW.LEFT)
            self.cam.retrieve_measure(self._cloud, sl.MEASURE.XYZRGBA)
            yield Frame(
                rgb=self._image.get_data()[..., [2, 1, 0]].copy(),  # BGRA -> RGB
                xyz=self._cloud.get_data()[..., :3].astype(np.float32),
            )

    def close(self) -> None:
        self.cam.close()


def collect_frames(source: FrameSource, n: int, skip: int = 0) -> list[Frame]:
    """Take `n` frames from a source, discarding the first `skip`.

    Depth on the first frames after a stream starts is noticeably noisier, and
    averaging boxes over a few frames is the cheapest defence against a single
    bad depth map (see `detections_to_scene_boxes(..., aggregate='median')`).
    """
    out: list[Frame] = []
    for i, frame in enumerate(source.frames()):
        if i < skip:
            continue
        out.append(frame)
        if len(out) >= n:
            break
    if not out:
        raise RuntimeError("frame source produced no frames")
    if len(out) < n:
        logger.warning("requested %d frames, source gave %d", n, len(out))
    return out
