"""Visualize a recorded episode from a ZED camera view.

Accepts either:
  - An episode trajectory file (episode_TIMESTAMP.mcap) — looks up the
    associated camera video file (episode_TIMESTAMP_CAMERA_video.{mp4,svo2}),
    or falls back to a trajectory-only view if the episode has no camera data.
  - An .mp4 or .svo2 camera sidecar file directly — plays it standalone.

.svo2 (native ZED) playback needs the ZED SDK (`pyzed`, this project's
optional `camera` extra: `uv sync --extra camera`).

Usage:
    python visualize_episode.py                                     # latest episode, third_person camera
    python visualize_episode.py data/episode_*.mcap                  # episode file, choose camera with --camera
    python visualize_episode.py data/episode_*_hand_video.svo2       # camera sidecar file directly
    python visualize_episode.py episode.mcap --speed 0.5 --camera hand

Controls:
    SPACE        pause / resume
    LEFT / RIGHT step one frame (while paused)
    q            quit
"""

import argparse
import re
import time
from pathlib import Path

import cv2
import numpy as np

from clear_franka.episode_io import find_latest_episode, load_episode

_VIDEO_EXTENSIONS = (".mp4", ".svo2")


# ── helpers ──────────────────────────────────────────────────────────────────


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "camera"


def find_video_file(episode_path: Path, camera_name: str, attrs: dict) -> Path | None:
    """Locate the camera sidecar for `episode_path`, or None if it has no camera data."""
    safe = _safe_name(camera_name)

    filename = attrs.get(f"camera.{safe}.video_file")
    if filename:
        candidate = episode_path.parent / filename
        if candidate.exists():
            return candidate

    stem = episode_path.stem
    for ext in ("mp4", "svo2"):
        candidate = episode_path.parent / f"{stem}_{safe}_video.{ext}"
        if candidate.exists():
            return candidate

    return None


def nearest_traj_idx(frame_ts: np.ndarray, traj_ts: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(traj_ts, frame_ts, side="left")
    return np.clip(idx, 0, len(traj_ts) - 1)


class _VideoFrames:
    """Sequential-read-friendly frame accessor over an mp4, indexable by frame number."""

    def __init__(self, path: Path):
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {path}")
        self.n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self._next_idx = 0

    def __getitem__(self, frame_idx: int) -> np.ndarray:
        # Sequential reads (the common case: playback and single-step forward)
        # avoid a seek; anything else (rewind, scrub) pays for one.
        if frame_idx != self._next_idx:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, bgr = self._cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame {frame_idx}/{self.n_frames}")
        self._next_idx = frame_idx + 1
        return bgr

    def close(self):
        self._cap.release()


class _SvoFrames:
    """Sequential-read-friendly frame accessor over a ZED .svo2 file, indexable by frame number.

    Needs the ZED SDK (`pyzed`, the optional `camera` extra) — imported lazily so
    the rest of this script works without it.
    """

    def __init__(self, path: Path):
        import pyzed.sl as sl

        self._sl = sl
        self._zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.set_from_svo_file(str(path))
        init_params.svo_real_time_mode = False
        status = self._zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open SVO file {path}: {status}")

        self._runtime = sl.RuntimeParameters()
        self._mat = sl.Mat()
        self.n_frames = self._zed.get_svo_number_of_frames()
        cam_config = self._zed.get_camera_information().camera_configuration
        self.width = cam_config.resolution.width
        self.height = cam_config.resolution.height
        self.fps = float(cam_config.fps) or 30.0
        self._next_idx = 0

    def __getitem__(self, frame_idx: int) -> np.ndarray:
        sl = self._sl
        # Sequential reads (the common case: playback and single-step forward)
        # avoid a seek; anything else (rewind, scrub) pays for one.
        if frame_idx != self._next_idx:
            self._zed.set_svo_position(frame_idx)
        err = self._zed.grab(self._runtime)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to read SVO frame {frame_idx}/{self.n_frames}: {err}")
        self._zed.retrieve_image(self._mat, sl.VIEW.LEFT)
        # BGRA -> BGR (drop alpha; no channel reorder needed, cv2 already wants BGR).
        bgr = self._mat.get_data()[:, :, :3].copy()
        self._next_idx = frame_idx + 1
        return bgr

    def close(self):
        self._zed.close()


def _open_frames(video_path: Path):
    if video_path.suffix == ".svo2":
        return _SvoFrames(video_path)
    return _VideoFrames(video_path)


# ── overlay ───────────────────────────────────────────────────────────────────


def draw_overlay(
    bgr: np.ndarray,
    frame_idx: int,
    n_frames: int,
    elapsed_s: float,
    duration_s: float,
    gripper_open=None,
    paused: bool = False,
    caption: str = "",
) -> np.ndarray:
    img = bgr.copy()
    h, w = img.shape[:2]
    bar_h = 38

    # Dark bar
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

    # Progress bar
    frac = frame_idx / max(n_frames - 1, 1)
    cv2.rectangle(img, (0, h - 3), (int(w * frac), h), (80, 200, 80), -1)

    # Info text
    gripper_str = ""
    if gripper_open is not None and np.isfinite(float(gripper_open)):
        state = "open" if round(float(gripper_open)) else "closed"
        gripper_str = f"  gripper={state}"
    pause_str = "  [PAUSED]" if paused else ""
    text = (
        f"frame {frame_idx + 1}/{n_frames}"
        f"  t={elapsed_s:.2f}s/{duration_s:.2f}s"
        f"{gripper_str}{pause_str}"
    )
    if caption:
        text = f"{caption}  {text}"
    cv2.putText(
        img, text, (8, h - bar_h + 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1, cv2.LINE_AA,
    )
    return img


# ── main loop ─────────────────────────────────────────────────────────────────


def run(episode_path: Path, camera_name: str, speed: float):
    print(f"File:     {episode_path}")

    if episode_path.suffix in _VIDEO_EXTENSIONS:
        # Passed a camera sidecar file directly — no trajectory data available.
        video_path = episode_path
        traj_ts = None
        gripper_data = None
        duration_s = None
    else:
        episode = load_episode(episode_path)
        traj_ts = episode["timestamps"]
        gripper_data = episode.get("gripper_open")
        attrs = episode["attrs"]
        duration_s = float(traj_ts[-1]) if len(traj_ts) else 0.0

        video_path = find_video_file(episode_path, camera_name, attrs)
        if video_path is None:
            print(f"Camera:   {camera_name}  |  no camera data recorded for this episode")
            _run_trajectory_only(traj_ts, gripper_data, duration_s, speed)
            return
        print(f"Video:    {video_path}")

    frames = _open_frames(video_path)
    n_frames, fps = frames.n_frames, frames.fps
    frame_ts = (
        np.linspace(0.0, duration_s, n_frames) if duration_s is not None
        else np.arange(n_frames, dtype=np.float64) / fps
    )
    if duration_s is None:
        duration_s = float(frame_ts[-1]) if n_frames > 0 else 0.0

    traj_idx = nearest_traj_idx(frame_ts, traj_ts) if traj_ts is not None else None

    print(
        f"Camera:   {camera_name}  |  {n_frames} frames @ {fps:.0f} fps"
        f"  |  {frames.width}x{frames.height}"
    )
    print(f"Duration: {duration_s:.2f}s  |  Playback speed: {speed}x")
    print("Controls: SPACE=pause  LEFT/RIGHT=step  q=quit")

    cv2.namedWindow("Episode Viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Episode Viewer", frames.width, frames.height)

    frame_idx = 0
    paused = False
    frame_period = 1.0 / (fps * speed)
    last_advance = time.monotonic()

    try:
        while True:
            grip = None
            if gripper_data is not None and traj_idx is not None:
                tidx = int(traj_idx[min(frame_idx, n_frames - 1)])
                if tidx < len(gripper_data):
                    grip = gripper_data[tidx]

            bgr = frames[frame_idx]
            display = draw_overlay(
                bgr, frame_idx, n_frames,
                float(frame_ts[frame_idx]), duration_s, grip, paused,
            )
            cv2.imshow("Episode Viewer", display)

            wait_ms = 50 if paused else max(1, int(frame_period * 1000 * 0.5))
            key = cv2.waitKey(wait_ms) & 0xFF

            if key == ord("q") or cv2.getWindowProperty("Episode Viewer", cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key == ord(" "):
                paused = not paused
                last_advance = time.monotonic()
            elif key in (81, 2):  # left arrow
                frame_idx = max(0, frame_idx - 1)
                paused = True
            elif key in (83, 3):  # right arrow
                frame_idx = min(n_frames - 1, frame_idx + 1)
                paused = True

            if not paused:
                now = time.monotonic()
                if now - last_advance >= frame_period:
                    frame_idx += 1
                    last_advance = now
                    if frame_idx >= n_frames:
                        frame_idx = n_frames - 1
                        paused = True
    finally:
        frames.close()
        cv2.destroyAllWindows()


def _run_trajectory_only(traj_ts: np.ndarray, gripper_data, duration_s: float, speed: float):
    """Scrub through the recorded trajectory with the same controls, no camera image."""
    n_frames = len(traj_ts)
    fps = max(n_frames / max(duration_s, 1e-6), 1.0)

    print(f"Duration: {duration_s:.2f}s  |  {n_frames} steps  |  Playback speed: {speed}x")
    print("Controls: SPACE=pause  LEFT/RIGHT=step  q=quit")

    cv2.namedWindow("Episode Viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Episode Viewer", 640, 100)

    frame_idx = 0
    paused = False
    frame_period = 1.0 / (fps * speed)
    last_advance = time.monotonic()

    while True:
        grip = gripper_data[frame_idx] if gripper_data is not None else None
        canvas = np.zeros((100, 640, 3), dtype=np.uint8)
        display = draw_overlay(
            canvas, frame_idx, n_frames,
            float(traj_ts[frame_idx]), duration_s, grip, paused,
            caption="[no camera]",
        )
        cv2.imshow("Episode Viewer", display)

        wait_ms = 50 if paused else max(1, int(frame_period * 1000 * 0.5))
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord("q") or cv2.getWindowProperty("Episode Viewer", cv2.WND_PROP_VISIBLE) < 1:
            break
        elif key == ord(" "):
            paused = not paused
            last_advance = time.monotonic()
        elif key in (81, 2):  # left arrow
            frame_idx = max(0, frame_idx - 1)
            paused = True
        elif key in (83, 3):  # right arrow
            frame_idx = min(n_frames - 1, frame_idx + 1)
            paused = True

        if not paused:
            now = time.monotonic()
            if now - last_advance >= frame_period:
                frame_idx += 1
                last_advance = now
                if frame_idx >= n_frames:
                    frame_idx = n_frames - 1
                    paused = True

    cv2.destroyAllWindows()


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a recorded episode from a ZED camera view.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "episode", nargs="?",
        help="Path to episode .mcap file (default: latest in --data-dir)",
    )
    parser.add_argument(
        "--camera", default="third_person",
        help="Camera name to visualize (default: third_person)",
    )
    parser.add_argument(
        "--data-dir", default="./data",
        help="Directory to search for latest episode (default: ./data)",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Playback speed multiplier (default: 1.0)",
    )
    args = parser.parse_args()

    episode_path = Path(args.episode) if args.episode else find_latest_episode(args.data_dir)
    if not episode_path.exists():
        raise FileNotFoundError(f"Episode file not found: {episode_path}")

    run(episode_path, args.camera, args.speed)


if __name__ == "__main__":
    main()
