"""Visualize a recorded episode from a ZED camera view.

Accepts either:
  - An episode trajectory file (episode_TIMESTAMP.h5) — looks up the
    associated ZED stream file for the chosen camera.
  - A ZED stream file directly (episode_TIMESTAMP_third_person_video.hdf5 /
    episode_TIMESTAMP_hand_video.hdf5) — plays it standalone.

Usage:
    python visualize_episode.py                                     # latest episode, third_person camera
    python visualize_episode.py data/episode_*.h5                   # episode file, choose camera with --camera
    python visualize_episode.py data/episode_*_hand_video.hdf5      # ZED stream file directly
    python visualize_episode.py episode.h5 --speed 0.5 --camera hand

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
import h5py
import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "camera"


def is_video_file(path: Path) -> bool:
    """Return True if the file is a ZED stream h5 (has an 'rgb' dataset)."""
    try:
        with h5py.File(path, "r") as f:
            return "rgb" in f
    except Exception:
        return False


def find_latest_episode(data_dir: str) -> Path:
    """Return the most recent episode or ZED stream file in data_dir."""
    d = Path(data_dir)
    # Prefer episode trajectory files; fall back to raw video files.
    episodes = sorted(d.glob("episode_*.h5"))
    if episodes:
        return episodes[-1]
    videos = sorted(d.glob("episode_*_video.hdf5"))
    if videos:
        return videos[-1]
    raise FileNotFoundError(f"No episode or video files found in {data_dir}")


def find_video_file(episode_path: Path, camera_name: str) -> Path:
    safe = _safe_name(camera_name)
    with h5py.File(episode_path, "r") as f:
        attr_key = f"{safe}_camera_video_file"
        if attr_key in f.attrs:
            candidate = episode_path.parent / f.attrs[attr_key]
            if candidate.exists():
                return candidate

    stem = episode_path.stem
    for ext in ("hdf5", "h5"):
        candidate = episode_path.parent / f"{stem}_{safe}_video.{ext}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No video file found for camera '{camera_name}'.\n"
        f"Expected: {episode_path.parent}/{stem}_{safe}_video.hdf5\n"
        f"Make sure the episode was recorded with --camera {camera_name} enabled."
    )


def nearest_traj_idx(frame_ts: np.ndarray, traj_ts: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(traj_ts, frame_ts, side="left")
    return np.clip(idx, 0, len(traj_ts) - 1)


# ── overlay ───────────────────────────────────────────────────────────────────


def draw_overlay(
    bgr: np.ndarray,
    frame_idx: int,
    n_frames: int,
    elapsed_s: float,
    duration_s: float,
    gripper_open=None,
    paused: bool = False,
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
    cv2.putText(
        img, text, (8, h - bar_h + 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1, cv2.LINE_AA,
    )
    return img


# ── main loop ─────────────────────────────────────────────────────────────────


def run(episode_path: Path, camera_name: str, speed: float):
    print(f"File:     {episode_path}")

    if is_video_file(episode_path):
        # Passed a ZED stream file directly — no trajectory data available.
        video_path = episode_path
        traj_ts = None
        gripper_data = None
        fps = 30.0
        duration_s = None
    else:
        # Passed an episode trajectory file — load trajectory and find video.
        with h5py.File(episode_path, "r") as ef:
            traj_ts = ef["timestamps"][:] if "timestamps" in ef else None
            gripper_data = ef["gripper_open"][:] if "gripper_open" in ef else None
            attrs = dict(ef.attrs)

        duration_s = float(attrs.get("duration_s", traj_ts[-1] if traj_ts is not None else 0.0))
        fps = float(attrs.get(f"{_safe_name(camera_name)}_camera_fps", 30))

        video_path = find_video_file(episode_path, camera_name)
        print(f"Video:    {video_path}")

    with h5py.File(video_path, "r") as vf:
        n_frames, frame_h, frame_w = vf["rgb"].shape[:3]
        if "timestamps" in vf:
            frame_ts = vf["timestamps"][:]
        elif duration_s is not None:
            frame_ts = np.linspace(0.0, duration_s, n_frames)
        else:
            frame_ts = np.arange(n_frames, dtype=np.float64) / fps

        if duration_s is None:
            duration_s = float(frame_ts[-1]) if n_frames > 0 else 0.0

        traj_idx = nearest_traj_idx(frame_ts, traj_ts) if traj_ts is not None else None

        print(
            f"Camera:   {camera_name}  |  {n_frames} frames @ {fps:.0f} fps"
            f"  |  {frame_w}x{frame_h}"
        )
        print(f"Duration: {duration_s:.2f}s  |  Playback speed: {speed}x")
        print("Controls: SPACE=pause  LEFT/RIGHT=step  q=quit")

        cv2.namedWindow("Episode Viewer", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Episode Viewer", frame_w, frame_h)

        frame_idx = 0
        paused = False
        frame_period = 1.0 / (fps * speed)
        last_advance = time.monotonic()

        while True:
            grip = None
            if gripper_data is not None and traj_idx is not None:
                tidx = int(traj_idx[min(frame_idx, n_frames - 1)])
                if tidx < len(gripper_data):
                    grip = gripper_data[tidx]

            rgb = vf["rgb"][frame_idx]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
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

    cv2.destroyAllWindows()


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a recorded episode from a ZED camera view.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "episode", nargs="?",
        help="Path to episode .h5 file (default: latest in --data-dir)",
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
