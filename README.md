# clear_franka

Teleoperation and trajectory recording for Franka Emika robots.

## Installation

```
uv sync
```

This installs the base dependencies (robot control via `zero-franky`/`franky-control`, SpaceMouse input, recording, visualization). It does **not** include camera support.

### Camera support

ZED camera capture (used for the point-cloud viewer and for `record: rgb`/`rgbd` episode recording) needs the `camera` extra:

```
uv sync --extra camera
```

This pulls in:
- `pyzed` — the ZED SDK Python bindings (requires the ZED SDK to be installed on the machine; see [stereolabs.com](https://www.stereolabs.com/developers/release/)).
- `opencv-python` — used by `calibrate_extrinsics.py`. Not needed for `record: rgb`/`rgbd`/`joints` themselves.

If you only use `record: joints` (no cameras at all), you can skip the `camera` extra entirely.

`record: rgb` also needs a **system** `ffmpeg` with `libx265` (for encoding — see below), which is a separate install from the Python deps above:

```
sudo apt install ffmpeg
```

The `opencv-python` wheel from PyPI bundles its own minimal FFmpeg that omits `libx264`/`libx265` (licensing/size reasons), so `record: rgb` shells out to the system `ffmpeg` binary directly (via `subprocess`, piping raw frames over stdin) rather than using `cv2.VideoWriter`.


## Recording modes

`teleop.<record>` (and `demonstrate.<record>`) controls what gets written to each episode:

| Mode | Contents |
| --- | --- |
| `joints` | Joint/robot state only, no cameras. |
| `rgb` | Joint/robot state + a left-view-only H.265 color video per camera (`.mp4`, via a piped system `ffmpeg`). No depth. Smaller files. |
| `rgbd` | Joint/robot state + native ZED stereo recording per camera (`.svo2`). Depth reconstructable at replay time. Larger files. |
