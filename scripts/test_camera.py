"""Quick standalone test for ZED 2i camera recording."""

import time
from pathlib import Path

from clear_franka.camera import ZedCamera

data_dir = Path("./data")
data_dir.mkdir(exist_ok=True)

print("Opening ZED 2i...")
cam = ZedCamera(resolution="HD720", fps=30, depth_mode="NEURAL")
cam.run()

video_path = str(data_dir / "test_video.hdf5")
start_time = time.monotonic()

print("Recording 5 seconds...")
cam.start_recording(video_path, start_time)
time.sleep(5)
timestamps, n_frames = cam.stop_recording()

cam.close()

print(f"\nResults:")
print(f"  Frames captured: {n_frames}")
print(f"  Duration: {timestamps[-1]:.2f}s" if timestamps is not None else "  No frames")
print(f"  Effective FPS: {n_frames / timestamps[-1]:.1f}" if n_frames > 0 else "")

# Verify the HDF5 file
import h5py
with h5py.File(video_path, "r") as f:
    print(f"\nHDF5 contents:")
    for key in f:
        print(f"  {key}: {f[key].shape} {f[key].dtype}")
