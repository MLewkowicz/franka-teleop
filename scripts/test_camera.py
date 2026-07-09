"""Record a short native ZED SVO2 file for a camera smoke test."""

import time
from pathlib import Path

from clear_franka.camera import ZedCamera


data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
video_path = str(data_dir / "test_video.svo2")

with ZedCamera() as camera:
    camera.run()
    camera.start_recording(video_path, svo_compression="H265_LOSSLESS")
    time.sleep(5.0)
    summary = camera.stop_recording()

print(f"Recorded {summary['frame_count']} frames to {video_path}")
print(f"Clock anchors: {summary}")
