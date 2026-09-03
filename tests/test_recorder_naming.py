"""Episode naming/indexing and artifact bookkeeping for TrajectoryRecorder.

A cherry-pick once grafted the old HDF5-era `start()` onto the MCAP recorder,
which both crashed on stale attributes and left the `episode_name` feature
inert (files still came out timestamped). These pin down the naming contract
and the artifact list that callers use to keep or discard an episode.
"""

import threading

import numpy as np
import pytest

from clear_franka.episode_io import find_latest_episode, load_episode
from clear_franka.recorder import TrajectoryRecorder


class FakeCamera:
    """Stands in for ZedCamera, asserting the keyword contract start() uses."""

    resolution = "HD720"
    fps = 30

    def __init__(self, camera_id="third_person", serial_number=37820861):
        self.camera_id = camera_id
        self.serial_number = serial_number
        self.started = []

    def start_recording(self, video_path, svo_compression="H264", format="svo"):
        self.started.append(video_path)
        # Touch the file so artifact_paths() can be checked against reality.
        with open(video_path, "wb") as f:
            f.write(b"fake video")

    def stop_recording(self):
        return {"frame_count": 7}


def record(save_dir, episode_name=None, camera_format="svo", cameras=None, steps=3,
           metadata=None):
    recorder = TrajectoryRecorder(
        save_dir=save_dir,
        cameras=cameras,
        episode_name=episode_name,
        camera_format=camera_format,
        metadata=metadata or {},
    )
    recorder.start()
    for _ in range(steps):
        recorder.step(
            np.zeros(3), np.eye(3), np.zeros(3), np.zeros(3), 0, True,
            joint_pos=np.zeros(7),
        )
    recorder.stop()
    return recorder


def test_unnamed_episodes_are_timestamped(tmp_path):
    recorder = record(tmp_path)
    path = recorder.last_saved_path
    assert path.exists()
    assert path.name.startswith("episode_") and path.suffix == ".mcap"


def test_named_episodes_use_the_next_free_index(tmp_path):
    for expected in range(3):
        recorder = record(tmp_path, episode_name="pick_cup")
        assert recorder.last_saved_path.name == f"pick_cup_{expected}.mcap"


def test_index_resumes_above_existing_episodes(tmp_path):
    (tmp_path / "pick_cup_0.mcap").write_bytes(b"")
    (tmp_path / "pick_cup_7.mcap").write_bytes(b"")
    recorder = record(tmp_path, episode_name="pick_cup")
    assert recorder.last_saved_path.name == "pick_cup_8.mcap"


def test_episode_name_is_sanitized(tmp_path):
    recorder = record(tmp_path, episode_name="pick the cup!")
    assert recorder.last_saved_path.name == "pick_the_cup_0.mcap"


def test_camera_sidecars_share_the_episode_stem(tmp_path):
    cam = FakeCamera()
    recorder = record(tmp_path, episode_name="pick_cup", cameras={"third_person": cam})
    assert recorder.episode_base == "pick_cup_0"
    assert cam.started == [str(tmp_path / "pick_cup_0_third_person_video.svo2")]


def test_rgb_format_writes_mp4_sidecars(tmp_path):
    cam = FakeCamera()
    record(tmp_path, episode_name="demo", camera_format="rgb",
           cameras={"third_person": cam})
    assert cam.started == [str(tmp_path / "demo_0_third_person_video.mp4")]


def test_artifact_paths_lists_every_written_file(tmp_path):
    cams = {"third_person": FakeCamera("third_person"), "hand": FakeCamera("hand", 10986074)}
    recorder = record(tmp_path, episode_name="demo", cameras=cams)
    artifacts = recorder.artifact_paths()
    assert {p.name for p in artifacts} == {
        "demo_0.mcap",
        "demo_0_third_person_video.svo2",
        "demo_0_hand_video.svo2",
    }
    assert all(p.exists() for p in artifacts)


def test_metadata_carries_name_and_index_for_replay_joins(tmp_path):
    cam = FakeCamera()
    recorder = record(tmp_path, episode_name="pick_cup", cameras={"third_person": cam},
                      metadata={"mode_title": "pick_cup"})
    attrs = load_episode(recorder.last_saved_path)["attrs"]
    assert attrs["episode_name"] == "pick_cup"
    assert attrs["episode_index"] == "0"
    assert attrs["episode_base"] == "pick_cup_0"
    assert attrs["mode_title"] == "pick_cup"
    assert attrs["camera.third_person.video_file"] == "pick_cup_0_third_person_video.svo2"


def test_unnamed_episodes_omit_the_index_metadata(tmp_path):
    recorder = record(tmp_path)
    attrs = load_episode(recorder.last_saved_path)["attrs"]
    assert "episode_index" not in attrs
    assert "episode_name" not in attrs


def test_samples_round_trip(tmp_path):
    recorder = record(tmp_path, episode_name="demo", steps=5)
    episode = load_episode(recorder.last_saved_path)
    assert episode["joint_pos"].shape == (5, 7)


def test_find_latest_episode_sees_named_episodes(tmp_path):
    """find_latest_episode globbed `episode_*.mcap`, so named demos were invisible."""
    recorder = record(tmp_path, episode_name="pick_cup")
    assert find_latest_episode(str(tmp_path)) == recorder.last_saved_path


def test_find_latest_episode_orders_by_write_time(tmp_path):
    first = record(tmp_path, episode_name="zzz").last_saved_path
    second = record(tmp_path, episode_name="aaa").last_saved_path
    # "aaa_0" sorts before "zzz_0" by name, so name ordering would pick the wrong one.
    assert find_latest_episode(str(tmp_path)) == second
    assert first.exists()


def test_failed_start_does_not_leak_the_writer_thread(tmp_path):
    """The writer thread is non-daemon and blocks on the queue: if start() raises
    without shutting it down, the process hangs on exit."""
    class BrokenCamera(FakeCamera):
        def start_recording(self, *args, **kwargs):
            raise RuntimeError("camera boom")

    before = threading.active_count()
    recorder = TrajectoryRecorder(save_dir=tmp_path, cameras={"c": BrokenCamera()})
    with pytest.raises(RuntimeError, match="camera boom"):
        recorder.start()
    assert not recorder.recording
    assert threading.active_count() == before


def test_step_before_start_is_ignored(tmp_path):
    recorder = TrajectoryRecorder(save_dir=tmp_path)
    assert recorder.step(np.zeros(3), np.eye(3), np.zeros(3), np.zeros(3), 0, True) is None
    assert recorder.artifact_paths() == []
