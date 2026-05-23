"""Franka Panda teleoperation and trajectory replay.

Usage:
    uv run python main.py                       # teleop (default)
    uv run python main.py mode=replay            # replay latest episode
    uv run python main.py mode=replay replay.episode=data/episode_20260407_150000.h5
    uv run python main.py mode=replay replay.speed=0.5
    uv run python main.py mode=calibrate calibration.camera_mount=hand
"""

import hydra
from omegaconf import DictConfig

from zero_franky import setup_zero_franky


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    setup_zero_franky(cfg.zero_franky.ip, cfg.zero_franky.port, pub_port=cfg.zero_franky.pub_port)

    if cfg.mode == "teleop":
        from teleop import run_teleop
        run_teleop(cfg)
    elif cfg.mode == "replay":
        from replay import run_replay
        run_replay(cfg)
    elif cfg.mode == "calibrate":
        from calibrate_extrinsics import run_calibration
        run_calibration(cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


if __name__ == "__main__":
    main()
