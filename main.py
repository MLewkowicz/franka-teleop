"""Franka Panda teleoperation and trajectory replay.

Usage:
    uv run python main.py                       # teleop (default)
    uv run python main.py mode=replay            # replay latest episode
    uv run python main.py mode=replay replay.episode=data/episode_20260407_150000.h5
    uv run python main.py mode=replay replay.speed=0.5
"""

import hydra
from omegaconf import DictConfig

from net_franky import setup_net_franky


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    setup_net_franky(cfg.net_franky.ip, cfg.net_franky.port)

    if cfg.mode == "teleop":
        from teleop import run_teleop
        run_teleop(cfg)
    elif cfg.mode == "replay":
        from replay import run_replay
        run_replay(cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


if __name__ == "__main__":
    main()
