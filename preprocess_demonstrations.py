"""Batch / re-run wrapper around `preprocess_episode`.

Usage:
    # Re-process every raw episode in `cfg.preprocess.input_dir` whose
    # processed copy does not yet exist:
    uv run python preprocess_demonstrations.py

    # Re-process a single raw episode:
    uv run python preprocess_demonstrations.py +episode=data/episode_20260528_120000.mcap

    # Overwrite existing processed copies (re-tune with new params):
    uv run python preprocess_demonstrations.py preprocess.overwrite=true

Replay can run the same preprocessing in memory before playback. This CLI is
for writing processed .npz copies for inspection, caching, or batch re-runs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from clear_franka.preprocess import preprocess_episode


logger = logging.getLogger(__name__)


def _iter_raw_episodes(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir does not exist: {input_dir}")
    return sorted(input_dir.glob("*.mcap"))


def _preprocess_kwargs(pre_cfg: DictConfig) -> dict:
    params = OmegaConf.to_container(pre_cfg, resolve=True)
    trim_cfg = params.get("trim", {})
    retime_cfg = params.get("retime", {})
    smooth_cfg = params.get("smooth", {})
    return {
        "trim_enabled": bool(trim_cfg.get("enabled", True)),
        "trim_time_window": float(trim_cfg.get("time_window", 0.3)),
        "trim_threshold": float(trim_cfg.get("threshold", 0.01)),
        "retime_enabled": bool(retime_cfg.get("enabled", False)),
        "retime_sample_uniform": bool(retime_cfg.get("sample_uniform", False)),
        "retime_path_tol": retime_cfg.get("path_tol", None),
        "retime_max_joint_vel": retime_cfg.get("max_joint_vel", None),
        "retime_max_joint_accel": retime_cfg.get("max_joint_accel", None),
        "smooth_enabled": bool(smooth_cfg.get("enabled", True)),
        "smooth_max_joint_vel": smooth_cfg["max_joint_vel"],
        "smooth_max_joint_accel": smooth_cfg["max_joint_accel"],
        "smooth_max_joint_jerk": smooth_cfg["max_joint_jerk"],
        "smooth_dt": float(smooth_cfg.get("dt", 0.001)),
        "gripper_dwell_s": float(params.get("gripper_dwell_s", 0.0)),
        "params_metadata": params,
    }


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    pre_cfg = cfg.get("preprocess", None)
    if pre_cfg is None:
        raise RuntimeError("conf/config.yaml is missing the `preprocess:` block")

    input_dir = Path(pre_cfg.get("input_dir", "./data"))
    output_dir = Path(pre_cfg.get("output_dir", "./data/processed"))
    overwrite = bool(pre_cfg.get("overwrite", False))
    preprocess_kwargs = _preprocess_kwargs(pre_cfg)

    # Single-episode mode via `+episode=PATH` Hydra override.
    single = cfg.get("episode", None)
    if single is not None:
        paths = [Path(single)]
    else:
        paths = _iter_raw_episodes(input_dir)
        # Exclude any episode that already lives under output_dir (don't re-process
        # processed files if they share the directory tree by accident).
        paths = [p for p in paths if output_dir.resolve() not in p.resolve().parents]

    if not paths:
        print(f"No raw episodes found in {input_dir}")
        return

    print(f"Found {len(paths)} raw episode(s); writing to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    ok_count = 0
    skip_count = 0
    fail_count = 0
    for raw in paths:
        out = output_dir / f"{raw.stem}.npz"
        if out.exists() and not overwrite:
            print(f"  [skip] {raw.name} (processed copy exists; pass preprocess.overwrite=true to redo)")
            skip_count += 1
            continue
        ok = preprocess_episode(raw, out, **preprocess_kwargs)
        if ok:
            print(f"  [ok]   {raw.name} -> {out}")
            ok_count += 1
        else:
            print(f"  [fail] {raw.name} (see logs above)")
            fail_count += 1

    print(f"Done: ok={ok_count} skip={skip_count} fail={fail_count}")


if __name__ == "__main__":
    main()
