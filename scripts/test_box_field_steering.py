"""Offline smoke test for PositionFieldSteering (no robot, no GPU).

Verifies the plumbing of the value-map position steering:
  * get_guidance returns a tensor matching the model_output shape, all finite;
  * the rotation slice [3:9] is always exactly zero (no rotation branch — the
    policy is fully in control of rotation);
  * the position slice [0:3] is non-zero once told the place stage is active;
  * both guidance modes (epsilon / dps) route to the right container shape;
  * get_guidance is constant zero outside `stage_indices` (grasp by default),
    without the module being swapped for None at the call site.

Run:  uv run python scripts/test_box_field_steering.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, "/home/clear/LangSteer")
sys.path.insert(0, str(ROOT))

from clear_franka.box_field_steering import PositionFieldSteering  # noqa: E402

H = 16
T = 100
GRIPPER_POS = np.array([0.60, -0.12, 0.45], dtype=np.float32)  # in front of rack
WS_MIN = np.array([0.45, -0.50, -0.15], dtype=np.float32)
WS_MAX = np.array([1.00, 0.80, 0.75], dtype=np.float32)
MAP_SIZE = 60
TARGET_WORLD = np.array([0.70, 0.20, 0.55], dtype=np.float32)  # off from GRIPPER_POS


class StubScheduler:
    """Minimal DDPM-style scheduler: monotone ᾱ from ~1 (t=0) to ~0 (t=T-1)."""

    def __init__(self, T: int = T):
        t = torch.linspace(0.0, 1.0, T)
        self.alphas_cumprod = torch.clamp(torch.cos(t * np.pi / 2) ** 2, 1e-6, 1 - 1e-6)


def _make_llm_artifact(path: Path) -> Path:
    """A tiny synthetic single-stage LLM value-map artifact for the plumbing
    test -- PositionFieldSteering only ever loads maps this way."""
    from voxposer.calvin_interface import pc2voxel
    from voxposer.value_map import ValueMap

    from clear_franka.value_map_llm.artifact import save_stages
    from clear_franka.value_map_llm.synthesize import SynthesizedStage

    affordance = np.zeros((MAP_SIZE,) * 3, dtype=np.float32)
    idx = pc2voxel(TARGET_WORLD[None, :], WS_MIN, WS_MAX, MAP_SIZE)[0]
    affordance[tuple(np.clip(idx, 0, MAP_SIZE - 1))] = 1.0

    vm = ValueMap(
        affordance=affordance, avoidance=None,
        workspace_bounds_min=WS_MIN, workspace_bounds_max=WS_MAX,
        map_size=MAP_SIZE, instruction="test place",
    )
    vm.smooth(obstacle_sigma=2.0)
    vm.precompute_gradients(avoidance_weight=1.0)

    stage = SynthesizedStage(
        label="place", affordance_query="target", avoidance_query=None,
        arrival_radius_m=0.08, value_map=vm, target_world=TARGET_WORLD,
    )
    return save_stages(path, [stage], task="test place")


def _cfg(mode: str, artifact_path: str) -> dict:
    return {
        "device": "cpu",
        "guidance_mode": mode,
        "horizon": H,
        "relative": True,
        "gripper_loc_bounds": [[-0.4056, -0.4340, -0.6588], [0.5103, 0.2986, 0.3674]],
        "llm": {"artifact_path": artifact_path},
        "position": {
            "guidance_strength": 0.5,
            "start_guidance_timestep": 10_000,
        },
    }


def _run_mode(mode: str, artifact_path: str) -> None:
    torch.manual_seed(0)
    cfg = _cfg(mode, artifact_path)

    steering = PositionFieldSteering(cfg)
    sched = StubScheduler()
    steering.set_position_scheduler(sched)
    steering.set_current_gripper_pos(GRIPPER_POS)
    # Default stage_indices: [1] -- must be told "place" is active or
    # get_guidance returns constant zero (see _check_stage_gating below).
    steering.set_deploy_stage(1)

    current_sample = torch.randn(1, H, 9) * 0.5   # x_t: pos[0:3] + 6D rot[3:9]
    model_output = torch.randn(1, H, 10) * 0.5    # eps + openness[9]
    t = 10

    g = steering.get_guidance(current_sample, t, None, model_output)

    # Shape + finiteness. epsilon adds to model_output (10), dps to x_t (9).
    expected_last = 10 if mode == "epsilon" else 9
    assert g.shape == (1, H, expected_last), (mode, g.shape)
    assert torch.isfinite(g).all(), f"{mode}: non-finite guidance"

    # No rotation branch — the rotation slice is always exactly zero.
    assert torch.count_nonzero(g[:, :, 3:9]) == 0, f"{mode}: rotation slice non-zero"
    assert g[:, :, :3].abs().sum() > 0, f"{mode}: position branch produced no delta"

    print(f"[{mode}] OK  shape={tuple(g.shape)}  "
          f"pos|delta|={g[:, :, :3].abs().mean().item():.4e}")


def _check_stage_gating(artifact_path: str) -> None:
    """Attached for both stages; get_guidance must be constant zero outside
    `stage_indices` (grasp, stage 0) and real once told stage 1 (place) is
    active -- no None-swap at the call site, no timestep delay."""
    cfg = _cfg("epsilon", artifact_path)
    steering = PositionFieldSteering(cfg)
    sched = StubScheduler()
    steering.set_position_scheduler(sched)
    steering.set_current_gripper_pos(GRIPPER_POS)

    current_sample = torch.randn(1, H, 9) * 0.5
    model_output = torch.randn(1, H, 10) * 0.5

    steering.set_deploy_stage(0)  # grasp -- not in stage_indices: [1]
    g_grasp = steering.get_guidance(current_sample, 10, None, model_output)
    assert torch.count_nonzero(g_grasp) == 0, "grasp stage produced non-zero guidance"

    steering.set_deploy_stage(1)  # place
    g_place = steering.get_guidance(current_sample, 10, None, model_output)
    assert torch.count_nonzero(g_place) > 0, "place stage produced zero guidance"

    print("[stage gating] OK  grasp=all-zero  place=non-zero")


def _check_field_direction(artifact_path: str) -> None:
    """Document the descent direction at a couple of probes (not an assertion)."""
    steering = PositionFieldSteering(_cfg("epsilon", artifact_path))
    vm = steering._value_map
    for label, pt in {
        "near gripper start": GRIPPER_POS,
        "near target": TARGET_WORLD,
    }.items():
        descent = -vm.gradient_at_world_points(pt)[0]
        n = np.linalg.norm(descent)
        unit = descent / n if n > 1e-9 else descent
        print(f"  descent @ {label:<18}: {np.array2string(unit, precision=2)}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact = str(_make_llm_artifact(Path(tmpdir) / "test_place.npz"))
        _run_mode("epsilon", artifact)
        _run_mode("dps", artifact)
        _check_stage_gating(artifact)
        _check_field_direction(artifact)
    print("\nAll PositionFieldSteering smoke checks passed.")
