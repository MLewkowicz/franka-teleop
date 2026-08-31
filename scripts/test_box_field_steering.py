"""Offline smoke test for CombinedBoxSteering (no robot, no GPU).

Verifies the plumbing of the combined rotation + value-map position steering:
  * get_guidance returns a tensor matching the model_output shape, all finite;
  * the rotation slice [3:9] is byte-for-byte what TargetRotationSteering alone
    produces (the position branch must not perturb rotation);
  * the position slice [0:3] is non-zero (the value-map branch is active) while
    rotation-only leaves it zero;
  * both guidance modes (epsilon / dps) route to the right container shape.

Run:  uv run python scripts/test_box_field_steering.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, "/home/clear/LangSteer")
sys.path.insert(0, str(ROOT))

from clear_franka.box_field_steering import CombinedBoxSteering  # noqa: E402
from steering.target_rotation import TargetRotationSteering  # noqa: E402

H = 16
T = 100
GRIPPER_POS = np.array([0.60, -0.12, 0.45], dtype=np.float32)  # in front of rack
GRIPPER_EULER = np.array([0.0, 0.0, 0.0], dtype=np.float32)


class StubScheduler:
    """Minimal DDPM-style scheduler: monotone ᾱ from ~1 (t=0) to ~0 (t=T-1)."""

    def __init__(self, T: int = T):
        t = torch.linspace(0.0, 1.0, T)
        self.alphas_cumprod = torch.clamp(torch.cos(t * np.pi / 2) ** 2, 1e-6, 1 - 1e-6)


def _cfg(mode: str) -> dict:
    return {
        "device": "cpu",
        "guidance_mode": mode,
        "target_euler": [-1.35, 1.31, 1.41],
        "guidance_strength": 0.6,
        "horizon": H,
        "start_guidance_timestep": 10_000,
        "relative": True,
        "gripper_loc_bounds": [[-0.4056, -0.4340, -0.6588], [0.5103, 0.2986, 0.3674]],
        "position": {
            "boxes_path": str(ROOT / "data" / "workspace_boxes.json"),
            "workspace_bounds_min": [0.45, -0.50, -0.15],
            "workspace_bounds_max": [1.00, 0.80, 0.75],
            "map_size": 60,
            "avoidance_weight": 2.0,
            "guidance_strength": 0.5,
            "start_guidance_timestep": 10_000,
        },
    }


def _run_mode(mode: str) -> None:
    torch.manual_seed(0)
    cfg = _cfg(mode)

    steering = CombinedBoxSteering(cfg)
    sched = StubScheduler()
    steering.set_rotation_scheduler(sched)
    steering.set_position_scheduler(sched)
    steering.set_current_gripper_pos(GRIPPER_POS)
    steering.set_current_gripper_rotation(GRIPPER_EULER)

    current_sample = torch.randn(1, H, 9) * 0.5   # x_t: pos[0:3] + 6D rot[3:9]
    model_output = torch.randn(1, H, 10) * 0.5    # eps + openness[9]
    t = 10

    g = steering.get_guidance(current_sample, t, None, model_output)

    # Shape + finiteness. epsilon adds to model_output (10), dps to x_t (9).
    expected_last = 10 if mode == "epsilon" else 9
    assert g.shape == (1, H, expected_last), (mode, g.shape)
    assert torch.isfinite(g).all(), f"{mode}: non-finite guidance"

    # Rotation-slice parity with TargetRotationSteering alone.
    rot_only = TargetRotationSteering(cfg)
    rot_only.set_rotation_scheduler(sched)
    rot_only.set_current_gripper_rotation(GRIPPER_EULER)
    g_rot = rot_only.get_guidance(current_sample, t, None, model_output)
    assert torch.allclose(g[:, :, 3:9], g_rot[:, :, 3:9]), f"{mode}: rotation slice drift"

    # Rotation-only must not touch position; combined must.
    assert g_rot[:, :, :3].abs().sum() == 0, f"{mode}: rotation wrote to position"
    assert g[:, :, :3].abs().sum() > 0, f"{mode}: position branch produced no delta"

    print(f"[{mode}] OK  shape={tuple(g.shape)}  "
          f"pos|delta|={g[:, :, :3].abs().mean().item():.4e}  "
          f"rot|delta|={g[:, :, 3:9].abs().mean().item():.4e}")


def _check_field_direction() -> None:
    """Document the descent direction at a couple of probes (not an assertion)."""
    steering = CombinedBoxSteering(_cfg("epsilon"))
    vm = steering._value_map
    for label, pt in {
        "in front of rack": np.array([0.54, -0.12, 0.52], np.float32),
        "cabinet surface": np.array([0.68, 0.34, 0.62], np.float32),
    }.items():
        descent = -vm.gradient_at_world_points(pt)[0]
        n = np.linalg.norm(descent)
        unit = descent / n if n > 1e-9 else descent
        print(f"  descent @ {label:<18}: {np.array2string(unit, precision=2)}")


if __name__ == "__main__":
    _run_mode("epsilon")
    _run_mode("dps")
    _check_field_direction()
    print("\nAll CombinedBoxSteering smoke checks passed.")
