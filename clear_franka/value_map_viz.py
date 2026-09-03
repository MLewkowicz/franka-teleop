"""Shared rendering + sanity checks for value maps.

Both `visualize_value_maps.py` (hand-built rack/cabinet maps) and
`synthesize_value_map.py` (LLM-synthesized maps) render through here, so a map
looks the same however it was built and the descent probe means the same thing.
Rendering itself is LangSteer's `voxposer.visualizer.ValueMapVisualizer`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def box_overlay(name: str, center, size) -> dict:
    """A wireframe overlay entry for `ValueMapVisualizer.visualize(objects=...)`."""
    center = np.asarray(center, dtype=np.float32)
    size = np.asarray(size, dtype=np.float32)
    return {
        "name": name,
        "_position_world": center,
        "obb_center_world": center,
        "obb_size": size,
        "obb_rotation": np.eye(3, dtype=np.float32),
        "aabb": np.stack([center - size / 2.0, center + size / 2.0]),
    }


def print_gradient_probes(vm, target_world) -> None:
    """Sample the descent (= -∇cost) at points around the target; it should
    point toward the target (and away from any avoidance region)."""
    t = np.asarray(target_world, dtype=np.float32)
    probes = {
        "0.15m -x of target": t + np.array([-0.15, 0.0, 0.0], np.float32),
        "0.15m -y of target": t + np.array([0.0, -0.15, 0.0], np.float32),
        "0.15m below target": t + np.array([0.0, 0.0, -0.15], np.float32),
        "0.20m -x,-z (approach)": t + np.array([-0.20, 0.0, -0.20], np.float32),
    }
    print(f"\nDescent sanity check (descent = -∇cost; should point toward target "
          f"{np.round(t, 3)}):")
    print(f"  {'probe':<26} {'world xyz':<24} {'descent dir':<22} cos_to_target")
    for label, pt in probes.items():
        grad = vm.gradient_at_world_points(pt)[0]
        descent = -grad
        n = np.linalg.norm(descent)
        unit = descent / n if n > 1e-9 else descent
        to_t = t - pt
        tn = np.linalg.norm(to_t)
        cos = float(np.dot(unit, to_t / tn)) if (n > 1e-9 and tn > 1e-9) else 0.0
        print(f"  {label:<26} {np.array2string(pt, precision=2):<24} "
              f"{np.array2string(unit, precision=2):<22} {cos:+.2f}")


def render(
    vm,
    *,
    out_dir: str | Path,
    filename: str,
    objects: list | None = None,
    scene_points: np.ndarray | None = None,
    scene_colors: np.ndarray | None = None,
    ee_pos_world: np.ndarray | None = None,
    quality: str = "medium",
    also_latest: bool = True,
    show: bool = False,
):
    """Write an interactive Plotly HTML for one value map. Returns the figure.

    Rendering is entirely LangSteer's `ValueMapVisualizer`: affordance and
    avoidance isosurfaces, object wireframes, the scene cloud and the EE marker.
    Passing `scene_points` is what makes the map legible on hardware — the
    isosurfaces float in empty space otherwise. Use `print_gradient_probes` to
    check which way the field actually pulls.
    """
    from voxposer.visualizer import ValueMapVisualizer

    out_dir = Path(out_dir)
    viz = ValueMapVisualizer({
        "visualization_save_dir": str(out_dir),
        "visualization_quality": quality,
    })
    if scene_points is not None:
        viz.update_scene_points(scene_points, scene_colors)
    fig = viz.visualize(
        vm,
        ee_pos_world=ee_pos_world,
        objects=objects,
        save=True,
        show=False,
        filename=filename,
    )

    if also_latest:
        fig.write_html(str(out_dir / "latest.html"))
    if show:
        fig.show()
    return fig
