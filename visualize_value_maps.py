"""Build and visualize the hardcoded place-stage value maps (no robot needed).

Constructs the affordance (wine-rack front face) + avoidance (cabinet and the
volume beneath it) value map from data/workspace_boxes.json and renders it as an
interactive Plotly HTML via LangSteer's ValueMapVisualizer: affordance in
Greens, avoidance in Reds, box wireframes overlaid. Also prints a gradient
descent sanity check at a few probe points and (optionally) overlays a cone
quiver of the descent field so you can eyeball where the policy would be pushed.

Usage:
    uv run python visualize_value_maps.py
    uv run python visualize_value_maps.py --avoidance-weight 1.5 --forward-extend 0.08
    uv run python visualize_value_maps.py --no-quiver --map-size 80

Output: outputs/value_maps/place.html (and latest.html). Open in a browser.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# LangSteer provides ValueMap / ValueMapVisualizer / voxel helpers.
LANGSTEER_PATH = "/home/clear/LangSteer"
sys.path.insert(0, LANGSTEER_PATH)

from clear_franka.value_maps import (  # noqa: E402
    CABINET,
    RACK,
    UNDERNEATH,
    build_place_value_map,
    load_boxes,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--boxes", default="data/workspace_boxes.json")
    p.add_argument("--out-dir", default="outputs/value_maps")
    p.add_argument("--map-size", type=int, default=100)
    p.add_argument("--avoidance-weight", type=float, default=1.0)
    p.add_argument("--obstacle-sigma", type=float, default=1.0)
    p.add_argument("--face-thickness", type=float, default=0.04)
    p.add_argument("--forward-extend", type=float, default=0.06)
    p.add_argument("--obstacle-back-extend", type=float, default=0.0,
                   help="extend obstacle +x face backward (m) so descent always points -x toward robot")
    p.add_argument("--wall-x-offset", type=float, default=0.0,
                   help="push avoidance wall slab +x into the obstacle (m), away from rack-approach corridor")
    p.add_argument("--avoidance-carve-radius", type=float, default=0.0,
                   help="soft Gaussian carve radius (m) around the basin where avoidance is zeroed")
    p.add_argument("--affordance-y-extent", type=float, default=None,
                   help="clip rack affordance seed y extent (m); narrow → stronger y pull to basin center")
    p.add_argument("--affordance-z-extent", type=float, default=None,
                   help="clip rack affordance seed z extent (m); narrow → stronger z pull to basin center")
    p.add_argument("--suppress-affordance-in-obstacles", action="store_true",
                   help="zero global EDT affordance inside cabinet/underneath (default off — keeps pull everywhere)")
    p.add_argument("--ws-min", type=float, nargs=3, default=[0.45, -0.50, -0.15])
    p.add_argument("--ws-max", type=float, nargs=3, default=[1.00, 0.80, 0.75])
    p.add_argument("--quality", default="medium",
                   choices=["low", "medium", "high", "best"])
    p.add_argument("--quiver", dest="quiver", action="store_true", default=True,
                   help="overlay a cone quiver of the descent field (default on)")
    p.add_argument("--no-quiver", dest="quiver", action="store_false")
    p.add_argument("--show", action="store_true", help="open in browser")
    return p.parse_args()


def _box_objects(boxes: dict) -> list[dict]:
    """Plain dicts the ValueMapVisualizer renders as labelled OBB wireframes."""
    labels = {RACK: "wine_rack", CABINET: "cabinet", UNDERNEATH: "underneath"}
    objs = []
    for name, label in labels.items():
        center = boxes[name]["center"]
        size = boxes[name]["size"]
        objs.append({
            "name": label,
            "_position_world": center,
            "obb_center_world": center,
            "obb_size": size,
            "obb_rotation": np.eye(3, dtype=np.float32),
            # aabb is only consulted in the non-OBB fallback, but the renderer
            # guards `if aabb is None: continue`, so supply a valid placeholder.
            "aabb": np.stack([center - size / 2.0, center + size / 2.0]),
        })
    return objs


def _print_gradient_probes(vm, boxes) -> None:
    """Sample the cost gradient at a few points; print the descent direction.

    Descent (= -gradient) should point toward the rack front face and away from
    the cabinet / underneath volumes.
    """
    rack_c = boxes[RACK]["center"]
    rack_h = boxes[RACK]["size"] / 2.0
    und_c = boxes[UNDERNEATH]["center"]
    und_h = boxes[UNDERNEATH]["size"] / 2.0
    # Trace the intended path: deep inside the underneath volume should push
    # OUT (expect -x toward the front); just in front of it / below the rack
    # should be drawn UP toward the rack front face.
    probes = {
        "underneath center": und_c.copy(),
        "underneath off-center": und_c + np.array([0.0, 0.30, 0.10], np.float32),
        "underneath front-low": np.array(
            [und_c[0] - und_h[0] + 0.03, rack_c[1], und_c[2] + 0.10], np.float32),
        "just front of underneath": np.array(
            [und_c[0] - und_h[0] - 0.06, rack_c[1], rack_c[2] - 0.10], np.float32),
        "in front of rack face": np.array(
            [rack_c[0] - rack_h[0] - 0.08, rack_c[1], rack_c[2]], dtype=np.float32),
        "above cabinet surface": boxes[CABINET]["center"] + np.array([0, 0, 0.05], np.float32),
    }
    print("\nGradient descent sanity check (descent = -∇cost, normalized):")
    print(f"  {'probe':<24} {'world xyz':<26} {'descent dir':<26} aff")
    for label, pt in probes.items():
        grad = vm.gradient_at_world_points(pt)[0]          # toward increasing cost
        descent = -grad
        n = np.linalg.norm(descent)
        unit = descent / n if n > 1e-9 else descent
        aff = float(vm.affordance_at_world_points(pt)[0])
        print(f"  {label:<24} {np.array2string(pt, precision=2):<26} "
              f"{np.array2string(unit, precision=2):<26} {aff:.2f}")


def _add_descent_quiver(fig, vm, stride: int = 12) -> None:
    """Append a Plotly cone trace of the (subsampled) descent field."""
    import plotly.graph_objects as go

    M = vm.map_size
    ws_min, ws_max = vm.workspace_bounds_min, vm.workspace_bounds_max
    idx = np.arange(0, M, stride)
    ii, jj, kk = np.meshgrid(idx, idx, idx, indexing="ij")
    vox = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=-1).astype(np.float32)
    world = vox / (M - 1) * (ws_max - ws_min) + ws_min
    grad = vm.gradient_at_world_points(world)              # (N, 3)
    descent = -grad
    # Drop near-zero vectors (flat regions) to keep the plot readable.
    mag = np.linalg.norm(descent, axis=1)
    keep = mag > (mag.max() * 0.05 if mag.max() > 0 else np.inf)
    world, descent = world[keep], descent[keep]
    fig.add_trace(go.Cone(
        x=world[:, 0], y=world[:, 1], z=world[:, 2],
        u=descent[:, 0], v=descent[:, 1], w=descent[:, 2],
        sizemode="scaled", sizeref=2.0, anchor="tail",
        colorscale="Blues", showscale=False, opacity=0.6, name="descent",
    ))


def main() -> int:
    args = _parse_args()
    from voxposer.visualizer import ValueMapVisualizer

    boxes = load_boxes(args.boxes)
    vm = build_place_value_map(
        boxes,
        ws_min=np.asarray(args.ws_min, dtype=np.float32),
        ws_max=np.asarray(args.ws_max, dtype=np.float32),
        map_size=args.map_size,
        avoidance_weight=args.avoidance_weight,
        obstacle_sigma=args.obstacle_sigma,
        face_thickness_m=args.face_thickness,
        forward_extend_m=args.forward_extend,
        obstacle_back_extend_m=args.obstacle_back_extend,
        wall_x_offset_m=args.wall_x_offset,
        avoidance_carve_radius_m=args.avoidance_carve_radius,
        suppress_affordance_in_obstacles=args.suppress_affordance_in_obstacles,
        affordance_y_extent_m=args.affordance_y_extent,
        affordance_z_extent_m=args.affordance_z_extent,
    )

    _print_gradient_probes(vm, boxes)

    out_dir = Path(args.out_dir)
    viz = ValueMapVisualizer({
        "visualization_save_dir": str(out_dir),
        "visualization_quality": args.quality,
    })
    fig = viz.visualize(
        vm,
        objects=_box_objects(boxes),
        save=True,
        show=False,
        filename="place",
    )

    if args.quiver:
        _add_descent_quiver(fig, vm)
        # Re-save with the quiver overlay (visualize() saved before we added it).
        fig.write_html(str(out_dir / "place.html"))
        fig.write_html(str(out_dir / "latest.html"))

    if args.show:
        fig.show()

    print(f"\nSaved: {out_dir / 'place.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
