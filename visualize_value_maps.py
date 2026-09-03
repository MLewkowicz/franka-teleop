"""Build and visualize the CURRENTLY ACTIVE place-stage value map (no robot).

Reads the live steering config (conf/config.yaml → deploy.steering) and builds
the exact value map the deploy would use for the active `place_mode`:

  * rack    (mode B) — wine-rack front-face affordance + cabinet/underneath
                       avoidance walls (build_place_value_map).
  * cabinet (mode A) — gentle point-attractor at the in-front approach target +
                       the closed-cabinet avoidance blob (build_center_attractor_value_map).

Renders an interactive Plotly HTML via LangSteer's ValueMapVisualizer:
affordance in Greens, avoidance in Reds, scene/target/avoidance boxes overlaid,
plus a printed descent probe so you can see which way the policy would be
pushed.

Usage:
    uv run python visualize_value_maps.py                 # active map from config
    uv run python visualize_value_maps.py --place-mode rack
    uv run python visualize_value_maps.py --map-size 80

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

from omegaconf import OmegaConf  # noqa: E402

from clear_franka.value_map_viz import (  # noqa: E402
    box_overlay as _box_obj,
    print_gradient_probes as _print_gradient_probes,
)
from clear_franka.value_maps import (  # noqa: E402
    CABINET,
    RACK,
    UNDERNEATH,
    build_center_attractor_value_map,
    build_place_value_map,
    front_face_center,
    load_boxes,
)


def _utf8_stdout() -> None:
    """Don't let an ASCII locale turn a nabla in a log line into a crash."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # already wrapped / not a TTY
            pass


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="conf/config.yaml",
                   help="Hydra config to read deploy.steering from")
    p.add_argument("--place-mode", default=None,
                   choices=["rack", "cabinet", "llm"],
                   help="override place_mode (default: read from config)")
    p.add_argument("--stage", type=int, default=0,
                   help="which position stage to render (0-based; cabinet / llm modes)")
    p.add_argument("--boxes", default=None,
                   help="workspace boxes json (default: from steering.position.boxes_path)")
    p.add_argument("--out-dir", default="outputs/value_maps")
    p.add_argument("--map-size", type=int, default=None, help="override map_size")
    p.add_argument("--quality", default="medium",
                   choices=["low", "medium", "high", "best"])
    p.add_argument("--show", action="store_true", help="open in browser")
    return p.parse_args()


def build_active_value_map(steer: dict, stage: int = 0):
    """Build the value map for the active place_mode from the steering config.

    Returns (vm, target_world, place_mode, ws_min, ws_max, context_boxes).
    `context_boxes` are wireframe overlays (scene boxes + target marker +
    avoidance blob) for the visualizer.
    """
    place_mode = str(steer.get("place_mode", "rack")).lower()
    pos = dict(steer.get("position", {}))
    boxes_path = pos.get("boxes_path", "data/workspace_boxes.json")
    boxes = load_boxes(boxes_path)
    ws_min = np.asarray(pos["workspace_bounds_min"], dtype=np.float32)
    ws_max = np.asarray(pos["workspace_bounds_max"], dtype=np.float32)
    map_size = int(pos.get("map_size", 100))

    # Scene wireframes for context (the real boxes), always shown. Missing
    # boxes are skipped — a scene JSON written by the perception pipeline has
    # semantic names, not the hand-edited box_00N ones.
    ctx = [
        _box_obj(label, boxes[key]["center"], boxes[key]["size"])
        for label, key in (("wine_rack", RACK), ("cabinet", CABINET),
                           ("underneath", UNDERNEATH))
        if key in boxes
    ]

    if place_mode == "llm":
        # Whatever synthesize_value_map.py last wrote for this scene + task.
        # The artifact owns its grid, so ws_min/ws_max/map_size above are
        # replaced by the ones the maps were actually built on.
        from clear_franka.perception import load_scene_boxes
        from clear_franka.value_map_llm import load_stages

        stages, meta = load_stages(dict(steer.get("llm", {}))["artifact_path"])
        stage = max(0, min(stage, len(stages) - 1))
        st = stages[stage]
        ws_min = np.asarray(meta["workspace_bounds_min"], dtype=np.float32)
        ws_max = np.asarray(meta["workspace_bounds_max"], dtype=np.float32)
        ctx = []
        if meta.get("boxes_path") and Path(meta["boxes_path"]).exists():
            ctx = [_box_obj(b.name, b.center, b.size)
                   for b in load_scene_boxes(meta["boxes_path"])]
        for i, other in enumerate(stages):
            ctx.append(_box_obj(f"stage{i}_{other.label}", other.target_world,
                                np.array([0.03, 0.03, 0.03])))
        print(f"task: {meta['task']!r}")
        print(f"stage {stage} '{st.label}': affordance={st.affordance_query!r} "
              f"avoidance={st.avoidance_query!r}")
        return st.value_map, st.target_world, place_mode, ws_min, ws_max, ctx

    if place_mode == "cabinet":
        cpos = dict(steer.get("cabinet", {}).get("position", {}))
        # Stage targets (or single fallback target). Render the chosen stage's
        # attractor map; mark all stage targets so the sequence is visible.
        stages_cfg = cpos.get("stages", None)
        if stages_cfg:
            # A stage target is either explicit coords (`target: [x,y,z]`) or a
            # workspace box referenced by name (`target_box: box_004`) -> its
            # center. Same resolution CombinedBoxSteering does, so the render
            # matches what actually steers.
            targets = [
                np.asarray(boxes[s["target_box"]]["center"], dtype=np.float32)
                if s.get("target_box") else np.asarray(s["target"], dtype=np.float32)
                for s in stages_cfg
            ]
        else:
            targets = [np.asarray(cpos["target"], dtype=np.float32)]
        stage = max(0, min(stage, len(targets) - 1))
        target = targets[stage]
        for i, t in enumerate(targets):
            ctx.append(_box_obj(f"stage{i+1}_target", t, np.array([0.03, 0.03, 0.03])))
        av = dict(cpos.get("avoidance", {}))
        avoidance_boxes = None
        if av.get("enabled", False) and av.get("box") is not None:
            b = av["box"]
            avoidance_boxes = [(np.asarray(b["center"], dtype=np.float32),
                                np.asarray(b["size"], dtype=np.float32))]
            ctx.append(_box_obj("closed_cabinet_avoid", b["center"], b["size"]))
        vm = build_center_attractor_value_map(
            target, ws_min=ws_min, ws_max=ws_max, map_size=map_size,
            seed_extent_m=float(cpos.get("seed_extent_m", 0.04)),
            avoidance_boxes=avoidance_boxes,
            avoidance_weight=float(av.get("weight", 0.0)),
            obstacle_sigma=float(av.get("sigma", 3.0)),
        )
        target_world = target
    else:
        face_thickness_m = float(pos.get("face_thickness_m", 0.04))
        forward_extend_m = float(pos.get("forward_extend_m", 0.06))
        basin_y_offset_m = float(pos.get("basin_y_offset_m", 0.0))
        basin_z_offset_m = float(pos.get("basin_z_offset_m", 0.0))
        rack_name = pos.get("rack", RACK)
        vm = build_place_value_map(
            boxes, ws_min=ws_min, ws_max=ws_max, map_size=map_size,
            avoidance_weight=float(pos.get("avoidance_weight", 2.0)),
            obstacle_sigma=float(pos.get("obstacle_sigma", 1.0)),
            suppress_affordance_in_obstacles=bool(
                pos.get("suppress_affordance_in_obstacles", False)),
            face_thickness_m=face_thickness_m,
            forward_extend_m=forward_extend_m,
            wall_x_offset_m=float(pos.get("wall_x_offset_m", 0.0)),
            avoidance_carve_radius_m=float(pos.get("avoidance_carve_radius_m", 0.0)),
            affordance_y_extent_m=(float(pos["affordance_y_extent_m"])
                                   if pos.get("affordance_y_extent_m") is not None else None),
            affordance_z_extent_m=(float(pos["affordance_z_extent_m"])
                                   if pos.get("affordance_z_extent_m") is not None else None),
            include_underneath_wall=bool(pos.get("include_underneath_wall", True)),
            basin_y_offset_m=basin_y_offset_m,
            basin_z_offset_m=basin_z_offset_m,
            rack=rack_name,
        )
        target_world = front_face_center(
            boxes[rack_name]["center"], boxes[rack_name]["size"],
            face_thickness_m=face_thickness_m, forward_extend_m=forward_extend_m,
            y_offset_m=basin_y_offset_m, z_offset_m=basin_z_offset_m,
        )
        ctx.append(_box_obj("basin_target", target_world, np.array([0.03, 0.03, 0.03])))

    return vm, target_world, place_mode, ws_min, ws_max, ctx


def main() -> int:
    _utf8_stdout()
    args = _parse_args()
    from voxposer.visualizer import ValueMapVisualizer

    cfg = OmegaConf.load(args.config)
    steer = OmegaConf.to_container(cfg.deploy.steering, resolve=True)
    if args.place_mode is not None:
        steer["place_mode"] = args.place_mode
    if args.boxes is not None:
        steer.setdefault("position", {})["boxes_path"] = args.boxes
    if args.map_size is not None:
        steer.setdefault("position", {})["map_size"] = args.map_size

    vm, target, place_mode, ws_min, ws_max, ctx = build_active_value_map(steer, stage=args.stage)
    print(f"Active place_mode = {place_mode}   rendering stage {args.stage} target = "
          f"{np.round(np.asarray(target), 3)}")
    _print_gradient_probes(vm, target)

    out_dir = Path(args.out_dir)
    viz = ValueMapVisualizer({
        "visualization_save_dir": str(out_dir),
        "visualization_quality": args.quality,
    })
    fig = viz.visualize(vm, objects=ctx, save=True, show=False, filename="place")

    fig.write_html(str(out_dir / "latest.html"))

    if args.show:
        fig.show()

    print(f"\nSaved: {out_dir / 'place.html'}  (place_mode={place_mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
