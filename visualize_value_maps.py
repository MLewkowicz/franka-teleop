"""Render a synthesized LLM value-map artifact (no robot).

Loads a `.npz` written by `synthesize_value_map.py` (or the scratch file
`deploy_diffuser_actor.py` regenerates at every launch,
`data/value_maps/place.npz`) and renders the requested stage as an
interactive Plotly HTML via LangSteer's ValueMapVisualizer: affordance in
Greens, avoidance in Reds, scene/target boxes overlaid, plus a printed
descent probe so you can see which way the policy would be pushed.

Usage:
    uv run python visualize_value_maps.py                       # default artifact/stage
    uv run python visualize_value_maps.py --artifact data/value_maps/place.npz --stage 0

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

from clear_franka.value_map_viz import (  # noqa: E402
    box_overlay as _box_obj,
    print_gradient_probes as _print_gradient_probes,
)


def _utf8_stdout() -> None:
    """Don't let an ASCII locale turn a nabla in a log line into a crash."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # already wrapped / not a TTY
            pass


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifact", default="data/value_maps/place.npz",
                   help="synthesized value-map .npz to render (from "
                        "synthesize_value_map.py or deploy's scratch file)")
    p.add_argument("--stage", type=int, default=0,
                   help="which position stage to render (0-based)")
    p.add_argument("--out-dir", default="outputs/value_maps")
    p.add_argument("--quality", default="medium",
                   choices=["low", "medium", "high", "best"])
    p.add_argument("--show", action="store_true", help="open in browser")
    return p.parse_args()


def build_active_value_map(artifact_path: str, stage: int = 0):
    """Load `artifact_path` and return the requested stage for rendering.

    Returns (vm, target_world, task, ws_min, ws_max, context_boxes).
    `context_boxes` are wireframe overlays (scene boxes + every stage's
    target marker) for the visualizer.
    """
    from clear_franka.perception import load_scene_boxes
    from clear_franka.value_map_llm import load_stages

    stages, meta = load_stages(artifact_path)
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
    return st.value_map, st.target_world, meta["task"], ws_min, ws_max, ctx


def main() -> int:
    _utf8_stdout()
    args = _parse_args()
    from voxposer.visualizer import ValueMapVisualizer

    vm, target, task, ws_min, ws_max, ctx = build_active_value_map(
        args.artifact, stage=args.stage
    )
    print(f"Artifact = {args.artifact}   rendering stage {args.stage} target = "
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

    print(f"\nSaved: {out_dir / 'place.html'}  (task={task!r})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
