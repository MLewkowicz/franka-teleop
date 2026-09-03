"""Task + object list -> SAM 3 boxes in the world frame -> LLM value maps.

The offline half of the steering pipeline. Nothing here runs on the robot: it
captures a scene, segments the objects you name, fits their bounding boxes in
the Franka base frame, asks an LLM to turn the task into steering stages over
those boxes, and writes a .npz that `deploy_diffuser_actor.py` loads with
`deploy.steering.place_mode=llm`.

Run it with the LangSteer interpreter, NOT `uv run` — like the deploy, this
needs torch/plotly/sam3, which live in LangSteer's venv, not franka-teleop's:

    LS=/home/clear/LangSteer/.venv/bin/python

    # from the live third-person ZED
    $LS synthesize_value_map.py \
        --task "put the marker in the plate" \
        --objects "marker,plate" --source live --viser

    # from a recorded SVO (no robot, no live camera)
    $LS synthesize_value_map.py \
        --task "..." --objects "marker,plate" \
        --source svo --svo data/test_video.svo2 --viser

    # iterate on prompts against boxes already on disk (no camera, no SAM 3)
    $LS synthesize_value_map.py \
        --task "..." --source boxes --boxes-in data/scene_boxes.json

Outputs:
    data/scene_boxes.json          detected boxes (fr3_link0)
    data/value_maps/<name>.npz     the artifact deploy loads
    outputs/value_maps/<name>_stage<i>.html   one interactive plot per stage
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# LangSteer provides ValueMap / ValueMapVisualizer / the voxel helpers.
LANGSTEER_PATH = "/home/clear/LangSteer"
sys.path.insert(0, LANGSTEER_PATH)

from omegaconf import OmegaConf  # noqa: E402

from clear_franka.geometry import load_T_cam2base  # noqa: E402
from clear_franka.perception import (  # noqa: E402
    Sam3Segmenter,
    SvoSource,
    ZedLiveSource,
    collect_frames,
    detections_to_scene_boxes,
    frame_to_world_cloud,
    label_scene_image,
    load_scene_boxes,
    load_scene_cloud,
    load_scene_image,
    match_boxes_to_frame_detections,
    save_scene_boxes,
    save_scene_cloud,
    save_scene_image,
)
from clear_franka.value_map_llm import (  # noqa: E402
    save_stages,
    synthesize_value_maps,
    workspace_bounds_from_boxes,
)
from clear_franka.value_map_viz import (  # noqa: E402
    box_overlay,
    print_gradient_probes,
    render,
)

logger = logging.getLogger("synthesize_value_map")


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
    p.add_argument("--task", default=None,
                   help="task specification handed to the planner (required "
                        "unless --skip-llm)")
    p.add_argument("--objects", default="",
                   help="comma-separated text prompts for SAM 3, e.g. "
                        "'wine rack,cabinet,table'")
    p.add_argument("--config", default="conf/config.yaml",
                   help="config providing cameras.* and perception.*")

    src = p.add_argument_group("scene capture")
    src.add_argument("--source", default="svo", choices=["svo", "live", "boxes"],
                     help="svo = recording, live = the ZED, boxes = reuse a "
                          "boxes JSON and skip perception entirely")
    src.add_argument("--svo", default="data/test_video.svo2")
    src.add_argument("--camera", default="third_person",
                     help="which cameras.* entry supplies frames + extrinsics")
    src.add_argument("--frames", type=int, default=5,
                     help="frames to fuse (per-corner median) into each box")
    src.add_argument("--skip-frames", type=int, default=2,
                     help="frames discarded before capture (depth settles)")
    src.add_argument("--sam3-checkpoint", default=None,
                     help="overrides perception.sam3_checkpoint from the config")
    src.add_argument("--bpe-path", default=None,
                     help="overrides perception.bpe_path from the config")
    src.add_argument("--confidence", type=float, default=None)
    src.add_argument("--max-per-label", type=int, default=2,
                     help="instances kept per text prompt, highest score first")
    src.add_argument("--boxes-in", default=None,
                     help="boxes JSON to read (--source boxes)")
    src.add_argument("--boxes-out", default="data/scene_boxes.json")
    src.add_argument("--cloud-out", default="data/scene_cloud.npz",
                     help="decimated scene cloud, overlaid on the value-map "
                          "plots and reused by --source boxes")
    src.add_argument("--image-out", default="data/scene_image.png",
                     help="workspace snapshot handed to the planner LLM, "
                          "cached for --source boxes reruns")
    src.add_argument("--cloud-points", type=int, default=40000,
                     help="max cloud points kept (Plotly renders one colour "
                          "string per point)")
    src.add_argument("--viser", action="store_true",
                     help="serve the point cloud + fitted boxes for inspection")

    vm = p.add_argument_group("value maps")
    vm.add_argument("--map-size", type=int, default=100)
    vm.add_argument("--bounds-margin", type=float, default=0.25,
                    help="margin (m) around the detected boxes when the grid "
                         "bounds are derived from the scene")
    vm.add_argument("--ws-min", default=None, help="explicit grid min, 'x,y,z'")
    vm.add_argument("--ws-max", default=None, help="explicit grid max, 'x,y,z'")
    vm.add_argument("--avoidance-weight", type=float, default=1.0)
    vm.add_argument("--obstacle-sigma", type=float, default=3.0)
    vm.add_argument("--provider", default=None, choices=["anthropic", "openai"],
                    help="overrides value_map_llm.provider from the config")
    vm.add_argument("--model", default=None,
                    help="overrides value_map_llm.model from the config")
    vm.add_argument("--cache-dir", default=None,
                    help="overrides value_map_llm.cache_dir from the config")
    vm.add_argument("--no-cache", action="store_true",
                    help="ignore cached LLM responses for this run")
    vm.add_argument("--skip-llm", action="store_true",
                    help="stop after perception; just write the boxes JSON")

    out = p.add_argument_group("output")
    out.add_argument("--name", default="place", help="artifact / plot stem")
    out.add_argument("--artifact-dir", default="data/value_maps")
    out.add_argument("--out-dir", default="outputs/value_maps")
    out.add_argument("--quality", default="medium",
                     choices=["low", "medium", "high", "best"])
    out.add_argument("--show", action="store_true", help="open plots in a browser")
    return p.parse_args()


ENV_KEY = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def _require_api_key(provider: str) -> None:
    """Fail early and say exactly what to export."""
    import os

    var = ENV_KEY[provider]
    if not os.environ.get(var):
        raise SystemExit(
            f"{var} is not set — the {provider} SDK reads the key from the "
            f"environment.\n  export {var}=...    (add it to ~/.zshrc to persist)"
        )


def _xyz(text: str | None) -> np.ndarray | None:
    if text is None:
        return None
    parts = [float(v) for v in text.replace(" ", "").split(",")]
    if len(parts) != 3:
        raise ValueError(f"expected 'x,y,z', got {text!r}")
    return np.asarray(parts, dtype=np.float32)


# ---------------------------------------------------------------------------
# Perception
# ---------------------------------------------------------------------------

def capture_boxes(args, cfg):
    """Segment the requested objects, fit world-frame boxes, keep the cloud.

    Returns (boxes, (points, colors), rgb) — the cloud and the snapshot both
    come from the same frame the boxes were fitted from, so a box that looks
    wrong in the plot is wrong against the very data that produced it.
    """
    classes = [c.strip() for c in args.objects.split(",") if c.strip()]
    if not classes:
        raise SystemExit(
            "--objects is required for --source svo/live (the list of things "
            "SAM 3 should segment for)"
        )

    cam_cfg = cfg.cameras[args.camera]
    T_cam2base = load_T_cam2base(cam_cfg.extrinsics_path)
    logger.info("extrinsics: %s", cam_cfg.extrinsics_path)

    checkpoint = args.sam3_checkpoint or cfg.get("perception", {}).get(
        "sam3_checkpoint", None
    )
    if not checkpoint:
        raise SystemExit(
            "no SAM 3 checkpoint: pass --sam3-checkpoint or set "
            "perception.sam3_checkpoint in the config"
        )
    confidence = (
        args.confidence
        if args.confidence is not None
        else float(cfg.get("perception", {}).get("confidence", 0.5))
    )
    bpe_path = args.bpe_path or cfg.get("perception", {}).get("bpe_path", None)

    camera = None
    if args.source == "svo":
        source = SvoSource(
            args.svo,
            depth_mode=str(cfg.get("perception", {}).get("depth_mode", "NEURAL")),
        )
    else:
        from clear_franka.camera import make_zed_camera

        camera = make_zed_camera(cfg, args.camera)
        source = ZedLiveSource(camera)

    try:
        frames = collect_frames(source, args.frames, skip=args.skip_frames)
        logger.info("captured %d frame(s) from %s", len(frames), args.source)
        segmenter = Sam3Segmenter(
            checkpoint, classes, confidence=confidence, bpe_path=bpe_path
        )
        per_frame = [(segmenter.detect(f.rgb), f.xyz) for f in frames]
        K = source.intrinsics
    finally:
        source.close()
        if camera is not None:
            camera.close()

    boxes = detections_to_scene_boxes(
        per_frame, T_cam2base, max_per_label=args.max_per_label
    )
    if not boxes:
        raise SystemExit(
            f"no objects detected for {classes} — lower --confidence, check the "
            "camera view, or rename the prompts"
        )
    save_scene_boxes(args.boxes_out, boxes)

    cloud = frame_to_world_cloud(
        frames[-1].xyz, frames[-1].rgb, T_cam2base, max_points=args.cloud_points
    )
    save_scene_cloud(args.cloud_out, *cloud)

    # Label the last frame with each box's own SAM footprint (not a reprojected
    # 3D box) so the planner can tell which physical object in the photo is
    # 'bowl' vs 'bowl_2' -- the box table's world-frame numbers alone give it
    # no way to make that correspondence.
    matches = match_boxes_to_frame_detections(
        boxes, per_frame[-1][0], args.max_per_label
    )
    labeled_image = label_scene_image(frames[-1].rgb, matches)
    save_scene_image(args.image_out, labeled_image)

    if args.viser:
        _serve_viser(cloud, boxes, rgb=frames[-1].rgb, K=K, T_cam2base=T_cam2base)
    return boxes, cloud, labeled_image


def _serve_viser(cloud, boxes, rgb=None, K=None, T_cam2base=None) -> None:
    """Show the scene cloud and the fitted boxes, both in the base frame.

    When `rgb`/`K`/`T_cam2base` are given, also drop a textured camera
    frustum at the shot's pose so the RGB frame the boxes were fit from can
    be checked against the cloud/boxes directly, in place.
    """
    import viser
    import viser.transforms as vtf

    from clear_franka.perception import aabb_edges

    world, colors = cloud
    server = viser.ViserServer()
    server.scene.add_point_cloud("/cloud", world, colors, point_size=0.005)
    for i, b in enumerate(boxes):
        lo, hi = b.aabb
        server.scene.add_line_segments(
            f"/box_{i}", aabb_edges(lo, hi), (255, 87, 51), line_width=3.0
        )
        server.scene.add_label(f"/box_{i}_label", b.name,
                               position=(lo[0], lo[1], hi[2]))

    if rgb is not None and K is not None and T_cam2base is not None:
        h, w = rgb.shape[:2]
        fov_y = 2.0 * np.arctan2(h / 2.0, K[1, 1])
        server.scene.add_camera_frustum(
            "/camera",
            fov=fov_y,
            aspect=w / h,
            image=rgb,
            wxyz=vtf.SO3.from_matrix(T_cam2base[:3, :3]).wxyz,
            position=T_cam2base[:3, 3],
            scale=0.15,
        )

    input("viser serving the scene in the base frame — press Enter to continue... ")


# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    _utf8_stdout()
    args = _parse_args()
    cfg = OmegaConf.load(args.config)

    # Resolve and validate the LLM side first: capturing a scene loads a 3.4GB
    # checkpoint, and there is no reason to pay for that before finding out the
    # task is missing or the key is not exported.
    llm = None
    if not args.skip_llm:
        if not args.task:
            raise SystemExit("--task is required unless --skip-llm is passed")
        llm_cfg = dict(cfg.get("value_map_llm", {}) or {})
        llm = {
            "provider": args.provider or llm_cfg.get("provider", "anthropic"),
            "model": args.model or llm_cfg.get("model", "claude-opus-5"),
            "cache_dir": args.cache_dir or llm_cfg.get(
                "cache_dir", "cache/value_map_llm"),
            "load_cache": not args.no_cache,
        }
        _require_api_key(llm["provider"])
        logger.info("planner/map LLM: %s/%s", llm["provider"], llm["model"])

    if args.source == "boxes":
        boxes_path = args.boxes_in or args.boxes_out
        boxes = load_scene_boxes(boxes_path)
        logger.info("loaded %d box(es) from %s", len(boxes), boxes_path)
        for b in boxes:
            logger.info("  %-20s center=%s size=%s", b.name,
                        np.array2string(b.center, precision=3),
                        np.array2string(b.size, precision=3))
        cloud = load_scene_cloud(args.cloud_out)
        if cloud is None:
            logger.info("no cached cloud at %s — plots will show boxes only",
                        args.cloud_out)
        workspace_image = load_scene_image(args.image_out)
        if workspace_image is None:
            logger.info("no cached snapshot at %s — planner runs text-only",
                        args.image_out)
    else:
        boxes, cloud, workspace_image = capture_boxes(args, cfg)
        boxes_path = args.boxes_out

    if args.skip_llm:
        print(f"\nBoxes written to {boxes_path} (--skip-llm; no value maps built)")
        return 0

    ws_min, ws_max = _xyz(args.ws_min), _xyz(args.ws_max)
    if ws_min is None or ws_max is None:
        ws_min, ws_max = workspace_bounds_from_boxes(boxes, args.bounds_margin)
    logger.info("grid: %d^3 over [%s, %s]", args.map_size,
                np.array2string(ws_min, precision=2),
                np.array2string(ws_max, precision=2))

    stages = synthesize_value_maps(
        args.task,
        boxes,
        workspace_bounds_min=ws_min,
        workspace_bounds_max=ws_max,
        map_size=args.map_size,
        avoidance_weight=args.avoidance_weight,
        obstacle_sigma=args.obstacle_sigma,
        workspace_image=workspace_image,
        llm=llm,
    )

    artifact = save_stages(
        Path(args.artifact_dir) / f"{args.name}.npz",
        stages,
        task=args.task,
        boxes_path=str(boxes_path),
        avoidance_weight=args.avoidance_weight,
        obstacle_sigma=args.obstacle_sigma,
    )

    overlays = [box_overlay(b.name, b.center, b.size) for b in boxes]
    for i, st in enumerate(stages):
        print(f"\n=== stage {i}: {st.label} ===")
        print(f"  affordance : {st.affordance_query}")
        print(f"  avoidance  : {st.avoidance_query}")
        print(f"  target     : {np.round(st.target_world, 3)}  "
              f"(arrival {st.arrival_radius_m:.3f}m)")
        print_gradient_probes(st.value_map, st.target_world)
        render(
            st.value_map,
            out_dir=args.out_dir,
            filename=f"{args.name}_stage{i}",
            objects=overlays + [
                box_overlay(f"{st.label}_target", st.target_world,
                            np.array([0.03, 0.03, 0.03]))
            ],
            scene_points=cloud[0] if cloud is not None else None,
            scene_colors=cloud[1] if cloud is not None else None,
            quality=args.quality,
            also_latest=(i == len(stages) - 1),
            show=args.show,
        )

    print(f"\nArtifact : {artifact}")
    print(f"Plots    : {Path(args.out_dir)}/{args.name}_stage*.html")
    print("\nTo steer with it:")
    print("  deploy.steering.place_mode=llm "
          f"deploy.steering.llm.artifact_path={artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
