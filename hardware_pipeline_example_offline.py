"""ZED SVO -> SAM 3 text-prompted masks -> 3D boxes in viser."""

import time
from dataclasses import dataclass

import numpy as np
import pyzed.sl as sl
import torch
import viser
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

SVO_PATH = "data/test_video.svo2"
SAM3_CHECKPOINT = "/home/aryannav/Downloads/sam3.pt"
CLASSES = ["table"]

DEPTH_MODE = sl.DEPTH_MODE.NEURAL  # drop to NEURAL_LIGHT if the GPU runs out of memory
DETECT_EVERY = 15  # frames between SAM 3 passes
STRIDE = 2  # point cloud downsample for the browser
CONFIDENCE = 0.5  # minimum detection score
TRIM_PCT = 2.0  # percentile trimmed off each end before fitting a box

PALETTE = [(255, 87, 51), (46, 204, 113), (52, 152, 219), (241, 196, 15), (155, 89, 182)]


@dataclass
class Frame:
    rgb: np.ndarray  # (H, W, 3) uint8
    xyz: np.ndarray  # (H, W, 3) float32, meters


@dataclass
class Detection:
    label: str
    score: float
    mask: np.ndarray  # (H, W) bool
    color: tuple[int, int, int]


class SvoReader:
    """Plays an SVO recording, looping forever."""

    def __init__(self, path):
        init = sl.InitParameters()
        init.set_from_svo_file(path)
        init.coordinate_units = sl.UNIT.METER
        init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP  # matches viser
        init.depth_mode = DEPTH_MODE

        self.cam = sl.Camera()
        self.cam.open(init)
        self._image = sl.Mat()
        self._cloud = sl.Mat()

    def __iter__(self):
        while True:
            if self.cam.grab() != sl.ERROR_CODE.SUCCESS:
                self.cam.set_svo_position(0)
                continue

            self.cam.retrieve_image(self._image, sl.VIEW.LEFT)
            self.cam.retrieve_measure(self._cloud, sl.MEASURE.XYZRGBA)
            yield Frame(
                rgb=self._image.get_data()[..., [2, 1, 0]].copy(),  # BGRA -> RGB
                xyz=self._cloud.get_data()[..., :3].copy(),
            )

    def close(self):
        self.cam.close()


class Sam3Segmenter:
    """Segments one image against a fixed list of text prompts."""

    def __init__(self, checkpoint, classes):
        model = build_sam3_image_model(checkpoint_path=checkpoint, load_from_HF=False)
        self.processor = Sam3Processor(model, confidence_threshold=CONFIDENCE)
        self.classes = classes

    def detect(self, rgb):
        detections = []
        # The vision backbone runs once here; each prompt below only re-runs the
        # text encoder and grounding head.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.processor.set_image(Image.fromarray(rgb))

            for i, label in enumerate(self.classes):
                out = self.processor.set_text_prompt(prompt=label, state=state)
                # set_text_prompt overwrites state in place, so copy out now.
                masks = out["masks"][:, 0].cpu().numpy()
                scores = out["scores"].float().cpu().numpy()
                detections += [
                    Detection(label, s, m, PALETTE[i % len(PALETTE)])
                    for m, s in zip(masks, scores)
                ]
        return detections


def mask_to_aabb(xyz, mask):
    """Backproject a 2D mask onto the cloud and fit an axis-aligned box."""
    points = xyz[mask]
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 10:
        return None
    return (
        np.percentile(points, TRIM_PCT, axis=0),
        np.percentile(points, 100 - TRIM_PCT, axis=0),
    )


def aabb_edges(lo, hi):
    """The 12 edges of a box, as (12, 2, 3) line segments."""
    corners = np.array(np.meshgrid(*zip(lo, hi), indexing="ij")).reshape(3, -1).T
    return np.array(
        [
            [a, b]
            for i, a in enumerate(corners)
            for b in corners[i + 1 :]
            if np.count_nonzero(~np.isclose(a, b)) == 1  # differ on exactly one axis
        ]
    )


class SceneView:
    """Draws the cloud and the detected boxes in viser."""

    def __init__(self):
        self.server = viser.ViserServer()
        self._box_handles = []

    def show_cloud(self, frame, detections):
        xyz = frame.xyz[::STRIDE, ::STRIDE]
        rgb = frame.rgb[::STRIDE, ::STRIDE].copy()
        for det in detections:  # tint the masked pixels so the mapping is visible
            rgb[det.mask[::STRIDE, ::STRIDE]] = det.color

        xyz, rgb = xyz.reshape(-1, 3), rgb.reshape(-1, 3)
        valid = np.isfinite(xyz).all(axis=1)
        self.server.scene.add_point_cloud(
            "/cloud", xyz[valid], rgb[valid], point_size=0.01
        )

    def show_detections(self, frame, detections):
        for handle in self._box_handles:  # clear boxes from the previous pass
            handle.remove()
        self._box_handles = []

        for i, det in enumerate(detections):
            box = mask_to_aabb(frame.xyz, det.mask)
            if box is None:
                continue
            lo, hi = box
            self._box_handles += [
                self.server.scene.add_line_segments(
                    f"/box_{i}", aabb_edges(lo, hi), det.color, thickness=0.004
                ),
                self.server.scene.add_label(
                    f"/box_{i}_label",
                    f"{det.label} {det.score:.2f}",
                    position=(lo[0], lo[1], hi[2]),
                ),
            ]
            print(f"  {det.label:12} {det.score:.3f}  size={np.round(hi - lo, 3)}")


def main():
    reader = SvoReader(SVO_PATH)
    segmenter = Sam3Segmenter(SAM3_CHECKPOINT, CLASSES)
    view = SceneView()

    detections = []
    for i, frame in enumerate(reader):
        if i % DETECT_EVERY == 0:
            print(f"frame {i}:")
            detections = segmenter.detect(frame.rgb)
            view.show_detections(frame, detections)
        view.show_cloud(frame, detections)
        time.sleep(1 / 30)


if __name__ == "__main__":
    main()