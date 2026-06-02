"""Preview exactly what the DiffuserActor image encoder will be fed.

Grabs one frame from the hand and third-person ZEDs, runs the SAME preprocessing
the deploy harness uses (CameraPreprocessor.process: center-square crop 720x1280
-> 720x720 -> resize 200x200), then applies the SAME model crop the policy applies
(DiffuserActorBasePolicy._prepare_rgb: [20:180, 20:180] -> 160x160 when
crop_images is set), and writes a montage PNG to data_dir.

Run:  python preview_model_input.py
No robot motion; cameras are grabbed synchronously (run() is not needed).
"""

import time
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    from deploy_diffuser_actor import _setup_cameras
    from clear_franka.diffuser_actor_io import save_model_input_preview

    cam_hand, cam_tp, pre_hand, pre_tp = _setup_cameras(cfg)
    try:
        tp_frame = cam_tp.grab_frame()
        hand_frame = cam_hand.grab_frame()
        if tp_frame is None or hand_frame is None:
            raise RuntimeError("Camera grab failed (got None) — check the ZEDs.")
        rgb_tp_full, depth_tp_full = tp_frame
        rgb_hand_full, depth_hand_full = hand_frame

        # The RGB crop is independent of the hand camera's base transform, so an
        # identity T_gripper2base is fine here (we only use the RGB output).
        rgb_tp_200, _ = pre_tp.process(rgb_tp_full, depth_tp_full)
        rgb_hand_200, _ = pre_hand.process(rgb_hand_full, depth_hand_full, np.eye(4))

        policy_cfg = OmegaConf.load(cfg.deploy.policy_config)
        crop_images = bool(policy_cfg.get("crop_images", True))

        out_path = Path(cfg.data_dir) / f"model_input_preview_{int(time.time())}.png"
        save_model_input_preview(
            rows=[
                ("front (third_person)", rgb_tp_full, rgb_tp_200),
                ("wrist (hand)", rgb_hand_full, rgb_hand_200),
            ],
            out_path=out_path,
            crop_images=crop_images,
        )
        n = 160 if crop_images else 200
        print(f"Saved model-input preview to {out_path}")
        print(f"  Encoder input per camera: {n}x{n}  (crop_images={crop_images})")
        print("  Left=raw center-crop (FOV the model gets), "
              "middle=200x200 with model-crop box, right=final encoder input.")
    finally:
        cam_hand.close()
        cam_tp.close()


if __name__ == "__main__":
    main()
