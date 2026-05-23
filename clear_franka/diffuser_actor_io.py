"""Glue between live ZED frames + Franka state and the 3D Diffuser Actor obs.

Mirrors the geometry done at training time in
LangSteer/scripts/convert_realworld_for_diffuser_actor.py so the live
observation feeds the policy data with the same coordinate frames, image
resolution, and unit conventions it was trained on.

Per camera, the conversion is:
    1. center-square crop the (720, 1280) ZED frame to 720×720
    2. resize that crop to 200×200 (RGB INTER_AREA, depth INTER_NEAREST)
    3. unproject depth → camera-frame XYZ via the adjusted intrinsics
    4. transform to base frame:
         third-person: T_cam2base (constant, from extrinsics_third_person.json)
         hand:         T_cam2gripper (constant)  ∘  T_gripper2base(t)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


IMG_SIZE = 200  # CalvinDataset crops [:, 20:180, 20:180] → 160×160 at train time


# ---------------------------------------------------------------------------
# Extrinsics loading
# ---------------------------------------------------------------------------

def load_extrinsics_json(path: Path | str) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Image / depth processing
# ---------------------------------------------------------------------------

def center_square_crop_resize(img: np.ndarray, out_size: int,
                              interp: int) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Center-crop to a square, resize to (out_size, out_size).

    Returns (resized, (crop_x, crop_y, crop_side)) so the intrinsics can be
    adjusted to match the same crop+resize.
    """
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = img[y0:y0 + side, x0:x0 + side]
    resized = cv2.resize(crop, (out_size, out_size), interpolation=interp)
    return resized, (x0, y0, side)


def adjust_K_for_crop_resize(K: np.ndarray, crop_x: int, crop_y: int,
                              crop_size: int, out_size: int) -> np.ndarray:
    """Scale a 3×3 intrinsic for the center-crop + isotropic resize above."""
    K = K.copy()
    K[0, 2] -= crop_x
    K[1, 2] -= crop_y
    s = out_size / crop_size
    K[0, 0] *= s
    K[1, 1] *= s
    K[0, 2] *= s
    K[1, 2] *= s
    return K


def depth_to_camera_xyz(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Unproject an HxW depth map (meters, NaN allowed) using intrinsics K.

    Returns (H, W, 3) XYZ in the camera frame. NaNs become 0 (camera origin).
    """
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64),
                       np.arange(h, dtype=np.float64))
    z = np.nan_to_num(depth.astype(np.float64), nan=0.0,
                      posinf=0.0, neginf=0.0)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def transform_xyz(xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4×4 homogeneous transform to an (H, W, 3) point cloud."""
    h, w, _ = xyz.shape
    flat = xyz.reshape(-1, 3)
    hom = np.concatenate([flat, np.ones((flat.shape[0], 1))], axis=1)
    out = (T @ hom.T).T[:, :3]
    return out.reshape(h, w, 3)


def make_T_gripper_to_base(ee_pos: np.ndarray, ee_rot: np.ndarray) -> np.ndarray:
    """4×4 transform from gripper (end-effector) frame to base frame."""
    T = np.eye(4)
    T[:3, :3] = ee_rot
    T[:3, 3] = ee_pos
    return T


# ---------------------------------------------------------------------------
# Pre-computed per-camera state
# ---------------------------------------------------------------------------

class CameraPreprocessor:
    """Caches the adjusted intrinsics + (for the static cam) the fixed T_cam2base.

    Designed for the inference inner loop: one allocation up front, then
    `process(rgb, depth, T_gripper2base)` per tick produces the 200×200 RGB
    (uint8 RGB, not normalised) and the 200×200×3 base-frame XYZ.
    """

    def __init__(self, extrinsics: dict, mount: str,
                 raw_h: int = 720, raw_w: int = 1280, out_size: int = IMG_SIZE):
        assert mount in ("hand", "third_person")
        self.mount = mount
        self.out_size = out_size

        K_raw = np.asarray(extrinsics["intrinsics"]["K"], dtype=np.float64)
        crop_x = (raw_w - raw_h) // 2
        crop_y = 0
        side = raw_h
        self.K = adjust_K_for_crop_resize(K_raw, crop_x, crop_y, side, out_size)
        self._crop_x = crop_x
        self._crop_y = crop_y
        self._side = side

        if mount == "third_person":
            self.T_cam2base = np.asarray(extrinsics["T_cam2base"], dtype=np.float64)
            self.T_cam2gripper = None
        else:
            self.T_cam2gripper = np.asarray(extrinsics["T_cam2gripper"], dtype=np.float64)
            self.T_cam2base = None  # populated per frame from gripper pose

    def process(self, rgb_full: np.ndarray, depth_full: np.ndarray,
                T_gripper2base: Optional[np.ndarray] = None
                ) -> tuple[np.ndarray, np.ndarray]:
        """Return (rgb_200_uint8, xyz_200_base_frame_float32).

        For the hand camera you must pass the current T_gripper2base; for the
        third-person camera it is ignored.
        """
        rgb_200, _ = center_square_crop_resize(rgb_full, self.out_size,
                                                cv2.INTER_AREA)
        depth_200, _ = center_square_crop_resize(depth_full, self.out_size,
                                                  cv2.INTER_NEAREST)
        cam_xyz = depth_to_camera_xyz(depth_200, self.K)
        if self.mount == "third_person":
            base_xyz = transform_xyz(cam_xyz, self.T_cam2base)
        else:
            if T_gripper2base is None:
                raise ValueError("hand camera needs the current T_gripper2base")
            gripper_xyz = transform_xyz(cam_xyz, self.T_cam2gripper)
            base_xyz = transform_xyz(gripper_xyz, T_gripper2base)
        return rgb_200, base_xyz.astype(np.float32)


# ---------------------------------------------------------------------------
# Action-side helpers
# ---------------------------------------------------------------------------

def ee_rot_to_euler_xyz(ee_rot: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → pytorch3d "XYZ" intrinsic Euler angles.

    Identical to what `convert_realworld_for_diffuser_actor.py` does at
    training time, so the gripper history fed to the policy uses the same
    Euler convention the model learned from.
    """
    # Local import: keeps clear_franka importable without LangSteer present
    # (LangSteer is only needed when actually instantiating the policy).
    import torch
    from training.policies.diffuser_actor.preprocessing.pytorch3d_transforms import (
        matrix_to_euler_angles,
    )

    R = torch.as_tensor(ee_rot, dtype=torch.float64)
    euler = matrix_to_euler_angles(R, "XYZ").cpu().numpy()
    return euler.astype(np.float32)


def euler_xyz_to_matrix(euler_xyz: np.ndarray) -> np.ndarray:
    """Inverse of ee_rot_to_euler_xyz — for sending Cartesian targets."""
    import torch
    from training.policies.diffuser_actor.preprocessing.pytorch3d_transforms import (
        euler_angles_to_matrix,
    )

    e = torch.as_tensor(euler_xyz, dtype=torch.float64)
    R = euler_angles_to_matrix(e, "XYZ").cpu().numpy()
    return R.astype(np.float64)
