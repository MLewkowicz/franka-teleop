"""Geometry and transform helpers for calibration and visualization."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def average_transforms(Ts: list[np.ndarray]) -> np.ndarray:
    """Average SE(3) samples with quaternion mean and arithmetic translation."""
    quats = np.array([R.from_matrix(T[:3, :3]).as_quat() for T in Ts])
    flip = np.sign(quats @ quats[0])
    flip[flip == 0] = 1
    quats = quats * flip[:, None]
    mean_quat = quats.mean(axis=0)
    mean_quat /= np.linalg.norm(mean_quat)

    T_mean = np.eye(4)
    T_mean[:3, :3] = R.from_quat(mean_quat).as_matrix()
    T_mean[:3, 3] = np.mean([T[:3, 3] for T in Ts], axis=0)
    return T_mean


def load_T_cam2base(extrinsics_path: str | Path) -> np.ndarray:
    with Path(extrinsics_path).expanduser().open("r") as f:
        payload = json.load(f)
    T_cam2base = np.asarray(payload["T_cam2base"], dtype=float)
    if T_cam2base.shape != (4, 4):
        raise ValueError(f"Expected T_cam2base to be 4x4, got {T_cam2base.shape}")
    return T_cam2base


def load_T_cam2gripper(extrinsics_path: str | Path) -> np.ndarray:
    with Path(extrinsics_path).expanduser().open("r") as f:
        payload = json.load(f)
    if "T_cam2gripper" in payload:
        T_cam2gripper = np.asarray(payload["T_cam2gripper"], dtype=float)
    elif "T_gripper2cam" in payload:
        T_cam2gripper = invert_transform(np.asarray(payload["T_gripper2cam"], dtype=float))
    else:
        raise KeyError(f"{extrinsics_path} does not contain T_cam2gripper or T_gripper2cam")
    if T_cam2gripper.shape != (4, 4):
        raise ValueError(f"Expected T_cam2gripper to be 4x4, got {T_cam2gripper.shape}")
    return T_cam2gripper
