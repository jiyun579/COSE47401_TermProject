"""Temporal experiments for MotionAGFormer (N, T, J, 3) predictions.

Both functions operate on the same root-relative, normalized 3D coordinates
used by the stock MotionAGFormer training loss and evaluation code.
"""

from __future__ import annotations

import numpy as np
import torch


def acceleration_consistency_loss(
    predicted: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Mean joint Euclidean error of per-frame second differences.

    For every joint: d2P[t] = P[t+2] - 2*P[t+1] + P[t].
    This matches the discrete second-difference error in loss/pose3d.py
    (acc_error), but is differentiable for training. Units are normalized
    coordinate/frame^2, not m/s^2.

    Shape: (batch, frames, joints, 3). No smoothing toward zero acceleration:
    high true acceleration is preserved when the prediction matches the GT.
    """
    if predicted.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: predicted={tuple(predicted.shape)}, "
            f"target={tuple(target.shape)}"
        )
    if predicted.ndim != 4 or predicted.shape[-1] != 3:
        raise ValueError("Expected tensors of shape (batch, frames, joints, 3)")
    if predicted.shape[1] < 3:
        return predicted.sum() * 0.0

    pred_acc = predicted[:, 2:] - 2.0 * predicted[:, 1:-1] + predicted[:, :-2]
    gt_acc = target[:, 2:] - 2.0 * target[:, 1:-1] + target[:, :-2]
    return torch.linalg.vector_norm(pred_acc - gt_acc, dim=-1).mean()


def savgol_pose_smooth(
    pose: np.ndarray,
    *,
    window_length: int = 9,
    polyorder: int = 3,
) -> np.ndarray:
    """Smooth *predictions only* along the temporal axis independently by clip.

    Shape: (N, T, J, 3). SG mode='interp' avoids zero padding on boundaries.
    window_length=0 disables smoothing (for baseline-versus-SG ablation).
    The function does not mutate the input.
    """
    pose = np.asarray(pose)
    if pose.ndim != 4 or pose.shape[-1] != 3:
        raise ValueError("Expected poses of shape (clips, frames, joints, 3)")
    if not np.issubdtype(pose.dtype, np.floating):
        raise TypeError("Pose coordinates must have a floating-point dtype")
    if window_length == 0:
        return pose.copy()
    if not isinstance(window_length, int) or window_length <= 0 or window_length % 2 == 0:
        raise ValueError("window_length must be an odd positive integer, or 0")
    if not isinstance(polyorder, int) or polyorder < 0 or polyorder >= window_length:
        raise ValueError("polyorder must be a nonnegative integer < window_length")
    if window_length > pose.shape[1]:
        raise ValueError("window_length must not exceed the clip length")

    # Import here so normal training with the acceleration loss doesn't
    # require SciPy just to import this module.
    from scipy.signal import savgol_filter

    return savgol_filter(
        pose,
        window_length=window_length,
        polyorder=polyorder,
        axis=1,
        mode="interp",
    ).astype(pose.dtype, copy=False)
