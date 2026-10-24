"""Fit a robust similarity from predicted camera centers to mesh-space anchors."""

import numpy as np
import torch


def fit_similarity(source, target):
    """Fit target = scale * rotation @ source + translation with a proper rotation."""
    source, target = np.asarray(source, float), np.asarray(target, float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("Similarity alignment requires matching [N,3] points, N >= 3")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Camera centers must be finite")
    source_mean, target_mean = source.mean(0), target.mean(0)
    x, y = source - source_mean, target - target_mean
    variance = (x**2).sum() / len(x)
    if variance <= 0:
        raise ValueError("Camera centers do not determine a scale")
    u, singular, vt = np.linalg.svd(y.T @ x / len(x))
    signs = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        signs[2, 2] = -1
    rotation = u @ signs @ vt
    scale = np.trace(np.diag(singular) @ signs) / variance
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def align_camera_centers(source, target, *, sample_size, threshold, iterations=5000, seed=0):
    """Fit Sim(3) with deterministic RANSAC, then refit on the best inliers.

    Score by inlier count and break ties by their RMS distance. Return the 4x4
    source-to-target matrix, winning hypothesis mask and refitted residuals.
    """
    source, target = np.asarray(source, float), np.asarray(target, float)
    if source.shape != target.shape or not 3 <= sample_size <= len(source):
        raise ValueError("Insufficient matching anchor camera centers")
    rng = np.random.default_rng(seed)
    best_inliers, best_score, best_rms = None, -1, np.inf
    for _ in range(iterations):
        indices = rng.choice(np.arange(len(source)), size=sample_size, replace=False)
        try:
            scale, rotation, translation = fit_similarity(source[indices], target[indices])
        except (np.linalg.LinAlgError, ValueError):
            continue
        predicted = (scale * (rotation @ source.T)).T + translation
        residuals = np.linalg.norm(target - predicted, axis=1)
        inliers = residuals < threshold
        score = int(inliers.sum())
        if score == 0:
            continue
        rms = np.sqrt(np.mean(residuals[inliers]**2))
        if score > best_score or (score == best_score and rms < best_rms):
            best_inliers, best_score, best_rms = inliers, score, rms
    if best_inliers is None or best_score < 3:
        raise RuntimeError("Camera alignment failed to find enough consistent anchors")
    scale, rotation, translation = fit_similarity(source[best_inliers], target[best_inliers])
    residuals = np.linalg.norm(target - (scale * (rotation @ source.T)).T - translation, axis=1)
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = scale * rotation, translation
    return matrix, best_inliers, residuals


def remove_rotation_scale(extrinsics):
    """Project each 3x3 block to SO(3), preserving the translation column."""
    result = np.asarray(extrinsics).copy()
    u, _, vt = np.linalg.svd(result[..., :3, :3], full_matrices=False)
    signs = np.zeros_like(result[..., :3, :3])
    signs[..., 0, 0] = signs[..., 1, 1] = 1
    signs[..., 2, 2] = np.sign(np.linalg.det(u @ vt))
    result[..., :3, :3] = u @ signs @ vt
    return result


def align_extrinsics(extrinsics, similarity):
    """Map camera-to-world poses through Sim(3) and return rigid extrinsics.

    Transform camera centers with the full similarity. Project the transformed
    orientation to SO(3) in FP32, then invert to world-to-camera convention.
    """
    poses = similarity @ torch.linalg.inv(extrinsics)
    poses = remove_rotation_scale(poses.float().cpu().numpy())
    return torch.linalg.inv(torch.from_numpy(poses).to(extrinsics))
