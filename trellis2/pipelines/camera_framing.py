"""Center estimated-camera canvases without changing mesh geometry or poses."""

import numpy as np


def center_camera_frames(scene, intrinsics, extrinsics):
    """Return normalized intrinsics containing the mesh with a 5% border.

    Apply one image-plane scale and translation to each calibration. Keep the
    original scale when it fits, and never alter world-to-camera extrinsics.
    If the mesh crosses a camera's near plane, retain that calibration and
    mark the view unusable for reference-view selection.
    """
    if len(intrinsics) != len(extrinsics):
        raise ValueError("Camera calibration and pose lists must have matching lengths")
    vertices = []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        mesh_vertices = np.asarray(scene.geometry[name].vertices)
        vertices.append(mesh_vertices @ transform[:3, :3].T + transform[:3, 3])
    if not vertices or not any(len(value) for value in vertices):
        raise ValueError("Camera framing requires mesh vertices")
    vertices = np.concatenate(vertices)
    if not np.isfinite(vertices).all():
        raise ValueError("Camera framing requires finite mesh vertices")
    calibrations, records = [], []
    for calibration, extrinsic in zip(intrinsics, extrinsics):
        calibration = np.asarray(calibration, dtype=np.float64).copy()
        extrinsic = np.asarray(extrinsic, dtype=np.float64)
        record = {"applied": False, "original_intrinsics": calibration.tolist()}
        camera = vertices @ extrinsic[:3, :3].T + extrinsic[:3, 3]
        if not np.isfinite(camera).all() or np.any(camera[:, 2] <= .01):
            record["reason"] = "The mesh crosses or lies behind the camera near plane"
        else:
            projected = camera @ calibration.T
            points = projected[:, :2] / projected[:, 2:3]
            lower, upper = points.min(0), points.max(0)
            extent = float((upper - lower).max())
            if not np.isfinite(points).all() or extent <= 0:
                record["reason"] = "The projected mesh has no finite spatial extent"
            else:
                scale = min(1.0, .9 / extent)
                translation = .5 - scale * (lower + upper) / 2
                transform = np.array([[scale, 0, translation[0]],
                                      [0, scale, translation[1]], [0, 0, 1]])
                calibration = transform @ calibration
                record.update(applied=True, reason=None,
                              image_plane_transform=transform.tolist(),
                              projected_bounds_before=[lower.tolist(), upper.tolist()],
                              projected_bounds_after=[(scale * lower + translation).tolist(),
                                                      (scale * upper + translation).tolist()])
        calibrations.append(calibration)
        records.append(record)
    return calibrations, records
