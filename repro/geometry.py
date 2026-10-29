"""Deterministic point-cloud views and exact point-to-pixel bookkeeping.

This is an independent renderer for the project. It is not the renderer used in
the companion report, whose source was not supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True)
class View:
    angles: tuple[int, int, int]
    rotation: np.ndarray


def rotation_matrix(ax: float, ay: float, az: float) -> np.ndarray:
    """Active xyz rotations in degrees, applied in x then y then z order."""
    x, y, z = np.deg2rad([ax, ay, az])
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]], dtype=np.float64)
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]], dtype=np.float64)
    rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def local_views() -> list[View]:
    """The 27 local rotations used for training; inference uses a fixed 15."""
    angles = [(0, 0, 0)] + [a for a in product((-15, 0, 15), repeat=3)
                           if a != (0, 0, 0)]
    return [View(a, rotation_matrix(*a)) for a in angles]


def inference_view_indices() -> list[int]:
    """Spread 15 views over the 27 local rotations, always including front."""
    views = local_views()
    selected = [(0, 0, 0)]
    # Greedy farthest-angle selection is deterministic and avoids 15 near copies.
    remaining = [v.angles for v in views[1:]]
    while len(selected) < 15:
        choice = max(remaining, key=lambda a: (min(sum((a[i] - b[i]) ** 2
                                                        for i in range(3))
                                                   for b in selected), a))
        selected.append(choice)
        remaining.remove(choice)
    return [next(i for i, v in enumerate(views) if v.angles == a)
            for a in selected]


def category_basis(normal_prototype: np.ndarray) -> np.ndarray:
    """PCA basis from one normal object, with deterministic axis signs."""
    centered = normal_prototype - normal_prototype.mean(axis=0)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    values, vectors = np.linalg.eigh(covariance)
    basis = vectors[:, np.argsort(values)[::-1]]
    for column in range(3):
        axis = basis[:, column]
        if axis[np.argmax(np.abs(axis))] < 0:
            basis[:, column] *= -1
    if np.linalg.det(basis) < 0:
        basis[:, 2] *= -1
    return basis


def normalize_points(points: np.ndarray, basis: np.ndarray) -> np.ndarray:
    centered = np.asarray(points, dtype=np.float64) - points.mean(axis=0)
    transformed = centered @ basis
    radius = np.linalg.norm(transformed, axis=1).max()
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Point cloud has zero or invalid radius")
    return (transformed / radius).astype(np.float32)


def project(points: np.ndarray, rotation: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthographic projection: pixel xy, camera depth, in-frame flag."""
    camera = np.asarray(points) @ rotation.T
    uv = np.empty((len(points), 2), dtype=np.float32)
    uv[:, 0] = (camera[:, 0] * 0.94 + 1.0) * 0.5 * (size - 1)
    uv[:, 1] = (1.0 - camera[:, 1] * 0.94) * 0.5 * (size - 1)
    in_frame = ((uv[:, 0] >= 0) & (uv[:, 0] < size) &
                (uv[:, 1] >= 0) & (uv[:, 1] < size))
    return uv, camera[:, 2].astype(np.float32), in_frame


def render(points: np.ndarray, rotation: np.ndarray, size: int = 224,
           visibility_tolerance: float = 0.012) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return RGB grayscale render, projected xy, visible flags and z winners.

    Brightness encodes camera depth on a black background. The depth buffer
    selects the frontmost raw point in each pixel; all point identities remain
    available for the correspondence and visibility calculation.
    """
    uv, depth, in_frame = project(points, rotation, size)
    xy = np.floor(uv).astype(np.int32)
    flat = xy[:, 1] * size + xy[:, 0]
    valid = np.flatnonzero(in_frame)
    if len(valid) == 0:
        raise ValueError("No points fall within the projected image")
    # Primary sort is pixel, secondary sort is descending depth, tertiary raw ID.
    order = np.lexsort((valid, -depth[valid], flat[valid]))
    sorted_ids = valid[order]
    sorted_pixels = flat[sorted_ids]
    first = np.r_[True, sorted_pixels[1:] != sorted_pixels[:-1]]
    winners = sorted_ids[first]
    depth_buffer = np.full(size * size, -np.inf, dtype=np.float32)
    depth_buffer[flat[winners]] = depth[winners]

    visible = np.zeros(len(points), dtype=bool)
    visible[valid] = (depth[valid] >= depth_buffer[flat[valid]] -
                      visibility_tolerance)
    # Monotone depth shading gives a reproducible 3-channel grayscale input.
    z = depth[winners]
    lo, hi = np.percentile(z, [2, 98])
    scale = max(float(hi - lo), 1e-6)
    gray = np.zeros(size * size, dtype=np.uint8)
    gray[flat[winners]] = np.rint(75 + 175 * np.clip((z - lo) / scale, 0, 1)).astype(np.uint8)
    rgb = np.repeat(gray.reshape(size, size, 1), 3, axis=2)
    return rgb, uv, visible, winners


def front_reference_indices(points: np.ndarray, rotation: np.ndarray,
                            grid_size: int = 64) -> np.ndarray:
    """One exact raw-point representative per occupied coarse front pixel."""
    _, _, _, winners = render(points, rotation, size=grid_size)
    return np.sort(winners)
