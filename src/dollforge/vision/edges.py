from __future__ import annotations

from collections import deque

import numpy as np


def _luminance(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("RGB image expected")
    return image[..., 0] * .2126 + image[..., 1] * .7152 + image[..., 2] * .0722


def sobel_edge_map(rgb: np.ndarray) -> np.ndarray:
    """Return normalized Sobel magnitude in [0, 1].

    Percentile normalization avoids one isolated high-contrast pixel making every
    useful doll boundary look weak.
    """
    gray = _luminance(rgb)
    padded = np.pad(gray, 1, mode="edge")
    gx = (
        -padded[:-2, :-2] + padded[:-2, 2:]
        - 2 * padded[1:-1, :-2] + 2 * padded[1:-1, 2:]
        - padded[2:, :-2] + padded[2:, 2:]
    )
    gy = (
        -padded[:-2, :-2] - 2 * padded[:-2, 1:-1] - padded[:-2, 2:]
        + padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:]
    )
    magnitude = np.sqrt(gx * gx + gy * gy)
    scale = float(np.percentile(magnitude, 98.0))
    if scale <= 1e-6:
        return np.zeros_like(magnitude, dtype=np.float32)
    return np.clip(magnitude / scale, 0.0, 1.0).astype(np.float32)


def _nearest_allowed(allowed: np.ndarray, x: int, y: int) -> tuple[int, int] | None:
    height, width = allowed.shape
    x = int(np.clip(x, 0, width - 1))
    y = int(np.clip(y, 0, height - 1))
    if allowed[y, x]:
        return x, y
    ys, xs = np.where(allowed)
    if len(xs) == 0:
        return None
    distances = (xs - x) ** 2 + (ys - y) ** 2
    index = int(np.argmin(distances))
    return int(xs[index]), int(ys[index])


def edge_guided_region(
    rgb: np.ndarray,
    allowed: np.ndarray,
    seed_xy: tuple[int, int],
    edges: np.ndarray | None = None,
    edge_threshold: float = .58,
    color_tolerance: float = .38,
) -> np.ndarray:
    """Connected region constrained by semantic prior, appearance and image edges.

    The semantic prior remains the outer safety envelope. Strong gradients stop the
    flood, while color continuity prevents the region from walking through adjacent
    clothes/body parts when their boundary is visible.
    """
    image = np.asarray(rgb, dtype=np.float32)
    allowed = np.asarray(allowed, dtype=bool)
    if image.shape[:2] != allowed.shape:
        raise ValueError("Image and allowed mask dimensions differ")
    edges = sobel_edge_map(image) if edges is None else np.asarray(edges, dtype=np.float32)
    if edges.shape != allowed.shape:
        raise ValueError("Edge map and allowed mask dimensions differ")

    seed = _nearest_allowed(allowed, *seed_xy)
    result = np.zeros_like(allowed)
    if seed is None:
        return result

    sx, sy = seed
    seed_color = image[sy, sx, :3].copy()
    queue: deque[tuple[int, int]] = deque([(sx, sy)])
    visited = np.zeros_like(allowed)
    visited[sy, sx] = True
    result[sy, sx] = True
    max_color_distance = max(1e-6, color_tolerance) * 255.0 * np.sqrt(3.0)

    while queue:
        x, y = queue.popleft()
        current = image[y, x, :3]
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if nx < 0 or ny < 0 or nx >= allowed.shape[1] or ny >= allowed.shape[0]:
                continue
            if visited[ny, nx]:
                continue
            visited[ny, nx] = True
            if not allowed[ny, nx]:
                continue

            edge_strength = float(max(edges[y, x], edges[ny, nx]))
            if edge_strength > edge_threshold:
                continue

            candidate = image[ny, nx, :3]
            seed_distance = float(np.linalg.norm(candidate - seed_color))
            local_distance = float(np.linalg.norm(candidate - current))
            combined = .72 * seed_distance + .28 * local_distance
            if combined > max_color_distance:
                continue

            result[ny, nx] = True
            queue.append((nx, ny))

    return result


def boundary_adherence(mask: np.ndarray, edges: np.ndarray) -> float:
    """Mean edge evidence around a binary mask boundary."""
    mask = np.asarray(mask, dtype=bool)
    edges = np.asarray(edges, dtype=np.float32)
    if mask.shape != edges.shape:
        raise ValueError("Mask and edge map dimensions differ")
    boundary = mask & (
        ~np.roll(mask, 1, 0)
        | ~np.roll(mask, -1, 0)
        | ~np.roll(mask, 1, 1)
        | ~np.roll(mask, -1, 1)
    )
    # Ignore wrap-around introduced by np.roll.
    boundary[[0, -1], :] = False
    boundary[:, [0, -1]] = False
    values = edges[boundary]
    return float(values.mean()) if len(values) else 0.0
