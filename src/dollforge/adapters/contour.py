from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter

from dollforge.adapters.baseline import png
from dollforge.contracts import MaskProposal, SegmentationRequest
from dollforge.domain.models import PartObservation, Provenance, Side


def foreground_mask(image_png: bytes, threshold: int) -> np.ndarray:
    image = Image.open(BytesIO(image_png)).convert("RGBA")
    rgba = np.asarray(image).astype(np.int16)
    rgb = rgba[:, :, :3]
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    background = np.median(border, axis=0)
    foreground = np.max(np.abs(rgb - background), axis=2) > threshold
    if np.min(rgba[:, :, 3]) < 255:
        foreground = rgba[:, :, 3] > 127
    cleaned = Image.fromarray((foreground * 255).astype("uint8"))
    cleaned = cleaned.filter(ImageFilter.MedianFilter(3))
    cleaned = cleaned.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    return np.asarray(cleaned) > 127


def _bounds(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) < 4:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _superellipse(shape: tuple[int, int], bounds: tuple[int, int, int, int],
                  cx: float, cy: float, rx: float, ry: float,
                  px: float = 2.4, py: float = 2.4) -> np.ndarray:
    height, width = shape
    x0, y0, x1, y1 = bounds
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    yy, xx = np.ogrid[:height, :width]
    nx = np.abs((xx - (x0 + cx * bw)) / max(1.0, rx * bw))
    ny = np.abs((yy - (y0 + cy * bh)) / max(1.0, ry * bh))
    return np.power(nx, px) + np.power(ny, py) <= 1.0


def _mask_from_prior(foreground: np.ndarray, prior: np.ndarray) -> np.ndarray:
    candidate = foreground & prior
    image = Image.fromarray((candidate * 255).astype("uint8"))
    image = image.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    return np.asarray(image) > 127


class ContourDollSegmenter:
    """CPU fallback specialized in doll silhouettes.

    Unlike v1, semantic regions are curved shape priors intersected with the true
    foreground. This avoids the rectangular masks visible in the first prototype.
    """

    model_id = "contour_rules_v2"
    model_version = "2.0.0"

    def predict(self, request: SegmentationRequest) -> list[MaskProposal]:
        foreground = foreground_mask(request.image_png, request.threshold)
        object_bounds = _bounds(foreground)
        if not object_bounds:
            return []

        front = request.view.label in ("front", "back")
        if front:
            image_left = "right" if request.view.label == "front" else "left"
            image_right = "left" if request.view.label == "front" else "right"
            specs = [
                ("head", "center", .50, .135, .50, .17, 2.2, 2.2),
                ("torso", "center", .50, .405, .30, .17, 3.2, 2.8),
                ("pelvis", "center", .50, .585, .25, .105, 3.0, 2.5),
                ("arm", image_left, .16, .43, .16, .20, 2.1, 2.3),
                ("arm", image_right, .84, .43, .16, .20, 2.1, 2.3),
                ("leg", image_left, .35, .77, .20, .18, 2.3, 2.5),
                ("leg", image_right, .65, .77, .20, .18, 2.3, 2.5),
                ("footwear", image_left, .34, .925, .24, .10, 3.0, 2.4),
                ("footwear", image_right, .66, .925, .24, .10, 3.0, 2.4),
            ]
        else:
            side = request.view.label if request.view.label in ("left", "right") else "unknown"
            specs = [
                ("head", "center", .50, .135, .50, .17, 2.2, 2.2),
                ("torso", "center", .46, .405, .34, .17, 3.0, 2.8),
                ("pelvis", "center", .46, .585, .30, .105, 3.0, 2.5),
                ("arm", side, .69, .43, .25, .20, 2.0, 2.3),
                ("leg", side, .48, .77, .33, .18, 2.4, 2.5),
                ("footwear", side, .50, .925, .38, .10, 3.0, 2.4),
            ]

        proposals: list[MaskProposal] = []
        for part_class, side, cx, cy, rx, ry, px, py in specs:
            prior = _superellipse(foreground.shape, object_bounds, cx, cy, rx, ry, px, py)
            mask = _mask_from_prior(foreground, prior)
            box = _bounds(mask)
            if not box:
                continue
            proposals.append(MaskProposal(
                part_class=part_class,
                side=side,
                bbox_xyxy=box,
                confidence=.52,
                provenance=Provenance(
                    type="rule_based",
                    source=self.model_id,
                    evidence=[request.view.normalized_artifact_id],
                    note="Prior curvo especializado em bonecos, recortado pelo contorno real da peça.",
                ),
                mask_png=png(Image.fromarray((mask * 255).astype("uint8"))),
            ))
        return proposals


def _row_profile(mask: np.ndarray, box: tuple[int, int, int, int], samples: int = 96) -> np.ndarray:
    x0, y0, x1, y1 = box
    width = max(1, x1 - x0)
    profile = np.zeros(samples, dtype=np.float32)
    for index in range(samples):
        y = min(y1 - 1, y0 + int((index + .5) / samples * max(1, y1 - y0)))
        xs = np.where(mask[y, x0:x1])[0]
        profile[index] = 0.0 if len(xs) == 0 else (xs.max() - xs.min() + 1) / width
    if samples >= 5:
        profile = np.convolve(profile, np.ones(5, dtype=np.float32) / 5, mode="same")
    return np.clip(profile, 0.05, 1.0)


def transfer_human_mask(source_mask_png: bytes, target_image_png: bytes, target_mask_png: bytes,
                        target: PartObservation, threshold: int) -> tuple[bytes, tuple[int, int, int, int]] | None:
    """Transfer a human correction as a project-local shape prior.

    It transfers vertical extent and the normalized row-width profile, while keeping
    the target view's own center/depth evidence. The result is always intersected
    with the target foreground, so it follows target-view curves instead of copying pixels.
    """

    source = np.asarray(Image.open(BytesIO(source_mask_png)).convert("L")) > 127
    target_old = np.asarray(Image.open(BytesIO(target_mask_png)).convert("L")) > 127
    source_box = _bounds(source)
    target_box = _bounds(target_old) or tuple(int(v) for v in target.bbox_xyxy)
    if not source_box or not target_box:
        return None

    foreground = foreground_mask(target_image_png, threshold)
    height, width = foreground.shape
    sx0, sy0, sx1, sy1 = source_box
    tx0, ty0, tx1, ty1 = target_box

    source_h = source.shape[0]
    src_cy = ((sy0 + sy1) / 2) / source_h
    src_h = (sy1 - sy0) / source_h
    tgt_cy = ((ty0 + ty1) / 2) / height
    tgt_h = (ty1 - ty0) / height

    cy = .68 * src_cy + .32 * tgt_cy
    part_h = np.clip(.68 * src_h + .32 * tgt_h, .015, .55)
    new_y0 = max(0, int((cy - part_h / 2) * height))
    new_y1 = min(height, int((cy + part_h / 2) * height))
    if new_y1 - new_y0 < 3:
        return None

    source_profile = _row_profile(source, source_box)
    target_profile = _row_profile(target_old, target_box)
    profile = .62 * source_profile + .38 * target_profile

    center_x = (tx0 + tx1) / 2
    base_width = max(3, tx1 - tx0)
    candidate = np.zeros_like(foreground)
    samples = len(profile)
    for y in range(new_y0, new_y1):
        t = (y - new_y0) / max(1, new_y1 - new_y0 - 1)
        idx = min(samples - 1, int(t * samples))
        half = max(2.0, base_width * float(profile[idx]) * .55)
        left = max(0, int(center_x - half))
        right = min(width, int(center_x + half) + 1)
        candidate[y, left:right] = True

    candidate &= foreground
    smoothed = Image.fromarray((candidate * 255).astype("uint8"))
    smoothed = smoothed.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
    mask = np.asarray(smoothed) > 127
    box = _bounds(mask)
    if not box:
        return None
    return png(Image.fromarray((mask * 255).astype("uint8"))), box
