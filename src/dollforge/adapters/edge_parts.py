"""Edge-driven visible-part proposals; anatomical labels remain reviewable priors."""
from io import BytesIO

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from dollforge.adapters.baseline import png
from dollforge.adapters.contour import _bounds, foreground_mask
from dollforge.contracts import MaskProposal, SegmentationRequest
from dollforge.domain.models import Provenance
from dollforge.vision.edges import boundary_adherence, sobel_edge_map


def neck_row(mask: np.ndarray) -> int:
    """Find a local silhouette constriction, allowing oversized collectible heads."""
    bounds = _bounds(mask)
    if bounds is None:
        raise ValueError("Empty foreground")
    x0, y0, x1, y1 = bounds
    height = y1 - y0
    widths = ndi.gaussian_filter1d(mask[:, x0:x1].sum(axis=1).astype(float),
                                  max(1, height * .008))
    lo, hi = y0 + int(height * .20), y0 + int(height * .62)
    span = max(2, int(height * .07))
    candidates = []
    for y in range(max(lo, span), min(hi, len(widths) - span)):
        before = widths[y - span:y].max()
        after = widths[y + 1:y + span + 1].max()
        prominence = min(before, after) - widths[y]
        if prominence > .04 * (x1 - x0):
            candidates.append((prominence / max(1, widths[y]), y))
    return max(candidates)[1] if candidates else y0 + round(height * .31)


def partition(rgb: np.ndarray, mask: np.ndarray,
              seeds: list[tuple[float, float]],
              extra_seeds: list[tuple[int, float, float]] | None = None,
              ) -> tuple[np.ndarray, np.ndarray]:
    """Competitive flooding on gradients. No fallback discards the edge evidence."""
    markers = np.zeros(mask.shape, dtype=np.int32)
    edges = sobel_edge_map(ndi.gaussian_filter(rgb.astype(float), (1, 1, 0)))
    # Prefer interior seed positions to avoid placing a seed directly on a boundary.
    interior = ndi.distance_transform_edt(mask)
    yy, xx = np.where(mask)
    if not len(xx):
        return markers, edges
    scale = max(mask.shape)
    points = [(index, x, y) for index, (x, y) in enumerate(seeds, 1)]
    points.extend(extra_seeds or [])
    for index, x, y in points:
        cost = ((xx - x) ** 2 + (yy - y) ** 2) / scale**2
        cost += .002 / np.maximum(interior[yy, xx], .1)
        cost[markers[yy, xx] != 0] = np.inf
        chosen = int(np.argmin(cost))
        if np.isfinite(cost[chosen]):
            markers[yy[chosen], xx[chosen]] = index
    labels = watershed(edges, markers, mask=mask, compactness=.002)
    return labels, edges


class EdgePartSegmenter:
    model_id = "contour_rules_v3"
    model_version = "3.0.0"

    def predict(self, request: SegmentationRequest) -> list[MaskProposal]:
        rgb = np.asarray(Image.open(BytesIO(request.image_png)).convert("RGB"))
        mask = foreground_mask(request.image_png, int(request.parameters.get(
            "foreground_threshold", request.threshold)))
        bounds = _bounds(mask)
        if bounds is None:
            return []
        x0, y0, x1, y1 = bounds
        width, height = x1 - x0, y1 - y0
        neck = neck_row(mask)
        override = request.parameters.get("neck_fraction")
        if override is not None:
            value = float(override)
            if not np.isfinite(value) or not .15 <= value <= .70:
                raise ValueError("neck_fraction must be between .15 and .70")
            neck = y0 + round(value * height)
        head_mask = mask.copy()
        head_mask[neck:] = False
        body_mask = mask & ~head_mask
        body_height = y1 - neck
        front = request.view.label in ("front", "back")
        left = "right" if request.view.label == "front" else "left"
        right = "left" if request.view.label == "front" else "right"
        if front:
            specs = [("torso", "center", .50, .20),
                     ("pelvis", "center", .50, .48),
                     ("arm", left, .12, .26), ("arm", right, .88, .26),
                     ("leg", left, .35, .68), ("leg", right, .65, .68),
                     ("footwear", left, .32, .94), ("footwear", right, .68, .94)]
        else:
            side = request.view.label if request.view.label in ("left", "right") else "unknown"
            specs = [("torso", "center", .43, .18),
                     ("pelvis", "center", .43, .48),
                     ("arm", side, .72, .28), ("leg", side, .48, .68),
                     ("footwear", side, .50, .94)]
        seeds = [(x0 + cx * width, neck + cy * body_height) for _, _, cx, cy in specs]
        # Several torso markers prevent a shirt print from capturing its only seed.
        extra = [(1, x0 + width * .50, neck + body_height * t)
                 for t in (.10, .32, .39)]
        labels, edges = partition(rgb, body_mask, seeds, extra)
        regions = [("head", "center", head_mask)]
        regions.extend((name, side, labels == i)
                       for i, (name, side, _, _) in enumerate(specs, 1))
        # Hair is opt-in: useful for dark hair / light face references, not a semantic detector.
        if request.parameters.get("split_dark_hair", False) and head_mask.any():
            head_labels, _ = partition(rgb, head_mask, [
                (x0 + width * .5, y0 + (neck - y0) * .20),
                (x0 + width * .5, y0 + (neck - y0) * .75)])
            hair, face = head_labels == 1, head_labels == 2
            if hair.sum() >= 12 and face.sum() >= 12:
                contrast = rgb[face].mean() - rgb[hair].mean()
                if contrast >= 30:
                    regions[0] = ("head", "center", face)
                    regions.insert(0, ("hair", "center", hair))
        proposals = []
        for name, side, region in regions:
            box = _bounds(region)
            if box is None or region.sum() < 12:
                continue
            proposals.append(MaskProposal(
                part_class=name, side=side, bbox_xyxy=box,
                confidence=float(np.clip(.35 + .25 * boundary_adherence(region, edges), .35, .60)),
                provenance=Provenance(type="rule_based", source=self.model_id,
                    evidence=[request.view.normalized_artifact_id],
                    note="Competitive edge watershed; silhouette neck anchor; semantic labels "
                         "are anatomical hypotheses requiring review. Hidden surfaces not inferred."),
                mask_png=png(Image.fromarray((region * 255).astype("uint8")))))
        return proposals
