"""Boundary-first segmentation for multiview doll references.

The segmenter deliberately separates two jobs:

* boundary discovery: image gradients and the foreground silhouette define
  where a region is allowed to end;
* semantic naming: seeds only name cells after the boundary graph is built.

This prevents a proportional box from inventing an arm, torso, or shoe.
Closed cells with weak boundaries are returned with conservative confidence so
the review UI can refine the boundary rather than silently accepting a prior.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from dollforge.adapters.baseline import png
from dollforge.adapters.contour import _bounds, foreground_mask
from dollforge.contracts import MaskProposal, SegmentationRequest
from dollforge.domain.models import Provenance
from dollforge.vision.edges import boundary_adherence, sobel_edge_map


def _normalized_edges(rgb: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    """Build a boundary cost map without assuming doll proportions."""
    smooth = ndi.gaussian_filter(rgb.astype(np.float32), sigma=(1.15, 1.15, 0))
    gradients = sobel_edge_map(smooth)

    # Strong silhouette edges are hard barriers. Interior edges remain soft so
    # watershed can use them when they form a meaningful closed boundary.
    outside = ndi.binary_dilation(foreground, iterations=1) & ~foreground
    gradients[outside] = 1.0

    # Mildly close hair/clothing texture without erasing actual boundaries.
    edge_image = Image.fromarray((gradients * 255).astype("uint8"))
    edge_image = edge_image.filter(ImageFilter.GaussianBlur(radius=.65))
    return np.asarray(edge_image, dtype=np.float32) / 255.0


def _seed_from_fraction(bounds: tuple[int, int, int, int],
                        x: float, y: float) -> tuple[int, int]:
    x0, y0, x1, y1 = bounds
    return round(x0 + (x1 - x0) * x), round(y0 + (y1 - y0) * y)


def _nearest_foreground(mask: np.ndarray, point: tuple[int, int]) -> tuple[int, int] | None:
    x, y = point
    if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x]:
        return x, y
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    index = int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))
    return int(xs[index]), int(ys[index])


def _marker_image(mask: np.ndarray,
                  seeds: list[tuple[str, str, tuple[int, int]]]) -> tuple[np.ndarray, list[tuple[str, str]]]:
    markers = np.zeros(mask.shape, dtype=np.int32)
    names: list[tuple[str, str]] = []
    for index, (part_class, side, point) in enumerate(seeds, 1):
        resolved = _nearest_foreground(mask, point)
        if resolved is None:
            continue
        x, y = resolved
        # A marker that lands on an edge is moved to the nearest high-distance
        # interior pixel. This avoids assigning a boundary to a semantic part.
        distance = ndi.distance_transform_edt(mask)
        window = max(2, min(mask.shape) // 60)
        y0, y1 = max(0, y - window), min(mask.shape[0], y + window + 1)
        x0, x1 = max(0, x - window), min(mask.shape[1], x + window + 1)
        local = distance[y0:y1, x0:x1]
        if local.size:
            ly, lx = np.unravel_index(int(np.argmax(local)), local.shape)
            x, y = x0 + lx, y0 + ly
        if markers[y, x] == 0:
            markers[y, x] = index
            names.append((part_class, side))
    return markers, names


def _region_confidence(region: np.ndarray, edges: np.ndarray,
                       boundary_threshold: float) -> float:
    adherence = boundary_adherence(region, edges)
    boundary_pixels = region & (
        ~ndi.binary_erosion(region, iterations=1)
    )
    values = edges[boundary_pixels]
    closure = float((values >= boundary_threshold).mean()) if len(values) else 0.0
    return float(np.clip(.25 + .40 * adherence + .35 * closure, .20, .95))


class BoundaryFirstSegmenter:
    """Discover closed image cells first, then attach semantic labels."""

    model_id = "boundary_cells_v1"
    model_version = "1.0.0"

    def predict(self, request: SegmentationRequest) -> list[MaskProposal]:
        image = Image.open(BytesIO(request.image_png)).convert("RGB")
        rgb = np.asarray(image)
        foreground = foreground_mask(
            request.image_png,
            int(request.parameters.get("foreground_threshold", request.threshold)),
        )
        bounds = _bounds(foreground)
        if bounds is None:
            return []

        x0, y0, x1, y1 = bounds
        edges = _normalized_edges(rgb, foreground)
        boundary_threshold = float(request.parameters.get("boundary_threshold", .34))

        # These are only candidate semantic seeds. They never define a box or
        # the extent of a part; the watershed boundaries do that.
        front = request.view.label in ("front", "back")
        left = "right" if request.view.label == "front" else "left"
        right = "left" if request.view.label == "front" else "right"
        if front:
            seed_specs = [
                ("head", "center", .50, .16),
                ("torso", "center", .50, .43),
                ("pelvis", "center", .50, .61),
                ("arm", left, .17, .46),
                ("arm", right, .83, .46),
                ("leg", left, .37, .78),
                ("leg", right, .63, .78),
                ("footwear", left, .35, .94),
                ("footwear", right, .65, .94),
            ]
        else:
            side = request.view.label if request.view.label in ("left", "right") else "unknown"
            seed_specs = [
                ("head", "center", .50, .16),
                ("torso", "center", .46, .43),
                ("pelvis", "center", .46, .61),
                ("arm", side, .70, .46),
                ("leg", side, .50, .78),
                ("footwear", side, .50, .94),
            ]
        seeds = [(name, side, _seed_from_fraction(bounds, cx, cy))
                 for name, side, cx, cy in seed_specs]
        markers, names = _marker_image(foreground, seeds)
        if not names:
            return []

        # Gradient is the barrier. A lower compactness preserves irregular
        # contours and lets the image, rather than the prior, own the shape.
        labels = watershed(
            edges,
            markers,
            mask=foreground,
            compactness=float(request.parameters.get("compactness", .0005)),
        )

        proposals: list[MaskProposal] = []
        for index, (part_class, side) in enumerate(names, 1):
            region = labels == index
            box = _bounds(region)
            if box is None or int(region.sum()) < 12:
                continue
            confidence = _region_confidence(region, edges, boundary_threshold)
            proposals.append(MaskProposal(
                part_class=part_class,
                side=side,
                bbox_xyxy=box,
                confidence=confidence,
                provenance=Provenance(
                    type="rule_based",
                    source=self.model_id,
                    evidence=[request.view.normalized_artifact_id],
                    note=(
                        "Célula delimitada por fronteiras de imagem; a semente apenas "
                        "atribui o rótulo sem definir o formato da peça."
                    ),
                ),
                mask_png=png(Image.fromarray((region * 255).astype("uint8"))),
            ))
        return proposals


def boundary_memory_record(proposal: MaskProposal, view_label: str,
                           mask_artifact_id: str, revision: int) -> dict:
    """Canonical record consumed by future human-in-the-loop runs."""
    return {
        "view_label": view_label,
        "part_class": proposal.part_class,
        "side": proposal.side,
        "bbox_xyxy": list(proposal.bbox_xyxy),
        "mask_artifact_id": mask_artifact_id,
        "revision": revision,
        "source": "human_boundary_refinement",
    }
