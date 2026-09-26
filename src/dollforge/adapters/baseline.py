"""CPU baselines. Heuristic proposals always require review; no pretrained weights."""
from io import BytesIO

import numpy as np
import trimesh
from PIL import Image, ImageFilter

from dollforge.contracts import MaskProposal, MeshCandidate, SegmentationRequest, VolumetryResult
from dollforge.domain.models import DollGraph, ImageView, PartObservation, Provenance, Side


def png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class SilhouetteSegmenter:
    model_id = "silhouette_rules_v1"
    model_version = "1.0.0"

    def predict(self, request: SegmentationRequest) -> list[MaskProposal]:
        image = Image.open(BytesIO(request.image_png)).convert("RGBA")
        rgba = np.asarray(image).astype(np.int16)
        rgb = rgba[:, :, :3]
        border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
        background = np.median(border, axis=0)
        foreground = np.max(np.abs(rgb - background), axis=2) > request.threshold
        if np.min(rgba[:, :, 3]) < 255:
            foreground = rgba[:, :, 3] > 127
        foreground = np.asarray(Image.fromarray((foreground * 255).astype("uint8"))
                                .filter(ImageFilter.MedianFilter(3))) > 0
        ys, xs = np.where(foreground)
        if len(xs) < 25:
            return []
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        width, height = x1 - x0, y1 - y0
        front = request.view.label == "front"
        back = request.view.label == "back"
        if front or back:
            image_left = "right" if front else "left"
            image_right = "left" if front else "right"
            regions = [
                ("head", "center", (0, 0, 1, .26)),
                ("torso", "center", (.27, .26, .73, .51)),
                ("pelvis", "center", (.27, .51, .73, .63)),
                ("arm", image_left, (0, .26, .27, .63)),
                ("arm", image_right, (.73, .26, 1, .63)),
                ("leg", image_left, (0, .63, .5, .91)),
                ("leg", image_right, (.5, .63, 1, .91)),
                ("footwear", image_left, (0, .91, .5, 1)),
                ("footwear", image_right, (.5, .91, 1, 1)),
            ]
        else:
            side = request.view.label if request.view.label in ("left", "right") else "unknown"
            regions = [("head", "center", (0, 0, 1, .26)),
                       ("torso", "center", (0, .26, .55, .51)),
                       ("arm", side, (.55, .26, 1, .63)),
                       ("pelvis", "center", (0, .51, .55, .63)),
                       ("leg", side, (0, .63, 1, .91)),
                       ("footwear", side, (0, .91, 1, 1))]
        proposals = []
        for part_class, side, region in regions:
            a, b, c, d = region
            box = (x0 + round(a * width), y0 + round(b * height),
                   x0 + round(c * width), y0 + round(d * height))
            mask = np.zeros(foreground.shape, dtype="uint8")
            mask[box[1]:box[3], box[0]:box[2]] = foreground[box[1]:box[3], box[0]:box[2]] * 255
            yy, xx = np.where(mask)
            if len(xx) < 4:
                continue
            proposals.append(MaskProposal(
                part_class=part_class, side=side,
                bbox_xyxy=(int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1),
                confidence=.35,
                provenance=Provenance(type="rule_based", source=self.model_id,
                                      evidence=[request.view.normalized_artifact_id],
                                      note="Divisão proporcional da silhueta; requer correção humana."),
                mask_png=png(Image.fromarray(mask)),
            ))
        return proposals


class EllipsoidReconstructor:
    model_id = "ellipsoid_multiview_v1"
    model_version = "1.0.0"

    def reconstruct(self, graph: DollGraph, observations: list[PartObservation],
                    views: list[ImageView],
                    volumetry: VolumetryResult | None = None) -> list[MeshCandidate]:
        by_id = {o.observation_id: o for o in observations}
        view_map = {v.view_id: v for v in views}
        bounds = {}
        for view in views:
            boxes = [o.bbox_xyxy for o in observations if o.view_id == view.view_id]
            if boxes:
                bounds[view.view_id] = (min(b[0] for b in boxes), min(b[1] for b in boxes),
                                        max(b[2] for b in boxes), max(b[3] for b in boxes))
        height = graph.scale.canonical_height
        output = []
        for part in graph.parts:
            obs = [by_id[o] for o in part.observation_ids]
            front = [o for o in obs if view_map[o.view_id].label in ("front", "back")]
            profile = [o for o in obs if view_map[o.view_id].label in ("left", "right")]
            reference = (front or obs)[0]
            x0, y0, x1, y1 = reference.bbox_xyxy
            bx0, by0, bx1, by1 = bounds[reference.view_id]
            factor = height / (by1 - by0)
            extents = [(x1 - x0) * factor, (x1 - x0) * factor * .7, (y1 - y0) * factor]
            if profile:
                widths = [(o.bbox_xyxy[2] - o.bbox_xyxy[0]) * height /
                          (bounds[o.view_id][3] - bounds[o.view_id][1]) for o in profile]
                extents[1] = float(np.median(widths))
            sign = -1 if view_map[reference.view_id].label == "back" else 1
            center_x = ((x0 + x1 - bx0 - bx1) / 2) * factor * sign
            if part.side == Side.CENTER:
                center_x = 0
            center_z = (by1 - (y0 + y1) / 2) * factor - height * .43
            mesh = trimesh.creation.icosphere(subdivisions=2, radius=1)
            mesh.apply_scale(np.maximum(extents, height * .01) / 2)
            mesh.apply_translation([center_x, 0, center_z])
            transform = np.eye(4)
            transform[:3, 3] = [center_x, 0, center_z]
            output.append(MeshCandidate(
                part_instance_id=part.part_instance_id,
                name=f"{part.part_class}__{part.side}__{str(part.part_instance_id)[:8]}",
                vertices=mesh.vertices.tolist(), faces=mesh.faces.tolist(), confidence=.3,
                transform=transform.flatten().tolist(),
                provenance=Provenance(type="derived_geometry", source=self.model_id,
                    evidence=[o.mask_artifact_id for o in obs],
                    note="Elipsoide ajustado às caixas multi-view; superfície e profundidade aproximadas."),
            ))
        return output
