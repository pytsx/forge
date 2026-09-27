from io import BytesIO
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw

from dollforge.adapters.boundary_segmenter import BoundaryFirstSegmenter
from dollforge.contracts import SegmentationRequest
from dollforge.domain.models import ImageQA, ImageView, Provenance


def image_png():
    image = Image.new("RGB", (240, 320), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((55, 15, 185, 125), fill=(80, 80, 80))
    draw.rounded_rectangle((78, 120, 162, 220), radius=20, fill=(110, 110, 110))
    draw.ellipse((42, 125, 87, 215), fill=(90, 90, 90))
    draw.ellipse((153, 125, 198, 215), fill=(90, 90, 90))
    draw.rectangle((82, 210, 115, 290), fill=(120, 120, 120))
    draw.rectangle((125, 210, 158, 290), fill=(120, 120, 120))
    output = BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def request():
    view = ImageView(
        project_id=uuid4(),
        label="front",
        original_artifact_id=uuid4(),
        normalized_artifact_id=uuid4(),
        qa=ImageQA(width=240, height=320, sharpness=100),
        provenance=Provenance(type="observed", source="test"),
    )
    return SegmentationRequest(view=view, image_png=image_png(), threshold=24, seed=42)


def test_boundary_segmenter_returns_exclusive_image_cells():
    result = BoundaryFirstSegmenter().predict(request())
    masks = []
    for proposal in result:
        mask = np.asarray(Image.open(BytesIO(proposal.mask_png)).convert("L")) > 127
        masks.append(mask)
    assert result
    assert np.stack(masks).sum(axis=0).max() <= 1


def test_boundary_segmenter_does_not_use_rectangular_bbox_as_mask():
    result = BoundaryFirstSegmenter().predict(request())
    fill_ratios = []
    for proposal in result:
        mask = np.asarray(Image.open(BytesIO(proposal.mask_png)).convert("L")) > 127
        x0, y0, x1, y1 = proposal.bbox_xyxy
        fill_ratios.append(mask[y0:y1, x0:x1].mean())
    assert min(fill_ratios) < .98
