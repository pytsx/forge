from io import BytesIO
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw

from dollforge.adapters.contour import ContourDollSegmenter, transfer_human_mask
from dollforge.contracts import SegmentationRequest
from dollforge.domain.models import ImageQA, ImageView, PartObservation, Provenance


def png(image):
    output = BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def view(label="front", size=(240, 320)):
    return ImageView(
        project_id=uuid4(),
        label=label,
        original_artifact_id=uuid4(),
        normalized_artifact_id=uuid4(),
        qa=ImageQA(width=size[0], height=size[1], sharpness=100),
        provenance=Provenance(type="observed", source="test"),
    )


def doll_image():
    image = Image.new("RGB", (240, 320), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((60, 15, 180, 115), fill=(80, 80, 80))
    draw.rounded_rectangle((75, 115, 165, 205), radius=28, fill=(80, 80, 80))
    draw.ellipse((45, 120, 90, 205), fill=(80, 80, 80))
    draw.ellipse((150, 120, 195, 205), fill=(80, 80, 80))
    draw.rounded_rectangle((75, 200, 115, 285), radius=16, fill=(80, 80, 80))
    draw.rounded_rectangle((125, 200, 165, 285), radius=16, fill=(80, 80, 80))
    draw.ellipse((60, 265, 120, 310), fill=(80, 80, 80))
    draw.ellipse((120, 265, 180, 310), fill=(80, 80, 80))
    return image


def test_contour_segmenter_masks_follow_foreground_not_full_boxes():
    image = doll_image()
    v = view()
    result = ContourDollSegmenter().predict(
        SegmentationRequest(view=v, image_png=png(image), threshold=24, seed=42)
    )
    head = next(item for item in result if item.part_class == "head")
    mask = np.asarray(Image.open(BytesIO(head.mask_png)).convert("L")) > 127
    x0, y0, x1, y1 = head.bbox_xyxy
    fill_ratio = mask[y0:y1, x0:x1].mean()

    assert 0.45 < fill_ratio < 0.92
    assert mask[y0, x0] == 0


def test_human_mask_transfer_preserves_target_view_evidence():
    target_image = doll_image()
    source = Image.new("L", (240, 320), 0)
    ImageDraw.Draw(source).ellipse((58, 18, 182, 116), fill=255)
    target = Image.new("L", (240, 320), 0)
    ImageDraw.Draw(target).ellipse((75, 20, 165, 116), fill=255)

    observation = PartObservation(
        view_id=uuid4(),
        **{"class": "head"},
        side="center",
        confidence=.35,
        bbox_xyxy=(75, 20, 166, 117),
        mask_artifact_id=uuid4(),
        provenance=Provenance(type="rule_based", source="test"),
    )

    result = transfer_human_mask(
        png(source), png(target_image), png(target), observation, threshold=24
    )

    assert result is not None
    transferred_png, box = result
    transferred = np.asarray(Image.open(BytesIO(transferred_png)).convert("L")) > 127
    assert transferred.sum() > 0
    assert box[1] < 35
    assert transferred[20:115, 60:180].mean() > 0.15
