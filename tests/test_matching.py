from io import BytesIO
from uuid import uuid4

from PIL import Image, ImageDraw

from dollforge.adapters.matching import MultiSignalMatcher
from dollforge.contracts import MatchingRequest
from dollforge.domain.models import ImageQA, ImageView, PartObservation, Provenance


def png(image):
    output = BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def view(project_id, label):
    return ImageView(
        project_id=project_id,
        label=label,
        original_artifact_id=uuid4(),
        normalized_artifact_id=uuid4(),
        qa=ImageQA(width=100, height=100, sharpness=100),
        provenance=Provenance(type="observed", source="test"),
    )


def observation(v, side, box, mask_id):
    return PartObservation(
        view_id=v.view_id,
        **{"class": "hand"},
        side=side,
        confidence=0.9,
        bbox_xyxy=box,
        mask_artifact_id=mask_id,
        provenance=Provenance(type="model_inferred", source="test"),
    )


def test_multisignal_matcher_groups_same_semantic_piece_across_views():
    project_id = uuid4()
    front = view(project_id, "front")
    left = view(project_id, "left")
    mask_a, mask_b = uuid4(), uuid4()
    a = observation(front, "left", (10, 40, 30, 70), mask_a)
    b = observation(left, "left", (12, 41, 31, 70), mask_b)

    image_a = Image.new("RGB", (100, 100), "white")
    image_b = Image.new("RGB", (100, 100), "white")
    ImageDraw.Draw(image_a).rectangle(a.bbox_xyxy, fill=(30, 40, 50))
    ImageDraw.Draw(image_b).rectangle(b.bbox_xyxy, fill=(32, 42, 52))
    mask1 = Image.new("L", (100, 100), 0)
    mask2 = Image.new("L", (100, 100), 0)
    ImageDraw.Draw(mask1).rectangle(a.bbox_xyxy, fill=255)
    ImageDraw.Draw(mask2).rectangle(b.bbox_xyxy, fill=255)

    result = MultiSignalMatcher().match(MatchingRequest(
        project_id=project_id,
        observations=[a, b],
        views=[front, left],
        image_png_by_view={front.view_id: png(image_a), left.view_id: png(image_b)},
        mask_png_by_observation={a.observation_id: png(mask1), b.observation_id: png(mask2)},
    ))

    assert len(result.parts) == 1
    assert set(result.parts[0].observation_ids) == {a.observation_id, b.observation_id}


def test_multisignal_matcher_never_merges_opposite_sides():
    project_id = uuid4()
    front = view(project_id, "front")
    back = view(project_id, "back")
    mask_a, mask_b = uuid4(), uuid4()
    a = observation(front, "left", (10, 40, 30, 70), mask_a)
    b = observation(back, "right", (10, 40, 30, 70), mask_b)
    image = png(Image.new("RGB", (100, 100), "gray"))
    mask = Image.new("L", (100, 100), 0)
    ImageDraw.Draw(mask).rectangle((10, 40, 30, 70), fill=255)
    mask_png = png(mask)

    result = MultiSignalMatcher().match(MatchingRequest(
        project_id=project_id,
        observations=[a, b],
        views=[front, back],
        image_png_by_view={front.view_id: image, back.view_id: image},
        mask_png_by_observation={a.observation_id: mask_png, b.observation_id: mask_png},
    ))

    assert len(result.parts) == 2
