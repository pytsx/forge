from io import BytesIO

import numpy as np
import pytest
from PIL import Image, ImageDraw
from test_contour_segmentation import png, view

from dollforge.adapters.edge_parts import EdgePartSegmenter, neck_row, partition
from dollforge.contracts import SegmentationRequest


def oversized():
    image = Image.new('RGB', (240, 320), 'white')
    d = ImageDraw.Draw(image)
    d.ellipse((35, 5, 205, 150), fill=(210, 170, 140))
    d.rectangle((107, 140, 133, 170), fill=(210, 170, 140))
    d.rectangle((60, 170, 180, 235), fill=(60, 60, 60))
    d.rectangle((70, 230, 105, 310), fill=(100, 110, 130))
    d.rectangle((135, 230, 170, 310), fill=(100, 110, 130))
    return image


@pytest.mark.parametrize('label', ['front', 'back', 'left', 'right'])
def test_oversized_head_never_becomes_arm_or_torso(label):
    image = oversized()
    result = EdgePartSegmenter().predict(SegmentationRequest(
        view=view(label), image_png=png(image), threshold=24, seed=42))
    masks = [np.asarray(Image.open(BytesIO(p.mask_png))) > 127 for p in result]
    assert np.stack(masks).sum(axis=0).max() == 1
    for p, mask in zip(result, masks):
        if p.part_class != 'head':
            assert not mask[:140].any()
    head = next(m for p, m in zip(result, masks) if p.part_class == 'head')
    assert head[100, 120]


def test_watershed_follows_boundary_when_seed_midpoint_is_wrong():
    rgb = np.full((80, 100, 3), 30, dtype=np.uint8)
    rgb[:, 65:] = 220
    labels, _ = partition(rgb, np.ones((80, 100), bool), [(15, 40), (80, 40)])
    assert (labels[:, :63] == 1).mean() > .97
    assert (labels[:, 67:] == 2).mean() > .97


def test_neck_tracks_large_head():
    from dollforge.adapters.contour import foreground_mask
    mask = foreground_mask(png(oversized()), 24)
    assert 140 <= neck_row(mask) <= 170


def test_blank_image_and_invalid_override():
    request = SegmentationRequest(view=view(), image_png=png(oversized()), threshold=24, seed=42,
                                  parameters={'neck_fraction': .9})
    with pytest.raises(ValueError):
        EdgePartSegmenter().predict(request)
    request.image_png = png(Image.new('RGB', (240, 320), 'white'))
    assert EdgePartSegmenter().predict(request) == []


def test_extra_markers_keep_shirt_print_in_same_part():
    rgb = np.full((100, 100, 3), 40, dtype=np.uint8)
    rgb[35:65, 35:65] = 220
    labels, _ = partition(rgb, np.ones((100, 100), bool), [(50, 50)],
                          [(1, 50, 20), (1, 50, 80)])
    assert np.all(labels == 1)


def test_dark_hair_split_is_opt_in_and_disjoint():
    image = oversized()
    ImageDraw.Draw(image).rectangle((55, 20, 185, 65), fill=(30, 30, 30))
    request = SegmentationRequest(view=view(), image_png=png(image), threshold=24, seed=42)
    assert 'hair' not in [p.part_class for p in EdgePartSegmenter().predict(request)]
    request.parameters['split_dark_hair'] = True
    parts = EdgePartSegmenter().predict(request)
    assert 'hair' in [p.part_class for p in parts]
    masks = [np.asarray(Image.open(BytesIO(p.mask_png))) > 127 for p in parts]
    assert np.stack(masks).sum(axis=0).max() == 1
