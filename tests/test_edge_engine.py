import numpy as np

from dollforge.vision.edges import boundary_adherence, edge_guided_region, sobel_edge_map


def test_sobel_edge_map_finds_internal_color_boundary():
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[:, :60] = [45, 45, 45]
    image[:, 60:] = [220, 220, 220]

    edges = sobel_edge_map(image)

    assert edges[:, 59:62].mean() > .75
    assert edges[:, 15:45].mean() < .05


def test_edge_guided_region_does_not_cross_strong_boundary():
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[:, :60] = [50, 70, 80]
    image[:, 60:] = [180, 190, 195]
    allowed = np.ones((80, 120), dtype=bool)
    edges = sobel_edge_map(image)

    region = edge_guided_region(
        image,
        allowed,
        seed_xy=(25, 40),
        edges=edges,
        edge_threshold=.45,
        color_tolerance=.55,
    )

    assert region[:, :55].mean() > .90
    assert region[:, 65:].mean() < .02


def test_boundary_adherence_scores_mask_on_real_edge_higher():
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[:, :60] = [30, 30, 30]
    image[:, 60:] = [230, 230, 230]
    edges = sobel_edge_map(image)

    aligned = np.zeros((80, 120), dtype=bool)
    aligned[10:70, 10:60] = True
    arbitrary = np.zeros((80, 120), dtype=bool)
    arbitrary[10:70, 10:42] = True

    assert boundary_adherence(aligned, edges) > boundary_adherence(arbitrary, edges)
