import numpy as np

from dollforge.volumetry.sdf import extract_zero_surface, signed_distance_field


def test_sdf_sign_convention_and_zero_surface():
    occupancy = np.zeros((16, 16, 16), dtype=bool)
    occupancy[4:12, 4:12, 4:12] = True
    sdf = signed_distance_field(occupancy, voxel_size=1.0)

    assert sdf[8, 8, 8] < 0
    assert sdf[0, 0, 0] > 0
    assert sdf[3, 8, 8] > 0
    assert sdf[4, 8, 8] < 0

    vertices, faces = extract_zero_surface(sdf, (0.0, 0.0, 0.0), 1.0)
    assert len(vertices) > 0
    assert len(faces) > 0
