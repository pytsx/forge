"""Compare original dense projection with bounded temporary buffers.

Run: python scripts/benchmark_projection.py --resolution 128 --repeats 3
Reports traced allocation peaks, not total process RSS or whole-pipeline speed.
"""
import argparse
import gc
import hashlib
import json
import platform
import statistics
import time
import tracemalloc
from uuid import uuid4

import numpy as np

from dollforge.domain.models import CameraEstimate, Provenance
from dollforge.volumetry.projection import project_world_points
from dollforge.volumetry.visual_hull import _grid, _inside_mask


def dense_grid(center, extents, resolution):
    size = float(max(extents) * 1.12) / resolution
    minimum = center - max(extents) * 1.12 / 2 + size / 2
    axes = [minimum[i] + np.arange(resolution, dtype=np.float64) * size for i in range(3)]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]), minimum, size


def dense_inside(points, camera, mask):
    pixels = project_world_points(points, camera)
    u = np.rint(pixels[:, 0]).astype(np.int64)
    v = np.rint(pixels[:, 1]).astype(np.int64)
    height, width = mask.shape
    valid = (u >= 0) & (v >= 0) & (u < width) & (v < height)
    inside = np.zeros(len(points), dtype=bool)
    inside[valid] = mask[v[valid], u[valid]]
    return inside


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, choices=range(32, 129), default=128)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    center, extents = np.zeros(3), np.array([40., 30., 60.])
    camera = CameraEstimate(view_id=uuid4(), label="front", yaw_deg=0, calibrated=True,
                            world_units_per_pixel=.5, principal_point_px=(128., 128.),
                            world_from_view=np.eye(4).ravel().tolist(),
                            provenance=Provenance(type="rule_based", source="benchmark"))
    mask = np.random.default_rng(42).random((256, 256)) > .5
    output = {"resolution": args.resolution, "repeats": args.repeats,
              "python": platform.python_version(), "numpy": np.__version__}
    for name, grid, inside in (("dense", dense_grid, dense_inside),
                               ("bounded", _grid, _inside_mask)):
        times, peaks, hashes = [], [], []
        for _ in range(args.repeats):
            gc.collect()
            tracemalloc.start()
            started = time.perf_counter()
            points, _, _ = grid(center, extents, args.resolution)
            result = inside(points, camera, mask)
            times.append(time.perf_counter() - started)
            peaks.append(tracemalloc.get_traced_memory()[1] / 1024**2)
            tracemalloc.stop()
            hashes.append(hashlib.sha256(result.tobytes()).hexdigest())
            del points, result
        assert len(set(hashes)) == 1
        output[name] = {"median_seconds": round(statistics.median(times), 4),
                        "peak_traced_mib": round(max(peaks), 2), "sha256": hashes[0]}
    output["identical"] = output["dense"]["sha256"] == output["bounded"]["sha256"]
    print(json.dumps(output, indent=2))
    if not output["identical"]:
        raise SystemExit("Projection output changed")


if __name__ == "__main__":
    main()
