import io
import unittest
from zipfile import ZipFile

import numpy as np

from repro.full_surface import matched_labels
from repro.geometry_memory import local_shape, local_shape_radius


class FullSurfaceContracts(unittest.TestCase):
    def test_reordered_duplicate_gt_and_unannotated_point(self):
        raw = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
        buffer = io.BytesIO()
        with ZipFile(buffer, "w") as archive:
            archive.writestr("Real3D-AD-PCD/airplane/gt/123_bulge.txt",
                             "1 0 0 0\n0 0 0 1\n0 0 0 1\n")
        buffer.seek(0)
        with ZipFile(buffer) as archive:
            labels = matched_labels(archive,
                                    "Real3D-AD-PCD/airplane/test/123_bulge.pcd",
                                    raw, "bulge")
        self.assertEqual(labels.tolist(), [1, 0, -1])

    def test_local_shape_is_invariant_to_uniform_scale(self):
        rng = np.random.default_rng(1)
        points = rng.normal(size=(300, 3))
        references = points[:20]
        a = local_shape(points, references)
        b = local_shape(points * 3, references * 3)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)

    def test_radius_shape_uses_fixed_spatial_support(self):
        rng = np.random.default_rng(2)
        points = rng.uniform(-0.5, 0.5, size=(500, 3))
        references = points[:15]
        a = local_shape_radius(points, references, radius=0.4)
        b = local_shape_radius(points * 2, references * 2, radius=0.8)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
