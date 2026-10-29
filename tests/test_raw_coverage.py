import unittest

import numpy as np

from repro.raw_coverage import visibility_for_sample


class RawCoverageTests(unittest.TestCase):
    def test_z_buffer_hides_deeper_point_at_same_pixel(self):
        raw = np.array([[0, 0, 0], [0, 0, 0.1]], dtype=np.float32)
        intrinsic = np.array([[500, 0, 335.5], [0, 500, 335.5], [0, 0, 1]])
        extrinsic = np.eye(4)
        extrinsic[2, 3] = 2.5
        camera = {"rotation": np.eye(3).tolist(),
                  "intrinsic_672": intrinsic.tolist(),
                  "extrinsic": extrinsic.tolist()}
        sample = {"basis": np.eye(3).tolist(),
                  "oriented_center": [0, 0, 0],
                  "normalization_radius": 1}
        result = visibility_for_sample(raw, sample, [camera])
        self.assertEqual(result.shape, (1, 2))
        self.assertEqual(result[0].tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
