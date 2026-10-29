import unittest

import numpy as np

from repro.geometry import (front_reference_indices, inference_view_indices,
                            local_views, render, rotation_matrix)


class RenderingContractTests(unittest.TestCase):
    def test_view_set_is_fixed_and_unique(self):
        views = local_views()
        self.assertEqual(len(views), 27)
        self.assertEqual(len({v.angles for v in views}), 27)
        ids = inference_view_indices()
        self.assertEqual(len(ids), 15)
        self.assertEqual(len(set(ids)), 15)
        self.assertEqual(ids[0], 0)

    def test_front_point_wins_depth_buffer_and_keeps_raw_identity(self):
        # Points 0 and 1 share xy; point 1 is closer to the camera.
        points = np.array([[0.0, 0.0, -0.3], [0.0, 0.0, 0.3],
                           [0.4, 0.0, 0.0]], dtype=np.float32)
        rgb, uv, visible, winners = render(points, rotation_matrix(0, 0, 0),
                                            size=64, visibility_tolerance=0.01)
        self.assertEqual(rgb.shape, (64, 64, 3))
        np.testing.assert_array_equal(uv[0], uv[1])
        self.assertFalse(visible[0])
        self.assertTrue(visible[1])
        self.assertIn(1, winners)
        self.assertNotIn(0, winners)
        references = front_reference_indices(points, rotation_matrix(0, 0, 0), 64)
        self.assertIn(1, references)
        self.assertNotIn(0, references)


if __name__ == "__main__":
    unittest.main()
