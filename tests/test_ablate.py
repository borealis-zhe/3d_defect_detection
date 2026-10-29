import unittest

import numpy as np

from repro.ablate import fused_scores


class FusionContractTests(unittest.TestCase):
    def test_baseline_visibility_and_unseen_fallback(self):
        scores = np.array([[1, 1, 2], [3, 8, 4], [5, 9, 6]], dtype=np.float32)
        visible = np.array([[True, False, False],
                            [True, False, False],
                            [False, True, False]])
        fused = fused_scores(scores, visible)
        np.testing.assert_allclose(fused["baseline_uniform"], [3, 6, 4])
        np.testing.assert_allclose(fused["visible_mean_control"], [2, 9, 4])
        np.testing.assert_allclose(fused["view_consensus"], [2, 9, 4])
        self.assertGreater(fused["view_disagreement"][0], 2)
        self.assertEqual(fused["view_disagreement"][2], 4)

    def test_fusion_is_invariant_to_view_order(self):
        scores = np.array([[0.1, 0.7], [0.8, 0.2], [0.3, 0.5]], dtype=np.float32)
        visible = np.array([[True, True], [False, True], [True, True]])
        first = fused_scores(scores, visible)
        second = fused_scores(scores[[2, 0, 1]], visible[[2, 0, 1]])
        for method in first:
            np.testing.assert_allclose(first[method], second[method], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
