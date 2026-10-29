import unittest

import numpy as np

from repro.coverage_ablate import greedy_coverage, selected


class CoverageSelectionTests(unittest.TestCase):
    def test_greedy_adds_new_reference_points(self):
        visible = np.zeros((27, 5), dtype=bool)
        visible[0, :2] = True
        visible[1, :3] = True
        visible[2, 2:] = True
        self.assertEqual(greedy_coverage(visible, 2), (0, 2))
        self.assertEqual(greedy_coverage(visible, 3)[:2], (0, 2))
        self.assertEqual(len(set(greedy_coverage(visible, 3))), 3)

    def test_selection_never_uses_labels(self):
        visible = np.zeros((27, 3), dtype=bool)
        visible[0, 0] = True
        visible[5, 1:] = True
        methods = selected(visible, tuple(range(15)), tuple(range(15)), 2)
        self.assertEqual(methods["object_greedy_27"], (0, 5))
        self.assertTrue(all(len(ids) == 2 for ids in methods.values()))


if __name__ == "__main__":
    unittest.main()
