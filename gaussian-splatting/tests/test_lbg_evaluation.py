import unittest

import numpy as np

from lifting.evaluation import instance_metrics
from lifting.tuning import cached_paths


class EvaluationTests(unittest.TestCase):
    def test_best_instance_and_fragment_union(self):
        truth = np.array([1, 1, 1, 1, 0, 0, 0, 0], bool)
        labels = np.array([0, 0, 1, 1, 0, 2, -1, -1])
        report = instance_metrics(labels, truth)
        self.assertEqual(report['reference_gaussians'], 4)
        self.assertEqual(report['assigned_reference'], 4)
        self.assertEqual(report['best']['instance'], 1)
        self.assertAlmostEqual(report['best']['iou'], 0.5)
        self.assertEqual(report['greedy_union']['instances'], [1, 0])
        self.assertAlmostEqual(report['greedy_union']['iou'], 0.8)

    def test_unassigned_result(self):
        report = instance_metrics(np.full(3, -1), np.array([1, 0, 1], bool))
        self.assertIsNone(report['best'])
        self.assertEqual(report['greedy_union']['instances'], [])
        self.assertEqual(report['assigned_reference'], 0)

    def test_rejects_shape_mismatch_and_empty_truth(self):
        with self.assertRaises(ValueError):
            instance_metrics(np.zeros(2), np.zeros(3, bool))
        with self.assertRaises(ValueError):
            instance_metrics(np.zeros(2), np.zeros(2, bool))

    def test_cached_paths_preserve_requested_order(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for filename, image in (("z.npz", "second"), ("a.npz", "first")):
                np.savez(root / filename, metadata=np.array(json.dumps({"image_name": image})),
                         object_masks=np.empty((0, 0), np.uint8))
            paths = cached_paths(root, ["first", "second"])
            self.assertEqual([path.name for path in paths], ["a.npz", "z.npz"])
            with self.assertRaisesRegex(ValueError, "exactly one"):
                cached_paths(root, ["missing"])


if __name__ == '__main__':
    unittest.main()
