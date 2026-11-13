import math
import unittest

import numpy as np

from lifting.query import rank_instances
from lifting.viewer import camera_parameters, frame_size, validate_pose


class ViewerTests(unittest.TestCase):
    def test_query_skips_empty_and_breaks_ties_by_id(self):
        records = [dict(id=0, gaussians=0), dict(id=1, gaussians=20), dict(id=2, gaussians=10)]
        features = np.array([[1., 0.], [0., 1.], [0., 1.]])
        self.assertEqual([r['id'] for r in rank_instances(features, records, [0, 1])], [1, 2])
        self.assertEqual(rank_instances(np.empty((0, 2)), [], [1, 0]), [])

    def test_saved_camera_focal_lengths_and_pose(self):
        saved = dict(width=800, height=600, fx=400, fy=300,
                     position=[1, 2, 3], rotation=np.eye(3).tolist())
        result = camera_parameters(saved)
        self.assertAlmostEqual(result['fov_x'], math.pi / 2)
        self.assertAlmostEqual(result['fov_y'], math.pi / 2)
        np.testing.assert_allclose(result['rotation'], np.eye(3))
        validate_pose(result['position'], result['rotation'])
        with self.assertRaises(ValueError):
            validate_pose([0, 0, 0], [[0, 0, 0]] * 3)

    def test_frame_caps_preserve_aspect(self):
        self.assertEqual(frame_size(1600, 900, 800), (800, 450))
        self.assertEqual(frame_size(600, 400, 800), (600, 400))
        with self.assertRaises(ValueError):
            frame_size(0, 400, 800)


if __name__ == '__main__':
    unittest.main()
