import unittest
import tempfile
from pathlib import Path
import numpy as np
from lifting.core import Fragment, Fusion, Instance, lift_masks, propagate, refine
from lifting.io import save_frame, load_level, fingerprint


class LiftingTests(unittest.TestCase):
    def test_max_ids_background_and_majority(self):
        ids = np.array([[0, 0, 1, -1], [0, 2, 2, 3]])
        masks = np.array([[[1, 1, 0, 0], [0, 0, 0, 0]], [[0, 0, 1, 1], [1, 1, 1, 1]]], bool)
        weights = np.ones_like(ids, float)
        weights[-1, -1] = 0
        fragments = lift_masks(ids, weights, masks, [1, .9], np.eye(2), np.eye(2), min_gaussians=1)
        self.assertEqual([f.ids.tolist() for f in fragments], [[0], [1, 2]])

    def test_parent_partition_and_overlap(self):
        ids = np.arange(4).reshape(2, 2)
        fragments = lift_masks(ids, np.ones((2, 2)), np.ones((2, 2, 2), bool), [.8, .9], np.eye(2), np.eye(2),
                               min_gaussians=1, parent_labels=np.array([0, 0, 1, -1]))
        self.assertEqual([(f.parent, f.ids.tolist()) for f in fragments], [(0, [0, 1]), (1, [2])])
        np.testing.assert_array_equal(fragments[0].clip, [0, 1])

    def test_empty_masks(self):
        self.assertEqual(lift_masks(np.zeros((2, 2), int), np.ones((2, 2)), np.empty((0, 2, 2), bool), [],
                                    np.empty((0, 2)), np.empty((0, 2))), [])

    def test_previous_frame_only_and_running_average(self):
        fusion = Fusion(6, .25, .5)
        def fragment(ids, feature=(1., 0.)):
            return Fragment(np.array(ids), np.array(feature), np.array(feature))
        fusion.add_frame([fragment([0, 1]), fragment([2, 3])])
        self.assertEqual(len(fusion.instances), 2)
        fusion.add_frame([fragment([1, 4], (.8, .6))])
        self.assertEqual(fusion.labels[4], 0)
        np.testing.assert_allclose(fusion.instances[0].clip, [.9, .3])
        fusion.add_frame([fragment([2, 5], (-1., 0.))])
        self.assertEqual(fusion.labels[5], 2)
        self.assertEqual(fusion.labels[2], 1)  # conflicting evidence ties preserve first label

    def test_no_same_frame_merge(self):
        f = Fragment(np.array([0, 1]), np.array([1., 0.]), np.array([1., 0.]))
        fusion = Fusion(2)
        fusion.add_frame([f, f])
        self.assertEqual(len(fusion.instances), 2)

    def test_parent_blocks_merge(self):
        fusion = Fusion(3)
        fusion.add_frame([Fragment(np.array([0, 1]), np.ones(2), np.ones(2), 0)])
        fusion.add_frame([Fragment(np.array([1, 2]), np.ones(2), np.ones(2), 1)])
        self.assertEqual(len(fusion.instances), 2)

    def test_refinement_split(self):
        rng = np.random.default_rng(3)
        xyz = np.concatenate([rng.normal(0, .001, (40, 3)), rng.normal(1, .001, (40, 3)), [[8, 8, 8]]])
        labels, objects = refine(np.zeros(81, np.int32), xyz, [Instance(0, 5, np.ones(2), np.ones(2))], .02, 10)
        self.assertEqual(len(objects), 2)
        self.assertEqual(labels[-1], -1)
        self.assertEqual({o.parent for o in objects}, {5})
        self.assertNotEqual(labels[0], labels[40])

    def test_conservative_propagation_and_parent_constraint(self):
        xyz = np.array([[0, 0, 0], [.01, 0, 0], [.02, 0, 0],
                        [1, 0, 0], [1.01, 0, 0], [1.02, 0, 0]], float)
        labels = np.array([0, 0, -1, 1, 1, -1])
        filled = propagate(labels, xyz, .03, neighbors=2, consensus=1)
        np.testing.assert_array_equal(filled, [0, 0, 0, 1, 1, 1])
        # Child 0 belongs to parent 4; the nearby unlabeled point belongs to
        # parent 5 and therefore must remain untouched.
        child = np.array([0, 0, -1, -1, -1, -1])
        parents = np.array([4, 4, 5, 5, 5, 5])
        instances = [Instance(0, 4, np.ones(2), np.ones(2))]
        constrained = propagate(child, xyz, .03, neighbors=2, consensus=1,
                                parent_labels=parents, instances=instances)
        self.assertEqual(constrained[2], -1)

    def test_propagation_validation(self):
        with self.assertRaises(ValueError):
            propagate(np.zeros(2), np.zeros((2, 3)), 0)
        with self.assertRaises(ValueError):
            propagate(np.zeros(2), np.zeros((3, 3)), 1)

    def test_cache_roundtrip_empty_and_nonempty(self):
        with tempfile.TemporaryDirectory() as temp:
            for count in (0, 2):
                masks = np.ones((count, 3, 5), bool)
                path = Path(temp) / 'frame.npz'
                save_frame(path, np.zeros((3, 5), int), np.ones((3, 5)),
                           {'object': (masks, np.ones(count), np.ones((count, 2)), np.ones((count, 3)))}, {'version': 1})
                np.testing.assert_array_equal(load_level(path, 'object')[2], masks)
        self.assertNotEqual(fingerprint({'image': 1}), fingerprint({'image': 2}))


if __name__ == '__main__':
    unittest.main()
