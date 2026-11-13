import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from lifting.io import digest_file, save_npz, export_assets, verify_checkpoint
from lifting.models import mask_nms


class MaskTests(unittest.TestCase):
    def test_mask_nms(self):
        a = np.zeros((4, 4), bool)
        a[:2, :2] = True
        b = np.zeros_like(a)
        b[2:, 2:] = True
        self.assertEqual(mask_nms(np.stack([a, a, b]), [0.7, .9, .8]), [1, 2])
        self.assertEqual(mask_nms(np.empty((0, 4, 4), bool), []), [])


@unittest.skipUnless(importlib.util.find_spec('plyfile'), 'plyfile required')
class AssetTests(unittest.TestCase):
    def test_exact_ply_subset_and_checkpoint_guard(self):
        from plyfile import PlyData, PlyElement
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vertices = np.zeros(4, dtype=[(name, 'f4') for name in
                ['x', 'y', 'z', 'opacity', 'scale_0', 'rot_0', 'f_dc_0', 'f_rest_0']])
            for i, name in enumerate(vertices.dtype.names):
                vertices[name] = np.arange(4) + i / 10
            checkpoint = root / 'original.ply'
            PlyData([PlyElement.describe(vertices, 'vertex')]).write(str(checkpoint))
            metadata = dict(format_version=1, checkpoint=str(checkpoint), checkpoint_sha256=digest_file(checkpoint))
            (root / 'scene.json').write_text(json.dumps(metadata))
            save_npz(root / 'segmentation.npz', object_labels=np.array([0, -1, 1, 0]))
            paths = export_assets(root, instance=0)
            np.testing.assert_array_equal(PlyData.read(paths[0])['vertex'].data, vertices[[0, 3]])
            self.assertEqual(digest_file(checkpoint), metadata['checkpoint_sha256'])
            with self.assertRaises(ValueError):
                export_assets(root, instance=99)
            with checkpoint.open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'Checkpoint differs'):
                verify_checkpoint(metadata)


if __name__ == '__main__':
    unittest.main()
