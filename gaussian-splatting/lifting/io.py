"""Portable, non-pickle caches and checkpoint-bound segmentation artifacts."""
import hashlib
import json
from pathlib import Path
import numpy as np


def digest_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def save_frame(path, ids, weights, levels, metadata):
    arrays = dict(ids=ids.astype(np.int32), weights=weights.astype(np.float32), metadata=np.array(json.dumps(metadata)))
    for name, (masks, scores, clip, dino) in levels.items():
        arrays[name + '_masks'] = np.packbits(masks.reshape(len(masks), ids.size), axis=1)
        arrays[name + '_scores'] = scores
        arrays[name + '_clip'] = clip
        arrays[name + '_dino'] = dino
    save_npz(path, **arrays)


def load_level(path, level):
    with np.load(path, allow_pickle=False) as data:
        ids, weights = data['ids'], data['weights']
        packed = data[level + '_masks']
        masks = np.unpackbits(packed, axis=1, count=ids.size).reshape(len(packed), *ids.shape).astype(bool)
        return ids, weights, masks, data[level + '_scores'], data[level + '_clip'], data[level + '_dino']


def read_result(directory):
    directory = Path(directory)
    metadata = json.loads((directory / 'scene.json').read_text())
    with np.load(directory / 'segmentation.npz', allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    if metadata['format_version'] != 1:
        raise ValueError('Unsupported LBG result version')
    return metadata, arrays


def verify_checkpoint(metadata, path=None):
    path = Path(path or metadata['checkpoint'])
    if digest_file(path) != metadata['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from the segmented checkpoint; Gaussian ordering must match')
    return path


def export_assets(directory, level='object', instance=None, destination=None):
    from plyfile import PlyData, PlyElement
    metadata, arrays = read_result(directory)
    checkpoint = verify_checkpoint(metadata)
    labels = arrays[level + '_labels']
    ply = PlyData.read(str(checkpoint))
    if len(ply['vertex']) != len(labels):
        raise ValueError('Label count does not match checkpoint')
    destination = Path(destination or Path(directory) / 'assets' / level)
    destination.mkdir(parents=True, exist_ok=True)
    choices = [instance] if instance is not None else np.unique(labels[labels >= 0]).tolist()
    paths = []
    for label in choices:
        selection = labels == label
        if not selection.any():
            raise ValueError(f'No Gaussians for {level} instance {label}')
        target = destination / f'{level}_{label:05d}.ply'
        if target.resolve() == checkpoint.resolve():
            raise ValueError('Cannot overwrite input checkpoint')
        vertices = ply['vertex'].data[selection].copy()
        PlyData([PlyElement.describe(vertices, 'vertex')], text=False).write(str(target))
        paths.append(str(target))
    return paths
