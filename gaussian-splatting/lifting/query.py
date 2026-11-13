"""Shared text-to-instance ranking for the CLI and viewer."""
import numpy as np
from .core import unit


def rank_instances(features, records, encoded, top_k=5):
    features = np.asarray(features)
    if not len(features):
        return []
    if features.ndim != 2 or len(records) != len(features):
        raise ValueError('Instance features and metadata disagree')
    scores = features @ unit(np.asarray(encoded))
    valid = [record for record in records if record['gaussians'] > 0]
    ranked = sorted(valid, key=lambda record: (-float(scores[record['id']]), record['id']))
    return [dict(record, cosine=float(scores[record['id']])) for record in ranked[:top_k]]
