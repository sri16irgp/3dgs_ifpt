"""Checkpoint-aligned 3D instance evaluation for LBG results."""
import json
from pathlib import Path

import numpy as np

from .io import read_result, verify_checkpoint


def reference_mask(checkpoint, reference):
    """Map an exact PLY subset back to its source checkpoint ordering."""
    from plyfile import PlyData
    from scipy.spatial import cKDTree

    source = PlyData.read(str(checkpoint))["vertex"].data
    target = PlyData.read(str(reference))["vertex"].data
    if source.dtype != target.dtype:
        raise ValueError(f"PLY schema differs from checkpoint: {reference}")
    source_xyz = np.column_stack((source["x"], source["y"], source["z"]))
    target_xyz = np.column_stack((target["x"], target["y"], target["z"]))
    distance, indices = cKDTree(source_xyz).query(target_xyz, k=1)
    if np.any(distance != 0):
        raise ValueError(f"Reference is not an exact checkpoint subset: {reference}")
    # XYZ could theoretically collide. Comparing complete structured records
    # makes the accepted mapping exact rather than proximity-based.
    if not np.array_equal(source[indices], target):
        raise ValueError(f"Reference records differ from checkpoint: {reference}")
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"Reference contains duplicate checkpoint records: {reference}")
    mask = np.zeros(len(source), dtype=bool)
    mask[indices] = True
    return mask


def instance_metrics(labels, truth):
    """Score each nonempty instance and a greedy union against one target."""
    labels = np.asarray(labels)
    truth = np.asarray(truth, dtype=bool)
    if labels.shape != truth.shape or labels.ndim != 1:
        raise ValueError("Labels and truth must be matching one-dimensional arrays")
    truth_count = int(truth.sum())
    if not truth_count:
        raise ValueError("Reference target is empty")
    valid = labels >= 0
    count = int(labels[valid].max()) + 1 if valid.any() else 0
    predicted = np.bincount(labels[valid], minlength=count)
    intersection = np.bincount(labels[truth & valid], minlength=count)
    union = predicted + truth_count - intersection
    iou = np.divide(intersection, union, out=np.zeros(count, dtype=float), where=union > 0)
    precision = np.divide(intersection, predicted, out=np.zeros(count, dtype=float), where=predicted > 0)
    recall = intersection / truth_count
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(count, dtype=float), where=(precision + recall) > 0)
    records = [dict(instance=int(i), iou=float(iou[i]), precision=float(precision[i]),
                    recall=float(recall[i]), f1=float(f1[i]), intersection=int(intersection[i]),
                    predicted=int(predicted[i])) for i in range(count) if predicted[i] > 0]
    records.sort(key=lambda item: (-item["iou"], item["instance"]))

    # Quantifies fragmentation without allowing arbitrary false-positive growth:
    # add the instance that most improves union IoU, stopping at the optimum.
    selected = np.zeros_like(truth)
    remaining = {record["instance"] for record in records}
    chosen, best_iou = [], 0.0
    while remaining:
        candidate = None
        candidate_iou = best_iou
        for instance in remaining:
            merged = selected | (labels == instance)
            overlap = int(np.count_nonzero(merged & truth))
            score = overlap / int(np.count_nonzero(merged | truth))
            if score > candidate_iou + 1e-12 or (score == candidate_iou and candidate is not None and instance < candidate):
                candidate, candidate_iou = instance, score
        if candidate is None:
            break
        selected |= labels == candidate
        remaining.remove(candidate)
        chosen.append(int(candidate))
        best_iou = candidate_iou
    overlap = int(np.count_nonzero(selected & truth))
    selected_count = int(selected.sum())
    greedy = dict(instances=chosen, iou=float(best_iou),
                  precision=overlap / selected_count if selected_count else 0.0,
                  recall=overlap / truth_count, intersection=overlap,
                  predicted=selected_count)
    return dict(reference_gaussians=truth_count,
                assigned_reference=int(np.count_nonzero(truth & valid)),
                best=records[0] if records else None, greedy_union=greedy,
                top_instances=records[:10])


def evaluate_result(result, references, level="object", output=None):
    metadata, arrays = read_result(result)
    checkpoint = verify_checkpoint(metadata)
    key = level + "_labels"
    if key not in arrays:
        raise ValueError(f"Result does not contain level: {level}")
    report = {
        "result": str(Path(result).resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "level": level,
        "targets": {},
    }
    for reference in references:
        path = Path(reference)
        report["targets"][path.stem] = instance_metrics(
            arrays[key], reference_mask(checkpoint, path))
    rendered = json.dumps(report, indent=2)
    if output:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(rendered + "\n")
        temporary.replace(destination)
    return report
