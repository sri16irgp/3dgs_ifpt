"""Parameter sweeps over reusable LBG frame caches."""
import itertools
import json
from pathlib import Path

import numpy as np

from .core import Fusion, lift_masks
from .evaluation import instance_metrics, reference_mask
from .io import load_level, read_result, verify_checkpoint


def cached_paths(cache, image_names, level="object"):
    """Resolve one cached frame per image name in original fusion order."""
    matches = {name: [] for name in image_names}
    for path in Path(cache).glob("*.npz"):
        with np.load(path, allow_pickle=False) as data:
            if "metadata" not in data or level + "_masks" not in data:
                continue
            metadata = json.loads(str(data["metadata"]))
        name = metadata.get("image_name")
        if name in matches:
            matches[name].append(path)
    ambiguous = {name: paths for name, paths in matches.items() if len(paths) != 1}
    if ambiguous:
        summary = ", ".join(f"{name} ({len(paths)})" for name, paths in ambiguous.items())
        raise ValueError(f"Expected exactly one compatible cache per source view: {summary}")
    return [matches[name][0] for name in image_names]


def tune_cached(result, cache, references, geometry_values, semantic_values,
                weight_values, min_gaussian_values, output=None):
    """Grid-search fusion settings using a fixed set of 2D proposals/features."""
    metadata, _ = read_result(result)
    checkpoint = verify_checkpoint(metadata)
    paths = cached_paths(cache, metadata["views"])
    frames = [load_level(path, "object") for path in paths]
    truths = {Path(reference).stem: reference_mask(checkpoint, reference)
              for reference in references}
    count = metadata["gaussian_count"]
    trials = []
    combinations = itertools.product(geometry_values, semantic_values,
                                     weight_values, min_gaussian_values)
    for geometry, semantic, weight, minimum in combinations:
        fusion = Fusion(count, geometry, semantic)
        for frame in frames:
            fusion.add_frame(lift_masks(*frame, min_weight=weight,
                                        min_gaussians=minimum))
        targets = {name: instance_metrics(fusion.labels, truth)
                   for name, truth in truths.items()}
        single = [target["best"]["iou"] if target["best"] else 0.0
                  for target in targets.values()]
        greedy = [target["greedy_union"]["iou"] for target in targets.values()]
        trials.append({
            "geometry_threshold": geometry,
            "semantic_threshold": semantic,
            "min_weight": weight,
            "min_gaussians": minimum,
            "instances": sum(1 for record in fusion.records() if record["gaussians"]),
            "mean_best_iou": float(np.mean(single)),
            "mean_greedy_iou": float(np.mean(greedy)),
            "targets": targets,
        })
    trials.sort(key=lambda trial: (-trial["mean_best_iou"],
                                   -trial["mean_greedy_iou"],
                                   trial["instances"],
                                   trial["geometry_threshold"],
                                   trial["semantic_threshold"]))
    report = {
        "source_result": str(Path(result).resolve()),
        "cache": str(Path(cache).resolve()),
        "views": metadata["views"],
        "objective": "mean best-single-instance 3D IoU across references",
        "best": trials[0] if trials else None,
        "trials": trials,
    }
    if output:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(destination)
    return report
