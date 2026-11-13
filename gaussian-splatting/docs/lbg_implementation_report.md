# Lifting by Gaussians: Implementation Report

Date: 2026-11-04  
Repository: `gaussian-splatting`  
Reference: *Lifting by Gaussians: A Simple, Fast and Flexible Method for 3D Instance Segmentation*

## Goal

This work extends the original 3D Gaussian Splatting repository with a
training-free 3D instance-segmentation pipeline based on Lifting by Gaussians
(LBG). The implementation consumes:

1. an existing trained 3DGS checkpoint;
2. the posed RGB training images already used by the scene; and
3. pretrained SAM, CLIP, and DINOv2 models.

It does not retrain or modify the Gaussian reconstruction. Instead, it extracts
2D masks and features, assigns each image pixel to its strongest contributing
Gaussian, merges the resulting 3D fragments across views, and produces an
object/part/subpart hierarchy.

## What was added

### Exact pixel-to-Gaussian contribution lookup

The local CUDA rasterizer was extended with an inference-only pass that returns:

- the Gaussian with maximum alpha-compositing contribution for every pixel;
- the corresponding `alpha * transmittance` weight; and
- `-1` with weight zero for background pixels.

The pass reuses the original rasterizer's sorted Gaussian buffers and matches
its opacity clamp, alpha cutoff, early transmittance termination, covariance,
and antialiasing behavior. The existing RGB/depth return values and backward
gradient interface remain unchanged.

Relevant code:

- `submodules/diff-gaussian-rasterization/rasterize_points.cu`
- `submodules/diff-gaussian-rasterization/rasterize_points.h`
- `submodules/diff-gaussian-rasterization/ext.cpp`
- `submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py`
- `gaussian_renderer/__init__.py`

### Foundation-model preprocessing

For every selected training image, the pipeline:

1. samples a regular point grid;
2. obtains SAM's three mask hypotheses for every point;
3. orders those hypotheses by area into object, part, and subpart levels;
4. filters low-quality and unstable masks;
5. removes duplicate masks with mask-IoU non-maximum suppression;
6. encodes masked crops with CLIP; and
7. pools DINOv2 patch features inside each mask.

CLIP and DINO features are normalized independently. The SAM hierarchy ordering
is an explicit implementation heuristic: SAM itself does not promise that its
three hypotheses represent semantic object/part/subpart levels.

Relevant code: `lifting/models.py`.

### 2D mask lifting

For each mask, pixels are mapped to the maximum-contributing Gaussian. Mask
overlap is resolved by SAM score, and repeated pixel evidence is reduced to one
Gaussian assignment by majority vote. Weak contributors and very small
fragments can be discarded.

This produces sparse per-view 3D fragments without optimizing new Gaussian
features or allocating a dense Gaussian-by-instance matrix.

Relevant code: `lifting/core.py`, function `lift_masks`.

### Incremental multi-view fusion

Incoming fragments are matched against the semantic map from the previous
frame using:

- geometric overlap: intersection divided by incoming-fragment size;
- semantic similarity: mean CLIP and DINO cosine similarity; and
- a combined ranking score: `overlap * (1 + similarity) / 2`.

Fragments below the configured geometric or semantic thresholds start new
instances. Matched instance features use a fragment-count running average.
Sparse per-Gaussian vote counts resolve conflicting assignments; equal evidence
preserves the earlier owner.

Relevant code: `lifting/core.py`, class `Fusion`.

### Hierarchical decomposition

The complete object map is built first. Part fragments are intersected with the
completed object labels and can merge only within their parent object. Subparts
are handled the same way inside completed parts.

This guarantees that a child instance cannot cross a parent boundary. The final
plant run verified this invariant for every assigned part and subpart Gaussian.

### Cleanup and completion

Two optional post-processing operations were implemented:

- `refine`: statistical outlier filtering, connected-component splitting, and
  conservative merging of small components;
- `propagate`: fills only unlabeled Gaussians when a configurable fraction of
  their nearest labeled neighbors agree within a scene-space radius.

Propagation never changes an existing label. At part and subpart levels it is
restricted to the already assigned parent, so it cannot violate the hierarchy.

### Caching and reproducibility

Every processed frame is saved as a compressed NPZ containing contributor IDs,
weights, bit-packed masks, mask scores, and CLIP/DINO features. Cache keys include
the checkpoint hash, resized RGB contents, camera matrices, model configuration,
SAM checkpoint hash, hierarchy level count, and rasterization mode.

Changing fusion thresholds or post-processing parameters therefore requires
only a CPU replay of cached fragments. `--cached-only` fails if any required
cache is absent and does not load the foundation models.

Relevant code: `lifting/io.py` and `lifting/pipeline.py`.

### Evaluation and parameter tuning

An evaluator maps reference PLY assets back to their exact checkpoint indices.
The plant references were confirmed to be exact, unique subsets of the same
2,004,727-Gaussian checkpoint:

- foliage: 128,110 Gaussians;
- pot: 2,207 Gaussians.

For every predicted instance, evaluation reports intersection, predicted size,
precision, recall, F1, and 3D IoU. It also reports a greedy union score to expose
fragmentation. A cached grid-search command evaluates fusion thresholds without
recomputing SAM, CLIP, or DINO features.

Relevant code:

- `lifting/evaluation.py`
- `lifting/tuning.py`
- `lbg.py evaluate`
- `lbg.py tune`

## Command-line interface

The entry point is `lbg.py`:

- `run`: preprocess, lift, fuse, refine, propagate, and save a segmentation;
- `render`: render hard instance maps or isolated RGB instances;
- `export`: export selected instances as lossless Gaussian PLY subsets;
- `query`: rank instances by CLIP text similarity;
- `evaluate`: compare instances with exact checkpoint-subset PLY references;
- `tune`: sweep fusion settings over existing frame caches.

Dependencies specific to this extension are documented in
`requirements-lbg.txt`. Detailed general usage is in `docs/lbg.md`.

## Plant-scene validation

### Initial sampled-view run

The first validation used 14 training views. It proved that the full workflow,
caching, querying, rendering, and asset export functioned, but reference recall
was low because only a small fraction of Gaussians had received evidence.

At object level, the best foliage instance had:

- IoU: 0.0864;
- precision: 0.8221;
- recall: 0.0880.

The pot was merged into the plant at object level. At part level, however, a
separate pot segment achieved 0.1188 IoU with 0.7662 precision. This established
that the pot belongs at the part level in this hierarchy.

### Threshold sweep

A 180-combination sweep over the 14-view cache tested geometric threshold,
semantic threshold, contribution weight, and minimum fragment size. The chosen
fusion settings were:

- geometric threshold: 0.05;
- semantic threshold: 0.4;
- minimum contribution weight: 0.005;
- minimum fragment size: 5 Gaussians.

These settings merged foliage fragments more effectively while retaining high
precision.

### Full hierarchy run

The final hierarchy used all 139 training views and kept 20 LLFF views held out.
It used 640-pixel image width, a 16×16 SAM point grid, and the tuned fusion
settings above.

Runtime on the NVIDIA Thor was 877.5 seconds. The result contained:

- 6 object instances;
- 53 part instances; and
- 175 subpart instances.

Before propagation, the best results were:

| Target | Level | Instance | IoU | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: |
| Foliage | Object | 3 | 0.2333 | 0.8099 | 0.2468 |
| Pot | Part | 6 | 0.2688 | 0.7862 | 0.2900 |

### Parent-constrained propagation

The maximum-contributor strategy leaves many never-winning Gaussians unlabeled.
A conservative five-neighbor completion study used 80% label consensus. The
validation-selected radii were:

- object: 0.015 scene units;
- part: 0.03 scene units;
- subpart: 0.03 scene units.

Replaying the completed caches with propagation took 53.7 seconds. Final metrics:

| Target | Level | Instance | IoU | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: |
| Foliage | Object | 3 | **0.6650** | 0.7944 | 0.8032 |
| Pot | Part | 6 | **0.3226** | 0.7456 | 0.3625 |
| Pot | Subpart | 1 | 0.3249 | 0.7701 | 0.3598 |

The part-level pot is preferred because it is the simpler hierarchical
representation and performs nearly identically to the subpart.

## Final outputs

Primary result directory:

`output/lbg_plant_full_hierarchy_propagated`

Important artifacts:

- `segmentation.npz`: final object, part, and subpart labels and features;
- `scene.json`: checkpoint hash, run configuration, hierarchy, and timing;
- `evaluation_object.json`: exact foliage/object evaluation;
- `evaluation_part.json`: exact pot/part evaluation;
- `evaluation_subpart.json`: exact subpart evaluation;
- `assets/foliage/object_00003.ply`: 129,522-Gaussian foliage object;
- `assets/pot/part_00006.ply`: 1,073-Gaussian pot part;
- `heldout_foliage/*.png`: foliage renders on held-out cameras;
- `heldout_pot/*.png`: pot renders on held-out cameras.

PLY export preserves every original selected vertex property, including opacity,
scale, rotation, and all spherical-harmonic coefficients. Export and render also
verify the checkpoint SHA-256 to prevent applying labels to a reordered model.

## Verification

The final regression suite contains 20 tests and passes completely. It covers:

- fragment lifting, overlap resolution, and background handling;
- previous-frame-only fusion and running feature averages;
- parent-constrained hierarchy behavior;
- refinement and conservative label propagation;
- cache serialization and deterministic cache ordering;
- exact PLY subset export and checkpoint mismatch rejection;
- evaluation metrics;
- contributor IDs versus independently color-coded alpha contributions;
- opacity cutoff and early termination;
- covariance, antialiasing, and nondefault CUDA streams; and
- preservation of the existing renderer's backward gradients.

Validation command:

```bash
/home/user/venvs/gaussian-splatting/bin/python \
  -m unittest discover -s tests -p 'test_lbg_*.py' -v
```

`python3 -m compileall -q lifting lbg.py` and `git diff --check` also pass.

## Reproduction commands

Full hierarchy preprocessing and fusion:

```bash
PYTHONPATH=.deps/lbg \
/home/user/venvs/gaussian-splatting/bin/python lbg.py run \
  -s data/plant \
  -m output/plant_10k \
  --iteration 10000 \
  --eval \
  --levels 3 \
  --geometry-threshold 0.05 \
  --semantic-threshold 0.4 \
  --min-weight 0.005 \
  --min-gaussians 5 \
  -o output/lbg_plant_full_hierarchy
```

Regenerate the completed result from cache:

```bash
/home/user/venvs/gaussian-splatting/bin/python lbg.py run \
  -s data/plant \
  -m output/plant_10k \
  --iteration 10000 \
  --eval \
  --levels 3 \
  --geometry-threshold 0.05 \
  --semantic-threshold 0.4 \
  --min-weight 0.005 \
  --min-gaussians 5 \
  --propagate-radius 0.015 0.03 0.03 \
  --cache output/lbg_plant_full_hierarchy/cache \
  --cached-only \
  -o output/lbg_plant_full_hierarchy_propagated
```

Evaluate the final pot part:

```bash
/home/user/venvs/gaussian-splatting/bin/python lbg.py evaluate \
  --result output/lbg_plant_full_hierarchy_propagated \
  --reference output/plant_10k/pot.ply \
  --level part \
  -o output/lbg_plant_full_hierarchy_propagated/evaluation_part.json
```

## Limitations

- The object/part/subpart interpretation of SAM's three mask hypotheses is a
  heuristic and may not transfer uniformly to other scenes.
- The pot remains incomplete and has sparse leaf/stem contamination around its
  upper rim. This agrees with its 74.6% precision and 36.2% recall.
- Propagation radii were selected using the plant reference assets. The reported
  propagated metrics are validation results, not an unbiased test benchmark.
- Only the maximum-contributing Gaussian receives direct pixel evidence; other
  visible blended Gaussians depend on spatial completion.
- Mini-Splatting reconstruction/resampling, 2DGS support, dense DINO feature
  lifting, and the paper's LERF/3DOVS evaluation datasets are not implemented.
- This is an independent implementation based on the paper, not the authors'
  released reference code, and it does not claim reproduction of paper metrics.

