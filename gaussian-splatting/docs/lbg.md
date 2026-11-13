# Lifting by Gaussians

This extension implements a training-free segmentation pipeline for **existing
3DGS checkpoints and posed RGB images**, based on
[Lifting by Gaussians, §3](https://arxiv.org/html/2502.00173v1#S3).
It does not retrain or modify the reconstruction. It supports COLMAP and Blender
scenes through the existing scene loader.

## Install

Use a working CUDA/PyTorch environment for this repository, Python 3.10 or newer.
The original `environment.yml` targets an older Python; use a separate modern
environment for this extension. Keep torch/torchvision matched to your GPU.

```bash
python -m pip install -r requirements-lbg.txt
python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization
```

Alternatively, in an editable installation, rebuild in place:

```bash
(cd submodules/diff-gaussian-rasterization && python setup.py build_ext --inplace)
```

Supply the official SAM checkpoint (`--sam-checkpoint`, default ViT-H).
CLIP ViT-L/14 OpenAI weights and DINOv2 ViT-B/14 with registers are downloaded
on first use into `.models/lbg`. DINOv2 uses Torch Hub; `--dino-repo` accepts
a GitHub `owner/repo:ref` or a local source checkout. Pin the ref and retain model
caches when reproducing a run. For offline runs, populate these caches first.
`--clip-pretrained` also accepts a local weights path.

On this workspace the existing GPU interpreter is
`/home/user/venvs/gaussian-splatting/bin/python`. Additional packages installed
for validation are isolated in `.deps/lbg`; prepend
`PYTHONPATH=.deps/lbg` when using that interpreter. They do not
replace its CUDA packages.

## Run

### Browser viewer on a CUDA host

The viewer uses the already computed `scene.json` and `segmentation.npz`. It
loads the matching Gaussian checkpoint and the model's saved `cameras.json`, so
the original source images are not needed for browsing. Install the web server
dependencies from `requirements-lbg.txt` in the working CUDA environment.
If the additional LBG packages live in the isolated directory described above,
run with `PYTHONPATH=.deps/lbg` from the repository root.

On Thor, start one scene per server launch:

```bash
PYTHONPATH=.deps/lbg python lbg.py serve \
  --result output/lbg_garden --port 8080
```

On the laptop, connect using your SSH user or configured host alias:

```bash
ssh -N -L 8080:127.0.0.1:8080 user@thor
```

Open `http://localhost:8080` on the laptop. The viewer binds to Thor's
localhost only. Restart it with a different `--result` to switch scenes. Use
`--checkpoint /new/path/point_cloud.ply` and `--cameras /new/path/cameras.json`
when artifacts have moved. The checkpoint must still match the segmentation's
recorded SHA-256. Use `--model-cache /new/cache` if CLIP model weights moved.

Left drag orbits, right drag pans, and the wheel zooms. Search selects the
highest-ranked object and tints it yellow while dimming the rest; Isolate,
Focus selected, Reset camera, Clear selection, and Download PLY are available.
The similarity displayed is a ranking score rather than a confidence estimate.
Rendering is capped at 15 requested frames per second and one GPU worker; the
browser displays actual frame rate, render time, and GPU memory. The viewer
needs the precomputed segmentation and does not perform segmentation itself.


A small smoke run uses three training views; it is not a full-scene evaluation:

```bash
python lbg.py run -s data/plant -m output/plant_10k --iteration 10000 \
  --eval -r 320 --view-stride 10 --max-views 3 --points-per-side 8 \
  -o output/lbg_plant_smoke
```

For fuller coverage, omit `--max-views` and use the default 640-pixel width and
16×16 point grid. `--view-stride` selects deterministic, loader-ordered views.
Use `--eval` when training used the held-out split; otherwise test images become
fusion inputs. Set `--antialiasing` to match the original reconstruction.
The CLI takes explicit scene options rather than evaluating the model's
`cfg_args`. It defaults to SH degree 3, black background, and no exposure crop.

```bash
python lbg.py run -s data/plant -m output/plant_10k --iteration 10000 \
  --eval -o output/lbg_plant
python lbg.py query --result output/lbg_plant --text 'a plant pot'
python lbg.py export --result output/lbg_plant --level object --instance 0
python lbg.py render -s data/plant -m output/plant_10k --iteration 10000 \
  --eval --result output/lbg_plant --split test -o output/lbg_plant/previews
python lbg.py render -s data/plant -m output/plant_10k --iteration 10000 \
  --eval --result output/lbg_plant --instance 0 --white-background \
  -o output/lbg_plant/asset_preview
python lbg.py evaluate --result output/lbg_plant \
  --reference output/plant_10k/foliage.ply \
  --reference output/plant_10k/pot.ply -o output/lbg_plant/evaluation.json
python lbg.py tune --result output/lbg_plant --cache output/lbg_plant/cache \
  --reference output/plant_10k/foliage.ply \
  --reference output/plant_10k/pot.ply -o output/lbg_plant/tuning.json
```

Use the actual instance ID reported by `query` or `scene.json`; IDs are not
class labels. Omit `--instance` on `export` to export all nonempty instances.
`--level part` and `--level subpart` select the child hierarchies.
Text queries rank CLIP cosine similarities, not calibrated probabilities.

Optional refinement: `--refine-radius R --min-cluster 30`. **R is in the scene's
coordinate units**, so there is deliberately no universal default. Inspect
scene scale before choosing it. `--levels 1` runs only the object level.

Optional completion: `--propagate-radius R` fills nearby unlabeled Gaussians
when at least 80% of their five nearest labeled neighbors agree. Pass one radius
to use it at every enabled level, or one value per object/part/subpart level.
For child levels, labels can propagate only inside the already assigned parent;
existing assignments never change. Tune radii in scene units on validation data.

## Algorithm and explicit implementation choices

1. The rasterizer exposes per-pixel `argmax(alpha * T)` Gaussian indices and
   maximum weights. It replays the original sorted splats and matches alpha
   cutoff, opacity clamp, transmittance early exit, covariance and antialiasing.
   Background is `-1`, weight zero. Ties retain the first depth-sorted splat.
   This is an inference-only second pass using original forward buffers; it
   avoids changing the RGB/depth/autograd interfaces.
2. SAM predicts three masks per grid prompt. This implementation orders the
   three hypotheses by area into object/part/subpart **as a heuristic**; SAM
   does not guarantee those semantic meanings. Filter by predicted IoU 0.88,
   stability 0.95, minimum 100 pixels; perform mask-IoU NMS at 0.8 per level.
   Larger `--points-per-side` improves sampling at additional cost.
3. Encode masked bounding-box crops with CLIP (outside-mask pixels are gray).
   Extract DINOv2 patch features from the full image, area-pool using each mask,
   and L2-normalize both feature families independently.
4. Higher-scoring masks own overlapping pixels. Discard contributors below
   weight 0.01. Assign each Gaussian to the mask winning the most pixels;
   ties use stable mask order. Discard fragments smaller than 10 Gaussians.
5. Match fragments only to the **previous frame's map and features**.
   Geometric overlap is intersection divided by incoming-fragment size.
   Semantic similarity is the mean of CLIP and DINO cosine similarities.
   Require overlap ≥0.25 and semantic similarity ≥0.75; choose the maximum
   `overlap * (1 + similarity) / 2`. Otherwise create a new instance.
   Features use a fragment-count running mean, normalized for comparisons.
   Per-Gaussian sparse vote counts resolve conflicting assignments; equal
   evidence retains the earlier owner. There is no dense Gaussian×object matrix.
6. Complete the whole-object map before parts; intersect lower-level fragments
   with each final parent and fuse only within that parent. Unassigned parent
   Gaussians remain unassigned at lower levels. Repeat for subparts.
7. Optional cleanup removes statistical outliers (16-neighbor mean distance,
   mean + 2 standard deviations), splits the radius-limited 16-neighbor graph
   into components, and retains components with at least `--min-cluster` points.
   Small components attach only if at least half their points are within R of
   the same salient component of the original instance. Otherwise discard them.
   Cleanup runs before the next hierarchy level, preserving parent containment.
8. Optional propagation fills only unassigned Gaussians when a configurable
   fraction of their nearest labeled neighbors agree within a scene-space
   radius. Existing labels stay fixed. Part and subpart propagation is restricted
   to each completed parent, so completion cannot violate the hierarchy.

The paper does not fully specify threshold selection, feature combination,
SAM hypothesis ordering, conflicting labels, or connected-component parameters.
These defaults make the implementation usable and reproducible given fixed
models and inputs; they are **not official author hyperparameters**. This is
an independent implementation, not a claim of reproduced paper metrics.
Mini-Splatting training/resampling, 2DGS support, dense feature lifting (appendix
G), and the manually annotated LERF/3DOVS benchmark datasets are not included.
Hard-ID previews use the dominant contributor; they do not reproduce an
alpha-blended semantic-mask benchmark protocol. Asset previews render the
original appearance after setting other instances' opacity to zero.

## Artifacts and caching

- `segmentation.npz`: int32 `<level>_labels` arrays of length N (`-1` unassigned),
  float32 `<level>_clip` and `<level>_dino` instance features.
- `scene.json`: checkpoint SHA-256 and original ordering, run configuration,
  input views, hierarchy/parent IDs, fragment counts, Gaussian counts and time.
  Empty instances can remain after conflict resolution; export/query skip them.
- `cache/*.npz`: contributor IDs/weights, bit-packed masks, scores and features.
  Keys include the checkpoint hash, actual resized RGB bytes, camera matrices,
  model configuration, SAM checkpoint hash and rasterization mode.
- `assets/<level>/*.ply`: exact subsets of original PLY vertices, including
  opacity, scales, rotations and all SH coefficients. Input PLY stays unchanged.
- Previews: deterministic colored PNGs plus integer label NPZs. Isolated RGB
  previews are PNG only. Labels are never stored as blended RGB IDs.

Reuse `--cache PATH` with another output directory or different fusion thresholds.
`--cached-only` fails on missing caches and does not load foundation models.
Do not replace weights or source code under an unchanged model identifier and
reuse caches; choose a new cache directory or identifier in that case. Export
and render reject a changed checkpoint, even if Gaussian counts still match.

Only maximally contributing Gaussians receive evidence. Invisible or never-winning
Gaussians remain unassigned, and low view coverage can leave holes in extracted
assets. Scene quality and multi-view mask consistency limit segmentation quality.

## Validation

```bash
python -m unittest discover -s tests -p 'test_lbg_*.py' -v
```

CPU tests cover ownership, geometric/semantic matching, frame snapshots,
parent containment, refinement, propagation, evaluation, cache serialization
and PLY exports. CUDA tests
compare contributor IDs/weights with independently color-coded Gaussian
contributions, test background/culling, antialiasing/covariance, nondefault
streams and existing backward gradients. CUDA tests skip without CUDA PyTorch.

### Local validation (2026-11-04)

On the NVIDIA Thor with the existing CUDA/PyTorch environment:

- Local CUDA extension built successfully; **20 CPU/CUDA tests passed**.
- `output/lbg_plant`: 14 training views (stride 10), width 640, 16×16 SAM grid,
  checkpoint `plant_10k/iteration_10000`, 2,004,727 Gaussians. Runtime: **133.2 s**
  including loading and preprocessing, with pretrained weights already cached.
- Result: **23 nonempty objects**, 176 parts, 305 subparts; 196,303 Gaussians
  assigned at object level. This is sampled-view coverage, not a full-scene run.
- `output/lbg_plant_replay`: cached-only execution reproduced every label and
  feature array exactly. All part/subpart labels passed parent-containment checks.
- Held-out hard-ID previews: `output/lbg_plant/previews/`. The maps show holes and
  fragmentation; no quantitative accuracy claim is made without ground truth.
- Text query, PLY export, and isolated RGB rendering exercised successfully.
  Instance 17 contains the **plant together with its pot**, not the pot alone.
  Export: `output/lbg_plant/assets/object/object_00017.ply`; isolated renders:
  `output/lbg_plant/asset_17/`. The highest text-scoring fragment was very small,
  illustrating why query ranking should be inspected rather than treated as a
  guaranteed object selection.

These outputs are local, ignored artifacts. No LERF/3DOVS metrics or full paper
experiment reproduction have been performed.

#### Full-view plant quality pass

`output/lbg_plant_full_hierarchy` processes all 139 training views while keeping
20 LLFF views held out. It uses geometry threshold 0.05, semantic threshold 0.4,
minimum contributor weight 0.005, and minimum fragment size 5. Preprocessing and
three-level fusion took **877.5 seconds** on the NVIDIA Thor and produced 6
objects, 53 parts, and 175 subparts.

The `foliage.ply` and `pot.ply` references are exact, unique subsets of the same
2,004,727-Gaussian checkpoint, so evaluation maps them back to checkpoint indices
without geometric tolerance. Before completion, best-instance 3D results were:

| Target / level | Instance | IoU | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| Foliage / object | 3 | 0.2333 | 0.8099 | 0.2468 |
| Pot / part | 6 | 0.2688 | 0.7862 | 0.2900 |

A validation-tuned, parent-constrained completion pass uses radii 0.015, 0.03,
and 0.03 for object, part, and subpart. Regenerating from cache took **53.7
seconds** and produced `output/lbg_plant_full_hierarchy_propagated`:

| Target / level | Instance | IoU | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| Foliage / object | 3 | **0.6650** | 0.7944 | 0.8032 |
| Pot / part | 6 | **0.3226** | 0.7456 | 0.3625 |
| Pot / subpart | 1 | 0.3249 | 0.7701 | 0.3598 |

The pot is a part of the plant-plus-pot scene object at the hierarchy's top
level. Held-out isolated renders are under `heldout_foliage/` and `heldout_pot/`.
The foliage asset is coherent; the pot is substantially more complete after
propagation but retains sparse leaf/stem contamination around its rim. Final PLY
exports are under `assets/foliage/object_00003.ply` and
`assets/pot/part_00006.ply`. These radii were selected using the plant references,
so the numbers are validation results, not an unbiased benchmark or paper-metric
reproduction.
