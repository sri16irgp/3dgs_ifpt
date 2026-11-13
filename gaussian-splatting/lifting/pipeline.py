import json
import hashlib
import time
from pathlib import Path
import numpy as np
from .core import LEVELS, Fusion, lift_masks, propagate, refine, unit
from .io import digest_file, fingerprint, save_frame, load_level, save_npz, read_result, verify_checkpoint


def load_scene(args):
    import torch
    from scene import Scene
    from scene.gaussian_model import GaussianModel
    from types import SimpleNamespace
    if args.iteration != -1 and args.iteration < 1:
        raise ValueError('LBG requires an existing checkpoint iteration')
    if not torch.cuda.is_available():
        raise RuntimeError('LBG scene rendering requires CUDA')
    dataset = SimpleNamespace(sh_degree=args.sh_degree, source_path=str(Path(args.source).resolve()),
        model_path=str(Path(args.model).resolve()), images=args.images, depths='', resolution=args.resolution,
        white_background=args.white_background, train_test_exp=False, data_device='cpu', eval=args.eval)
    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    pipeline = SimpleNamespace(debug=False, antialiasing=args.antialiasing, compute_cov3D_python=False, convert_SHs_python=False)
    checkpoint = Path(dataset.model_path) / 'point_cloud' / f'iteration_{scene.loaded_iter}' / 'point_cloud.ply'
    return scene, gaussians, pipeline, checkpoint


def run(args):
    import torch
    from gaussian_renderer import render_contributors
    from .models import FoundationModels
    start = time.monotonic()
    scene, gaussians, pipeline, checkpoint = load_scene(args)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    config = {key: getattr(args, key) for key in (
        'sam_checkpoint', 'sam_type', 'points_per_side', 'sam_batch', 'pred_iou', 'stability', 'min_mask_pixels',
        'mask_nms', 'clip_model', 'clip_pretrained', 'dino_repo', 'dino_model', 'dino_resolution', 'feature_batch', 'model_cache')}
    config['device'] = 'cuda'
    config['sam_checkpoint'] = str(Path(config['sam_checkpoint']).resolve())
    model_signature = dict(config, sam_sha256=digest_file(config['sam_checkpoint']))
    checkpoint_hash = digest_file(checkpoint)
    views = scene.getTrainCameras()[::args.view_stride]
    if args.max_views:
        views = views[:args.max_views]
    if not views:
        raise ValueError('No training views selected')
    cache_root = Path(args.cache or destination / 'cache')
    cache_root.mkdir(parents=True, exist_ok=True)
    paths, names, models = [], [], None
    for index, camera in enumerate(views):
        image = (camera.original_image[:3].permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        signature = fingerprint(dict(version=2, model=model_signature, checkpoint=checkpoint_hash,
            image=hashlib.sha256(image.tobytes()).hexdigest(),
            view=camera.world_view_transform.cpu().tolist(), projection=camera.full_proj_transform.cpu().tolist(),
            antialiasing=args.antialiasing, levels=args.levels))
        path = cache_root / f'{signature}.npz'
        if not path.exists():
            if args.cached_only:
                raise FileNotFoundError(f'Missing frame cache for {camera.image_name}: {path}')
            if models is None:
                models = FoundationModels(config)
            with torch.inference_mode():
                contribution = render_contributors(camera, gaussians, pipeline)
                ids = contribution['gaussian_ids'].cpu().numpy()
                weights = contribution['weights'].cpu().numpy()
                groups = models.masks(image)
                groups = groups[:args.levels]
                all_masks = np.concatenate([masks for masks, _ in groups])
                all_clip, all_dino = models.features(image, all_masks)
                levels, offset = {}, 0
                for level, (masks, scores) in zip(LEVELS, groups):
                    end = offset + len(masks)
                    levels[level] = masks, scores, all_clip[offset:end], all_dino[offset:end]
                    offset = end
                save_frame(path, ids, weights, levels, dict(signature=signature, image_name=camera.image_name))
        paths.append(path)
        names.append(camera.image_name)
        print(f'Prepared {index+1}/{len(views)}: {camera.image_name}', flush=True)
    del models
    torch.cuda.empty_cache()
    arrays, hierarchy, parent = {}, {}, None
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    propagation = args.propagate_radius
    if len(propagation) == 1:
        propagation = propagation * args.levels
    for level_index, level in enumerate(LEVELS[:args.levels]):
        fusion = Fusion(len(xyz), args.geometry_threshold, args.semantic_threshold)
        for index, path in enumerate(paths):
            data = load_level(path, level)
            fragments = lift_masks(*data, min_weight=args.min_weight, min_gaussians=args.min_gaussians, parent_labels=parent)
            fusion.add_frame(fragments)
            print(f'Fused {level} {index+1}/{len(paths)}: {len(fusion.instances)} instances', flush=True)
        labels, instances = fusion.labels, fusion.instances
        if args.refine_radius:
            labels, instances = refine(labels, xyz, instances, args.refine_radius, args.min_cluster)
        if propagation[level_index]:
            labels = propagate(labels, xyz, propagation[level_index],
                               args.propagate_neighbors, args.propagate_consensus,
                               parent_labels=parent, instances=instances if parent is not None else None)
        arrays[level + '_labels'] = labels
        arrays[level + '_clip'] = np.stack([unit(o.clip) for o in instances]) if instances else np.empty((0, 0), np.float32)
        arrays[level + '_dino'] = np.stack([unit(o.dino) for o in instances]) if instances else np.empty((0, 0), np.float32)
        hierarchy[level] = [dict(id=o.id, parent=o.parent, fragments=o.count,
            gaussians=int(np.count_nonzero(labels == o.id))) for o in instances]
        parent = labels
    metadata = dict(format_version=1, method='LBG independent implementation', checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=checkpoint_hash, gaussian_count=len(xyz), config=vars(args), foundation_models=config,
        views=names, instances=hierarchy, seconds=time.monotonic()-start)
    save_npz(destination / 'segmentation.npz', **arrays)
    temporary = destination / 'scene.json.tmp'
    temporary.write_text(json.dumps(metadata, indent=2))
    temporary.replace(destination / 'scene.json')
    print(f'Saved segmentation to {destination} ({metadata["seconds"]:.1f}s)', flush=True)


def render_result(args):
    import torch
    from PIL import Image
    from gaussian_renderer import render_contributors, render
    metadata, arrays = read_result(args.result)
    scene, gaussians, pipe, checkpoint = load_scene(args)
    verify_checkpoint(metadata, checkpoint)
    labels = arrays[args.level + '_labels']
    if args.instance is not None:
        import copy
        if not np.any(labels == args.instance):
            raise ValueError(f'No Gaussians for instance {args.instance}')
        gaussians = copy.copy(gaussians)
        gaussians._opacity = gaussians._opacity.detach().clone()
        gaussians._opacity[torch.as_tensor(labels != args.instance, device='cuda')] = -torch.inf
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cameras = scene.getTestCameras() if args.split == 'test' else scene.getTrainCameras()
    if not cameras:
        raise ValueError('Requested camera split is empty; use --eval for held-out COLMAP views')
    cameras = cameras[::args.view_stride]
    if args.max_views:
        cameras = cameras[:args.max_views]
    color = np.random.default_rng(0).integers(40, 256, size=(max(int(labels.max())+1, 1), 3), dtype=np.uint8)
    for i, camera in enumerate(cameras):
        with torch.inference_mode():
            if args.instance is not None:
                background = torch.full((3,), float(args.white_background), device='cuda')
                rgb = render(camera, gaussians, pipe, background)['render']
                rgb = (rgb.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
                Image.fromarray(rgb).save(out / f'{i:05d}.png')
                continue
            output = render_contributors(camera, gaussians, pipe)
        ids, weights = output['gaussian_ids'].cpu().numpy(), output['weights'].cpu().numpy()
        valid = (ids >= 0) & (weights >= args.min_weight)
        image_labels = np.full(ids.shape, -1, np.int32)
        image_labels[valid] = labels[ids[valid]]
        rgb = np.zeros((*ids.shape, 3), np.uint8)
        rgb[image_labels >= 0] = color[image_labels[image_labels >= 0]]
        Image.fromarray(rgb).save(out / f'{i:05d}.png')
        save_npz(out / f'{i:05d}.npz', labels=image_labels, image_name=np.array(camera.image_name))
    kind = 'isolated RGB' if args.instance is not None else 'segmentation'
    print(f'Saved {len(cameras)} {kind} views to {out}')


def query(args):
    import torch
    import open_clip
    metadata, arrays = read_result(args.result)
    c = metadata['foundation_models']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _, _ = open_clip.create_model_and_transforms(c['clip_model'], pretrained=c['clip_pretrained'],
        device=device, cache_dir=c['model_cache'], force_quick_gelu=c['clip_pretrained'] == 'openai')
    model.eval()
    features = arrays[args.level + '_clip']
    if not len(features):
        print('[]')
        return
    with torch.inference_mode():
        encoded = model.encode_text(open_clip.get_tokenizer(c['clip_model'])([args.text]).to(device)).float().cpu().numpy()[0]
    from .query import rank_instances
    print(json.dumps(rank_instances(features, metadata['instances'][args.level], encoded, args.top_k), indent=2))
