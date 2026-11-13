#!/usr/bin/env python3
"""Training-free 3D instance segmentation. See docs/lbg.md."""
import argparse


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def probability(value):
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError('must be between 0 and 1')
    return number


def scene_options(parser):
    parser.add_argument('-s', '--source', required=True, help='COLMAP or Blender dataset directory')
    parser.add_argument('-m', '--model', required=True, help='Existing trained 3DGS model directory')
    parser.add_argument('--iteration', type=int, default=-1)
    parser.add_argument('--sh-degree', type=int, default=3)
    parser.add_argument('--images', default='images')
    parser.add_argument('-r', '--resolution', type=positive, default=640, help='Image width, or downsample factor 1/2/4/8')
    parser.add_argument('--eval', action='store_true', help='Use the existing loader train/test split')
    parser.add_argument('--white-background', action='store_true')
    parser.add_argument('--antialiasing', action='store_true', help='Match this to reconstruction settings')
    parser.add_argument('--view-stride', type=positive, default=1)
    parser.add_argument('--max-views', type=positive, default=None)
    parser.add_argument('--min-weight', type=probability, default=0.01)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run', help='Generate masks/features, lift, and fuse a trained scene')
    scene_options(run)
    run.add_argument('-o', '--output', required=True)
    run.add_argument('--cache', help='Reusable frame-cache directory')
    run.add_argument('--cached-only', action='store_true', help='Fail rather than recomputing missing caches')
    run.add_argument('--levels', type=int, choices=(1, 2, 3), default=3)
    run.add_argument('--sam-checkpoint', default='checkpoints/sam_vit_h_4b8939.pth')
    run.add_argument('--sam-type', choices=('vit_h', 'vit_l', 'vit_b'), default='vit_h')
    run.add_argument('--points-per-side', type=positive, default=16)
    run.add_argument('--sam-batch', type=positive, default=16)
    run.add_argument('--pred-iou', type=probability, default=0.88)
    run.add_argument('--stability', type=probability, default=0.95)
    run.add_argument('--min-mask-pixels', type=positive, default=100)
    run.add_argument('--mask-nms', type=probability, default=0.8)
    run.add_argument('--clip-model', default='ViT-L-14')
    run.add_argument('--clip-pretrained', default='openai')
    run.add_argument('--dino-repo', default='facebookresearch/dinov2:main', help='Torch Hub repo[:ref] or local source directory')
    run.add_argument('--dino-model', default='dinov2_vitb14_reg')
    run.add_argument('--dino-resolution', type=positive, default=518)
    run.add_argument('--feature-batch', type=positive, default=16)
    run.add_argument('--model-cache', default='.models/lbg')
    run.add_argument('--geometry-threshold', type=probability, default=0.25)
    run.add_argument('--semantic-threshold', type=probability, default=0.75, help='Mean CLIP/DINO cosine similarity')
    run.add_argument('--min-gaussians', type=positive, default=10)
    run.add_argument('--refine-radius', type=float, default=0, help='Optional cleanup radius in scene units; 0 disables')
    run.add_argument('--min-cluster', type=positive, default=30)
    run.add_argument('--propagate-radius', nargs='+', type=float, default=[0],
                     help='Unassigned-label fill radii in scene units: one value or one per hierarchy level')
    run.add_argument('--propagate-neighbors', type=positive, default=5)
    run.add_argument('--propagate-consensus', type=probability, default=0.8)

    export = commands.add_parser('export', help='Export original Gaussian PLY properties for selected instances')
    export.add_argument('--result', required=True)
    export.add_argument('--level', choices=('object', 'part', 'subpart'), default='object')
    export.add_argument('--instance', type=int)
    export.add_argument('-o', '--output')

    render = commands.add_parser('render', help='Render hard instance IDs and color previews')
    scene_options(render)
    render.add_argument('--result', required=True)
    render.add_argument('--instance', type=int, help='Render an isolated RGB asset instead of segmentation IDs')
    render.add_argument('--level', choices=('object', 'part', 'subpart'), default='object')
    render.add_argument('--split', choices=('train', 'test'), default='test')
    render.add_argument('-o', '--output', required=True)

    query = commands.add_parser('query', help='Rank instances by CLIP text similarity')
    query.add_argument('--result', required=True)
    query.add_argument('--text', required=True)
    query.add_argument('--level', choices=('object', 'part', 'subpart'), default='object')
    query.add_argument('--top-k', type=positive, default=5)

    serve = commands.add_parser('serve', help='Serve a CUDA-backed browser viewer for one LBG result')
    serve.add_argument('--result', required=True)
    serve.add_argument('--checkpoint', help='Override checkpoint path after moving the result')
    serve.add_argument('--cameras', help='Override cameras.json path after moving the result')
    serve.add_argument('--host', default='127.0.0.1')
    serve.add_argument('--port', type=int, default=8080)
    serve.add_argument('--model-cache', help='Override the saved CLIP model cache path')

    evaluate = commands.add_parser('evaluate', help='Score 3D instances against exact checkpoint-subset PLY references')
    evaluate.add_argument('--result', required=True)
    evaluate.add_argument('--reference', action='append', required=True, help='Reference PLY; repeat for multiple targets')
    evaluate.add_argument('--level', choices=('object', 'part', 'subpart'), default='object')
    evaluate.add_argument('-o', '--output', help='Optional JSON report path')

    tune = commands.add_parser('tune', help='Sweep fusion thresholds over an existing frame cache')
    tune.add_argument('--result', required=True, help='Result whose ordered views define the sweep input')
    tune.add_argument('--cache', required=True)
    tune.add_argument('--reference', action='append', required=True)
    tune.add_argument('--geometry', nargs='+', type=probability, default=[0.1, 0.25, 0.4])
    tune.add_argument('--semantic', nargs='+', type=probability, default=[0.5, 0.65, 0.75, 0.85])
    tune.add_argument('--min-weight', nargs='+', type=probability, default=[0.005, 0.01, 0.02])
    tune.add_argument('--min-gaussians', nargs='+', type=positive, default=[5, 10, 25])
    tune.add_argument('-o', '--output', help='Optional JSON report path')
    args = parser.parse_args()
    if hasattr(args, 'iteration') and args.iteration != -1 and args.iteration < 1:
        parser.error('iteration must be -1 (latest) or a positive checkpoint iteration')
    if args.command == 'run':
        if args.refine_radius < 0 or (args.refine_radius and args.min_cluster < 2):
            parser.error('refine-radius must be nonnegative and refinement needs min-cluster >= 2')
        if any(radius < 0 for radius in args.propagate_radius):
            parser.error('propagate-radius values must be nonnegative')
        if len(args.propagate_radius) not in (1, args.levels):
            parser.error('propagate-radius needs one value or one value per enabled hierarchy level')
        if args.propagate_consensus == 0:
            parser.error('propagate-consensus must be greater than zero')
        from lifting.pipeline import run
        run(args)
    elif args.command == 'render':
        from lifting.pipeline import render_result
        render_result(args)
    elif args.command == 'export':
        from lifting.io import export_assets
        for path in export_assets(args.result, args.level, args.instance, args.output):
            print(path)
    elif args.command == 'query':
        from lifting.pipeline import query
        query(args)
    elif args.command == 'serve':
        from lifting.viewer import serve
        serve(args)
    elif args.command == 'evaluate':
        import json
        from lifting.evaluation import evaluate_result
        print(json.dumps(evaluate_result(args.result, args.reference, args.level, args.output), indent=2))
    else:
        import json
        from lifting.tuning import tune_cached
        report = tune_cached(args.result, args.cache, args.reference, args.geometry,
                             args.semantic, args.min_weight, args.min_gaussians, args.output)
        print(json.dumps(report['best'], indent=2))


if __name__ == '__main__':
    main()
