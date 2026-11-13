"""One-scene CUDA viewer served to a browser over an SSH tunnel."""
import asyncio
import io
import json
import math
from pathlib import Path
import tempfile
import threading
import time

import numpy as np

from .io import read_result, verify_checkpoint
from .query import rank_instances


def camera_parameters(record):
    """Return a saved camera's C2W pose and horizontal/vertical field of view."""
    width, height = int(record['width']), int(record['height'])
    rotation = np.asarray(record['rotation'], dtype=np.float64)
    position = np.asarray(record['position'], dtype=np.float64)
    if width < 1 or height < 1 or rotation.shape != (3, 3) or position.shape != (3,):
        raise ValueError('Invalid saved camera geometry')
    if not np.isfinite(rotation).all() or not np.isfinite(position).all():
        raise ValueError('Nonfinite saved camera geometry')
    return dict(position=position.tolist(), rotation=rotation.tolist(), width=width, height=height,
                fov_x=2 * math.atan(width / (2 * float(record['fx']))),
                fov_y=2 * math.atan(height / (2 * float(record['fy']))))


def frame_size(width, height, max_dimension):
    width, height = int(width), int(height)
    if not 1 <= width <= 8192 or not 1 <= height <= 8192:
        raise ValueError('Invalid viewport dimensions')
    scale = min(1.0, max_dimension / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def validate_pose(position, rotation):
    p, r = np.asarray(position, dtype=np.float64), np.asarray(rotation, dtype=np.float64)
    if p.shape != (3,) or r.shape != (3, 3) or not np.isfinite(p).all() or not np.isfinite(r).all():
        raise ValueError('Invalid camera pose')
    if not np.allclose(r.T @ r, np.eye(3), atol=0.02) or np.linalg.det(r) < 0.98:
        raise ValueError('Camera rotation must be orthonormal')
    return p, r


def make_camera(position, rotation, width, height, fov_y):
    import torch
    from scene.cameras import MiniCam
    from utils.graphics_utils import getProjectionMatrix
    p, r = validate_pose(position, rotation)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3], c2w[:3, 3] = r, p
    w2c = torch.as_tensor(np.linalg.inv(c2w).T.copy(), device='cuda')
    fov_x = 2 * math.atan((width / height) * math.tan(fov_y / 2))
    projection = getProjectionMatrix(0.01, 100.0, fov_x, fov_y).T.cuda()
    return MiniCam(width, height, fov_y, fov_x, 0.01, 100.0, w2c, w2c @ projection)


class ViewerScene:
    def __init__(self, result, checkpoint=None, cameras=None, model_cache=None):
        import torch
        import open_clip
        from scene.gaussian_model import GaussianModel
        from types import SimpleNamespace

        if not torch.cuda.is_available():
            raise RuntimeError('The web viewer requires CUDA on Thor')
        self.result = Path(result).resolve()
        self.metadata, arrays = read_result(self.result)
        self.checkpoint = verify_checkpoint(self.metadata, checkpoint)
        self.labels = np.asarray(arrays['object_labels'])
        self.features = np.asarray(arrays['object_clip'])
        self.records = self.metadata['instances']['object']
        if len(self.labels) != self.metadata['gaussian_count'] or len(self.records) != len(self.features):
            raise ValueError('Object labels, features, and metadata disagree')
        if not self.records or not any(record['gaussians'] > 0 for record in self.records):
            raise ValueError('No nonempty object instances in this result')
        if self.features.ndim != 2 or self.features.shape[1] < 1:
            raise ValueError('Missing CLIP object features')
        if any(record['id'] != i for i, record in enumerate(self.records)):
            raise ValueError('Object metadata IDs are not aligned with features')
        if np.any((self.labels >= len(self.records)) | (self.labels < -1)):
            raise ValueError('Object labels are outside the instance range')
        root = Path(cameras) if cameras else self.checkpoint.parents[2] / 'cameras.json'
        if not root.is_file():
            raise FileNotFoundError(f'Saved camera poses not found: {root}; use --cameras')
        saved = json.loads(root.read_text())
        if not saved:
            raise ValueError('Saved camera pose list is empty')
        self.initial = camera_parameters(saved[0])
        degree = int(self.metadata.get('config', {}).get('sh_degree', 3))
        self.gaussians = GaussianModel(degree)
        self.gaussians.load_ply(str(self.checkpoint))
        if len(self.gaussians.get_xyz) != len(self.labels):
            raise ValueError('Checkpoint Gaussian count differs from labels')
        self.center = np.median(self.gaussians.get_xyz.detach().cpu().numpy(), axis=0).tolist()
        self.radius = float(np.linalg.norm(np.asarray(self.initial['position']) - self.center))
        self.radius = max(self.radius, 0.1)
        self.mask_colors = torch.zeros((len(self.labels), 3), device='cuda')
        self.label_tensors = torch.as_tensor(self.labels, device='cuda')
        self.focus_cache = {}
        self.pipe = SimpleNamespace(debug=False, antialiasing=bool(self.metadata['config'].get('antialiasing', False)),
                                    compute_cov3D_python=False, convert_SHs_python=False)
        config = self.metadata['foundation_models']
        cache = model_cache or config['model_cache']
        self.clip, _, _ = open_clip.create_model_and_transforms(
            config['clip_model'], pretrained=config['clip_pretrained'], device='cuda',
            cache_dir=cache, force_quick_gelu=config['clip_pretrained'] == 'openai')
        self.clip.eval()
        self.tokenizer = open_clip.get_tokenizer(config['clip_model'])
        self.lock = threading.Lock()

    def info(self):
        return dict(scene=self.result.name, initial=self.initial, center=self.center, radius=self.radius,
                    objects=sum(record['gaussians'] > 0 for record in self.records))

    def query(self, prompt):
        import torch
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 512:
            raise ValueError('Enter a text prompt of at most 512 characters')
        with self.lock, torch.inference_mode():
            encoded = self.clip.encode_text(self.tokenizer([prompt.strip()]).cuda()).float().cpu().numpy()[0]
            ranked = rank_instances(self.features, self.records, encoded, 1)
        return ranked[0] if ranked else None

    def render_frame(self, request):
        import copy
        import torch
        from PIL import Image
        from gaussian_renderer import render

        selected = request.get('instance')
        if selected is not None and (not isinstance(selected, int) or selected < 0 or
                                     selected >= len(self.records) or self.records[selected]['gaussians'] < 1):
            raise ValueError('Unknown object instance')
        w, h = frame_size(request['width'], request['height'], 800 if request.get('moving') else 1280)
        fov_y = float(request.get('fov_y', self.initial['fov_y']))
        if not math.isfinite(fov_y) or not 0.2 < fov_y < 2.8:
            raise ValueError('Invalid field of view')
        camera = make_camera(request['position'], request['rotation'], w, h, fov_y)
        background = torch.zeros(3, device='cuda')
        started = time.monotonic()
        with self.lock, torch.inference_mode():
            if selected is not None and request.get('isolate'):
                # The clone changes only the opacity tensor, never the loaded checkpoint.
                model = copy.copy(self.gaussians)
                model._opacity = self.gaussians._opacity.detach().clone()
                model._opacity[self.label_tensors != selected] = -torch.inf
                pixels = render(camera, model, self.pipe, background)['render']
                rgb = pixels.permute(1, 2, 0).cpu().numpy()
            else:
                pixels = render(camera, self.gaussians, self.pipe, background)['render']
                rgb = pixels.permute(1, 2, 0).cpu().numpy()
                if selected is not None:
                    self.mask_colors.zero_()
                    self.mask_colors[self.label_tensors == selected] = 1
                    # Rendering a white selected instance through the complete scene
                    # gives its visible alpha contribution, including occlusion.
                    mask = render(camera, self.gaussians, self.pipe, background,
                                  override_color=self.mask_colors)['render'][0].cpu().numpy()
                    rgb = rgb * (0.38 + 0.62 * mask[..., None])
                    rgb = rgb * (1 - 0.65 * mask[..., None]) + np.array([1.0, 0.82, 0.08]) * (0.65 * mask[..., None])
            image = Image.fromarray(np.uint8(np.clip(rgb * 255, 0, 255)))
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=85)
        return buffer.getvalue(), dict(render_ms=round((time.monotonic() - started) * 1000, 1), width=w, height=h,
                                       gpu_mb=round(torch.cuda.memory_allocated() / (1024 ** 2), 1))

    def export(self, instance):
        from plyfile import PlyData, PlyElement
        if instance < 0 or instance >= len(self.records) or not self.records[instance]['gaussians']:
            raise ValueError('Unknown object instance')
        with self.lock:
            vertices = PlyData.read(str(self.checkpoint))['vertex'].data[self.labels == instance].copy()
            with tempfile.NamedTemporaryFile(prefix='lbg_object_', suffix='.ply', delete=False) as stream:
                path = Path(stream.name)
            try:
                PlyData([PlyElement.describe(vertices, 'vertex')], text=False).write(str(path))
            except Exception:
                path.unlink(missing_ok=True)
                raise
        return path

    def focus(self, instance):
        if instance < 0 or instance >= len(self.records) or not self.records[instance]['gaussians']:
            raise ValueError('Unknown object instance')
        with self.lock:
            if instance not in self.focus_cache:
                points = self.gaussians.get_xyz[self.label_tensors == instance].detach().cpu().numpy()
                center = np.median(points, axis=0)
                radius = float(np.percentile(np.linalg.norm(points - center, axis=1), 90))
                self.focus_cache[instance] = dict(center=center.tolist(), radius=max(radius, 0.05))
            return self.focus_cache[instance]


def create_app(scene):
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse
    from pydantic import BaseModel
    from starlette.background import BackgroundTask

    app = FastAPI(title='LBG Viewer')
    static = Path(__file__).parent / 'viewer_static'
    no_cache = {'Cache-Control': 'no-store'}

    class Prompt(BaseModel):
        text: str

    @app.get('/')
    def index():
        return FileResponse(static / 'index.html', headers=no_cache)

    @app.get('/app.js')
    def javascript():
        return FileResponse(static / 'app.js', media_type='text/javascript', headers=no_cache)

    @app.get('/style.css')
    def stylesheet():
        return FileResponse(static / 'style.css', media_type='text/css', headers=no_cache)

    @app.get('/api/scene')
    def info():
        return scene.info()

    @app.post('/api/query')
    def query(prompt: Prompt):
        try:
            return {'result': scene.query(prompt.text)}
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

    @app.get('/api/export/{instance}')
    def export(instance: int):
        try:
            path = scene.export(instance)
        except ValueError as error:
            raise HTTPException(404, str(error)) from error
        return FileResponse(path, filename=f'object_{instance:05d}.ply', media_type='application/octet-stream',
                            background=BackgroundTask(path.unlink, missing_ok=True))

    @app.get('/api/focus/{instance}')
    def focus(instance: int):
        try:
            return scene.focus(instance)
        except ValueError as error:
            raise HTTPException(404, str(error)) from error

    @app.websocket('/ws')
    async def frames(socket: WebSocket):
        await socket.accept()
        last = 0.0
        try:
            while True:
                request = await socket.receive_json()
                identifier = request.get('id')
                try:
                    elapsed = time.monotonic() - last
                    if elapsed < 1 / 15:
                        await asyncio.sleep(1 / 15 - elapsed)
                    payload, stats = await asyncio.to_thread(scene.render_frame, request)
                    last = time.monotonic()
                    await socket.send_json(dict(type='frame', id=identifier, **stats))
                    await socket.send_bytes(payload)
                except (ValueError, KeyError, TypeError, AttributeError, RuntimeError) as error:
                    await socket.send_json(dict(type='error', id=identifier, message=str(error)))
        except WebSocketDisconnect:
            pass

    return app


def serve(args):
    import uvicorn
    if args.host not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('Viewer binds only to localhost; use an SSH tunnel for remote access')
    scene = ViewerScene(args.result, args.checkpoint, args.cameras, args.model_cache)
    print(f'Open http://localhost:{args.port} through an SSH tunnel', flush=True)
    uvicorn.run(create_app(scene), host=args.host, port=args.port, workers=1)
