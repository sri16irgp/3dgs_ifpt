"""Optional SAM / CLIP / DINOv2 front end. Models load only when needed."""
from pathlib import Path
import numpy as np
from PIL import Image
from .core import unit


def mask_nms(masks, scores, threshold=0.8):
    """Deterministic mask IoU NMS with bounding-box rejection."""
    kept = []
    areas = masks.reshape(len(masks), -1).sum(1) if len(masks) else []
    boxes = []
    for mask in masks:
        y, x = np.nonzero(mask)
        boxes.append((x.min(), y.min(), x.max(), y.max()))
    for i in np.argsort(-np.asarray(scores), kind='stable'):
        a = boxes[i]
        duplicate = False
        for j in kept:
            b = boxes[j]
            if a[0] > b[2] or b[0] > a[2] or a[1] > b[3] or b[1] > a[3]:
                continue
            intersection = np.count_nonzero(masks[i] & masks[j])
            if intersection / (areas[i] + areas[j] - intersection) > threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(int(i))
    return kept


class FoundationModels:
    def __init__(self, config):
        import torch
        import open_clip
        from segment_anything import sam_model_registry, SamPredictor
        self.torch = torch
        self.config = config
        self.device = config['device']
        checkpoint = Path(config['sam_checkpoint'])
        if not checkpoint.is_file():
            raise FileNotFoundError(f'SAM checkpoint not found: {checkpoint}')
        self.sam = SamPredictor(sam_model_registry[config['sam_type']](checkpoint=str(checkpoint)).to(self.device).eval())
        self.clip, _, self.clip_transform = open_clip.create_model_and_transforms(
            config['clip_model'], pretrained=config['clip_pretrained'], device=self.device,
            cache_dir=config['model_cache'], force_quick_gelu=config['clip_pretrained'] == 'openai')
        self.clip.eval()
        torch.hub.set_dir(str(Path(config['model_cache']) / 'torch_hub'))
        repo = config['dino_repo']
        self.dino = torch.hub.load(repo, config['dino_model'], source='local' if Path(repo).is_dir() else 'github',
                                   trust_repo=True).to(self.device).eval()

    def masks(self, image):
        """Area-order each point's SAM hypotheses into whole/part/subpart.

        This is an explicit heuristic, not a claim SAM predicts semantic levels.
        """
        torch, c = self.torch, self.config
        h, w = image.shape[:2]
        grid = (np.arange(c['points_per_side']) + 0.5) / c['points_per_side']
        xx, yy = np.meshgrid(grid * w, grid * h)
        points = np.stack((xx.ravel(), yy.ravel()), axis=-1)
        groups = [[], [], []]
        scores = [[], [], []]
        with torch.inference_mode():
            self.sam.set_image(image)
            for start in range(0, len(points), c['sam_batch']):
                coords = torch.as_tensor(points[start:start+c['sam_batch']], dtype=torch.float32, device=self.device)
                coords = self.sam.transform.apply_coords_torch(coords, (h, w))
                logits, quality, _ = self.sam.predict_torch(coords[:, None], torch.ones((len(coords), 1), device=self.device),
                    multimask_output=True, return_logits=True)
                threshold = self.sam.model.mask_threshold
                binary = logits > threshold
                stability = ((logits > threshold + 1).sum((-1, -2)) /
                             (logits > threshold - 1).sum((-1, -2)).clamp_min(1))
                areas = binary.sum((-1, -2))
                order = torch.argsort(areas, dim=1, descending=True, stable=True)
                for batch in range(len(coords)):
                    for level, j in enumerate(order[batch].tolist()):
                        if quality[batch, j] < c['pred_iou'] or stability[batch, j] < c['stability'] or areas[batch, j] < c['min_mask_pixels']:
                            continue
                        groups[level].append(binary[batch, j].cpu().numpy())
                        scores[level].append(float(quality[batch, j]))
            self.sam.reset_image()
        result = []
        for masks, quality in zip(groups, scores):
            masks = np.asarray(masks, dtype=bool).reshape(-1, h, w)
            quality = np.asarray(quality, dtype=np.float32)
            keep = mask_nms(masks, quality, c['mask_nms'])
            result.append((masks[keep], quality[keep]))
        return result

    def features(self, image, masks):
        torch, c = self.torch, self.config
        if not len(masks):
            return np.empty((0, self.clip.visual.output_dim), np.float32), np.empty((0, self.dino.embed_dim), np.float32)
        import torch.nn.functional as F
        # DINO patch features use the full frame; mask pooling keeps spatial context.
        h, w = image.shape[:2]
        scale = c['dino_resolution'] / max(h, w)
        dh, dw = max(14, round(h * scale / 14) * 14), max(14, round(w * scale / 14) * 14)
        with torch.inference_mode():
            tensor = torch.from_numpy(image.copy()).permute(2, 0, 1)[None].float().to(self.device) / 255
            tensor = F.interpolate(tensor, size=(dh, dw), mode='bilinear', align_corners=False)
            mean = tensor.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
            std = tensor.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
            patches = self.dino.forward_features((tensor - mean) / std)['x_norm_patchtokens'][0].float()
            clip_features, dino_features = [], []
            for start in range(0, len(masks), c['feature_batch']):
                batch = masks[start:start+c['feature_batch']]
                crops = []
                for mask in batch:
                    ys, xs = np.nonzero(mask)
                    crop = image[ys.min():ys.max()+1, xs.min():xs.max()+1].copy()
                    crop[~mask[ys.min():ys.max()+1, xs.min():xs.max()+1]] = 127
                    crops.append(self.clip_transform(Image.fromarray(crop)))
                cf = self.clip.encode_image(torch.stack(crops).to(self.device)).float()
                pooled = F.interpolate(torch.as_tensor(batch, dtype=torch.float32, device=self.device)[:, None],
                                       size=(dh // 14, dw // 14), mode='area').flatten(1)
                df = pooled @ patches / pooled.sum(1, keepdim=True).clamp_min(1e-8)
                clip_features.append(cf.cpu().numpy())
                dino_features.append(df.cpu().numpy())
        return unit(np.concatenate(clip_features)), unit(np.concatenate(dino_features))
