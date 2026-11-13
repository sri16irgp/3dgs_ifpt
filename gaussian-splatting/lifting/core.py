"""Deterministic, sparse mask lifting and incremental fragment fusion.

No foundation-model or CUDA imports: the algorithm can be tested on CPU.
Thresholds and conflict handling are implementation choices (see docs/lbg.md).
"""
from dataclasses import dataclass, field
import numpy as np

LEVELS = ('object', 'part', 'subpart')


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


@dataclass
class Fragment:
    ids: np.ndarray
    clip: np.ndarray
    dino: np.ndarray
    parent: int = -1


@dataclass
class Instance:
    id: int
    parent: int
    clip: np.ndarray
    dino: np.ndarray
    count: int = 1
    votes: dict = field(default_factory=dict, repr=False)


def lift_masks(ids, weights, masks, scores, clip, dino, min_weight=0.01,
               min_gaussians=10, parent_labels=None):
    """Resolve mask overlaps by score, then Gaussian conflicts by pixel majority.

    Each Gaussian contributes to at most one fragment per frame and level.
    Child fragments are intersected with each final parent instance.
    """
    ids, weights, masks = np.asarray(ids), np.asarray(weights), np.asarray(masks, bool)
    if ids.ndim != 2 or weights.shape != ids.shape or masks.shape[1:] != ids.shape:
        raise ValueError('Contributor maps and masks must have matching HxW dimensions')
    if len(masks) != len(scores) or len(masks) != len(clip) or len(masks) != len(dino):
        raise ValueError('Mask, score and feature counts must match')
    if not len(masks):
        return []
    owner = np.full(ids.shape, -1, np.int32)
    # Stable ordering makes equal-score conflicts reproducible.
    for i in np.argsort(-np.asarray(scores), kind='stable'):
        owner[masks[i] & (owner < 0)] = i
    valid = (owner >= 0) & (ids >= 0) & (weights >= min_weight)
    n = len(masks)
    pairs, counts = np.unique(ids[valid].astype(np.int64) * n + owner[valid], return_counts=True)
    order = np.argsort(-counts, kind='stable')
    pairs = pairs[order]
    _, first = np.unique(pairs // n, return_index=True)
    winners = pairs[first]
    fragments = []
    for i in range(n):
        members = winners[winners % n == i] // n
        if parent_labels is None:
            partitions = [(-1, members)]
        else:
            parents = parent_labels[members]
            partitions = [(int(p), members[parents == p]) for p in np.unique(parents) if p >= 0]
        for parent, members in partitions:
            if len(members) >= min_gaussians:
                fragments.append(Fragment(members, unit(clip[i]), unit(dino[i]), parent))
    return fragments


class Fusion:
    def __init__(self, gaussian_count, geometric_threshold=0.25, semantic_threshold=0.75):
        if not 0 <= geometric_threshold <= 1 or not -1 <= semantic_threshold <= 1:
            raise ValueError('Invalid fusion thresholds')
        self.labels = np.full(gaussian_count, -1, np.int32)
        self.support = np.zeros(gaussian_count, np.int32)
        self.instances = []
        self.geometric_threshold = geometric_threshold
        self.semantic_threshold = semantic_threshold

    def add_frame(self, fragments):
        # Match only against the previous frame's map, including its features.
        previous = self.labels.copy()
        old_features = [(unit(o.clip), unit(o.dino)) for o in self.instances]
        decisions = []
        for fragment in fragments:
            ids = np.unique(fragment.ids)
            if not len(ids):
                continue
            if ids.min() < 0 or ids.max() >= len(previous):
                raise ValueError('Gaussian index outside checkpoint')
            candidates, counts = np.unique(previous[ids], return_counts=True)
            best, best_score = None, -np.inf
            for candidate, count in zip(candidates, counts):
                if candidate < 0:
                    continue
                obj = self.instances[candidate]
                if obj.parent != fragment.parent:
                    continue
                geometry = count / len(ids)
                cf, df = old_features[candidate]
                semantic = float((np.dot(cf, unit(fragment.clip)) + np.dot(df, unit(fragment.dino))) / 2)
                score = geometry * (semantic + 1) / 2
                if geometry >= self.geometric_threshold and semantic >= self.semantic_threshold and score > best_score:
                    best, best_score = int(candidate), score
            decisions.append((fragment, ids, best))
        for fragment, ids, match in decisions:
            if match is None:
                match = len(self.instances)
                obj = Instance(match, fragment.parent, unit(fragment.clip), unit(fragment.dino))
                self.instances.append(obj)
            else:
                obj = self.instances[match]
                obj.clip = (obj.clip * obj.count + unit(fragment.clip)) / (obj.count + 1)
                obj.dino = (obj.dino * obj.count + unit(fragment.dino)) / (obj.count + 1)
                obj.count += 1
            for gid in ids:
                gid = int(gid)
                votes = obj.votes.get(gid, 0) + 1
                obj.votes[gid] = votes
                # Equal evidence preserves the earlier instance.
                if votes > self.support[gid]:
                    self.labels[gid], self.support[gid] = match, votes

    def records(self):
        return [dict(id=o.id, parent=o.parent, fragments=o.count,
                     gaussians=int(np.count_nonzero(self.labels == o.id))) for o in self.instances]


def refine(labels, xyz, instances, radius, min_cluster=30, neighbors=16,
           std_ratio=2.0, merge_overlap=0.5):
    """Optional statistical filtering and bounded kNN split/merge.

    Radius is in scene coordinates. Small clusters only attach within their
    original instance; unrelated instances and hierarchy parents never merge.
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    if radius <= 0 or min_cluster < 2 or neighbors < 1:
        raise ValueError('Refinement needs positive radius, cluster >= 2 and neighbors >= 1')
    result = np.full_like(labels, -1)
    output = []
    for old in instances:
        ids = np.flatnonzero(labels == old.id)
        if len(ids) < min_cluster:
            continue
        points = xyz[ids]
        dist, _ = cKDTree(points).query(points, k=min(neighbors + 1, len(ids)))
        avg = dist[:, 1:].mean(axis=1)
        keep = avg <= avg.mean() + std_ratio * avg.std()
        ids, points = ids[keep], points[keep]
        if len(ids) < min_cluster:
            continue
        dist, near = cKDTree(points).query(points, k=min(neighbors + 1, len(ids)))
        rows = np.broadcast_to(np.arange(len(ids))[:, None], near.shape)
        valid = (dist <= radius) & (rows != near)
        graph = coo_matrix((np.ones(valid.sum()), (rows[valid], near[valid])), shape=(len(ids), len(ids))).tocsr()
        _, components = connected_components(graph, directed=False)
        counts = np.bincount(components)
        salient = np.flatnonzero(counts >= min_cluster)
        if not len(salient):
            continue
        selected = np.isin(components, salient)
        tree = cKDTree(points[selected])
        for small in np.flatnonzero(counts < min_cluster):
            members = np.flatnonzero(components == small)
            d, nearest = tree.query(points[members])
            close = d <= radius
            if close.mean() < merge_overlap:
                continue
            targets, votes = np.unique(components[selected][nearest[close]], return_counts=True)
            target = targets[np.argmax(votes)]
            if votes.max() / len(members) >= merge_overlap:
                components[members] = target
        for component in salient:
            new_id = len(output)
            result[ids[components == component]] = new_id
            output.append(Instance(new_id, old.parent, old.clip.copy(), old.dino.copy(), old.count))
    return result, output


def propagate(labels, xyz, radius, neighbors=5, consensus=0.8,
              parent_labels=None, instances=None):
    """Fill nearby unlabeled Gaussians using local instance-label consensus.

    Child labels only propagate inside their already assigned parent. Existing
    assignments never change, making this a conservative completion pass.
    """
    from scipy.spatial import cKDTree

    if radius <= 0 or neighbors < 1 or not 0 < consensus <= 1:
        raise ValueError("Propagation needs positive radius/neighbors and consensus in (0, 1]")
    labels = np.asarray(labels)
    xyz = np.asarray(xyz)
    if labels.ndim != 1 or xyz.shape != (len(labels), 3):
        raise ValueError("Labels and xyz dimensions do not match")
    result = labels.copy()
    if parent_labels is None:
        groups = [(None, np.flatnonzero(labels >= 0), np.flatnonzero(labels < 0))]
    else:
        parent_labels = np.asarray(parent_labels)
        if parent_labels.shape != labels.shape or instances is None:
            raise ValueError("Child propagation needs matching parents and instances")
        instance_parent = np.full(max(len(instances), int(labels.max()) + 1 if np.any(labels >= 0) else 0), -1, np.int32)
        for instance in instances:
            instance_parent[instance.id] = instance.parent
        groups = []
        for parent in np.unique(parent_labels[parent_labels >= 0]):
            mapped_parent = np.full(len(labels), -1, np.int32)
            assigned = labels >= 0
            if assigned.any():
                mapped_parent[assigned] = instance_parent[labels[assigned]]
            source = np.flatnonzero(assigned & (mapped_parent == parent))
            candidates = np.flatnonzero((labels < 0) & (parent_labels == parent))
            groups.append((int(parent), source, candidates))
    required = int(np.ceil(neighbors * consensus))
    for _, source, candidates in groups:
        if not len(source) or not len(candidates):
            continue
        k = min(neighbors, len(source))
        distance, nearest = cKDTree(xyz[source]).query(xyz[candidates], k=k)
        if k == 1:
            distance = distance[:, None]
            nearest = nearest[:, None]
        neighbor_labels = labels[source[nearest]]
        within_labels = np.where(distance <= radius, neighbor_labels, -1)
        best_label = np.full(len(candidates), -1, labels.dtype)
        best_count = np.zeros(len(candidates), np.int16)
        # k is deliberately small; loop over columns while processing all
        # candidates vectorially to avoid a per-Gaussian Python loop.
        for column in range(k):
            candidate_label = within_labels[:, column]
            counts = ((within_labels == candidate_label[:, None]) &
                      (candidate_label[:, None] >= 0)).sum(axis=1)
            improve = counts > best_count
            best_count[improve] = counts[improve]
            best_label[improve] = candidate_label[improve]
        accepted = best_count >= min(required, k)
        result[candidates[accepted]] = best_label[accepted]
    return result
