"""Physical compression, grouping and permutation-invariant expert fusion.

REAM equations/grouping/alignment follow SamsungSAILMontreal/ream (84a3030);
HC uses average-linkage output means and frequency fusion (no permutation).
See experiments/exp14/README.md for exact baseline conventions.
"""
import copy
import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F
from .protocol import validate_groups
from .quant import QuantLinear, dense_weight, set_weight

def order(scores, ids=None):
    ids = list(range(len(scores))) if ids is None else list(ids)
    return sorted(ids, key=lambda i: (-float(scores[i]), i))

def similarity(stats, ream=False):
    gram = stats["gate_gram"].float()
    norms = gram.diag().clamp_min(1e-20).sqrt()
    cosine = (gram / (norms[:, None] * norms[None, :])).clamp(-1, 1)
    if ream:
        distance = (2 - 2 * cosine).clamp_min(0).sqrt()
        gate = 1 - distance / distance.max().clamp_min(1e-20)
        out = (F.normalize(stats["gated_mean"].float(), dim=1) @
               F.normalize(stats["gated_mean"].float(), dim=1).T + 1) / 2
    else:
        gram = stats["prob_gram"].float()
        norms = gram.diag().clamp_min(1e-20).sqrt()
        gate = gram / (norms[:, None] * norms[None, :])
        out = F.normalize(stats["gated_mean"].float(), dim=1)
        out = out @ out.T
    return (gate + out) / 2

def balanced_groups(ids, count, sim, saliency, random=False, seed=42):
    ids = list(ids)
    if not 0 < count <= len(ids):
        raise ValueError("Invalid residual group budget")
    if random:
        ids = np.random.default_rng(seed).permutation(ids).tolist()
        return [order(saliency, g.tolist()) for g in np.array_split(ids, count)]
    # Saliency anchors; assign most similar pairs subject to fixed balanced capacity.
    anchors = order(saliency, ids)[:count]
    groups = [[a] for a in anchors]
    capacities = [len(ids)//count + (i < len(ids)%count) for i in range(count)]
    unassigned = set(ids) - set(anchors)
    pairs = sorted(((-float(sim[a, i]), g, i) for g, a in enumerate(anchors) for i in unassigned))
    for _, g, i in pairs:
        if i in unassigned and len(groups[g]) < capacities[g]:
            groups[g].append(i); unassigned.remove(i)
    assert not unassigned
    return [order(saliency, g) for g in groups]

def groups_for(job, stats, full_groups=None):
    e = len(stats["saliency"]); k = job["k"]; method = job["method"]
    scores = stats["saliency"]
    if method == "REAP":
        groups = [[i] for i in order(scores)[:k]]
        return groups, "centroid", False, scores
    if method == "HC":
        features = stats["output_mean"].float().numpy()
        labels = cut_tree(linkage(features, method="average", metric="euclidean"), n_clusters=k).ravel()
        groups = []
        for label in range(k):
            ids = np.flatnonzero(labels == label).tolist()
            centroid = min(ids, key=lambda i: (np.linalg.norm(features[i]-features[ids].mean(0)), i))
            groups.append([centroid] + [i for i in ids if i != centroid])
        return groups, "logical", False, stats["frequency"]
    if method == "REAM":
        scores = stats["ream_saliency"].clone()
        positive = scores[scores > 0]
        if not len(positive):
            raise ValueError("REAM cannot fuse a layer with all-zero saliency")
        scores[scores == 0] = min(.5, float(positive.min()))
        anchors = order(scores)[:k]; assigned = set(anchors); groups = []
        sim = similarity(stats, ream=True)
        for a in anchors:
            candidates = sorted(set(range(e))-assigned, key=lambda i: (-float(sim[a, i]), i))
            group = [a] + candidates[:15]  # official group_size=16 incl. centroid
            assigned.update(group); groups.append(group)
        if len(assigned) != e:
            raise ValueError("REAM group capacity does not cover E")
        return groups, "centroid", True, scores
    if method != "SPRM":
        raise ValueError(method)
    variant = job["variant"]; p = job["protected"]
    if not 0 <= p < k:
        raise ValueError("SPRM needs at least one Super-Expert")
    if full_groups is not None and variant in ["no_alignment", "uniform_fusion", "centroid_router"]:
        groups = full_groups
    else:
        protected = order(scores)[:p]
        if variant == "random_protect":
            protected = np.random.default_rng(job["seed"]).choice(e, p, replace=False).tolist()
        residual = [i for i in range(e) if i not in protected]
        groups = [[i] for i in protected] + balanced_groups(
            residual, k-p, similarity(stats), scores,
            random=variant == "random_group", seed=job["seed"])
    return groups, ("centroid" if variant == "centroid_router" else "grouped"), variant != "no_alignment", (
        torch.ones_like(scores) if variant == "uniform_fusion" else scores)

@torch.no_grad()
def weight_features(experts, rank=64, device="cpu"):
    # Joint PCA of neuron signatures [gate row, up row, down column].
    features = torch.cat([F.normalize(torch.cat([
        dense_weight(e.gate_proj).cpu(), dense_weight(e.up_proj).cpu(),
        dense_weight(e.down_proj).cpu().T], dim=1), dim=1) for e in experts])
    features = features.to(device)
    center = features.mean(0, keepdim=True); features -= center
    rank = min(rank, min(features.shape)-1)
    with torch.random.fork_rng(devices=[features.device.index] if features.is_cuda else []):
        torch.manual_seed(0)
        _, s, v = torch.pca_lowrank(features, q=rank, center=False)
    projected = features @ (v * s)
    return F.normalize(projected.reshape(len(experts), -1, rank), dim=-1).cpu()

@torch.no_grad()
def fuse(experts, ids, scores, activations=None, weights=None, align=True, alignment_device=None):
    if len(ids) == 1:
        return experts[ids[0]]
    merged = copy.deepcopy(experts[ids[0]])
    coeff = scores[ids].float().clamp_min(0)
    coeff = coeff / coeff.sum() if coeff.sum() > 0 else torch.ones_like(coeff)/len(ids)
    acc = {n: torch.zeros_like(dense_weight(getattr(merged, n)), dtype=torch.float32)
           for n in ["gate_proj", "up_proj", "down_proj"]}
    ref = ids[0]
    for i, alpha in zip(ids, coeff):
        perm = None
        if align and i != ref:
            if activations is None or weights is None:
                raise ValueError("Alignment requires frozen activation and weight features")
            a = F.normalize(activations[ref].float().T.to(alignment_device), dim=1)
            b = F.normalize(activations[i].float().T.to(alignment_device), dim=1)
            cost = torch.cdist(a, b) + torch.cdist(weights[ref].to(a.device), weights[i].to(a.device))
            _, cols = linear_sum_assignment(cost.cpu().numpy())
            perm = torch.as_tensor(cols, device=dense_weight(experts[i].gate_proj).device)
        for name in acc:
            w = dense_weight(getattr(experts[i], name))
            if perm is not None:
                w = w[:, perm] if name == "down_proj" else w[perm]
            acc[name].add_(w.to(acc[name].device), alpha=float(alpha))
    for name, w in acc.items():
        set_weight(getattr(merged, name), w)
    return merged

class CompactMoE(nn.Module):
    """K physical FFNs; source router retained only for grouped/logical routing."""
    def __init__(self, source, groups, mode, experts=None):
        super().__init__()
        validate_groups(groups, len(source.experts), len(groups), partial=mode == "centroid")
        self.groups, self.mode = groups, mode
        self.num_experts, self.top_k = len(groups), source.top_k
        if self.top_k > self.num_experts:
            raise ValueError("Compression K smaller than original active top-k")
        self.norm_topk_prob = source.norm_topk_prob
        self.experts = nn.ModuleList(experts if experts is not None else [source.experts[g[0]] for g in groups])
        gate_tensor = next(source.gate.parameters(), None)
        if gate_tensor is None:
            gate_tensor = next(source.gate.buffers())
        if mode == "centroid":
            indices = [g[0] for g in groups]
            if isinstance(source.gate, QuantLinear):
                self.gate = QuantLinear(len(groups), source.gate.in_features,
                                        source.gate.spec, device=gate_tensor.device)
                with torch.no_grad():
                    if source.gate.spec["kind"] == "w4":
                        self.gate.weight_packed.copy_(source.gate.weight_packed[indices])
                        self.gate.weight_scale.copy_(source.gate.weight_scale[indices])
                        self.gate.weight_shape.copy_(torch.tensor(
                            [len(groups), source.gate.in_features], device=gate_tensor.device))
                    else:
                        self.gate.set_dense(source.gate.dense()[indices])
            else:
                self.gate = nn.Linear(source.gate.in_features, len(groups), bias=False,
                                      device=gate_tensor.device, dtype=gate_tensor.dtype)
                with torch.no_grad():
                    self.gate.weight.copy_(source.gate.weight[indices])
        else:
            self.gate = source.gate
        mapping = torch.empty(len(source.experts), dtype=torch.long, device=gate_tensor.device)
        # REAP/centroid has no source map; -1 prevents accidental use.
        if mode == "centroid":
            mapping.fill_(-1)
        else:
            for i, group in enumerate(groups):
                mapping[group] = i
        self.register_buffer("source_to_physical", mapping)
        if hasattr(source, "shared_expert"):
            self.shared_expert = source.shared_expert
            self.shared_expert_gate = source.shared_expert_gate
        self.register_buffer("call_hist", torch.zeros(self.top_k+1, dtype=torch.long), persistent=False)

    @property
    def calls(self):
        return self.call_hist.cpu().tolist()

    def routing(self, x):
        logits = self.gate(x)
        if self.mode == "grouped":
            logits = torch.stack([torch.logsumexp(logits[:, g].float(), -1) for g in self.groups], -1)
        weights = logits.float().softmax(-1)
        weights, selected = weights.topk(self.top_k, dim=-1)
        if self.norm_topk_prob:
            weights = weights / weights.sum(-1, keepdim=True)
        if self.mode == "logical":
            selected = self.source_to_physical[selected]
        return logits, weights, selected

    def forward(self, hidden_states):
        shape = hidden_states.shape; x = hidden_states.reshape(-1, shape[-1])
        logits, weights, selected = self.routing(x)
        physical = torch.zeros(x.shape[0], self.num_experts, dtype=weights.dtype, device=x.device)
        physical.scatter_add_(1, selected, weights)
        counts = (physical > 0).sum(-1)
        self.call_hist.add_(torch.bincount(counts, minlength=self.top_k+1))
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            idx = (physical[:, i] > 0).nonzero().flatten()
            if len(idx):
                value = expert(x[idx]) * physical[idx, i, None].to(x.dtype)
                out.index_add_(0, idx, value.to(x.dtype))
        if hasattr(self, "shared_expert"):
            out += self.shared_expert(x) * torch.sigmoid(self.shared_expert_gate(x))
        return out.reshape(shape), logits
