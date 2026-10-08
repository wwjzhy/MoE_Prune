"""Layerwise statistics, sequential REAM and resumable physical compression."""
import gc
import json
import os
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from .methods import CompactMoE, groups_for, fuse, weight_features
from .protocol import SETTINGS, digest, atomic_json, validate_groups
from .checkpoint import save

def atomic_torch(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(value, tmp); os.replace(tmp, path)

def tree(value, device):
    if torch.is_tensor(value):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {k: tree(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(tree(v, device) for v in value)
    return value

class CaptureDone(Exception):
    pass

@torch.no_grad()
def initial_states(net, samples, device):
    states = []
    def capture(module, args, kwargs):
        states.append((tree(args[0], "cpu"), tree(kwargs, "cpu")))
        raise CaptureDone
    handle = net.model.layers[0].register_forward_pre_hook(capture, with_kwargs=True)
    net.model.embed_tokens.to(device); net.model.rotary_emb.to(device)
    try:
        for ids in samples:
            ids = torch.tensor([ids], device=device, dtype=torch.long)
            try:
                net(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            except CaptureDone:
                pass
    finally:
        handle.remove(); net.model.embed_tokens.cpu(); net.model.rotary_emb.cpu()
    return states

class Statistics:
    def __init__(self, moe, samples, method="shared"):
        self.moe = moe; self.e = len(moe.experts); self.n = 0; self.sequences = 0
        h = moe.gate.in_features
        self.frequency = torch.zeros(self.e, dtype=torch.float64)
        self.saliency_sum = torch.zeros(self.e, dtype=torch.float64)
        self.ream_sum = torch.zeros(self.e, dtype=torch.float64)
        self.output_sum = torch.zeros(self.e, h)
        self.gated_sum = torch.zeros(self.e, h)
        self.gate_gram = torch.zeros(self.e, self.e)
        self.prob_gram = torch.zeros(self.e, self.e)
        self.activations = []
        self.per_sequence = max(1, SETTINGS["alignment_samples"] // samples)
        self.method = method

    @torch.no_grad()
    def __call__(self, module, args, output):
        x = args[0].reshape(-1, args[0].shape[-1])
        if self.n == 0:
            for name in ["frequency", "saliency_sum", "ream_sum", "output_sum",
                         "gated_sum", "gate_gram", "prob_gram"]:
                setattr(self, name, getattr(self, name).to(x.device))
        logits = module.gate(x).float()
        prob = logits.softmax(-1)
        values, selected = prob.topk(module.top_k, -1)
        active = torch.zeros_like(prob).scatter_(1, selected, 1).bool()
        normalized = prob / values.sum(-1, keepdim=True) if module.norm_topk_prob else prob
        self.frequency += active.sum(0)
        self.gate_gram += logits.T @ logits
        self.prob_gram += prob.T @ prob
        gen = torch.Generator().manual_seed(42+self.sequences)
        sample = torch.randperm(len(x), generator=gen)[:self.per_sequence].to(x.device)
        acts = []
        for i, expert in enumerate(module.experts):
            if self.method == "REAP":
                hit = active[:, i]
                if hit.any():
                    out = expert(x[hit]).float()
                    self.saliency_sum[i] += (out.norm(dim=-1) * normalized[hit, i]).sum()
                continue
            act = expert.act_fn(expert.gate_proj(x)) * expert.up_proj(x)
            out = expert.down_proj(act).float()
            norm = out.norm(dim=-1)
            self.saliency_sum[i] += (norm * normalized[:, i] * active[:, i]).sum()
            # Official REAM implementation evaluates gated expert outputs and
            # multiplies their norms by unnormalized top-k probabilities again.
            self.ream_sum[i] += (norm * prob[:, i].square() * active[:, i]).sum()/active[:, i].sum().clamp_min(1)
            self.output_sum[i] += out.sum(0)
            self.gated_sum[i] += (out * prob[:, i, None]).sum(0)
            if self.method != "HC":
                acts.append(act[sample])
        if acts:
            self.activations.append(torch.stack(acts).cpu())
        self.n += len(x); self.sequences += 1

    def result(self):
        return tree(dict(saliency=(self.saliency_sum/self.frequency.clamp_min(1)).float(),
                    ream_saliency=(self.ream_sum/self.sequences).float(),
                    frequency=self.frequency.float(), output_mean=self.output_sum/self.n,
                    gated_mean=self.gated_sum/self.n, gate_gram=self.gate_gram,
                    prob_gram=self.prob_gram, activations=torch.cat(self.activations, dim=1)
                    if self.activations else torch.empty(self.e, 0, 0),
                    valid_tokens=self.n, sequences=self.sequences), "cpu")

@torch.no_grad()
def replay(layer, states, device, observer=None):
    layer.to(device)
    handle = layer.mlp.register_forward_hook(observer) if observer else None
    outputs = []
    try:
        for hidden, kwargs in states:
            result = layer(tree(hidden, device), **tree(kwargs, device))
            result = result[0] if isinstance(result, tuple) else result
            outputs.append((tree(result, "cpu"), kwargs))
    finally:
        if handle:
            handle.remove()
        layer.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return outputs

def locked_identity(path, identity):
    from .run import lock
    path = Path(path)
    with lock(path.with_name(path.name+".lock")):
        if path.exists():
            if json.loads(path.read_text()) != identity:
                raise ValueError(f"Resume/cache identity mismatch: {path}")
        else:
            atomic_json(path, identity)

@torch.no_grad()
def ensure_stats(net, samples, cache, identity, device="cuda", method="shared"):
    """Teacher trajectory cache; callers must hold a cache lock."""
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    locked_identity(cache/"identity.json", identity)
    progress = cache/"progress.json"; rolling = cache/"states.pt"
    bundle = torch.load(rolling, weights_only=True) if rolling.exists() else None
    next_layer = bundle["next_layer"] if bundle else 0
    if next_layer == len(net.model.layers):
        return
    states = bundle["states"] if bundle else initial_states(net, samples, device)
    for i in range(next_layer, len(net.model.layers)):
        layer = net.model.layers[i]
        collector = Statistics(layer.mlp, len(samples), method)
        states = replay(layer, states, device, collector)
        stats = collector.result()
        if method in ["shared", "SPRM"]:
            stats["weight_features"] = weight_features(layer.mlp.experts, SETTINGS["pca_rank"], device)
        atomic_torch(cache/f"layer-{i:03d}.pt", stats)
        atomic_torch(rolling, dict(next_layer=i+1, states=states))
        atomic_json(progress, dict(next_layer=i+1))
        print(f"teacher stats layer {i+1}/{len(net.model.layers)}", flush=True)
        del collector, stats; gc.collect()

@torch.no_grad()
def compress(net, tokenizer, samples, cache, output, job, identity, device="cuda", full_meta=None):
    output = Path(output); progress_dir = output.parent/(output.name+".resume")
    progress_dir.mkdir(parents=True, exist_ok=True)
    locked_identity(progress_dir/"identity.json", identity)
    metadata = dict(protocol=1, identity=identity, job=job, layers={},
                    backend="native-payload/reference-dequant-gemm" if
                    getattr(net.config, "quantization_config", None) else "bf16-hf")
    sequential = job["method"] == "REAM"
    prefix = 0
    while (progress_dir/f"layer-{prefix:03d}.pt").exists():
        record = torch.load(progress_dir/f"layer-{prefix:03d}.pt", weights_only=True)
        source = net.model.layers[prefix].mlp
        compact = CompactMoE(source, record["info"]["groups"], record["info"]["mode"])
        compact.load_state_dict(record["state"])
        net.model.layers[prefix].mlp = compact
        metadata["layers"][str(prefix)] = record["info"]
        prefix += 1
    if sequential and prefix < len(net.model.layers):
        # Layer records and states are one atomic bundle, so a crash cannot mix
        # different sequential trajectories on resume.
        states = (torch.load(progress_dir/f"layer-{prefix-1:03d}.pt", weights_only=True)["next_states"]
                  if prefix else initial_states(net, samples, device))
    for i in range(prefix, len(net.model.layers)):
        start = time.monotonic()
        layer = net.model.layers[i]; source = layer.mlp
        if sequential:
            collector = Statistics(source, len(samples))
            replay(layer, states, device, collector)
            stats = collector.result(); del collector
        else:
            stats = torch.load(Path(cache)/f"layer-{i:03d}.pt", weights_only=True)
        full_groups = full_meta["layers"][str(i)]["groups"] if full_meta else None
        groups, mode, align, scores = groups_for(job, stats, full_groups)
        validate_groups(groups, len(source.experts), job["k"], partial=job["method"] == "REAP")
        features = stats.get("weight_features") if align else None
        if align and features is None:
            features = weight_features(source.experts, SETTINGS["pca_rank"], device)
        # CPU fusion avoids retaining E dequantized FFNs on the GPU.
        source.cpu()
        experts = [fuse(source.experts, g, scores, stats["activations"], features, align,
                       alignment_device=device) for g in groups]
        compact = CompactMoE(source, groups, mode, experts)
        layer.mlp = compact
        info = dict(groups=groups, mode=mode, physical_experts=len(groups),
                    logical_router_slots=compact.gate.out_features,
                    protected=job.get("protected"), alignment=align,
                    seconds=time.monotonic()-start)
        metadata["layers"][str(i)] = info
        record = dict(info=info, state=compact.state_dict())
        if sequential:
            states = replay(layer, states, device)
            record["next_states"] = states
        atomic_torch(progress_dir/f"layer-{i:03d}.pt", record)
        print(f"{job['id']} compressed layer {i+1}/{len(net.model.layers)}", flush=True)
        del stats, features, source; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # Keep source E in architecture config; exp14.json is the physical topology.
    net.config.exp14_physical_experts = job["k"]
    save(net, tokenizer, output, metadata)
    return metadata
