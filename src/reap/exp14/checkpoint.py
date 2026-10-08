"""Streaming source loader and reloadable, unique-expert checkpoints."""
import json
import os
import hashlib
from pathlib import Path
import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from .methods import CompactMoE
from .quant import QuantLinear, quant_spec
from .protocol import atomic_json

def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024*1024):
            result.update(chunk)
    return result.hexdigest()

def source_path(model, revision=None):
    if Path(model).is_dir():
        return Path(model)
    return Path(snapshot_download(model, revision=revision, allow_patterns=[
        "*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken", "merges.txt", "vocab.json"]))

def headers(path):
    entries = {}
    files = sorted(Path(path).glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors checkpoint in {path}")
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in entries:
                    raise ValueError(f"Duplicate tensor: {key}")
                entries[key] = (file, f.get_slice(key).get_shape())
    return entries

def load(model, revision=None, device="cpu"):
    path = source_path(model, revision)
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    if config.model_type not in ["qwen2_moe", "qwen3_moe"]:
        raise ValueError(f"Exp14 supports Qwen2/3 MoE, not {config.model_type}")
    # Keep source E in config: compact metadata replaces the FFNs before loading.
    with init_empty_weights(include_buffers=False):
        net = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
        meta_path = path / "exp14.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            for name, expected_hash in meta.get("checksums", {}).items():
                if Path(name).name != name or file_hash(path/name) != expected_hash:
                    raise ValueError(f"Checkpoint checksum mismatch: {name}")
            for index, info in meta["layers"].items():
                layer = net.model.layers[int(index)]
                layer.mlp = CompactMoE(layer.mlp, info["groups"], info["mode"])
        entries = headers(path)
        spec = quant_spec(config)
        if spec:
            # Replace only linears actually represented by quantized payloads.
            for name, module in list(net.named_modules()):
                if not isinstance(module, torch.nn.Linear):
                    continue
                marker = name + (".weight_scale_inv" if spec["kind"] == "fp8" else ".weight_packed")
                if marker not in entries:
                    continue
                shapes = {key[len(name)+1:]: shape for key, (_, shape) in entries.items() if key.startswith(name+".")}
                parent_name, leaf = name.rsplit(".", 1)
                setattr(net.get_submodule(parent_name), leaf,
                        QuantLinear(module.out_features, module.in_features, spec, shapes, device="meta"))
    expected = set(net.state_dict())
    unexpected = set(entries) - expected
    # tied embedding weights can legitimately omit lm_head.weight.
    if unexpected:
        raise ValueError(f"Unrecognized checkpoint payload (no fallback): {sorted(unexpected)[:12]}")
    loaded = set()
    for file in sorted({item[0] for item in entries.values()}):
        state = {}
        with safe_open(file, framework="pt", device="cpu") as f:
            for key in f.keys():
                value = f.get_tensor(key)
                # Ordinary compute weights BF16, quant buffers retain source dtype.
                module_name = key.rsplit(".", 1)[0]
                is_quant = isinstance(net.get_submodule(module_name), QuantLinear)
                if value.is_floating_point() and not is_quant:
                    value = value.to(torch.bfloat16)
                state[key] = value
        net.load_state_dict(state, strict=False, assign=True); loaded.update(state)
        del state
    net.tie_weights()
    remaining = [name for name, value in net.state_dict().items() if value.is_meta]
    if remaining:
        raise ValueError(f"Checkpoint missing tensors: {remaining[:12]}")
    net.eval().requires_grad_(False)
    return net.to(device), AutoTokenizer.from_pretrained(path, local_files_only=True), path

def save(net, tokenizer, path, metadata):
    path = Path(path)
    staging = path.with_name(path.name+".incomplete")
    staging.mkdir(parents=True, exist_ok=True)
    net.cpu().save_pretrained(staging, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(staging)
    metadata = dict(metadata, checksums={p.name: file_hash(p)
                    for p in staging.iterdir() if p.is_file() and p.name!="exp14.json"})
    atomic_json(staging/"exp14.json", metadata)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite completed checkpoint {path}")
    os.replace(staging, path)
