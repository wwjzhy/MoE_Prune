"""Native checkpoint payloads with an explicit reference dequant-GEMM backend.

No quantized integer/FP8 codes are averaged. Only a merged expert is dequantized
and re-encoded; original buffers remain bit-identical. This is not a fused
quantized CUDA kernel and must not be used to claim quantized throughput.
"""
import torch
from torch import nn
from torch.nn import functional as F

def pack4(q):
    if q.ndim != 2 or q.shape[1] % 8:
        raise ValueError("Packed W4 requires columns divisible by 8")
    codes = (q.to(torch.int64) + 8)
    if (codes < 0).any() or (codes > 15).any():
        raise ValueError("W4 signed codes outside [-8, 7]")
    codes = codes.reshape(q.shape[0], -1, 8)
    shifts = torch.arange(8, device=q.device) * 4
    return (codes << shifts).sum(-1).to(torch.int32)

def unpack4(packed, columns):
    shifts = torch.arange(8, device=packed.device) * 4
    codes = ((packed.to(torch.int64).unsqueeze(-1) >> shifts) & 15)
    return (codes.reshape(packed.shape[0], -1)[:, :columns] - 8).to(torch.int8)

def encode_fp8(weight, block=(128, 128)):
    weight = weight.float()
    h, w = weight.shape
    bh, bw = block
    ph, pw = (h + bh - 1) // bh * bh, (w + bw - 1) // bw * bw
    tiles = F.pad(weight, (0, pw-w, 0, ph-h)).reshape(ph//bh, bh, pw//bw, bw).permute(0, 2, 1, 3)
    scale = tiles.abs().amax((-1, -2)).clamp_min(1e-12) / 448.
    q = (tiles / scale[..., None, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    q = q.permute(0, 2, 1, 3).reshape(ph, pw)[:h, :w].contiguous()
    return q, scale

class QuantLinear(nn.Module):
    def __init__(self, rows, columns, spec, shapes=None, device=None):
        super().__init__()
        self.in_features, self.out_features, self.spec = columns, rows, spec
        shapes = shapes or {}
        def buffer(name, shape, dtype):
            self.register_buffer(name, torch.empty(shapes.get(name, shape), dtype=dtype, device=device))
        if spec["kind"] == "fp8":
            bh, bw = spec["block"]
            buffer("weight", (rows, columns), torch.float8_e4m3fn)
            buffer("weight_scale_inv", ((rows+bh-1)//bh, (columns+bw-1)//bw), torch.float32)
        elif spec["kind"] == "w4":
            group = spec["group_size"]
            if columns % group or columns % 8:
                raise ValueError("Unsupported W4 shape/group (no silent fallback)")
            buffer("weight_packed", (rows, columns//8), torch.int32)
            buffer("weight_scale", (rows, columns//group), torch.float16)
            buffer("weight_shape", (2,), torch.int64)
        else:
            raise ValueError(spec)

    def dense(self):
        if self.spec["kind"] == "fp8":
            bh, bw = self.spec["block"]
            scale = self.weight_scale_inv.repeat_interleave(bh, 0).repeat_interleave(bw, 1)
            return self.weight.float() * scale[:self.out_features, :self.in_features]
        q = unpack4(self.weight_packed, self.in_features)
        scale = self.weight_scale.repeat_interleave(self.spec["group_size"], 1)
        return q.float() * scale.float()

    def forward(self, x):
        if self.spec["kind"] == "fp8" and self.spec.get("dynamic_activation", True):
            block = self.spec["block"][1]
            if x.shape[-1] % block:
                raise ValueError("FP8 dynamic input blocks must divide the input dimension")
            t = x.float().reshape(*x.shape[:-1], -1, block)
            s = t.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448.
            x = ((t / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s).reshape_as(x).to(x.dtype)
        return F.linear(x, self.dense().to(x.dtype))

    @torch.no_grad()
    def set_dense(self, weight):
        if self.spec["kind"] == "fp8":
            q, s = encode_fp8(weight, self.spec["block"])
            self.weight.copy_(q); self.weight_scale_inv.copy_(s)
        else:
            group = self.spec["group_size"]
            tiles = weight.float().reshape(self.out_features, -1, group)
            # Symmetric signed RTN in the original compressed-tensors pack format.
            scale = tiles.abs().amax(-1).clamp_min(1e-12) / 7.
            q = (tiles / scale[..., None]).round().clamp(-8, 7).to(torch.int8)
            self.weight_packed.copy_(pack4(q.reshape_as(weight)))
            self.weight_scale.copy_(scale)
            self.weight_shape.copy_(torch.tensor(weight.shape, device=weight.device))
        return self

def dense_weight(linear):
    return linear.dense() if isinstance(linear, QuantLinear) else linear.weight.float()

@torch.no_grad()
def set_weight(linear, weight):
    if isinstance(linear, QuantLinear):
        linear.set_dense(weight)
    else:
        linear.weight.copy_(weight.to(linear.weight))

def quant_spec(config):
    q = getattr(config, "quantization_config", None)
    if not q:
        return None
    if q["quant_method"] == "fp8":
        if q.get("fmt", "e4m3") != "e4m3" or q.get("activation_scheme") != "dynamic":
            raise ValueError("Only block E4M3/dynamic FP8 is supported")
        return dict(kind="fp8", block=q.get("weight_block_size", [128, 128]), dynamic_activation=True)
    if q["quant_method"] == "compressed-tensors":
        if q.get("format") != "pack-quantized":
            raise ValueError("Expected compressed-tensors pack-quantized format")
        groups = list(q["config_groups"].values())
        specs = [g["weights"] for g in groups]
        first = specs[0]
        if any(s != first for s in specs) or first["num_bits"] != 4 or not first["symmetric"]:
            raise ValueError("Only uniform symmetric W4 weight groups are supported")
        if first.get("strategy") != "group" or any(g.get("input_activations") for g in groups):
            raise ValueError("Expected weight-only group W4A16")
        return dict(kind="w4", group_size=first["group_size"])
    raise ValueError(f"Unsupported quantizer: {q['quant_method']}")
