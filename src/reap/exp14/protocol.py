"""Frozen budgets, jobs and content-addressed experiment records."""
import hashlib
import json
import os
from pathlib import Path

VERSION = 1
MODELS = {
    "Q3": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Q15": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    "F8": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    "W4": "RedHatAI/Qwen3-30B-A3B-Instruct-2507-quantized.w4a16",
}
MC8 = ["arc_challenge", "arc_easy", "boolq", "hellaswag", "mmlu", "openbookqa", "rte", "winogrande"]
X_TASKS = ["humaneval", "mbpp", "livecodebench", "gsm8k", "math500"]
SETTINGS = {
    "seed": 42, "sequences": 3072, "max_calibration_length": 512,
    "max_input_length": 2048, "max_new_tokens": 1024,
    "alignment_samples": 32768, "pca_rank": 64, "ream_group_size": 16,
    "lcb_release": "release_v6", "lcb_start": "2025-01-01", "lcb_end": "2025-07-31",
    "code_timeout": 120, "num_fewshot": 0, "sampling": False,
    "stats_version": 1,
}

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)

def jobs():
    result = []
    def add(section, model, track, k, method, host, protected=None, variant="full", depends=None):
        name = f"{section}-{model}-{track}-K{k}-{method}"
        if method == "SPRM":
            name += f"-P{protected}-{variant}"
        result.append(dict(id=name, section=section, model=model, track=track, k=k,
                           method=method, protected=protected, variant=variant,
                           host=host, depends=depends, seed=42))
        return name
    for track, host in [("X", "A"), ("G", "B")]:
        for k, p in [(42, 34), (32, 26)]:
            for method in ["REAP", "HC", "REAM", "SPRM"]:
                add("main", "Q3", track, k, method, host, p if method == "SPRM" else None)
    for p in [0, 3, 6, 9, 12]:
        full = add("ablation", "Q15", "X", 15, "SPRM", "C", p)
    for variant in ["random_protect", "random_group", "no_alignment", "uniform_fusion", "centroid_router"]:
        add("ablation", "Q15", "X", 15, "SPRM", "C", 12, variant, full)
    for model in ["F8", "W4"]:
        for track in ["X", "G"]:
            for method in ["REAP", "REAM", "SPRM"]:
                add("quant", model, track, 32, method, "D" if track == "X" else "B",
                    26 if method == "SPRM" else None)
    for method in ["REAP", "HC", "REAM", "SPRM"]:
        add("efficiency", "Q15", "G", 15, method, "A", 12 if method == "SPRM" else None)
    for model, host in [("Q3", "B"), ("Q15", "B"), ("F8", "D"), ("W4", "D")]:
        add("reference", model, "X" if model == "Q15" else "GX", 0, "Teacher", host)
    assert len(result) == 46
    assert len([j for j in result if j["method"] != "Teacher"]) == 42
    return result

def benchmarks(job):
    if job["section"] == "efficiency":
        return []
    return (["mc8"] if "G" in job["track"] else []) + (X_TASKS if "X" in job["track"] else [])

def validate_groups(groups, e, k, partial=False):
    flat = [i for group in groups for i in group]
    if len(groups) != k or any(not g for g in groups) or len(set(flat)) != len(flat):
        raise ValueError("Groups must be nonempty, disjoint, and exactly K")
    if any(i < 0 or i >= e for i in flat) or (not partial and sorted(flat) != list(range(e))):
        raise ValueError("Residual coverage / source expert index mismatch")
