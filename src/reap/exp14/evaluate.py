"""Verified HF adapters, sample-level generation resume and strict grading."""
import json
import re
import subprocess
import time
from functools import lru_cache
from decimal import Decimal, InvalidOperation
from pathlib import Path
import torch
from .protocol import SETTINGS, MC8, atomic_json, digest
from .compress import locked_identity
FENCE = chr(96)*3
CHUNK_SIZE = 32

@lru_cache(maxsize=None)
def evaluation_rows(root, task):
    root = Path(root)
    rows = json.loads((root/f"evaluation/{task}.json").read_text())
    frozen = json.loads((root/"evaluation/manifest.json").read_text())["datasets"][task]
    if not rows or digest(rows) != frozen["hash"] or len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"Frozen evaluation rows changed or duplicate IDs: {task}")
    return rows

def chunks(root, output, task):
    rows = evaluation_rows(str(root), task)
    folder = Path(output)/f"{task}-chunks"
    return [(start, min(start+CHUNK_SIZE, len(rows)), folder/f"{start:06d}")
            for start in range(0, len(rows), CHUNK_SIZE)]

def pending_chunks(root, output, task):
    return [c for c in chunks(root, output, task) if not c[2].with_suffix(".done.json").exists()]

def generation_ready(root, output, task):
    return task != "mc8" and not pending_chunks(root, output, task)

def read_sample(path, row):
    sample = json.loads(path.read_text())
    if sample["id"] != row["id"] or sample["prompt_hash"] != digest(row["prompt"]):
        raise ValueError("Sample resume identity mismatch")
    return sample

def generate_task(net, tokenizer, root, output, task, identity, device):
    """Primary and helpers claim identical 32-sample chunks without duplication."""
    from .run import lock
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    locked_identity(output/"identity.json", identity)
    rows = evaluation_rows(str(root), task)
    sample_dir = output/f"{task}-samples"; sample_dir.mkdir(exist_ok=True)
    completed = 0
    for start, stop, base in pending_chunks(root, output, task):
        try:
            with lock(base.with_suffix(".lock"), blocking=False):
                if base.with_suffix(".done.json").exists():
                    continue
                began = time.monotonic()
                for row in rows[start:stop]:
                    path = sample_dir/(digest(row["id"])+".json")
                    if path.exists():
                        read_sample(path, row)
                    else:
                        sample_started = time.monotonic()
                        sample = generate(net, tokenizer, row, task, device)
                        sample["prompt_hash"] = digest(row["prompt"])
                        sample["generation_seconds"] = time.monotonic()-sample_started
                        atomic_json(path, sample)
                atomic_json(base.with_suffix(".done.json"), dict(
                    count=stop-start, seconds=time.monotonic()-began,
                    identity_hash=digest(identity)))
                completed += stop-start
        except BlockingIOError:
            continue
    return completed

def grade_task(root, output, task, identity, image="moe-exp14-grader:1"):
    from .run import lock
    root, output = Path(root), Path(output)
    locked_identity(output/"identity.json", identity)
    result_path = output/f"{task}.json"
    with lock(output/f"{task}-grading.lock"):
        if result_path.exists():
            if json.loads(result_path.read_text())["identity_hash"] != digest(identity):
                raise ValueError("Grader result identity mismatch")
            return
        if not generation_ready(root, output, task):
            raise ValueError("Cannot grade incomplete generations")
        rows = evaluation_rows(str(root), task)
        generated = [read_sample(output/f"{task}-samples"/(digest(r["id"])+".json"), r) for r in rows]
        began = time.monotonic()
        if task in ["gsm8k", "math500"]:
            correct = [gsm_correct(s["response"], r["answer"]) if task=="gsm8k" else
                       math_correct(s["response"], r["answer"]) for r,s in zip(rows, generated)]
            result = dict(accuracy=sum(correct)/len(rows), count=len(rows),
                          correct=correct, grader="flexible_numeric" if task=="gsm8k" else "math-verify-0.8.0")
        else:
            result = grade_code(task, rows, generated, root, output, image)
        result.update(identity_hash=digest(identity), grading_seconds=time.monotonic()-began)
        atomic_json(result_path, result)

def flexible_number(text):
    matches = re.findall(r"-?(?:\d[\d,]*)(?:\.\d+)?", text)
    return matches[-1].replace(",", "") if matches else None

def gsm_correct(response, answer):
    value = flexible_number(response)
    if value is None:
        return False
    try:
        return Decimal(value) == Decimal(answer.replace(",", ""))
    except InvalidOperation:
        return False

def math_correct(response, answer):
    from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig
    configs = [LatexExtractionConfig(), ExprExtractionConfig()]
    gold = parse("$"+answer+"$", extraction_config=configs)
    pred = parse(response, extraction_config=configs)
    return bool(gold and pred and verify(gold, pred))

@torch.no_grad()
def generate(net, tokenizer, row, task, device):
    system = ("You are an expert Python programmer. You will be given a question "
              "(problem specification) and will generate a correct Python program "
              "that matches the specification and passes all tests.") if task=="livecodebench" else "You are a helpful assistant."
    prompt = row["prompt"]
    if task in ["humaneval", "mbpp"]:
        prompt = "Write a complete Python solution. Return only the code in a Python code block.\n\n" + prompt
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    tokenizer.truncation_side = "left"
    inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt", truncation=True,
                       max_length=SETTINGS["max_input_length"]).to(device)
    out = net.generate(**inputs, do_sample=False, max_new_tokens=SETTINGS["max_new_tokens"],
                       pad_token_id=tokenizer.eos_token_id, use_cache=True)
    return dict(id=row["id"], response=tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True),
                input_tokens=inputs.input_ids.shape[1], new_tokens=out.shape[1]-inputs.input_ids.shape[1])

def mc8(net, tokenizer, root, output):
    from datasets import DatasetDict, load_from_disk
    from lm_eval import evaluator
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager, get_task_dict
    from unittest.mock import patch
    lm = HFLM(pretrained=net, tokenizer=tokenizer, batch_size=1, max_length=2048)
    manager = TaskManager()
    scores, keys = {}, {}
    # Store each task separately; MMLU is one task with its subject aggregation.
    for task_name in MC8:
        done = output/f"mc8-{task_name}.json"
        if done.exists():
            item = json.loads(done.read_text())
        else:
            def frozen_download(task, *args, **kwargs):
                name = task.config.task
                cache = root/"evaluation/mc8-cache"/name
                if not cache.exists():
                    raise ValueError(f"MC8 frozen split missing: {name}")
                task.dataset = DatasetDict({p.name: load_from_disk(str(p)) for p in cache.iterdir() if p.is_dir()})
                task.config.num_fewshot = 0
            # Substitute the download BEFORE harness initialization caches task_docs
            # and samplers. No worker resolves a remote dataset's mutable main.
            with patch.object(ConfigurableTask, "download", frozen_download):
                task_dict = get_task_dict([task_name], manager)
            result = evaluator.evaluate(lm=lm, task_dict=task_dict, log_samples=True,
                                        bootstrap_iters=1000, verbosity="INFO")
            results = result["results"]
            record = results[task_name]
            key = "acc_norm,none" if "acc_norm,none" in record else "acc,none"
            if key not in record:
                raise ValueError(f"Missing expected MC8 accuracy metric for {task_name}: {list(record)}")
            item = dict(score=record[key], metric=key, results=results)
            atomic_json(done, item)
        scores[task_name] = item["score"]; keys[task_name] = item["metric"]
    return dict(mc_average=sum(scores.values())/8, tasks=scores, metric_keys=keys)

def grade_code(task, rows, generated, root, output, image):
    """Execute generated code only in a no-network, read-only-root container."""
    work = output/f"{task}-grade"; work.mkdir(parents=True, exist_ok=True)
    atomic_json(work/"input.json", {"task": task, "rows": rows, "generated": generated,
                                   "timeout": SETTINGS["code_timeout"]})
    command = ["docker", "run", "--rm", "--network=none", "--read-only", "--cap-drop=ALL",
               "--security-opt=no-new-privileges", "--pids-limit=256", "--memory=8g", "--cpus=4",
               "--tmpfs=/tmp:rw,nosuid,size=2g", "--user", f"{__import__('os').getuid()}:{__import__('os').getgid()}",
               "-e", "HOME=/tmp", "-e", "XDG_CACHE_HOME=/tmp/cache",
               "-e", "HUMANEVAL_OVERRIDE_PATH=/data/humaneval-plus.jsonl",
               "-e", "MBPP_OVERRIDE_PATH=/data/mbpp-plus.jsonl",
               "-v", f"{work.resolve()}:/work:rw",
               "-v", f"{(root/'evaluation').resolve()}:/data:ro",
               image, "python", "-m", "reap.exp14.grade", "/work/input.json"]
    subprocess.run(command, check=True, timeout=24*3600)
    result = json.loads((work/"result.json").read_text())
    if result.get("count") != len(rows) or not 0 <= result["pass@1"] <= 1:
        raise ValueError("Incomplete/invalid code grading result")
    return result

def evaluate(net, tokenizer, root, output, tasks, identity, device="cuda", image="moe-exp14-grader:1",
             defer_grading=False):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    locked_identity(output/"identity.json", identity)
    net.to(device).eval()
    for task in tasks:
        result_path = output/f"{task}.json"
        if result_path.exists():
            continue
        if task == "mc8":
            result = mc8(net, tokenizer, root, output)
            result["identity_hash"] = digest(identity)
            atomic_json(result_path, result)
        else:
            generate_task(net, tokenizer, root, output, task, identity, device)
            if not defer_grading:
                grade_task(root, output, task, identity, image)
    from .methods import CompactMoE
    atomic_json(output/"routing_calls.json", {
        str(i): {"top_k": layer.mlp.top_k, "unique_physical_calls_histogram": layer.mlp.calls,
                 "note": "Includes prefill and decode tokens; reset at start of this process"}
        for i, layer in enumerate(net.model.layers) if isinstance(layer.mlp, CompactMoE)})
