"""Frozen evaluation data, deterministic calibration and contamination filtering."""
import hashlib
import json
import random
import re
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer
from .protocol import SETTINGS, MODELS, atomic_json, digest
FENCE = chr(96)*3
SOURCES = {
    "c4": ("allenai/c4", "en", "train", "text"),
    "math": ("AI-MO/NuminaMath-1.5", None, "train", "problem+solution"),
    "code": ("bigcode/the-stack-smol", None, "train", "content"),
}
def normalize(text):
    return " ".join(re.findall(r"\w+", text.lower()))

class Decontaminator:
    """Reject exact shared 13-word spans (not a fuzzy decontamination claim)."""
    def __init__(self, questions):
        self.spans = set()
        for question in questions:
            words = normalize(question).split()
            self.spans.update(" ".join(words[i:i+13]) for i in range(max(0, len(words)-12)))
    def matches(self, text):
        words = normalize(text).split()
        return any(" ".join(words[i:i+13]) in self.spans for i in range(max(0, len(words)-12)))

def dataset_revision(name):
    return HfApi().dataset_info(name).sha

def write_jsonl(path, rows):
    path = Path(path); temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text("".join(json.dumps(row, default=str)+"\n" for row in rows))
    temp.replace(path)

def freeze_evaluation(root):
    root = Path(root); output = root/"evaluation"
    if (output/"manifest.json").exists():
        return json.loads((output/"manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    from evalplus.data.humaneval import _ready_human_eval_plus_path
    from evalplus.data.mbpp import _ready_mbpp_plus_path
    manifest = {"settings": SETTINGS, "datasets": {}}
    questions = []
    for task, getter, raw_path in [
        ("humaneval", get_human_eval_plus, _ready_human_eval_plus_path),
        ("mbpp", get_mbpp_plus, _ready_mbpp_plus_path)]:
        problems = getter()
        raw_rows = [json.loads(line) for line in Path(raw_path()).read_text().splitlines()]
        write_jsonl(output/f"{task}-plus.jsonl", raw_rows)
        rows = []
        for key, p in problems.items():
            prompt = p["prompt"]
            if task == "mbpp":
                prompt += "\n\n" + "\n".join(p.get("test_list", []))
            rows.append(dict(id=key, prompt=prompt, entry_point=p["entry_point"]))
            questions.append(prompt)
        atomic_json(output/f"{task}.json", rows)
        manifest["datasets"][task] = {"hash": digest(rows), "plus_hash": digest(raw_rows), "count": len(rows)}
    for task, name, subset, question_field in [
        ("gsm8k", "openai/gsm8k", "main", "question"),
        ("math500", "HuggingFaceH4/MATH-500", None, "problem")]:
        revision = dataset_revision(name)
        ds = load_dataset(name, subset, split="test", revision=revision)
        rows = [dict(id=str(i), prompt=row[question_field] +
                     "\nPlease reason step by step and put the final answer in \\boxed{}.",
                     answer=(row["answer"].split("####")[-1].strip() if task=="gsm8k" else row["answer"]))
                for i, row in enumerate(ds)]
        questions.extend(row[question_field] for row in ds)
        atomic_json(output/f"{task}.json", rows)
        manifest["datasets"][task] = dict(id=name, revision=revision, count=len(rows), hash=digest(rows))
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    name = "livecodebench/code_generation_lite"; revision = dataset_revision(name)
    ds = load_dataset(name, split="test", revision=revision,
                      version_tag=SETTINGS["lcb_release"], trust_remote_code=True)
    rows = []
    actual_dates = []
    for row in ds:
        p = CodeGenerationProblem(**row)
        date = p.contest_date.date().isoformat()
        if not SETTINGS["lcb_start"] <= date <= SETTINGS["lcb_end"]:
            continue
        if p.starter_code:
            fmt = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
            source_code = p.starter_code
        else:
            fmt = ("Read the inputs from stdin solve the problem and write the answer to stdout "
                   "(do not directly test on the sample inputs). Enclose your code within delimiters "
                   "as follows. Ensure that when the python program runs, it reads the inputs, "
                   "runs the algorithm and writes output to STDOUT.")
            source_code = "# YOUR CODE HERE"
        prompt = f"### Question:\n{p.question_content}\n\n### Format: {fmt}\n{FENCE}python\n{source_code}\n{FENCE}\n\n### Answer: (use the provided format with backticks)\n\n"
        rows.append(dict(id=p.question_id, prompt=prompt, tests=p.get_evaluation_sample()))
        actual_dates.append(date)
        questions.append(p.question_content)
    rows.sort(key=lambda row: row["id"])
    if not rows:
        raise ValueError("LCB release/date filter produced zero tasks")
    atomic_json(output/"livecodebench.json", rows)
    manifest["datasets"]["livecodebench"] = dict(id=name, revision=revision, count=len(rows), hash=digest(rows),
                                                actual_start=min(actual_dates), actual_end=max(actual_dates),
                                                release=SETTINGS["lcb_release"])
    # Serialize harness evaluation splits: every worker reads these exact records.
    from lm_eval.tasks import TaskManager, get_task_dict
    from .protocol import MC8
    task_dict = get_task_dict(MC8, TaskManager())
    mc_manifest = {}
    def leaves(tasks):
        for name, task in tasks.items():
            if isinstance(task, dict):
                yield from leaves(task)
            else:
                yield name, task
    for name, task in leaves(task_dict):
        if not hasattr(task, "eval_docs"):
            continue
        docs = list(task.eval_docs)
        rendered = [str(task.doc_to_text(doc)) for doc in docs]
        questions.extend(rendered)
        for split_name, split in task.dataset.items():
            split.save_to_disk(str(output/"mc8-cache"/name/split_name))
        mc_manifest[name] = {"count": len(docs), "text_hash": digest(rendered), "config": str(task.config)}
    manifest["mc8"] = mc_manifest
    atomic_json(output/"questions.json", questions)
    atomic_json(output/"manifest.json", manifest)
    return manifest

def sample_source(kind, count, tokenizer, decontam):
    name, subset, split, field = SOURCES[kind]
    revision = dataset_revision(name)
    dataset = load_dataset(name, subset, split=split, revision=revision, streaming=True)
    dataset = dataset.shuffle(seed=42, buffer_size=10000)
    rows, rejected = [], {"overlap": 0, "short": 0, "excluded_source": 0}
    seen = set()
    for index, row in enumerate(dataset):
        if kind == "math":
            if str(row.get("source", "")).lower() not in ["cn_k12", "olympiads"]:
                rejected["excluded_source"] += 1; continue
            text = row["problem"] + "\n" + row["solution"]
        else:
            text = row[field]
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        if text_hash in seen or decontam.matches(text):
            rejected["overlap"] += 1; continue
        ids = tokenizer(text, add_special_tokens=True, truncation=True, max_length=512)["input_ids"]
        if len(ids) < 32:
            rejected["short"] += 1; continue
        rows.append(dict(source=kind, source_index=index, source_id=row.get("id", row.get("path")),
                         text=text, text_hash=text_hash, ids=ids))
        seen.add(text_hash)
        if len(rows) == count:
            break
    if len(rows) != count:
        raise ValueError(f"Insufficient clean {kind} samples: {len(rows)}/{count}")
    return rows, dict(id=name, subset=subset, split=split, revision=revision,
                     field=field, rejected=rejected, shuffle_buffer=10000,
                     source_filter=["cn_k12", "olympiads"] if kind=="math" else None)

def prepare(root, model_ids=None):
    root = Path(root); manifest = freeze_evaluation(root)
    decontam = Decontaminator(json.loads((root/"evaluation/questions.json").read_text()))
    model_ids = model_ids or MODELS
    if (root/"models.json").exists():
        frozen_models = json.loads((root/"models.json").read_text())
        if {k:v["id"] for k,v in frozen_models.items()} != model_ids:
            raise ValueError("Model IDs changed; use a fresh root")
        model_revisions = {k:v["revision"] for k,v in frozen_models.items()}
    else:
        model_revisions = {key: HfApi().model_info(model).sha for key, model in model_ids.items()}
        atomic_json(root/"models.json", {key: dict(id=model_ids[key], revision=rev)
                                          for key, rev in model_revisions.items()})
    tokenizer = AutoTokenizer.from_pretrained(model_ids["Q3"], revision=model_revisions["Q3"])
    for track, kinds in [("G", [("c4", 3072)]), ("X", [("math", 1536), ("code", 1536)])]:
        raw_path = root/f"calibration/{track}/raw.json"
        if raw_path.exists():
            raw = json.loads(raw_path.read_text())
        else:
            rows, sources = [], {}
            for kind, count in kinds:
                part, info = sample_source(kind, count, tokenizer, decontam)
                rows.extend(part); sources[kind] = info
            if track == "X":
                random.Random(42).shuffle(rows)
            raw = dict(rows=rows, sources=sources, seed=42, max_length=512,
                       decontamination={"rule": "exact normalized 13-word overlap",
                                        "evaluation_hash": digest(manifest)}, raw_hash=digest(rows))
            atomic_json(raw_path, raw)
        for key, model in model_ids.items():
            target = root/f"calibration/{track}/{key}.json"
            if target.exists():
                continue
            tok = AutoTokenizer.from_pretrained(model, revision=model_revisions[key])
            ids = [tok(row["text"], add_special_tokens=True, truncation=True, max_length=512)["input_ids"]
                   for row in raw["rows"]]
            frozen = dict(model=model, revision=model_revisions[key], track=track,
                          ids=ids, token_hash=digest(ids), raw_hash=raw["raw_hash"],
                          source_manifest=raw["sources"], effective_tokens=sum(map(len, ids)),
                          lengths=[len(row) for row in ids], cross_sample_packing=False,
                          num_sequences=3072, max_length=512)
            atomic_json(target, frozen)
    atomic_json(root/"models.json", {key: dict(id=model_ids[key], revision=rev)
                                      for key, rev in model_revisions.items()})
    atomic_json(root/"prepared.json", {"protocol": SETTINGS, "evaluation_hash": digest(manifest)})
