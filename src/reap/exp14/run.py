"""Exp14 CLI: prepare, stats, one job, four-node queue, aggregate."""
import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from .protocol import SETTINGS, MODELS, jobs, benchmarks, atomic_json, digest

@contextmanager
def lock(path, blocking=True):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)

def read(path):
    return json.loads(Path(path).read_text())

def implementation_hash():
    return digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in Path(__file__).parent.glob("*.py")})

def manifest(root):
    return dict(settings=SETTINGS, models=read(root/"models.json"),
                prepared=read(root/"prepared.json"),
                versions={name: importlib.metadata.version(name) for name in [
                    "torch", "transformers", "datasets", "scipy", "lm-eval", "evalplus", "math-verify"]},
                code_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                code_diff_hash=digest(subprocess.check_output(["git", "diff", "HEAD"], text=True)),
                implementation_hash=implementation_hash(),
                jobs=jobs())

def identity(root, job):
    frozen = read(root/"run-manifest.json")
    model = frozen["models"][job["model"]]
    return dict(job=job, protocol_hash=digest(frozen), model=model,
                evaluation_hash=frozen["prepared"]["evaluation_hash"],
                calibration_hash=None if job["method"]=="Teacher" else
                    read(root/f"calibration/{job['track']}/{job['model']}.json")["token_hash"])

def preflight(root, gpu_required=True, grading_required=True):
    import torch
    if read(root/"prepared.json")["protocol"] != SETTINGS:
        raise ValueError("Prepared protocol differs from code. Use a new run root.")
    frozen = read(root/"run-manifest.json")
    if frozen["implementation_hash"] != implementation_hash():
        raise ValueError("Implementation changed after prepare; use a new run root")
    for name, version in frozen["versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Runtime version mismatch: {name}")
    if gpu_required:
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise ValueError("Each worker must see exactly one CUDA GPU")
        if "H20" not in torch.cuda.get_device_name(0):
            raise ValueError("Formal Exp14 runs require H20 (use tests for CPU smoke)")
    commands = (["nvidia-smi"] + (["docker"] if grading_required else [])) if gpu_required else []
    for command in commands:
        if not shutil.which(command):
            raise ValueError(f"Missing runtime: {command}")
    if gpu_required and grading_required:
        image = os.environ.get("EXP14_GRADER_IMAGE", "moe-exp14-grader:1")
        subprocess.run(["docker", "image", "inspect", image], check=True, stdout=subprocess.DEVNULL)

def calibration(root, model, track):
    data = read(root/f"calibration/{track}/{model}.json")
    if len(data["ids"]) != 3072 or any(not row or len(row)>512 for row in data["ids"]):
        raise ValueError("Exp14 calibration must have 3072 nonempty, unpacked rows of <=512 tokens")
    if digest(data["ids"]) != data["token_hash"]:
        raise ValueError("Calibration tokens were modified after freezing")
    return data

def stats_job(root, model, track, device="cuda"):
    from .checkpoint import load
    from .compress import ensure_stats
    import torch
    preflight(root, gpu_required=device=="cuda", grading_required=False)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    data = calibration(root, model, track)
    cache = root/f"stats/{model}-{track}"
    cache_identity = dict(model=data["model"], revision=data["revision"],
                          tokens=data["token_hash"], observer=SETTINGS["stats_version"],
                          alignment_samples=SETTINGS["alignment_samples"])
    with lock(cache/"lock"):
        if (cache/"progress.json").exists():
            layers = 24 if model=="Q15" else 48
            if read(cache/"progress.json")["next_layer"] == layers:
                return
        net, _, _ = load(data["model"], data["revision"])
        ensure_stats(net, data["ids"], cache, cache_identity, device)

def run_job(root, job, device="cuda", defer_grading=False):
    import torch
    from .checkpoint import load
    from .compress import ensure_stats, compress, locked_identity
    from .evaluate import evaluate
    from .resources import Resources
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    preflight(root, gpu_required=device=="cuda", grading_required=not defer_grading and any(
        task in ["humaneval","mbpp","livecodebench"] for task in benchmarks(job)))
    output = root/"jobs"/job["id"]; output.mkdir(parents=True, exist_ok=True)
    checkpoint = output/"checkpoint"
    ident = identity(root, job)
    with lock(output/"lock"):
        locked_identity(output/"identity.json", ident)
        if (output/"status.json").exists() and read(output/"status.json")["state"]=="complete":
            return
        if (output/"evaluation/failure.json").exists():
            os.replace(output/"evaluation/failure.json", output/f"failure-{time.time_ns()}.json")
        atomic_json(output/"status.json", dict(state="running", pid=os.getpid(), started=time.time()))
        try:
            if job["section"]=="efficiency" and not (output/"compression-resources.json").exists():
                # A partial cold timing cannot be reconstructed from layer caches.
                # Preserve the attempt and restart this measurement from clean source.
                audit = output/("interrupted-"+str(time.time_ns()))
                candidates = [output/"cold-stats", output/"checkpoint.resume",
                              output/"checkpoint.incomplete", checkpoint]
                for path in candidates:
                    if path.exists():
                        audit.mkdir(exist_ok=True)
                        os.replace(path, audit/path.name)
            model = read(root/"models.json")[job["model"]]
            full_meta = None
            if job["depends"]:
                dependency = root/"jobs"/job["depends"]/"checkpoint/exp14.json"
                if not dependency.exists():
                    raise ValueError(f"Dependency compression not ready: {job['depends']}")
                full_meta = read(dependency)
            if job["method"]!="Teacher" and not checkpoint.exists():
                data = calibration(root, job["model"], job["track"])
                if job["section"]!="efficiency" and job["method"]!="REAM":
                    stats_job(root, job["model"], job["track"], device)
                with Resources() as resources:
                    with resources.stage("source_load"):
                        net, tokenizer, _ = load(model["id"], model["revision"])
                    expected = (24,60,4) if job["model"]=="Q15" else (48,128,8)
                    architecture = (len(net.model.layers), len(net.model.layers[0].mlp.experts),
                                    net.model.layers[0].mlp.top_k)
                    if architecture != expected:
                        raise ValueError(f"Source architecture mismatch: {architecture} != {expected}")
                    cache = root/f"stats/{job['model']}-{job['track']}"
                    if job["section"]=="efficiency" and job["method"]!="REAM":
                        cache = output/"cold-stats"
                        with resources.stage("calibration"):
                            ensure_stats(net, data["ids"], cache,
                                dict(tokens=data["token_hash"], source=model, observer=1), device,
                                method=job["method"])
                    with resources.stage("compression_and_save"):
                        compress(net, tokenizer, data["ids"], cache, checkpoint, job, ident,
                                 device=device, full_meta=full_meta)
                    metrics = resources.result()
                    metrics["checkpoint_bytes"] = sum(p.stat().st_size for p in checkpoint.rglob("*") if p.is_file())
                    metrics["cold"] = job["section"]=="efficiency"
                    metrics["cached_gpu_hours"] = (
                        (metrics["stage_seconds"]["source_load"]+
                         metrics["stage_seconds"]["compression_and_save"])/3600
                         if job["method"]!="REAM" else None)
                    atomic_json(output/"compression-resources.json", metrics)
                del net, tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            tasks = benchmarks(job)
            if tasks:
                with Resources() as resources:
                    with resources.stage("load_and_evaluate"):
                        net, tokenizer, _ = load(str(checkpoint) if job["method"]!="Teacher" else model["id"],
                                                 None if job["method"]!="Teacher" else model["revision"])
                        evaluate(net, tokenizer, root, output/"evaluation", tasks, ident, device,
                                 os.environ.get("EXP14_GRADER_IMAGE", "moe-exp14-grader:1"),
                                 defer_grading=defer_grading)
                    atomic_json(output/"evaluation-resources.json", resources.result())
            if (output/"evaluation/failure.json").exists():
                raise RuntimeError("Evaluation helper/grader failed; see evaluation/failure.json")
            atomic_json(output/"status.json", dict(
                state="waiting_evaluation" if defer_grading and tasks else "complete", finished=time.time()))
            if defer_grading:
                atomic_json(output/"gpu-phase-done.json", dict(identity_hash=digest(ident)))
        except BaseException as error:
            atomic_json(output/"status.json", dict(state="failed", error=str(error),
                         traceback=traceback.format_exc(), finished=time.time()))
            raise

def available_memory():
    if Path("/proc/meminfo").exists():
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])*1024
    return os.sysconf("SC_PHYS_PAGES")*os.sysconf("SC_PAGE_SIZE")

def process_rss(pid):
    status = Path(f"/proc/{pid}/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])*1024
    return 0

def launch(root, host, gpus, section=None, dry_run=False, steal=False):
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("GPU list must be nonempty and unique")
    pending = [job for job in jobs() if job["host"]==host and (not section or job["section"]==section)]
    if dry_run:
        print(json.dumps(pending, indent=2)); return
    # Fixed host assignment + job locks prevents duplicate work across restart.
    # Efficiency waits for this host's queue to drain and uses one card in isolation.
    with lock(root/f"host-{host}.lock", blocking=False):
        active = {}; failed = []
        logs = root/"logs"; logs.mkdir(exist_ok=True)
        estimate = {"Q3": 95, "Q15": 65, "F8": 60, "W4": 45} # GiB resident + PCA/activations
        while pending or active or (steal and any(
                j["section"]!="efficiency" and j["id"] not in failed and
                (not section or j["section"]==section) and
                (not (root/"jobs"/j["id"]/"status.json").exists() or
                 read(root/"jobs"/j["id"]/"status.json")["state"] not in ["complete","failed"]) for j in jobs())):
            for gpu, (process, stream, job, claim) in list(active.items()):
                if process.poll() is not None:
                    stream.close(); claim.close(); del active[gpu]
                    if process.returncode:
                        failed.append(job["id"])
            for gpu in gpus:
                if gpu in active:
                    continue
                chosen = None
                candidates = list(pending)
                if steal and not any(j["section"]=="efficiency" for j in pending):
                    candidates += [j for j in jobs() if j["host"]!=host and
                                   j["section"]!="efficiency" and j["id"] not in failed and
                                   (not section or j["section"]==section)]
                claim = None
                for job in candidates:
                    status = root/"jobs"/job["id"]/"status.json"
                    if status.exists() and read(status)["state"]=="complete":
                        if job in pending:
                            chosen=job; break
                        continue
                    if job not in pending and status.exists() and read(status)["state"]=="failed":
                        continue
                    if job["depends"] and not (root/"jobs"/job["depends"]/"checkpoint/exp14.json").exists():
                        if job["depends"] in failed:
                            raise RuntimeError(f"Dependency failed: {job['depends']}")
                        continue
                    if job["section"]=="efficiency":
                        if active or any(j["section"]!="efficiency" for j in pending):
                            continue
                        # Check every GPU and every CPU worker on this machine.
                        used = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                    "--format=csv,noheader,nounits"], text=True)
                        if any(int(line)>500 for line in used.splitlines()):
                            raise RuntimeError("Efficiency requires an idle machine; other GPU process detected")
                    reserved = sum(max(0, estimate[j["model"]]*1024**3-process_rss(p.pid))
                                   for p, _, j, _ in active.values())
                    if available_memory()-reserved < estimate[job["model"]]*1024**3:
                        continue
                    claim_path=root/"claims"/(job["id"]+".lock")
                    claim_path.parent.mkdir(parents=True, exist_ok=True)
                    candidate_claim=claim_path.open("a")
                    try:
                        fcntl.flock(candidate_claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        candidate_claim.close(); continue
                    claim=candidate_claim; chosen=job; break
                if chosen is None:
                    continue
                if chosen in pending:
                    pending.remove(chosen)
                status = root/"jobs"/chosen["id"]/"status.json"
                if status.exists() and read(status)["state"]=="complete":
                    if claim:
                        claim.close()
                    continue
                env=os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                           TOKENIZERS_PARALLELISM="false", WORLD_SIZE="1", RANK="0", LOCAL_RANK="0")
                for name in ["MASTER_ADDR", "MASTER_PORT", "RAY_ADDRESS"]:
                    env.pop(name, None)
                stream=(logs/f"{chosen['id']}.log").open("a")
                command=[sys.executable, "-m", "reap.exp14.run", "job", "--root", str(root), "--id", chosen["id"]]
                process=subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
                active[gpu]=(process, stream, chosen, claim)
                print(f"{host}: GPU {gpu} -> {chosen['id']}", flush=True)
                # Account for not-yet-resident new workers, before /proc catches up.
                if len(active)>1:
                    time.sleep(1)
            if pending and not active and not steal:
                unmet=[j["id"] for j in pending if not j["depends"] or
                        (root/"jobs"/j["depends"]/"checkpoint/exp14.json").exists()]
                if unmet:
                    raise RuntimeError("No job fits available host RAM; free RAM or use a larger host")
                raise RuntimeError("Unmet external dependencies; run their assigned host first")
            if active or steal:
                time.sleep(2)
        if failed:
            raise RuntimeError("Jobs failed (resume with same launch command): "+", ".join(failed))

def aggregate(root):
    rows=[]
    for job in jobs():
        path=root/"jobs"/job["id"]; scores={}
        for task in benchmarks(job):
            file=path/f"evaluation/{task}.json"
            if file.exists():
                data=read(file); scores[task]=data.get("mc_average",data.get("pass@1",data.get("accuracy")))
        resources=read(path/"compression-resources.json") if (path/"compression-resources.json").exists() else {}
        status=read(path/"status.json") if (path/"status.json").exists() else {"state":"not_started"}
        primary=read(path/"evaluation-resources.json") if (path/"evaluation-resources.json").exists() else {}
        helpers=[read(p) for p in (path/"evaluation").glob("helper-resources-*.json")]
        evaluation_resources=dict(primary=primary,helpers=helpers,
            gpu_hours=primary.get("gpu_hours",0)+sum(h["gpu_hours"] for h in helpers))
        rows.append({**job,"state":status["state"],"scores":scores,"resources":resources,
                     "evaluation_resources":evaluation_resources})
    atomic_json(root/"summary.json", rows)
    with (root/"summary.csv").open("w", newline="") as stream:
        columns=["id","section","model","track","k","method","protected","variant","state","mc8",*["humaneval","mbpp","livecodebench","gsm8k","math500"],"gpu_hours","checkpoint_bytes"]
        writer=csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
        for row in rows:
            writer.writerow({**{k:row.get(k) for k in columns if k in row},
                             **row["scores"], **{k:row["resources"].get(k) for k in ["gpu_hours","checkpoint_bytes"]}})

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("command", choices=["plan","prepare","prefetch","stats","job","launch","aggregate","eval-helper","grade-task"])
    parser.add_argument("--root", type=Path, default=Path("artifacts/exp14"))
    parser.add_argument("--id"); parser.add_argument("--host", choices=list("ABCD"))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--model", choices=list(MODELS)); parser.add_argument("--track", choices=["G","X"])
    parser.add_argument("--section", choices=["main","ablation","quant","reference","efficiency"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--steal", action="store_true", help="Shared-root tail backfill; prefetch all models first")
    parser.add_argument("--adaptive-eval", action="store_true", help="Share generation chunks and run CPU graders separately")
    parser.add_argument("--defer-grading", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--task", choices=["humaneval","mbpp","livecodebench","gsm8k","math500"])
    args=parser.parse_args(); root=args.root.resolve()
    if args.command=="plan":
        print(json.dumps({"settings":SETTINGS,"jobs":jobs()},indent=2)); return
    if args.command=="prepare":
        from .data import prepare
        with lock(root/"prepare.lock"):
            prepare(root)
            current=manifest(root)
            if (root/"run-manifest.json").exists() and read(root/"run-manifest.json")!=current:
                raise ValueError("Existing run identity differs; use a new root")
            atomic_json(root/"run-manifest.json",current)
    elif args.command=="prefetch":
        from .checkpoint import source_path
        selected = {args.model} if args.model else {
            j["model"] for j in jobs() if not args.host or j["host"]==args.host}
        models = read(root/"models.json")
        for model in sorted(selected):
            path = source_path(models[model]["id"], models[model]["revision"])
            print(f"{model}: {path}", flush=True)
    elif args.command=="stats":
        if not args.model or not args.track:
            parser.error("stats needs --model and --track")
        stats_job(root,args.model,args.track)
    elif args.command in ["job","eval-helper","grade-task"]:
        job=next((j for j in jobs() if j["id"]==args.id),None)
        if not job:
            parser.error("Unknown --id; see plan")
        if args.command=="job":
            run_job(root,job,defer_grading=args.defer_grading)
        else:
            if args.task not in benchmarks(job):
                parser.error("Task must belong to the selected job")
            from .adaptive import evaluation_worker
            evaluation_worker(root,job,args.task,args.command=="grade-task")
    elif args.command=="launch":
        if not args.host:
            parser.error("launch needs --host")
        if args.adaptive_eval and not args.dry_run:
            from .adaptive import launch_adaptive
            launch_adaptive(root,args.host,args.gpus.split(","),args.section,args.steal)
        else:
            launch(root,args.host,args.gpus.split(","),args.section,args.dry_run,args.steal)
    else:
        aggregate(root)
if __name__=="__main__":
    main()
