"""Bounded GPU tail helpers + independent CPU grading; shared POSIX locks."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from .protocol import jobs, benchmarks, atomic_json, digest
from .run import lock, read, identity, preflight, available_memory, process_rss
from .evaluate import pending_chunks, generation_ready, generate_task, grade_task

MAX_HELPERS = 2  # Additional GPUs per checkpoint, across all four nodes.
MAX_GRADERS = 2  # Per node: 8 CPUs / 16 GiB maximum Docker grading reservation.
RAM_GIB = {"Q3":95, "Q15":65, "F8":60, "W4":45}

def claim(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close(); return None
    return stream

def state(root, job):
    path = root/"jobs"/job["id"]/"status.json"
    return read(path)["state"] if path.exists() else "not_started"

def finish_ready(root, job):
    output = root/"jobs"/job["id"]
    if state(root, job) != "waiting_evaluation":
        return
    if (output/"evaluation/failure.json").exists():
        atomic_json(output/"status.json", dict(state="failed", error="Evaluation worker failed",
                                               finished=time.time()))
        return
    if not (output/"gpu-phase-done.json").exists():
        return
    ownership=claim(root/"claims"/(job["id"]+".lock"))
    if not ownership:
        return  # Primary still owns the GPU phase, possibly on another host.
    ownership.close()
    ident = identity(root, job)
    if read(output/"gpu-phase-done.json")["identity_hash"] != digest(ident):
        raise ValueError("GPU phase identity mismatch")
    for task in benchmarks(job):
        result = output/f"evaluation/{task}.json"
        if not result.exists():
            return
        if read(result)["identity_hash"] != digest(ident):
            raise ValueError("Evaluation result identity mismatch")
    atomic_json(output/"status.json", dict(state="complete", finished=time.time()))

def tail_candidates(root, selected):
    """Largest estimated remaining generation first, using completed-chunk timing."""
    candidates = []
    for job in selected:
        if state(root, job) not in ["running", "waiting_evaluation"]:
            continue
        output = root/"jobs"/job["id"]/"evaluation"
        if not (output/"identity.json").exists() or (output/"failure.json").exists():
            continue
        for task in benchmarks(job):
            if task=="mc8" or (output/f"{task}.json").exists():
                continue
            todo = pending_chunks(root, output, task)
            if not todo:
                continue
            # Avoid launching a model replica when every remaining chunk is already claimed.
            available = False
            for _, _, base in todo:
                stream = claim(base.with_suffix(".lock"))
                if stream:
                    stream.close(); available = True; break
            if not available:
                continue
            done = [read(p) for p in (output/f"{task}-chunks").glob("*.done.json")]
            count = sum(d["count"] for d in done)
            seconds_per_sample = sum(d["seconds"] for d in done)/count if count else 1.0
            remaining = sum(stop-start for start,stop,_ in todo)
            candidates.append((remaining*max(.001, seconds_per_sample), job, task))
    return sorted(candidates, key=lambda item: (-item[0], item[1]["id"], item[2]))

def evaluation_worker(root, job, task, grading=False):
    import torch
    torch.set_num_threads(4)
    preflight(root, gpu_required=not grading, grading_required=False)
    output = root/"jobs"/job["id"]/"evaluation"
    ident = identity(root, job)
    try:
        if grading:
            image = os.environ.get("EXP14_GRADER_IMAGE", "moe-exp14-grader:1")
            if task not in ["gsm8k","math500"]:
                subprocess.run(["docker","image","inspect",image], check=True, stdout=subprocess.DEVNULL)
            grade_task(root, output, task, ident, image)
        else:
            from .checkpoint import load
            from .resources import Resources
            from .methods import CompactMoE
            source = read(root/"models.json")[job["model"]]
            with Resources() as resources:
                with resources.stage("helper_load_and_generate"):
                    net, tok, _ = load(str(output.parent/"checkpoint") if job["method"]!="Teacher" else source["id"],
                                       None if job["method"]!="Teacher" else source["revision"])
                    net.to("cuda").eval()
                    generated = generate_task(net, tok, root, output, task, ident, "cuda")
                metrics = resources.result()
                metrics.update(task=task, generated=generated, identity_hash=digest(ident),
                    routing_calls={str(i): layer.mlp.calls for i,layer in enumerate(net.model.layers)
                                   if isinstance(layer.mlp,CompactMoE)})
                atomic_json(output/f"helper-resources-{os.getpid()}-{time.time_ns()}.json", metrics)
    except BaseException as error:
        atomic_json(output/"failure.json", dict(task=task, grading=grading, error=str(error),
                                               traceback=traceback.format_exc()))
        raise

def launch_adaptive(root, host, gpus, section=None, steal=False):
    if not gpus or len(gpus)!=len(set(gpus)):
        raise ValueError("GPU list must be nonempty and unique")
    own = [j for j in jobs() if j["host"]==host and (not section or j["section"]==section)]
    selected = own + ([j for j in jobs() if j["host"]!=host and j["section"]!="efficiency"
                       and (not section or j["section"]==section)] if steal else [])
    logs = root/"logs"; logs.mkdir(parents=True, exist_ok=True)
    atomic_json(root/f"scheduler-{host}.json", dict(adaptive=True, chunk_size=32,
        max_helpers_per_checkpoint=MAX_HELPERS, max_graders=MAX_GRADERS, gpus=gpus, steal=steal))
    with lock(root/f"host-{host}.lock", blocking=False):
        active, graders, failed = {}, {}, set()
        def spawn(job, command, task, gpu, slot):
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES="" if gpu is None else gpu, OMP_NUM_THREADS="4",
                       MKL_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false",
                       WORLD_SIZE="1", RANK="0", LOCAL_RANK="0")
            for name in ["MASTER_ADDR","MASTER_PORT","RAY_ADDRESS"]:
                env.pop(name, None)
            stream = (logs/f"{job['id']}-{command}-{task or 'all'}-{time.time_ns()}.log").open("a")
            args = [sys.executable,"-m","reap.exp14.run",command,"--root",str(root),"--id",job["id"]]
            args += ["--task",task] if task else ["--defer-grading"]
            # Child retains the claim if a launcher is killed; duplicate launchers cannot steal it.
            process = subprocess.Popen(args,env=env,stdout=stream,stderr=subprocess.STDOUT,
                                       pass_fds=(slot.fileno(),))
            print(f"{host}: {'CPU' if gpu is None else 'GPU '+gpu} -> {job['id']} / {command} / {task or 'all'}",
                  flush=True)
            return (process,stream,job,slot,command)
        while True:
            for workers in [active, graders]:
                for key,(process,stream,job,slot,kind) in list(workers.items()):
                    if process.poll() is not None:
                        stream.close(); slot.close(); del workers[key]
                        if process.returncode:
                            failed.add(job["id"])
                            atomic_json(root/"jobs"/job["id"]/"status.json",
                                        dict(state="failed", error=f"{kind} process exited {process.returncode}"))
            for job in selected:
                finish_ready(root, job)
            # Freshly failed OWN jobs may be retried once on a new launch, not in this loop.
            unfinished = [j for j in selected if j["id"] not in failed and state(root,j)!="complete"
                          and (j in own or state(root,j)!="failed")]
            if not unfinished and not active and not graders:
                break
            for job in selected:
                if job["id"] in failed or state(root,job)=="failed" and job not in own:
                    continue
                if (root/"jobs"/job["id"]/"evaluation/failure.json").exists() and state(root,job)!="failed":
                    failed.add(job["id"])
            efficiency_waits = any(j["section"]=="efficiency" and state(root,j)!="complete"
                                   and j["id"] not in failed for j in own)
            drain = efficiency_waits and all(state(root,j) in ["complete","failed"] or j["id"] in failed
                                            for j in own if j["section"]!="efficiency")
            isolated = any(j["section"]=="efficiency" for _,_,j,_,_ in active.values())
            # CPU grading does not occupy a GPU slot. No CPU queue runs during cold efficiency.
            if not isolated and not drain:
                for job in selected:
                    if len(graders)>=MAX_GRADERS:
                        break
                    output = root/"jobs"/job["id"]/"evaluation"
                    if state(root,job) not in ["running","waiting_evaluation"] or (output/"failure.json").exists():
                        continue
                    for task in benchmarks(job):
                        if task=="mc8" or (output/f"{task}.json").exists() or not generation_ready(root,output,task):
                            continue
                        if len(graders)>=MAX_GRADERS:
                            break
                        slot=claim(output/f"{task}-grade-claim.lock")
                        if slot:
                            token=(job["id"],task)
                            graders[token]=spawn(job,"grade-task",task,None,slot)
            for gpu in gpus:
                if gpu in active or isolated:
                    continue
                chosen = None
                for job in unfinished:
                    if state(root,job)=="waiting_evaluation":
                        continue
                    if state(root,job)=="running":
                        probe=claim(root/"jobs"/job["id"]/"lock")
                        if not probe:
                            continue
                        probe.close()  # Stale running status: recover only if both ownership locks are free.
                    if job["id"] in failed:
                        continue
                    if drain and job["section"]!="efficiency":
                        continue
                    if job["depends"] and not (root/"jobs"/job["depends"]/"checkpoint/exp14.json").exists():
                        if state(root,next(j for j in jobs() if j["id"]==job["depends"]))=="failed":
                            failed.add(job["id"])
                        continue
                    if job["section"]=="efficiency":
                        if active or graders or not drain:
                            continue
                        used=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used",
                                                      "--format=csv,noheader,nounits"],text=True)
                        if any(int(line)>500 for line in used.splitlines()):
                            raise RuntimeError("Efficiency requires an idle machine")
                    reserved=sum(max(0,RAM_GIB[j["model"]]*1024**3-process_rss(p.pid))
                                 for p,_,j,_,_ in active.values())
                    if available_memory()-reserved < RAM_GIB[job["model"]]*1024**3+16*1024**3:
                        continue
                    slot=claim(root/"claims"/(job["id"]+".lock"))
                    if slot:
                        chosen=(job,slot); break
                if chosen:
                    job,slot=chosen
                    active[gpu]=spawn(job,"job",None,gpu,slot)
                    if job["section"]=="efficiency":
                        isolated=True
                    continue
                if drain:
                    continue
                # Fresh configurations have priority; idle slots assist slow code/math generation.
                for _,job,task in tail_candidates(root, selected):
                    if job["id"] in failed:
                        continue
                    reserved=sum(max(0,RAM_GIB[j["model"]]*1024**3-process_rss(p.pid))
                                 for p,_,j,_,_ in active.values())
                    if available_memory()-reserved < RAM_GIB[job["model"]]*1024**3+16*1024**3:
                        continue
                    slot=next((s for n in range(MAX_HELPERS)
                               if (s:=claim(root/"jobs"/job["id"]/f"helper-slot-{n}.lock")) is not None),None)
                    if slot:
                        active[gpu]=spawn(job,"eval-helper",task,gpu,slot); break
            # No active work, but external queues may still own dependency/claim locks.
            if not active and not graders and not steal and unfinished:
                ready_jobs=[j for j in unfinished if state(root,j) in ["not_started","failed"] and
                            (not j["depends"] or (root/"jobs"/j["depends"]/"checkpoint/exp14.json").exists())]
                if ready_jobs and available_memory() < min(
                        RAM_GIB[j["model"]]*1024**3+16*1024**3 for j in ready_jobs):
                    raise RuntimeError("No ready job fits host RAM; preserve the memory guard")
                tails=tail_candidates(root, [j for j in unfinished if j["id"] not in failed])
                if tails and available_memory() < min(RAM_GIB[j["model"]]*1024**3+16*1024**3
                                                      for _,j,_ in tails):
                    raise RuntimeError("No evaluation replica fits host RAM; preserve the memory guard")
            time.sleep(2)
        if failed:
            raise RuntimeError("Failed jobs, inspect logs then resume: "+", ".join(sorted(failed)))
