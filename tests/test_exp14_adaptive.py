"""Concurrency/resume contracts; generated code is never executed by these tests."""
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import pytest
from reap.exp14.protocol import atomic_json, digest
from reap.exp14 import evaluate, adaptive

def frozen(root, n=70):
    rows=[dict(id=str(i),prompt=f"question {i}",answer=str(i)) for i in range(n)]
    atomic_json(root/"evaluation/gsm8k.json",rows)
    atomic_json(root/"evaluation/manifest.json",dict(datasets={"gsm8k":dict(hash=digest(rows))}))
    return rows

def fake_generator(calls):
    def generate(net,tok,row,task,device):
        calls.append(row["id"]); time.sleep(.001)
        return dict(id=row["id"],response=row["id"],input_tokens=2,new_tokens=1)
    return generate

def test_chunks_concurrent_no_duplicates(tmp_path,monkeypatch):
    rows=frozen(tmp_path); calls=[]
    monkeypatch.setattr(evaluate,"generate",fake_generator(calls))
    output=tmp_path/"samples"
    with ThreadPoolExecutor(max_workers=3) as pool:
        counts=list(pool.map(lambda _:evaluate.generate_task(None,None,tmp_path,output,"gsm8k",{"run":1},"cpu"),range(3)))
    assert sum(counts)==len(rows)
    assert Counter(calls)==Counter(r["id"] for r in rows)
    assert evaluate.generation_ready(tmp_path,output,"gsm8k")
    assert not (output/"gsm8k.json").exists()  # GPU phase never waits for a grader.
    evaluate.grade_task(tmp_path,output,"gsm8k",{"run":1})
    result=json.loads((output/"gsm8k.json").read_text())
    assert result["accuracy"]==1 and result["count"]==70

def test_crash_partial_chunk_resumes(tmp_path,monkeypatch):
    frozen(tmp_path,40); calls=[]
    good=fake_generator(calls)
    def interrupted(*args):
        if len(calls)==3:
            raise RuntimeError("preemption")
        return good(*args)
    monkeypatch.setattr(evaluate,"generate",interrupted)
    output=tmp_path/"samples"
    with pytest.raises(RuntimeError,match="preemption"):
        evaluate.generate_task(None,None,tmp_path,output,"gsm8k",{"run":2},"cpu")
    assert not evaluate.generation_ready(tmp_path,output,"gsm8k")
    with pytest.raises(ValueError,match="incomplete"):
        evaluate.grade_task(tmp_path,output,"gsm8k",{"run":2})
    monkeypatch.setattr(evaluate,"generate",good)
    evaluate.generate_task(None,None,tmp_path,output,"gsm8k",{"run":2},"cpu")
    assert len(calls)==40 and len(set(calls))==40
    evaluate.grade_task(tmp_path,output,"gsm8k",{"run":2})
    with pytest.raises(ValueError,match="identity"):
        evaluate.grade_task(tmp_path,output,"gsm8k",{"run":"wrong"})

def test_global_helper_cap_and_release(tmp_path):
    slots=[adaptive.claim(tmp_path/f"slot-{i}") for i in range(adaptive.MAX_HELPERS)]
    assert all(slots) and adaptive.claim(tmp_path/"slot-0") is None
    slots[0].close()
    replacement=adaptive.claim(tmp_path/"slot-0")
    assert replacement
    replacement.close(); slots[1].close()

def test_ready_requires_gpu_exit_and_every_score(tmp_path,monkeypatch):
    j=dict(id="toy",section="main",track="X",host="A",model="Q15")
    monkeypatch.setattr(adaptive,"identity",lambda *a:{"run":3})
    monkeypatch.setattr(adaptive,"benchmarks",lambda j:["gsm8k","math500"])
    output=tmp_path/"jobs/toy"
    atomic_json(output/"status.json",{"state":"waiting_evaluation"})
    atomic_json(output/"evaluation/gsm8k.json",{"identity_hash":digest({"run":3})})
    adaptive.finish_ready(tmp_path,j)
    assert adaptive.state(tmp_path,j)=="waiting_evaluation"
    atomic_json(output/"gpu-phase-done.json",{"identity_hash":digest({"run":3})})
    adaptive.finish_ready(tmp_path,j)
    assert adaptive.state(tmp_path,j)=="waiting_evaluation"
    atomic_json(output/"evaluation/math500.json",{"identity_hash":digest({"run":3})})
    primary_claim=adaptive.claim(tmp_path/"claims/toy.lock")
    adaptive.finish_ready(tmp_path,j)
    assert adaptive.state(tmp_path,j)=="waiting_evaluation"
    primary_claim.close()
    adaptive.finish_ready(tmp_path,j)
    assert adaptive.state(tmp_path,j)=="complete"

def test_eta_and_claimed_chunk_skip(tmp_path,monkeypatch):
    frozen(tmp_path,70)
    j=dict(id="toy",section="main",track="X",host="A",model="Q15")
    monkeypatch.setattr(adaptive,"benchmarks",lambda j:["gsm8k"])
    output=tmp_path/"jobs/toy/evaluation"
    atomic_json(output.parent/"status.json",{"state":"running"})
    atomic_json(output/"identity.json",{"run":4})
    locked=[adaptive.claim(base.with_suffix(".lock")) for _,_,base in evaluate.chunks(tmp_path,output,"gsm8k")]
    assert adaptive.tail_candidates(tmp_path,[j])==[]
    locked[0].close()
    assert adaptive.tail_candidates(tmp_path,[j])[0][0]==70
    for stream in locked[1:]:
        stream.close()

@pytest.mark.parametrize("initial",["not_started","running","failed"])
def test_gpu_slot_reused_while_cpu_grades(tmp_path,monkeypatch,initial):
    frozen(tmp_path,1)
    matrix=[dict(id=f"toy-{i}",section="main",track="X",host="A",model="Q15",depends=None) for i in range(2)]
    calls=[]; events=[]; live_graders=[]
    monkeypatch.setattr(evaluate,"generate",fake_generator(calls))
    monkeypatch.setattr(adaptive,"jobs",lambda:matrix)
    monkeypatch.setattr(adaptive,"benchmarks",lambda j:["gsm8k"])
    monkeypatch.setattr(adaptive,"identity",lambda root,j:{"id":j["id"]})
    monkeypatch.setattr(adaptive,"available_memory",lambda:10**13)
    monkeypatch.setattr(adaptive,"process_rss",lambda pid:0)
    monkeypatch.setattr(adaptive.time,"sleep",lambda t:None)
    if initial!="not_started":
        atomic_json(tmp_path/"jobs/toy-0/status.json",{"state":initial,"pid":999999})
    if initial=="failed":
        atomic_json(tmp_path/"jobs/toy-0/evaluation/failure.json",{"error":"previous interrupted grader"})
    class Process:
        def __init__(self,args,env,**kw):
            self.returncode=0; self.pid=999999; self.ticks=0
            kind=args[3]; j=next(j for j in matrix if j["id"]==args[args.index("--id")+1])
            output=tmp_path/"jobs"/j["id"]
            if kind=="job":
                failure=output/"evaluation/failure.json"
                if failure.exists():
                    failure.rename(output/"previous-failure.json")
                events.append(("job",j["id"],any(p.ticks<3 for p in live_graders)))
                evaluate.generate_task(None,None,tmp_path,output/"evaluation","gsm8k",{"id":j["id"]},"cpu")
                atomic_json(output/"gpu-phase-done.json",{"identity_hash":digest({"id":j["id"]})})
                atomic_json(output/"status.json",{"state":"waiting_evaluation"})
            else:
                assert kind=="grade-task" and env["CUDA_VISIBLE_DEVICES"]==""
                live_graders.append(self)
                evaluate.grade_task(tmp_path,output/"evaluation","gsm8k",{"id":j["id"]})
        def poll(self):
            self.ticks+=1
            return 0 if self not in live_graders or self.ticks>=3 else None
    monkeypatch.setattr(adaptive.subprocess,"Popen",Process)
    adaptive.launch_adaptive(tmp_path,"A",["0"])
    assert events==[("job","toy-0",False),("job","toy-1",True)]
    assert all(adaptive.state(tmp_path,j)=="complete" for j in matrix)
