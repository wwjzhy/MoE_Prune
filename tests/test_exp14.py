"""CPU contract tests: no GPU, downloads or benchmark execution required."""
import copy
import json
from pathlib import Path
import pytest
import torch
from torch import nn
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from reap.exp14.protocol import jobs, validate_groups, benchmarks, digest
from reap.exp14.methods import CompactMoE, balanced_groups, groups_for, fuse, weight_features
from reap.exp14.quant import QuantLinear, pack4, unpack4, encode_fp8
from reap.exp14.checkpoint import load, save
from reap.exp14.compress import Statistics, initial_states, replay, ensure_stats, compress
from reap.exp14.data import Decontaminator
from reap.exp14.evaluate import gsm_correct, math_correct
torch.set_num_threads(1)

def model(kind="q3", quant=None):
    torch.manual_seed(42)
    config = dict(vocab_size=32, hidden_size=16, intermediate_size=32,
                  moe_intermediate_size=16, shared_expert_intermediate_size=16,
                  num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                  head_dim=8, num_experts=8, num_experts_per_tok=2,
                  max_position_embeddings=64, norm_topk_prob=True,
                  attn_implementation="eager")
    net = (Qwen3MoeForCausalLM(Qwen3MoeConfig(**config)) if kind=="q3" else
           Qwen2MoeForCausalLM(Qwen2MoeConfig(**config))).eval()
    rotary = copy.deepcopy(net.model.rotary_emb)
    net.to(torch.bfloat16)
    net.model.rotary_emb = rotary  # HF regenerates nonpersistent RoPE in FP32.
    if quant:
        for name, linear in list(net.named_modules()):
            if isinstance(linear, nn.Linear) and (".experts." in name or name.endswith(".mlp.gate")):
                q = QuantLinear(linear.out_features, linear.in_features, quant).set_dense(linear.weight)
                parent, leaf = name.rsplit(".",1)
                setattr(net.get_submodule(parent), leaf, q)
        if quant["kind"]=="fp8":
            net.config.quantization_config = dict(quant_method="fp8", fmt="e4m3", activation_scheme="dynamic",
                                                  weight_block_size=quant["block"])
        else:
            net.config.quantization_config = dict(quant_method="compressed-tensors", format="pack-quantized",
                config_groups={"g":dict(weights=dict(num_bits=4, group_size=8, symmetric=True, strategy="group"),
                                        input_activations=None)})
    tok = Tokenizer(WordLevel({str(i):i for i in range(32)}, unk_token="0"))
    tok.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="0", eos_token="1", pad_token="0")
    return net, tokenizer

def fake_stats(e=8, d=16):
    torch.manual_seed(42)
    gate = torch.rand(12,e)
    return dict(saliency=torch.arange(1,e+1).float(), ream_saliency=torch.arange(1,e+1).float(),
                frequency=torch.arange(1,e+1).float(), gate_gram=gate.T@gate, prob_gram=gate.T@gate,
                output_mean=torch.rand(e,d), gated_mean=torch.rand(e,d), activations=torch.rand(e,12,d))

def job(method="SPRM", variant="full", p=3):
    return dict(id="toy-"+method+"-"+variant, method=method, k=4, protected=p, variant=variant, seed=42)

def equal_payload(a,b):
    for key in a:
        x,y=a[key],b[key]
        if x.dtype==torch.float8_e4m3fn:
            assert torch.equal(x.view(torch.int8),y.view(torch.int8))
        else:
            assert torch.equal(x,y)

def test_matrix_and_budgets():
    matrix=jobs()
    assert len(matrix)==46 and len({j["id"] for j in matrix})==46
    assert sum(j["section"]=="ablation" for j in matrix)==10
    assert not any(j["protected"]==15 for j in matrix)
    assert len(benchmarks(next(j for j in matrix if j["track"]=="GX")))==6
    for e,k,p in [(128,42,34),(128,32,26),(60,15,12),(60,15,0),(60,15,3),(60,15,6),(60,15,9)]:
        s=fake_stats(e)
        j=dict(job(),k=k,protected=p)
        groups,*_=groups_for(j,s)
        validate_groups(groups,e,k)
        sizes=[len(g) for g in groups[p:]]
        assert max(sizes)-min(sizes)<=1 and all(len(g)==1 for g in groups[:p])

def test_routing_probability_mass_and_hc_mapping():
    net,_=model()
    source=net.model.layers[0].mlp
    groups=[[0,1,2],[3],[4,5],[6,7]]
    compact=CompactMoE(source,groups,"grouped")
    x=torch.randn(7,16,dtype=torch.bfloat16)
    logits,weights,selected=compact.routing(x)
    original=source.gate(x).float().softmax(-1)
    expected=torch.stack([original[:,g].sum(-1) for g in groups],-1)
    assert torch.allclose(logits.softmax(-1),expected,atol=1e-6)
    logical=CompactMoE(source,groups,"logical")
    _,weight,selected=logical.routing(x)
    assert torch.allclose(weight.sum(-1),torch.ones(len(x)))
    assert selected.max()<4
    # Logical-slot aggregation executes each physical FFN once, with summed mass.
    output,_=logical(x[None])
    original_logits=source.gate(x).float()
    probs,indices=original_logits.softmax(-1).topk(2,-1)
    probs/=probs.sum(-1,keepdim=True)
    manual=torch.zeros_like(x)
    mapping={old:g for g,ids in enumerate(groups) for old in ids}
    for t in range(len(x)):
        physical={}
        for r in range(2):
            g=mapping[int(indices[t,r])]
            physical[g]=physical.get(g,0)+probs[t,r]
        for g,w in physical.items():
            manual[t]+=logical.experts[g](x[t:t+1])[0]*w.to(x.dtype)
    assert torch.allclose(output[0].float(),manual.float(),atol=.002)

def test_neuron_alignment_and_multiple_fusion():
    net,_=model(); a=net.model.layers[0].mlp.experts[0].float()
    b=copy.deepcopy(a); perm=torch.randperm(16)
    with torch.no_grad():
        b.gate_proj.weight.copy_(a.gate_proj.weight[perm])
        b.up_proj.weight.copy_(a.up_proj.weight[perm])
        b.down_proj.weight.copy_(a.down_proj.weight[:,perm])
    x=torch.randn(32,16)
    assert torch.allclose(a(x),b(x),atol=1e-6)
    c=copy.deepcopy(b); experts=[a,b,c]
    acts=torch.stack([e.act_fn(e.gate_proj(x))*e.up_proj(x) for e in experts]).detach()
    features=weight_features(experts,rank=12)
    merged=fuse(experts,[0,1,2],torch.ones(3),acts,features)
    assert torch.allclose(a(x),merged(x),atol=1e-6)
    assert merged is not a
    equal_payload(a.state_dict(),experts[0].state_dict())

@pytest.mark.parametrize("kind",["q3","q2"])
@pytest.mark.parametrize("method",["REAP","HC","REAM","SPRM"])
def test_checkpoint_and_protection(tmp_path,kind,method):
    net,tok=model(kind); stats=fake_stats(); source=net.model.layers[0].mlp
    groups,mode,align,scores=groups_for(job(method),stats)
    before=copy.deepcopy(source.experts[groups[0][0]].state_dict())
    features=weight_features(source.experts,rank=8) if align else None
    experts=[fuse(source.experts,g,scores,stats["activations"],features,align) for g in groups]
    net.model.layers[0].mlp=CompactMoE(source,groups,mode,experts)
    if method in ["REAP","SPRM"]:
        equal_payload(before,experts[0].state_dict())
    if kind=="q2":
        assert net.model.layers[0].mlp.shared_expert is source.shared_expert
    ids=torch.tensor([[2,3,4]])
    with torch.no_grad(): expected=net(ids).logits
    path=tmp_path/"model"
    save(net,tok,path,dict(layers={"0":dict(groups=groups,mode=mode)}))
    restored,_,_=load(str(path))
    assert len(restored.model.layers[0].mlp.experts)==4
    assert torch.allclose(expected,restored(ids).logits,atol=.0001)
    assert restored.generate(ids,max_new_tokens=2,do_sample=False).shape==(1,5)

@pytest.mark.parametrize("spec",[dict(kind="w4",group_size=8),dict(kind="fp8",block=[8,8],dynamic_activation=True)])
@pytest.mark.parametrize("mode",["grouped","centroid","logical"])
def test_quantized_checkpoint(tmp_path,spec,mode):
    net,tok=model(quant=spec); source=net.model.layers[0].mlp
    groups=[[0],[1],[2],[3,4,5,6,7]]
    protected=copy.deepcopy(source.experts[0].state_dict())
    features=weight_features(source.experts,rank=8)
    stats=fake_stats()
    merged=fuse(source.experts,groups[-1],stats["saliency"],stats["activations"],features)
    compact=CompactMoE(source,groups,mode,[source.experts[g[0]] for g in groups[:-1]]+[merged])
    net.model.layers[0].mlp=compact
    ids=torch.tensor([[2,3,4]])
    with torch.no_grad(): expected=net(ids).logits
    save(net,tok,tmp_path/"model",dict(layers={"0":dict(groups=groups,mode=mode)}))
    restored,_,_=load(str(tmp_path/"model"))
    equal_payload(protected,restored.model.layers[0].mlp.experts[0].state_dict())
    assert isinstance(restored.model.layers[0].mlp.experts[-1].gate_proj,QuantLinear)
    assert torch.allclose(expected,restored(ids).logits,atol=.0001)

def test_quant_codes():
    q=torch.arange(-8,8,dtype=torch.int8).reshape(2,8)
    packed=pack4(q)
    assert packed[0,0].item()==0x76543210
    assert packed[1,0].item()==-19088744  # 0xfedcba98 signed int32
    assert torch.equal(unpack4(packed,8),q)
    weight=torch.randn(19,21)
    q,s=encode_fp8(weight,[8,8])
    dense=q.float()*s.repeat_interleave(8,0).repeat_interleave(8,1)[:19,:21]
    assert torch.mean((dense-weight).square())<.005

@pytest.mark.parametrize("method",["SPRM","REAM"])
def test_layerwise_stats_and_resume(tmp_path,method,monkeypatch):
    net,tok=model(); original=copy.deepcopy(net)
    samples=[[2,3,4,5],[4,5,6]]
    cache=tmp_path/"stats"
    ensure_stats(net,samples,cache,dict(tokens=digest(samples)),"cpu")
    assert json.loads((cache/"progress.json").read_text())["next_layer"]==2
    stats=torch.load(cache/"layer-000.pt",weights_only=True)
    assert stats["valid_tokens"]==7 and int(stats["frequency"].sum())==14
    import reap.exp14.compress as module
    save_fn=module.save
    def stop_before_save(*a,**kw):
        raise RuntimeError("simulated preemption")
    monkeypatch.setattr(module,"save",stop_before_save)
    with pytest.raises(RuntimeError,match="preemption"):
        compress(net,tok,samples,cache,tmp_path/"checkpoint",job(method),{"test":1},"cpu")
    monkeypatch.setattr(module,"save",save_fn)
    meta=compress(original,tok,samples,cache,tmp_path/"checkpoint",job(method),{"test":1},"cpu")
    loaded,_,_=load(str(tmp_path/"checkpoint"))
    ids=torch.tensor([[2,3,4]])
    assert torch.allclose(net(ids).logits,loaded(ids).logits,atol=.0001)
    assert len(meta["layers"])==2
    with pytest.raises(ValueError,match="identity"):
        ensure_stats(original,samples,cache,{"tokens":"wrong"},"cpu")

def test_decontamination_and_math():
    question="A sufficiently long mathematical problem has thirteen or more distinct words for an exact overlap detection rule"
    d=Decontaminator([question])
    assert d.matches("preface "+question+" solution")
    assert not d.matches("unrelated calibration text")
    assert gsm_correct("Reasoning 5 then answer is 1,234.", "1234")
    assert not gsm_correct("No answer", "0")
    assert math_correct(r"Therefore \boxed{\frac{1}{2}}", "0.5")

def test_calibration_freeze_and_mix(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from reap.exp14 import data
    def eval_freeze(root):
        path=Path(root)/"evaluation"; path.mkdir(parents=True)
        data.atomic_json(path/"questions.json",[])
        return {"hash":"fixed-eval"}
    class TokenizerStub:
        def __call__(self,text,**kwargs):
            return {"input_ids":[2+(len(text)%10)]*32}
    class SourceStub:
        def shuffle(self,**kwargs):
            assert kwargs=={"seed":42,"buffer_size":10000}
            return self
        def __iter__(self):
            for i in range(4000):
                yield dict(id=str(i),source="cn_k12",problem=f"problem {i}",
                           solution=f"solution {i}",text=f"c4 document {i}",content=f"code document {i}")
    monkeypatch.setattr(data,"freeze_evaluation",eval_freeze)
    monkeypatch.setattr(data,"dataset_revision",lambda name:"dataset-sha")
    monkeypatch.setattr(data,"load_dataset",lambda *a,**kw:SourceStub())
    monkeypatch.setattr(data,"HfApi",lambda:SimpleNamespace(model_info=lambda model:SimpleNamespace(sha="model-sha")))
    monkeypatch.setattr(data.AutoTokenizer,"from_pretrained",lambda *a,**kw:TokenizerStub())
    data.prepare(tmp_path)
    mixed=json.loads((tmp_path/"calibration/X/raw.json").read_text())
    assert len(mixed["rows"])==3072
    assert sum(r["source"]=="math" for r in mixed["rows"])==1536
    assert sum(r["source"]=="code" for r in mixed["rows"])==1536
    assert len(set(r["source"] for r in mixed["rows"][:20]))==2
    frozen=json.loads((tmp_path/"calibration/G/Q3.json").read_text())
    assert frozen["effective_tokens"]==3072*32
    assert frozen["token_hash"]==digest(frozen["ids"])
    assert frozen["source_manifest"]["c4"]["split"]=="train"

def test_harness_accepts_compact_model():
    from lm_eval.models.huggingface import HFLM
    from lm_eval.api.instance import Instance
    net,tok=model()
    source=net.model.layers[0].mlp
    net.model.layers[0].mlp=CompactMoE(source,[[0],[1],[2],[3,4,5,6,7]],"grouped")
    lm=HFLM(pretrained=net,tokenizer=tok,batch_size=1,max_length=64)
    request=Instance(request_type="loglikelihood",doc={},arguments=("2"," 3"),idx=0)
    score,_=lm.loglikelihood([request])[0]
    assert isinstance(score,float) and torch.isfinite(torch.tensor(score))

def test_calibration_identity_and_scheduler(tmp_path,capsys):
    from reap.exp14.run import calibration,launch,aggregate
    path=tmp_path/"calibration/G"; path.mkdir(parents=True)
    from reap.exp14.protocol import atomic_json
    rows=[[2,3]]*3072
    atomic_json(path/"Q3.json",dict(ids=rows,token_hash=digest(rows)))
    assert len(calibration(tmp_path,"Q3","G")["ids"])==3072
    launch(tmp_path,"C",["0","1"],dry_run=True)
    assert len(json.loads(capsys.readouterr().out))==10
    with pytest.raises(ValueError,match="unique"):
        launch(tmp_path,"A",["0","0"],dry_run=True)
    aggregate(tmp_path)
    assert len(json.loads((tmp_path/"summary.json").read_text()))==46

def test_checkpoint_tamper_detection(tmp_path):
    net,tok=model()
    save(net,tok,tmp_path/"model",{"layers":{}})
    config=tmp_path/"model/config.json"
    config.write_text(config.read_text()+" ")
    with pytest.raises(ValueError,match="checksum"):
        load(str(tmp_path/"model"))

def test_selective_cold_reap_matches_shared():
    net,_=model(); source=net.model.layers[0].mlp
    x=torch.randn(1,7,16,dtype=torch.bfloat16)
    all_stats=Statistics(source,1,"shared")
    prune_stats=Statistics(source,1,"REAP")
    all_stats(source,(x,),None); prune_stats(source,(x,),None)
    assert torch.allclose(all_stats.result()["saliency"],prune_stats.result()["saliency"],atol=.0001)
    assert prune_stats.result()["activations"].numel()==0

def test_mc8_frozen_dataset_adapter(tmp_path,monkeypatch):
    from datasets import Dataset
    from lm_eval.api.task import ConfigurableTask
    from reap.exp14.evaluate import mc8
    from reap.exp14.protocol import MC8
    import lm_eval.tasks
    for name in MC8:
        dataset=Dataset.from_dict({"question":["2 3"],"choices":[["4","5"]],"answer":[0]})
        dataset.save_to_disk(str(tmp_path/"evaluation/mc8-cache"/name/"test"))
    def get_tasks(names,manager):
        name=names[0]
        return {name:ConfigurableTask(config=dict(task=name,dataset_path="must-not-download",
            test_split="test",output_type="multiple_choice",doc_to_text="question",doc_to_choice="choices",
            doc_to_target="answer",num_fewshot=0,
            metric_list=[dict(metric="acc",aggregation="mean",higher_is_better=True),
                         dict(metric="acc_norm",aggregation="mean",higher_is_better=True)]))}
    monkeypatch.setattr(lm_eval.tasks,"get_task_dict",get_tasks)
    net,tok=model()
    output=tmp_path/"scores"; output.mkdir()
    result=mc8(net,tok,tmp_path,output)
    assert len(result["tasks"])==8 and 0<=result["mc_average"]<=1
    assert all(key=="acc_norm,none" for key in result["metric_keys"].values())
