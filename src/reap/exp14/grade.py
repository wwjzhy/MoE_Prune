"""Container-only official graders (never run generated code in the host worker)."""
import json
import sys
from pathlib import Path

def main(path):
    path = Path(path); data = json.loads(path.read_text())
    task, rows, samples = data["task"], data["rows"], data["generated"]
    if len(rows) != len(samples) or [r["id"] for r in rows] != [s["id"] for s in samples]:
        raise ValueError("Grader input alignment mismatch")
    if task == "livecodebench":
        from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
        from lcb_runner.utils.extraction_utils import extract_code
        from lcb_runner.lm_styles import LMStyle
        solutions = [[extract_code(s["response"], LMStyle.OpenAIChat)] for s in samples]
        metrics, tests, metadata = codegen_metrics([r["tests"] for r in rows], solutions,
                k_list=[1], num_process_evaluate=4, timeout=data["timeout"])
        result = {"pass@1": metrics["pass@1"], "count": len(rows), "details": tests,
                  "metadata": metadata, "grader": "LiveCodeBench"}
    else:
        from evalplus.sanitize import sanitize
        from evalplus.evaluate import evaluate
        from evalplus.data.utils import CACHE_DIR
        Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
        solutions = [{"task_id": r["id"], "solution": sanitize(s["response"], entrypoint=r["entry_point"])}
                     for r,s in zip(rows, samples)]
        sample_path = path.parent/"solutions.jsonl"
        sample_path.write_text("".join(json.dumps(s)+"\n" for s in solutions))
        raw_path = path.parent/"solutions_eval_results.json"
        evaluate(dataset=task, samples=str(sample_path), base_only=False, parallel=4,
                 min_time_limit=data["timeout"])
        raw = json.loads(raw_path.read_text())
        if len(raw["eval"]) != len(rows):
            raise ValueError("EvalPlus omitted tasks")
        statuses = list(raw["eval"].values())
        if any(len(items) != 1 for items in statuses):
            raise ValueError("Exp14 requires exactly one completion per problem")
        score = sum(items[0]["base_status"] == items[0]["plus_status"] == "pass"
                    for items in statuses)/len(rows)
        result = {"pass@1": score, "count": len(rows),
                  "grader": "EvalPlus-expanded-tests", "raw_results": raw}
    (path.parent/"result.json").write_text(json.dumps(result, default=str))
if __name__ == "__main__":
    main(sys.argv[1])
