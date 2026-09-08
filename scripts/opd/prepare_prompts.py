#!/usr/bin/env python3
"""Write a small prompt file for OPD.

verl wants parquet with a chat `prompt` column. slime still uses JSONL `{prompt: str}`.
Format follows the output suffix, or `--format`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_PROMPTS = [
    "Explain why the sky appears blue in short sentences.",
    "What is the difference between a compiler and an interpreter?",
    "Summarize the water cycle in one paragraph.",
    "Is 17 a prime number? Give a brief reason.",
    "Write a short story about a robot learning to cook.",
    "What causes seasons on Earth?",
    "Define overfitting in machine learning.",
    "Compare RAM and ROM.",
    "Why do we see lightning before we hear thunder?",
    "Give three tips for writing a clear email.",
    "What is photosynthesis?",
    "Explain entropy in everyday language.",
    "Who was Ada Lovelace and why is she important?",
    "How does a binary search work?",
    "What is the greenhouse effect?",
    "Describe the difference between weather and climate.",
]


def infer_format(path: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return "parquet"
    return "jsonl"


def rows(repeat: int) -> list[dict]:
    out = []
    for _ in range(max(1, repeat)):
        for text in DEFAULT_PROMPTS:
            out.append(
                {
                    "data_source": "opd_calib",
                    "prompt": [{"role": "user", "content": text}],
                    "ability": "general",
                    "reward_model": {"style": "rule", "ground_truth": ""},
                    "extra_info": {"split": "train", "index": len(out)},
                    "raw_prompt": text,
                }
            )
    return out


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps({"prompt": rec["raw_prompt"]}, ensure_ascii=False) + "\n")


def write_parquet(path: Path, records: list[dict]) -> None:
    from datasets import Dataset

    payload = [{k: v for k, v in rec.items() if k != "raw_prompt"} for rec in records]
    Dataset.from_list(payload).to_parquet(str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=8, help="Repeat the seed prompts this many times.")
    parser.add_argument("--format", choices=["jsonl", "parquet"], default=None)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fmt = infer_format(args.output, args.format)
    records = rows(args.repeat)
    if fmt == "parquet":
        write_parquet(args.output, records)
    else:
        write_jsonl(args.output, records)
    print(f"wrote {len(records)} prompts ({fmt}) -> {args.output}")


if __name__ == "__main__":
    main()
