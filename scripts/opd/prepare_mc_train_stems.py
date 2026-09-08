#!/usr/bin/env python3
"""Build verl parquet of 8-task *train* question stems for Exp #3.

Never uses test/validation. Skips MMLU (no public train split in lm-eval).
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Callable

from datasets import Dataset, load_dataset

StemFn = Callable[[dict[str, Any]], str | None]


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def hellaswag_stem(row: dict[str, Any]) -> str | None:
    return _text(row.get("ctx") or row.get("context"))


def arc_stem(row: dict[str, Any]) -> str | None:
    return _text(row.get("question"))


def winogrande_stem(row: dict[str, Any]) -> str | None:
    return _text(row.get("sentence"))


def piqa_stem(row: dict[str, Any]) -> str | None:
    return _text(row.get("goal"))


def openbookqa_stem(row: dict[str, Any]) -> str | None:
    return _text(row.get("question_stem") or row.get("question"))


def boolq_stem(row: dict[str, Any]) -> str | None:
    question = _text(row.get("question"))
    if not question:
        return None
    passage = _text(row.get("passage"))
    if passage:
        return f"{passage}\n\n{question}"
    return question


SOURCES: list[tuple[str, str | None, str, StemFn]] = [
    ("hellaswag", None, "train", hellaswag_stem),
    ("ai2_arc", "ARC-Easy", "train", arc_stem),
    ("ai2_arc", "ARC-Challenge", "train", arc_stem),
    ("winogrande", "winogrande_xl", "train", winogrande_stem),
    ("piqa", None, "train", piqa_stem),
    ("openbookqa", None, "train", openbookqa_stem),
    ("google/boolq", None, "train", boolq_stem),
]


def make_row(text: str, index: int, source: str) -> dict[str, Any]:
    return {
        "data_source": source,
        "prompt": [{"role": "user", "content": text}],
        "ability": "mc_stem",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {"split": "train", "index": index},
    }


def load_stems(max_per_source: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path, subset, split, stem_fn in SOURCES:
        label = path if subset is None else f"{path}/{subset}"
        try:
            kwargs: dict[str, Any] = {"split": split}
            if subset is not None:
                kwargs["name"] = subset
            ds = load_dataset(path, **kwargs)
        except Exception as exc:
            print(f"[skip] {label}: {exc}")
            continue
        n = 0
        for row in ds:
            if n >= max_per_source:
                break
            text = stem_fn(row)
            if not text or text in seen:
                continue
            seen.add(text)
            rows.append(make_row(text, len(rows), label))
            n += 1
        print(f"[ok] {label}: {n} stems")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-per-source", type=int, default=2000)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    rows = load_stems(args.max_per_source)
    if not rows:
        raise SystemExit("no train stems loaded")
    Dataset.from_list(rows).to_parquet(args.output)
    print(f"wrote {len(rows)} prompts -> {args.output}")


if __name__ == "__main__":
    main()
