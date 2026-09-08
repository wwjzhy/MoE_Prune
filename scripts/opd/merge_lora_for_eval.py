#!/usr/bin/env python3
"""Merge a PEFT adapter into the HC student HF weights for lm-eval."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True, help="HC-SMoE (or other) student HF dir")
    parser.add_argument("--adapter", type=Path, required=True, help="PEFT adapter dir (adapter_config.json)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (args.base / "config.json").is_file():
        raise SystemExit(f"base model missing config.json: {args.base}")
    if not (args.adapter / "adapter_config.json").is_file():
        raise SystemExit(f"adapter missing adapter_config.json: {args.adapter}")
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"load base {args.base}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print(f"load adapter {args.adapter}")
    model = PeftModel.from_pretrained(model, str(args.adapter))
    model = model.merge_and_unload()
    print(f"save merged {args.output}")
    model.save_pretrained(args.output)
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    tok.save_pretrained(args.output)
    print("done")


if __name__ == "__main__":
    main()
