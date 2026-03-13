#!/usr/bin/env python3
"""
EmotWen fine-tune pipeline CLI.

Usage:
    python main.py <stage> [key=value ...]

Stages:
    data_prep   Load, filter and save training data
    sft         Two-stage supervised fine-tuning
    eval        Evaluate SFT model (decides whether GRPO is needed)
    grpo        GRPO reinforcement step (auto-skipped if not needed)
    all         Run all stages in order

Config overrides are passed as key=value pairs and applied on top of
the defaults in src/config.py.  Values are parsed as JSON where possible,
falling back to plain strings.

Examples:
    python main.py data_prep max_empathetic=5000 max_daily_dialog=1000
    python main.py sft stage=1
    python main.py eval judge_provider=anthropic
    python main.py grpo skip_if_not_needed=true
    python main.py all max_empathetic=2000
"""

import argparse
import json
import sys


def _parse_overrides(pairs: list[str]) -> dict:
    overrides = {}
    for pair in pairs:
        if "=" not in pair:
            print(f"Warning: skipping malformed override {pair!r} (expected key=value)", file=sys.stderr)
            continue
        key, _, raw = pair.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        overrides[key] = value
    return overrides


STAGES = {
    "data_prep": "src.data_prep",
    "sft": "src.train_sft",
    "eval": "src.evaluate",
    "grpo": "src.train_grpo",
}


def run_stage(name: str, overrides: dict) -> dict:
    import importlib
    mod = importlib.import_module(STAGES[name])
    print(f"\n{'='*60}")
    print(f"  Stage: {name}")
    if overrides:
        print(f"  Overrides: {overrides}")
    print(f"{'='*60}\n")
    return mod.run(overrides or None)


def main():
    parser = argparse.ArgumentParser(
        description="EmotWen fine-tune pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "stage",
        choices=[*STAGES, "all"],
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="key=value",
        help="Config overrides (e.g. max_empathetic=5000)",
    )
    args = parser.parse_args()
    overrides = _parse_overrides(args.overrides)

    if args.stage == "all":
        for name in STAGES:
            results = run_stage(name, overrides)
            print(f"\nResults ({name}):", results)
    else:
        results = run_stage(args.stage, overrides)
        print(f"\nResults:", results)


if __name__ == "__main__":
    main()
