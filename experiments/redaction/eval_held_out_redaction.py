#!/usr/bin/env python3
"""Held-out evaluation for the redaction experiment (both arms).

Scans ``results/redaction_experiment/`` for runs (disclosed and
``_redacted``), and for each compiles + times its ``best.metal`` and
``00_seed.metal`` on the task's ``held_out_sizes``. Writes one tidy row per
(task, model, arm) to ``held_out_redaction.json``.

This reuses the production evaluator (``task.evaluate_candidate`` on
``held_out_sizes``) unchanged --- the held-out probe is identical to the
disclosed arm's; the only thing that differed between arms was what the
model saw during the search.

Dry by default (lists what it would evaluate); pass --go to run on the GPU.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

EXP_DIR = REPO / "results" / "redaction_experiment"
OUT_JSON = Path(__file__).resolve().parent / "held_out_redaction.json"

# run dir name: <task>_<model>_<timestamp>[_redacted]
_RUN_RE = re.compile(r"^(?P<task>.+)_(?P<model>[^_]+(?:-[^_]+)*)_\d{8}_\d{6}(?P<red>_redacted)?$")


def _parse_run(dirname: str) -> tuple[str, str, str] | None:
    """Return (task, model, arm) or None. Task list is the disclosed three."""
    arm = "redacted" if dirname.endswith("_redacted") else "disclosed"
    base = dirname[:-9] if arm == "redacted" else dirname
    for task in ("keccak_f1600_batch", "kyber_ntt", "wots_chain"):
        if base.startswith(task + "_"):
            model = base[len(task) + 1:].rsplit("_", 2)[0]
            return task, model, arm
    return None


def _discover() -> list[tuple[Path, str, str, str]]:
    runs = []
    if not EXP_DIR.is_dir():
        return runs
    for d in sorted(EXP_DIR.iterdir()):
        if not d.is_dir():
            continue
        parsed = _parse_run(d.name)
        if parsed and (d / "best.metal").exists():
            runs.append((d, *parsed))
    return runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true",
                    help="Actually compile+time on the GPU (default: dry list).")
    args = ap.parse_args()

    runs = _discover()
    print(f"# discovered {len(runs)} run(s) under {EXP_DIR}")
    for d, task, model, arm in runs:
        print(f"  {arm:9s} {task:20s} {model:24s} {d.name}")
    if not runs:
        print("  (none yet — run run_redaction_sweep.sh --go first)")
        return 0
    if not args.go:
        print("\nDry run: pass --go to evaluate held-out on the GPU.")
        return 0

    from metal_zk import tasks  # noqa: F401  (registers tasks)
    from metal_zk.harness import MetalHarness
    from metal_zk.hardware import detect_chip
    from metal_zk.task import get_task

    chip = detect_chip()
    harness = MetalHarness()
    seed_cache: dict[str, dict] = {}
    rows = []

    for d, task_name, model, arm in runs:
        task = get_task(task_name)
        held = task.spec.held_out_sizes

        def _frac(source: str):
            res = task.evaluate_candidate(harness, chip, source, sizes=held)
            sr = res.size_results[0] if res.size_results else None
            return {
                "correct": bool(sr and sr.correct),
                "frac": (sr.fraction_of_ceiling if (sr and sr.correct) else None),
                "fail_reason": res.fail_reason,
                "held_label": held[0].label if held else None,
            }

        if task_name not in seed_cache:
            seed_cache[task_name] = _frac((d / "00_seed.metal").read_text())
        seed_row = seed_cache[task_name]
        best_row = _frac((d / "best.metal").read_text())

        summary = json.loads((d / "summary.json").read_text())
        ho = (best_row["frac"] / seed_row["frac"]
              if best_row["frac"] and seed_row["frac"] else None)
        rows.append({
            "task": task_name, "model": model, "arm": arm,
            "run_dir": d.name,
            "id_speedup": summary.get("improvement"),
            "held_label": seed_row["held_label"],
            "seed_held_frac": seed_row["frac"],
            "best_held_frac": best_row["frac"],
            "held_correct": best_row["correct"],
            "ho_speedup": ho,
            "held_fail_reason": best_row["fail_reason"],
        })
        print(f"  -> {arm:9s} {task_name:20s} {model:24s} "
              f"ID {summary.get('improvement')}  HO {ho}")

    OUT_JSON.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {OUT_JSON} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
