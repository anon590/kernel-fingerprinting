#!/usr/bin/env python3
"""Evaluate seed and best candidates on each task's held-out config.

For every (task, model) pair with results in ``results/`` we pick the
latest run (by directory timestamp) that has ``00_seed.metal``,
``best.metal`` and ``summary.json``. We then compile + time both
kernels on the task's ``held_out_sizes`` and persist the rows to
``results/_run_logs/held_out_results.json``.

The seed source is identical for every model targeting the same task,
so we evaluate it once per task and reuse the result across models.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/Users/anon/metal-zk")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metal_zk import tasks  # noqa: F401  (registers tasks)
from metal_zk.harness import MetalHarness
from metal_zk.hardware import detect_chip
from metal_zk.task import get_task

from _common import MODELS, ROOT, latest_runs


def _evaluate(task, harness, chip, source, held):
    res = task.evaluate_candidate(harness, chip, source, sizes=held)
    rows = []
    for hsize in held:
        sr = next(
            (r for r in res.size_results if r.size_label == hsize.label), None
        )
        rows.append({
            "label": hsize.label,
            "compile_ok": res.compile_ok,
            "pipeline_ok": res.pipeline_error is None,
            "correct": sr.correct if sr else False,
            "frac": (sr.fraction_of_ceiling if (sr and sr.correct) else None),
            "achieved": (sr.achieved if (sr and sr.correct) else None),
            "achieved_unit": (sr.achieved_unit if sr else ""),
            "ceiling": (sr.ceiling if sr else None),
            "ceiling_unit": (sr.ceiling_unit if sr else ""),
            "gpu_ms": (sr.gpu_seconds * 1e3 if (sr and sr.correct) else None),
            "fail_reason": res.fail_reason,
        })
    return rows


def main() -> int:
    chip = detect_chip()
    harness = MetalHarness()
    print(f"# Held-out evaluation, chip={chip.name}")
    print(f"  peak DRAM BW:     {chip.peak_bw_gb_s:.0f} GB/s")
    print(f"  peak int64 mul:   {chip.peak_int64_mul_gops:.0f} Gops/s")
    print(f"  peak int64 bitop: {chip.peak_int64_bitop_gops:.0f} Gops/s")
    print(f"  peak int64 rot:   {chip.peak_int64_rotate_gops:.0f} Gops/s")
    print(f"  models in scope:  {', '.join(MODELS)}")
    print()

    runs = latest_runs()
    by_task: dict[str, dict[str, Path]] = {}
    for (task_name, model), run_dir in runs.items():
        by_task.setdefault(task_name, {})[model] = run_dir

    seed_cache: dict[str, list[dict]] = {}
    rows: list[dict] = []

    for task_name in sorted(by_task):
        try:
            task = get_task(task_name)
        except KeyError:
            print(f"## {task_name}: unknown task (not registered) — skipped")
            continue
        held = task.spec.held_out_sizes
        if not held:
            print(f"## {task_name}: no held-out sizes — skipped")
            continue

        for model, run_dir in sorted(by_task[task_name].items()):
            seed_path = run_dir / "00_seed.metal"
            best_path = run_dir / "best.metal"
            if not (seed_path.exists() and best_path.exists()):
                print(f"## {task_name} / {model}: missing seed or best — skipped")
                continue

            if task_name not in seed_cache:
                print(f"  evaluating SEED for {task_name} (held-out)...", flush=True)
                seed_cache[task_name] = _evaluate(
                    task, harness, chip, seed_path.read_text(), held,
                )
            seed_rows = seed_cache[task_name]

            print(f"  evaluating BEST for {task_name} / {model}...", flush=True)
            best_rows = _evaluate(
                task, harness, chip, best_path.read_text(), held,
            )

            for sr, br in zip(seed_rows, best_rows):
                rows.append({
                    "task": task_name,
                    "model": model,
                    "run_dir": run_dir.name,
                    "held_label": sr["label"],
                    "seed_compile_ok": sr["compile_ok"],
                    "seed_correct": sr["correct"],
                    "seed_frac": sr["frac"],
                    "seed_achieved": sr["achieved"],
                    "seed_unit": sr["achieved_unit"],
                    "seed_gpu_ms": sr["gpu_ms"],
                    "seed_fail": sr["fail_reason"],
                    "best_compile_ok": br["compile_ok"],
                    "best_correct": br["correct"],
                    "best_frac": br["frac"],
                    "best_achieved": br["achieved"],
                    "best_unit": br["achieved_unit"],
                    "best_gpu_ms": br["gpu_ms"],
                    "best_fail": br["fail_reason"],
                    "ceiling": sr["ceiling"] if sr["ceiling"] is not None else br["ceiling"],
                    "ceiling_unit": sr["ceiling_unit"] or br["ceiling_unit"],
                })

    # Markdown table
    print("\n## Held-out: seed vs best (fraction of binding roofline)\n")
    print("| Task | Model | Held-out config | Seed frac | Best frac | "
          "Seed (abs) | Best (abs) | Speedup | Notes |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        seed_f = f"{r['seed_frac']*100:.2f}%" if r['seed_frac'] is not None else "FAIL"
        best_f = f"{r['best_frac']*100:.2f}%" if r['best_frac'] is not None else "FAIL"
        seed_a = (
            f"{r['seed_achieved']:.1f} {r['seed_unit']}"
            if r['seed_achieved'] is not None else "—"
        )
        best_a = (
            f"{r['best_achieved']:.1f} {r['best_unit']}"
            if r['best_achieved'] is not None else "—"
        )
        sp = (
            f"{r['best_frac']/r['seed_frac']:.2f}×"
            if (r['seed_frac'] and r['best_frac']) else "—"
        )
        notes = []
        if not r['seed_compile_ok']:
            notes.append(f"seed COMPILE FAIL: {r['seed_fail']}")
        elif not r['seed_correct']:
            notes.append(f"seed CORRECTNESS FAIL: {r['seed_fail']}")
        if not r['best_compile_ok']:
            notes.append(f"best COMPILE FAIL: {r['best_fail']}")
        elif not r['best_correct']:
            notes.append(f"best CORRECTNESS FAIL: {r['best_fail']}")
        note_str = "; ".join(notes)
        print(
            f"| {r['task']} | {r['model']} | {r['held_label']} | {seed_f} | "
            f"{best_f} | {seed_a} | {best_a} | **{sp}** | {note_str} |"
        )

    out = ROOT / "_run_logs" / "held_out_results.json"
    out.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nRaw results: {out.relative_to(ROOT.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
