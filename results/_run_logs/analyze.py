#!/usr/bin/env python3
"""Summarize metal-zk results across all (task, model) latest runs.

Combines:
- In-distribution scores from ``summary.json`` / ``best_result.json``.
- Held-out scores from ``_run_logs/held_out_results.json`` (run
  ``eval_held_out.py`` first; the held-out section is silently
  skipped if that file is absent).

Output is a markdown report on stdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (
    MODELS, ROOT, latest_runs, load_summary, load_best_result,
    best_iter_from_history, candidate_stats,
)


def _fmt_pct(x: float | None) -> str:
    return f"{x*100:.2f}%" if x is not None else "—"


def _fmt_sp(num: float | None, den: float | None) -> str:
    if num is None or not den:
        return "—"
    return f"{num/den:.2f}×"


def main() -> int:
    runs = latest_runs()
    if not runs:
        print("No runs found under results/.")
        return 1

    by_task: dict[str, dict[str, Path]] = {}
    for (task, model), run_dir in runs.items():
        by_task.setdefault(task, {})[model] = run_dir
    tasks_sorted = sorted(by_task)
    models_sorted = sorted({m for d in by_task.values() for m in d})

    held_path = ROOT / "_run_logs" / "held_out_results.json"
    held_rows: list[dict] = []
    if held_path.exists():
        held_rows = json.loads(held_path.read_text())
    held_lookup: dict[tuple[str, str], dict] = {
        (r["task"], r["model"]): r for r in held_rows
    }

    # ---------- 1. In-distribution per-(task,model) ----------
    print("# metal-zk benchmark summary\n")
    print(f"Tasks:  {', '.join(tasks_sorted)}")
    print(f"Models: {', '.join(models_sorted)}")
    print(
        f"\n_Scope: latest usable run (has `summary.json`) per (task, model), "
        f"restricted to {', '.join(MODELS)}. Other models on disk — e.g. the "
        f"exploratory gemini-3.5-flash sweeps — are excluded via "
        f"`_common.MODELS`._"
    )
    missing = [
        (t, m) for t in tasks_sorted for m in models_sorted
        if m not in by_task[t]
    ]
    if missing:
        miss_str = ", ".join(f"`{t}/{m}`" for t, m in missing)
        print(
            f"\n_No completed run (omitted from every section): {miss_str}._"
        )
    print()

    print("## In-distribution scores (gmean fraction-of-ceiling over in-dist sizes)\n")
    print("| Task | Model | iters | Seed | Best | Speedup | Best iter | "
          "Compile fails | Correctness fails | Wall time | Run dir |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    rows: list[dict] = []
    for task in tasks_sorted:
        for model in models_sorted:
            run_dir = by_task[task].get(model)
            if run_dir is None:
                continue
            summary = load_summary(run_dir)
            stats = candidate_stats(summary)
            seed = summary["seed_score"]
            best = summary["best_score"]
            best_iter = best_iter_from_history(summary)
            speedup = best / seed if (seed and best) else None
            rows.append({
                "task": task, "model": model, "summary": summary,
                "stats": stats, "speedup": speedup, "best_iter": best_iter,
                "run_dir": run_dir,
            })
            print(
                f"| {task} | {model} | {summary['n_iterations']} | "
                f"{seed:.4f} | {best:.4f} | "
                f"**{speedup:.2f}×** | {best_iter} | "
                f"{stats['compile_fails']}/{stats['n_candidates']} | "
                f"{stats['correct_fails']}/{stats['n_candidates']} | "
                f"{stats['wall_seconds']/60:.1f} min | {run_dir.name} |"
            )

    # ---------- 2. Per-size breakdown for best (in-dist) ----------
    print("\n## Best-candidate per-size breakdown (in-dist, fraction of ceiling)\n")
    for r in rows:
        br = load_best_result(r["run_dir"])
        parts = [
            f"{s['label']}: {s['fraction_of_ceiling']*100:.1f}%"
            for s in br["sizes"]
        ]
        print(f"- **{r['task']} / {r['model']}** — " + " | ".join(parts))

    # ---------- 3. Held-out section ----------
    if not held_rows:
        print(
            "\n_(Held-out section skipped: run `python results/_run_logs/"
            "eval_held_out.py` to populate `held_out_results.json`.)_"
        )
        return 0

    print("\n## Held-out: seed vs best (one new config per task)\n")
    print(
        "_Each task is evaluated on a single held-out config never seen during "
        "the search. `Generalisation` = held-out best frac ÷ in-dist best gmean "
        "(≈1.0 means the held-out config matches in-dist quality; >1 means it "
        "transferred even better). `Speedup vs seed` = held-out best frac ÷ "
        "held-out seed frac._\n"
    )
    print("| Task | Model | Held-out | Seed frac | Best frac | "
          "In-dist best (gmean) | Held-out best (abs) | Generalisation | "
          "Speedup vs seed | Notes |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        h = held_lookup.get((r["task"], r["model"]))
        if h is None:
            continue
        seed_f = h["seed_frac"]
        best_f = h["best_frac"]
        in_dist_best = r["summary"]["best_score"]
        held_abs = (
            f"{h['best_achieved']:.1f} {h['best_unit']}"
            if h.get("best_achieved") is not None else "—"
        )
        notes = []
        if not h["seed_compile_ok"]:
            notes.append("seed compile fail")
        elif not h["seed_correct"]:
            notes.append("seed correctness fail")
        if not h["best_compile_ok"]:
            notes.append("best compile fail")
        elif not h["best_correct"]:
            notes.append("best correctness fail")
        note_str = "; ".join(notes)
        print(
            f"| {r['task']} | {r['model']} | {h['held_label']} | "
            f"{_fmt_pct(seed_f)} | {_fmt_pct(best_f)} | "
            f"{in_dist_best:.4f} | {held_abs} | "
            f"{_fmt_sp(best_f, in_dist_best)} | {_fmt_sp(best_f, seed_f)} | "
            f"{note_str} |"
        )

    # ---------- 4. Cross-model comparison per task ----------
    if len(models_sorted) > 1:
        print("\n## Cross-model comparison (in-dist best, held-out best)\n")
        header = ["Task"]
        for m in models_sorted:
            header.append(f"{m} in-dist")
            header.append(f"{m} held-out")
        print("| " + " | ".join(header) + " |")
        print("|" + "|".join(["---"] * len(header)) + "|")
        for task in tasks_sorted:
            row = [task]
            for m in models_sorted:
                run_dir = by_task[task].get(m)
                if run_dir is None:
                    row.append("—")
                    row.append("—")
                    continue
                summary = load_summary(run_dir)
                row.append(f"{summary['best_score']:.4f}")
                h = held_lookup.get((task, m))
                if h is None or h["best_frac"] is None:
                    row.append("—" if h is None else "FAIL")
                else:
                    row.append(f"{h['best_frac']*100:.2f}%")
            print("| " + " | ".join(row) + " |")

    return 0


if __name__ == "__main__":
    sys.exit(main())
