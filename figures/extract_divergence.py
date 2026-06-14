#!/usr/bin/env python3
"""Extract the (1+1) incumbent divergence trajectory for exemplar cells.

For each exemplar (task, model) run we reconstruct the incumbent lineage
from ``summary.json`` (the (1+1) loop keeps the latest ``is_new_best``
candidate) and, for every iteration's incumbent kernel, evaluate the
task's held-out config *retroactively* (post-hoc, the search is over).

Two signals per iteration:
  - in-distribution score the loop actually saw (from summary.json),
  - held-out fraction-of-ceiling measured now on the same incumbent.

Both are normalised to the seed so the y-axis is "x seed". The held-out
seed + all incumbents are measured back-to-back in one session so the
ratios are internally consistent despite absolute-timing machine noise.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/Users/anon/metal-zk")
sys.path.insert(0, str(ROOT))

from metal_zk import tasks  # noqa: F401  (registers tasks)
from metal_zk.harness import MetalHarness
from metal_zk.hardware import detect_chip
from metal_zk.task import get_task

N_REPS = 7  # extra reps to dampen held-out timing noise

EXEMPLARS = [
    {
        "run_dir": "multilinear_sumcheck_round_claude-opus-4-7_20260517_173349",
        "task": "multilinear_sumcheck_round",
        "label": "sumcheck / Opus 4.7",
        "fingerprint_iter": 4,
        "fingerprint_note": r"$\mathtt{if\ (d==2)}$ fast path enters",
    },
    {
        "run_dir": "logup_gkr_gemini-3.1-pro-preview_20260515_133024",
        "task": "logup_gkr",
        "label": "logup / Gemini 3.1",
        "fingerprint_iter": 3,
        "fingerprint_note": "wrong Barrett constant enters",
    },
    {
        "run_dir": "binius_clmul_claude-opus-4-7_20260517_172042",
        "task": "binius_clmul",
        "label": "binius / Opus 4.7",
        "fingerprint_iter": 5,
        "fingerprint_note": "windowed-scan rewrite (no predicate)",
    },
]


def incumbent_lineage(history):
    """Return list of incumbent source-iter index per iteration 0..N.

    The (1+1) loop keeps the latest is_new_best candidate. Seed (iter 0)
    is always the initial incumbent.
    """
    lineage = []
    cur = None
    for h in sorted(history, key=lambda r: r["iteration"]):
        if h["is_new_best"]:
            cur = h["iteration"]
        lineage.append((h["iteration"], cur))
    return lineage


def metal_path(run_dir: Path, src_iter: int) -> Path:
    if src_iter == 0:
        return run_dir / "00_seed.metal"
    return run_dir / f"{src_iter:02d}_candidate.metal"


def eval_held(task, harness, chip, source, held):
    res = task.evaluate_candidate(
        harness, chip, source, sizes=held, n_reps=N_REPS,
    )
    sr = res.size_results[0] if res.size_results else None
    correct = bool(sr and sr.correct)
    frac = (sr.fraction_of_ceiling if correct else None)
    return {
        "compile_ok": res.compile_ok,
        "correct": correct,
        "frac": frac,
        "fail": None if correct else res.fail_reason,
    }


def main() -> int:
    chip = detect_chip()
    harness = MetalHarness()
    out = {"chip": chip.name, "n_reps": N_REPS, "runs": []}

    for ex in EXEMPLARS:
        run_dir = ROOT / "results" / ex["run_dir"]
        summary = json.loads((run_dir / "summary.json").read_text())
        task = get_task(ex["task"])
        held = task.spec.held_out_sizes
        held_label = held[0].label
        history = summary["history"]
        lineage = incumbent_lineage(history)
        score_by_iter = {h["iteration"]: h["score"] for h in history}
        seed_in = summary["seed_score"]

        print(f"\n## {ex['label']}  ({held_label})", flush=True)

        # held-out eval cache keyed by source iteration (distinct incumbents)
        held_cache: dict[int, dict] = {}

        def held_for(src_iter: int) -> dict:
            if src_iter not in held_cache:
                src = metal_path(run_dir, src_iter).read_text()
                held_cache[src_iter] = eval_held(task, harness, chip, src, held)
                hc = held_cache[src_iter]
                tag = (f"frac={hc['frac']:.5f}" if hc["correct"]
                       else f"FAIL ({str(hc['fail'])[:30]})")
                print(f"   incumbent src=iter{src_iter}: {tag}", flush=True)
            return held_cache[src_iter]

        seed_held = held_for(0)
        seed_held_frac = seed_held["frac"]

        iters = []
        for it, src_iter in lineage:
            in_score = score_by_iter.get(src_iter)
            in_ratio = (in_score / seed_in) if (in_score and seed_in) else None
            hc = held_for(src_iter)
            ho_ratio = (
                hc["frac"] / seed_held_frac
                if (hc["correct"] and seed_held_frac) else None
            )
            iters.append({
                "iter": it,
                "incumbent_src": src_iter,
                "in_dist_ratio": in_ratio,
                "held_out_ratio": ho_ratio,
                "held_out_correct": hc["correct"],
            })

        out["runs"].append({
            "task": ex["task"],
            "label": ex["label"],
            "held_label": held_label,
            "fingerprint_iter": ex["fingerprint_iter"],
            "fingerprint_note": ex["fingerprint_note"],
            "seed_in_dist": seed_in,
            "seed_held_out_frac": seed_held_frac,
            "best_in_dist_ratio": summary["improvement"],
            "iters": iters,
        })

    dest = ROOT / "figures" / "divergence_data.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
