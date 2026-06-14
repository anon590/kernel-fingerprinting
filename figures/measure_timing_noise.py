#!/usr/bin/env python3
"""Characterize within-configuration timing noise for ZK seeds.

Two noise figures, both on the *production* timing path (no LLM, no
evolution -- pure timing of the seed kernels already in-tree):

1. Per-configuration single-rep CV: re-run one rep (n_warmup=3,
   n_measure=10, median GPU time) R times and report the run-to-run CV
   of that median time, per configuration.

2. Production score CV: re-run the full incumbent-scoring path
   (evaluate_candidate with n_reps=3 -> median rep, gmean of
   fraction-of-roofline over the in-distribution sizes) R times and
   report the CV of the resulting score. This is the exact quantity the
   (1+1) promotion rule compares and the 1.05x win threshold is applied
   to, so its CV is the relevant noise floor.

Both are measurement-precision of a *fixed* kernel; neither speaks to
search-trajectory variance (the n=1-sweep limitation).

Spread across roofline anchors: NTT (BW/int-mul crossover, with
sub-0.1 ms SLC-resident sizes), Poseidon2 (int-mul bound), Keccak (bitop
bound), MSM (modular, multi-kernel + tree reduction).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from metal_zk import tasks  # noqa: F401  (registers tasks)
from metal_zk.harness import MetalHarness
from metal_zk.hardware import detect_chip
from metal_zk.task import get_task

OUT_JSON = Path(__file__).resolve().parent / "timing_noise_data.json"

PROBE_TASKS = [
    "goldilocks_ntt",
    "poseidon2_hash",
    "keccak_f1600_batch",
    "montgomery_msm",
]
R_SIZE = 12   # single-rep re-measurements per configuration
R_SCORE = 8   # full production-path re-measurements of the score


def cv(xs: list[float]) -> float:
    m = statistics.mean(xs)
    return statistics.pstdev(xs) / m if m > 0 else float("nan")


def main() -> int:
    chip = detect_chip()
    harness = MetalHarness()
    print(f"chip: {chip.name}  device: {harness.device_name()}")
    print(f"single-rep: R={R_SIZE} (n_warmup=3, n_measure=10, median)")
    print(f"score:      R={R_SCORE} (production evaluate_candidate, n_reps=3)\n")

    per_config_cvs: list[float] = []
    score_cvs: list[float] = []
    dump = {
        "chip": chip.name,
        "r_size": R_SIZE,
        "r_score": R_SCORE,
        "tasks": {},
    }
    for name in PROBE_TASKS:
        task = get_task(name)
        src = task.spec.seed_path.read_text()
        cr = harness.compile(src)
        if cr.error is not None:
            print(f"[{name}] compile error: {cr.error}")
            continue
        pipelines, perr = harness.make_pipelines(cr.library, task.spec.kernel_names)
        if perr is not None:
            print(f"[{name}] pipeline error: {perr}")
            continue

        print(f"[{name}]")
        dump["tasks"][name] = {"configs": [], "scores": []}
        sizes = list(task.spec.sizes) + list(task.spec.held_out_sizes)
        for size in sizes:
            held = size in task.spec.held_out_sizes
            tag = "HO" if held else "ID"
            times = []
            ok = True
            for _ in range(R_SIZE):
                res = task.evaluate_size(harness, pipelines, size, chip, 3, 10)
                ok = ok and res.correct
                times.append(res.gpu_seconds)
            c = cv(times)
            if not held:
                per_config_cvs.append(c)
            dump["tasks"][name]["configs"].append({
                "label": size.label,
                "held_out": held,
                "gpu_seconds": times,
                "cv": c,
            })
            lo, hi = min(times), max(times)
            print(f"   {tag} {size.label:>18s}: "
                  f"median {statistics.median(times)*1e3:9.3f} ms  "
                  f"single-rep CV {c*100:5.2f}%  "
                  f"max/min {hi/lo:.3f}  {'OK' if ok else 'FAIL'}")

        # Production-path score CV (in-distribution sizes only -> the score).
        scores = []
        for _ in range(R_SCORE):
            cand = task.evaluate_candidate(
                harness, chip, src, n_warmup=3, n_measure=10, n_reps=3,
            )
            if cand.score is not None:
                scores.append(cand.score)
        if scores:
            sc = cv(scores)
            score_cvs.append(sc)
            dump["tasks"][name]["scores"] = scores
            print(f"   -> production score: median {statistics.median(scores):.4f}  "
                  f"CV {sc*100:.2f}%  (R={len(scores)})")
        print()

    print(f"single-rep CV across ID configs: median {statistics.median(per_config_cvs)*100:.2f}%  "
          f"max {max(per_config_cvs)*100:.2f}%  (n={len(per_config_cvs)})")
    print(f"production score CV across tasks: median {statistics.median(score_cvs)*100:.2f}%  "
          f"max {max(score_cvs)*100:.2f}%  (n={len(score_cvs)})")
    OUT_JSON.write_text(json.dumps(dump, indent=2))
    print(f"raw data -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
