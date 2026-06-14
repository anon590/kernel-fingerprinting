#!/usr/bin/env python3
"""CLI entry for the Metal-ZK kernel evolution benchmark.

Examples:
    # Verify that a seed compiles + passes bit-exact correctness + times:
    python run_benchmark.py --task goldilocks_ntt --evaluate-seed-only

    # Run 3 LLM iterations with Claude on poseidon2_hash:
    python run_benchmark.py --task poseidon2_hash --model claude-opus-4-7 --iterations 3

    # Gemini / OpenAI work the same way as in metal-kernels.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from metal_zk import tasks  # noqa: F401  (registers tasks)
from metal_zk.evolve import evolve
from metal_zk.harness import MetalHarness
from metal_zk.hardware import detect_chip
from metal_zk.task import get_task, list_tasks


def evaluate_seed(name: str, n_reps: int) -> int:
    chip = detect_chip()
    harness = MetalHarness()
    task = get_task(name)
    src = task.spec.seed_path.read_text()
    result = task.evaluate_candidate(
        harness, chip, src, n_warmup=3, n_measure=10, n_reps=n_reps,
    )
    print(f"=== seed evaluation: task={name} ===")
    print(f"  chip: {chip.name}")
    print(f"    peak fp32:   {chip.peak_fp32_gflops:.0f} GFLOPS")
    print(f"    peak int64 mul (est): {chip.peak_int64_mul_gops:.0f} Gops/s")
    print(f"    peak DRAM BW: {chip.peak_bw_gb_s:.0f} GB/s")
    print(f"  device: {harness.device_name()}")
    print(f"  n_reps: {n_reps}")
    print(f"  compile_ok: {result.compile_ok}")
    print(f"  score (gmean fraction-of-ceiling): {result.score}")
    if result.fail_reason:
        print(f"  fail_reason: {result.fail_reason}")
    for s in result.size_results:
        ok = "OK" if s.correct else "FAIL"
        reps = s.extra.get("gpu_seconds_per_rep")
        rep_str = ""
        if reps and len(reps) > 1:
            lo, hi = min(reps), max(reps)
            rep_str = f"  [reps {lo*1e3:.2f}-{hi*1e3:.2f} ms]"
        print(
            f"  {s.size_label:>16s} [{ok}]: err={s.error_value} ({s.error_kind}), "
            f"{s.gpu_seconds*1e3:7.2f} ms, "
            f"{s.achieved:7.1f} {s.achieved_unit} "
            f"({s.fraction_of_ceiling*100:.1f}% of {s.ceiling:.0f} {s.ceiling_unit})"
            f"{rep_str}"
        )
    return 0 if (result.score is not None) else 1


async def run_evolution(args) -> int:
    task = get_task(args.task)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # The _redacted suffix keeps the disclosed and redacted arms in separate
    # run directories so a sweep over both never collides.
    suffix = "_redacted" if args.redact_held_out else ""
    out_dir = (
        Path(args.output_dir)
        / f"{args.task}_{args.model}_{timestamp}{suffix}"
    )
    await evolve(
        task,
        model=args.model,
        n_iterations=args.iterations,
        output_dir=out_dir,
        n_warmup=args.warmup,
        n_measure=args.measure,
        n_reps=args.reps,
        redact_held_out=args.redact_held_out,
    )
    return 0


def dry_run_prompt(args) -> int:
    """Build and print the iteration-1 prompt for both arms without calling
    the LLM or the GPU. Lets you eyeball exactly what redaction removes (and
    confirms the denylist passes) before spending any sweep budget."""
    from dataclasses import replace as _replace

    from metal_zk.prompts import build_initial_prompt
    from metal_zk.redact import apply_redactions, assert_no_disclosure
    from metal_zk.task import CandidateResult

    task = get_task(args.task)
    spec = task.spec
    seed_src = spec.seed_path.read_text()
    # A placeholder seed result so the prompt builder has a baseline block;
    # values are irrelevant to the disclosure text we are inspecting.
    stub = CandidateResult(
        compile_ok=True, compile_error=None, pipeline_error=None,
        size_results=[], score=1.0, fail_reason=None,
    )

    disclosed = build_initial_prompt(spec, seed_src, stub)
    print("=" * 78)
    print(f"DISCLOSED prompt — task={spec.name} ({len(disclosed)} chars)")
    print("=" * 78)
    print(disclosed)

    if not spec.redactions:
        print(f"\n[!] task {spec.name!r} declares no redactions "
              "(not one of the disclosed tasks); nothing to redact.")
        return 0

    desc2, sig2, seed2, manifest = apply_redactions(
        spec.description, spec.kernel_signatures, seed_src, spec.redactions,
    )
    redacted = build_initial_prompt(
        _replace(spec, description=desc2, kernel_signatures=sig2), seed2, stub,
    )
    assert_no_disclosure(redacted, spec.held_out_denylist)
    print("\n" + "=" * 78)
    print(f"REDACTED prompt — task={spec.name} ({len(redacted)} chars)")
    print(f"redactions applied: {len(manifest)}; "
          f"denylist OK ({spec.held_out_denylist})")
    print("=" * 78)
    print(redacted)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Metal-ZK kernel evolution benchmark CLI"
    )
    parser.add_argument(
        "--task", required=True,
        choices=list_tasks() or ["<no tasks registered>"],
        help="Which task to run.",
    )
    parser.add_argument(
        "--model", default="claude-sonnet-4-6",
        help="LLM to use. Claude (e.g. claude-opus-4-7, claude-sonnet-4-6), "
             "Gemini (e.g. gemini-3.1-pro-preview), "
             "or OpenAI (e.g. gpt-5.5, gpt-5, o4-mini).",
    )
    parser.add_argument(
        "--iterations", type=int, default=3,
        help="Number of LLM iterations.",
    )
    parser.add_argument(
        "--warmup", type=int, default=3,
        help="Warmup dispatches per timing measurement.",
    )
    parser.add_argument(
        "--measure", type=int, default=10,
        help="Measured dispatches per timing measurement.",
    )
    parser.add_argument(
        "--reps", type=int, default=3,
        help="Number of independent rep evaluations per size; the "
             "median rep is reported. Dampens cache-boundary noise.",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Directory under which run outputs are written.",
    )
    parser.add_argument(
        "--evaluate-seed-only", action="store_true",
        help="Just compile + run + verify the seed kernel; skip the LLM loop.",
    )
    parser.add_argument(
        "--redact-held-out", action="store_true",
        help="Controlled grade-C experiment: strip the held-out "
             "configuration's identity from everything the model sees "
             "(description, signatures, seed comments). Evaluation is "
             "unchanged. Only the three disclosed tasks "
             "(keccak_f1600_batch, kyber_ntt, wots_chain) support this; "
             "others raise. Output dir gets a _redacted suffix.",
    )
    parser.add_argument(
        "--dry-run-prompt", action="store_true",
        help="Print the iteration-1 prompt for the disclosed and redacted "
             "arms and exit (no LLM, no GPU). Use to verify what redaction "
             "removes before launching a sweep.",
    )
    args = parser.parse_args()

    if args.dry_run_prompt:
        return dry_run_prompt(args)
    if args.evaluate_seed_only:
        return evaluate_seed(args.task, n_reps=args.reps)
    return asyncio.run(run_evolution(args))


if __name__ == "__main__":
    raise SystemExit(main())
