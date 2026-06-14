"""Evolution loop: prompt LLM iteratively to improve a Metal kernel.

Mirrors the iterative-improvement structure from llm_self_play.py:
- evaluate the seed
- ask the LLM for an improvement, given previous attempt + incumbent best
- evaluate the candidate; if it beats the incumbent, promote it
- repeat for N iterations, persisting all artefacts to disk
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .harness import MetalHarness
from .hardware import ChipSpec, detect_chip
from .llm import call_llm, log
from .prompts import (
    SYSTEM_PROMPT,
    build_initial_prompt,
    build_iteration_prompt,
    extract_metal_source,
)
from .redact import apply_redactions, assert_no_disclosure
from .task import CandidateResult, Task


@dataclass
class IterationRecord:
    iteration: int
    role: str            # "seed", "candidate", "skipped"
    compile_ok: bool
    correct: bool
    score: float | None
    fail_reason: str | None
    source_path: str
    elapsed_s: float
    is_new_best: bool
    # Redacted arm only: held-out-identity phrases the MODEL emitted (in its
    # response or candidate) despite redaction --- the knowledge-driven
    # enumeration signal. Empty list when not applicable / none found.
    model_named_held_out: list[str] = field(default_factory=list)


def _result_summary(r: CandidateResult) -> dict:
    return {
        "compile_ok": r.compile_ok,
        "correct": r.score is not None,
        "score": r.score,
        "fail_reason": r.fail_reason,
        "sizes": [
            {
                "label": s.size_label,
                "correct": s.correct,
                "error_value": s.error_value,
                "error_kind": s.error_kind,
                "gpu_seconds": s.gpu_seconds,
                "achieved": s.achieved,
                "achieved_unit": s.achieved_unit,
                "ceiling": s.ceiling,
                "ceiling_unit": s.ceiling_unit,
                "fraction_of_ceiling": s.fraction_of_ceiling,
            }
            for s in r.size_results
        ],
    }


async def evolve(
    task: Task,
    *,
    model: str,
    n_iterations: int,
    output_dir: Path,
    n_warmup: int = 3,
    n_measure: int = 10,
    n_reps: int = 3,
    chip: ChipSpec | None = None,
    redact_held_out: bool = False,
) -> dict:
    """Run the evolution loop. Returns a summary dict; writes everything to disk.

    When ``redact_held_out`` is set, the held-out configuration's identity is
    stripped from everything the model sees (task description, kernel-signature
    block, and the seed comments shown in the prompt) per the task's declared
    redactions (metal_zk/redact.py). The in-distribution and held-out
    *evaluations* are unchanged --- disclosure is the only manipulated variable.
    The task must declare redactions; otherwise this raises, so a "redacted"
    run can never silently duplicate the disclosed arm.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    chip = chip or detect_chip()
    harness = MetalHarness()

    log(f"\n=== Metal kernel evolution: task={task.spec.name} ===")
    log(f"  model: {model}")
    log(f"  chip:  {chip.name} (peak {chip.peak_fp32_gflops:.0f} GFLOPS fp32, "
        f"{chip.peak_int64_mul_gops:.0f} Gops/s int64 mul [est], "
        f"{chip.peak_bw_gb_s:.0f} GB/s DRAM)")
    log(f"  timing: n_warmup={n_warmup}, n_measure={n_measure}, n_reps={n_reps}")
    log(f"  output: {output_dir}")

    history: list[IterationRecord] = []

    # --- Seed evaluation ----------------------------------------------------
    # seed_src is the ORIGINAL source: it is what gets evaluated and what is
    # persisted as 00_seed.metal, identical across the disclosed and redacted
    # arms. seed_display / prompt_spec are what the model SEES; under
    # redaction they have the held-out identity removed.
    seed_src = task.spec.seed_path.read_text()
    if redact_held_out:
        if not task.spec.redactions:
            raise ValueError(
                f"task {task.spec.name!r} declares no redactions; "
                "--redact-held-out is meaningless (it would duplicate the "
                "disclosed arm). Only the disclosed tasks support it."
            )
        desc2, sig2, seed_display, manifest = apply_redactions(
            task.spec.description, task.spec.kernel_signatures,
            seed_src, task.spec.redactions,
        )
        prompt_spec = replace(
            task.spec, description=desc2, kernel_signatures=sig2,
        )
        # Integrity gate: verify OUR redaction removed the held-out identity
        # from the text we author --- the redacted description, signature, and
        # displayed seed. This is the only content the experiment controls; it
        # is checked once, here, not per iteration. We deliberately do NOT
        # police model-generated content: a candidate that itself names the
        # held-out (e.g. Claude recalling that Kyber's NTT sibling is
        # Dilithium) is exactly the knowledge-driven enumeration this
        # experiment measures, so it is recorded per iteration (below), never
        # suppressed --- otherwise the loop, which feeds each candidate back
        # into the next prompt, would abort precisely on the cells where the
        # effect is strongest, biasing the result.
        assert_no_disclosure(
            desc2 + "\n" + sig2 + "\n" + seed_display,
            task.spec.held_out_denylist,
        )
        (output_dir / "redaction_manifest.json").write_text(json.dumps(
            {"task": task.spec.name, "redactions": manifest,
             "denylist": task.spec.held_out_denylist}, indent=2,
        ))
        log(f"  [redact] held-out identity removed from authored prompt text "
            f"({len(manifest)} redaction(s)); model self-disclosure recorded, "
            f"not suppressed")
    else:
        prompt_spec = task.spec
        seed_display = seed_src

    log("\n[seed] Evaluating seed kernel...")
    t0 = time.time()
    seed_result = task.evaluate_candidate(
        harness, chip, seed_src,
        n_warmup=n_warmup, n_measure=n_measure, n_reps=n_reps,
    )
    seed_elapsed = time.time() - t0
    seed_path = output_dir / "00_seed.metal"
    seed_path.write_text(seed_src)
    history.append(IterationRecord(
        iteration=0, role="seed",
        compile_ok=seed_result.compile_ok,
        correct=seed_result.score is not None,
        score=seed_result.score,
        fail_reason=seed_result.fail_reason,
        source_path=str(seed_path),
        elapsed_s=seed_elapsed,
        is_new_best=True,
    ))
    if seed_result.score is None:
        raise RuntimeError(
            f"Seed kernel failed evaluation: {seed_result.fail_reason}"
        )
    log(f"  seed score: {seed_result.score:.4f}  ({seed_elapsed:.1f}s)")

    # Incumbent/previous sources shown in prompts use the displayed (possibly
    # redacted) seed so iteration prompts never re-leak the held-out identity;
    # they are replaced by the model's own candidates after iteration 1.
    best_source = seed_display
    best_result = seed_result
    prev_source = seed_display
    prev_result = seed_result

    # --- Iteration loop -----------------------------------------------------
    for i in range(1, n_iterations + 1):
        log(f"\n--- iteration {i}/{n_iterations} ---")

        if i == 1:
            user_prompt = build_initial_prompt(prompt_spec, seed_display, seed_result)
        else:
            user_prompt = build_iteration_prompt(
                prompt_spec, prev_source, prev_result,
                best_source, best_result,
                history=[
                    {"iteration": h.iteration, "compile_ok": h.compile_ok,
                     "correct": h.correct, "score": h.score,
                     "is_new_best": h.is_new_best}
                    for h in history
                ],
            )

        # Save prompt for debuggability.
        (output_dir / f"{i:02d}_prompt.md").write_text(user_prompt)

        t0 = time.time()
        try:
            full_text, reasoning = await call_llm(
                SYSTEM_PROMPT, user_prompt, model,
            )
        except Exception as e:
            log(f"  LLM call failed: {e}")
            history.append(IterationRecord(
                iteration=i, role="skipped",
                compile_ok=False, correct=False, score=None,
                fail_reason=f"LLM call failed: {e}",
                source_path="",
                elapsed_s=time.time() - t0,
                is_new_best=False,
            ))
            continue
        llm_elapsed = time.time() - t0

        (output_dir / f"{i:02d}_response.md").write_text(full_text)
        if reasoning:
            (output_dir / f"{i:02d}_reasoning.md").write_text(reasoning)

        candidate_src = extract_metal_source(full_text)
        if candidate_src is None:
            log("  Could not extract a Metal source block from response.")
            history.append(IterationRecord(
                iteration=i, role="skipped",
                compile_ok=False, correct=False, score=None,
                fail_reason="no metal block in response",
                source_path=str(output_dir / f"{i:02d}_response.md"),
                elapsed_s=llm_elapsed,
                is_new_best=False,
            ))
            continue

        cand_path = output_dir / f"{i:02d}_candidate.metal"
        cand_path.write_text(candidate_src)

        # Record (do not suppress) any held-out identity the MODEL itself
        # surfaced in its response or candidate despite redaction --- the
        # knowledge-driven enumeration signal.
        model_named: list[str] = []
        if redact_held_out:
            blob = full_text + "\n" + candidate_src
            model_named = sorted(
                {p for p in task.spec.held_out_denylist if p in blob}
            )
            if model_named:
                log(f"  [knowledge] model named held-out identity unprompted: "
                    f"{model_named}")

        log(f"  generated in {llm_elapsed:.1f}s; evaluating...")
        t1 = time.time()
        result = task.evaluate_candidate(
            harness, chip, candidate_src,
            n_warmup=n_warmup, n_measure=n_measure, n_reps=n_reps,
        )
        eval_elapsed = time.time() - t1
        elapsed = llm_elapsed + eval_elapsed

        is_new_best = (
            result.score is not None
            and (best_result.score is None
                 or result.score > best_result.score)
        )

        rec = IterationRecord(
            iteration=i, role="candidate",
            compile_ok=result.compile_ok,
            correct=result.score is not None,
            score=result.score,
            fail_reason=result.fail_reason,
            source_path=str(cand_path),
            elapsed_s=elapsed,
            is_new_best=is_new_best,
            model_named_held_out=model_named,
        )
        history.append(rec)

        if not result.compile_ok:
            log(f"  compile failed: {result.compile_error}")
        elif result.score is None:
            log(f"  incorrect: {result.fail_reason}")
        else:
            log(f"  score = {result.score:.4f} "
                f"(seed = {seed_result.score:.4f}, "
                f"best = {best_result.score:.4f})")
            for s in result.size_results:
                reps = s.extra.get("gpu_seconds_per_rep")
                rep_str = ""
                if reps and len(reps) > 1:
                    lo, hi = min(reps), max(reps)
                    rep_str = f"  [reps {lo*1e3:.2f}-{hi*1e3:.2f} ms]"
                log(f"    {s.size_label:>16s}: "
                    f"{s.gpu_seconds*1e3:7.2f} ms, "
                    f"{s.achieved:7.1f} {s.achieved_unit} "
                    f"({s.fraction_of_ceiling*100:.1f}%){rep_str}")

        if is_new_best:
            log("  NEW INCUMBENT")
            best_source = candidate_src
            best_result = result

        # Result-detail JSON for this iteration.
        (output_dir / f"{i:02d}_result.json").write_text(
            json.dumps(_result_summary(result), indent=2)
        )

        prev_source = candidate_src
        prev_result = result

    # --- Save history + best ------------------------------------------------
    (output_dir / "history.json").write_text(
        json.dumps([asdict(h) for h in history], indent=2)
    )
    (output_dir / "best.metal").write_text(best_source)
    (output_dir / "best_result.json").write_text(
        json.dumps(_result_summary(best_result), indent=2)
    )

    # Iterations where the model named the held-out identity unprompted
    # (redacted arm only) --- the knowledge-driven enumeration signal.
    named_iters = sorted(
        {p for h in history for p in h.model_named_held_out}
    )
    named_iter_nums = [h.iteration for h in history if h.model_named_held_out]
    summary = {
        "task": task.spec.name,
        "model": model,
        "chip": chip.name,
        "redacted": redact_held_out,
        "model_named_held_out": named_iters,
        "model_named_held_out_iterations": named_iter_nums,
        "n_iterations": n_iterations,
        "seed_score": seed_result.score,
        "best_score": best_result.score,
        "improvement": (
            best_result.score / seed_result.score
            if best_result.score and seed_result.score
            else None
        ),
        "history": [asdict(h) for h in history],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"\n=== done. seed={seed_result.score:.4f}, "
        f"best={best_result.score:.4f}, output={output_dir} ===")
    return summary
