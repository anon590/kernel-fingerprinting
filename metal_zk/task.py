"""Task abstraction for the Metal-ZK kernel benchmark.

A ``Task`` packages everything needed to (a) describe a problem to an LLM,
(b) compile a candidate ``.metal`` source, (c) dispatch the kernels at
multiple problem sizes, (d) verify correctness against a CPU reference,
(e) compute achieved throughput, and (f) score the candidate as a
fraction of the architectural roofline.

Differences from metal-kernels:
- ZK tasks default to bit-exact correctness. ``SizeResult.error_kind`` is
  expected to be ``"bit_exact"`` and ``error_value`` is the number of
  mismatched output elements (0 on pass). The score-zeroing logic in
  :func:`evaluate_candidate` is unchanged: any size that returns
  ``correct=False`` zeros the candidate.
- The roofline anchor is declared per-task in ``SizeResult.ceiling_unit``
  ("GB/s" for BW-bound NTT at large N; "Gops/s (int64 mul)" for
  compute-bound Poseidon2 / Montgomery). The fitness gmean is unchanged.
- Determinism gate (PLAN.md Methodology section 3): every ZK task gets a
  free determinism check because the reference is bit-exact. We don't
  yet vary threadgroup counts to probe it, but ``error_kind="bit_exact"``
  + ``error_value=0`` is the contractual signal.
"""

from __future__ import annotations

import math
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from .harness import MetalHarness
from .hardware import ChipSpec
from .redact import Redaction


@dataclass
class TaskSize:
    """One problem size for a task."""
    label: str
    params: dict[str, Any]


@dataclass
class TaskSpec:
    name: str
    description: str
    kernel_signatures: str          # text block shown to the LLM
    kernel_names: list[str]         # functions to extract from compiled lib
    seed_path: Path
    sizes: list[TaskSize]
    held_out_sizes: list[TaskSize] = field(default_factory=list)
    # Controlled grade-C experiment (see metal_zk/redact.py): substrings
    # that disclose the held-out configuration, and the neutral text to
    # swap in under --redact-held-out. ``held_out_denylist`` lists phrases
    # that must NOT survive in a redacted prompt (asserted at run time).
    redactions: list[Redaction] = field(default_factory=list)
    held_out_denylist: list[str] = field(default_factory=list)


@dataclass
class SizeResult:
    size_label: str
    correct: bool
    error_value: float              # task-specific error metric (0 if bit-exact)
    error_kind: str                 # "bit_exact" for ZK; "max_abs" etc. allowed
    gpu_seconds: float
    achieved: float                 # task-declared throughput
    achieved_unit: str
    ceiling: float
    ceiling_unit: str
    fraction_of_ceiling: float
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateResult:
    compile_ok: bool
    compile_error: str | None
    pipeline_error: str | None
    size_results: list[SizeResult]
    score: float | None             # None if any size failed correctness
    fail_reason: str | None         # human-readable hint for the LLM


class Task(ABC):
    """Base class for Metal-ZK benchmark tasks."""

    spec: TaskSpec

    @abstractmethod
    def evaluate_size(
        self,
        harness: MetalHarness,
        pipelines: dict[str, object],
        size: TaskSize,
        chip: ChipSpec,
        n_warmup: int,
        n_measure: int,
    ) -> SizeResult:
        ...

    def evaluate_candidate(
        self,
        harness: MetalHarness,
        chip: ChipSpec,
        source: str,
        n_warmup: int = 3,
        n_measure: int = 10,
        n_reps: int = 3,
        sizes: list[TaskSize] | None = None,
    ) -> CandidateResult:
        """Compile + run + verify + score a candidate ``.metal`` source.

        Each size is evaluated ``n_reps`` independent times; the median
        rep (by ``gpu_seconds``) is reported. Independent reps dampen
        cache-boundary noise, where the same kernel can land at
        substantially different fractions of peak depending on whether
        the working set caught SLC residency from a prior run. With
        ``n_reps=1`` the behavior collapses to the original single-shot
        path.
        """
        sizes = sizes if sizes is not None else self.spec.sizes
        n_reps = max(1, int(n_reps))
        cr = harness.compile(source)
        if cr.error is not None:
            return CandidateResult(
                compile_ok=False, compile_error=cr.error,
                pipeline_error=None, size_results=[], score=None,
                fail_reason=f"compile error: {cr.error}",
            )
        pipelines, perr = harness.make_pipelines(cr.library, self.spec.kernel_names)
        if perr is not None:
            return CandidateResult(
                compile_ok=True, compile_error=None,
                pipeline_error=perr, size_results=[], score=None,
                fail_reason=f"pipeline error: {perr}",
            )

        size_results: list[SizeResult] = []
        for size in sizes:
            rep_results: list[SizeResult] = []
            for _rep in range(n_reps):
                try:
                    res = self.evaluate_size(
                        harness, pipelines, size, chip, n_warmup, n_measure,
                    )
                except Exception as e:
                    size_results.append(SizeResult(
                        size_label=size.label, correct=False,
                        error_value=float("inf"), error_kind="exception",
                        gpu_seconds=0.0, achieved=0.0, achieved_unit="",
                        ceiling=0.0, ceiling_unit="",
                        fraction_of_ceiling=0.0,
                        extra={"exception": str(e)},
                    ))
                    return CandidateResult(
                        compile_ok=True, compile_error=None,
                        pipeline_error=None,
                        size_results=size_results,
                        score=None,
                        fail_reason=f"runtime error at size {size.label}: {e}",
                    )
                rep_results.append(res)
                if not res.correct:
                    # No point doing more reps for a rejected candidate.
                    break

            combined = _combine_reps(rep_results)
            size_results.append(combined)
            if not combined.correct:
                return CandidateResult(
                    compile_ok=True, compile_error=None,
                    pipeline_error=None,
                    size_results=size_results,
                    score=None,
                    fail_reason=(
                        f"correctness failed at size {size.label}: "
                        f"{combined.error_kind}={combined.error_value}"
                    ),
                )

        # All sizes correct -> gmean of fraction_of_ceiling.
        fractions = [r.fraction_of_ceiling for r in size_results]
        log_sum = sum(math.log(max(f, 1e-12)) for f in fractions)
        score = math.exp(log_sum / max(len(fractions), 1))
        return CandidateResult(
            compile_ok=True, compile_error=None, pipeline_error=None,
            size_results=size_results, score=score, fail_reason=None,
        )


# ------------------------------------------------------------------
# Multi-rep combiner
# ------------------------------------------------------------------

def _combine_reps(reps: list[SizeResult]) -> SizeResult:
    """Combine N independent SizeResult reps into one by picking the
    median rep (by ``gpu_seconds``) and carrying its anchor/unit
    decisions through. ``correct`` is AND across reps, ``error_value``
    is max. The per-rep timings are recorded in ``extra``.

    Picking the whole median rep (rather than recomputing achieved
    from the median time) keeps the anchor choice consistent — at
    cache-boundary sizes the BW vs int-mul anchor can flip between
    reps, and we want the anchor that goes with the reported time.
    """
    if not reps:
        raise ValueError("no rep results to combine")
    if len(reps) == 1:
        return reps[0]
    sorted_reps = sorted(reps, key=lambda r: r.gpu_seconds)
    median_rep = sorted_reps[len(sorted_reps) // 2]
    times = [r.gpu_seconds for r in reps]
    extra = dict(median_rep.extra)
    extra["gpu_seconds_per_rep"] = times
    extra["gpu_seconds_median"] = float(statistics.median(times))
    extra["n_reps"] = len(reps)
    return replace(
        median_rep,
        correct=all(r.correct for r in reps),
        error_value=max(r.error_value for r in reps),
        extra=extra,
    )


# Convenient throughput formulas
def gb_per_s(bytes_moved: float, seconds: float) -> float:
    return bytes_moved / seconds / 1e9


def gflops(flops: float, seconds: float) -> float:
    return flops / seconds / 1e9


def gops_per_s(ops: float, seconds: float) -> float:
    """Generic Gops/s — for integer kernels we count modmuls or rotates,
    not FLOPs."""
    return ops / seconds / 1e9


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

_TASK_REGISTRY: dict[str, type[Task]] = {}


def register_task(name: str):
    """Decorator: register a Task subclass under ``name``."""
    def decorator(cls):
        _TASK_REGISTRY[name] = cls
        return cls
    return decorator


def get_task(name: str) -> Task:
    if name not in _TASK_REGISTRY:
        raise KeyError(
            f"unknown task: {name!r}. Available: {sorted(_TASK_REGISTRY)}"
        )
    return _TASK_REGISTRY[name]()


def list_tasks() -> list[str]:
    return sorted(_TASK_REGISTRY)
