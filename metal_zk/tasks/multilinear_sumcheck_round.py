"""Multilinear sumcheck-round -- Z13 task.

One round of a degree-``d`` sumcheck on a product polynomial

    g(x) = f_0(x) * f_1(x) * ... * f_{d-1}(x)

where each ``f_i: {0,1}^k -> F_p`` is multilinear, stored as a length
``2^k`` table of evaluations on the Boolean hypercube. The kernel
folds the first variable and emits

  (A) the univariate ``h(X) = sum_{x' in {0,1}^{k-1}} prod_i f_i(X, x')``
      represented by its evaluations at ``X in {0, 1, ..., d}``;
  (B) the folded factor tables ``f_i_new[j] = f_i(r, j)`` for the
      next round, where ``r`` is a verifier-supplied round challenge.

Sizes (PLAN.md Section "Regime taxonomy"):

  * in-distribution: Goldilocks, ``d = 2``,
    ``2^k`` in ``{2^14, 2^16, 2^18}`` evaluations
  * held-out:        BabyBear,   ``d = 3``, ``2^k = 2^18``

The held-out probe flips on candidates that:
  - hardcode ``d = 2`` (e.g. emit only three ``h`` evaluations or
    fuse the ``t > 1`` extrapolation around a fixed unroll);
  - hardcode the Goldilocks reduction macro and ignore
    ``prime_kind``;
  - hardcode the linear-extrapolation step at ``t = 2`` instead of
    looping over ``t in [2, d]``;
  - hardcode the partial-buffer stride at ``3 * tgid`` (correct for
    ``d = 2``) instead of ``(d + 1) * tgid``.

Two-kernel pipeline:

  * ``sumcheck_round_h`` -- one threadgroup of 256 threads owns 256
    consecutive pair indices in ``[0, half)`` where ``half = 2^(k-1)``;
    it tree-reduces 256 per-pair products per ``t in [0, d]`` into
    ``d + 1`` tile partials written to ``partial[tgid * (d+1) + t]``.
  * ``sumcheck_fold``     -- one thread per output ``(poly_i, j)``;
    writes ``f_out[poly_i * half + j] = f_i^(0)[j]
    + r * (f_i^(1)[j] - f_i^(0)[j])``.

The host folds ``partial[K * (d+1)]`` into ``h_evals[d+1]`` on the CPU
(``K * (d+1)`` mod-adds, trivially sub-millisecond) and verifies
bit-exact agreement with the reference plus the sumcheck consistency
identity ``h(0) + h(1) == sum_x prod_i f_i(x)`` -- which catches
indexing bugs that a same-buggy reference comparison would miss.

Roofline
========

Per-pair algorithmic-lower-bound modmul count (the count an
optimized candidate would actually emit, used as the int-mul anchor):

  * h-compute per pair, per t:
      t in {0, 1}            : 0 extrapolation muls + (d - 1) product muls
      t in {2, ..., d}       : d extrapolation muls + (d - 1) product muls
    Summed over t:          (d - 1) * (2 * d + 1) modmuls per pair.
  * fold per pair:          d modmuls per pair (one ``r * delta`` per
                            factor, accumulated into the new table).

  Combined: ``(d - 1) * (2 * d + 1) + d = 2 * d ** 2 - 1`` modmuls
  per pair, ``(2 * d ** 2 - 1) * half`` total.
    d = 2: 7 muls / pair.
    d = 3: 17 muls / pair.

Per-round byte traffic (DRAM read + write, single-pass model):

  * read  f_in   : 8 * d * 2^k                 = 16 * d * half bytes
  * write f_out  : 8 * d * half                =  8 * d * half bytes
  * write partial: 8 * K * (d+1)               ~ negligible
  Total ``~ 24 * d * half`` bytes.

The task reports the achieved fraction against the **binding**
roofline ``max(achieved_mul / peak_int64_mul, achieved_bw / peak_bw)``
and tags ``extra["anchor"]`` so the analysis layer knows which
ceiling the fraction is against. The kernel work is fully ALU-bound
in principle (no atomics, no sequential per-element dependence), but
the table is small enough that DRAM BW binds at the larger sizes on
Apple Silicon.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import Metal

from ..reference.sumcheck import (
    PRIME_NAMES, compute_reference_cached, generate_inputs, prime_of,
)
from ..task import (
    Task, TaskSize, TaskSpec, SizeResult,
    gb_per_s, gops_per_s, register_task,
)


_SEED = (Path(__file__).resolve().parent.parent.parent
         / "seeds" / "multilinear_sumcheck_round.metal")

_TG_WIDTH = 256                             # fixed by the kernel contract


def _muls_per_pair(d: int) -> int:
    """Algorithmic-lower-bound mod-mul count per pair for one round.

    Derivation: see module docstring. (d-1)*(2d+1) for the h-compute
    + d for the fold-into-next-table = 2*d**2 - 1.
    """
    return 2 * d * d - 1


@register_task("multilinear_sumcheck_round")
class MultilinearSumcheckRoundTask(Task):
    spec = TaskSpec(
        name="multilinear_sumcheck_round",
        description=(
            "One degree-d sumcheck round on a product polynomial "
            "g(x) = f_0(x) * f_1(x) * ... * f_{d-1}(x), where each "
            "f_i: {0,1}^k -> F_p is multilinear, stored as a length "
            "2^k_log table of evaluations on the Boolean hypercube. "
            "The kernel folds the FIRST variable: it emits (A) the "
            "univariate round polynomial h(X) = sum_{x' in "
            "{0,1}^(k-1)} prod_i f_i(X, x'), represented by its d+1 "
            "evaluations h(0), h(1), ..., h(d); and (B) the folded "
            "factor tables f_i_new[j] = f_i(r, j) for j in [0, "
            "2^(k-1)), where r is the verifier-supplied round "
            "challenge in [0, p).\n\n"
            "Layout convention. The variable being folded is the most "
            "significant bit of the hypercube index, so for "
            "j in [0, 2^(k-1)) the X = 0 and X = 1 slices are\n"
            "  f_i^(0)[j] = f_in[i * 2^k_log + j]\n"
            "  f_i^(1)[j] = f_in[i * 2^k_log + j + 2^(k-1)]\n"
            "The multilinear extension along the first variable, "
            "evaluated at any X in F_p, is the unique affine "
            "interpolant\n"
            "  f_i(X, j) = f_i^(0)[j] + X * (f_i^(1)[j] - f_i^(0)[j])   (mod p)\n"
            "so the kernel must produce, in one round,\n"
            "  h(t)       = sum_{j in [0, 2^(k-1))} prod_i f_i(t, j)\n"
            "               for t in {0, 1, ..., d}\n"
            "  f_i_new[j] = f_i(r, j)\n"
            "               for i in [0, d) and j in [0, 2^(k-1)).\n\n"
            "Two-kernel pipeline (host issues both in ONE compute "
            "command encoder; the serial encoder gives an implicit "
            "barrier so kernel B observes A's writes):\n"
            "  Dispatch 1 (sumcheck_round_h): each threadgroup owns "
            "    256 consecutive pair indices in [0, half) where "
            "    half = 2^(k_log - 1). For each pair index j the "
            "    thread contributes the d+1 per-pair products "
            "    prod_i f_i(t, j); the threadgroup cooperatively "
            "    reduces 256 contributions per t into one tile sum "
            "    and writes d+1 contiguous ulongs to "
            "    partial[tgid * (d+1) + t]. Threads with gid >= "
            "    half contribute 0 (additive identity for the sum).\n"
            "  Dispatch 2 (sumcheck_fold): one thread per output "
            "    (poly_i, j); writes one folded coefficient to "
            "    f_out[poly_i * half + j]. Guard against gid >= d * "
            "    half (the grid is rounded up to a multiple of the TG "
            "    width).\n\n"
            "The host then sums partial[0..K-1] per t on the CPU "
            "(K = ceil(half / 256), ~1 KB total -- intentionally "
            "untimed) to obtain h_evals[0..d+1], and cross-checks the "
            "sumcheck consistency identity h(0) + h(1) == sum_x "
            "prod_i f_i(x). A candidate whose h_evals matches a "
            "same-buggy reference but indexes the linear extension "
            "the wrong way silently fails this identity.\n\n"
            "Field selection (constant prime_kind):\n"
            "  0 = Goldilocks   p = 2^64 - 2^32 + 1\n"
            "  1 = BabyBear     p = 2^31 - 2^27 + 1 = 2013265921\n"
            "Both reductions, the per-pair t-loop, and the "
            "threadgroup geometry must dispatch on the RUNTIME values "
            "of prime_kind, d_deg, and k_log. Baking any of them in "
            "as a compile-time constant -- a specific reduction "
            "macro, a fixed unroll over t, a hardcoded buffer "
            "stride, ... -- violates the kernel contract.\n\n"
            "All field elements (f_in, partial, f_out, r) are "
            "canonical uint64 in [0, p); a non-canonical output is "
            "treated as a correctness failure even if its residue "
            "class matches the reference."
        ),
        kernel_signatures=(
            "kernel void sumcheck_round_h(\n"
            "    device const ulong *f_in       [[buffer(0)]],\n"
            "    device       ulong *partial    [[buffer(1)]],\n"
            "    constant uint      &k_log      [[buffer(2)]],\n"
            "    constant uint      &d_deg      [[buffer(3)]],\n"
            "    constant uint      &prime_kind [[buffer(4)]],\n"
            "    uint gid  [[thread_position_in_grid]],\n"
            "    uint tid  [[thread_position_in_threadgroup]],\n"
            "    uint tgid [[threadgroup_position_in_grid]]);\n"
            "\n"
            "kernel void sumcheck_fold(\n"
            "    device const ulong *f_in       [[buffer(0)]],\n"
            "    device       ulong *f_out      [[buffer(1)]],\n"
            "    constant ulong     &r          [[buffer(2)]],\n"
            "    constant uint      &k_log      [[buffer(3)]],\n"
            "    constant uint      &d_deg      [[buffer(4)]],\n"
            "    constant uint      &prime_kind [[buffer(5)]],\n"
            "    uint gid [[thread_position_in_grid]]);\n"
            "\n"
            "Dispatch geometry (host-fixed):\n"
            "  sumcheck_round_h:\n"
            "    threadsPerGrid        = (K * 256, 1, 1)   K = ceil(half / 256)\n"
            "    threadsPerThreadgroup = (256, 1, 1)        // FIXED at TG_WIDTH=256\n"
            "  sumcheck_fold:\n"
            "    threadsPerGrid        = (d * half rounded up to TG width, 1, 1)\n"
            "    threadsPerThreadgroup = (min(d * half, 256), 1, 1)\n"
            "\n"
            "The 256-wide threadgroup is part of the host-kernel "
            "contract for sumcheck_round_h: K = ceil(half / 256) is "
            "baked into the host-side partial[] allocation, so the "
            "kernel must emit exactly one (d+1)-element tile sum per "
            "256 consecutive pair indices."
        ),
        kernel_names=["sumcheck_round_h", "sumcheck_fold"],
        seed_path=_SEED,
        sizes=[
            TaskSize("gold_k14_d2",
                     {"k_log": 14, "d": 2, "prime_kind": 0}),
            TaskSize("gold_k16_d2",
                     {"k_log": 16, "d": 2, "prime_kind": 0}),
            TaskSize("gold_k18_d2",
                     {"k_log": 18, "d": 2, "prime_kind": 0}),
        ],
        held_out_sizes=[
            TaskSize("bb_k18_d3",
                     {"k_log": 18, "d": 3, "prime_kind": 1}),
        ],
    )

    def evaluate_size(self, harness, pipelines, size, chip, n_warmup, n_measure):
        k_log = int(size.params["k_log"])
        d = int(size.params["d"])
        prime_kind = int(size.params["prime_kind"])
        if k_log < 1:
            raise ValueError(f"k_log={k_log} must be >= 1")
        if d < 1 or d > 3:
            raise ValueError(
                f"d={d} out of supported range [1, 3]; the seed kernel "
                f"caps MAX_D at 3"
            )

        prime = prime_of(prime_kind)
        n = 1 << k_log
        half = n >> 1
        K = (half + _TG_WIDTH - 1) // _TG_WIDTH

        # Deterministic inputs + cached reference. Seed salted with
        # (k_log, d, prime_kind) so every size has an independent
        # cache entry.
        gen_seed = (
            0x5C0D_E_BEEF                      # noqa: E501  (mnemonic constant)
            + k_log * 1_000_003
            + d * 0x9E37_79B9
            + prime_kind * 0xBF58_476D
        ) & ((1 << 64) - 1)
        f_tables, r = generate_inputs(k_log, d, prime_kind, gen_seed)
        ref = compute_reference_cached(f_tables, r, prime_kind)
        claim_ref      = int(ref["claim"])
        h_evals_ref    = ref["h_evals"]            # uint64[d+1]
        f_out_ref      = ref["f_out"]              # uint64[d, half]

        # ---- Buffers ----
        # f_in is the d factor tables concatenated in row-major:
        # f_in[i * n + j] is the j-th eval of f_i.
        f_in_flat = np.ascontiguousarray(f_tables.reshape(d * n), dtype=np.uint64)
        b_f_in    = harness.buf_from_np(f_in_flat)
        b_partial = harness.buf_zeros(K * (d + 1) * 8)        # ulong[K * (d+1)]
        b_f_out   = harness.buf_zeros(d * half * 8)           # ulong[d * half]
        b_r       = harness.buf_scalar(int(r),     np.uint64)
        b_k_log   = harness.buf_scalar(k_log,      np.uint32)
        b_d       = harness.buf_scalar(d,          np.uint32)
        b_pk      = harness.buf_scalar(prime_kind, np.uint32)

        pso_h    = pipelines["sumcheck_round_h"]
        pso_fold = pipelines["sumcheck_fold"]

        # Kernel A geometry: TG_WIDTH is fixed at 256 by the kernel
        # contract. Sanity-check that the candidate's compiled pipeline
        # supports a 256-wide threadgroup.
        max_tg_h = int(pso_h.maxTotalThreadsPerThreadgroup())
        if max_tg_h < _TG_WIDTH:
            raise RuntimeError(
                f"sumcheck_round_h pipeline supports only {max_tg_h}-"
                f"wide threadgroups; the host requires {_TG_WIDTH} "
                f"(kernel contract)."
            )
        grid_h = K * _TG_WIDTH

        # Kernel B geometry: one thread per output element.
        max_tg_f = int(pso_fold.maxTotalThreadsPerThreadgroup())
        tew_f    = int(pso_fold.threadExecutionWidth())
        out_n    = d * half
        tg_f     = min(max_tg_f, max(tew_f, 256))
        tg_f     = max(1, min(tg_f, out_n))
        grid_f   = ((out_n + tg_f - 1) // tg_f) * tg_f

        view_partial = harness.np_view(b_partial, np.uint64, K * (d + 1))
        view_f_out   = harness.np_view(b_f_out,   np.uint64, d * half)

        def reset():
            view_partial[:] = 0
            view_f_out[:]   = 0

        def dispatch(enc):
            # Dispatch 1: per-tile partial sums of h(t) for t in [0, d].
            enc.setComputePipelineState_(pso_h)
            enc.setBuffer_offset_atIndex_(b_f_in,    0, 0)
            enc.setBuffer_offset_atIndex_(b_partial, 0, 1)
            enc.setBuffer_offset_atIndex_(b_k_log,   0, 2)
            enc.setBuffer_offset_atIndex_(b_d,       0, 3)
            enc.setBuffer_offset_atIndex_(b_pk,      0, 4)
            enc.dispatchThreads_threadsPerThreadgroup_(
                Metal.MTLSizeMake(grid_h, 1, 1),
                Metal.MTLSizeMake(_TG_WIDTH, 1, 1),
            )

            # Dispatch 2: fold each factor table along the first
            # variable. Independent of dispatch 1 (no read-after-write
            # dependency); the serial encoder still orders them.
            enc.setComputePipelineState_(pso_fold)
            enc.setBuffer_offset_atIndex_(b_f_in,  0, 0)
            enc.setBuffer_offset_atIndex_(b_f_out, 0, 1)
            enc.setBuffer_offset_atIndex_(b_r,     0, 2)
            enc.setBuffer_offset_atIndex_(b_k_log, 0, 3)
            enc.setBuffer_offset_atIndex_(b_d,     0, 4)
            enc.setBuffer_offset_atIndex_(b_pk,    0, 5)
            enc.dispatchThreads_threadsPerThreadgroup_(
                Metal.MTLSizeMake(grid_f, 1, 1),
                Metal.MTLSizeMake(tg_f, 1, 1),
            )

        for _ in range(n_warmup):
            reset()
            harness.time_dispatch(dispatch)
        samples = []
        for _ in range(n_measure):
            reset()
            samples.append(harness.time_dispatch(dispatch))
        gpu_s = float(np.median(samples))

        # Final correctness pass.
        reset()
        harness.time_dispatch(dispatch)
        got_partial = view_partial.copy()
        got_f_out   = view_f_out.copy().reshape(d, half)

        # ---- Host-side fold of partials into h_evals[d+1] ----
        # K * (d+1) mod-adds in Python int.
        got_h = [0] * (d + 1)
        for tgid in range(K):
            for t in range(d + 1):
                got_h[t] = (got_h[t]
                            + int(got_partial[tgid * (d + 1) + t])) % prime
        got_h_np = np.array(got_h, dtype=np.uint64)

        # ---- Bit-exact comparisons ----
        mismatches_h    = int(np.sum(got_h_np != h_evals_ref))
        mismatches_fold = int(np.sum(got_f_out != f_out_ref))

        # ---- Sumcheck consistency identity ----
        # h(0) + h(1) should equal the pre-round claim. This is a
        # mathematical invariant of correctly computed h_evals; we
        # surface it so an LLM gets a clean diagnostic when a kernel
        # produces a bit-exact-but-internally-incoherent answer (e.g.
        # mis-routed t-index, swapped low/high halves -- the bit-exact
        # h check might still pass if both reference and kernel share
        # the same indexing bug, but h(0)+h(1) won't match the claim).
        h0_plus_h1 = (got_h[0] + got_h[1]) % prime
        consistency_ok = (h0_plus_h1 == claim_ref)
        consistency_err = 0 if consistency_ok else 1

        # ---- Canonicality (any output >= p is a fault) ----
        non_canonical_h    = int(np.sum(got_h_np   >= np.uint64(prime)))
        non_canonical_fold = int(np.sum(got_f_out  >= np.uint64(prime)))
        non_canonical = non_canonical_h + non_canonical_fold

        mismatches = mismatches_h + mismatches_fold + consistency_err
        correct = (mismatches == 0 and non_canonical == 0)

        # ---- Roofline ----
        muls_per_pair = _muls_per_pair(d)
        muls_total = muls_per_pair * half
        achieved_mul = gops_per_s(muls_total, gpu_s)
        ceiling_mul  = float(chip.peak_int64_mul_gops)

        # Single-pass BW model: read d * n ulongs, write d * half
        # ulongs + K * (d+1) ulongs partials (the latter is tiny but
        # included so a candidate that grows the partial buffer is
        # honestly accounted).
        bytes_total = float(
            8 * d * n               # read f_in
            + 8 * d * half          # write f_out
            + 8 * K * (d + 1)       # write partial
        )
        achieved_bw = gb_per_s(bytes_total, gpu_s)
        ceiling_bw  = float(chip.peak_bw_gb_s)

        frac_mul = achieved_mul / ceiling_mul if ceiling_mul > 0 else 0.0
        frac_bw  = achieved_bw  / ceiling_bw  if ceiling_bw  > 0 else 0.0
        if frac_mul >= frac_bw:
            achieved, achieved_unit = achieved_mul, "Gmodmul/s (int64)"
            ceiling,  ceiling_unit  = ceiling_mul,  "Gops/s (int64 mul, est)"
            fraction = frac_mul
            anchor = "int64_mul"
        else:
            achieved, achieved_unit = achieved_bw,  "GB/s"
            ceiling,  ceiling_unit  = ceiling_bw,   "GB/s"
            fraction = frac_bw
            anchor = "dram_bw"

        return SizeResult(
            size_label=size.label,
            correct=correct,
            error_value=mismatches + non_canonical,
            error_kind="bit_exact",
            gpu_seconds=gpu_s,
            achieved=achieved,
            achieved_unit=achieved_unit,
            ceiling=ceiling,
            ceiling_unit=ceiling_unit,
            fraction_of_ceiling=fraction,
            extra={
                "k_log": k_log,
                "d": d,
                "prime_kind": prime_kind,
                "prime_name": PRIME_NAMES[prime_kind],
                "n": n,
                "half": half,
                "K_tiles": K,
                "r": int(r),
                "claim": claim_ref,
                "mismatches_h": mismatches_h,
                "mismatches_fold": mismatches_fold,
                "consistency_err": consistency_err,
                "non_canonical_h": non_canonical_h,
                "non_canonical_fold": non_canonical_fold,
                "muls_per_pair": muls_per_pair,
                "muls_total": muls_total,
                "bytes_total": bytes_total,
                "achieved_mul_gops": achieved_mul,
                "achieved_bw_gb_s": achieved_bw,
                "anchor": anchor,
            },
        )
