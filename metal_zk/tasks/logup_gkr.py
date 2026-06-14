"""LogUp lookup-argument running product -- Z7 task.

Batched LogUp core (Haebock 2022; back-end of Jolt / Lasso / Plonkish v2).
Given a table ``T[M]`` and a witness column ``w[N]`` where each
``w_i := T[witness_idx[i]]``, the task computes:

  (1) multiplicities ``m[j] = #{ i : witness_idx[i] == j }``
      for ``j`` in ``[0, M)``,
  (2) the running product

        P = prod_{i=0..N-1} 1/(alpha - w_i)
           * prod_{j=0..M-1} m_j / (alpha - T_j)            (mod p)

      where ``alpha`` is a verifier challenge (host-fixed so that
      ``alpha`` is outside the set of table values -- no zero
      denominators).

Sizes (PLAN.md S Regime taxonomy):

  * in-distribution:  Goldilocks, M in {2^12, 2^16, 2^20}, N = 2*M
  * held-out:         BabyBear,   M = 2^18,                N = 2*M

The 2x ratio between witness count and table size makes multiplicities
non-trivial (each table entry receives one guaranteed hit from the
permutation prefix of ``witness_idx`` plus a Poisson-1 tail from the
uniform suffix). The held-out probe flips on candidates that hardcode
the Goldilocks reduction macro, the Goldilocks 64-bit limb size, or
the implicit assumption that ``alpha`` fits in 31 bits.

Roofline:

  Per-element algorithmic cost (canonical lower bound used as the
  modmul anchor):
    - 3 modmuls / element for batched modular inversion;
    - 1 modmul / element for the numerator ``num_k * inv_k``;
    - 1 modmul / element to fold into the running product.
  Total ~ 5 * (N + M) Goldilocks (resp. BabyBear) modmuls.

  Minimal bytes the algorithm must move through DRAM:
    - read table[M]                 : 8 * M
    - read witness_idx[N]           : 4 * N
    - write multiplicities[M]       : 4 * M
    - write partial[K]              : 8 * K   (K = ceil((N+M) / 256))
  Total ~ 12 M + 4 N + small.

  The task reports the achieved fraction against the **binding**
  roofline ``max(achieved_mul / peak_int64_mul, achieved_bw / peak_bw)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import Metal

from ..harness import MetalHarness
from ..hardware import ChipSpec
from ..reference.logup import (
    PRIME_NAMES, compute_reference_cached, generate_inputs, prime_of,
)
from ..task import (
    Task, TaskSize, TaskSpec, SizeResult,
    gb_per_s, gops_per_s, register_task,
)


_SEED = Path(__file__).resolve().parent.parent.parent / "seeds" / "logup_gkr.metal"
_TG_WIDTH = 256                                 # fixed by the kernel contract


@register_task("logup_gkr")
class LogUpGKRTask(Task):
    spec = TaskSpec(
        name="logup_gkr",
        description=(
            "Batched LogUp lookup-argument core (Haebock 2022; back-end "
            "of Jolt / Lasso / Plonkish v2). Given a table T[M] and a "
            "witness column w[N] where each w_i := T[witness_idx[i]], "
            "compute (1) multiplicities m[j] = #{ i : witness_idx[i] "
            "== j } for j in [0, M), and (2) the running product\n"
            "  P = prod_{i=0..N-1} 1/(alpha - w_i)\n"
            "     * prod_{j=0..M-1} m_j / (alpha - T_j)   (mod p)\n"
            "where alpha is a verifier challenge. The host fixes alpha "
            "so that alpha is outside the set of table values -- no "
            "zero denominators arise.\n\n"
            "Combined fingerprint stream of length N + M:\n"
            "  k <  N:   x_k = T[witness_idx[k]],  num_k = 1\n"
            "  k >= N:   x_k = T[k - N],            num_k = m[k - N]\n\n"
            "The host issues two dispatches in a single compute command "
            "encoder. Their serial ordering provides the implicit "
            "barrier so the second dispatch sees the first's atomic "
            "writes:\n"
            "  Dispatch 1 (logup_count_mult): one thread per witness "
            "    row; atomically increments multiplicities[witness_idx[i]].\n"
            "  Dispatch 2 (logup_partial_product): each threadgroup "
            "    of TG_WIDTH = 256 threads owns 256 consecutive indices "
            "    in [0, N+M). Each thread computes num_k * 1/(alpha - "
            "    x_k); threadgroup-cooperatively reduces the 256 terms "
            "    into one tile product written to partial[tgid]. Threads "
            "    with k >= N+M contribute the multiplicative identity "
            "    (1). The host then multiplies partial[0..K-1] (K = "
            "    ceil((N+M)/256)) on the CPU to obtain the final running "
            "    product (the sub-millisecond host fold is intentionally "
            "    untimed).\n\n"
            "Field selection (constant prime_kind):\n"
            "  0 = Goldilocks  p = 2^64 - 2^32 + 1\n"
            "  1 = BabyBear    p = 2^31 - 2^27 + 1 = 2013265921\n"
            "Both reductions are runtime-dispatched on prime_kind; a "
            "candidate that hardcodes the Goldilocks reduction macro, "
            "or assumes 64-bit limbs are needed, silently fails the "
            "held-out BabyBear probe.\n\n"
            "All field elements (table, alpha, partial[]) are canonical "
            "uint64 in [0, p); a non-canonical output element is a "
            "correctness failure even if its residue class matches. "
            "Multiplicities are canonical uint32 counts (promoted to "
            "ulong only when used as the numerator)."
        ),
        kernel_signatures=(
            "kernel void logup_count_mult(\n"
            "    device const uint  *witness_idx    [[buffer(0)]],\n"
            "    device atomic_uint *multiplicities [[buffer(1)]],\n"
            "    constant uint      &N              [[buffer(2)]],\n"
            "    uint i [[thread_position_in_grid]]);\n"
            "\n"
            "kernel void logup_partial_product(\n"
            "    device const ulong *table          [[buffer(0)]],\n"
            "    device const uint  *witness_idx    [[buffer(1)]],\n"
            "    device const uint  *multiplicities [[buffer(2)]],\n"
            "    device       ulong *partial        [[buffer(3)]],\n"
            "    constant uint      &N              [[buffer(4)]],\n"
            "    constant uint      &M              [[buffer(5)]],\n"
            "    constant uint      &prime_kind     [[buffer(6)]],\n"
            "    constant ulong     &alpha          [[buffer(7)]],\n"
            "    uint gid  [[thread_position_in_grid]],\n"
            "    uint tid  [[thread_position_in_threadgroup]],\n"
            "    uint tgid [[threadgroup_position_in_grid]]);\n"
            "\n"
            "Dispatch geometry (host-fixed):\n"
            "  logup_count_mult:\n"
            "    threadsPerGrid        = (N rounded up to TG width, 1, 1)\n"
            "    threadsPerThreadgroup = (min(N, 256), 1, 1)\n"
            "  logup_partial_product:\n"
            "    threadsPerGrid        = (K * 256, 1, 1)   K = ceil((N+M)/256)\n"
            "    threadsPerThreadgroup = (256, 1, 1)        // FIXED at TG_WIDTH=256\n"
            "\n"
            "The 256-wide threadgroup is part of the host-kernel contract "
            "for logup_partial_product: K = ceil((N+M) / 256) is baked into "
            "the host-side partial[] allocation, so the kernel must emit "
            "exactly one tile-product per 256 consecutive indices. The "
            "host pre-zeroes multiplicities[M] before every dispatch. "
            "The two dispatches share a single MTLComputeCommandEncoder; "
            "the implicit cross-dispatch barrier in serial mode gives "
            "logup_partial_product a coherent view of "
            "multiplicities written by logup_count_mult."
        ),
        kernel_names=["logup_count_mult", "logup_partial_product"],
        seed_path=_SEED,
        sizes=[
            TaskSize("gold_M4K",  {"M": 1 << 12, "N": 1 << 13, "prime_kind": 0}),
            TaskSize("gold_M64K", {"M": 1 << 16, "N": 1 << 17, "prime_kind": 0}),
            TaskSize("gold_M1M",  {"M": 1 << 20, "N": 1 << 21, "prime_kind": 0}),
        ],
        held_out_sizes=[
            TaskSize("bb_M256K",  {"M": 1 << 18, "N": 1 << 19, "prime_kind": 1}),
        ],
    )

    def evaluate_size(self, harness, pipelines, size, chip, n_warmup, n_measure):
        M = int(size.params["M"])
        N = int(size.params["N"])
        prime_kind = int(size.params["prime_kind"])
        prime = prime_of(prime_kind)
        total = N + M
        K = (total + _TG_WIDTH - 1) // _TG_WIDTH

        # Deterministic inputs + cached reference. The seed is salted
        # with (M, N, prime_kind) so each size has an independent cache
        # entry.
        gen_seed = (
            0xC0DECAFE
            + N * 31
            + M * 1009
            + prime_kind * 5_000_011
        )
        table, witness_idx, alpha = generate_inputs(M, N, prime_kind, gen_seed)
        mult_ref, prod_ref = compute_reference_cached(
            table, witness_idx, alpha, prime_kind,
        )

        # Buffers.
        b_table = harness.buf_from_np(table.astype(np.uint64))
        b_widx  = harness.buf_from_np(witness_idx.astype(np.uint32))
        b_mult  = harness.buf_zeros(M * 4)                   # uint32[M]
        b_partial = harness.buf_zeros(K * 8)                 # uint64[K]
        b_N     = harness.buf_scalar(N, np.uint32)
        b_M     = harness.buf_scalar(M, np.uint32)
        b_pk    = harness.buf_scalar(prime_kind, np.uint32)
        b_alpha = harness.buf_scalar(alpha, np.uint64)

        pso_count = pipelines["logup_count_mult"]
        pso_prod  = pipelines["logup_partial_product"]

        # Kernel A geometry.
        max_tg_a = int(pso_count.maxTotalThreadsPerThreadgroup())
        tew_a    = int(pso_count.threadExecutionWidth())
        tg_a     = min(max_tg_a, max(tew_a, 256))
        tg_a     = max(1, min(tg_a, N))
        grid_a   = ((N + tg_a - 1) // tg_a) * tg_a

        # Kernel B geometry: TG_WIDTH is fixed at 256 by the kernel
        # contract. Sanity-check that the candidate's compiled pipeline
        # supports a 256-wide threadgroup.
        max_tg_b = int(pso_prod.maxTotalThreadsPerThreadgroup())
        if max_tg_b < _TG_WIDTH:
            raise RuntimeError(
                f"logup_partial_product pipeline supports only "
                f"{max_tg_b}-wide threadgroups; the host requires "
                f"{_TG_WIDTH} (kernel contract)."
            )
        grid_b = K * _TG_WIDTH

        view_mult    = harness.np_view(b_mult,    np.uint32, M)
        view_partial = harness.np_view(b_partial, np.uint64, K)

        def reset():
            view_mult[:]    = 0
            view_partial[:] = 0

        def dispatch(enc):
            # Dispatch 1: count multiplicities.
            enc.setComputePipelineState_(pso_count)
            enc.setBuffer_offset_atIndex_(b_widx, 0, 0)
            enc.setBuffer_offset_atIndex_(b_mult, 0, 1)
            enc.setBuffer_offset_atIndex_(b_N,    0, 2)
            enc.dispatchThreads_threadsPerThreadgroup_(
                Metal.MTLSizeMake(grid_a, 1, 1),
                Metal.MTLSizeMake(tg_a, 1, 1),
            )

            # Dispatch 2: per-tile fingerprint inversion + product
            # reduction. Serial-mode encoder gives an implicit barrier
            # against the atomic writes above.
            enc.setComputePipelineState_(pso_prod)
            enc.setBuffer_offset_atIndex_(b_table,   0, 0)
            enc.setBuffer_offset_atIndex_(b_widx,    0, 1)
            enc.setBuffer_offset_atIndex_(b_mult,    0, 2)
            enc.setBuffer_offset_atIndex_(b_partial, 0, 3)
            enc.setBuffer_offset_atIndex_(b_N,       0, 4)
            enc.setBuffer_offset_atIndex_(b_M,       0, 5)
            enc.setBuffer_offset_atIndex_(b_pk,      0, 6)
            enc.setBuffer_offset_atIndex_(b_alpha,   0, 7)
            enc.dispatchThreads_threadsPerThreadgroup_(
                Metal.MTLSizeMake(grid_b, 1, 1),
                Metal.MTLSizeMake(_TG_WIDTH, 1, 1),
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
        got_mult    = view_mult.copy()
        got_partial = view_partial.copy()

        # Host-side fold of partial[K] into the final running product.
        prod_got = 1
        for v in got_partial.tolist():
            prod_got = (prod_got * int(v)) % prime

        # Non-canonical detection: any partial > p is a fault signal.
        non_canonical = int(np.sum(got_partial >= np.uint64(prime)))

        mismatches_mult = int(np.sum(got_mult != mult_ref))
        mismatches_prod = int(prod_got != prod_ref)
        # Combined error metric for the LLM feedback path: mults +
        # whether the product matched. Both must be zero for correct.
        mismatches = mismatches_mult + mismatches_prod
        correct = (mismatches == 0)

        # ---- Roofline ----
        # Modmul anchor: 3 muls/elem batched inversion + 1 num*inv +
        # 1 fold-into-running-product = 5 * (N + M).
        muls_total = 5 * total
        achieved_mul = gops_per_s(muls_total, gpu_s)
        ceiling_mul  = float(chip.peak_int64_mul_gops)

        # Minimal-bytes model: table read + witness_idx read + mult
        # write + partial write. Field-element load is 8B; uint32 is 4B.
        bytes_total = float(8 * M + 4 * N + 4 * M + 8 * K)
        achieved_bw  = gb_per_s(bytes_total, gpu_s)
        ceiling_bw   = float(chip.peak_bw_gb_s)

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
            error_value=mismatches,
            error_kind="bit_exact",
            gpu_seconds=gpu_s,
            achieved=achieved,
            achieved_unit=achieved_unit,
            ceiling=ceiling,
            ceiling_unit=ceiling_unit,
            fraction_of_ceiling=fraction,
            extra={
                "M": M, "N": N, "total": total, "K": K,
                "prime_kind": prime_kind,
                "prime_name": PRIME_NAMES[prime_kind],
                "alpha": int(alpha),
                "non_canonical": non_canonical,
                "mismatches_mult": mismatches_mult,
                "mismatches_prod": mismatches_prod,
                "muls_total": muls_total,
                "bytes_total": bytes_total,
                "achieved_mul_gops": achieved_mul,
                "achieved_bw_gb_s": achieved_bw,
                "anchor": anchor,
            },
        )
