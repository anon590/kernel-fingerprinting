"""Goldilocks NTT — Z2 task.

One-stage-per-dispatch Stockham radix-2 NTT over the 64-bit Goldilocks
field. Forward direction (positive exponent in the twiddle), bit-exact
against the pure-Python reference in :mod:`metal_zk.reference.goldilocks`.

Sizes (PLAN.md §Regime taxonomy):
  * in-distribution: N in {2^14, 2^16, 2^18}
  * held-out:        N = 2^20

The held-out N catches candidates that hardcoded the stage layout for
the in-distribution log_N values, or that hardcoded the twiddle table
length / DRAM-buffer dimensions.

Roofline:
  Per stage we touch 20 bytes/element (8 read in + 8 write out + 4 from
  the half-sized twiddle table, amortised). Across ``log_N`` stages the
  candidate moves ``20 * N * log_N`` bytes — BW-bound for large N.
  Compute is ``(N/2) * log_N`` Goldilocks modmuls per NTT. We report
  the achieved fraction against the **binding** roofline:
  ``max(achieved_bw / peak_bw, achieved_mul / peak_int64_mul)``. This
  keeps cross-N comparisons honest: a candidate that saturates DRAM at
  N=2^20 and saturates int-mul at N=2^14 reports ~1.0 at both ends.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import Metal

from ..harness import MetalHarness
from ..hardware import ChipSpec
from ..reference.goldilocks import (
    P, root_of_unity, random_field_elements, ntt_forward_cached,
)
from ..task import (
    Task, TaskSize, TaskSpec, SizeResult,
    gb_per_s, gops_per_s, register_task,
)


_SEED = Path(__file__).resolve().parent.parent.parent / "seeds" / "goldilocks_ntt.metal"


def _twiddle_table(N: int) -> np.ndarray:
    """tw[i] = omega_N^i for i in [0, N/2), as uint64."""
    log_n = (N - 1).bit_length()
    assert 1 << log_n == N
    w = root_of_unity(log_n)
    tw = np.empty(N // 2, dtype=np.uint64)
    v = 1
    for i in range(N // 2):
        tw[i] = v
        v = (v * w) % P
    return tw


@register_task("goldilocks_ntt")
class GoldilocksNTTTask(Task):
    spec = TaskSpec(
        name="goldilocks_ntt",
        description=(
            "Forward Number-Theoretic Transform of length N = 2^log_N over "
            "the Goldilocks prime field p = 2^64 - 2^32 + 1. Convention:\n"
            "  Y[k] = sum_{n=0}^{N-1} X[n] * omega_N^(k * n)   (mod p)\n"
            "where omega_N is the primitive N-th root of unity in Goldilocks "
            "(the host computes it from the standard plonky2 / risc0 root "
            "g_root_2^32 = 1753635133440165772 and uploads tw[i] = omega_N^i "
            "for i in [0, N/2) as the twiddle buffer).\n\n"
            "Storage is row-major ulong[N]; one independent NTT per task "
            "dispatch (batch=1). Bit-exact correctness against a Python "
            "bigint reference — any mismatched output element rejects the "
            "candidate.\n\n"
            "The host implements one butterfly stage per kernel dispatch "
            "and ping-pongs (in_data, out_data) across log_N dispatches. "
            "Per-stage indexing contract: at stage s, the butterfly pair "
            "index k in [0, N/2) decomposes as\n"
            "  j = k >> s, r = k & ((1 << s) - 1),\n"
            "the read offsets are (j * 2^s + r) and (j * 2^s + r + N/2), "
            "and the write offsets are (j * 2^(s+1) + r) and the same + 2^s. "
            "The twiddle for this butterfly is tw[r * (N >> (s + 1))].\n\n"
            "All arithmetic is over Goldilocks; outputs MUST be canonical "
            "(< p), i.e. a value in [p, 2^64) is treated as a mismatch "
            "even if its residue class matches."
        ),
        kernel_signatures=(
            "kernel void goldilocks_ntt_stage(\n"
            "    device const ulong *in_data    [[buffer(0)]],\n"
            "    device       ulong *out_data   [[buffer(1)]],\n"
            "    device const ulong *twiddles   [[buffer(2)]],\n"
            "    constant uint      &stage_idx  [[buffer(3)]],\n"
            "    constant uint      &log_N      [[buffer(4)]],\n"
            "    uint k [[thread_position_in_grid]]);\n"
            "\n"
            "Dispatch geometry (host-fixed, identical across stages):\n"
            "  threadsPerGrid        = (N/2, 1, 1)\n"
            "  threadsPerThreadgroup = (min(N/2, 256), 1, 1)\n"
            "Each thread owns exactly one butterfly pair; guard against "
            "k >= N/2 (the grid is rounded up to a multiple of the TG "
            "width). The host invokes the kernel log_N times in one "
            "command buffer with stage_idx = 0, 1, ..., log_N - 1, "
            "ping-ponging (in_data, out_data) between two device buffers; "
            "the final NTT result lands in the buffer selected by the "
            "parity of log_N. Twiddle and log_N buffers are bound once."
        ),
        kernel_names=["goldilocks_ntt_stage"],
        seed_path=_SEED,
        sizes=[
            TaskSize("N2_14", {"n": 1 << 14}),   # 16 K, ~256 KB ws, SLC-resident
            TaskSize("N2_16", {"n": 1 << 16}),   # 64 K, ~1 MB
            TaskSize("N2_18", {"n": 1 << 18}),   # 256 K, ~4 MB
        ],
        held_out_sizes=[
            TaskSize("N2_20", {"n": 1 << 20}),   # 1 M, ~16 MB DRAM-bound
        ],
    )

    def evaluate_size(self, harness, pipelines, size, chip, n_warmup, n_measure):
        N = int(size.params["n"])
        if N & (N - 1) != 0:
            raise ValueError(f"goldilocks_ntt size N={N} must be a power of two")
        log_N = (N - 1).bit_length()
        if log_N > 32:
            raise ValueError(
                f"log_N={log_N} exceeds Goldilocks 2-adicity (32)"
            )

        # Input + reference.
        x = random_field_elements(N, seed=0xC0FFEE + N).astype(np.uint64)
        y_ref = ntt_forward_cached(x)

        tw = _twiddle_table(N)

        bA = harness.buf_from_np(x)
        bB = harness.buf_zeros(int(x.nbytes))
        bT = harness.buf_from_np(tw)
        b_log = harness.buf_scalar(log_N, np.uint32)
        b_stage_indices = harness.buf_from_np(
            np.arange(log_N, dtype=np.uint32)
        )

        pso = pipelines["goldilocks_ntt_stage"]
        max_tg = int(pso.maxTotalThreadsPerThreadgroup())
        tew = int(pso.threadExecutionWidth())
        half_N = N // 2
        tg_w = min(max_tg, max(tew, 256), half_N if half_N >= tew else tew)
        tg_w = max(tg_w, 1)
        grid_w = ((half_N + tg_w - 1) // tg_w) * tg_w

        view_A = harness.np_view(bA, np.uint64, N)
        view_B = harness.np_view(bB, np.uint64, N)

        def reset():
            view_A[:] = x
            view_B[:] = 0

        def dispatch(enc):
            enc.setComputePipelineState_(pso)
            enc.setBuffer_offset_atIndex_(bT, 0, 2)
            enc.setBuffer_offset_atIndex_(b_log, 0, 4)
            for s in range(log_N):
                if s % 2 == 0:
                    enc.setBuffer_offset_atIndex_(bA, 0, 0)
                    enc.setBuffer_offset_atIndex_(bB, 0, 1)
                else:
                    enc.setBuffer_offset_atIndex_(bB, 0, 0)
                    enc.setBuffer_offset_atIndex_(bA, 0, 1)
                enc.setBuffer_offset_atIndex_(b_stage_indices, s * 4, 3)
                enc.dispatchThreads_threadsPerThreadgroup_(
                    Metal.MTLSizeMake(grid_w, 1, 1),
                    Metal.MTLSizeMake(tg_w, 1, 1),
                )

        # Warmup + timed measurements.
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
        final = view_B if (log_N % 2 == 1) else view_A
        got = final.copy()

        mismatches = int(np.sum(got != y_ref))
        # Out-of-range outputs (>= p) also count as failures — the host
        # reference is canonical, so a non-canonical Metal output (a
        # value in [p, 2^64)) would print as a mismatch even if the
        # residue class is the same. We surface this explicitly so the
        # LLM gets a clear feedback signal.
        non_canonical = int(np.sum(got >= np.uint64(P)))
        correct = (mismatches == 0)

        bytes_per_stage = float(N) * 8.0 * 2.0 + float(N // 2) * 8.0
        bytes_total = bytes_per_stage * log_N
        achieved_bw = gb_per_s(bytes_total, gpu_s)
        ceiling_bw = float(chip.peak_bw_gb_s)

        modmuls_per_stage = N // 2
        modmuls_total = modmuls_per_stage * log_N
        achieved_mul = gops_per_s(modmuls_total, gpu_s)
        ceiling_mul = float(chip.peak_int64_mul_gops)

        # Pick the binding roofline (whichever fraction is larger).
        frac_bw = achieved_bw / ceiling_bw if ceiling_bw > 0 else 0.0
        frac_mul = achieved_mul / ceiling_mul if ceiling_mul > 0 else 0.0
        if frac_mul >= frac_bw:
            achieved, achieved_unit = achieved_mul, "Gmodmul/s (int64)"
            ceiling, ceiling_unit = ceiling_mul, "Gops/s (int64 mul, est)"
            fraction = frac_mul
            anchor = "int64_mul"
        else:
            achieved, achieved_unit = achieved_bw, "GB/s"
            ceiling, ceiling_unit = ceiling_bw, "GB/s"
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
                "N": N, "log_N": log_N,
                "non_canonical": non_canonical,
                "achieved_bw_gb_s": achieved_bw,
                "achieved_mul_gops": achieved_mul,
                "bytes_total": bytes_total,
                "modmuls_total": modmuls_total,
                "anchor": anchor,
            },
        )
