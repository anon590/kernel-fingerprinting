"""Poseidon2 hash — Z3 task.

Batched Poseidon2 permutation over Goldilocks, one thread per sponge.
In-distribution sizes vary the batch at fixed arity t=3; the held-out
probe at t=4 forces the candidate to NOT hardcode round constants /
MDS-matrix shape — both arities ship distinct constants generated
deterministically by ``metal_zk.reference.poseidon2``.

Sizes (PLAN.md §Regime taxonomy):
  * in-distribution: t=3, batch in {2^12, 2^16, 2^20}
  * held-out:        t=4, batch = 2^18 (different MDS + round constants)

Roofline:
  Per Poseidon2 permutation at t=3, R_F=8, R_P=22 with alpha=7 S-box:
    - S-boxes: 4 muls each, 8*3 + 22*1 = 46 S-boxes -> ~184 muls
    - Full-round matvecs: 9 muls per matvec * 9 matvecs (1 pre + 8 rounds)
                          = 81 muls
    - Partial-round int-mds: 3 muls per matvec * 22 = 66 muls
  Total ~ 331 Goldilocks mod-muls per sponge. At t=4 the count rises
  by the matvec dimension (16 vs 9 per ext matvec, 4 vs 3 per int);
  per-sponge total ~ 488.

  Bytes / sponge: 2*t*8 = 48 (t=3) or 64 (t=4) for the load+store. The
  constants buffers are tiny and shared via cache; not counted.

  The task reports the achieved fraction against the **binding** roofline
  ``max(achieved_mul / peak_int64_mul, achieved_bw / peak_bw)``. At the
  in-distribution batches Poseidon2 is firmly compute-bound on Apple
  Silicon, so the int-mul anchor dominates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import Metal

from ..harness import MetalHarness
from ..hardware import ChipSpec
from ..reference.poseidon2 import (
    Poseidon2Goldilocks, R_F, R_P, permute_batch_cached,
)
from ..reference.goldilocks import random_field_elements
from ..task import (
    Task, TaskSize, TaskSpec, SizeResult,
    gb_per_s, gops_per_s, register_task,
)


_SEED = Path(__file__).resolve().parent.parent.parent / "seeds" / "poseidon2_hash.metal"


def _modmuls_per_sponge(t: int, r_f: int, r_p: int) -> int:
    """Counting model (matches PLAN.md / module docstring above)."""
    sbox = 4                                # x^7 = 4 modmuls
    n_sbox = r_f * t + r_p * 1
    matvec_ext = t * t
    n_matvec_ext = 1 + r_f                  # 1 pre + r_f
    matvec_int = t
    n_matvec_int = r_p
    return n_sbox * sbox + matvec_ext * n_matvec_ext + matvec_int * n_matvec_int


@register_task("poseidon2_hash")
class Poseidon2HashTask(Task):
    spec = TaskSpec(
        name="poseidon2_hash",
        description=(
            "Batched Poseidon2 permutation over the Goldilocks field "
            "(p = 2^64 - 2^32 + 1, S-box alpha = 7, R_F = 8 full rounds "
            "split 4+4, R_P = 22 partial rounds). Each of ``batch`` "
            "independent sponges runs the same permutation on its own "
            "length-t state vector. The output is the full permuted state "
            "(NOT a sponge truncation): out_state[idx, :] = "
            "Permute(in_state[idx, :]).\n\n"
            "The arity ``t``, the round-count parameters, and the round "
            "constants / MDS coefficients are all bound as device or "
            "constant buffers (see the buffer layout below); the kernel "
            "must use the runtime values rather than compile-time "
            "constants. The host always passes a t-square ``ext_mds`` "
            "and a t-length ``int_diag`` in row-major order; the "
            "internal-MDS convention is M_I = J + diag(int_diag) where "
            "J is the all-ones matrix, i.e. the per-thread internal "
            "matvec is\n"
            "  y[i] = sum(state) + int_diag[i] * state[i].\n\n"
            "The external matvec is the generic dense form: "
            "y[i] = sum_j ext_mds[i * t + j] * state[j].\n\n"
            "Algorithm (executed by the seed):\n"
            "  state <- ext_mds * state\n"
            "  for r in 0..R_F/2:        # first half-full rounds\n"
            "    state[i] += rc_ext[r, i] for all i\n"
            "    state[i] = state[i]^7  for all i\n"
            "    state <- ext_mds * state\n"
            "  for r in 0..R_P:           # partial rounds\n"
            "    state[0] += rc_int[r]\n"
            "    state[0] = state[0]^7\n"
            "    state <- (J + diag(int_diag)) * state\n"
            "  for r in R_F/2..R_F:       # second half-full rounds\n"
            "    (same shape as first half)\n\n"
            "All arithmetic is in Goldilocks; bit-exact correctness "
            "against a Python bigint reference. Outputs MUST be canonical "
            "(< p); a non-canonical value with the same residue class still "
            "counts as a mismatch."
        ),
        kernel_signatures=(
            "kernel void poseidon2_hash(\n"
            "    device const ulong *in_state    [[buffer(0)]],\n"
            "    device       ulong *out_state   [[buffer(1)]],\n"
            "    device const ulong *rc_ext      [[buffer(2)]],\n"
            "    device const ulong *rc_int      [[buffer(3)]],\n"
            "    device const ulong *ext_mds     [[buffer(4)]],\n"
            "    device const ulong *int_diag    [[buffer(5)]],\n"
            "    constant uint      &t           [[buffer(6)]],\n"
            "    constant uint      &r_f         [[buffer(7)]],\n"
            "    constant uint      &r_p         [[buffer(8)]],\n"
            "    constant uint      &batch       [[buffer(9)]],\n"
            "    uint idx [[thread_position_in_grid]]);\n"
            "\n"
            "Dispatch (host-fixed):\n"
            "  threadsPerGrid        = (batch, 1, 1)\n"
            "  threadsPerThreadgroup = (min(batch, 64), 1, 1)\n"
            "Each thread runs ONE sponge end-to-end; guard against "
            "idx >= batch (the grid is rounded up to a multiple of the "
            "TG width).\n"
            "\n"
            "All test sizes satisfy t <= 4 and R_F <= 8, R_P <= 32; "
            "thread-private state arrays of size 4 and round-constant "
            "tables of size 32 are sufficient. Threadgroup-cooperative "
            "and simdgroup schemes are valid as long as the external "
            "buffer layout above is preserved."
        ),
        kernel_names=["poseidon2_hash"],
        seed_path=_SEED,
        sizes=[
            TaskSize("t3_B4K",   {"t": 3, "batch": 1 << 12}),
            TaskSize("t3_B64K",  {"t": 3, "batch": 1 << 16}),
            TaskSize("t3_B1M",   {"t": 3, "batch": 1 << 20}),
        ],
        held_out_sizes=[
            TaskSize("t4_B256K", {"t": 4, "batch": 1 << 18}),
        ],
    )

    def evaluate_size(self, harness, pipelines, size, chip, n_warmup, n_measure):
        t = int(size.params["t"])
        batch = int(size.params["batch"])
        cfg = Poseidon2Goldilocks(t=t)

        # Random input states.
        flat_in = random_field_elements(batch * t, seed=0xBADBEEF + batch * 31 + t)
        states = flat_in.reshape(batch, t)

        # CPU reference is disk-cached (sha256 of t + input bytes);
        # essential at t3_B1M / t4_B256K where pure-Python bigint
        # Poseidon2 takes ~90s and would re-run every iter * rep.
        y_ref = permute_batch_cached(cfg, states)

        # Buffers.
        bA = harness.buf_from_np(flat_in)
        bB = harness.buf_zeros(int(flat_in.nbytes))
        b_rc_ext = harness.buf_from_np(cfg.round_constants_external.ravel())
        b_rc_int = harness.buf_from_np(cfg.round_constants_internal)
        b_ext_mds = harness.buf_from_np(cfg.ext_mds.ravel())
        b_int_diag = harness.buf_from_np(cfg.int_diag)
        b_t = harness.buf_scalar(t, np.uint32)
        b_rf = harness.buf_scalar(cfg.r_f, np.uint32)
        b_rp = harness.buf_scalar(cfg.r_p, np.uint32)
        b_batch = harness.buf_scalar(batch, np.uint32)

        pso = pipelines["poseidon2_hash"]
        max_tg = int(pso.maxTotalThreadsPerThreadgroup())
        tew = int(pso.threadExecutionWidth())
        tg_w = min(max_tg, max(tew, 64))
        grid_w = ((batch + tg_w - 1) // tg_w) * tg_w

        view_B = harness.np_view(bB, np.uint64, batch * t)

        def reset():
            view_B[:] = 0

        def dispatch(enc):
            enc.setComputePipelineState_(pso)
            enc.setBuffer_offset_atIndex_(bA, 0, 0)
            enc.setBuffer_offset_atIndex_(bB, 0, 1)
            enc.setBuffer_offset_atIndex_(b_rc_ext, 0, 2)
            enc.setBuffer_offset_atIndex_(b_rc_int, 0, 3)
            enc.setBuffer_offset_atIndex_(b_ext_mds, 0, 4)
            enc.setBuffer_offset_atIndex_(b_int_diag, 0, 5)
            enc.setBuffer_offset_atIndex_(b_t, 0, 6)
            enc.setBuffer_offset_atIndex_(b_rf, 0, 7)
            enc.setBuffer_offset_atIndex_(b_rp, 0, 8)
            enc.setBuffer_offset_atIndex_(b_batch, 0, 9)
            enc.dispatchThreads_threadsPerThreadgroup_(
                Metal.MTLSizeMake(grid_w, 1, 1),
                Metal.MTLSizeMake(tg_w, 1, 1),
            )

        for _ in range(n_warmup):
            reset()
            harness.time_dispatch(dispatch)
        samples = []
        for _ in range(n_measure):
            reset()
            samples.append(harness.time_dispatch(dispatch))
        gpu_s = float(np.median(samples))

        # Correctness pass.
        reset()
        harness.time_dispatch(dispatch)
        got = view_B.copy().reshape(batch, t)

        mismatches = int(np.sum(got != y_ref))
        correct = (mismatches == 0)

        # Per-sponge modmul count.
        muls_per = _modmuls_per_sponge(t, cfg.r_f, cfg.r_p)
        muls_total = muls_per * batch
        achieved_mul = gops_per_s(muls_total, gpu_s)
        ceiling_mul = float(chip.peak_int64_mul_gops)

        bytes_total = batch * t * 8 * 2     # load + store
        achieved_bw = gb_per_s(bytes_total, gpu_s)
        ceiling_bw = float(chip.peak_bw_gb_s)

        frac_mul = achieved_mul / ceiling_mul if ceiling_mul > 0 else 0.0
        frac_bw = achieved_bw / ceiling_bw if ceiling_bw > 0 else 0.0
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
                "t": t, "batch": batch,
                "r_f": cfg.r_f, "r_p": cfg.r_p, "alpha": cfg.alpha,
                "muls_per_sponge": muls_per,
                "muls_total": muls_total,
                "achieved_bw_gb_s": achieved_bw,
                "achieved_mul_gops": achieved_mul,
                "anchor": anchor,
            },
        )
