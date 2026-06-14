"""Multi-scalar multiplication -- Z1 task.

Naive-friendly two-kernel decomposition: ``montgomery_msm_pair``
computes ``t_i = s_i * P_i`` per pair, ``montgomery_msm_reduce``
tree-reduces the N partial products in place. The host issues one
pair dispatch and ``log2(n_pairs)`` reduce dispatches.

Sizes (PLAN.md Z1 specifies 2^16 / 2^18 / 2^20 for BLS in-dist and
2^17 BN254 held-out; we shift two octaves down because the naive
seed's per-pair cost -- 256 doublings + ~128 adds in 6-limb
Montgomery -- is ~200x heavier than the per-element work in
neighbouring tasks. The 4x size spread and the BN254 held-out
twist are preserved):
  * in-distribution: BLS12-381 G1, N in {2^12, 2^14, 2^16}
  * held-out:        BN254 G1,     N = 2^13

The held-out probe flips together on three independent overfit modes:

  - **the modulus**: BLS12-381 q (~381 bits, all 6 limbs alive)
    swaps for BN254 q (~254 bits, top 2 limbs of q are 0). A
    candidate that hardcodes BLS12-381's q silently produces wrong
    modular reductions on BN254;
  - **the CIOS scalar** ``q_inv_neg`` (i.e. ``-q^-1 mod 2^64``)
    changes; a candidate that hardcodes BLS's scalar silently
    breaks reductions even if it reads ``q`` from the buffer;
  - **the limb count**: a candidate that specialises to only the
    bottom 4 limbs (because "BN254 fits in 256 bits") silently
    fails BLS12-381 by dropping the top 2 limbs of the
    arithmetic.

Coordinate / representation convention:
  - All field elements in Montgomery form with R = 2^384 (uniform
    across both curves; BN254 is zero-padded in its top two limbs).
  - Points stored in Jacobian (X, Y, Z) layout; ``Z = 0`` is the
    point at infinity. 18 ulongs per point.
  - Scalars stored as 4-ulong little-endian limbs (both curves'
    scalar fields fit in 256 bits).

Bit-exactness: Jacobian representations are non-unique, so we
normalize *both* sides of the comparison to **affine Montgomery**
form before the bit-exact check. The reference exploits the
algebraic structure of the inputs (``P_i = (a + b * i) * G``) to
compute the expected result in milliseconds rather than minutes;
see ``metal_zk.reference.msm`` for the shortcut.

Roofline:
  Per pair, the seed runs 256 Jacobian doublings + ~128 Jacobian
  additions. Counting only base-field multiplications (the
  Montgomery-mul-bound work):
    modmuls_per_pair = 256 * 10 + 128 * 16 = 4608
  Per tree-reduce step there is exactly one Jacobian addition (16
  modmuls). With ``n_pairs - 1`` reduce-step adds total, the
  reduce path contributes ``16 * (n_pairs - 1)`` modmuls.

  Bytes touched: scalars ``32 * N`` read, points_in ``144 * N``
  read, scratch ``144 * N`` write + ``144 * (2N - 2)``
  read+write across the tree levels. Compute dominates BW by a
  wide margin at every test size.

  We report ``max(achieved_mul/peak_mul, achieved_bw/peak_bw)``
  as the binding ceiling, mirroring ``goldilocks_ntt`` and
  ``merkle_build``. Both the seed and any improved candidate are
  expected to be int-mul bound on Apple Silicon.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import Metal

from ..harness import MetalHarness
from ..hardware import ChipSpec
from ..reference.msm import (
    BLS12_381_G1, BN254_G1, Montgomery, N_LIMBS,
    MODMULS_PER_ADD, MODMULS_PER_DOUBLE, SCALAR_BITS_SCANNED,
    gen_inputs, get_curve, int_to_limbs, jac_to_affine_mont, JacMont,
    limbs_to_int,
)
from ..task import (
    Task, TaskSize, TaskSpec, SizeResult,
    gb_per_s, gops_per_s, register_task,
)


_SEED = Path(__file__).resolve().parent.parent.parent / "seeds" / "montgomery_msm.metal"


def _modmuls_per_pair() -> int:
    """Cost of one naive double-and-add at the fixed 256-bit scan."""
    n_doubles = SCALAR_BITS_SCANNED
    n_adds = SCALAR_BITS_SCANNED // 2
    return n_doubles * MODMULS_PER_DOUBLE + n_adds * MODMULS_PER_ADD


@register_task("montgomery_msm")
class MontgomeryMsmTask(Task):
    spec = TaskSpec(
        name="montgomery_msm",
        description=(
            "Multi-scalar multiplication on a short-Weierstrass elliptic "
            "curve. Given ``n_pairs`` pairs of (256-bit scalar ``s_i``, "
            "Jacobian point ``P_i``), compute the single curve point "
            "``R = sum_i s_i * P_i`` and emit it in Jacobian Montgomery "
            "form. The in-distribution sizes use BLS12-381 G1 "
            "(q ~ 381 bits, b = 4); the held-out size uses BN254 G1 "
            "(q ~ 254 bits, b = 3).\n\n"
            "Field representation: all elements live in Montgomery form "
            "with R = 2^384, six 64-bit limbs. The base-field modulus "
            "``q`` (6 ulongs, little-endian) and the CIOS scalar "
            "``q_inv_neg`` (``-q^-1 mod 2^64``) are bound as device / "
            "constant buffers; both **must** be read at runtime. A "
            "candidate that hardcodes the in-distribution modulus or "
            "its Montgomery constants silently produces wrong output "
            "on the held-out probe.\n\n"
            "Coordinate convention: 6-limb Jacobian "
            "``(X, Y, Z)``, little-endian limbs, affine point is "
            "``(X / Z^2, Y / Z^3)``, ``Z = 0`` represents the point "
            "at infinity. Per point: 18 ulongs.\n\n"
            "Scalars: 4-ulong little-endian limbs (both curves' "
            "scalar fields fit in 256 bits).\n\n"
            "Bit-exact correctness: the host normalizes the GPU "
            "Jacobian output to affine Montgomery form via one "
            "base-field inversion, then compares the (X_aff_mont, "
            "Y_aff_mont) pair against the algebraic reference. A "
            "non-canonical limb (>= q) counts as a mismatch even if "
            "the residue class agrees.\n\n"
            "Threadgroup-cooperative and simdgroup-cooperative "
            "implementations are valid so long as the external buffer "
            "layout above is preserved and the ``pair`` + "
            "``log2(n_pairs)`` x ``reduce`` dispatch schedule is "
            "honored (the pair kernel sees each (scalar, point) pair "
            "exactly once; each reduce dispatch sees the current tree "
            "level via ``half_count``)."
        ),
        kernel_signatures=(
            "kernel void montgomery_msm_pair(\n"
            "    device const ulong *scalars      [[buffer(0)]],\n"
            "    device const ulong *points_in    [[buffer(1)]],\n"
            "    device       ulong *scratch      [[buffer(2)]],\n"
            "    device const ulong *q            [[buffer(3)]],\n"
            "    constant ulong     &q_inv_neg    [[buffer(4)]],\n"
            "    constant uint      &n_pairs      [[buffer(5)]],\n"
            "    uint idx [[thread_position_in_grid]]);\n"
            "\n"
            "kernel void montgomery_msm_reduce(\n"
            "    device       ulong *scratch      [[buffer(0)]],\n"
            "    device const ulong *q            [[buffer(1)]],\n"
            "    constant ulong     &q_inv_neg    [[buffer(2)]],\n"
            "    constant uint      &half_count   [[buffer(3)]],\n"
            "    uint idx [[thread_position_in_grid]]);\n"
            "\n"
            "Dispatch (host-fixed):\n"
            "  montgomery_msm_pair: one thread per (s_i, P_i); guard "
            "against idx >= n_pairs; grid rounded up to a multiple of "
            "the TG width.\n"
            "  montgomery_msm_reduce: invoked log2(n_pairs) times in a "
            "single compute command encoder with ``half_count`` "
            "successively halving (n_pairs/2, n_pairs/4, ..., 1). One "
            "thread per active slot; thread t reads scratch[t] and "
            "scratch[t + half_count], adds them in Jacobian form, and "
            "writes the sum back to scratch[t]. The serial command "
            "encoder gives read-after-write between levels with no "
            "explicit barriers required.\n"
            "  threadsPerThreadgroup = (min(grid_w, 64), 1, 1) for both "
            "kernels in the seed; cooperative implementations may pick "
            "a different tile width but must honor the buffer layout "
            "and the half_count contract in reduce."
        ),
        kernel_names=["montgomery_msm_pair", "montgomery_msm_reduce"],
        seed_path=_SEED,
        sizes=[
            TaskSize("bls_N4K",  {"curve": "bls12_381_g1", "n_pairs": 1 << 12}),
            TaskSize("bls_N16K", {"curve": "bls12_381_g1", "n_pairs": 1 << 14}),
            TaskSize("bls_N64K", {"curve": "bls12_381_g1", "n_pairs": 1 << 16}),
        ],
        held_out_sizes=[
            TaskSize("bn254_N8K",
                     {"curve": "bn254_g1", "n_pairs": 1 << 13}),
        ],
    )

    def evaluate_size(self, harness, pipelines, size, chip, n_warmup, n_measure):
        curve_name = str(size.params["curve"])
        n_pairs = int(size.params["n_pairs"])
        if n_pairs <= 0 or (n_pairs & (n_pairs - 1)) != 0:
            raise ValueError(
                f"montgomery_msm requires power-of-two n_pairs; got {n_pairs}"
            )
        curve = get_curve(curve_name)
        mont = Montgomery(curve.q, N_LIMBS)

        # Seed mixes (curve, n_pairs) so each test gets its own cache
        # entry and the held-out probe doesn't share a precompute file
        # with any in-dist size.
        seed = 0xA15B12 ^ (n_pairs * 1009) ^ hash(curve.name) & 0xFFFFFFFF
        inputs = gen_inputs(curve, n_pairs, seed=seed)

        # Host buffers.
        b_scalars = harness.buf_from_np(inputs.scalars_u64.ravel())
        b_points_in = harness.buf_from_np(inputs.points_in_u64.ravel())
        b_scratch = harness.buf_zeros(n_pairs * 3 * N_LIMBS * 8)
        q_limbs = int_to_limbs(curve.q, N_LIMBS)
        b_q = harness.buf_from_np(q_limbs)
        b_q_inv_neg = harness.buf_scalar(mont.q_inv_neg_64, np.uint64)
        b_n_pairs = harness.buf_scalar(n_pairs, np.uint32)

        # Per-level half-count scalars (one ulong per level: half =
        # n_pairs/2, /4, ..., 1).
        log_n = (n_pairs - 1).bit_length()
        half_counts = np.array(
            [n_pairs >> (lvl + 1) for lvl in range(log_n)], dtype=np.uint32,
        )
        b_half_counts = harness.buf_from_np(half_counts)

        pso_pair = pipelines["montgomery_msm_pair"]
        pso_reduce = pipelines["montgomery_msm_reduce"]
        tew_p = int(pso_pair.threadExecutionWidth())
        max_tg_p = int(pso_pair.maxTotalThreadsPerThreadgroup())
        tg_w_p = min(max_tg_p, max(tew_p, 64))
        grid_pair = ((n_pairs + tg_w_p - 1) // tg_w_p) * tg_w_p
        tew_r = int(pso_reduce.threadExecutionWidth())
        max_tg_r = int(pso_reduce.maxTotalThreadsPerThreadgroup())
        tg_w_r = min(max_tg_r, max(tew_r, 64))

        view_scratch = harness.np_view(
            b_scratch, np.uint64, n_pairs * 3 * N_LIMBS,
        )

        def reset():
            view_scratch[:] = 0

        def dispatch(enc):
            # Phase 1: per-pair scalar multiplications.
            enc.setComputePipelineState_(pso_pair)
            enc.setBuffer_offset_atIndex_(b_scalars, 0, 0)
            enc.setBuffer_offset_atIndex_(b_points_in, 0, 1)
            enc.setBuffer_offset_atIndex_(b_scratch, 0, 2)
            enc.setBuffer_offset_atIndex_(b_q, 0, 3)
            enc.setBuffer_offset_atIndex_(b_q_inv_neg, 0, 4)
            enc.setBuffer_offset_atIndex_(b_n_pairs, 0, 5)
            enc.dispatchThreads_threadsPerThreadgroup_(
                Metal.MTLSizeMake(grid_pair, 1, 1),
                Metal.MTLSizeMake(tg_w_p, 1, 1),
            )
            # Phase 2: tree reduction.
            enc.setComputePipelineState_(pso_reduce)
            enc.setBuffer_offset_atIndex_(b_scratch, 0, 0)
            enc.setBuffer_offset_atIndex_(b_q, 0, 1)
            enc.setBuffer_offset_atIndex_(b_q_inv_neg, 0, 2)
            for lvl in range(log_n):
                hc = int(half_counts[lvl])
                tg_lvl = min(tg_w_r, max(1, hc))
                grid_lvl = ((hc + tg_lvl - 1) // tg_lvl) * tg_lvl
                enc.setBuffer_offset_atIndex_(b_half_counts, lvl * 4, 3)
                enc.dispatchThreads_threadsPerThreadgroup_(
                    Metal.MTLSizeMake(grid_lvl, 1, 1),
                    Metal.MTLSizeMake(tg_lvl, 1, 1),
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
        result_limbs = view_scratch[:3 * N_LIMBS].copy()

        # Normalize GPU output to affine Montgomery.
        gpu_X = limbs_to_int(result_limbs[0:N_LIMBS])
        gpu_Y = limbs_to_int(result_limbs[N_LIMBS:2 * N_LIMBS])
        gpu_Z = limbs_to_int(result_limbs[2 * N_LIMBS:3 * N_LIMBS])

        # Canonical-range check: every limb output must satisfy
        # the value < q. We detect non-canonical limbs *before* trying
        # to convert to affine -- a non-canonical Z would still
        # invert cleanly but the canonical reference uses canonical
        # output. If any coord has value >= q we flag a mismatch.
        non_canonical = sum(
            int(v >= curve.q) for v in (gpu_X, gpu_Y, gpu_Z)
        )

        gpu_jac = JacMont(X=gpu_X, Y=gpu_Y, Z=gpu_Z)

        gpu_aff = jac_to_affine_mont(gpu_jac, mont)
        ref_aff_u64 = inputs.expected_aff_mont_u64

        if gpu_aff is None and ref_aff_u64 is None:
            mismatches = 0
        elif gpu_aff is None or ref_aff_u64 is None:
            mismatches = 1
        else:
            gpu_x_mont, gpu_y_mont = gpu_aff
            ref_x_mont = limbs_to_int(ref_aff_u64[:N_LIMBS])
            ref_y_mont = limbs_to_int(ref_aff_u64[N_LIMBS:2 * N_LIMBS])
            mismatches = int(
                (gpu_x_mont != ref_x_mont) + (gpu_y_mont != ref_y_mont)
            )

        mismatches += non_canonical
        correct = (mismatches == 0)

        # Roofline.
        muls_pair = _modmuls_per_pair() * n_pairs
        muls_reduce = MODMULS_PER_ADD * (n_pairs - 1)
        muls_total = muls_pair + muls_reduce
        achieved_mul = gops_per_s(muls_total, gpu_s)
        ceiling_mul = float(chip.peak_int64_mul_gops)

        # BW: scalars (32 N) + points_in (144 N) read once for the
        # pair kernel; scratch (144 N) written by the pair kernel,
        # then for the reduce levels each level reads 144 *
        # half_count * 2 and writes 144 * half_count, with
        # sum(half_count) = N - 1, giving an extra 144 * 3 * (N-1)
        # roughly. Conservative total:
        bytes_pair_in = (32.0 + 144.0) * float(n_pairs)
        bytes_pair_out = 144.0 * float(n_pairs)
        bytes_reduce = 3.0 * 144.0 * float(n_pairs - 1)
        bytes_total = bytes_pair_in + bytes_pair_out + bytes_reduce
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
                "curve": curve.name,
                "n_pairs": n_pairs,
                "muls_per_pair": _modmuls_per_pair(),
                "muls_total": muls_total,
                "bytes_total": bytes_total,
                "achieved_mul_gops": achieved_mul,
                "achieved_bw_gb_s": achieved_bw,
                "non_canonical": non_canonical,
                "anchor": anchor,
            },
        )
