"""Batched binary-field carry-less multiplication -- Z11 task.

Each instance multiplies one pair of binary-field elements; one thread
per pair. In-distribution sizes exercise ``GF(2^128)`` with the
AES-GCM irreducible polynomial ``R(x) = x^128 + x^7 + x^2 + x + 1``;
the held-out probe exercises ``GF(2^256)`` via the Fan-Hasan tower
``GF(2^128)[v] / (v^2 + v + alpha)``.

Sizes (PLAN.md S Regime taxonomy):
  * in-distribution: GF(2^128), ``N`` in {2^16, 2^18, 2^20}
  * held-out:        GF(2^256) Fan-Hasan tower, ``N = 2^18``

The held-out probe flips on candidates that:
  - hardcode ``field_words = 2`` (the tower needs 4 limbs per element);
  - ignore the ``tower`` flag (the tower path needs 3-5 inner muls
    plus an ``alpha`` scaling, not a single 128-bit clmul + reduce);
  - hardcode the 128-bit irreducible polynomial as the only mod path
    (the tower has no irreducible polynomial -- the reduction is
    ``v^2 = v + alpha`` and uses the bound ``alpha`` constants).

Roofline
========

Per product:
  * Bit-op anchor: GF(2^128) carry-less multiplication is essentially
    a dense 128 x 128 bit-AND mesh plus a 7-XOR reduction network.
    Packed as ``ulong`` ops, ``128^2 / 64 = 256`` u64 ops is the
    structural lower bound for one mul + reduction. The tower mul
    requires three GF(2^128) sub-products plus an ``alpha`` scaling
    over the tower; we count ``4 * 256 = 1024`` u64 bitops per
    tower product (treating the alpha scaling as a fourth GF(2^128)
    mul). This is a conservative lower bound: an optimal Karatsuba
    candidate that exploits a sparse ``alpha`` can land *below* the
    count and report >100% fraction-of-ceiling, matching the
    ``andn``-fusion overshoot we already see on Z8.

  * BW anchor: ``2 * 8 * field_words`` input bytes + ``8 *
    field_words`` output bytes per product, so 48 B per GF(2^128)
    product and 96 B per tower product. At the test sizes,
    GF(2^128) is firmly compute-bound on Apple Silicon (the bit-op
    anchor binds), but at the largest in-dist size the BW anchor
    is reported alongside so a memory-throughput candidate isn't
    silently penalised.

The task reports the **binding** roofline ``max(achieved_bitop /
peak_int64_bitop, achieved_bw / peak_bw)`` and tags ``extra["anchor"]``
so the analysis layer knows which ceiling the fraction is against.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import Metal

from ..reference.binary_field import (
    FieldParams, GF128, GF256_TOWER, params_for,
    multiply_cached, random_inputs,
)
from ..task import (
    Task, TaskSize, TaskSpec, SizeResult,
    gb_per_s, gops_per_s, register_task,
)


_SEED = Path(__file__).resolve().parent.parent.parent / "seeds" / "binius_clmul.metal"


# Structural lower bound for a packed GF(2^128) carry-less mul plus
# reduction, expressed as u64 bit-ops. Used as the per-product count
# in the bit-op roofline; see module docstring.
_BITOPS_PER_GF128_PRODUCT: int = 256
_BITOPS_PER_TOWER_PRODUCT: int = 4 * _BITOPS_PER_GF128_PRODUCT


def _bitops_per_product(params: FieldParams) -> int:
    return _BITOPS_PER_TOWER_PRODUCT if params.tower else _BITOPS_PER_GF128_PRODUCT


def _bytes_per_product(params: FieldParams) -> int:
    # 2 inputs + 1 output, each ``field_words`` u64 limbs.
    return 3 * 8 * params.field_words


@register_task("binius_clmul")
class BiniusClmulTask(Task):
    spec = TaskSpec(
        name="binius_clmul",
        description=(
            "Batched binary-field carry-less multiplication, one product "
            "per thread. Two parameter sets are exercised; the runtime "
            "``tower`` flag selects between them and the kernel MUST "
            "branch on the flag rather than baking either path in as a "
            "compile-time constant.\n\n"
            "tower = 0 -- GF(2^128). Each element is two ``ulong`` "
            "limbs in little-endian polynomial order (limb 0 holds the "
            "coefficients of x^0..x^63; limb 1 holds x^64..x^127). The "
            "product is computed in GF(2)[x] (every '+' is XOR; there "
            "are zero integer multiplies) and reduced modulo the "
            "AES-GCM irreducible polynomial "
            "R(x) = x^128 + x^7 + x^2 + x + 1. The standard two-stage "
            "fold suffices: stage 1 folds the upper 128 bits into the "
            "lower 128 via the low pattern 1 + x + x^2 + x^7, leaving "
            "a residual at most 7 bits long; stage 2 folds the residual "
            "once more, after which the result has degree < 128.\n\n"
            "tower = 1 -- GF(2^256) via the Fan-Hasan tower "
            "GF(2^128)[v] / (v^2 + v + alpha). Each element is four "
            "``ulong`` limbs: limbs 0, 1 hold the v^0 coefficient "
            "a_0 in GF(2^128); limbs 2, 3 hold the v^1 coefficient "
            "a_1. With (a_0 + a_1 v) (b_0 + b_1 v) = c_0 + c_1 v and "
            "v^2 = v + alpha (the consequence of v^2 + v + alpha = 0 "
            "in characteristic 2),\n"
            "    c_0 = a_0 b_0 + alpha * (a_1 b_1)\n"
            "    c_1 = a_0 b_1 + a_1 b_0 + a_1 b_1\n"
            "The ``alpha`` operand is supplied via the bound "
            "``alpha_lo`` / ``alpha_hi`` scalars and is itself an "
            "element of GF(2^128).\n\n"
            "Buffer layout: for ``field_words = 2 + 2 * tower`` limbs "
            "per element, ``a``, ``b``, ``c`` are flat ``ulong`` arrays "
            "of length ``batch * field_words``; element i occupies "
            "limbs ``[i * field_words .. i * field_words + field_words)``. "
            "Outputs are the raw 64-bit polynomial coefficient patterns; "
            "the host compares bit-exactly against a CPU GF(2^128) / "
            "tower reference."
        ),
        kernel_signatures=(
            "kernel void binius_clmul(\n"
            "    device const ulong *a         [[buffer(0)]],\n"
            "    device const ulong *b         [[buffer(1)]],\n"
            "    device       ulong *c         [[buffer(2)]],\n"
            "    constant ulong     &alpha_lo  [[buffer(3)]],\n"
            "    constant ulong     &alpha_hi  [[buffer(4)]],\n"
            "    constant uint      &tower     [[buffer(5)]],\n"
            "    constant uint      &batch     [[buffer(6)]],\n"
            "    uint idx [[thread_position_in_grid]]);\n"
            "\n"
            "Dispatch (host-fixed):\n"
            "  threadsPerGrid        = (batch, 1, 1)\n"
            "  threadsPerThreadgroup = (min(batch, 64), 1, 1)\n"
            "Each thread processes ONE product end-to-end; guard "
            "against ``idx >= batch`` (the grid is rounded up to a "
            "multiple of the TG width). Threadgroup- or simdgroup-"
            "cooperative implementations are valid so long as the "
            "external buffer layout above and the canonical-output "
            "contract are preserved."
        ),
        kernel_names=["binius_clmul"],
        seed_path=_SEED,
        sizes=[
            TaskSize(
                "gf128_N64K",
                {"variant": "gf128", "n": 1 << 16,
                 "seed": 0xB1B1_C000 + (1 << 16)},
            ),
            TaskSize(
                "gf128_N256K",
                {"variant": "gf128", "n": 1 << 18,
                 "seed": 0xB1B1_C000 + (1 << 18)},
            ),
            TaskSize(
                "gf128_N1M",
                {"variant": "gf128", "n": 1 << 20,
                 "seed": 0xB1B1_C000 + (1 << 20)},
            ),
        ],
        held_out_sizes=[
            TaskSize(
                "gf256_tower_N256K",
                {"variant": "gf256_tower", "n": 1 << 18,
                 "seed": 0xB1B1_C000 + (1 << 18) + 7},
            ),
        ],
    )

    def evaluate_size(self, harness, pipelines, size, chip, n_warmup, n_measure):
        variant = str(size.params["variant"])
        n = int(size.params["n"])
        seed = int(size.params["seed"])
        params = params_for(variant)
        fw = params.field_words

        a_in, b_in = random_inputs(n, params, seed=seed)
        c_ref = multiply_cached(a_in, b_in, params)

        bA = harness.buf_from_np(np.ascontiguousarray(a_in, dtype=np.uint64))
        bB = harness.buf_from_np(np.ascontiguousarray(b_in, dtype=np.uint64))
        bC = harness.buf_zeros(int(n * fw * 8))
        b_alpha_lo = harness.buf_scalar(params.alpha_lo, np.uint64)
        b_alpha_hi = harness.buf_scalar(params.alpha_hi, np.uint64)
        b_tower = harness.buf_scalar(params.tower, np.uint32)
        b_batch = harness.buf_scalar(n, np.uint32)

        pso = pipelines["binius_clmul"]
        max_tg = int(pso.maxTotalThreadsPerThreadgroup())
        tew = int(pso.threadExecutionWidth())
        tg_w = min(max_tg, max(tew, 64))
        grid_w = ((n + tg_w - 1) // tg_w) * tg_w

        view_C = harness.np_view(bC, np.uint64, n * fw)

        def reset():
            view_C[:] = 0

        def dispatch(enc):
            enc.setComputePipelineState_(pso)
            enc.setBuffer_offset_atIndex_(bA, 0, 0)
            enc.setBuffer_offset_atIndex_(bB, 0, 1)
            enc.setBuffer_offset_atIndex_(bC, 0, 2)
            enc.setBuffer_offset_atIndex_(b_alpha_lo, 0, 3)
            enc.setBuffer_offset_atIndex_(b_alpha_hi, 0, 4)
            enc.setBuffer_offset_atIndex_(b_tower, 0, 5)
            enc.setBuffer_offset_atIndex_(b_batch, 0, 6)
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

        # Final correctness pass.
        reset()
        harness.time_dispatch(dispatch)
        got = view_C.copy()

        mismatches = int(np.sum(got != c_ref))
        correct = (mismatches == 0)

        bitops_per_prod = _bitops_per_product(params)
        bitops_total = float(n) * float(bitops_per_prod)
        achieved_bitop = gops_per_s(bitops_total, gpu_s)
        ceiling_bitop = float(chip.peak_int64_bitop_gops)

        bytes_total = float(n) * float(_bytes_per_product(params))
        achieved_bw = gb_per_s(bytes_total, gpu_s)
        ceiling_bw = float(chip.peak_bw_gb_s)

        frac_bitop = (
            achieved_bitop / ceiling_bitop if ceiling_bitop > 0 else 0.0
        )
        frac_bw = (
            achieved_bw / ceiling_bw if ceiling_bw > 0 else 0.0
        )
        if frac_bitop >= frac_bw:
            achieved, achieved_unit = achieved_bitop, "Gbitops/s (u64)"
            ceiling, ceiling_unit = ceiling_bitop, "Gops/s (u64 bitop, est)"
            fraction = frac_bitop
            anchor = "int64_bitop"
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
                "variant": variant,
                "n": n,
                "field_words": fw,
                "tower": params.tower,
                "alpha_lo": params.alpha_lo,
                "alpha_hi": params.alpha_hi,
                "bitops_per_product": bitops_per_prod,
                "bitops_total": bitops_total,
                "bytes_total": bytes_total,
                "achieved_bitop_gops": achieved_bitop,
                "achieved_bw_gb_s": achieved_bw,
                "anchor": anchor,
            },
        )
