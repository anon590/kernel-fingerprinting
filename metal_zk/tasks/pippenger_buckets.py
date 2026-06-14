"""Pippenger bucket scatter -- Z9 task.

The atomic-scatter half of Pippenger's MSM. Given ``N`` 256-bit
scalars and ``N`` Jacobian Montgomery points on BLS12-381 G1, scatter
each point into the bucket addressed by its windowed scalar value,
accumulating via Jacobian EC addition. The output is a table of
``NUM_WINDOWS * (2^w - 1)`` Jacobian bucket sums; the subsequent
``aggregate / weighted-sum`` phase of Pippenger is outside this task's
scope (the lever isolated here is the on-GPU 384-bit EC scatter).

Sizes (PLAN.md Z9 specifies N in {2^16, 2^18, 2^20} at w=16; we shift
two octaves down to keep the bigint reference build tractable in pure
Python -- one Jacobian add costs ~70 us in CPython at 256-bit, so the
N=2^20 reference would take ~ 20 min per cache miss. The 4x spread
across in-dist sizes and the held-out twist axis are preserved):

  * in-distribution: uniform 256-bit scalars, N in {2^12, 2^14, 2^16}
  * held-out:        Zipf-1.5 scalars,        N = 2^14

The held-out probe targets the **bucket-traffic-distribution overfit**
lever. Both in-dist and held-out use BLS12-381 G1 (the modulus overfit
is already covered by Z1 ``montgomery_msm``); the only axis that flips
is the per-window bucket histogram:

  - uniform: each of the 2^w-1 buckets receives ~ N/(2^w-1) hits;
    contention on any single bucket is low.
  - Zipf-1.5: bucket 1 absorbs ~ 38% of all traffic, bucket 2 ~ 14%,
    the top-1% of buckets carry ~ 10^3 x the median bucket's traffic.
    A candidate that tuned its scatter strategy for uniform contention
    (e.g. a per-bucket spinlock that is contention-free at uniform
    sizes) catastrophically serialises on the hot buckets.

Window decomposition. ``w = WINDOW_BITS = 16`` is fixed; we process
``NUM_WINDOWS = 4`` consecutive 16-bit windows = the bottom 64 bits of
each scalar. The kernel reads ``num_windows`` and ``window_bits`` at
runtime so the same kernel handles ``num_windows in {1, ..., 16}`` if
desired; the seed exercises the 4-window case.

Bit-exactness: the seed accumulates buckets in a non-deterministic
order (spinlock acquisition order depends on thread scheduling), so
the Jacobian representation of each bucket varies run-to-run. We
normalize *both* the GPU output and the algebraic reference to
canonical affine Montgomery form ``(X / Z^2, Y / Z^3) * R mod q``
before the bit-exact compare. Non-canonical limbs on the GPU side
count as mismatches.

Roofline. Total per-dispatch modmul work:

    modmuls = NUM_WINDOWS * N * MODMULS_PER_BUCKET_ADD     # = 16 * 4 * N

Bytes moved (per dispatch): scalars (32 * N) + points (144 * N) read
once; each scatter loads + stores a Jacobian bucket (288 bytes per
scatter on average; non-empty buckets) for a total dominated by the
EC arithmetic, not the input streaming. We report the binding ceiling
``max(achieved_mul/peak_mul, achieved_bw/peak_bw)`` and surface both
fractions in ``extra`` -- on Apple Silicon at these sizes the int-mul
anchor binds.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import Metal

from ..harness import MetalHarness
from ..hardware import ChipSpec
from ..reference.msm import (
    BLS12_381_G1, Montgomery, N_LIMBS, int_to_limbs,
)
from ..reference.pippenger import (
    MODMULS_PER_BUCKET_ADD, NUM_WINDOWS, PippengerInputs,
    WINDOW_BITS, WINDOW_BUCKETS,
    gen_inputs, normalize_gpu_buckets,
)
from ..task import (
    Task, TaskSize, TaskSpec, SizeResult,
    gb_per_s, gops_per_s, register_task,
)


_SEED = Path(__file__).resolve().parent.parent.parent / "seeds" / "pippenger_buckets.metal"


@register_task("pippenger_buckets")
class PippengerBucketsTask(Task):
    spec = TaskSpec(
        name="pippenger_buckets",
        description=(
            "Pippenger bucket-scatter on a short-Weierstrass elliptic "
            "curve. Given ``n_pairs`` 256-bit scalars and ``n_pairs`` "
            "Jacobian Montgomery points on BLS12-381 G1, compute the "
            "``num_windows * (2^w - 1)`` bucket sums of Pippenger's "
            "MSM. For each pair index ``i`` and window index ``k in "
            "[0, num_windows)``, extract the ``w``-bit window value "
            "``b = (s_i >> (k*w)) & ((1 << w) - 1)``. If ``b == 0`` "
            "the pair contributes nothing to window ``k``; otherwise "
            "add ``P_i`` (Jacobian Montgomery) to ``buckets[k][b-1]``. "
            "Buckets start as the point at infinity (all-zero "
            "Jacobian).\n\n"
            "Field representation: six-limb Montgomery form "
            "(``R = 2^384``); the base-field modulus ``q`` (6 ulongs, "
            "little-endian) and the CIOS scalar "
            "``q_inv_neg = -q^-1 mod 2^64`` are bound as device / "
            "constant buffers and must be read at runtime.\n\n"
            "Window decomposition (host-bound runtime parameters):\n"
            "  * ``window_bits`` = 16\n"
            "  * ``num_windows`` = 4\n"
            "The kernel processes the bottom "
            "``num_windows * window_bits = 64`` bits of each scalar. "
            "Buckets are addressed in [1, 2^w); index ``b = 0`` is "
            "elided. The output buffer's slot ``[k][b - 1]`` holds "
            "the sum for window ``k`` and bucket value ``b``.\n\n"
            "Coordinate convention: 6-limb Jacobian ``(X, Y, Z)`` in "
            "Montgomery form, little-endian limbs, 18 ulongs per "
            "point. ``Z == 0`` represents the point at infinity "
            "(the initial state of every bucket).\n\n"
            "Scalars: 4-ulong little-endian limbs (256-bit).\n\n"
            "Bit-exact correctness: the order in which a bucket's "
            "contributing points are summed is implementation-defined, "
            "so the Jacobian representation of each bucket may vary. "
            "The host normalizes every GPU bucket "
            "``(X, Y, Z)`` to affine Montgomery "
            "``(X / Z^2, Y / Z^3) * R mod q`` via one batched modular "
            "inversion and compares ``(X_aff_mont, Y_aff_mont)`` "
            "limb-for-limb against the CPU reference. A non-canonical "
            "limb (>= q) on the GPU side counts as a mismatch even if "
            "the residue class matches.\n\n"
            "The kernel must read ``q``, ``q_inv_neg``, ``n_pairs``, "
            "``num_windows`` and ``window_bits`` at runtime. "
            "Threadgroup-cooperative and simdgroup-cooperative "
            "implementations are valid so long as the external buffer "
            "layout above is preserved and the final bucket buffer "
            "is in Jacobian Montgomery form ready for host-side "
            "affine normalization."
        ),
        kernel_signatures=(
            "kernel void pippenger_bucket_scatter(\n"
            "    device const ulong *scalars     [[buffer(0)]],\n"
            "    device const ulong *points_in   [[buffer(1)]],\n"
            "    device       ulong *buckets     [[buffer(2)]],\n"
            "    device const ulong *q           [[buffer(3)]],\n"
            "    constant ulong     &q_inv_neg   [[buffer(4)]],\n"
            "    constant uint      &n_pairs     [[buffer(5)]],\n"
            "    constant uint      &num_windows [[buffer(6)]],\n"
            "    constant uint      &window_bits [[buffer(7)]],\n"
            "    uint idx [[thread_position_in_grid]]);\n"
            "\n"
            "Buffer sizes (host-allocated):\n"
            "  * scalars:     n_pairs * 4 ulongs\n"
            "  * points_in:   n_pairs * 18 ulongs (Jacobian Montgomery)\n"
            "  * buckets:     num_windows * (2^window_bits - 1) * 18 "
            "ulongs (zeroed before each dispatch)\n"
            "  * q:           6 ulongs\n"
            "\n"
            "Dispatch (host-fixed by the seed): one thread per "
            "(window, bucket). Total grid width is "
            "``num_windows * ((1 << window_bits) - 1)`` rounded up "
            "to the threadgroup width. Thread ``idx`` decodes to "
            "``(window_idx, bucket_value - 1) = (idx / num_buckets, "
            "idx % num_buckets)`` where ``num_buckets = "
            "(1 << window_bits) - 1``; guard against ``idx >= "
            "num_windows * num_buckets``. The seed uses "
            "``threadsPerThreadgroup = (min(grid_w, 64), 1, 1)``. "
            "Alternative thread / threadgroup layouts are valid as "
            "long as the external buffer layout is preserved and "
            "every output bucket slot is populated with the correct "
            "Jacobian Montgomery sum on completion."
        ),
        kernel_names=["pippenger_bucket_scatter"],
        seed_path=_SEED,
        sizes=[
            TaskSize("uniform_N4K",  {"distribution": "uniform", "n_pairs": 1 << 12}),
            TaskSize("uniform_N16K", {"distribution": "uniform", "n_pairs": 1 << 14}),
            TaskSize("uniform_N64K", {"distribution": "uniform", "n_pairs": 1 << 16}),
        ],
        held_out_sizes=[
            TaskSize("zipf15_N16K",
                     {"distribution": "zipf-1.5", "n_pairs": 1 << 14}),
        ],
    )

    def evaluate_size(self, harness, pipelines, size, chip, n_warmup, n_measure):
        distribution = str(size.params["distribution"])
        n_pairs = int(size.params["n_pairs"])
        if n_pairs <= 0:
            raise ValueError(
                f"pippenger_buckets requires n_pairs > 0; got {n_pairs}"
            )
        curve = BLS12_381_G1
        mont = Montgomery(curve.q, N_LIMBS)

        # Distinct seed per (n_pairs, distribution) so the disk cache
        # for the held-out probe is independent from the in-dist ones.
        # Python's built-in ``hash(str)`` is randomised per interpreter
        # invocation, so we use a stable SHA-256-derived hash to keep
        # the cache files stable across runs.
        dist_hash = int.from_bytes(
            hashlib.sha256(distribution.encode()).digest()[:4], "little",
        )
        seed = 0xC0DE9A0 ^ (n_pairs * 1009) ^ dist_hash
        inputs: PippengerInputs = gen_inputs(curve, n_pairs, seed, distribution)

        # ---------------- Host buffer allocations ----------------
        bucket_slots = NUM_WINDOWS * WINDOW_BUCKETS
        bucket_buf_words = bucket_slots * 3 * N_LIMBS

        b_scalars   = harness.buf_from_np(inputs.scalars_u64.ravel())
        b_points_in = harness.buf_from_np(inputs.points_in_u64.ravel())
        b_buckets   = harness.buf_zeros(bucket_buf_words * 8)        # ulong = 8 bytes
        b_q         = harness.buf_from_np(int_to_limbs(curve.q, N_LIMBS))
        b_q_inv_neg = harness.buf_scalar(mont.q_inv_neg_64, np.uint64)
        b_n_pairs   = harness.buf_scalar(n_pairs, np.uint32)
        b_nw        = harness.buf_scalar(NUM_WINDOWS, np.uint32)
        b_wb        = harness.buf_scalar(WINDOW_BITS, np.uint32)

        pso = pipelines["pippenger_bucket_scatter"]
        tew = int(pso.threadExecutionWidth())
        max_tg = int(pso.maxTotalThreadsPerThreadgroup())
        tg_w = min(max_tg, max(tew, 64))
        grid_total = NUM_WINDOWS * WINDOW_BUCKETS
        grid_w = ((grid_total + tg_w - 1) // tg_w) * tg_w

        view_buckets = harness.np_view(b_buckets, np.uint64, bucket_buf_words)

        def reset():
            # Zero the bucket buffer (Jacobian infinity) before every
            # dispatch so each rep observes the same starting state.
            view_buckets[:] = 0

        def dispatch(enc):
            enc.setComputePipelineState_(pso)
            enc.setBuffer_offset_atIndex_(b_scalars,   0, 0)
            enc.setBuffer_offset_atIndex_(b_points_in, 0, 1)
            enc.setBuffer_offset_atIndex_(b_buckets,   0, 2)
            enc.setBuffer_offset_atIndex_(b_q,         0, 3)
            enc.setBuffer_offset_atIndex_(b_q_inv_neg, 0, 4)
            enc.setBuffer_offset_atIndex_(b_n_pairs,   0, 5)
            enc.setBuffer_offset_atIndex_(b_nw,        0, 6)
            enc.setBuffer_offset_atIndex_(b_wb,        0, 7)
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

        # ----------- Final correctness pass + normalization -----------
        reset()
        harness.time_dispatch(dispatch)
        result = view_buckets.reshape(NUM_WINDOWS, WINDOW_BUCKETS, 3 * N_LIMBS).copy()
        affine_gpu, is_inf_gpu, non_canonical = normalize_gpu_buckets(result, mont)

        # Bit-exact compare in affine Montgomery space.
        ref_aff = inputs.expected_buckets_aff_u64
        ref_inf = inputs.expected_is_infinity
        # An infinity flag mismatch is one mismatch; an affine value
        # mismatch (when both sides are non-infinity) counts as one
        # mismatch per non-matching (k, b).
        inf_mismatch = int(np.sum(is_inf_gpu != ref_inf))
        both_finite = (~is_inf_gpu) & (~ref_inf)
        # affine[k][b][i] mismatch across the 12 limbs collapses to a
        # single per-bucket mismatch (np.any over the last axis).
        aff_neq = np.any(affine_gpu != ref_aff, axis=-1) & both_finite
        aff_mismatch = int(np.sum(aff_neq))
        mismatches = inf_mismatch + aff_mismatch + int(non_canonical)
        correct = (mismatches == 0)

        # ---------------- Roofline ----------------
        muls_total = NUM_WINDOWS * n_pairs * MODMULS_PER_BUCKET_ADD
        achieved_mul = gops_per_s(muls_total, gpu_s)
        ceiling_mul = float(chip.peak_int64_mul_gops)

        # BW: scalars (32 N) + points_in (144 N) read once per dispatch;
        # each (window, pair) scatter loads + stores one Jacobian
        # bucket (288 bytes). Bucket I/O dominates because it happens
        # ``num_windows`` times per pair.
        bytes_in = (32.0 + 144.0) * float(n_pairs)
        bytes_bucket_rw = 288.0 * float(NUM_WINDOWS) * float(n_pairs)
        bytes_total = bytes_in + bytes_bucket_rw
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
                "distribution": distribution,
                "n_pairs": n_pairs,
                "num_windows": NUM_WINDOWS,
                "window_bits": WINDOW_BITS,
                "muls_total": muls_total,
                "bytes_total": bytes_total,
                "achieved_mul_gops": achieved_mul,
                "achieved_bw_gb_s": achieved_bw,
                "non_canonical": int(non_canonical),
                "inf_mismatch": inf_mismatch,
                "aff_mismatch": aff_mismatch,
                "anchor": anchor,
            },
        )
