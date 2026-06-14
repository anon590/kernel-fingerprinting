"""Kyber / Dilithium negacyclic NTT -- Z6 task.

Batched forward Cooley-Tukey NTT over the small Kyber / Dilithium prime
field. The kernel runs a single threadgroup per polynomial (n / 2
threads); each thread owns one butterfly per stage. All parameters
that distinguish Kyber from Dilithium -- the modulus ``q``, the
twiddle table, the level count -- are bound at runtime through device
or constant buffers.

Sizes (PLAN.md "Regime taxonomy"):
  * in-distribution: Kyber (q=3329, n=256, n_levels=7),
                     batch in {1, 16, 256}
  * held-out:        Dilithium (q=8380417, n=256, n_levels=8),
                     batch = 64

The held-out probe flips on candidates that:
  - hardcode ``q = 3329`` in their reduction (Dilithium needs 32-bit
    Barrett constants);
  - hardcode 16-bit ``ushort`` storage / packing (Dilithium
    coefficients span 23 bits, do not fit);
  - hardcode the 7-level loop structure (Dilithium does 8 stages with
    a final ``len = 1`` butterfly);
  - hardcode the 128-entry zetas table length (Dilithium's table has
    256 entries; reading past 128 returns garbage for a hardcoded
    bound).

Roofline
========

Per polynomial:
  * Modmul anchor: 1 ``zeta * a[j+len]`` modmul per butterfly,
    ``n / 2`` butterflies per stage, ``n_levels`` stages
    -> ``n_levels * n / 2`` modmuls. Total: ``batch * n_levels * n/2``.
  * BW anchor (single-pass model):
      read+write coeffs : ``2 * 4 * n`` bytes per polynomial
      read zetas        : ``4 * (1 << n_levels)`` bytes per polynomial
                          (cache-resident across the batch in practice;
                          counted once per polynomial as the worst case)
    Total bytes: ``batch * (8 n + 4 * (1 << n_levels))``.

We report the achieved fraction against the **binding** roofline
``max(achieved_mul / peak_int64_mul, achieved_bw / peak_bw)`` -- same
convention as goldilocks_ntt / merkle_build / fri_round.

Note on the modmul ceiling: the per-modmul work here is 32-bit
(``uint32 * uint32 -> uint64``) rather than the 64-bit work that
``peak_int64_mul_gops`` was calibrated on. The reported fraction is
therefore a *lower* bound on the candidate's true utilisation of the
int-mul pipe; a candidate that exploits the smaller modulus to pack
multiple lanes into a single 64-bit multiply (the canonical Apple-GPU
lever for q < 2^16) can push the *effective* modmul count well above
the naive ``batch * n_levels * n/2`` -- which is exactly the
optimisation surface Z6 is supposed to expose. The held-out probe at
q = 8380417 rules this packing out, so a candidate whose entire score
came from packing will collapse on the held-out.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import Metal

from ..reference.kyber import (
    KYBER, DILITHIUM, NttParams,
    make_zetas, ntt_forward_cached, random_inputs,
)
from ..redact import Redaction
from ..task import (
    Task, TaskSize, TaskSpec, SizeResult,
    gb_per_s, gops_per_s, register_task,
)


_SEED = Path(__file__).resolve().parent.parent.parent / "seeds" / "kyber_ntt.metal"

# --- Controlled grade-C experiment: redact the held-out (Dilithium) modulus.
# The description never names 8380417; the only LLM-visible disclosure is the
# seed's buffer comment. The redacted form keeps the runtime-modulus contract
# (already stated in the description: "q fits in a 32-bit unsigned integer")
# without naming the held-out Dilithium prime. See metal_zk/redact.py.
_REDACTIONS = [
    Redaction(
        find="(modulus; 3329 or 8380417)",
        replace="(modulus; bound at runtime, fits in 32 bits)",
        note="removes Dilithium held-out modulus from seed comment",
    ),
]
_DENYLIST = ["8380417", "Dilithium"]


def _params_for(variant: str) -> NttParams:
    if variant == "kyber":
        return KYBER
    if variant == "dilithium":
        return DILITHIUM
    raise ValueError(f"unknown kyber_ntt variant: {variant!r}")


@register_task("kyber_ntt")
class KyberNttTask(Task):
    spec = TaskSpec(
        name="kyber_ntt",
        description=(
            "Batched forward Cooley-Tukey NTT over a small prime field "
            "for a negacyclic polynomial ring Z_q[X] / (X^n + 1). Each "
            "polynomial has length n; the modulus q, the polynomial "
            "length n, the number of NTT stages n_levels, and the "
            "precomputed twiddle table are all bound at runtime so a "
            "single kernel runs every parameter set the host supplies.\n\n"
            "Convention (matches the FIPS 203 / FIPS 204 / pqclean "
            "reference C implementations):\n"
            "  k = 1\n"
            "  for level = 0..n_levels:\n"
            "      len = n >> (level + 1)\n"
            "      for start = 0, 2*len, ..., n - 2*len:\n"
            "          z = zetas[k++]\n"
            "          for j = start..start + len - 1:\n"
            "              t          = (z * a[j+len]) mod q\n"
            "              a[j+len]   = (a[j] - t)       mod q\n"
            "              a[j]       = (a[j] + t)       mod q\n"
            "Equivalent per-thread mapping (ltid in [0, n/2) owns one "
            "butterfly per level):\n"
            "  group_idx   = ltid / len\n"
            "  j_in_group  = ltid - group_idx * len   (= ltid mod len)\n"
            "  j           = (group_idx << 1) * len + j_in_group\n"
            "  zeta_index  = (1 << level) + group_idx\n\n"
            "Zetas table (host-precomputed, length 1 << n_levels):\n"
            "  zetas[k] = zeta^bit_reverse(k, n_levels)  mod q\n"
            "where zeta is a primitive 2^(n_levels+1)-th root of unity "
            "in F_q. The forward NTT consumes zetas[1..(1 << n_levels) "
            "- 1] in increasing index order; zetas[0] = 1 is the unread "
            "identity element.\n\n"
            "Bounds for kernel design: q fits in a 32-bit unsigned "
            "integer; n is a power of two with n <= 256; n_levels <= 8 "
            "(so the zetas table has at most 256 entries). The kernel "
            "MUST read q, n, and n_levels from their bound buffers and "
            "load every twiddle from the zetas buffer at runtime; "
            "compile-time constants for any of these values are "
            "incorrect.\n\n"
            "Storage: ``uint32`` per coefficient, in-place. The host "
            "writes the input coefficients into ``coeffs`` and reads "
            "the result back from the same buffer; ``coeffs`` is "
            "(batch * n) uint values in row-major order (polynomial "
            "p's coefficients live at offsets p*n .. p*n + n - 1).\n\n"
            "All inputs are canonical: a[i] in [0, q). Outputs MUST "
            "also be canonical -- a value in [q, 2^32) with the same "
            "residue class still counts as a mismatch on the bit-exact "
            "reference comparison."
        ),
        kernel_signatures=(
            "kernel void kyber_ntt(\n"
            "    device       uint *coeffs     [[buffer(0)]],\n"
            "    device const uint *zetas      [[buffer(1)]],\n"
            "    constant uint     &q           [[buffer(2)]],\n"
            "    constant uint     &n           [[buffer(3)]],\n"
            "    constant uint     &n_levels    [[buffer(4)]],\n"
            "    constant uint     &batch       [[buffer(5)]],\n"
            "    uint tgid [[threadgroup_position_in_grid]],\n"
            "    uint ltid [[thread_position_in_threadgroup]]);\n"
            "\n"
            "Dispatch (host-provided):\n"
            "  threadsPerGrid        = (batch * (n/2), 1, 1)\n"
            "  threadsPerThreadgroup = (n/2, 1, 1)\n"
            "Each threadgroup owns ONE polynomial; tgid in [0, batch) "
            "selects the polynomial, ltid in [0, n/2) owns one butterfly "
            "per level. Every test size uses n = 256, so n/2 = 128 "
            "threads per threadgroup is sufficient; a static threadgroup "
            "scratch of size 256 covers every case. Threadgroup-"
            "cooperative and simdgroup schemes are valid as long as the "
            "buffer layout and the canonical-output contract are "
            "preserved."
        ),
        kernel_names=["kyber_ntt"],
        seed_path=_SEED,
        redactions=_REDACTIONS,
        held_out_denylist=_DENYLIST,
        sizes=[
            TaskSize("kyb_B1",   {"variant": "kyber", "batch": 1}),
            TaskSize("kyb_B16",  {"variant": "kyber", "batch": 16}),
            TaskSize("kyb_B256", {"variant": "kyber", "batch": 256}),
        ],
        held_out_sizes=[
            TaskSize("dil_B64",  {"variant": "dilithium", "batch": 64}),
        ],
    )

    def evaluate_size(self, harness, pipelines, size, chip, n_warmup, n_measure):
        variant = str(size.params["variant"])
        batch = int(size.params["batch"])
        params = _params_for(variant)
        n = params.n
        n_levels = params.n_levels
        q = params.q
        half_n = n // 2

        # Inputs + reference. Seed salted with (variant, batch) so the
        # held-out probe has its own distinct input set.
        seed = 0xCAFEB10C + batch * 17 + (1_000_003 if variant == "dilithium" else 0)
        coeffs_in = random_inputs(batch, params, seed=seed)
        coeffs_ref = ntt_forward_cached(coeffs_in, params)

        zetas = make_zetas(params)

        # Buffers.
        b_coeffs   = harness.buf_from_np(np.ascontiguousarray(coeffs_in.ravel(),
                                                              dtype=np.uint32))
        b_zetas    = harness.buf_from_np(np.ascontiguousarray(zetas, dtype=np.uint32))
        b_q        = harness.buf_scalar(q,        np.uint32)
        b_n        = harness.buf_scalar(n,        np.uint32)
        b_nlvls    = harness.buf_scalar(n_levels, np.uint32)
        b_batch    = harness.buf_scalar(batch,    np.uint32)

        pso = pipelines["kyber_ntt"]
        max_tg = int(pso.maxTotalThreadsPerThreadgroup())
        if half_n > max_tg:
            raise RuntimeError(
                f"kyber_ntt: pipeline supports {max_tg} threads/tg, "
                f"need {half_n} (n/2 with n = {n}). Candidate kernel is "
                f"using too much threadgroup memory or registers."
            )
        tg_w   = half_n
        grid_w = batch * tg_w

        view_coeffs = harness.np_view(b_coeffs, np.uint32, batch * n)

        def reset():
            view_coeffs[:] = coeffs_in.ravel()

        def dispatch(enc):
            enc.setComputePipelineState_(pso)
            enc.setBuffer_offset_atIndex_(b_coeffs, 0, 0)
            enc.setBuffer_offset_atIndex_(b_zetas,  0, 1)
            enc.setBuffer_offset_atIndex_(b_q,      0, 2)
            enc.setBuffer_offset_atIndex_(b_n,      0, 3)
            enc.setBuffer_offset_atIndex_(b_nlvls,  0, 4)
            enc.setBuffer_offset_atIndex_(b_batch,  0, 5)
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
        got = view_coeffs.copy().reshape(batch, n)

        non_canonical = int(np.sum(got >= np.uint32(q)))
        mismatches = int(np.sum(got != coeffs_ref))
        correct = (mismatches == 0)

        # Modmul roofline.
        muls_per_poly = n_levels * half_n
        muls_total    = batch * muls_per_poly
        achieved_mul = gops_per_s(muls_total, gpu_s)
        ceiling_mul  = float(chip.peak_int64_mul_gops)

        # BW roofline. Coeffs are 4 bytes each, read once + written once;
        # zetas are 4 bytes each, read once per polynomial in the worst
        # case (cache-resident across batch in practice -- but using the
        # worst case keeps the roofline an honest *upper bound* on the
        # candidate's memory traffic).
        bytes_total = float(
            batch * (2 * 4 * n + 4 * params.n_zetas)
        )
        achieved_bw = gb_per_s(bytes_total, gpu_s)
        ceiling_bw  = float(chip.peak_bw_gb_s)

        frac_mul = achieved_mul / ceiling_mul if ceiling_mul > 0 else 0.0
        frac_bw  = achieved_bw  / ceiling_bw  if ceiling_bw  > 0 else 0.0
        if frac_mul >= frac_bw:
            achieved, achieved_unit = achieved_mul, "Gmodmul/s (u32)"
            ceiling,  ceiling_unit  = ceiling_mul,  "Gops/s (int64 mul, est)"
            fraction = frac_mul
            anchor = "int64_mul"
        else:
            achieved, achieved_unit = achieved_bw, "GB/s"
            ceiling,  ceiling_unit  = ceiling_bw,  "GB/s"
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
                "variant": variant, "q": q, "n": n, "n_levels": n_levels,
                "batch": batch,
                "n_zetas": params.n_zetas,
                "muls_total": muls_total,
                "bytes_total": bytes_total,
                "achieved_mul_gops": achieved_mul,
                "achieved_bw_gb_s": achieved_bw,
                "non_canonical": non_canonical,
                "anchor": anchor,
            },
        )
