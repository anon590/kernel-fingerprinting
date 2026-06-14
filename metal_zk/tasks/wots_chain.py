"""WOTS+ / SPHINCS+ chain — Z10 task.

Batched sequential hash chains: given ``n_chains`` ``n_bytes``-byte
seeds, apply the Keccak-256 inner hash ``w`` times in sequence (each
digest truncated back to ``n_bytes`` before re-absorbing) and emit the
chain tip. Independent across chains, strictly sequential within each
chain — the exact lever the PLAN.md Z10 entry asks the candidate to
explore (thread-per-chain vs simdgroup-cooperative lockstep vs blocked).

Sizes (PLAN.md S Regime taxonomy):
  * in-distribution: n_bytes = 16, n_chains = 2^16, w in {16, 64, 256}
  * held-out:        n_bytes = 32, n_chains = 2^17, w = 32
    (SPHINCS+-256s style: 256-bit hash output + different chain count)

The held-out probe flips together on three independent overfit modes:

  - the absorb / squeeze lane count (``n_lanes`` 2 -> 4): a candidate
    that hardcodes "two ulongs in, two ulongs out" silently fails;
  - the byte position of the SHA3 domain pad (lane 2 -> lane 4): a
    candidate that hardcodes the in-distribution padding lane silently
    corrupts the digest;
  - the chain count (2^16 -> 2^17): a candidate that bakes a fixed
    grid dimension or threadgroup tile into its host-side launch
    silently mis-dispatches.

The chain-length axis ``w`` is varied within the in-distribution sweep
itself (not held-out) so that schedule-shape candidates (latency hiding
across chain depth) get the right gradient on the in-dist sizes.

Roofline:
  Each chain step is exactly one Keccak-f[1600] permutation (single
  absorb block because ``n_bytes < rate_bytes``, single squeeze block
  because ``n_bytes < rate_bytes``). Total perms = ``n_chains * w``.
  Structural bitop count per permutation matches Z8's accounting:
  ``24 * (20 + 5 + 5 + 25 + 24 + 75 + 1) ~ 3720`` u64 bit operations.

  Bytes touched per chain: ``2 * n_bytes`` (one seed read + one tip
  write); the chained intermediate digests live in thread-private
  registers and never touch DRAM in a sensible implementation. The
  inner-hash round constants and rho offsets are tiny constants.

  The task reports the achieved fraction against the **binding**
  roofline ``max(achieved_bitop / peak_int64_bitop, achieved_bw /
  peak_bw)``. Z10 is essentially the Z8 inner kernel wrapped in a
  sequential loop with negligible IO, so the int-bitop anchor
  dominates at every test size; the BW anchor is surfaced in
  ``extra`` for cross-task sanity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import Metal

from ..harness import MetalHarness
from ..hardware import ChipSpec
from ..reference.wots import (
    BITOPS_PER_PERMUTATION,
    chain_batch_cached, random_seeds,
)
from ..redact import Redaction
from ..task import (
    Task, TaskSize, TaskSpec, SizeResult,
    gb_per_s, gops_per_s, register_task,
)


_SEED = Path(__file__).resolve().parent.parent.parent / "seeds" / "wots_chain.metal"

# --- Controlled grade-C experiment: redact the held-out (n_bytes=32) identity
# from the description (two parentheticals) and the seed comments (two lines).
# The runtime-parameterization contract --- "read n_bytes and w from buffers,
# hardcoding produces wrong output" --- is preserved; only the value the
# held-out probe uses is removed. Finds are minimal substrings so they match
# both the one-line description and the "//"-prefixed seed comments. See
# metal_zk/redact.py.
_REDACTIONS = [
    Redaction(
        find="(in-distribution n_bytes=16, held-out n_bytes=32; rate_bytes=136)",
        replace=("(``n_bytes`` is bound at runtime and varies across the "
                 "configurations the kernel is scored on; rate_bytes=136)"),
        note="description: removes held-out n_bytes=32",
    ),
    Redaction(
        find=("(``w`` in {16, 64, 256} in the in-distribution sweep, "
              "``n_bytes`` 16 -> 32 between in-distribution and held-out)"),
        replace=("(``w`` in {16, 64, 256} among the baseline sizes shown; "
                 "both ``w`` and ``n_bytes`` are bound at runtime and vary "
                 "across the configurations the kernel is scored on)"),
        note="description: removes the n_bytes 16->32 held-out hint",
    ),
    Redaction(
        find="held-out n_bytes=32; rate_bytes=136), so every chain step is a",
        replace="rate_bytes=136), so every chain step is a",
        note="seed comment: removes held-out n_bytes=32",
    ),
    Redaction(
        find="(chunk size; 16 in-dist, 32 held-out)",
        replace="(chunk size; bound at runtime)",
        note="seed comment: removes held-out n_bytes=32",
    ),
]
_DENYLIST = ["held-out", "n_bytes=32", "16 -> 32", "32 held-out", "16 in-dist"]


@register_task("wots_chain")
class WotsChainTask(Task):
    spec = TaskSpec(
        name="wots_chain",
        description=(
            "Batched WOTS+ / SPHINCS+-style hash chains. Given "
            "``n_chains`` independent ``n_bytes``-byte seeds, apply "
            "the Keccak-256 inner hash ``w`` times in sequence per "
            "chain (each digest truncated to ``n_bytes`` bytes before "
            "feeding into the next iteration) and write the chain tip "
            "to the output. The chains are embarrassingly parallel; "
            "the ``w``-step iteration along each chain is strictly "
            "sequential.\n\n"
            "Inner hash: Keccak-f[1600] with the FIPS 202 SHA3-256 "
            "sponge framing -- rate = 136 bytes (17 lanes), "
            "capacity = 64 bytes, domain pad byte = 0x06. State "
            "convention: the 1600-bit state is a 5x5 array of "
            "64-bit lanes; lane k = x + 5*y holds bytes "
            "8*k .. 8*k + 7 of the sponge state in little-endian.\n\n"
            "All test sizes have ``n_bytes < rate_bytes`` "
            "(in-distribution n_bytes=16, held-out n_bytes=32; "
            "rate_bytes=136), so every chain step collapses to a "
            "single-block absorb + single-block squeeze of "
            "``n_lanes = n_bytes / 8`` state lanes:\n"
            "  state                          := 0\n"
            "  state[lane 0..n_lanes-1]       := previous_chunk\n"
            "  state[lane n_lanes, byte 0]    ^= 0x06   # SHA3 domain\n"
            "  state[lane 16, byte 7]         ^= 0x80   # FIPS 202 final pad\n"
            "  state                          := Keccak-f1600(state)\n"
            "  next_chunk                     := state[lane 0..n_lanes-1]\n\n"
            "On the first chain step the absorb is the seed; on every "
            "subsequent step the absorb is the n_lanes-lane truncation "
            "of the previous Keccak-f1600 output. After ``w`` steps the "
            "first n_lanes state lanes are written to the output as "
            "the chain tip.\n\n"
            "The kernel must read ``n_bytes`` and ``w`` from the bound "
            "device buffers rather than treating them as compile-time "
            "constants; both vary across the test sizes "
            "(``w`` in {16, 64, 256} in the in-distribution sweep, "
            "``n_bytes`` 16 -> 32 between in-distribution and held-out). "
            "Hardcoding either value silently produces wrong output, "
            "not just slow output.\n\n"
            "Correctness is bit-exact against ``hashlib.sha3_256`` "
            "iterated ``w`` times with ``n_bytes``-byte truncation; "
            "any mismatched output ulong rejects the candidate."
        ),
        kernel_signatures=(
            "kernel void wots_chain(\n"
            "    device const ulong *seeds    [[buffer(0)]],\n"
            "    device       ulong *tips     [[buffer(1)]],\n"
            "    constant uint      &n_chains [[buffer(2)]],\n"
            "    constant uint      &n_bytes  [[buffer(3)]],\n"
            "    constant uint      &w        [[buffer(4)]],\n"
            "    uint idx [[thread_position_in_grid]]);\n"
            "\n"
            "Dispatch (host-fixed):\n"
            "  threadsPerGrid        = (n_chains, 1, 1)\n"
            "  threadsPerThreadgroup = (min(n_chains, 64), 1, 1)\n"
            "Each thread processes ONE chain end-to-end; guard "
            "against idx >= n_chains (the grid is rounded up to a "
            "multiple of the TG width). ``seeds`` is laid out as "
            "n_chains consecutive runs of ``n_bytes / 8`` ulongs; "
            "``tips`` likewise. The external buffer layout above "
            "must be preserved and the per-chain sequential "
            "semantics honored: each chain's step ``j+1`` must read "
            "the digest produced by its own step ``j`` (cross-chain "
            "mixing of intermediate digests would be a correctness "
            "bug)."
        ),
        kernel_names=["wots_chain"],
        seed_path=_SEED,
        redactions=_REDACTIONS,
        held_out_denylist=_DENYLIST,
        sizes=[
            TaskSize(
                "w16_C64K",
                {"n_chains": 1 << 16, "n_bytes": 16, "w": 16,
                 "seed": 0xC4A1_0000 + 16},
            ),
            TaskSize(
                "w64_C64K",
                {"n_chains": 1 << 16, "n_bytes": 16, "w": 64,
                 "seed": 0xC4A1_0000 + 64},
            ),
            TaskSize(
                "w256_C64K",
                {"n_chains": 1 << 16, "n_bytes": 16, "w": 256,
                 "seed": 0xC4A1_0000 + 256},
            ),
        ],
        held_out_sizes=[
            TaskSize(
                "sphincs256s_w32_C128K",
                {"n_chains": 1 << 17, "n_bytes": 32, "w": 32,
                 "seed": 0xC4A1_5005 + 32},
            ),
        ],
    )

    def evaluate_size(self, harness, pipelines, size, chip, n_warmup, n_measure):
        n_chains = int(size.params["n_chains"])
        n_bytes = int(size.params["n_bytes"])
        w = int(size.params["w"])
        seed = int(size.params["seed"])

        if n_bytes % 8:
            raise ValueError(f"n_bytes must be a multiple of 8; got {n_bytes}")
        if not 0 < n_bytes < 136:
            raise ValueError(
                f"need 0 < n_bytes < 136 (SHA3-256 rate); got {n_bytes}"
            )
        if w < 1:
            raise ValueError(f"w must be >= 1; got {w}")

        n_lanes = n_bytes // 8

        seeds_u64 = random_seeds(n_chains, n_bytes, seed=seed)
        y_ref = chain_batch_cached(seeds_u64, n_bytes, w)

        bA = harness.buf_from_np(seeds_u64.ravel())
        bB = harness.buf_zeros(int(n_chains * n_lanes * 8))
        b_chains = harness.buf_scalar(n_chains, np.uint32)
        b_nbytes = harness.buf_scalar(n_bytes, np.uint32)
        b_w = harness.buf_scalar(w, np.uint32)

        pso = pipelines["wots_chain"]
        max_tg = int(pso.maxTotalThreadsPerThreadgroup())
        tew = int(pso.threadExecutionWidth())
        tg_w = min(max_tg, max(tew, 64))
        grid_w = ((n_chains + tg_w - 1) // tg_w) * tg_w

        view_B = harness.np_view(bB, np.uint64, n_chains * n_lanes)

        def reset():
            view_B[:] = 0

        def dispatch(enc):
            enc.setComputePipelineState_(pso)
            enc.setBuffer_offset_atIndex_(bA, 0, 0)
            enc.setBuffer_offset_atIndex_(bB, 0, 1)
            enc.setBuffer_offset_atIndex_(b_chains, 0, 2)
            enc.setBuffer_offset_atIndex_(b_nbytes, 0, 3)
            enc.setBuffer_offset_atIndex_(b_w, 0, 4)
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
        got = view_B.copy().reshape(n_chains, n_lanes)

        mismatches = int(np.sum(got != y_ref))
        correct = (mismatches == 0)

        # Roofline: int-bitop anchor (compute-bound) and BW anchor (IO);
        # pick the binding one.
        n_perms_per_chain = w
        n_perms_total = n_chains * n_perms_per_chain
        bitops_total = n_perms_total * BITOPS_PER_PERMUTATION
        achieved_bitop = gops_per_s(bitops_total, gpu_s)
        ceiling_bitop = float(chip.peak_int64_bitop_gops)

        bytes_total = float(n_chains) * float(2 * n_bytes)
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
                "n_chains": n_chains,
                "n_bytes": n_bytes,
                "w": w,
                "n_perms_per_chain": n_perms_per_chain,
                "n_perms_total": n_perms_total,
                "bitops_total": bitops_total,
                "bytes_total": bytes_total,
                "achieved_bitop_gops": achieved_bitop,
                "achieved_bw_gb_s": achieved_bw,
                "anchor": anchor,
            },
        )
