"""Keccak-f[1600] batched sponge — Z8 task.

Batched Keccak sponge, one thread per instance. In-distribution sizes
hold the SHA3-256 mode parameters (rate=136 bytes, domain=0x06,
out=32 bytes); the held-out probe switches to SHAKE128
(rate=168 bytes, domain=0x1F, out=256 bytes), which forces multiple
permutations during squeeze.

Sizes (PLAN.md S Regime taxonomy):
  * in-distribution: SHA3-256, batch in {2^14, 2^18, 2^22}
  * held-out:        SHAKE128, batch = 2^20, out_bytes = 256

All test sizes have ``msg_bytes = 32`` and use a ``ulong``-aligned
host-side buffer layout (msg_bytes, rate_bytes, out_bytes all
multiples of 8).

Roofline:
  Per Keccak-f[1600] permutation, the structural bitop count is about
  155 ops/round * 24 rounds = ~3720 u64 bit operations (theta XORs +
  rotates, rho rotates, chi NOT/AND/XOR, iota XOR). Each instance runs
  ``ceil(out_bytes / rate_bytes)`` permutations end-to-end -- one for
  the in-distribution SHA3-256 sizes, two for the held-out SHAKE128
  size that squeezes 256 bytes through a 168-byte rate. Bytes touched
  per instance: ``msg_bytes + out_bytes`` (load message, write output);
  constants (RC, rho offsets) are small and amortise across threads.

  The task reports the achieved fraction against the **binding**
  roofline ``max(achieved_bitop / peak_int64_bitop, achieved_bw /
  peak_bw)``. At realistic sizes Keccak-f[1600] is firmly bitop-bound
  on Apple Silicon, but at very large batches the output write can
  approach the DRAM ceiling, so we expose both anchors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import Metal

from ..harness import MetalHarness
from ..hardware import ChipSpec
from ..reference.keccak import (
    SHA3_256_RATE_BYTES, SHA3_256_DOMAIN,
    SHAKE128_RATE_BYTES, SHAKE128_DOMAIN,
    hash_batch_cached, random_messages,
)
from ..redact import Redaction
from ..task import (
    Task, TaskSize, TaskSpec, SizeResult,
    gb_per_s, gops_per_s, register_task,
)


_SEED = Path(__file__).resolve().parent.parent.parent / "seeds" / "keccak_f1600_batch.metal"

# --- Controlled grade-C experiment: redact the held-out (SHAKE128) identity.
# The disclosing sentence names the held-out mode and its exact parameters;
# the redacted form keeps the runtime-parameterization contract (the kernel
# must still read rate/out/domain from buffers and may face unseen configs)
# but does not reveal that the unseen config is SHAKE128 (rate=168,
# domain=0x1F, out=256). See metal_zk/redact.py.
_REDACTIONS = [
    Redaction(
        find=(
            "In-distribution sizes use the SHA3-256 mode (rate=136, "
            "domain=0x06, out=32); the held-out size uses SHAKE128 "
            "(rate=168, domain=0x1F, out=256, requires multiple squeeze "
            "permutations). The kernel must use the runtime values of "
            "``rate_bytes``, ``out_bytes`` and ``domain`` rather than "
            "compile-time constants."
        ),
        replace=(
            "The baseline sizes below use the SHA3-256 mode (rate=136, "
            "domain=0x06, out=32). The kernel is scored on several "
            "(rate_bytes, out_bytes, domain) parameter sets, including "
            "configurations not listed among the baseline sizes, and "
            "out_bytes may exceed rate_bytes. The kernel must use the "
            "runtime values of ``rate_bytes``, ``out_bytes`` and "
            "``domain`` rather than compile-time constants."
        ),
        note="removes SHAKE128 held-out identity; keeps generic runtime contract",
    ),
]
_DENYLIST = ["SHAKE128", "rate=168", "domain=0x1F", "out=256", "held-out"]

# Per-permutation bitop count (theta + rho + chi + iota, summed over 24 rounds).
_BITOPS_PER_PERMUTATION: int = 24 * (20 + 5 + 5 + 25 + 24 + 75 + 1)


@register_task("keccak_f1600_batch")
class KeccakF1600BatchTask(Task):
    spec = TaskSpec(
        name="keccak_f1600_batch",
        description=(
            "Batched Keccak-f[1600] sponge over fixed-length messages. "
            "Each of ``batch`` independent instances absorbs ``msg_bytes`` "
            "bytes of input, applies the standard FIPS 202 padding, runs "
            "the 24-round Keccak-f[1600] permutation, and squeezes "
            "``out_bytes`` bytes of output. All test sizes satisfy "
            "``msg_bytes < rate_bytes`` (single absorb block) and "
            "``msg_bytes``, ``rate_bytes``, ``out_bytes`` are all "
            "multiples of 8, so the host packs message and output as "
            "``ulong`` arrays.\n\n"
            "State convention: the 1600-bit state is a 5x5 array of "
            "64-bit lanes; lane k (for k in 0..25) corresponds to byte "
            "positions 8*k .. 8*k + 7 of the sponge state in "
            "little-endian, i.e. lane k holds bytes at the (x, y) cell "
            "with x = k % 5 and y = k / 5. The seed shows the standard "
            "round constants ``RC[24]`` and rho offsets "
            "``r[x][y]`` from FIPS 202.\n\n"
            "Permutation: 24 rounds of theta -> rho -> pi -> chi -> iota "
            "as defined in FIPS 202. Concretely, with A the (5,5) state "
            "of 64-bit lanes:\n"
            "  theta:  C[x]      = A[x,0] ^ A[x,1] ^ A[x,2] ^ A[x,3] ^ A[x,4];\n"
            "          D[x]      = C[x-1] ^ rotl(C[x+1], 1);\n"
            "          A[x,y]   ^= D[x].\n"
            "  rho:    A'[x,y]   = rotl(A[x,y], r[x][y]).\n"
            "  pi:     A''[y, (2*x + 3*y) %% 5] = A'[x, y]\n"
            "          (equivalently A''[x, y] = A'[(x + 3*y) %% 5, x]).\n"
            "  chi:    A'''[x,y] = A''[x,y] ^ ((~A''[(x+1)%%5, y]) & A''[(x+2)%%5, y]).\n"
            "  iota:   A''''[0,0] = A'''[0,0] ^ RC[round].\n\n"
            "Sponge protocol (msg_bytes < rate_bytes, single absorb block):\n"
            "  1. Initialise the state to zero.\n"
            "  2. XOR ``msg_bytes / 8`` input lanes into state lanes "
            "     0 .. msg_bytes/8 - 1 (little-endian byte stream).\n"
            "  3. XOR the domain byte (low 8 bits of ``domain``) into "
            "     byte position ``msg_bytes`` (lane ``msg_bytes/8``, "
            "     byte 0 of that lane).\n"
            "  4. XOR 0x80 into byte position ``rate_bytes - 1`` "
            "     (lane ``rate_bytes/8 - 1``, byte 7 of that lane).\n"
            "  5. Apply Keccak-f[1600].\n"
            "  6. Output the first ``rate_bytes / 8`` lanes of state.\n"
            "  7. If more output is needed, apply Keccak-f[1600] again "
            "     and output the next ``rate_bytes / 8`` lanes; repeat "
            "     until ``out_bytes / 8`` lanes have been written. The "
            "     final chunk may be shorter than the rate.\n\n"
            "In-distribution sizes use the SHA3-256 mode "
            "(rate=136, domain=0x06, out=32); the held-out size uses "
            "SHAKE128 (rate=168, domain=0x1F, out=256, requires "
            "multiple squeeze permutations). The kernel must use the "
            "runtime values of ``rate_bytes``, ``out_bytes`` and "
            "``domain`` rather than compile-time constants. "
            "Correctness is bit-exact against ``hashlib.sha3_256`` / "
            "``hashlib.shake_128``; any mismatched output ulong rejects "
            "the candidate."
        ),
        kernel_signatures=(
            "kernel void keccak_f1600_batch(\n"
            "    device const ulong *in_data    [[buffer(0)]],\n"
            "    device       ulong *out_data   [[buffer(1)]],\n"
            "    constant uint      &batch      [[buffer(2)]],\n"
            "    constant uint      &msg_bytes  [[buffer(3)]],\n"
            "    constant uint      &rate_bytes [[buffer(4)]],\n"
            "    constant uint      &out_bytes  [[buffer(5)]],\n"
            "    constant uint      &domain     [[buffer(6)]],\n"
            "    uint idx [[thread_position_in_grid]]);\n"
            "\n"
            "Dispatch (host-fixed):\n"
            "  threadsPerGrid        = (batch, 1, 1)\n"
            "  threadsPerThreadgroup = (min(batch, 64), 1, 1)\n"
            "Each thread processes ONE instance end-to-end; guard "
            "against idx >= batch (the grid is rounded up to a "
            "multiple of the TG width). All test sizes have "
            "msg_bytes = 32. ``in_data`` is laid out as batch "
            "consecutive runs of ``msg_bytes / 8`` ulongs; ``out_data`` "
            "as batch consecutive runs of ``out_bytes / 8`` ulongs. "
            "Threadgroup-cooperative and simdgroup-cooperative "
            "implementations are valid so long as the external buffer "
            "layout above is preserved."
        ),
        kernel_names=["keccak_f1600_batch"],
        seed_path=_SEED,
        redactions=_REDACTIONS,
        held_out_denylist=_DENYLIST,
        sizes=[
            TaskSize(
                "sha3_256_B16K",
                {"batch": 1 << 14, "msg_bytes": 32,
                 "rate_bytes": SHA3_256_RATE_BYTES, "out_bytes": 32,
                 "domain": SHA3_256_DOMAIN, "seed": 0xCAFE0000 + (1 << 14)},
            ),
            TaskSize(
                "sha3_256_B256K",
                {"batch": 1 << 18, "msg_bytes": 32,
                 "rate_bytes": SHA3_256_RATE_BYTES, "out_bytes": 32,
                 "domain": SHA3_256_DOMAIN, "seed": 0xCAFE0000 + (1 << 18)},
            ),
            TaskSize(
                "sha3_256_B4M",
                {"batch": 1 << 22, "msg_bytes": 32,
                 "rate_bytes": SHA3_256_RATE_BYTES, "out_bytes": 32,
                 "domain": SHA3_256_DOMAIN, "seed": 0xCAFE0000 + (1 << 22)},
            ),
        ],
        held_out_sizes=[
            TaskSize(
                "shake128_B1M_out256",
                {"batch": 1 << 20, "msg_bytes": 32,
                 "rate_bytes": SHAKE128_RATE_BYTES, "out_bytes": 256,
                 "domain": SHAKE128_DOMAIN, "seed": 0xCAFE0000 + (1 << 20) + 7},
            ),
        ],
    )

    def evaluate_size(self, harness, pipelines, size, chip, n_warmup, n_measure):
        batch = int(size.params["batch"])
        msg_bytes = int(size.params["msg_bytes"])
        rate_bytes = int(size.params["rate_bytes"])
        out_bytes = int(size.params["out_bytes"])
        domain = int(size.params["domain"])
        seed = int(size.params["seed"])

        if msg_bytes % 8 or rate_bytes % 8 or out_bytes % 8:
            raise ValueError(
                "msg_bytes / rate_bytes / out_bytes must all be multiples "
                f"of 8; got {msg_bytes}, {rate_bytes}, {out_bytes}"
            )
        if not 0 < msg_bytes < rate_bytes:
            raise ValueError(
                f"need 0 < msg_bytes < rate_bytes; got {msg_bytes} vs "
                f"{rate_bytes}"
            )

        out_lanes = out_bytes // 8

        messages = random_messages(batch, msg_bytes, seed=seed)
        y_ref = hash_batch_cached(
            messages, msg_bytes, rate_bytes, out_bytes, domain,
        )

        bA = harness.buf_from_np(messages.ravel())
        bB = harness.buf_zeros(int(batch * out_lanes * 8))
        b_batch = harness.buf_scalar(batch, np.uint32)
        b_msg = harness.buf_scalar(msg_bytes, np.uint32)
        b_rate = harness.buf_scalar(rate_bytes, np.uint32)
        b_out = harness.buf_scalar(out_bytes, np.uint32)
        b_domain = harness.buf_scalar(domain, np.uint32)

        pso = pipelines["keccak_f1600_batch"]
        max_tg = int(pso.maxTotalThreadsPerThreadgroup())
        tew = int(pso.threadExecutionWidth())
        tg_w = min(max_tg, max(tew, 64))
        grid_w = ((batch + tg_w - 1) // tg_w) * tg_w

        view_B = harness.np_view(bB, np.uint64, batch * out_lanes)

        def reset():
            view_B[:] = 0

        def dispatch(enc):
            enc.setComputePipelineState_(pso)
            enc.setBuffer_offset_atIndex_(bA, 0, 0)
            enc.setBuffer_offset_atIndex_(bB, 0, 1)
            enc.setBuffer_offset_atIndex_(b_batch, 0, 2)
            enc.setBuffer_offset_atIndex_(b_msg, 0, 3)
            enc.setBuffer_offset_atIndex_(b_rate, 0, 4)
            enc.setBuffer_offset_atIndex_(b_out, 0, 5)
            enc.setBuffer_offset_atIndex_(b_domain, 0, 6)
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
        got = view_B.copy().reshape(batch, out_lanes)

        mismatches = int(np.sum(got != y_ref))
        correct = (mismatches == 0)

        # Permutations per instance = ceil(out_bytes / rate_bytes).
        n_perms = (out_bytes + rate_bytes - 1) // rate_bytes
        bitops_per_instance = n_perms * _BITOPS_PER_PERMUTATION
        bitops_total = bitops_per_instance * batch
        achieved_bitop = gops_per_s(bitops_total, gpu_s)
        ceiling_bitop = float(chip.peak_int64_bitop_gops)

        bytes_total = float(batch) * float(msg_bytes + out_bytes)
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
                "batch": batch,
                "msg_bytes": msg_bytes,
                "rate_bytes": rate_bytes,
                "out_bytes": out_bytes,
                "domain": domain,
                "n_perms_per_instance": n_perms,
                "bitops_per_instance": bitops_per_instance,
                "bitops_total": bitops_total,
                "bytes_total": bytes_total,
                "achieved_bitop_gops": achieved_bitop,
                "achieved_bw_gb_s": achieved_bw,
                "anchor": anchor,
            },
        )
