"""Per-chip microbenchmarks for the roofline anchors PLAN.md methodology
section 1 + Open Questions section 6 call for.

Three kernels (defined inline as MSL strings):

- ``u64_xor_throughput``: tight loop of 16 independent XOR accumulators
  per thread; reports raw u64-bitop throughput in Gops/s. This is the
  natural ceiling for the theta / chi / iota path of Keccak-f[1600]
  (XOR + AND + NOT, no rotation).
- ``u64_rotate_throughput``: same loop shape, body is
  ``(x << k) | (x >> (64 - k))`` for compile-time-distinct k values.
  Reports the u64-rotate throughput in Gops/s. Each rotation expands
  to 2 shifts + 1 OR on Apple Silicon (no hardware u64 funnel-shift),
  so we expect roughly ``rotate_gops <= xor_gops / 3``.
- ``u64_mul_throughput``: same loop shape, body is a full
  ``uint64 x uint64 -> u128 lo/hi`` multiplication via the 4-way
  ``32 x 32 -> 64`` split that every Z1/Z2/Z3 seed already uses for
  ``gold_mul`` / ``mont_mul``. Reports the achievable u64-mul rate in
  Gops/s. This is the fitness anchor for compute-bound modular-
  arithmetic kernels (Goldilocks NTT, Poseidon2 S-box, Montgomery
  mul). Each composite mul expands to ~4 u32 muls + ~7 adds/shifts on
  Apple Silicon, so we expect ``mul_gops`` to be a small fraction of
  ``xor_gops``.

Anti-fold guard
---------------
A naive chain ``x = rotl(x, k)`` repeated ``N`` times is, in principle,
equal to ``rotl(x, k * N mod 64)`` -- a value the LLVM optimiser could
fold even when ``N`` is loop-variable. We avoid this by arranging the
16 accumulators into a *cyclic ring*: every iteration each ``x_i``
becomes ``f(x_i) ^ x_{(i+1) mod 16}``. The within-iter ring closes
on the just-updated ``x_0``, so the transformation over ``N`` iters
is some linear map on the 16-dim u64 state space that the compiler
cannot pre-evaluate for runtime ``N``. The linearity check (``time(2N)
/ time(N)`` must be ~2.0) confirms the guard held.

Each kernel does ``K`` ops per accumulator per iteration:
``u64_xor_throughput`` does 1 XOR; ``u64_rotate_throughput`` does
1 rotation + 1 XOR (the latter is the ring-closure XOR);
``u64_mul_throughput`` does 1 full ``umul128`` + 1 XOR (used to mix
the high and low halves of the 128-bit product back into the
accumulator -- this forces the compiler to actually compute both
halves rather than dead-coding the unused one). We *count only the
named op* when computing Gops/s.

Cache
-----
Successful runs are cached under
``~/.cache/metal-zk/microbench/<chip>_<macos>.json``. ``hardware.py``
loads the cached values when present and overrides the default
estimates of ``peak_int64_bitop_gops``, ``peak_int64_rotate_gops``,
and ``peak_int64_mul_gops`` on ``ChipSpec``.
"""

from __future__ import annotations

import json
import os
import platform
import re
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import Metal

from .harness import MetalHarness
from .hardware import ChipSpec, detect_chip


# Number of independent u64 accumulators per thread. With 16 in-flight
# chains the compiler can hide the ~4-cycle ALU latency.
ACCUMULATORS_PER_THREAD: int = 16

# 16 distinct rotation amounts in [1, 63], used both at compile time
# (in the rotate kernel body) and at host time (for the ops/iter count).
ROTATION_AMOUNTS: tuple[int, ...] = (
    1, 7, 13, 17, 23, 29, 31, 37,
    41, 43, 47, 53, 59, 3, 11, 19,
)


_MICROBENCH_SOURCE: str = r"""
#include <metal_stdlib>
using namespace metal;

// ---------------- u64 XOR throughput ----------------
// 16 accumulators in a cyclic ring; each iter does 16 XORs. The
// within-iter ring closure (x_15 ^= updated_x_0) plus the cross-iter
// dependency prevents any compile-time loop-fold.
kernel void u64_xor_throughput(
    device const ulong *seed_data  [[buffer(0)]],
    device       ulong *out_data   [[buffer(1)]],
    constant uint      &n_iters    [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    ulong s = seed_data[idx];
    ulong x0  = s ^ 0x01ul, x1  = s ^ 0x02ul, x2  = s ^ 0x03ul,
          x3  = s ^ 0x04ul, x4  = s ^ 0x05ul, x5  = s ^ 0x06ul,
          x6  = s ^ 0x07ul, x7  = s ^ 0x08ul, x8  = s ^ 0x09ul,
          x9  = s ^ 0x0Aul, x10 = s ^ 0x0Bul, x11 = s ^ 0x0Cul,
          x12 = s ^ 0x0Dul, x13 = s ^ 0x0Eul, x14 = s ^ 0x0Ful,
          x15 = s ^ 0x10ul;

    for (uint i = 0u; i < n_iters; ++i) {
        x0  ^= x1;  x1  ^= x2;  x2  ^= x3;  x3  ^= x4;
        x4  ^= x5;  x5  ^= x6;  x6  ^= x7;  x7  ^= x8;
        x8  ^= x9;  x9  ^= x10; x10 ^= x11; x11 ^= x12;
        x12 ^= x13; x13 ^= x14; x14 ^= x15; x15 ^= x0;
    }
    out_data[idx] = x0 ^ x1 ^ x2  ^ x3  ^ x4  ^ x5  ^ x6  ^ x7
                  ^ x8 ^ x9 ^ x10 ^ x11 ^ x12 ^ x13 ^ x14 ^ x15;
}

// ---------------- u64 rotate throughput ----------------
// 16 accumulators with 16 distinct compile-time rotation amounts in
// [1, 63]; ring closure as in the XOR kernel. Per iter: 16 rotates +
// 16 XORs. We count rotations only -- the XOR is the anti-fold ring
// closure, and the resulting Gops/s is the rotation rate achievable
// in code that also does one XOR per rotation, which matches the
// Keccak ``x ^ rotl(y, k)`` shape.
#define ROT(x, k) (((x) << (k)) | ((x) >> (64 - (k))))

kernel void u64_rotate_throughput(
    device const ulong *seed_data  [[buffer(0)]],
    device       ulong *out_data   [[buffer(1)]],
    constant uint      &n_iters    [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    ulong s = seed_data[idx];
    ulong x0  = s ^ 0x01ul, x1  = s ^ 0x02ul, x2  = s ^ 0x03ul,
          x3  = s ^ 0x04ul, x4  = s ^ 0x05ul, x5  = s ^ 0x06ul,
          x6  = s ^ 0x07ul, x7  = s ^ 0x08ul, x8  = s ^ 0x09ul,
          x9  = s ^ 0x0Aul, x10 = s ^ 0x0Bul, x11 = s ^ 0x0Cul,
          x12 = s ^ 0x0Dul, x13 = s ^ 0x0Eul, x14 = s ^ 0x0Ful,
          x15 = s ^ 0x10ul;

    for (uint i = 0u; i < n_iters; ++i) {
        x0  = ROT(x0,   1) ^ x1;   x1  = ROT(x1,   7) ^ x2;
        x2  = ROT(x2,  13) ^ x3;   x3  = ROT(x3,  17) ^ x4;
        x4  = ROT(x4,  23) ^ x5;   x5  = ROT(x5,  29) ^ x6;
        x6  = ROT(x6,  31) ^ x7;   x7  = ROT(x7,  37) ^ x8;
        x8  = ROT(x8,  41) ^ x9;   x9  = ROT(x9,  43) ^ x10;
        x10 = ROT(x10, 47) ^ x11;  x11 = ROT(x11, 53) ^ x12;
        x12 = ROT(x12, 59) ^ x13;  x13 = ROT(x13,  3) ^ x14;
        x14 = ROT(x14, 11) ^ x15;  x15 = ROT(x15, 19) ^ x0;
    }
    out_data[idx] = x0 ^ x1 ^ x2  ^ x3  ^ x4  ^ x5  ^ x6  ^ x7
                  ^ x8 ^ x9 ^ x10 ^ x11 ^ x12 ^ x13 ^ x14 ^ x15;
}

// ---------------- u64 full multiplication throughput ----------------
// Full 64x64 -> 128 multiplication via the 4-way 32x32 -> 64 split
// that every Z1/Z2/Z3 seed uses (gold_mul, mont_mul, ...). Per iter:
// 16 umul128 ops in a cyclic ring. The lo ^ hi reduction at each
// step forces both halves of the 128-bit product to participate,
// preventing the compiler from dead-coding either half.
inline ulong2 _umul128(ulong a, ulong b) {
    uint a0 = (uint)a, a1 = (uint)(a >> 32);
    uint b0 = (uint)b, b1 = (uint)(b >> 32);
    ulong p00 = (ulong)a0 * (ulong)b0;
    ulong p01 = (ulong)a0 * (ulong)b1;
    ulong p10 = (ulong)a1 * (ulong)b0;
    ulong p11 = (ulong)a1 * (ulong)b1;
    ulong mid = (p00 >> 32) + (p01 & 0xFFFFFFFFul) + (p10 & 0xFFFFFFFFul);
    ulong lo  = (p00 & 0xFFFFFFFFul) | (mid << 32);
    ulong hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return ulong2(lo, hi);
}

kernel void u64_mul_throughput(
    device const ulong *seed_data  [[buffer(0)]],
    device       ulong *out_data   [[buffer(1)]],
    constant uint      &n_iters    [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    ulong s = seed_data[idx];
    ulong x0  = s ^ 0x01ul, x1  = s ^ 0x02ul, x2  = s ^ 0x03ul,
          x3  = s ^ 0x04ul, x4  = s ^ 0x05ul, x5  = s ^ 0x06ul,
          x6  = s ^ 0x07ul, x7  = s ^ 0x08ul, x8  = s ^ 0x09ul,
          x9  = s ^ 0x0Aul, x10 = s ^ 0x0Bul, x11 = s ^ 0x0Cul,
          x12 = s ^ 0x0Dul, x13 = s ^ 0x0Eul, x14 = s ^ 0x0Ful,
          x15 = s ^ 0x10ul;

    for (uint i = 0u; i < n_iters; ++i) {
        ulong2 p;
        p = _umul128(x0,  x1);   x0  = p.x ^ p.y;
        p = _umul128(x1,  x2);   x1  = p.x ^ p.y;
        p = _umul128(x2,  x3);   x2  = p.x ^ p.y;
        p = _umul128(x3,  x4);   x3  = p.x ^ p.y;
        p = _umul128(x4,  x5);   x4  = p.x ^ p.y;
        p = _umul128(x5,  x6);   x5  = p.x ^ p.y;
        p = _umul128(x6,  x7);   x6  = p.x ^ p.y;
        p = _umul128(x7,  x8);   x7  = p.x ^ p.y;
        p = _umul128(x8,  x9);   x8  = p.x ^ p.y;
        p = _umul128(x9,  x10);  x9  = p.x ^ p.y;
        p = _umul128(x10, x11);  x10 = p.x ^ p.y;
        p = _umul128(x11, x12);  x11 = p.x ^ p.y;
        p = _umul128(x12, x13);  x12 = p.x ^ p.y;
        p = _umul128(x13, x14);  x13 = p.x ^ p.y;
        p = _umul128(x14, x15);  x14 = p.x ^ p.y;
        p = _umul128(x15, x0);   x15 = p.x ^ p.y;
    }
    out_data[idx] = x0 ^ x1 ^ x2  ^ x3  ^ x4  ^ x5  ^ x6  ^ x7
                  ^ x8 ^ x9 ^ x10 ^ x11 ^ x12 ^ x13 ^ x14 ^ x15;
}
"""


@dataclass
class MicrobenchResult:
    chip_name: str
    device_name: str
    macos_version: str
    n_threads: int
    n_iters: int
    n_warmup: int
    n_measure: int
    accumulators_per_thread: int
    u64_xor_gops: float
    u64_rotate_gops: float
    u64_mul_gops: float
    u64_xor_median_s: float
    u64_rotate_median_s: float
    u64_mul_median_s: float
    u64_xor_iqr_s: float
    u64_rotate_iqr_s: float
    u64_mul_iqr_s: float
    linearity_ratio: float          # time(2N) / time(N); expect ~2.0
    methodology: str

    def as_dict(self) -> dict:
        return {
            "chip_name": self.chip_name,
            "device_name": self.device_name,
            "macos_version": self.macos_version,
            "n_threads": self.n_threads,
            "n_iters": self.n_iters,
            "n_warmup": self.n_warmup,
            "n_measure": self.n_measure,
            "accumulators_per_thread": self.accumulators_per_thread,
            "u64_xor_gops": self.u64_xor_gops,
            "u64_rotate_gops": self.u64_rotate_gops,
            "u64_mul_gops": self.u64_mul_gops,
            "u64_xor_median_s": self.u64_xor_median_s,
            "u64_rotate_median_s": self.u64_rotate_median_s,
            "u64_mul_median_s": self.u64_mul_median_s,
            "u64_xor_iqr_s": self.u64_xor_iqr_s,
            "u64_rotate_iqr_s": self.u64_rotate_iqr_s,
            "u64_mul_iqr_s": self.u64_mul_iqr_s,
            "linearity_ratio": self.linearity_ratio,
            "methodology": self.methodology,
        }


# ---------------- Cache plumbing ----------------

def _macos_version() -> str:
    """Stable per-chip-family cache key suffix; macOS proxies for the
    Metal driver version since the driver ships with the OS."""
    try:
        v = platform.mac_ver()[0]
    except Exception:
        v = ""
    if v:
        return v
    try:
        out = subprocess.check_output(
            ["uname", "-r"], text=True,
        ).strip()
        return f"darwin-{out}"
    except Exception:
        return "unknown"


def _safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def cache_dir() -> Path:
    root = Path(
        os.environ.get("METAL_ZK_CACHE", "~/.cache/metal-zk")
    ).expanduser()
    d = root / "microbench"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_path(chip_name: str, macos: str | None = None) -> Path:
    macos = macos if macos is not None else _macos_version()
    fname = f"{_safe_filename(chip_name)}__{_safe_filename(macos)}.json"
    return cache_dir() / fname


def load_cached(chip_name: str) -> dict | None:
    p = cache_path(chip_name)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def save_cached(result: MicrobenchResult) -> Path:
    p = cache_path(result.chip_name, result.macos_version)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(result.as_dict(), indent=2))
    os.replace(tmp, p)
    return p


# ---------------- Kernel runners ----------------

def _dispatch_kernel(
    harness: MetalHarness,
    pso,
    bA, bB, b_iters,
    n_threads: int,
) -> callable:
    max_tg = int(pso.maxTotalThreadsPerThreadgroup())
    tew = int(pso.threadExecutionWidth())
    tg_w = min(max_tg, max(tew, 256))
    grid_w = ((n_threads + tg_w - 1) // tg_w) * tg_w

    def dispatch(enc):
        enc.setComputePipelineState_(pso)
        enc.setBuffer_offset_atIndex_(bA, 0, 0)
        enc.setBuffer_offset_atIndex_(bB, 0, 1)
        enc.setBuffer_offset_atIndex_(b_iters, 0, 2)
        enc.dispatchThreads_threadsPerThreadgroup_(
            Metal.MTLSizeMake(grid_w, 1, 1),
            Metal.MTLSizeMake(tg_w, 1, 1),
        )
    return dispatch


def _time_kernel(
    harness: MetalHarness, pso, bA, bB, b_iters,
    n_threads: int, n_warmup: int, n_measure: int,
) -> tuple[float, float, list[float]]:
    dispatch = _dispatch_kernel(harness, pso, bA, bB, b_iters, n_threads)
    for _ in range(n_warmup):
        harness.time_dispatch(dispatch)
    samples = [harness.time_dispatch(dispatch) for _ in range(n_measure)]
    arr = np.array(samples)
    q1, q3 = np.percentile(arr, [25, 75])
    return float(np.median(arr)), float(q3 - q1), samples


def run_microbench(
    harness: MetalHarness | None = None,
    chip: ChipSpec | None = None,
    n_threads: int = 1 << 16,
    n_iters: int = 1 << 12,
    n_warmup: int = 3,
    n_measure: int = 10,
    linearity_check: bool = True,
) -> MicrobenchResult:
    """Compile the microbench kernels and measure XOR + rotate throughput.

    Returns a :class:`MicrobenchResult` with derived Gops/s figures and
    a linearity ratio (time at 2N / time at N; must be ~2.0 for the
    anti-fold guard to be working).
    """
    harness = harness or MetalHarness()
    chip = chip or detect_chip()

    cr = harness.compile(_MICROBENCH_SOURCE)
    if cr.error is not None:
        raise RuntimeError(f"microbench compile error: {cr.error}")
    pipelines, perr = harness.make_pipelines(
        cr.library,
        ["u64_xor_throughput", "u64_rotate_throughput", "u64_mul_throughput"],
    )
    if perr is not None:
        raise RuntimeError(f"microbench pipeline error: {perr}")

    seed = np.arange(n_threads, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)
    bA = harness.buf_from_np(seed)
    bB = harness.buf_zeros(int(seed.nbytes))
    b_iters = harness.buf_scalar(n_iters, np.uint32)
    b_iters_2x = harness.buf_scalar(n_iters * 2, np.uint32)

    pso_xor = pipelines["u64_xor_throughput"]
    pso_rot = pipelines["u64_rotate_throughput"]
    pso_mul = pipelines["u64_mul_throughput"]

    xor_med, xor_iqr, _ = _time_kernel(
        harness, pso_xor, bA, bB, b_iters, n_threads, n_warmup, n_measure,
    )
    rot_med, rot_iqr, _ = _time_kernel(
        harness, pso_rot, bA, bB, b_iters, n_threads, n_warmup, n_measure,
    )
    mul_med, mul_iqr, _ = _time_kernel(
        harness, pso_mul, bA, bB, b_iters, n_threads, n_warmup, n_measure,
    )

    # Anti-fold linearity check: doubling n_iters should ~double time
    # for the slowest kernel (the multiplication path; biggest signal-
    # to-noise margin).
    if linearity_check:
        mul_med_2x, _, _ = _time_kernel(
            harness, pso_mul, bA, bB, b_iters_2x, n_threads,
            max(1, n_warmup // 2), max(2, n_measure // 2),
        )
        linearity = mul_med_2x / mul_med if mul_med > 0 else float("nan")
    else:
        linearity = float("nan")

    ops_total = float(n_threads) * float(n_iters) * float(ACCUMULATORS_PER_THREAD)
    u64_xor_gops = ops_total / xor_med / 1e9
    u64_rotate_gops = ops_total / rot_med / 1e9
    u64_mul_gops = ops_total / mul_med / 1e9

    methodology = (
        f"{ACCUMULATORS_PER_THREAD} u64 accumulators in a cyclic ring "
        f"per thread, {n_iters} inner-loop iterations, {n_threads} "
        "threads. Each iter applies one named op (XOR or rotation) per "
        "accumulator and XORs in the next-accumulator value to close "
        "the ring -- this prevents the LLVM optimiser from collapsing "
        "the chain into a closed-form linear map. Linearity check: "
        "doubling n_iters must ~double the measured GPU time "
        "(expected ratio ~2.0)."
    )

    return MicrobenchResult(
        chip_name=chip.name,
        device_name=harness.device_name(),
        macos_version=_macos_version(),
        n_threads=n_threads,
        n_iters=n_iters,
        n_warmup=n_warmup,
        n_measure=n_measure,
        accumulators_per_thread=ACCUMULATORS_PER_THREAD,
        u64_xor_gops=u64_xor_gops,
        u64_rotate_gops=u64_rotate_gops,
        u64_mul_gops=u64_mul_gops,
        u64_xor_median_s=xor_med,
        u64_rotate_median_s=rot_med,
        u64_mul_median_s=mul_med,
        u64_xor_iqr_s=xor_iqr,
        u64_rotate_iqr_s=rot_iqr,
        u64_mul_iqr_s=mul_iqr,
        linearity_ratio=linearity,
        methodology=methodology,
    )


__all__ = [
    "ACCUMULATORS_PER_THREAD",
    "ROTATION_AMOUNTS",
    "MicrobenchResult",
    "cache_dir",
    "cache_path",
    "load_cached",
    "save_cached",
    "run_microbench",
]
