"""Apple Silicon chip detection + roofline lookup for Metal-ZK.

Per-chip ceilings:

- ``peak_fp32_gflops``: same FP32 throughput as Metal-Sci. Kept for
  cross-task convenience (FRI round / Merkle reductions sometimes report
  achieved-vs-FP32 for sanity; not the primary fitness anchor).
- ``peak_bw_gb_s``: DRAM streaming bandwidth. The fitness anchor for any
  ZK kernel that is memory-bound (e.g. NTT at large N).
- ``peak_int64_mul_gops``: sustained ``uint64 x uint64 -> u128 lo/hi``
  throughput. The fitness anchor for compute-bound modular-arithmetic
  kernels (NTT at small N, Poseidon2 S-box, Montgomery mul).
- ``peak_int64_bitop_gops``: sustained ``uint64`` XOR/AND throughput.
  The fitness anchor for bit-hash kernels (Keccak-f[1600] theta + chi
  + iota path).
- ``peak_int64_rotate_gops``: sustained ``uint64`` left-rotation
  throughput. The binding ceiling for the rho stage of Keccak-f[1600]
  (and for any kernel whose ALU work is rotate-dominated).

The default values are *estimates* derived from
``peak_fp32_gflops`` (see ``_int64_mul_estimate`` /
``_int64_bitop_estimate`` / ``_int64_rotate_estimate``). They are
overridden, when present, by a cached microbenchmark under
``~/.cache/metal-zk/microbench/`` keyed by chip name + macOS version;
see :mod:`metal_zk.microbench` for the kernels and methodology.

A kernel that reports ``fraction_of_ceiling = achieved / ceiling`` is
honest about which ceiling it picked: each task names its anchor
explicitly in ``SizeResult.achieved_unit`` / ``ceiling_unit``.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ChipSpec:
    name: str
    peak_fp32_gflops: float       # FP32 GFLOPS (FMA-counted)
    peak_bw_gb_s: float           # DRAM bandwidth, GB/s
    peak_int64_mul_gops: float    # Estimated int64 mul throughput, Gops/s
    peak_int64_bitop_gops: float  # u64 XOR/AND throughput, Gops/s
    peak_int64_rotate_gops: float # u64 rotate throughput, Gops/s
    # "estimate" if derived from FP32 GFLOPS heuristics; "microbench"
    # if loaded from a cached run of metal_zk.microbench.
    roofline_source: str = "estimate"


def _int64_mul_estimate(fp32_gflops: float) -> float:
    """Heuristic: ~half of FP32 mul-add rate divided by 4 (one u64 mul
    decomposes into 4 u32 muls on Apple's integer pipe). Refined by a
    chip-level microbenchmark in a follow-up; the value is documented
    in CandidateResult JSON so a calibration pass can re-emit it later.
    """
    return 0.5 * fp32_gflops / 4.0


def _int64_bitop_estimate(fp32_gflops: float) -> float:
    """Heuristic: one u64 bitop ~ two u32 bitops; the integer ALU pipe
    runs at roughly one op per ALU per cycle while FP32 GFLOPS counts
    each FMA as two ops, so u64-bitop throughput ~= fp32_gflops / 4.
    Refined by :mod:`metal_zk.microbench`.
    """
    return fp32_gflops / 4.0


def _int64_rotate_estimate(fp32_gflops: float) -> float:
    """Heuristic: each u64 rotation expands to 2 shifts + 1 OR on Apple
    Silicon (no hardware u64 funnel-shift), so its throughput is
    roughly ``bitop_estimate / 3``. Refined by
    :mod:`metal_zk.microbench`; the cached value typically lands
    *lower* than this heuristic because of register-pressure /
    instruction-scheduling effects on the wider expansion.
    """
    return _int64_bitop_estimate(fp32_gflops) / 3.0


def _spec(name: str, fp32: float, bw: float,
          int64_mul: float | None = None,
          int64_bitop: float | None = None,
          int64_rotate: float | None = None) -> ChipSpec:
    return ChipSpec(
        name=name,
        peak_fp32_gflops=fp32,
        peak_bw_gb_s=bw,
        peak_int64_mul_gops=int64_mul if int64_mul is not None
                            else _int64_mul_estimate(fp32),
        peak_int64_bitop_gops=int64_bitop if int64_bitop is not None
                              else _int64_bitop_estimate(fp32),
        peak_int64_rotate_gops=int64_rotate if int64_rotate is not None
                              else _int64_rotate_estimate(fp32),
    )


# Conservative defaults per chip family.
_CHIP_TABLE: dict[str, ChipSpec] = {
    "Apple M1":         _spec("Apple M1",         2_600,  68),
    "Apple M1 Pro":     _spec("Apple M1 Pro",     4_500, 200),
    "Apple M1 Max":     _spec("Apple M1 Max",     7_800, 400),
    "Apple M1 Ultra":   _spec("Apple M1 Ultra",  21_000, 800),
    "Apple M2":         _spec("Apple M2",         3_600, 100),
    "Apple M2 Pro":     _spec("Apple M2 Pro",     6_800, 200),
    "Apple M2 Max":     _spec("Apple M2 Max",    13_600, 400),
    "Apple M2 Ultra":   _spec("Apple M2 Ultra",  27_200, 800),
    "Apple M3":         _spec("Apple M3",         4_100, 100),
    "Apple M3 Pro":     _spec("Apple M3 Pro",     7_400, 150),
    "Apple M3 Max":     _spec("Apple M3 Max",    14_200, 300),
    "Apple M4":         _spec("Apple M4",         4_600, 120),
    "Apple M4 Pro":     _spec("Apple M4 Pro",     9_200, 273),
    "Apple M4 Max":     _spec("Apple M4 Max",    18_000, 546),
}


def _read_chip_name() -> str:
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True,
        ).strip()
    except Exception:
        out = ""
    if out:
        return out
    try:
        sp = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType"], text=True,
        )
        m = re.search(r"Chip:\s*(.+)", sp)
        if m:
            return m.group(1).strip()
    except Exception:
        pass
    return "Unknown"


def _apply_microbench(spec: ChipSpec) -> ChipSpec:
    """If the chip has a cached microbench run, override the bitop,
    rotate, and mul ceilings with the measured Gops/s and tag the
    source. Missing fields fall back to whatever default the spec
    already had (so a legacy two-kernel cache still works)."""
    try:
        from .microbench import load_cached
    except ImportError:
        return spec
    cached = load_cached(spec.name)
    if not cached:
        return spec
    xor = cached.get("u64_xor_gops")
    rot = cached.get("u64_rotate_gops")
    mul = cached.get("u64_mul_gops")
    if not xor or not rot:
        return spec
    return replace(
        spec,
        peak_int64_bitop_gops=float(xor),
        peak_int64_rotate_gops=float(rot),
        peak_int64_mul_gops=float(mul) if mul else spec.peak_int64_mul_gops,
        roofline_source="microbench",
    )


def detect_chip() -> ChipSpec:
    name = _read_chip_name()
    if name in _CHIP_TABLE:
        return _apply_microbench(_CHIP_TABLE[name])
    for key, spec in _CHIP_TABLE.items():
        if key in name:
            return _apply_microbench(spec)
    return _apply_microbench(_spec(name or "Unknown", 2_000, 80))
