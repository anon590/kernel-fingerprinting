#!/usr/bin/env python3
"""Run the per-chip Metal-ZK rotation / XOR microbench and cache the
result for ``hardware.detect_chip()`` to pick up.

Examples:
    # Default: 65536 threads x 4096 inner iters x 16 accumulators.
    python run_microbench.py

    # Tighter timing run (slower, more stable).
    python run_microbench.py --measure 30

    # Re-run even if the cache already has an entry.
    python run_microbench.py --force
"""

from __future__ import annotations

import argparse
import sys

from metal_zk.harness import MetalHarness
from metal_zk.hardware import (
    _int64_bitop_estimate, _int64_mul_estimate, _int64_rotate_estimate,
    detect_chip,
)
from metal_zk.microbench import (
    cache_path, load_cached, run_microbench, save_cached,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--threads", type=int, default=1 << 16,
        help="Number of GPU threads launched (default: 2^16 = 65536).",
    )
    parser.add_argument(
        "--iters", type=int, default=1 << 12,
        help="Inner-loop iterations per thread (default: 2^12 = 4096).",
    )
    parser.add_argument(
        "--warmup", type=int, default=3,
        help="Untimed warmup dispatches before each measurement.",
    )
    parser.add_argument(
        "--measure", type=int, default=10,
        help="Timed dispatches per kernel; median + IQR are reported.",
    )
    parser.add_argument(
        "--no-linearity-check", action="store_true",
        help="Skip the 2N-iter linearity check (saves ~half the wall time).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if a cached result is already present.",
    )
    args = parser.parse_args()

    chip = detect_chip()
    cached = load_cached(chip.name)
    if cached and not args.force:
        print(f"[microbench] cached result already present for "
              f"{chip.name!r}:")
        print(f"   {cache_path(chip.name)}")
        print(f"   u64_xor_gops    = {cached.get('u64_xor_gops'):.1f}")
        print(f"   u64_rotate_gops = {cached.get('u64_rotate_gops'):.1f}")
        mul = cached.get("u64_mul_gops")
        if mul is not None:
            print(f"   u64_mul_gops    = {float(mul):.1f}")
        print("   pass --force to re-run.")
        return 0

    harness = MetalHarness()
    print(f"=== Metal-ZK microbench ===")
    print(f"  chip:    {chip.name}")
    print(f"  device:  {harness.device_name()}")
    print(f"  threads: {args.threads}")
    print(f"  iters:   {args.iters}")
    print(f"  warmup:  {args.warmup}   measure: {args.measure}")
    print()

    result = run_microbench(
        harness=harness, chip=chip,
        n_threads=args.threads, n_iters=args.iters,
        n_warmup=args.warmup, n_measure=args.measure,
        linearity_check=not args.no_linearity_check,
    )

    # Stability summary.
    xor_iqr_pct = (result.u64_xor_iqr_s / result.u64_xor_median_s) * 100.0
    rot_iqr_pct = (result.u64_rotate_iqr_s / result.u64_rotate_median_s) * 100.0
    mul_iqr_pct = (result.u64_mul_iqr_s / result.u64_mul_median_s) * 100.0
    print(f"[u64_xor]    {result.u64_xor_median_s*1e3:7.3f} ms median, "
          f"IQR {xor_iqr_pct:.2f}%")
    print(f"             -> {result.u64_xor_gops:7.1f} Gops/s")
    print(f"[u64_rotate] {result.u64_rotate_median_s*1e3:7.3f} ms median, "
          f"IQR {rot_iqr_pct:.2f}%")
    print(f"             -> {result.u64_rotate_gops:7.1f} Gops/s")
    print(f"             rotate / xor = "
          f"{result.u64_rotate_gops / result.u64_xor_gops:.3f}")
    print(f"[u64_mul]    {result.u64_mul_median_s*1e3:7.3f} ms median, "
          f"IQR {mul_iqr_pct:.2f}%")
    print(f"             -> {result.u64_mul_gops:7.1f} Gops/s")
    print(f"             mul / xor    = "
          f"{result.u64_mul_gops / result.u64_xor_gops:.3f}")

    # Anti-fold guard (timed against the mul kernel, which has the
    # biggest signal-to-noise margin).
    if not args.no_linearity_check:
        lin = result.linearity_ratio
        verdict = "OK" if 1.85 <= lin <= 2.15 else "WARN (possible compiler fold)"
        print(f"\n  linearity (time(2N) / time(N)): {lin:.3f}   [{verdict}]")

    # Compare against the previous estimates.
    xor_est = _int64_bitop_estimate(chip.peak_fp32_gflops)
    rot_est = _int64_rotate_estimate(chip.peak_fp32_gflops)
    mul_est = _int64_mul_estimate(chip.peak_fp32_gflops)
    print(f"\nVs previous heuristic on {chip.name}:")
    print(f"  est. bitop  = {xor_est:7.1f} Gops/s   |  "
          f"measured = {result.u64_xor_gops:7.1f}   "
          f"({result.u64_xor_gops / xor_est * 100:5.1f}% of est.)")
    print(f"  est. rotate = {rot_est:7.1f} Gops/s   |  "
          f"measured = {result.u64_rotate_gops:7.1f}   "
          f"({result.u64_rotate_gops / rot_est * 100:5.1f}% of est.)")
    print(f"  est. mul    = {mul_est:7.1f} Gops/s   |  "
          f"measured = {result.u64_mul_gops:7.1f}   "
          f"({result.u64_mul_gops / mul_est * 100:5.1f}% of est.)")

    # Stability warning.
    if xor_iqr_pct > 5.0 or rot_iqr_pct > 5.0 or mul_iqr_pct > 5.0:
        print("\n[warning] timing IQR > 5%; re-run with --measure 30 for "
              "a quieter reading.")
    if not args.no_linearity_check and not (1.85 <= result.linearity_ratio <= 2.15):
        print(f"\n[warning] linearity {result.linearity_ratio:.3f} "
              "suggests the compiler may have collapsed the chain; "
              "the throughput numbers are NOT trustworthy. Inspect "
              "the kernel disassembly or strengthen the anti-fold "
              "guard before relying on this result.")
        return 2

    path = save_cached(result)
    print(f"\ncached -> {path}")
    print("hardware.detect_chip() will now use the measured values.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
