**Optimization rationale:** The kernel signature forces one butterfly per thread, but the bottleneck is global memory traffic — each stage does a full read/write of N ulongs. The incumbent already hits 92% on N=2^18 (cache-friendly) but only 16-28% on smaller sizes, suggesting per-element compute overhead and divergent branches dominate. Improvements:

1. **Branchless `gold_sub`/`gold_add`** using `select` to avoid divergent control flow.
2. **Branchless `gold_reduce128`** — replace conditional subtracts with `select`/arithmetic on overflow masks.
3. **Skip the `s==0` branch entirely** by always doing the multiply path; but since `r==0` always at s=0 means `tw[0]==1`, we can just unconditionally multiply (twiddles[0]=1 by host convention). This removes a stage-divergent branch from every thread.
4. **Pre-fetch twiddle and inputs in parallel** so the compiler can issue loads concurrently.
5. **Use `select` and `as_type<uint2>`** for cheaper hi/lo splits.

These reduce instruction count per butterfly and eliminate warp divergence, which especially helps small-N where launch and compute overhead dominate.

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    ulong c = select((ulong)0, EPSILON, t < a);
    t += c;
    ulong d = select((ulong)0, P_GOLD, t >= P_GOLD);
    return t - d;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    ulong c = select((ulong)0, EPSILON, t > a);
    return t - c;
}

// 128-bit reduction mod Goldilocks p = 2^64 - 2^32 + 1.
// x = x_lo + 2^64 * x_hi, with x_hi = x_hi_hi*2^32 + x_hi_lo
// x mod p = x_lo + x_hi_lo * (2^32 - 1) - x_hi_hi  (mod p)
inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    uint2 hi_parts = as_type<uint2>(x_hi);
    ulong x_hi_lo  = (ulong)hi_parts.x;
    ulong x_hi_hi  = (ulong)hi_parts.y;

    // t0 = x_lo - x_hi_hi  (mod p); subtract EPSILON if it wrapped
    ulong t0 = x_lo - x_hi_hi;
    t0 -= select((ulong)0, EPSILON, t0 > x_lo);

    // t1 = x_hi_lo * (2^32 - 1) ; both operands < 2^32, fits in u64
    ulong t1 = (x_hi_lo << 32) - x_hi_lo;

    // t2 = t0 + t1  (mod p)
    ulong t2 = t0 + t1;
    t2 += select((ulong)0, EPSILON, t2 < t0);

    // canonicalize
    t2 -= select((ulong)0, P_GOLD, t2 >= P_GOLD);
    return t2;
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong lo = a * b;
    ulong hi = mulhi(a, b);
    return gold_reduce128(lo, hi);
}

kernel void goldilocks_ntt_stage(
    device const ulong *in_data    [[buffer(0)]],
    device       ulong *out_data   [[buffer(1)]],
    device const ulong *twiddles   [[buffer(2)]],
    constant uint      &stage_idx  [[buffer(3)]],
    constant uint      &log_N      [[buffer(4)]],
    uint k [[thread_position_in_grid]])
{
    uint lg     = log_N;
    uint half_N = 1u << (lg - 1u);
    if (k >= half_N) return;

    uint s         = stage_idx;
    uint span_half = 1u << s;
    uint mask      = span_half - 1u;

    uint r  = k & mask;
    uint j  = k >> s;

    uint i0 = (j << s) + r;
    uint i1 = i0 + half_N;
    uint o0 = (j << (s + 1u)) + r;
    uint o1 = o0 + span_half;

    uint tw_stride = 1u << (lg - 1u - s);
    uint tw_idx    = r * tw_stride;

    // Issue all three loads as early as possible so the compiler can
    // overlap their latency.
    ulong u = in_data[i0];
    ulong x = in_data[i1];
    ulong w = twiddles[tw_idx];   // twiddles[0] == 1 covers stage 0

    ulong v = gold_mul(x, w);

    out_data[o0] = gold_add(u, v);
    out_data[o1] = gold_sub(u, v);
}
```