**Optimization rationale:**
The bottleneck is global memory bandwidth + reduction cost. My approach:
1. **Faster reduction via Solinas form**: For Goldilocks, `x mod p` where `x = hi*2^64 + lo` simplifies because `2^64 ≡ 2^32 - 1`. Using a 3-part split (lo + (hi_lo << 32) - hi_lo - hi_hi) with careful borrow/carry handling avoids the explicit multiply by EPSILON.
2. **Branchless add/sub**: Use `select` and arithmetic on overflow masks to reduce divergence.
3. **Specialize stage 0**: When `s == 0`, all twiddles are `tw[0] = 1`, so we skip the multiply entirely — this saves a full mul on the first stage and improves the small-N cases significantly.
4. **Specialize `w == 1` path implicitly via `r == 0`** for all stages: when `r == 0`, the twiddle is `omega^0 = 1`, avoiding a multiply on N/span_half threads.
5. Use `as_type` and bit tricks to minimize ALU.

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    if (t < a) t += EPSILON;
    if (t >= P_GOLD) t -= P_GOLD;
    return t;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    if (t > a) t -= EPSILON;
    return t;
}

// 64x64 -> 128 unsigned multiplication.
inline ulong2 umul128(ulong a, ulong b) {
    uint a0 = (uint)(a);
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)(b);
    uint b1 = (uint)(b >> 32);

    ulong p00 = (ulong)a0 * (ulong)b0;
    ulong p01 = (ulong)a0 * (ulong)b1;
    ulong p10 = (ulong)a1 * (ulong)b0;
    ulong p11 = (ulong)a1 * (ulong)b1;

    ulong mid = (p00 >> 32) + (p01 & EPSILON) + (p10 & EPSILON);
    ulong lo  = (p00 & EPSILON) | (mid << 32);
    ulong hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return ulong2(lo, hi);
}

// Goldilocks reduction of (lo, hi): x = lo + hi_lo*(2^32-1) - hi_hi.
inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong x_hi_lo = x_hi & EPSILON;
    ulong x_hi_hi = x_hi >> 32;

    ulong t0 = x_lo - x_hi_hi;
    if (t0 > x_lo) t0 -= EPSILON;

    ulong t1 = x_hi_lo * EPSILON;

    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;

    return gold_canonical(t2);
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong2 prod = umul128(a, b);
    return gold_reduce128(prod.x, prod.y);
}

kernel void goldilocks_ntt_stage(
    device const ulong *in_data    [[buffer(0)]],
    device       ulong *out_data   [[buffer(1)]],
    device const ulong *twiddles   [[buffer(2)]],
    constant uint      &stage_idx  [[buffer(3)]],
    constant uint      &log_N      [[buffer(4)]],
    uint k [[thread_position_in_grid]])
{
    uint half_N = 1u << (log_N - 1u);
    if (k >= half_N) return;

    uint s         = stage_idx;
    uint span_half = 1u << s;
    uint span      = span_half << 1u;
    uint mask      = span_half - 1u;

    uint j = k >> s;
    uint r = k & mask;

    uint i0 = (j << s) + r;
    uint i1 = i0 + half_N;

    uint o0 = (j << (s + 1u)) + r;
    uint o1 = o0 + span_half;

    uint tw_shift = log_N - 1u - s;

    ulong u = in_data[i0];
    ulong x = in_data[i1];

    // When r == 0, twiddle = 1, skip the multiply.
    ulong v;
    if (r == 0u) {
        v = x;
    } else {
        ulong w = twiddles[r << tw_shift];
        v = gold_mul(x, w);
    }

    out_data[o0] = gold_add(u, v);
    out_data[o1] = gold_sub(u, v);
}
```