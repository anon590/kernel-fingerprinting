#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD   = 0xFFFFFFFF00000001ul;  // 2^64 - 2^32 + 1
constant ulong EPSILON  = 0x00000000FFFFFFFFul;  // 2^32 - 1

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

// Goldilocks multiply with fused reduction.
// Uses 32x32 -> 64 native multiplies and the identity
//   x_hi_lo * (2^32 - 1) = (x_hi_lo << 32) - x_hi_lo
// to avoid a 64-bit multiply in the reduction.
inline ulong gold_mul(ulong a, ulong b) {
    uint a0 = (uint)(a);
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)(b);
    uint b1 = (uint)(b >> 32);

    ulong p00 = (ulong)a0 * (ulong)b0;
    ulong p01 = (ulong)a0 * (ulong)b1;
    ulong p10 = (ulong)a1 * (ulong)b0;
    ulong p11 = (ulong)a1 * (ulong)b1;

    // Combine into 128-bit product (lo, hi).
    ulong mid = (p00 >> 32) + (p01 & EPSILON) + (p10 & EPSILON);
    ulong lo  = (p00 & EPSILON) | (mid << 32);
    ulong hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);

    // Reduce: result = lo + (hi_lo) * (2^32 - 1) - hi_hi   (mod p)
    ulong hi_lo = hi & EPSILON;
    ulong hi_hi = hi >> 32;

    // t0 = lo - hi_hi  (with borrow fixup: subtract EPSILON on borrow)
    ulong t0 = lo - hi_hi;
    if (t0 > lo) t0 -= EPSILON;

    // t1 = hi_lo * (2^32 - 1) = (hi_lo << 32) - hi_lo  (fits in u64; no overflow since hi_lo < 2^32)
    ulong t1 = (hi_lo << 32) - hi_lo;

    // t2 = t0 + t1  (with carry fixup: add EPSILON on carry)
    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;

    return gold_canonical(t2);
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
    uint span_half = 1u << s;          // 2^s
    uint span      = span_half << 1u;  // 2^(s+1)
    uint mask      = span_half - 1u;

    uint j  = k >> s;                  // s may be 0; (k >> 0) == k
    uint r  = k & mask;                // mask=0 when s=0 -> r=0

    uint i0 = j * span_half + r;
    uint i1 = i0 + half_N;
    uint o0 = j * span + r;
    uint o1 = o0 + span_half;

    uint tw_stride = 1u << (log_N - 1u - s);
    ulong w = twiddles[r * tw_stride];

    ulong u = in_data[i0];
    ulong v = gold_mul(in_data[i1], w);

    out_data[o0] = gold_add(u, v);
    out_data[o1] = gold_sub(u, v);
}