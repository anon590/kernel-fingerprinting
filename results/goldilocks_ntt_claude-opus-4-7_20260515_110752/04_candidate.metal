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

// Full 64x64 -> 128 multiply followed by Goldilocks reduction.
// Inputs assumed < p (canonical). Output canonical in [0, p).
inline ulong gold_mul(ulong a, ulong b) {
    uint a0 = (uint)(a);
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)(b);
    uint b1 = (uint)(b >> 32);

    ulong p00 = (ulong)a0 * (ulong)b0;
    ulong p01 = (ulong)a0 * (ulong)b1;
    ulong p10 = (ulong)a1 * (ulong)b0;
    ulong p11 = (ulong)a1 * (ulong)b1;

    // Assemble 128-bit product.
    ulong mid = (p00 >> 32) + (p01 & EPSILON) + (p10 & EPSILON);
    ulong lo  = (p00 & EPSILON) | (mid << 32);
    ulong hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);

    // Goldilocks reduction:
    //   x = lo + hi_lo * 2^64 + hi_hi * 2^96   (split hi = hi_hi:hi_lo)
    //   2^64 ≡ 2^32 - 1   (mod p)
    //   2^96 ≡ 2^64 * 2^32 ≡ (2^32 - 1) * 2^32 ≡ 2^64 - 2^32
    //                       ≡ (2^32 - 1) - 2^32 ≡ -1   (mod p)
    // So x ≡ lo + hi_lo*(2^32 - 1) - hi_hi  (mod p).
    ulong hi_lo = hi & EPSILON;
    ulong hi_hi = hi >> 32;

    // t = lo - hi_hi (mod 2^64 - we want mod p, fix borrow)
    ulong t = lo - hi_hi;
    if (t > lo) t -= EPSILON;

    // t += hi_lo * (2^32 - 1) = (hi_lo << 32) - hi_lo
    ulong shifted = hi_lo << 32;
    ulong add_val = shifted - hi_lo;  // exact in u64; hi_lo < 2^32

    ulong r = t + add_val;
    if (r < t) r += EPSILON;

    return gold_canonical(r);
}

kernel void goldilocks_ntt_stage(
    device const ulong *in_data    [[buffer(0)]],
    device       ulong *out_data   [[buffer(1)]],
    device const ulong *twiddles   [[buffer(2)]],
    constant uint      &stage_idx  [[buffer(3)]],
    constant uint      &log_N      [[buffer(4)]],
    uint k [[thread_position_in_grid]])
{
    uint lN     = log_N;
    uint half_N = 1u << (lN - 1u);
    if (k >= half_N) return;

    uint s         = stage_idx;
    uint span_half = 1u << s;          // 2^s
    uint mask      = span_half - 1u;

    uint j = k >> s;
    uint r = k & mask;

    uint i0 = (j << s) + r;
    uint i1 = i0 + half_N;

    uint o0 = (j << (s + 1u)) + r;
    uint o1 = o0 + span_half;

    uint tw_shift = lN - 1u - s;
    uint tw_idx   = r << tw_shift;

    ulong u = in_data[i0];
    ulong x = in_data[i1];

    // Stage 0 (s==0) always has r==0 -> twiddle = 1, skip mul.
    // For other stages, r==0 lanes also skip; this is a warp-coherent
    // branch only when s==0 (entire grid), otherwise divergent but cheap.
    ulong v;
    if (tw_idx == 0u) {
        v = x;
    } else {
        ulong w = twiddles[tw_idx];
        v = gold_mul(x, w);
    }

    out_data[o0] = gold_add(u, v);
    out_data[o1] = gold_sub(u, v);
}