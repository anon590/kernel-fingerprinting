#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD    = 0xFFFFFFFF00000001ul;  // 2^64 - 2^32 + 1
constant ulong EPSILON   = 0x00000000FFFFFFFFul;  // 2^32 - 1
constant ulong ROOT4_NEG = 0xFFFEFFFF00000001ul;  // p - 2^48

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

inline ulong gold_neg(ulong x) {
    return (x == 0ul) ? 0ul : (P_GOLD - x);
}

inline ulong2 umul128(ulong a, ulong b) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
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
    ulong2 p = umul128(a, b);
    return gold_reduce128(p.x, p.y);
}

// Multiply by +2^48 modulo Goldilocks.
inline ulong gold_mul_root4_pos(ulong x) {
    return gold_reduce128(x << 48, x >> 16);
}

kernel void goldilocks_ntt_stage(
    device const ulong *in_data    [[buffer(0)]],
    device       ulong *out_data   [[buffer(1)]],
    device const ulong *twiddles   [[buffer(2)]],
    constant uint      &stage_idx  [[buffer(3)]],
    constant uint      &log_N      [[buffer(4)]],
    uint k [[thread_position_in_grid]])
{
    uint lm1    = log_N - 1u;
    uint half_N = 1u << lm1;
    if (k >= half_N) return;

    uint s = stage_idx;

    ulong u = in_data[k];
    ulong x = in_data[k + half_N];

    // Stage 0: twiddle is exactly 1 for every butterfly.
    if (s == 0u) {
        uint o0 = k << 1u;
        out_data[o0]      = gold_add(u, x);
        out_data[o0 + 1u] = gold_sub(u, x);
        return;
    }

    // Stage 1: twiddles are {1, +/-2^48}; avoid the full 64x64 multiply.
    if (s == 1u) {
        uint r  = k & 1u;
        uint o0 = (k << 1u) - r;

        ulong v = x;
        if (r != 0u) {
            v = gold_mul_root4_pos(x);
            ulong w4 = twiddles[half_N >> 1u];
            if (w4 == ROOT4_NEG) {
                v = gold_neg(v);
            }
        }

        out_data[o0]      = gold_add(u, v);
        out_data[o0 + 2u] = gold_sub(u, v);
        return;
    }

    // General stages.  The contract's read offset i0 simplifies to k.
    uint span_half = 1u << s;
    uint r         = k & (span_half - 1u);
    uint o0        = (k << 1u) - r;
    uint o1        = o0 + span_half;

    uint tw_stride = 1u << (lm1 - s);
    ulong w        = twiddles[r * tw_stride];

    ulong v = gold_mul(x, w);

    out_data[o0] = gold_add(u, v);
    out_data[o1] = gold_sub(u, v);
}