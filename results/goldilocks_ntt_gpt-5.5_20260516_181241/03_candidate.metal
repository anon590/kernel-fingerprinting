#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD    = 0xFFFFFFFF00000001ul;  // 2^64 - 2^32 + 1
constant ulong EPSILON   = 0x00000000FFFFFFFFul;  // 2^32 - 1
constant ulong ROOT4_POS = 0x0001000000000000ul;  // 2^48
constant ulong ROOT4_NEG = 0xFFFEFFFF00000001ul;  // p - 2^48

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    t += (t < a) ? EPSILON : 0ul;
    t -= (t >= P_GOLD) ? P_GOLD : 0ul;
    return t;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    t -= (t > a) ? EPSILON : 0ul;
    return t;
}

inline ulong gold_neg(ulong x) {
    return (x == 0ul) ? 0ul : (P_GOLD - x);
}

// Fold x_lo + (hi_lo + hi_hi * 2^32) * 2^64 modulo p.
// hi_lo must be < 2^32.
inline ulong gold_reduce_fold(ulong x_lo, ulong hi_lo, ulong hi_hi) {
    ulong t0 = x_lo - hi_hi;
    t0 -= (t0 > x_lo) ? EPSILON : 0ul;

    ulong t1 = hi_lo * EPSILON;

    ulong t2 = t0 + t1;
    t2 += (t2 < t0) ? EPSILON : 0ul;

    return gold_canonical(t2);
}

inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    return gold_reduce_fold(x_lo, x_hi & EPSILON, x_hi >> 32);
}

// Exact Goldilocks multiplication using explicit 32-bit limb mul/mulhi.
inline ulong gold_mul(ulong a, ulong b) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    uint p00_lo = a0 * b0;
    uint p00_hi = mulhi(a0, b0);

    uint p01_lo = a0 * b1;
    uint p01_hi = mulhi(a0, b1);

    uint p10_lo = a1 * b0;
    uint p10_hi = mulhi(a1, b0);

    uint p11_lo = a1 * b1;
    uint p11_hi = mulhi(a1, b1);

    ulong s1 = (ulong)p00_hi + (ulong)p01_lo + (ulong)p10_lo;
    uint  c1 = (uint)s1;
    ulong carry1 = s1 >> 32;

    ulong s2 = (ulong)p01_hi + (ulong)p10_hi + (ulong)p11_lo + carry1;
    uint  c2 = (uint)s2;
    ulong c3 = (ulong)p11_hi + (s2 >> 32);

    ulong x_lo = ((ulong)c1 << 32) | (ulong)p00_lo;
    return gold_reduce_fold(x_lo, (ulong)c2, c3);
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
    uint half_N = 1u << (log_N - 1u);
    if (k >= half_N) return;

    uint s = stage_idx;

    ulong u = in_data[k];
    ulong x = in_data[k + half_N];

    // Stage 0: twiddle is 1 for every butterfly.
    if (s == 0u) {
        uint o0 = k << 1;
        out_data[o0]      = gold_add(u, x);
        out_data[o0 + 1u] = gold_sub(u, x);
        return;
    }

    // Stage 1: twiddles are {1, +/-2^48}; avoid full 64x64 multiplication.
    if (s == 1u) {
        uint r  = k & 1u;
        uint o0 = (k << 1) - r;

        ulong y = gold_mul_root4_pos(x);
        ulong w4 = twiddles[half_N >> 1u];
        if (w4 == ROOT4_NEG) {
            y = gold_neg(y);
        }

        ulong v = (r == 0u) ? x : y;
        out_data[o0]      = gold_add(u, v);
        out_data[o0 + 2u] = gold_sub(u, v);
        return;
    }

    uint span_half = 1u << s;
    uint r         = k & (span_half - 1u);

    uint o0 = (k << 1) - r;
    uint o1 = o0 + span_half;

    uint tw_idx = r << (log_N - 1u - s);
    ulong v = gold_mul(x, twiddles[tw_idx]);

    out_data[o0] = gold_add(u, v);
    out_data[o1] = gold_sub(u, v);
}