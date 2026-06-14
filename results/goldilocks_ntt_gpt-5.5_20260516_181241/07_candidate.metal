#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD    = 0xFFFFFFFF00000001ul;  // 2^64 - 2^32 + 1
constant ulong EPSILON   = 0x00000000FFFFFFFFul;  // 2^32 - 1
constant ulong ROOT4_NEG = 0xFFFEFFFF00000001ul;  // p - 2^48

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    t += (t < a) ? EPSILON : 0ul;
    return (t >= P_GOLD) ? (t - P_GOLD) : t;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    t -= (t > a) ? EPSILON : 0ul;
    return t;
}

inline ulong gold_neg(ulong x) {
    return (x == 0ul) ? 0ul : (P_GOLD - x);
}

// Goldilocks reduction of a 128-bit integer x = x_hi * 2^64 + x_lo.
// Since 2^64 == 2^32 - 1 (mod p):
//   x == x_lo + (x_hi_lo * (2^32 - 1)) - x_hi_hi.
inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong hi_lo = x_hi & EPSILON;
    ulong hi_hi = x_hi >> 32;

    ulong t0 = x_lo - hi_hi;
    t0 -= (t0 > x_lo) ? EPSILON : 0ul;

    // hi_lo * (2^32 - 1), with hi_lo < 2^32.
    ulong t1 = (hi_lo << 32) - hi_lo;

    ulong t2 = t0 + t1;
    t2 += (t2 < t0) ? EPSILON : 0ul;

    return (t2 >= P_GOLD) ? (t2 - P_GOLD) : t2;
}

// 64x64 -> 128 via four 32x32->64 products.
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

inline ulong gold_mul(ulong a, ulong b) {
    ulong2 p = umul128(a, b);
    return gold_reduce128(p.x, p.y);
}

// Multiply by +2^48 modulo Goldilocks.
inline ulong gold_mul_root4_pos(ulong x) {
    ulong lo = x << 48;
    ulong hi = x >> 16;
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
    uint half_N = 1u << (log_N - 1u);
    if (k >= half_N) return;

    uint s = stage_idx;

    ulong u = in_data[k];
    ulong x = in_data[k + half_N];

    // Stage 0: twiddle is 1 and outputs are adjacent.
    if (s == 0u) {
        device ulong2 *out2 = (device ulong2 *)out_data;
        out2[k] = ulong2(gold_add(u, x), gold_sub(u, x));
        return;
    }

    // Stage 1: twiddles are 1 and +/- 2^48, avoiding a general multiply.
    if (s == 1u) {
        uint r = k & 1u;

        ulong v_root = gold_mul_root4_pos(x);
        if (twiddles[half_N >> 1u] == ROOT4_NEG) {
            v_root = gold_neg(v_root);
        }

        ulong v = (r == 0u) ? x : v_root;

        uint o0 = (k << 1u) - r;
        out_data[o0]      = gold_add(u, v);
        out_data[o0 + 2u] = gold_sub(u, v);
        return;
    }

    uint m = 1u << s;
    uint r = k & (m - 1u);

    uint tw_stride = half_N >> s;
    ulong w = twiddles[r * tw_stride];
    ulong v = gold_mul(x, w);

    uint o0 = (k << 1u) - r;
    out_data[o0]     = gold_add(u, v);
    out_data[o0 + m] = gold_sub(u, v);
}