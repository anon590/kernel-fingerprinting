#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD       = 0xFFFFFFFF00000001ul;  // 2^64 - 2^32 + 1
constant ulong EPSILON      = 0x00000000FFFFFFFFul;  // 2^32 - 1
constant ulong ROOT4_POS    = 0x0001000000000000ul;  // 2^48, square = -1
constant ulong ROOT8_24_POS = 0x0000000001000000ul;  // 2^24, square = 2^48
constant ulong ROOT8_72_POS = 0x000000FFFFFFFF00ul;  // 2^72 mod p = 2^40 - 2^8

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

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

// Fold a 128-bit integer lo + hi*2^64 modulo p.
// Uses 2^64 == 2^32 - 1 and 2^96 == -1.
inline ulong gold_reduce128_fold(ulong lo, ulong hi) {
    ulong hi_lo = hi & EPSILON;
    ulong hi_hi = hi >> 32;

    ulong sub = hi_lo + hi_hi;
    ulong t = lo - sub;
    t -= (t > lo) ? EPSILON : 0ul;

    ulong add = hi_lo << 32;
    ulong r = t + add;
    r += (r < t) ? EPSILON : 0ul;

    return (r >= P_GOLD) ? (r - P_GOLD) : r;
}

// Incumbent-style 32x32->64 partial products, good on the smaller cases.
inline ulong2 umul128_wide(ulong a, ulong b) {
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

inline ulong gold_reduce128_wide(ulong lo, ulong hi) {
    ulong hi_lo = hi & EPSILON;
    ulong hi_hi = hi >> 32;

    ulong t0 = lo - hi_hi;
    if (t0 > lo) t0 -= EPSILON;

    ulong t1 = hi_lo * EPSILON;

    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;

    return gold_canonical(t2);
}

inline ulong gold_mul_wide(ulong a, ulong b) {
    ulong2 p = umul128_wide(a, b);
    return gold_reduce128_wide(p.x, p.y);
}

// Explicit mulhi-based 32-bit limb multiplication. This was faster for the
// largest benchmark size in the prior measurements.
inline ulong gold_mul_limb(ulong a, ulong b) {
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

    uint s1 = p00_hi + p01_lo;
    uint c1 = (s1 < p00_hi) ? 1u : 0u;

    uint z1 = s1 + p10_lo;
    c1 += (z1 < s1) ? 1u : 0u;

    uint s2 = p01_hi + p10_hi;
    uint c2 = (s2 < p01_hi) ? 1u : 0u;

    uint s3 = s2 + p11_lo;
    c2 += (s3 < s2) ? 1u : 0u;

    uint z2 = s3 + c1;
    c2 += (z2 < s3) ? 1u : 0u;

    uint z3 = p11_hi + c2;

    ulong lo = ((ulong)z1 << 32) | (ulong)p00_lo;

    ulong sub = (ulong)z2 + (ulong)z3;
    ulong t = lo - sub;
    t -= (t > lo) ? EPSILON : 0ul;

    ulong add = ((ulong)z2) << 32;
    ulong r = t + add;
    r += (r < t) ? EPSILON : 0ul;

    return (r >= P_GOLD) ? (r - P_GOLD) : r;
}

inline ulong gold_mul_selected(ulong a, ulong b, uint log_N) {
    if (log_N >= 18u) {
        return gold_mul_limb(a, b);
    } else {
        return gold_mul_wide(a, b);
    }
}

// Exact multiplication by small powers of two in the field.
inline ulong gold_mul_pow2_8(ulong x) {
    return gold_reduce128_fold(x << 8, x >> 56);
}

inline ulong gold_mul_pow2_24(ulong x) {
    return gold_reduce128_fold(x << 24, x >> 40);
}

inline ulong gold_mul_pow2_40(ulong x) {
    return gold_reduce128_fold(x << 40, x >> 24);
}

inline ulong gold_mul_pow2_48(ulong x) {
    return gold_reduce128_fold(x << 48, x >> 16);
}

// 2^72 == 2^40 - 2^8 mod p.
inline ulong gold_mul_pow2_72(ulong x) {
    return gold_sub(gold_mul_pow2_40(x), gold_mul_pow2_8(x));
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

    // Stage 0: twiddle is 1 for every butterfly.
    if (s == 0u) {
        ulong u = in_data[k];
        ulong x = in_data[k + half_N];

        uint o = k << 1u;
        out_data[o]      = gold_add(u, x);
        out_data[o + 1u] = gold_sub(u, x);
        return;
    }

    // Stage 1: twiddles are {1, +/-2^48}; avoid generic multiplication.
    if (s == 1u) {
        uint r = k & 1u;

        ulong u = in_data[k];
        ulong x = in_data[k + half_N];

        ulong root4 = twiddles[half_N >> 1u];
        ulong y = gold_mul_pow2_48(x);
        if (root4 != ROOT4_POS) {
            y = gold_neg(y);
        }

        ulong v = (r == 0u) ? x : y;

        uint o0 = (k << 1u) - r;
        out_data[o0]      = gold_add(u, v);
        out_data[o0 + 2u] = gold_sub(u, v);
        return;
    }

    // Stage 2: twiddles are 8th roots, all +/-2^24 or +/-2^72, with
    // the square equal to +/-2^48. Replace full field mul by shifts/folds.
    if (s == 2u) {
        uint r = k & 3u;

        ulong u = in_data[k];
        ulong x = in_data[k + half_N];

        uint stride = half_N >> 2u;
        ulong root8 = twiddles[stride];
        ulong root4 = twiddles[stride << 1u];

        ulong y24 = gold_mul_pow2_24(x);
        ulong y48 = gold_mul_pow2_48(x);
        ulong y72 = gold_mul_pow2_72(x);

        ulong y1;
        ulong y2;
        ulong y3;

        if (root4 == ROOT4_POS) {
            bool neg8 = (root8 != ROOT8_24_POS);
            y1 = neg8 ? gold_neg(y24) : y24;
            y2 = y48;
            y3 = neg8 ? gold_neg(y72) : y72;
        } else {
            bool neg8 = (root8 != ROOT8_72_POS);
            y1 = neg8 ? gold_neg(y72) : y72;
            y2 = gold_neg(y48);
            y3 = neg8 ? gold_neg(y24) : y24;
        }

        ulong v = (r == 0u) ? x : ((r == 1u) ? y1 : ((r == 2u) ? y2 : y3));

        uint o0 = (k << 1u) - r;
        out_data[o0]      = gold_add(u, v);
        out_data[o0 + 4u] = gold_sub(u, v);
        return;
    }

    // Last stage has contiguous output and twiddle index k.
    if (s == (log_N - 1u)) {
        ulong u = in_data[k];
        ulong x = in_data[k + half_N];
        ulong w = twiddles[k];
        ulong v = gold_mul_selected(x, w, log_N);

        out_data[k]          = gold_add(u, v);
        out_data[k + half_N] = gold_sub(u, v);
        return;
    }

    uint m = 1u << s;
    uint r = k & (m - 1u);

    ulong u = in_data[k];
    ulong x = in_data[k + half_N];

    uint tw_stride = half_N >> s;
    ulong w = twiddles[r * tw_stride];
    ulong v = gold_mul_selected(x, w, log_N);

    uint o0 = (k << 1u) - r;
    out_data[o0]     = gold_add(u, v);
    out_data[o0 + m] = gold_sub(u, v);
}