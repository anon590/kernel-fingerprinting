#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD       = 0xFFFFFFFF00000001ul;  // 2^64 - 2^32 + 1
constant ulong EPSILON      = 0x00000000FFFFFFFFul;  // 2^32 - 1

constant ulong ROOT4_POS    = 0x0001000000000000ul;  // 2^48, square = -1
constant ulong ROOT8_24_POS = 0x0000000001000000ul;  // 2^24
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

// 64x64 -> 128 via 32-bit limbs. The 32x32 products fit exactly in ulong.
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

// Reduce lo + hi*2^64 mod p, using 2^64 = 2^32 - 1.
// Explicit shift/sub avoids relying on constant-multiply lowering.
inline ulong gold_reduce128(ulong lo, ulong hi) {
    ulong hi_lo = hi & EPSILON;
    ulong hi_hi = hi >> 32;

    ulong t0 = lo - hi_hi;
    t0 -= (t0 > lo) ? EPSILON : 0ul;

    ulong t1 = (hi_lo << 32) - hi_lo;

    ulong t2 = t0 + t1;
    t2 += (t2 < t0) ? EPSILON : 0ul;

    return gold_canonical(t2);
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong2 p = umul128(a, b);
    return gold_reduce128(p.x, p.y);
}

// Generic fold for x * 2^shift, represented as a 128-bit value.
inline ulong gold_reduce128_fold(ulong lo, ulong hi) {
    ulong hi_lo = hi & EPSILON;
    ulong hi_hi = hi >> 32;

    ulong sub = hi_lo + hi_hi;
    ulong t = lo - sub;
    t -= (t > lo) ? EPSILON : 0ul;

    ulong add = hi_lo << 32;
    ulong r = t + add;
    r += (r < t) ? EPSILON : 0ul;

    return gold_canonical(r);
}

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

    ulong u = in_data[k];
    ulong x = in_data[k + half_N];

    // Stage 0: twiddle is 1 and output pairs are contiguous.
    if (s == 0u) {
        uint o = k << 1u;
        out_data[o]      = gold_add(u, x);
        out_data[o + 1u] = gold_sub(u, x);
        return;
    }

    // For the large case, avoid full field multiplies in the first
    // nontrivial root stages. These are uniform per dispatch and cheap.
    if (log_N >= 18u) {
        if (s == 1u) {
            uint r = k & 1u;
            ulong v = x;

            if (r != 0u) {
                ulong y = gold_mul_pow2_48(x);
                ulong root4 = twiddles[half_N >> 1u];
                v = (root4 == ROOT4_POS) ? y : gold_neg(y);
            }

            uint o0 = (k << 1u) - r;
            out_data[o0]      = gold_add(u, v);
            out_data[o0 + 2u] = gold_sub(u, v);
            return;
        }

        if (s == 2u) {
            uint r = k & 3u;
            ulong v = x;

            if (r != 0u) {
                uint stride = half_N >> 2u;
                ulong root4 = twiddles[stride << 1u];

                if (root4 == ROOT4_POS) {
                    if (r == 2u) {
                        v = gold_mul_pow2_48(x);
                    } else {
                        ulong root8 = twiddles[stride];
                        bool neg8 = (root8 != ROOT8_24_POS);
                        v = (r == 1u) ? gold_mul_pow2_24(x) : gold_mul_pow2_72(x);
                        v = neg8 ? gold_neg(v) : v;
                    }
                } else {
                    if (r == 2u) {
                        v = gold_neg(gold_mul_pow2_48(x));
                    } else {
                        ulong root8 = twiddles[stride];
                        bool neg8 = (root8 != ROOT8_72_POS);
                        v = (r == 1u) ? gold_mul_pow2_72(x) : gold_mul_pow2_24(x);
                        v = neg8 ? gold_neg(v) : v;
                    }
                }
            }

            uint o0 = (k << 1u) - r;
            out_data[o0]      = gold_add(u, v);
            out_data[o0 + 4u] = gold_sub(u, v);
            return;
        }
    }

    // Last stage: r = k, twiddle index = k, and output is contiguous.
    if (s == (log_N - 1u)) {
        ulong v = gold_mul(x, twiddles[k]);
        out_data[k]          = gold_add(u, v);
        out_data[k + half_N] = gold_sub(u, v);
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