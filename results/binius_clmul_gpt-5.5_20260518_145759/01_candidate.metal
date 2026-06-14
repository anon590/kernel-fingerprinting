#include <metal_stdlib>
using namespace metal;

// ----------------------------------------------------------------------
// Carry-less 32x32 -> 64 using 4 interleaved bit lanes.
//
// For 32-bit operands each lane has only 8 bits, so ordinary base-16
// products have coefficients < 16 and therefore no carries between
// nibbles.  Masking the four residue classes gives the exact GF(2)
// polynomial product.
// ----------------------------------------------------------------------

inline ulong umul32wide(uint x, uint y)
{
    return (((ulong)mulhi(x, y)) << 32u) | (ulong)(x * y);
}

inline ulong clmul32_full(uint x, uint y)
{
    const uint  A0 = 0x11111111u;
    const uint  A1 = 0x22222222u;
    const uint  A2 = 0x44444444u;
    const uint  A3 = 0x88888888u;

    const ulong M0 = 0x1111111111111111ul;
    const ulong M1 = 0x2222222222222222ul;
    const ulong M2 = 0x4444444444444444ul;
    const ulong M3 = 0x8888888888888888ul;

    uint x0 = x & A0, x1 = x & A1, x2 = x & A2, x3 = x & A3;
    uint y0 = y & A0, y1 = y & A1, y2 = y & A2, y3 = y & A3;

    ulong z0 = umul32wide(x0, y0) ^ umul32wide(x1, y3)
             ^ umul32wide(x2, y2) ^ umul32wide(x3, y1);
    ulong z1 = umul32wide(x0, y1) ^ umul32wide(x1, y0)
             ^ umul32wide(x2, y3) ^ umul32wide(x3, y2);
    ulong z2 = umul32wide(x0, y2) ^ umul32wide(x1, y1)
             ^ umul32wide(x2, y0) ^ umul32wide(x3, y3);
    ulong z3 = umul32wide(x0, y3) ^ umul32wide(x1, y2)
             ^ umul32wide(x2, y1) ^ umul32wide(x3, y0);

    return (z0 & M0) | (z1 & M1) | (z2 & M2) | (z3 & M3);
}

// ----------------------------------------------------------------------
// Carry-less 64x64 -> 128 by Karatsuba over 32-bit halves.
// ----------------------------------------------------------------------

inline void clmul64_full(
    ulong x, ulong y,
    thread ulong &lo, thread ulong &hi)
{
    uint x0 = (uint)x;
    uint x1 = (uint)(x >> 32u);
    uint y0 = (uint)y;
    uint y1 = (uint)(y >> 32u);

    ulong p0 = clmul32_full(x0, y0);
    ulong p2 = clmul32_full(x1, y1);
    ulong pm = clmul32_full(x0 ^ x1, y0 ^ y1);
    ulong mid = pm ^ p0 ^ p2;

    lo = p0 ^ (mid << 32u);
    hi = (mid >> 32u) ^ p2;
}

// ----------------------------------------------------------------------
// Carry-less 128x128 -> 256 by Karatsuba over 64-bit halves.
// ----------------------------------------------------------------------

inline void clmul128_unreduced(
    ulong a_lo, ulong a_hi, ulong b_lo, ulong b_hi,
    thread ulong &t0, thread ulong &t1,
    thread ulong &t2, thread ulong &t3)
{
    ulong z0_lo, z0_hi;
    ulong z2_lo, z2_hi;
    ulong zm_lo, zm_hi;

    clmul64_full(a_lo, b_lo, z0_lo, z0_hi);
    clmul64_full(a_hi, b_hi, z2_lo, z2_hi);
    clmul64_full(a_lo ^ a_hi, b_lo ^ b_hi, zm_lo, zm_hi);

    ulong z1_lo = zm_lo ^ z0_lo ^ z2_lo;
    ulong z1_hi = zm_hi ^ z0_hi ^ z2_hi;

    t0 = z0_lo;
    t1 = z0_hi ^ z1_lo;
    t2 = z2_lo ^ z1_hi;
    t3 = z2_hi;
}

// ----------------------------------------------------------------------
// Sparse multiplier scan, used only for the tower alpha multiply.
// The scanned operand is uniform across the dispatch, so loop divergence
// is avoided when alpha is low/medium weight.
// ----------------------------------------------------------------------

inline void clmul128_unreduced_sparse_multiplier(
    ulong x_lo, ulong x_hi, ulong m_lo, ulong m_hi,
    thread ulong &t0, thread ulong &t1,
    thread ulong &t2, thread ulong &t3)
{
    t0 = 0ul; t1 = 0ul; t2 = 0ul; t3 = 0ul;

    ulong ml = m_lo;
    while (ml != 0ul) {
        uint s = (uint)ctz(ml);
        if (s == 0u) {
            t0 ^= x_lo;
            t1 ^= x_hi;
        } else {
            t0 ^= x_lo << s;
            t1 ^= (x_hi << s) | (x_lo >> (64u - s));
            t2 ^= x_hi >> (64u - s);
        }
        ml &= (ml - 1ul);
    }

    ulong mh = m_hi;
    while (mh != 0ul) {
        uint s = (uint)ctz(mh);
        if (s == 0u) {
            t1 ^= x_lo;
            t2 ^= x_hi;
        } else {
            t1 ^= x_lo << s;
            t2 ^= (x_hi << s) | (x_lo >> (64u - s));
            t3 ^= x_hi >> (64u - s);
        }
        mh &= (mh - 1ul);
    }
}

// ----------------------------------------------------------------------
// Reduction modulo x^128 + x^7 + x^2 + x + 1.
// ----------------------------------------------------------------------

inline void gcm_reduce(
    ulong t0, ulong t1, ulong t2, ulong t3,
    thread ulong &r_lo, thread ulong &r_hi)
{
    ulong d_lo0 = t2 ^ (t2 << 1u) ^ (t2 << 2u) ^ (t2 << 7u);
    ulong d_lo1 = t3
                ^ ((t3 << 1u) | (t2 >> 63u))
                ^ ((t3 << 2u) | (t2 >> 62u))
                ^ ((t3 << 7u) | (t2 >> 57u));
    ulong d_hi  = (t3 >> 63u) ^ (t3 >> 62u) ^ (t3 >> 57u);

    t0 ^= d_lo0;
    t1 ^= d_lo1;
    t0 ^= d_hi ^ (d_hi << 1u) ^ (d_hi << 2u) ^ (d_hi << 7u);

    r_lo = t0;
    r_hi = t1;
}

inline void gf128_mul(
    ulong a_lo, ulong a_hi, ulong b_lo, ulong b_hi,
    thread ulong &c_lo, thread ulong &c_hi)
{
    ulong t0, t1, t2, t3;
    clmul128_unreduced(a_lo, a_hi, b_lo, b_hi, t0, t1, t2, t3);
    gcm_reduce(t0, t1, t2, t3, c_lo, c_hi);
}

inline void gf128_mul_sparse_multiplier(
    ulong x_lo, ulong x_hi, ulong m_lo, ulong m_hi,
    thread ulong &c_lo, thread ulong &c_hi)
{
    ulong t0, t1, t2, t3;
    clmul128_unreduced_sparse_multiplier(x_lo, x_hi, m_lo, m_hi, t0, t1, t2, t3);
    gcm_reduce(t0, t1, t2, t3, c_lo, c_hi);
}

inline void gf128_mul_alpha(
    ulong x_lo, ulong x_hi, ulong alpha_lo, ulong alpha_hi,
    thread ulong &c_lo, thread ulong &c_hi)
{
    uint alpha_weight = (uint)popcount(alpha_lo) + (uint)popcount(alpha_hi);

    if (alpha_weight <= 64u) {
        gf128_mul_sparse_multiplier(x_lo, x_hi, alpha_lo, alpha_hi, c_lo, c_hi);
    } else {
        gf128_mul(alpha_lo, alpha_hi, x_lo, x_hi, c_lo, c_hi);
    }
}

// ----------------------------------------------------------------------
// Required kernel.
// ----------------------------------------------------------------------

kernel void binius_clmul(
    device const ulong *a         [[buffer(0)]],
    device const ulong *b         [[buffer(1)]],
    device       ulong *c         [[buffer(2)]],
    constant ulong     &alpha_lo  [[buffer(3)]],
    constant ulong     &alpha_hi  [[buffer(4)]],
    constant uint      &tower     [[buffer(5)]],
    constant uint      &batch     [[buffer(6)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= batch) return;

    if (tower == 0u) {
        size_t base = (size_t)idx * (size_t)2;

        ulong a_lo = a[base + 0];
        ulong a_hi = a[base + 1];
        ulong b_lo = b[base + 0];
        ulong b_hi = b[base + 1];

        ulong c_lo, c_hi;
        gf128_mul(a_lo, a_hi, b_lo, b_hi, c_lo, c_hi);

        c[base + 0] = c_lo;
        c[base + 1] = c_hi;
    } else {
        size_t base = (size_t)idx * (size_t)4;

        ulong a0_lo = a[base + 0], a0_hi = a[base + 1];
        ulong a1_lo = a[base + 2], a1_hi = a[base + 3];
        ulong b0_lo = b[base + 0], b0_hi = b[base + 1];
        ulong b1_lo = b[base + 2], b1_hi = b[base + 3];

        // Extension-field Karatsuba:
        // mix = (a0+a1)(b0+b1) = m00 + m01 + m10 + m11
        // c1  = m01 + m10 + m11 = mix + m00
        ulong m00_lo, m00_hi;
        ulong m11_lo, m11_hi;
        ulong mix_lo, mix_hi;
        ulong am_lo,  am_hi;

        gf128_mul(a0_lo, a0_hi, b0_lo, b0_hi, m00_lo, m00_hi);
        gf128_mul(a1_lo, a1_hi, b1_lo, b1_hi, m11_lo, m11_hi);
        gf128_mul(a0_lo ^ a1_lo, a0_hi ^ a1_hi,
                  b0_lo ^ b1_lo, b0_hi ^ b1_hi,
                  mix_lo, mix_hi);

        gf128_mul_alpha(m11_lo, m11_hi, alpha_lo, alpha_hi, am_lo, am_hi);

        c[base + 0] = m00_lo ^ am_lo;
        c[base + 1] = m00_hi ^ am_hi;
        c[base + 2] = mix_lo ^ m00_lo;
        c[base + 3] = mix_hi ^ m00_hi;
    }
}