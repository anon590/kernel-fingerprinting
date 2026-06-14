#include <metal_stdlib>
using namespace metal;

// ----------------------------------------------------------------------
// Carry-less 16x16 -> 32 using 3 interleaved bit lanes.
// ----------------------------------------------------------------------

inline uint clmul16_full(uint x, uint y)
{
    const uint A0 = 0x00009249u;
    const uint A1 = 0x00002492u;
    const uint A2 = 0x00004924u;

    const uint M0 = 0x49249249u;
    const uint M1 = 0x92492492u;
    const uint M2 = 0x24924924u;

    uint x0 = x & A0;
    uint x1 = x & A1;
    uint x2 = x & A2;
    uint y0 = y & A0;
    uint y1 = y & A1;
    uint y2 = y & A2;

    uint z0 = (x0 * y0) ^ (x1 * y2) ^ (x2 * y1);
    uint z1 = (x0 * y1) ^ (x1 * y0) ^ (x2 * y2);
    uint z2 = (x0 * y2) ^ (x1 * y1) ^ (x2 * y0);

    return (z0 & M0) | (z1 & M1) | (z2 & M2);
}

// ----------------------------------------------------------------------
// Carry-less 32x32 -> 64, returned as two 32-bit limbs.
// ----------------------------------------------------------------------

inline uint2 clmul32_u32(uint x, uint y)
{
    uint x0 = x & 0xffffu;
    uint x1 = x >> 16u;
    uint y0 = y & 0xffffu;
    uint y1 = y >> 16u;

    uint p0 = clmul16_full(x0, y0);
    uint p2 = clmul16_full(x1, y1);
    uint pm = clmul16_full(x0 ^ x1, y0 ^ y1);
    uint mid = pm ^ p0 ^ p2;

    return uint2(p0 ^ (mid << 16u), (mid >> 16u) ^ p2);
}

// ----------------------------------------------------------------------
// Carry-less 64x64 -> 128, returned as four 32-bit limbs.
// Inputs are little-endian 32-bit limbs.
// ----------------------------------------------------------------------

inline uint4 clmul64_u32(uint a0, uint a1, uint b0, uint b1)
{
    uint2 p0 = clmul32_u32(a0, b0);
    uint2 p2 = clmul32_u32(a1, b1);
    uint2 pm = clmul32_u32(a0 ^ a1, b0 ^ b1);
    uint2 mid = pm ^ p0 ^ p2;

    return uint4(p0.x,
                 p0.y ^ mid.x,
                 mid.y ^ p2.x,
                 p2.y);
}

// ----------------------------------------------------------------------
// Reduction modulo x^128 + x^7 + x^2 + x + 1.
// lo = limbs 0..3, hi = limbs 4..7 of the unreduced product.
// ----------------------------------------------------------------------

inline uint4 gcm_reduce_u32(uint4 lo, uint4 hi)
{
    uint h0 = hi.x;
    uint h1 = hi.y;
    uint h2 = hi.z;
    uint h3 = hi.w;

    uint r0 = lo.x
            ^ h0
            ^ (h0 << 1u)
            ^ (h0 << 2u)
            ^ (h0 << 7u);

    uint r1 = lo.y
            ^ h1
            ^ ((h1 << 1u) | (h0 >> 31u))
            ^ ((h1 << 2u) | (h0 >> 30u))
            ^ ((h1 << 7u) | (h0 >> 25u));

    uint r2 = lo.z
            ^ h2
            ^ ((h2 << 1u) | (h1 >> 31u))
            ^ ((h2 << 2u) | (h1 >> 30u))
            ^ ((h2 << 7u) | (h1 >> 25u));

    uint r3 = lo.w
            ^ h3
            ^ ((h3 << 1u) | (h2 >> 31u))
            ^ ((h3 << 2u) | (h2 >> 30u))
            ^ ((h3 << 7u) | (h2 >> 25u));

    uint residual = (h3 >> 31u) ^ (h3 >> 30u) ^ (h3 >> 25u);
    r0 ^= residual ^ (residual << 1u) ^ (residual << 2u) ^ (residual << 7u);

    return uint4(r0, r1, r2, r3);
}

// ----------------------------------------------------------------------
// GF(2^128) multiply using 32-bit-limb Karatsuba composition.
// uint4 order is little-endian polynomial order.
// ----------------------------------------------------------------------

inline uint4 gf128_mul_u32(uint4 a, uint4 b)
{
    uint4 z0 = clmul64_u32(a.x, a.y, b.x, b.y);
    uint4 z2 = clmul64_u32(a.z, a.w, b.z, b.w);
    uint4 zm = clmul64_u32(a.x ^ a.z, a.y ^ a.w,
                           b.x ^ b.z, b.y ^ b.w);

    uint4 z1 = zm ^ z0 ^ z2;

    uint4 lo = uint4(z0.x,
                     z0.y,
                     z0.z ^ z1.x,
                     z0.w ^ z1.y);

    uint4 hi = uint4(z2.x ^ z1.z,
                     z2.y ^ z1.w,
                     z2.z,
                     z2.w);

    return gcm_reduce_u32(lo, hi);
}

// ----------------------------------------------------------------------
// Sparse 128x128 carry-less multiply by a chosen multiplier, uint limbs.
// Used for low-weight alpha in the tower path.
// ----------------------------------------------------------------------

inline uint4 gf128_mul_sparse_multiplier_u32(uint4 x, uint4 m)
{
    uint t0 = 0u, t1 = 0u, t2 = 0u, t3 = 0u;
    uint t4 = 0u, t5 = 0u, t6 = 0u, t7 = 0u;

    uint mw = m.x;
    while (mw != 0u) {
        uint s = (uint)ctz(mw);
        if (s == 0u) {
            t0 ^= x.x; t1 ^= x.y; t2 ^= x.z; t3 ^= x.w;
        } else {
            uint r = 32u - s;
            t0 ^= x.x << s;
            t1 ^= (x.y << s) | (x.x >> r);
            t2 ^= (x.z << s) | (x.y >> r);
            t3 ^= (x.w << s) | (x.z >> r);
            t4 ^= x.w >> r;
        }
        mw &= mw - 1u;
    }

    mw = m.y;
    while (mw != 0u) {
        uint s = (uint)ctz(mw);
        if (s == 0u) {
            t1 ^= x.x; t2 ^= x.y; t3 ^= x.z; t4 ^= x.w;
        } else {
            uint r = 32u - s;
            t1 ^= x.x << s;
            t2 ^= (x.y << s) | (x.x >> r);
            t3 ^= (x.z << s) | (x.y >> r);
            t4 ^= (x.w << s) | (x.z >> r);
            t5 ^= x.w >> r;
        }
        mw &= mw - 1u;
    }

    mw = m.z;
    while (mw != 0u) {
        uint s = (uint)ctz(mw);
        if (s == 0u) {
            t2 ^= x.x; t3 ^= x.y; t4 ^= x.z; t5 ^= x.w;
        } else {
            uint r = 32u - s;
            t2 ^= x.x << s;
            t3 ^= (x.y << s) | (x.x >> r);
            t4 ^= (x.z << s) | (x.y >> r);
            t5 ^= (x.w << s) | (x.z >> r);
            t6 ^= x.w >> r;
        }
        mw &= mw - 1u;
    }

    mw = m.w;
    while (mw != 0u) {
        uint s = (uint)ctz(mw);
        if (s == 0u) {
            t3 ^= x.x; t4 ^= x.y; t5 ^= x.z; t6 ^= x.w;
        } else {
            uint r = 32u - s;
            t3 ^= x.x << s;
            t4 ^= (x.y << s) | (x.x >> r);
            t5 ^= (x.z << s) | (x.y >> r);
            t6 ^= (x.w << s) | (x.z >> r);
            t7 ^= x.w >> r;
        }
        mw &= mw - 1u;
    }

    return gcm_reduce_u32(uint4(t0, t1, t2, t3),
                          uint4(t4, t5, t6, t7));
}

inline uint4 gf128_mul_alpha_weighted_u32(uint4 x, uint4 alpha, uint alpha_weight)
{
    if (alpha_weight <= 64u) {
        return gf128_mul_sparse_multiplier_u32(x, alpha);
    } else {
        return gf128_mul_u32(x, alpha);
    }
}

inline uint4 load_u128_u32(device const uint *p, uint base)
{
    return uint4(p[base + 0u],
                 p[base + 1u],
                 p[base + 2u],
                 p[base + 3u]);
}

inline void store_u128_u32(device uint *p, uint base, uint4 v)
{
    p[base + 0u] = v.x;
    p[base + 1u] = v.y;
    p[base + 2u] = v.z;
    p[base + 3u] = v.w;
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
    if (idx >= batch) {
        return;
    }

    device const uint *au = reinterpret_cast<device const uint *>(a);
    device const uint *bu = reinterpret_cast<device const uint *>(b);
    device       uint *cu = reinterpret_cast<device       uint *>(c);

    if (tower == 0u) {
        uint base = idx << 2u;

        uint4 av = load_u128_u32(au, base);
        uint4 bv = load_u128_u32(bu, base);
        uint4 cv = gf128_mul_u32(av, bv);

        store_u128_u32(cu, base, cv);
        return;
    }

    uint base = idx << 3u;

    uint4 a0 = load_u128_u32(au, base + 0u);
    uint4 a1 = load_u128_u32(au, base + 4u);
    uint4 b0 = load_u128_u32(bu, base + 0u);
    uint4 b1 = load_u128_u32(bu, base + 4u);

    uint4 alpha = uint4((uint)alpha_lo,
                        (uint)(alpha_lo >> 32u),
                        (uint)alpha_hi,
                        (uint)(alpha_hi >> 32u));

    uint alpha_weight = 0u;
    if ((idx & 31u) == 0u) {
        alpha_weight = (uint)popcount(alpha.x)
                     + (uint)popcount(alpha.y)
                     + (uint)popcount(alpha.z)
                     + (uint)popcount(alpha.w);
    }
    alpha_weight = simd_broadcast(alpha_weight, 0);

    uint4 m00 = gf128_mul_u32(a0, b0);
    uint4 m11 = gf128_mul_u32(a1, b1);
    uint4 mix = gf128_mul_u32(a0 ^ a1, b0 ^ b1);
    uint4 am  = gf128_mul_alpha_weighted_u32(m11, alpha, alpha_weight);

    store_u128_u32(cu, base + 0u, m00 ^ am);
    store_u128_u32(cu, base + 4u, mix ^ m00);
}