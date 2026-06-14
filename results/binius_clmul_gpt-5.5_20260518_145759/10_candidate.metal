#include <metal_stdlib>
using namespace metal;

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

inline void clmul64_full(
    ulong x, ulong y,
    thread ulong &lo, thread ulong &hi)
{
    uint x0 = (uint)x;
    uint x1 = (uint)(x >> 32u);
    uint y0 = (uint)y;
    uint y1 = (uint)(y >> 32u);

    uint2 p0 = clmul32_u32(x0, y0);
    uint2 p2 = clmul32_u32(x1, y1);
    uint2 pm = clmul32_u32(x0 ^ x1, y0 ^ y1);
    uint2 mid = pm ^ p0 ^ p2;

    lo = ((ulong)p0.x) | (((ulong)(p0.y ^ mid.x)) << 32u);
    hi = ((ulong)(mid.y ^ p2.x)) | (((ulong)p2.y) << 32u);
}

inline void clmul64_u32_parts(
    uint a0, uint a1, uint b0, uint b1,
    thread uint &r0, thread uint &r1,
    thread uint &r2, thread uint &r3)
{
    uint2 p0 = clmul32_u32(a0, b0);
    uint2 p2 = clmul32_u32(a1, b1);
    uint2 pm = clmul32_u32(a0 ^ a1, b0 ^ b1);
    uint2 mid = pm ^ p0 ^ p2;

    r0 = p0.x;
    r1 = p0.y ^ mid.x;
    r2 = mid.y ^ p2.x;
    r3 = p2.y;
}

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

    t0 = z0_lo;
    t1 = z0_hi ^ zm_lo ^ z0_lo ^ z2_lo;
    t2 = z2_lo ^ zm_hi ^ z0_hi ^ z2_hi;
    t3 = z2_hi;
}

inline void gcm_reduce64(
    ulong t0, ulong t1, ulong t2, ulong t3,
    thread ulong &r_lo, thread ulong &r_hi)
{
    ulong q = (t3 >> 62u) ^ (t3 >> 57u);

    r_lo = t0
         ^ t2 ^ (t2 << 1u) ^ (t2 << 2u) ^ (t2 << 7u)
         ^ q  ^ (q  << 1u) ^ (q  << 2u) ^ (q  << 7u);

    r_hi = t1
         ^ t3 ^ (t3 << 1u) ^ (t3 << 2u) ^ (t3 << 7u)
         ^ (t2 >> 63u) ^ (t2 >> 62u) ^ (t2 >> 57u);
}

inline void gf128_mul64(
    ulong a_lo, ulong a_hi, ulong b_lo, ulong b_hi,
    thread ulong &c_lo, thread ulong &c_hi)
{
    ulong t0, t1, t2, t3;
    clmul128_unreduced(a_lo, a_hi, b_lo, b_hi, t0, t1, t2, t3);
    gcm_reduce64(t0, t1, t2, t3, c_lo, c_hi);
}

inline void gcm_reduce_u32_out(
    uint l0, uint l1, uint l2, uint l3,
    uint h0, uint h1, uint h2, uint h3,
    thread uint &r0, thread uint &r1,
    thread uint &r2, thread uint &r3)
{
    r0 = l0
       ^ h0 ^ (h0 << 1u) ^ (h0 << 2u) ^ (h0 << 7u);

    r1 = l1
       ^ h1 ^ (h1 << 1u) ^ (h1 << 2u) ^ (h1 << 7u)
       ^ (h0 >> 31u) ^ (h0 >> 30u) ^ (h0 >> 25u);

    r2 = l2
       ^ h2 ^ (h2 << 1u) ^ (h2 << 2u) ^ (h2 << 7u)
       ^ (h1 >> 31u) ^ (h1 >> 30u) ^ (h1 >> 25u);

    r3 = l3
       ^ h3 ^ (h3 << 1u) ^ (h3 << 2u) ^ (h3 << 7u)
       ^ (h2 >> 31u) ^ (h2 >> 30u) ^ (h2 >> 25u);

    uint q = (h3 >> 30u) ^ (h3 >> 25u);
    r0 ^= q ^ (q << 1u) ^ (q << 2u) ^ (q << 7u);
}

inline void gf128_mul_u32_limbs(
    uint a0, uint a1, uint a2, uint a3,
    uint b0, uint b1, uint b2, uint b3,
    thread uint &r0, thread uint &r1,
    thread uint &r2, thread uint &r3)
{
    uint z00, z01, z02, z03;
    uint z20, z21, z22, z23;
    uint zm0, zm1, zm2, zm3;

    clmul64_u32_parts(a0, a1, b0, b1, z00, z01, z02, z03);
    clmul64_u32_parts(a2, a3, b2, b3, z20, z21, z22, z23);
    clmul64_u32_parts(a0 ^ a2, a1 ^ a3,
                      b0 ^ b2, b1 ^ b3,
                      zm0, zm1, zm2, zm3);

    uint l0 = z00;
    uint l1 = z01;
    uint l2 = z02 ^ zm0 ^ z00 ^ z20;
    uint l3 = z03 ^ zm1 ^ z01 ^ z21;

    uint h0 = z20 ^ zm2 ^ z02 ^ z22;
    uint h1 = z21 ^ zm3 ^ z03 ^ z23;
    uint h2 = z22;
    uint h3 = z23;

    gcm_reduce_u32_out(l0, l1, l2, l3, h0, h1, h2, h3,
                       r0, r1, r2, r3);
}

inline void clmul128_unreduced_sparse_multiplier(
    ulong x_lo, ulong x_hi, ulong m_lo, ulong m_hi,
    thread ulong &t0, thread ulong &t1,
    thread ulong &t2, thread ulong &t3)
{
    t0 = 0ul;
    t1 = 0ul;
    t2 = 0ul;
    t3 = 0ul;

    ulong ml = m_lo;
    while (ml != 0ul) {
        uint s = (uint)ctz(ml);
        if (s == 0u) {
            t0 ^= x_lo;
            t1 ^= x_hi;
        } else {
            uint r = 64u - s;
            t0 ^= x_lo << s;
            t1 ^= (x_hi << s) | (x_lo >> r);
            t2 ^= x_hi >> r;
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
            uint r = 64u - s;
            t1 ^= x_lo << s;
            t2 ^= (x_hi << s) | (x_lo >> r);
            t3 ^= x_hi >> r;
        }
        mh &= (mh - 1ul);
    }
}

inline void gf128_mul_sparse_multiplier64(
    ulong x_lo, ulong x_hi, ulong m_lo, ulong m_hi,
    thread ulong &c_lo, thread ulong &c_hi)
{
    ulong t0, t1, t2, t3;
    clmul128_unreduced_sparse_multiplier(x_lo, x_hi, m_lo, m_hi, t0, t1, t2, t3);
    gcm_reduce64(t0, t1, t2, t3, c_lo, c_hi);
}

inline void gf128_xtime64(
    ulong x_lo, ulong x_hi,
    thread ulong &r_lo, thread ulong &r_hi)
{
    ulong carry = x_hi >> 63u;
    r_hi = (x_hi << 1u) | (x_lo >> 63u);
    r_lo = (x_lo << 1u) ^ (carry * 0x87ul);
}

inline void gf128_mul_xk64(
    ulong x_lo, ulong x_hi, uint k,
    thread ulong &c_lo, thread ulong &c_hi)
{
    if (k == 0u) {
        c_lo = x_lo;
        c_hi = x_hi;
        return;
    }

    ulong t0 = 0ul;
    ulong t1 = 0ul;
    ulong t2 = 0ul;
    ulong t3 = 0ul;

    if (k < 64u) {
        uint r = 64u - k;
        t0 = x_lo << k;
        t1 = (x_hi << k) | (x_lo >> r);
        t2 = x_hi >> r;
    } else if (k == 64u) {
        t1 = x_lo;
        t2 = x_hi;
    } else {
        uint s = k - 64u;
        uint r = 64u - s;
        t1 = x_lo << s;
        t2 = (x_hi << s) | (x_lo >> r);
        t3 = x_hi >> r;
    }

    gcm_reduce64(t0, t1, t2, t3, c_lo, c_hi);
}

inline void gf128_mul_alpha64(
    ulong x_lo, ulong x_hi,
    ulong alpha_lo_v, ulong alpha_hi_v,
    thread ulong &c_lo, thread ulong &c_hi)
{
    if (alpha_hi_v == 0ul && alpha_lo_v == 0ul) {
        c_lo = 0ul;
        c_hi = 0ul;
        return;
    }

    if (alpha_hi_v == 0ul && alpha_lo_v == 1ul) {
        c_lo = x_lo;
        c_hi = x_hi;
        return;
    }

    if (alpha_hi_v == 0ul && alpha_lo_v == 2ul) {
        gf128_xtime64(x_lo, x_hi, c_lo, c_hi);
        return;
    }

    uint aw = (uint)popcount(alpha_lo_v) + (uint)popcount(alpha_hi_v);

    if (aw == 1u) {
        uint k = (alpha_lo_v != 0ul)
               ? (uint)ctz(alpha_lo_v)
               : (64u + (uint)ctz(alpha_hi_v));
        gf128_mul_xk64(x_lo, x_hi, k, c_lo, c_hi);
    } else if (aw <= 64u) {
        gf128_mul_sparse_multiplier64(x_lo, x_hi, alpha_lo_v, alpha_hi_v, c_lo, c_hi);
    } else {
        gf128_mul64(alpha_lo_v, alpha_hi_v, x_lo, x_hi, c_lo, c_hi);
    }
}

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

    if (tower == 0u) {
        if (batch > 65536u && batch < 1048576u) {
            device const ulong2 *a2p = reinterpret_cast<device const ulong2 *>(a);
            device const ulong2 *b2p = reinterpret_cast<device const ulong2 *>(b);
            device       ulong2 *c2p = reinterpret_cast<device       ulong2 *>(c);

            ulong2 av = a2p[idx];
            ulong2 bv = b2p[idx];

            ulong c_lo, c_hi;
            gf128_mul64(av.x, av.y, bv.x, bv.y, c_lo, c_hi);

            c2p[idx] = ulong2(c_lo, c_hi);
        } else {
            device const uint *au = reinterpret_cast<device const uint *>(a);
            device const uint *bu = reinterpret_cast<device const uint *>(b);
            device       uint *cu = reinterpret_cast<device       uint *>(c);

            uint ubase = idx << 2u;

            uint a0 = au[ubase + 0u];
            uint a1 = au[ubase + 1u];
            uint a2 = au[ubase + 2u];
            uint a3 = au[ubase + 3u];

            uint b0 = bu[ubase + 0u];
            uint b1 = bu[ubase + 1u];
            uint b2 = bu[ubase + 2u];
            uint b3 = bu[ubase + 3u];

            uint r0, r1, r2, r3;
            gf128_mul_u32_limbs(a0, a1, a2, a3,
                                b0, b1, b2, b3,
                                r0, r1, r2, r3);

            cu[ubase + 0u] = r0;
            cu[ubase + 1u] = r1;
            cu[ubase + 2u] = r2;
            cu[ubase + 3u] = r3;
        }
        return;
    }

    uint base = idx << 2u;

    ulong a0_lo = a[base + 0u];
    ulong a0_hi = a[base + 1u];
    ulong a1_lo = a[base + 2u];
    ulong a1_hi = a[base + 3u];

    ulong b0_lo = b[base + 0u];
    ulong b0_hi = b[base + 1u];
    ulong b1_lo = b[base + 2u];
    ulong b1_hi = b[base + 3u];

    ulong m00_lo, m00_hi;
    gf128_mul64(a0_lo, a0_hi, b0_lo, b0_hi, m00_lo, m00_hi);

    ulong mix_lo, mix_hi;
    gf128_mul64(a0_lo ^ a1_lo, a0_hi ^ a1_hi,
                b0_lo ^ b1_lo, b0_hi ^ b1_hi,
                mix_lo, mix_hi);

    c[base + 2u] = mix_lo ^ m00_lo;
    c[base + 3u] = mix_hi ^ m00_hi;

    if (alpha_hi == 0ul && alpha_lo == 0ul) {
        c[base + 0u] = m00_lo;
        c[base + 1u] = m00_hi;
        return;
    }

    ulong m11_lo, m11_hi;
    gf128_mul64(a1_lo, a1_hi, b1_lo, b1_hi, m11_lo, m11_hi);

    ulong am_lo, am_hi;
    gf128_mul_alpha64(m11_lo, m11_hi, alpha_lo, alpha_hi, am_lo, am_hi);

    c[base + 0u] = m00_lo ^ am_lo;
    c[base + 1u] = m00_hi ^ am_hi;
}