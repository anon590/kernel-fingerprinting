#include <metal_stdlib>
using namespace metal;

// ----------------------------------------------------------------------
// 16x16-bit carry-less multiply into 32-bit result
// Uses native 32-bit operations heavily optimized for Apple GPUs.
// ----------------------------------------------------------------------
inline uint clmul16(uint a, uint b) {
    uint res = 0;
    res ^= select(0u, a << 0,  bool(b & (1u << 0)));
    res ^= select(0u, a << 1,  bool(b & (1u << 1)));
    res ^= select(0u, a << 2,  bool(b & (1u << 2)));
    res ^= select(0u, a << 3,  bool(b & (1u << 3)));
    res ^= select(0u, a << 4,  bool(b & (1u << 4)));
    res ^= select(0u, a << 5,  bool(b & (1u << 5)));
    res ^= select(0u, a << 6,  bool(b & (1u << 6)));
    res ^= select(0u, a << 7,  bool(b & (1u << 7)));
    res ^= select(0u, a << 8,  bool(b & (1u << 8)));
    res ^= select(0u, a << 9,  bool(b & (1u << 9)));
    res ^= select(0u, a << 10, bool(b & (1u << 10)));
    res ^= select(0u, a << 11, bool(b & (1u << 11)));
    res ^= select(0u, a << 12, bool(b & (1u << 12)));
    res ^= select(0u, a << 13, bool(b & (1u << 13)));
    res ^= select(0u, a << 14, bool(b & (1u << 14)));
    res ^= select(0u, a << 15, bool(b & (1u << 15)));
    return res;
}

// ----------------------------------------------------------------------
// 32x32-bit carry-less multiply via Karatsuba
// ----------------------------------------------------------------------
inline ulong clmul32(uint a, uint b) {
    uint a_lo = a & 0xFFFFu;
    uint a_hi = a >> 16;
    uint b_lo = b & 0xFFFFu;
    uint b_hi = b >> 16;
    
    uint L = clmul16(a_lo, b_lo);
    uint H = clmul16(a_hi, b_hi);
    uint M = clmul16(a_lo ^ a_hi, b_lo ^ b_hi) ^ L ^ H;
    
    return (ulong)L ^ ((ulong)M << 16) ^ ((ulong)H << 32);
}

// ----------------------------------------------------------------------
// 64x64-bit carry-less multiply via Karatsuba
// ----------------------------------------------------------------------
inline void clmul64(ulong a, ulong b, thread ulong &r_lo, thread ulong &r_hi) {
    uint a_lo = (uint)a;
    uint a_hi = (uint)(a >> 32);
    uint b_lo = (uint)b;
    uint b_hi = (uint)(b >> 32);
    
    ulong L = clmul32(a_lo, b_lo);
    ulong H = clmul32(a_hi, b_hi);
    ulong M = clmul32(a_lo ^ a_hi, b_lo ^ b_hi) ^ L ^ H;
    
    r_lo = L ^ (M << 32);
    r_hi = H ^ (M >> 32);
}

// ----------------------------------------------------------------------
// 128x128-bit unreduced multiply via Karatsuba
// ----------------------------------------------------------------------
inline void clmul128_unreduced(
    ulong a_lo, ulong a_hi, ulong b_lo, ulong b_hi,
    thread ulong &t0, thread ulong &t1,
    thread ulong &t2, thread ulong &t3)
{
    ulong L_lo, L_hi;
    clmul64(a_lo, b_lo, L_lo, L_hi);
    
    ulong H_lo, H_hi;
    clmul64(a_hi, b_hi, H_lo, H_hi);
    
    ulong M_lo, M_hi;
    clmul64(a_lo ^ a_hi, b_lo ^ b_hi, M_lo, M_hi);
    
    M_lo ^= L_lo ^ H_lo;
    M_hi ^= L_hi ^ H_hi;
    
    t0 = L_lo;
    t1 = L_hi ^ M_lo;
    t2 = H_lo ^ M_hi;
    t3 = H_hi;
}

// ----------------------------------------------------------------------
// Two-stage GCM-style reduction modulo R(x) = x^128 + x^7 + x^2 + x + 1
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

// ----------------------------------------------------------------------
// GF(2^128) full multiply (clmul + reduce)
// ----------------------------------------------------------------------
inline void gf128_mul(
    ulong a_lo, ulong a_hi, ulong b_lo, ulong b_hi,
    thread ulong &c_lo, thread ulong &c_hi)
{
    ulong t0, t1, t2, t3;
    clmul128_unreduced(a_lo, a_hi, b_lo, b_hi, t0, t1, t2, t3);
    gcm_reduce(t0, t1, t2, t3, c_lo, c_hi);
}

// ----------------------------------------------------------------------
// Kernel
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
        // GF(2^128) vectorized loads
        device const ulong2 *a2 = (device const ulong2 *)a;
        device const ulong2 *b2 = (device const ulong2 *)b;
        device       ulong2 *c2 = (device       ulong2 *)c;
        
        ulong2 av = a2[idx];
        ulong2 bv = b2[idx];

        ulong c_lo, c_hi;
        gf128_mul(av.x, av.y, bv.x, bv.y, c_lo, c_hi);

        c2[idx] = ulong2(c_lo, c_hi);
    } else {
        // GF(2^256) vectorized loads
        device const ulong4 *a4 = (device const ulong4 *)a;
        device const ulong4 *b4 = (device const ulong4 *)b;
        device       ulong4 *c4 = (device       ulong4 *)c;
        
        ulong4 av = a4[idx];
        ulong4 bv = b4[idx];

        ulong a0_lo = av.x, a0_hi = av.y;
        ulong a1_lo = av.z, a1_hi = av.w;
        ulong b0_lo = bv.x, b0_hi = bv.y;
        ulong b1_lo = bv.z, b1_hi = bv.w;

        // Optimized Fan-Hasan tower computation (4 products instead of 5)
        ulong m00_lo, m00_hi; gf128_mul(a0_lo, a0_hi, b0_lo, b0_hi, m00_lo, m00_hi);
        ulong m11_lo, m11_hi; gf128_mul(a1_lo, a1_hi, b1_lo, b1_hi, m11_lo, m11_hi);
        ulong m_sum_lo, m_sum_hi; gf128_mul(a0_lo ^ a1_lo, a0_hi ^ a1_hi, b0_lo ^ b1_lo, b0_hi ^ b1_hi, m_sum_lo, m_sum_hi);

        ulong am_lo, am_hi;
        gf128_mul(alpha_lo, alpha_hi, m11_lo, m11_hi, am_lo, am_hi);

        ulong c0_lo = m00_lo ^ am_lo;
        ulong c0_hi = m00_hi ^ am_hi;
        ulong c1_lo = m_sum_lo ^ m00_lo;
        ulong c1_hi = m_sum_hi ^ m00_hi;

        c4[idx] = ulong4(c0_lo, c0_hi, c1_lo, c1_hi);
    }
}