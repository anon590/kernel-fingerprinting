#include <metal_stdlib>
using namespace metal;

// ----------------------------------------------------------------------
// 32x32 -> 64-bit carry-less multiply base case
// ----------------------------------------------------------------------
inline ulong clmul32(uint a, uint b) {
    ulong res = 0;
    ulong a_ul = a;
    #pragma unroll(32)
    for (uint i = 0; i < 32; i++) {
        ulong mask = 0ul - (ulong)((b >> i) & 1u);
        res ^= (a_ul << i) & mask;
    }
    return res;
}

// ----------------------------------------------------------------------
// 64x64 -> 128-bit carry-less multiply via Karatsuba
// ----------------------------------------------------------------------
inline void clmul64(ulong a, ulong b, thread ulong &res_lo, thread ulong &res_hi) {
    uint a_lo = (uint)a;
    uint a_hi = (uint)(a >> 32);
    uint b_lo = (uint)b;
    uint b_hi = (uint)(b >> 32);
    
    ulong z0 = clmul32(a_lo, b_lo);
    ulong z2 = clmul32(a_hi, b_hi);
    ulong z1 = clmul32(a_lo ^ a_hi, b_lo ^ b_hi) ^ z0 ^ z2;
    
    res_lo = z0 ^ (z1 << 32);
    res_hi = z2 ^ (z1 >> 32);
}

// ----------------------------------------------------------------------
// 128x128 -> 256-bit carry-less multiply via Karatsuba
// ----------------------------------------------------------------------
inline void clmul128_unreduced(
    ulong a_lo, ulong a_hi, ulong b_lo, ulong b_hi,
    thread ulong &t0, thread ulong &t1,
    thread ulong &t2, thread ulong &t3)
{
    ulong z0_lo, z0_hi; clmul64(a_lo, b_lo, z0_lo, z0_hi);
    ulong z2_lo, z2_hi; clmul64(a_hi, b_hi, z2_lo, z2_hi);
    ulong z1_lo, z1_hi; clmul64(a_lo ^ a_hi, b_lo ^ b_hi, z1_lo, z1_hi);
    
    z1_lo ^= z0_lo ^ z2_lo;
    z1_hi ^= z0_hi ^ z2_hi;
    
    t0 = z0_lo;
    t1 = z0_hi ^ z1_lo;
    t2 = z2_lo ^ z1_hi;
    t3 = z2_hi;
}

// ----------------------------------------------------------------------
// Helper shifts for AES-GCM irreducible polynomial reduction
// ----------------------------------------------------------------------
inline ulong fold_low(ulong x) {
    return x ^ (x << 1u) ^ (x << 2u) ^ (x << 7u);
}

inline ulong fold_high(ulong x) {
    return (x >> 63u) ^ (x >> 62u) ^ (x >> 57u);
}

// ----------------------------------------------------------------------
// Two-stage GCM-style reduction
// ----------------------------------------------------------------------
inline void gcm_reduce(
    ulong t0, ulong t1, ulong t2, ulong t3,
    thread ulong &r_lo, thread ulong &r_hi)
{
    ulong d_lo0 = fold_low(t2);
    ulong d_lo1 = fold_low(t3) ^ fold_high(t2);
    ulong d_hi  = fold_high(t3);
    
    t0 ^= d_lo0;
    t1 ^= d_lo1;
    t0 ^= fold_low(d_hi);
    
    r_lo = t0;
    r_hi = t1;
}

// ----------------------------------------------------------------------
// GF(2^128) full multiply (clmul + reduce)
// ----------------------------------------------------------------------
inline ulong2 gf128_mul(ulong2 a, ulong2 b) {
    ulong t0, t1, t2, t3;
    clmul128_unreduced(a[0], a[1], b[0], b[1], t0, t1, t2, t3);
    
    ulong c_lo, c_hi;
    gcm_reduce(t0, t1, t2, t3, c_lo, c_hi);
    return ulong2(c_lo, c_hi);
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
        // GF(2^128) using optimal vectorized loads and stores
        device const ulong2 *a2 = (device const ulong2 *)a;
        device const ulong2 *b2 = (device const ulong2 *)b;
        device       ulong2 *c2 = (device       ulong2 *)c;
        
        c2[idx] = gf128_mul(a2[idx], b2[idx]);
    } else {
        // GF(2^256) Fan-Hasan tower leveraging Karatsuba for multiplication reduction
        device const ulong4 *a4 = (device const ulong4 *)a;
        device const ulong4 *b4 = (device const ulong4 *)b;
        device       ulong4 *c4 = (device       ulong4 *)c;

        ulong4 a_val = a4[idx];
        ulong4 b_val = b4[idx];

        ulong2 a0 = a_val.xy;
        ulong2 a1 = a_val.zw;
        ulong2 b0 = b_val.xy;
        ulong2 b1 = b_val.zw;

        ulong2 m00 = gf128_mul(a0, b0);
        ulong2 m11 = gf128_mul(a1, b1);
        ulong2 m_sum = gf128_mul(a0 ^ a1, b0 ^ b1);

        ulong2 am = gf128_mul(ulong2(alpha_lo, alpha_hi), m11);

        ulong2 c0 = m00 ^ am;
        // In the Fan-Hasan tower v^2 = v + alpha, Karatsuba naturally
        // simplifies the v^1 coefficient to m_sum ^ m00.
        ulong2 c1 = m_sum ^ m00;

        c4[idx] = ulong4(c0, c1);
    }
}