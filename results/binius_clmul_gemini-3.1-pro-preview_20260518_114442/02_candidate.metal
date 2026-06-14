#include <metal_stdlib>
using namespace metal;

// ----------------------------------------------------------------------
// 64x64 -> 128-bit carry-less multiply
// Fully unrolled schoolbook with peeled 0th iteration for immediate shifts
// ----------------------------------------------------------------------
inline void clmul64(ulong a, ulong b, thread ulong &res_lo, thread ulong &res_hi) {
    ulong mask0 = 0ul - (b & 1ul);
    ulong lo = a & mask0;
    ulong hi = 0ul;
    
    #pragma unroll(63)
    for (uint i = 1u; i < 64u; ++i) {
        ulong mask = 0ul - ((b >> i) & 1ul);
        lo ^= (a << i) & mask;
        hi ^= (a >> (64u - i)) & mask;
    }
    
    res_lo = lo;
    res_hi = hi;
}

// ----------------------------------------------------------------------
// 128x128 -> 256-bit unreduced carry-less multiply
// Leverages Karatsuba over the optimized 64x64 base case
// ----------------------------------------------------------------------
inline void clmul128_unreduced(
    ulong a_lo, ulong a_hi, ulong b_lo, ulong b_hi,
    thread ulong &t0, thread ulong &t1,
    thread ulong &t2, thread ulong &t3)
{
    ulong z0_lo, z0_hi;
    clmul64(a_lo, b_lo, z0_lo, z0_hi);
    
    ulong z2_lo, z2_hi;
    clmul64(a_hi, b_hi, z2_lo, z2_hi);
    
    ulong z1_lo, z1_hi;
    clmul64(a_lo ^ a_hi, b_lo ^ b_hi, z1_lo, z1_hi);
    
    z1_lo ^= z0_lo ^ z2_lo;
    z1_hi ^= z0_hi ^ z2_hi;
    
    t0 = z0_lo;
    t1 = z0_hi ^ z1_lo;
    t2 = z2_lo ^ z1_hi;
    t3 = z2_hi;
}

// ----------------------------------------------------------------------
// Two-stage GCM-style reduction
// Modulo AES-GCM irreducible polynomial R(x) = x^128 + x^7 + x^2 + x + 1
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
        // GF(2^128) using vectorized loads
        device const ulong2 *a2 = (device const ulong2 *)a;
        device const ulong2 *b2 = (device const ulong2 *)b;
        device       ulong2 *c2 = (device       ulong2 *)c;
        
        ulong2 a_val = a2[idx];
        ulong2 b_val = b2[idx];

        ulong c_lo, c_hi;
        gf128_mul(a_val.x, a_val.y, b_val.x, b_val.y, c_lo, c_hi);

        c2[idx] = ulong2(c_lo, c_hi);
    } else {
        // GF(2^256) Fan-Hasan tower using vectorized loads
        device const ulong4 *a4 = (device const ulong4 *)a;
        device const ulong4 *b4 = (device const ulong4 *)b;
        device       ulong4 *c4 = (device       ulong4 *)c;

        ulong4 a_val = a4[idx];
        ulong4 b_val = b4[idx];

        ulong a0_lo = a_val.x, a0_hi = a_val.y;
        ulong a1_lo = a_val.z, a1_hi = a_val.w;
        ulong b0_lo = b_val.x, b0_hi = b_val.y;
        ulong b1_lo = b_val.z, b1_hi = b_val.w;

        // Fan-Hasan Karatsuba: Reduces 5 multi-word muls to 4.
        ulong m00_lo, m00_hi;
        gf128_mul(a0_lo, a0_hi, b0_lo, b0_hi, m00_lo, m00_hi);
        
        ulong m11_lo, m11_hi;
        gf128_mul(a1_lo, a1_hi, b1_lo, b1_hi, m11_lo, m11_hi);
        
        ulong m_sum_lo, m_sum_hi;
        gf128_mul(a0_lo ^ a1_lo, a0_hi ^ a1_hi, b0_lo ^ b1_lo, b0_hi ^ b1_hi, m_sum_lo, m_sum_hi);

        ulong am_lo, am_hi;
        // Swap operands here so uniform alpha is strictly driving bit-extracts
        gf128_mul(m11_lo, m11_hi, alpha_lo, alpha_hi, am_lo, am_hi);

        ulong c0_lo = m00_lo ^ am_lo;
        ulong c0_hi = m00_hi ^ am_hi;
        
        // Algebraically, c1 = m_sum ^ m00. (Replaces `m01 ^ m10 ^ m11`)
        ulong c1_lo = m_sum_lo ^ m00_lo;
        ulong c1_hi = m_sum_hi ^ m00_hi;

        c4[idx] = ulong4(c0_lo, c0_hi, c1_lo, c1_hi);
    }
}