1. **Replaced 64-bit emulation with purely native 32-bit GCM Reduction**: The previous `gcm_reduce` used `ulong` shifts which synthesized into expensive 64-bit emulation instructions on Apple's native 32-bit ALU. I introduced a pure `uint2`-based reduction leveraging exact overflow calculations (`mul_P` and `shift_P`) which perfectly bounds all intermediate degrees in 32-bit registers.
2. **Dynamic Small-Constant Fast-Paths**: In `tower == 1`, multiplying by the uniform constant `alpha` is a massive bottleneck. I added zero-divergence dispatch branches that check if `alpha` fits in 32-bits, 64-bits, or is a small integer (e.g. `1` or `2`). In the frequent case where `alpha == 2`, a full `clmul128` transforms into 4 cycle-cheap register shifts!
3. **Memory safety**: Removed aligned `ulong4` casting over buffers to guarantee no unaligned segment faults, using explicit scalar array dereferencing instead.

```metal
#include <metal_stdlib>
using namespace metal;

// ----------------------------------------------------------------------
// 16x16-bit carry-less multiply into 32-bit result
// Single accumulator avoids register spilling in deep inlined Karatsuba trees
// ----------------------------------------------------------------------
inline __attribute__((always_inline))
uint clmul16(uint a, uint b) {
    uint res = 0;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        res ^= select(0u, a << i, bool(b & (1u << i)));
    }
    return res;
}

// ----------------------------------------------------------------------
// 32x32-bit carry-less multiply
// Returns `uint2` to completely avoid 64-bit emulation on 32-bit ALUs
// ----------------------------------------------------------------------
inline __attribute__((always_inline))
uint2 clmul32(uint a, uint b) {
    uint a_lo = a & 0xFFFFu;
    uint a_hi = a >> 16u;
    uint b_lo = b & 0xFFFFu;
    uint b_hi = b >> 16u;
    
    uint L = clmul16(a_lo, b_lo);
    uint H = clmul16(a_hi, b_hi);
    uint M = clmul16(a_lo ^ a_hi, b_lo ^ b_hi) ^ L ^ H;
    
    uint res_lo = L ^ (M << 16u);
    uint res_hi = H ^ (M >> 16u);
    return uint2(res_lo, res_hi);
}

// ----------------------------------------------------------------------
// 64x64-bit carry-less multiply
// ----------------------------------------------------------------------
inline __attribute__((always_inline))
void clmul64(uint2 a, uint2 b, thread uint2 &r_lo, thread uint2 &r_hi) {
    uint2 L = clmul32(a.x, b.x);
    uint2 H = clmul32(a.y, b.y);
    uint2 M = clmul32(a.x ^ a.y, b.x ^ b.y);
    
    M ^= L ^ H;
    
    r_lo = uint2(L.x, L.y ^ M.x);
    r_hi = uint2(H.x ^ M.y, H.y);
}

// ----------------------------------------------------------------------
// 128x128-bit unreduced multiply
// ----------------------------------------------------------------------
inline __attribute__((always_inline))
void clmul128_unreduced(
    uint2 a_lo, uint2 a_hi, uint2 b_lo, uint2 b_hi,
    thread uint2 &t0, thread uint2 &t1,
    thread uint2 &t2, thread uint2 &t3)
{
    uint2 L_lo, L_hi;
    clmul64(a_lo, b_lo, L_lo, L_hi);
    
    uint2 H_lo, H_hi;
    clmul64(a_hi, b_hi, H_lo, H_hi);
    
    uint2 M_lo, M_hi;
    clmul64(a_lo ^ a_hi, b_lo ^ b_hi, M_lo, M_hi);
    
    M_lo ^= L_lo ^ H_lo;
    M_hi ^= L_hi ^ H_hi;
    
    t0 = L_lo;
    t1 = L_hi ^ M_lo;
    t2 = H_lo ^ M_hi;
    t3 = H_hi;
}

// ----------------------------------------------------------------------
// Optimized clmul128_unreduced when a_hi == 0
// ----------------------------------------------------------------------
inline __attribute__((always_inline))
void clmul128_unreduced_64(
    uint2 a_lo, uint2 b_lo, uint2 b_hi,
    thread uint2 &t0, thread uint2 &t1,
    thread uint2 &t2, thread uint2 &t3)
{
    uint2 L_lo, L_hi;
    clmul64(a_lo, b_lo, L_lo, L_hi);
    
    uint2 M_lo, M_hi;
    clmul64(a_lo, b_lo ^ b_hi, M_lo, M_hi);
    
    M_lo ^= L_lo;
    M_hi ^= L_hi;
    
    t0 = L_lo;
    t1 = L_hi ^ M_lo;
    t2 = M_hi;
    t3 = uint2(0);
}

// ----------------------------------------------------------------------
// Optimized clmul128_unreduced when a fits in 32 bits
// ----------------------------------------------------------------------
inline __attribute__((always_inline))
void clmul128_unreduced_32(
    uint a_lo, uint2 b_lo, uint2 b_hi,
    thread uint2 &t0, thread uint2 &t1,
    thread uint2 &t2, thread uint2 &t3)
{
    uint2 M0 = clmul32(a_lo, b_lo.x);
    uint2 M1 = clmul32(a_lo, b_lo.y);
    uint2 M2 = clmul32(a_lo, b_hi.x);
    uint2 M3 = clmul32(a_lo, b_hi.y);
    
    t0 = uint2(M0.x, M0.y ^ M1.x);
    t1 = uint2(M1.y ^ M2.x, M2.y ^ M3.x);
    t2 = uint2(M3.y, 0u);
    t3 = uint2(0u);
}

// ----------------------------------------------------------------------
// Pure 32-bit GCM reduction modulo R(x) = x^128 + x^7 + x^2 + x + 1
// ----------------------------------------------------------------------
inline __attribute__((always_inline))
uint mul_P(uint x) {
    return x ^ (x << 1u) ^ (x << 2u) ^ (x << 7u);
}

inline __attribute__((always_inline))
uint shift_P(uint x) {
    return (x >> 31u) ^ (x >> 30u) ^ (x >> 25u);
}

inline __attribute__((always_inline))
void gcm_reduce_uint2(
    uint2 t0, uint2 t1, uint2 t2, uint2 t3,
    thread uint2 &r_lo, thread uint2 &r_hi)
{
    uint2 d_lo0;
    d_lo0.x = mul_P(t2.x);
    d_lo0.y = mul_P(t2.y) ^ shift_P(t2.x);
            
    uint2 d_lo1;
    d_lo1.x = mul_P(t3.x) ^ shift_P(t2.y);
    d_lo1.y = mul_P(t3.y) ^ shift_P(t3.x);
            
    uint d_hi_u = shift_P(t3.y);
    uint d_hi_P = mul_P(d_hi_u);
    
    t0.x ^= d_lo0.x ^ d_hi_P;
    t0.y ^= d_lo0.y;
    
    t1.x ^= d_lo1.x;
    t1.y ^= d_lo1.y;
    
    r_lo = t0;
    r_hi = t1;
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
        uint i2 = idx * 2;
        uint2 a_lo = as_type<uint2>(a[i2]);
        uint2 a_hi = as_type<uint2>(a[i2 + 1]);
        uint2 b_lo = as_type<uint2>(b[i2]);
        uint2 b_hi = as_type<uint2>(b[i2 + 1]);

        uint2 t0, t1, t2, t3;
        clmul128_unreduced(a_lo, a_hi, b_lo, b_hi, t0, t1, t2, t3);

        uint2 c_lo, c_hi;
        gcm_reduce_uint2(t0, t1, t2, t3, c_lo, c_hi);

        c[i2]     = as_type<ulong>(c_lo);
        c[i2 + 1] = as_type<ulong>(c_hi);
    } else {
        uint i4 = idx * 4;
        uint2 a0_lo = as_type<uint2>(a[i4]);
        uint2 a0_hi = as_type<uint2>(a[i4 + 1]);
        uint2 a1_lo = as_type<uint2>(a[i4 + 2]);
        uint2 a1_hi = as_type<uint2>(a[i4 + 3]);
        
        uint2 b0_lo = as_type<uint2>(b[i4]);
        uint2 b0_hi = as_type<uint2>(b[i4 + 1]);
        uint2 b1_lo = as_type<uint2>(b[i4 + 2]);
        uint2 b1_hi = as_type<uint2>(b[i4 + 3]);

        uint2 t0_00, t1_00, t2_00, t3_00;
        clmul128_unreduced(a0_lo, a0_hi, b0_lo, b0_hi, t0_00, t1_00, t2_00, t3_00);

        uint2 t0_11, t1_11, t2_11, t3_11;
        clmul128_unreduced(a1_lo, a1_hi, b1_lo, b1_hi, t0_11, t1_11, t2_11, t3_11);

        uint2 t0_sum, t1_sum, t2_sum, t3_sum;
        clmul128_unreduced(
            a0_lo ^ a1_lo, a0_hi ^ a1_hi,
            b0_lo ^ b1_lo, b0_hi ^ b1_hi,
            t0_sum, t1_sum, t2_sum, t3_sum);

        uint2 m11_lo, m11_hi;
        gcm_reduce_uint2(t0_11, t1_11, t2_11, t3_11, m11_lo, m11_hi);

        uint2 t0_am, t1_am, t2_am, t3_am;
        uint2 alo = as_type<uint2>(alpha_lo);
        uint2 ahi = as_type<uint2>(alpha_hi);
        
        // Zero-divergence dynamic bypass using the uniform alpha
        if ((ahi.x | ahi.y | alo.y) == 0u) {
            if (alo.x == 2u) {
                // Multiplying by polynomial 'x' is just a shift by 1.
                t0_am = uint2(m11_lo.x << 1u, (m11_lo.y << 1u) | (m11_lo.x >> 31u));
                t1_am = uint2((m11_hi.x << 1u) | (m11_lo.y >> 31u), (m11_hi.y << 1u) | (m11_hi.x >> 31u));
                t2_am = uint2(m11_hi.y >> 31u, 0u);
                t3_am = uint2(0u);
            } else if (alo.x == 1u) {
                t0_am = m11_lo;
                t1_am = m11_hi;
                t2_am = uint2(0u);
                t3_am = uint2(0u);
            } else if (alo.x == 0u) {
                t0_am = uint2(0u);
                t1_am = uint2(0u);
                t2_am = uint2(0u);
                t3_am = uint2(0u);
            } else {
                clmul128_unreduced_32(alo.x, m11_lo, m11_hi, t0_am, t1_am, t2_am, t3_am);
            }
        } else if ((ahi.x | ahi.y) == 0u) {
            clmul128_unreduced_64(alo, m11_lo, m11_hi, t0_am, t1_am, t2_am, t3_am);
        } else {
            clmul128_unreduced(alo, ahi, m11_lo, m11_hi, t0_am, t1_am, t2_am, t3_am);
        }

        uint2 c0_lo, c0_hi;
        gcm_reduce_uint2(
            t0_00 ^ t0_am, t1_00 ^ t1_am,
            t2_00 ^ t2_am, t3_00 ^ t3_am,
            c0_lo, c0_hi);

        uint2 c1_lo, c1_hi;
        gcm_reduce_uint2(
            t0_sum ^ t0_00, t1_sum ^ t1_00,
            t2_sum ^ t2_00, t3_sum ^ t3_00,
            c1_lo, c1_hi);

        c[i4]     = as_type<ulong>(c0_lo);
        c[i4 + 1] = as_type<ulong>(c0_hi);
        c[i4 + 2] = as_type<ulong>(c1_lo);
        c[i4 + 3] = as_type<ulong>(c1_hi);
    }
}
```