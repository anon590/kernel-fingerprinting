Replaced the 32-bit Karatsuba-based `clmul32` and `clmul16` primitives with a natively 64-bit fully unrolled `clmul64` using `ulong`. Apple Silicon GPUs possess fully pipelined 64-bit ALUs; processing 64-bit limbs natively halves the total loop iterations and completely eliminates the register pressure and instructions wasted on recursive Karatsuba splits. To maximize Instruction-Level Parallelism (ILP), the 63-iteration unrolled loop interleaves its conditional additions across 4 independent `ulong` accumulators, breaking the XOR dependency chain and saturating the execution ports. Finally, in the `tower=1` branch, passing the uniform `alpha` parameter as the right-hand operand to `clmul128_unreduced` guarantees that the bit-test conditions for the carry-less multiply are uniform across the simdgroup, minimizing warp divergence.

```metal
#include <metal_stdlib>
using namespace metal;

// ----------------------------------------------------------------------
// 64x64-bit carry-less multiply
// Native 64-bit fully unrolled branchless loop with 4-way ILP accumulation.
// ----------------------------------------------------------------------
inline __attribute__((always_inline))
void clmul64(ulong a, ulong b, thread ulong &r_lo, thread ulong &r_hi) {
    ulong r0_lo = select(0ul, a, (b & 1ul) != 0), r0_hi = 0;
    ulong r1_lo = 0, r1_hi = 0;
    ulong r2_lo = 0, r2_hi = 0;
    ulong r3_lo = 0, r3_hi = 0;

    #pragma unroll
    for (int i = 1; i < 64; i++) {
        bool bit = (b & (1ul << i)) != 0;
        ulong m_lo = select(0ul, a << i, bit);
        ulong m_hi = select(0ul, a >> (64 - i), bit);
        
        if (i % 4 == 0) {
            r0_lo ^= m_lo; r0_hi ^= m_hi;
        } else if (i % 4 == 1) {
            r1_lo ^= m_lo; r1_hi ^= m_hi;
        } else if (i % 4 == 2) {
            r2_lo ^= m_lo; r2_hi ^= m_hi;
        } else {
            r3_lo ^= m_lo; r3_hi ^= m_hi;
        }
    }
    
    r_lo = r0_lo ^ r1_lo ^ r2_lo ^ r3_lo;
    r_hi = r0_hi ^ r1_hi ^ r2_hi ^ r3_hi;
}

// ----------------------------------------------------------------------
// 128x128-bit unreduced multiply
// ----------------------------------------------------------------------
inline __attribute__((always_inline))
void clmul128_unreduced(
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
inline __attribute__((always_inline))
void gcm_reduce(
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
        device const ulong2 *a2 = (device const ulong2 *)a;
        device const ulong2 *b2 = (device const ulong2 *)b;
        device       ulong2 *c2 = (device       ulong2 *)c;
        
        ulong2 av = a2[idx];
        ulong2 bv = b2[idx];

        ulong t0, t1, t2, t3;
        clmul128_unreduced(av.x, av.y, bv.x, bv.y, t0, t1, t2, t3);

        ulong c_lo, c_hi;
        gcm_reduce(t0, t1, t2, t3, c_lo, c_hi);

        c2[idx] = ulong2(c_lo, c_hi);
    } else {
        device const ulong4 *a4 = (device const ulong4 *)a;
        device const ulong4 *b4 = (device const ulong4 *)b;
        device       ulong4 *c4 = (device       ulong4 *)c;
        
        ulong4 av = a4[idx];
        ulong4 bv = b4[idx];

        ulong a0_lo = av.x, a0_hi = av.y;
        ulong a1_lo = av.z, a1_hi = av.w;
        ulong b0_lo = bv.x, b0_hi = bv.y;
        ulong b1_lo = bv.z, b1_hi = bv.w;

        ulong t0_00, t1_00, t2_00, t3_00;
        clmul128_unreduced(a0_lo, a0_hi, b0_lo, b0_hi, t0_00, t1_00, t2_00, t3_00);

        ulong t0_11, t1_11, t2_11, t3_11;
        clmul128_unreduced(a1_lo, a1_hi, b1_lo, b1_hi, t0_11, t1_11, t2_11, t3_11);

        ulong t0_sum, t1_sum, t2_sum, t3_sum;
        clmul128_unreduced(
            a0_lo ^ a1_lo, a0_hi ^ a1_hi,
            b0_lo ^ b1_lo, b0_hi ^ b1_hi,
            t0_sum, t1_sum, t2_sum, t3_sum);

        ulong m11_lo, m11_hi;
        gcm_reduce(t0_11, t1_11, t2_11, t3_11, m11_lo, m11_hi);

        ulong t0_am, t1_am, t2_am, t3_am;
        // alpha_lo / alpha_hi passed as right-hand operand 'b' to optimize uniform bit tests
        clmul128_unreduced(
            m11_lo,   m11_hi, 
            alpha_lo, alpha_hi, 
            t0_am, t1_am, t2_am, t3_am);

        ulong c0_lo, c0_hi;
        gcm_reduce(
            t0_00 ^ t0_am,
            t1_00 ^ t1_am,
            t2_00 ^ t2_am,
            t3_00 ^ t3_am,
            c0_lo, c0_hi);

        ulong c1_lo, c1_hi;
        gcm_reduce(
            t0_sum ^ t0_00,
            t1_sum ^ t1_00,
            t2_sum ^ t2_00,
            t3_sum ^ t3_00,
            c1_lo, c1_hi);

        c4[idx] = ulong4(c0_lo, c0_hi, c1_lo, c1_hi);
    }
}
```