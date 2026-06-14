To drastically optimize the computation for GF(2^256) (`tower == 1`), we can utilize threadgroup-level cooperation to compute a shared lookup table for `alpha * x^k mod R(x)`. Because `alpha` is constant across all threads, one thread per threadgroup can iteratively generate this 128-entry table (taking just ~500 cycles). Then, all threads replace their expensive 128x128 carry-less multiplication and degree reduction step for `alpha * m11` with fast table lookups. 

This optimization skips one whole `clmul128` operation and one `gcm_reduce` per thread!

A critical detail: the guard `if (idx >= batch) return;` **must** be placed *after* `threadgroup_barrier` so threads that are out-of-bounds still participate in the sync step, avoiding a GPU hang.

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
    uint a_hi = a >> 16;
    uint b_lo = b & 0xFFFFu;
    uint b_hi = b >> 16;
    
    uint L = clmul16(a_lo, b_lo);
    uint H = clmul16(a_hi, b_hi);
    uint M = clmul16(a_lo ^ a_hi, b_lo ^ b_hi) ^ L ^ H;
    
    // Exact 32-bit boundary alignment avoids all cross-word shifting
    uint res_lo = L ^ (M << 16);
    uint res_hi = H ^ (M >> 16);
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
    uint idx [[thread_position_in_grid]],
    uint tg_tid [[thread_index_in_threadgroup]])
{
    // Precompute `alpha * x^k mod R(x)` cooperatively in threadgroup memory.
    // By amortizing the generation of this 128-entry table, we replace a heavy
    // 128x128 clmul AND a full `gcm_reduce` per thread with 128 fast lookups!
    threadgroup ulong2 alpha_table[128];
    if (tower == 1u) {
        if (tg_tid == 0) {
            ulong2 alpha_val = ulong2(alpha_lo, alpha_hi);
            #pragma unroll(1)
            for (int k = 0; k < 128; k++) {
                alpha_table[k] = alpha_val;
                ulong msb = alpha_val.y >> 63;
                alpha_val.y = (alpha_val.y << 1) | (alpha_val.x >> 63);
                alpha_val.x = (alpha_val.x << 1);
                if (msb) {
                    alpha_val.x ^= 0x87ul;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Safely check OOB AFTER the barrier; missing the barrier would hang the GPU.
    if (idx >= batch) return;

    if (tower == 0u) {
        device const ulong2 *a2 = (device const ulong2 *)a;
        device const ulong2 *b2 = (device const ulong2 *)b;
        device       ulong2 *c2 = (device       ulong2 *)c;
        
        ulong2 av = a2[idx];
        ulong2 bv = b2[idx];

        uint2 a_lo = as_type<uint2>(av.x), a_hi = as_type<uint2>(av.y);
        uint2 b_lo = as_type<uint2>(bv.x), b_hi = as_type<uint2>(bv.y);

        uint2 t0, t1, t2, t3;
        clmul128_unreduced(a_lo, a_hi, b_lo, b_hi, t0, t1, t2, t3);

        ulong c_lo, c_hi;
        gcm_reduce(as_type<ulong>(t0), as_type<ulong>(t1), 
                   as_type<ulong>(t2), as_type<ulong>(t3), 
                   c_lo, c_hi);

        c2[idx] = ulong2(c_lo, c_hi);
    } else {
        device const ulong4 *a4 = (device const ulong4 *)a;
        device const ulong4 *b4 = (device const ulong4 *)b;
        device       ulong4 *c4 = (device       ulong4 *)c;
        
        ulong4 av = a4[idx];
        ulong4 bv = b4[idx];

        uint2 a0_lo = as_type<uint2>(av.x), a0_hi = as_type<uint2>(av.y);
        uint2 a1_lo = as_type<uint2>(av.z), a1_hi = as_type<uint2>(av.w);
        uint2 b0_lo = as_type<uint2>(bv.x), b0_hi = as_type<uint2>(bv.y);
        uint2 b1_lo = as_type<uint2>(bv.z), b1_hi = as_type<uint2>(bv.w);

        uint2 t0_00, t1_00, t2_00, t3_00;
        clmul128_unreduced(a0_lo, a0_hi, b0_lo, b0_hi, t0_00, t1_00, t2_00, t3_00);

        uint2 t0_11, t1_11, t2_11, t3_11;
        clmul128_unreduced(a1_lo, a1_hi, b1_lo, b1_hi, t0_11, t1_11, t2_11, t3_11);

        uint2 t0_sum, t1_sum, t2_sum, t3_sum;
        clmul128_unreduced(
            a0_lo ^ a1_lo, a0_hi ^ a1_hi,
            b0_lo ^ b1_lo, b0_hi ^ b1_hi,
            t0_sum, t1_sum, t2_sum, t3_sum);

        ulong m11_lo, m11_hi;
        gcm_reduce(as_type<ulong>(t0_11), as_type<ulong>(t1_11), 
                   as_type<ulong>(t2_11), as_type<ulong>(t3_11), 
                   m11_lo, m11_hi);

        ulong2 t_am = 0;
        #pragma unroll(8)
        for (int k = 0; k < 64; k++) {
            t_am ^= select(ulong2(0), alpha_table[k], bool((m11_lo >> k) & 1ul));
            t_am ^= select(ulong2(0), alpha_table[k + 64], bool((m11_hi >> k) & 1ul));
        }

        ulong c0_lo, c0_hi;
        gcm_reduce(
            as_type<ulong>(t0_00),
            as_type<ulong>(t1_00),
            as_type<ulong>(t2_00),
            as_type<ulong>(t3_00),
            c0_lo, c0_hi);

        // t_am is ALREADY reduced modulo R(x) via the precomputed table!
        c0_lo ^= t_am.x;
        c0_hi ^= t_am.y;

        ulong c1_lo, c1_hi;
        gcm_reduce(
            as_type<ulong>(t0_sum ^ t0_00),
            as_type<ulong>(t1_sum ^ t1_00),
            as_type<ulong>(t2_sum ^ t2_00),
            as_type<ulong>(t3_sum ^ t3_00),
            c1_lo, c1_hi);

        c4[idx] = ulong4(c0_lo, c0_hi, c1_lo, c1_hi);
    }
}
```