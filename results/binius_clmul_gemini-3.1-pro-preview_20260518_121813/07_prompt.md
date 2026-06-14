## Task: binius_clmul

Batched binary-field carry-less multiplication, one product per thread. Two parameter sets are exercised; the runtime ``tower`` flag selects between them and the kernel MUST branch on the flag rather than baking either path in as a compile-time constant.

tower = 0 -- GF(2^128). Each element is two ``ulong`` limbs in little-endian polynomial order (limb 0 holds the coefficients of x^0..x^63; limb 1 holds x^64..x^127). The product is computed in GF(2)[x] (every '+' is XOR; there are zero integer multiplies) and reduced modulo the AES-GCM irreducible polynomial R(x) = x^128 + x^7 + x^2 + x + 1. The standard two-stage fold suffices: stage 1 folds the upper 128 bits into the lower 128 via the low pattern 1 + x + x^2 + x^7, leaving a residual at most 7 bits long; stage 2 folds the residual once more, after which the result has degree < 128.

tower = 1 -- GF(2^256) via the Fan-Hasan tower GF(2^128)[v] / (v^2 + v + alpha). Each element is four ``ulong`` limbs: limbs 0, 1 hold the v^0 coefficient a_0 in GF(2^128); limbs 2, 3 hold the v^1 coefficient a_1. With (a_0 + a_1 v) (b_0 + b_1 v) = c_0 + c_1 v and v^2 = v + alpha (the consequence of v^2 + v + alpha = 0 in characteristic 2),
    c_0 = a_0 b_0 + alpha * (a_1 b_1)
    c_1 = a_0 b_1 + a_1 b_0 + a_1 b_1
The ``alpha`` operand is supplied via the bound ``alpha_lo`` / ``alpha_hi`` scalars and is itself an element of GF(2^128).

Buffer layout: for ``field_words = 2 + 2 * tower`` limbs per element, ``a``, ``b``, ``c`` are flat ``ulong`` arrays of length ``batch * field_words``; element i occupies limbs ``[i * field_words .. i * field_words + field_words)``. Outputs are the raw 64-bit polynomial coefficient patterns; the host compares bit-exactly against a CPU GF(2^128) / tower reference.

## Required kernel signature(s)

```
kernel void binius_clmul(
    device const ulong *a         [[buffer(0)]],
    device const ulong *b         [[buffer(1)]],
    device       ulong *c         [[buffer(2)]],
    constant ulong     &alpha_lo  [[buffer(3)]],
    constant ulong     &alpha_hi  [[buffer(4)]],
    constant uint      &tower     [[buffer(5)]],
    constant uint      &batch     [[buffer(6)]],
    uint idx [[thread_position_in_grid]]);

Dispatch (host-fixed):
  threadsPerGrid        = (batch, 1, 1)
  threadsPerThreadgroup = (min(batch, 64), 1, 1)
Each thread processes ONE product end-to-end; guard against ``idx >= batch`` (the grid is rounded up to a multiple of the TG width). Threadgroup- or simdgroup-cooperative implementations are valid so long as the external buffer layout above and the canonical-output contract are preserved.
```

## Your previous attempt

```metal
#include <metal_stdlib>
using namespace metal;

// ----------------------------------------------------------------------
// Space 8 bits into 32 bits (1 bit every 4 bits)
// ----------------------------------------------------------------------
inline __attribute__((always_inline))
uint4 expand32(uint x) {
    uint4 y = uint4(x, x >> 8u, x >> 16u, x >> 24u) & 0xFFu;
    y = (y | (y << 12u)) & 0x000F000Fu;
    y = (y | (y <<  6u)) & 0x03030303u;
    y = (y | (y <<  3u)) & 0x11111111u;
    return y;
}

struct exp64_t { uint4 lo, hi; };
struct exp128_t { exp64_t lo, hi; };

inline __attribute__((always_inline))
exp64_t expand64(uint2 x) { return { expand32(x.x), expand32(x.y) }; }

inline __attribute__((always_inline))
exp128_t expand128(uint4 x) { return { expand64(x.xy), expand64(x.zw) }; }

// ----------------------------------------------------------------------
// Core multiplication primitives using uncompressed expanded arrays
// ----------------------------------------------------------------------
struct uncomp32_t { uint4 lo, hi; };
struct uncomp64_t { uint4 w0, w1, w2, w3; };
struct uncomp128_t { uint4 w0, w1, w2, w3, w4, w5, w6, w7; };

inline __attribute__((always_inline))
uncomp32_t mul32_uncompressed(uint4 A, uint4 B) {
    uint2 U0 = uint2(A.x * B.x, mulhi(A.x, B.x));
    uint2 U1 = uint2(A.x * B.y, mulhi(A.x, B.y)) ^ uint2(A.y * B.x, mulhi(A.y, B.x));
    uint2 U2 = uint2(A.x * B.z, mulhi(A.x, B.z)) ^ uint2(A.y * B.y, mulhi(A.y, B.y)) ^ uint2(A.z * B.x, mulhi(A.z, B.x));
    uint2 U3 = uint2(A.x * B.w, mulhi(A.x, B.w)) ^ uint2(A.y * B.z, mulhi(A.y, B.z)) ^ uint2(A.z * B.y, mulhi(A.z, B.y)) ^ uint2(A.w * B.x, mulhi(A.w, B.x));
    uint2 U4 = uint2(A.y * B.w, mulhi(A.y, B.w)) ^ uint2(A.z * B.z, mulhi(A.z, B.z)) ^ uint2(A.w * B.y, mulhi(A.w, B.y));
    uint2 U5 = uint2(A.z * B.w, mulhi(A.z, B.w)) ^ uint2(A.w * B.z, mulhi(A.w, B.z));
    uint2 U6 = uint2(A.w * B.w, mulhi(A.w, B.w));
    
    uncomp32_t res;
    res.lo = uint4(U0.x, U0.y ^ U1.x, U1.y ^ U2.x, U2.y ^ U3.x);
    res.hi = uint4(U3.y ^ U4.x, U4.y ^ U5.x, U5.y ^ U6.x, U6.y);
    return res;
}

inline __attribute__((always_inline))
uncomp64_t mul64_uncompressed(exp64_t A, exp64_t B) {
    uncomp32_t L = mul32_uncompressed(A.lo, B.lo);
    uncomp32_t H = mul32_uncompressed(A.hi, B.hi);
    uncomp32_t M = mul32_uncompressed(A.lo ^ A.hi, B.lo ^ B.hi);
    
    M.lo ^= L.lo ^ H.lo;
    M.hi ^= L.hi ^ H.hi;
    
    uncomp64_t res;
    res.w0 = L.lo;
    res.w1 = L.hi ^ M.lo;
    res.w2 = H.lo ^ M.hi;
    res.w3 = H.hi;
    return res;
}

inline __attribute__((always_inline))
uncomp128_t mul128_uncompressed(exp128_t A, exp128_t B) {
    uncomp64_t L = mul64_uncompressed(A.lo, B.lo);
    uncomp64_t H = mul64_uncompressed(A.hi, B.hi);
    
    exp64_t A_M = { A.lo.lo ^ A.hi.lo, A.lo.hi ^ A.hi.hi };
    exp64_t B_M = { B.lo.lo ^ B.hi.lo, B.lo.hi ^ B.hi.hi };
    uncomp64_t M = mul64_uncompressed(A_M, B_M);
    
    M.w0 ^= L.w0 ^ H.w0;
    M.w1 ^= L.w1 ^ H.w1;
    M.w2 ^= L.w2 ^ H.w2;
    M.w3 ^= L.w3 ^ H.w3;
    
    uncomp128_t res;
    res.w0 = L.w0;
    res.w1 = L.w1;
    res.w2 = L.w2 ^ M.w0;
    res.w3 = L.w3 ^ M.w1;
    res.w4 = H.w0 ^ M.w2;
    res.w5 = H.w1 ^ M.w3;
    res.w6 = H.w2;
    res.w7 = H.w3;
    return res;
}

// ----------------------------------------------------------------------
// Compress uncompressed bitwise sums back into bytes
// ----------------------------------------------------------------------
inline __attribute__((always_inline))
uint4 compress_uint4(uint4 W) {
    W &= 0x11111111u;
    W = (W | (W >> 3u)) & 0x03030303u;
    W = (W | (W >> 6u)) & 0x000F000Fu;
    W = (W | (W >> 12u)) & 0x000000FFu;
    return W;
}

inline __attribute__((always_inline))
uint2 compress_words(uint4 W_lo, uint4 W_hi) {
    uint4 c_lo = compress_uint4(W_lo);
    uint4 c_hi = compress_uint4(W_hi);
    
    uint res_x = c_lo.x | (c_lo.y << 8u) | (c_lo.z << 16u) | (c_lo.w << 24u);
    uint res_y = c_hi.x | (c_hi.y << 8u) | (c_hi.z << 16u) | (c_hi.w << 24u);
    return uint2(res_x, res_y);
}

inline __attribute__((always_inline))
void clmul128_unreduced(
    uint4 a, uint4 b,
    thread uint2 &t0, thread uint2 &t1,
    thread uint2 &t2, thread uint2 &t3)
{
    exp128_t A = expand128(a);
    exp128_t B = expand128(b);
    
    uncomp128_t U = mul128_uncompressed(A, B);
    
    t0 = compress_words(U.w0, U.w1);
    t1 = compress_words(U.w2, U.w3);
    t2 = compress_words(U.w4, U.w5);
    t3 = compress_words(U.w6, U.w7);
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

        uint2 t0, t1, t2, t3;
        clmul128_unreduced(as_type<uint4>(av), as_type<uint4>(bv), t0, t1, t2, t3);

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

        uint4 a0 = as_type<uint4>(av.xy);
        uint4 a1 = as_type<uint4>(av.zw);
        uint4 b0 = as_type<uint4>(bv.xy);
        uint4 b1 = as_type<uint4>(bv.zw);

        uint2 t0_00, t1_00, t2_00, t3_00;
        clmul128_unreduced(a0, b0, t0_00, t1_00, t2_00, t3_00);

        uint2 t0_11, t1_11, t2_11, t3_11;
        clmul128_unreduced(a1, b1, t0_11, t1_11, t2_11, t3_11);

        uint2 t0_sum, t1_sum, t2_sum, t3_sum;
        clmul128_unreduced(a0 ^ a1, b0 ^ b1, t0_sum, t1_sum, t2_sum, t3_sum);

        ulong m11_lo, m11_hi;
        gcm_reduce(as_type<ulong>(t0_11), as_type<ulong>(t1_11), 
                   as_type<ulong>(t2_11), as_type<ulong>(t3_11), 
                   m11_lo, m11_hi);

        uint4 alpha_val = uint4(as_type<uint2>(alpha_lo), as_type<uint2>(alpha_hi));
        uint4 m11_val = uint4(as_type<uint2>(m11_lo), as_type<uint2>(m11_hi));

        uint2 t0_am, t1_am, t2_am, t3_am;
        clmul128_unreduced(alpha_val, m11_val, t0_am, t1_am, t2_am, t3_am);

        ulong c0_lo, c0_hi;
        gcm_reduce(
            as_type<ulong>(t0_00 ^ t0_am),
            as_type<ulong>(t1_00 ^ t1_am),
            as_type<ulong>(t2_00 ^ t2_am),
            as_type<ulong>(t3_00 ^ t3_am),
            c0_lo, c0_hi);

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

Result of previous attempt:
        gf128_N64K: correct, 0.28 ms, 60.5 Gbitops/s (u64) (10.5% of 577 Gops/s (u64 bitop, est))
       gf128_N256K: correct, 0.29 ms, 234.8 Gbitops/s (u64) (40.7% of 577 Gops/s (u64 bitop, est))
         gf128_N1M: correct, 1.15 ms, 233.0 Gbitops/s (u64) (40.3% of 577 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.2580

## Current best (incumbent)

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
    uint idx [[thread_position_in_grid]])
{
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

        // Reduce m11 before multiplying by alpha
        ulong m11_lo, m11_hi;
        gcm_reduce(as_type<ulong>(t0_11), as_type<ulong>(t1_11), 
                   as_type<ulong>(t2_11), as_type<ulong>(t3_11), 
                   m11_lo, m11_hi);

        uint2 t0_am, t1_am, t2_am, t3_am;
        clmul128_unreduced(
            as_type<uint2>(alpha_lo), as_type<uint2>(alpha_hi), 
            as_type<uint2>(m11_lo),   as_type<uint2>(m11_hi), 
            t0_am, t1_am, t2_am, t3_am);

        // Delayed Reduction for C0 and C1 saves 1 whole reduction step!
        ulong c0_lo, c0_hi;
        gcm_reduce(
            as_type<ulong>(t0_00 ^ t0_am),
            as_type<ulong>(t1_00 ^ t1_am),
            as_type<ulong>(t2_00 ^ t2_am),
            as_type<ulong>(t3_00 ^ t3_am),
            c0_lo, c0_hi);

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

Incumbent result:
        gf128_N64K: correct, 0.24 ms, 69.5 Gbitops/s (u64) (12.0% of 577 Gops/s (u64 bitop, est))
       gf128_N256K: correct, 0.26 ms, 258.7 Gbitops/s (u64) (44.8% of 577 Gops/s (u64 bitop, est))
         gf128_N1M: correct, 1.00 ms, 268.9 Gbitops/s (u64) (46.6% of 577 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.2928

## History

- iter  0: compile=OK | correct=True | score=0.08524265746282575
- iter  1: compile=OK | correct=True | score=0.28809715771171646
- iter  2: compile=OK | correct=True | score=0.03708096677861069
- iter  3: compile=OK | correct=True | score=0.198292471298144
- iter  4: compile=OK | correct=True | score=0.29281315101745703
- iter  5: compile=OK | correct=True | score=0.21573500629857395
- iter  6: compile=OK | correct=True | score=0.2580485037872807

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
