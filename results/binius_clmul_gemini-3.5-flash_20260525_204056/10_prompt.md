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

inline ulong fold_low(ulong x) {
    return x ^ (x << 1u) ^ (x << 2u) ^ (x << 7u);
}

inline ulong fold_high(ulong x) {
    return (x >> 63u) ^ (x >> 62u) ^ (x >> 57u);
}

inline void gcm_reduce(
    ulong t0, ulong t1, ulong t2, ulong t3,
    thread ulong &r_lo, thread ulong &r_hi)
{
    r_lo = t0 ^ fold_low(t2 ^ fold_high(t3));
    r_hi = t1 ^ fold_low(t3) ^ fold_high(t2);
}

inline uint clmul16(uint A, uint B)
{
    uint accum = 0;
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        accum ^= (B & (1u << j)) ? (A << j) : 0u;
    }
    return accum;
}

inline ulong clmul32_via_16(uint A, uint B)
{
    uint a_lo = A & 0xFFFFu;
    uint a_hi = A >> 16;
    uint b_lo = B & 0xFFFFu;
    uint b_hi = B >> 16;

    uint low = clmul16(a_lo, b_lo);
    uint high = clmul16(a_hi, b_hi);
    uint mid = clmul16(a_lo ^ a_hi, b_lo ^ b_hi);

    mid ^= low ^ high;

    return (ulong)low ^ ((ulong)mid << 16) ^ ((ulong)high << 32);
}

inline ulong2 clmul64(ulong A, ulong B)
{
    uint a_lo = (uint)A;
    uint a_hi = (uint)(A >> 32);
    uint b_lo = (uint)B;
    uint b_hi = (uint)(B >> 32);

    ulong low = clmul32_via_16(a_lo, b_lo);
    ulong high = clmul32_via_16(a_hi, b_hi);
    ulong mid = clmul32_via_16(a_lo ^ a_hi, b_lo ^ b_hi);

    mid ^= low ^ high;

    ulong r_lo = low ^ (mid << 32);
    ulong r_hi = high ^ (mid >> 32);
    return ulong2(r_lo, r_hi);
}

inline void clmul128_unreduced(
    ulong a_lo, ulong a_hi, ulong b_lo, ulong b_hi,
    thread ulong &t0, thread ulong &t1,
    thread ulong &t2, thread ulong &t3)
{
    ulong2 low = clmul64(a_lo, b_lo);
    ulong2 high = clmul64(a_hi, b_hi);
    ulong2 mid = clmul64(a_lo ^ a_hi, b_lo ^ b_hi);

    mid ^= low ^ high;

    t0 = low.x;
    t1 = low.y ^ mid.x;
    t2 = high.x ^ mid.y;
    t3 = high.y;
}

inline void gf128_mul(
    ulong a_lo, ulong a_hi, ulong b_lo, ulong b_hi,
    thread ulong &c_lo, thread ulong &c_hi)
{
    ulong t0, t1, t2, t3;
    clmul128_unreduced(a_lo, a_hi, b_lo, b_hi, t0, t1, t2, t3);
    gcm_reduce(t0, t1, t2, t3, c_lo, c_hi);
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
    if (idx >= batch) return;

    if (tower == 0u) {
        device const ulong2 *a2 = (device const ulong2 *)a;
        device const ulong2 *b2 = (device const ulong2 *)b;
        device       ulong2 *c2 = (device       ulong2 *)c;

        ulong2 va = a2[idx];
        ulong2 vb = b2[idx];

        ulong c_lo, c_hi;
        gf128_mul(va.x, va.y, vb.x, vb.y, c_lo, c_hi);

        c2[idx] = ulong2(c_lo, c_hi);
    } else {
        device const ulong4 *a4 = (device const ulong4 *)a;
        device const ulong4 *b4 = (device const ulong4 *)b;
        device       ulong4 *c4 = (device       ulong4 *)c;

        ulong4 va = a4[idx];
        ulong4 vb = b4[idx];

        ulong a0_lo = va.x, a0_hi = va.y;
        ulong a1_lo = va.z, a1_hi = va.w;
        ulong b0_lo = vb.x, b0_hi = vb.y;
        ulong b1_lo = vb.z, b1_hi = vb.w;

        // Unreduced 256-bit multiplications
        ulong t00_0, t00_1, t00_2, t00_3;
        clmul128_unreduced(a0_lo, a0_hi, b0_lo, b0_hi, t00_0, t00_1, t00_2, t00_3);

        ulong t11_0, t11_1, t11_2, t11_3;
        clmul128_unreduced(a1_lo, a1_hi, b1_lo, b1_hi, t11_0, t11_1, t11_2, t11_3);

        ulong tadd_0, tadd_1, tadd_2, tadd_3;
        clmul128_unreduced(a0_lo ^ a1_lo, a0_hi ^ a1_hi, b0_lo ^ b1_lo, b0_hi ^ b1_hi, tadd_0, tadd_1, tadd_2, tadd_3);

        // We only reduce t11 first to multiply by alpha
        ulong m11_lo, m11_hi;
        gcm_reduce(t11_0, t11_1, t11_2, t11_3, m11_lo, m11_hi);

        // Multiply alpha by m11
        ulong tal_0, tal_1, tal_2, tal_3;
        clmul128_unreduced(alpha_lo, alpha_hi, m11_lo, m11_hi, tal_0, tal_1, tal_2, tal_3);

        // c0 = reduce(t00 ^ tal)
        ulong c0_lo, c0_hi;
        gcm_reduce(t00_0 ^ tal_0, t00_1 ^ tal_1, t00_2 ^ tal_2, t00_3 ^ tal_3, c0_lo, c0_hi);

        // c1 = reduce(tadd ^ t00)
        ulong c1_lo, c1_hi;
        gcm_reduce(tadd_0 ^ t00_0, tadd_1 ^ t00_1, tadd_2 ^ t00_2, tadd_3 ^ t00_3, c1_lo, c1_hi);

        c4[idx] = ulong4(c0_lo, c0_hi, c1_lo, c1_hi);
    }
}
```

Result of previous attempt:
        gf128_N64K: correct, 0.27 ms, 63.0 Gbitops/s (u64) (10.9% of 577 Gops/s (u64 bitop, est))
       gf128_N256K: correct, 0.89 ms, 75.3 Gbitops/s (u64) (13.0% of 577 Gops/s (u64 bitop, est))
         gf128_N1M: correct, 1.37 ms, 196.1 Gbitops/s (u64) (34.0% of 577 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.1690

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

inline ulong fold_low(ulong x) {
    return x ^ (x << 1u) ^ (x << 2u) ^ (x << 7u);
}

inline ulong fold_high(ulong x) {
    return (x >> 63u) ^ (x >> 62u) ^ (x >> 57u);
}

inline void gcm_reduce(
    ulong t0, ulong t1, ulong t2, ulong t3,
    thread ulong &r_lo, thread ulong &r_hi)
{
    r_lo = t0 ^ fold_low(t2) ^ fold_low(fold_high(t3));
    r_hi = t1 ^ fold_low(t3) ^ fold_high(t2);
}

inline ulong clmul32(uint A, uint B)
{
    ulong accum = 0;
    ulong a = A;
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        accum ^= ((B & (1u << j)) ? (a << j) : 0ul);
    }
    return accum;
}

inline ulong2 clmul64(ulong A, ulong B)
{
    uint a_lo = (uint)A;
    uint a_hi = (uint)(A >> 32);
    uint b_lo = (uint)B;
    uint b_hi = (uint)(B >> 32);

    ulong low = clmul32(a_lo, b_lo);
    ulong high = clmul32(a_hi, b_hi);
    ulong mid = clmul32(a_lo ^ a_hi, b_lo ^ b_hi);

    mid ^= low ^ high;

    ulong r_lo = low ^ (mid << 32);
    ulong r_hi = high ^ (mid >> 32);
    return ulong2(r_lo, r_hi);
}

inline void clmul128_unreduced(
    ulong a_lo, ulong a_hi, ulong b_lo, ulong b_hi,
    thread ulong &t0, thread ulong &t1,
    thread ulong &t2, thread ulong &t3)
{
    ulong2 low = clmul64(a_lo, b_lo);
    ulong2 high = clmul64(a_hi, b_hi);
    ulong2 mid = clmul64(a_lo ^ a_hi, b_lo ^ b_hi);

    mid ^= low ^ high;

    t0 = low.x;
    t1 = low.y ^ mid.x;
    t2 = high.x ^ mid.y;
    t3 = high.y;
}

inline void gf128_mul(
    ulong a_lo, ulong a_hi, ulong b_lo, ulong b_hi,
    thread ulong &c_lo, thread ulong &c_hi)
{
    ulong t0, t1, t2, t3;
    clmul128_unreduced(a_lo, a_hi, b_lo, b_hi, t0, t1, t2, t3);
    gcm_reduce(t0, t1, t2, t3, c_lo, c_hi);
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
    if (idx >= batch) return;

    if (tower == 0u) {
        device const ulong2 *a2 = (device const ulong2 *)a;
        device const ulong2 *b2 = (device const ulong2 *)b;
        device       ulong2 *c2 = (device       ulong2 *)c;

        ulong2 va = a2[idx];
        ulong2 vb = b2[idx];

        ulong c_lo, c_hi;
        gf128_mul(va.x, va.y, vb.x, vb.y, c_lo, c_hi);

        c2[idx] = ulong2(c_lo, c_hi);
    } else {
        device const ulong4 *a4 = (device const ulong4 *)a;
        device const ulong4 *b4 = (device const ulong4 *)b;
        device       ulong4 *c4 = (device       ulong4 *)c;

        ulong4 va = a4[idx];
        ulong4 vb = b4[idx];

        ulong a0_lo = va.x, a0_hi = va.y;
        ulong a1_lo = va.z, a1_hi = va.w;
        ulong b0_lo = vb.x, b0_hi = vb.y;
        ulong b1_lo = vb.z, b1_hi = vb.w;

        ulong m00_lo, m00_hi;
        gf128_mul(a0_lo, a0_hi, b0_lo, b0_hi, m00_lo, m00_hi);

        ulong m11_lo, m11_hi;
        gf128_mul(a1_lo, a1_hi, b1_lo, b1_hi, m11_lo, m11_hi);

        ulong madd_lo, madd_hi;
        gf128_mul(a0_lo ^ a1_lo, a0_hi ^ a1_hi, b0_lo ^ b1_lo, b0_hi ^ b1_hi, madd_lo, madd_hi);

        ulong am_lo, am_hi;
        gf128_mul(alpha_lo, alpha_hi, m11_lo, m11_hi, am_lo, am_hi);

        ulong c0_lo = m00_lo ^ am_lo;
        ulong c0_hi = m00_hi ^ am_hi;
        ulong c1_lo = madd_lo ^ m00_lo;
        ulong c1_hi = madd_hi ^ m00_hi;

        c4[idx] = ulong4(c0_lo, c0_hi, c1_lo, c1_hi);
    }
}
```

Incumbent result:
        gf128_N64K: correct, 0.18 ms, 93.3 Gbitops/s (u64) (16.2% of 577 Gops/s (u64 bitop, est))
       gf128_N256K: correct, 0.35 ms, 191.1 Gbitops/s (u64) (33.1% of 577 Gops/s (u64 bitop, est))
         gf128_N1M: correct, 1.35 ms, 198.6 Gbitops/s (u64) (34.4% of 577 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.2640

## History

- iter  2: compile=OK | correct=True | score=0.2639807813260745
- iter  3: compile=OK | correct=True | score=0.2345643579070991
- iter  4: compile=OK | correct=True | score=0.1816052029084452
- iter  5: compile=OK | correct=True | score=0.182439594679636
- iter  6: compile=OK | correct=True | score=0.21516589854758555
- iter  7: compile=OK | correct=True | score=0.027920129901165772
- iter  8: compile=OK | correct=True | score=0.04614378164267573
- iter  9: compile=OK | correct=True | score=0.1689933212723367

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
