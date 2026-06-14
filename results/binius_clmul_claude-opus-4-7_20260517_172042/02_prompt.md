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

// 4-bit windowed clmul: 128x128 -> 256 bits as (t0,t1,t2,t3).
// Precompute 16 multiples of a (each 192 bits: lo, mid, hi), then
// scan b in 4-bit nibbles from high to low, shifting accumulator
// left by 4 each step and XOR-ing the table lookup.
inline void clmul128_unreduced(
    ulong a_lo, ulong a_hi, ulong b_lo, ulong b_hi,
    thread ulong &t0, thread ulong &t1,
    thread ulong &t2, thread ulong &t3)
{
    // Build table T[k] = a * k for k = 0..15, as 192-bit triple.
    // a fits in 128 bits; a*15 fits in 132 bits, so 192 bits is plenty.
    ulong tlo[16];
    ulong tmd[16];
    ulong thi[16];

    // k = 0
    tlo[0] = 0ul; tmd[0] = 0ul; thi[0] = 0ul;
    // k = 1: a
    tlo[1] = a_lo;
    tmd[1] = a_hi;
    thi[1] = 0ul;
    // k = 2: a << 1
    tlo[2] = a_lo << 1;
    tmd[2] = (a_hi << 1) | (a_lo >> 63);
    thi[2] = a_hi >> 63;
    // k = 4: a << 2
    tlo[4] = a_lo << 2;
    tmd[4] = (a_hi << 2) | (a_lo >> 62);
    thi[4] = a_hi >> 62;
    // k = 8: a << 3
    tlo[8] = a_lo << 3;
    tmd[8] = (a_hi << 3) | (a_lo >> 61);
    thi[8] = a_hi >> 61;

    // Combine via XOR for the rest
    // k = 3 = 1^2
    tlo[3] = tlo[1] ^ tlo[2]; tmd[3] = tmd[1] ^ tmd[2]; thi[3] = thi[1] ^ thi[2];
    // k = 5 = 1^4
    tlo[5] = tlo[1] ^ tlo[4]; tmd[5] = tmd[1] ^ tmd[4]; thi[5] = thi[1] ^ thi[4];
    // k = 6 = 2^4
    tlo[6] = tlo[2] ^ tlo[4]; tmd[6] = tmd[2] ^ tmd[4]; thi[6] = thi[2] ^ thi[4];
    // k = 7 = 3^4
    tlo[7] = tlo[3] ^ tlo[4]; tmd[7] = tmd[3] ^ tmd[4]; thi[7] = thi[3] ^ thi[4];
    // k = 9 = 1^8
    tlo[9] = tlo[1] ^ tlo[8]; tmd[9] = tmd[1] ^ tmd[8]; thi[9] = thi[1] ^ thi[8];
    // k = 10 = 2^8
    tlo[10] = tlo[2] ^ tlo[8]; tmd[10] = tmd[2] ^ tmd[8]; thi[10] = thi[2] ^ thi[8];
    // k = 11 = 3^8
    tlo[11] = tlo[3] ^ tlo[8]; tmd[11] = tmd[3] ^ tmd[8]; thi[11] = thi[3] ^ thi[8];
    // k = 12 = 4^8
    tlo[12] = tlo[4] ^ tlo[8]; tmd[12] = tmd[4] ^ tmd[8]; thi[12] = thi[4] ^ thi[8];
    // k = 13 = 5^8
    tlo[13] = tlo[5] ^ tlo[8]; tmd[13] = tmd[5] ^ tmd[8]; thi[13] = thi[5] ^ thi[8];
    // k = 14 = 6^8
    tlo[14] = tlo[6] ^ tlo[8]; tmd[14] = tmd[6] ^ tmd[8]; thi[14] = thi[6] ^ thi[8];
    // k = 15 = 7^8
    tlo[15] = tlo[7] ^ tlo[8]; tmd[15] = tmd[7] ^ tmd[8]; thi[15] = thi[7] ^ thi[8];

    // Accumulator (lo, mid, hi, top) = 4 * 64 = 256 bits
    ulong r0 = 0ul, r1 = 0ul, r2 = 0ul, r3 = 0ul;

    // Process b_hi first (high nibbles), then b_lo.
    // We scan from MSB nibble down, shifting acc left by 4 each step.
    // Total 32 nibbles.

    // b_hi nibbles, from bit 60..0
    for (int s = 60; s >= 0; s -= 4) {
        // shift acc left by 4
        ulong n3 = (r3 << 4) | (r2 >> 60);
        ulong n2 = (r2 << 4) | (r1 >> 60);
        ulong n1 = (r1 << 4) | (r0 >> 60);
        ulong n0 = (r0 << 4);
        uint k = (uint)((b_hi >> s) & 0xFul);
        r0 = n0 ^ tlo[k];
        r1 = n1 ^ tmd[k];
        r2 = n2 ^ thi[k];
        r3 = n3;
    }
    // b_lo nibbles
    for (int s = 60; s >= 0; s -= 4) {
        ulong n3 = (r3 << 4) | (r2 >> 60);
        ulong n2 = (r2 << 4) | (r1 >> 60);
        ulong n1 = (r1 << 4) | (r0 >> 60);
        ulong n0 = (r0 << 4);
        uint k = (uint)((b_lo >> s) & 0xFul);
        r0 = n0 ^ tlo[k];
        r1 = n1 ^ tmd[k];
        r2 = n2 ^ thi[k];
        r3 = n3;
    }

    t0 = r0;
    t1 = r1;
    t2 = r2;
    t3 = r3;
}

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
        size_t base = (size_t)idx * (size_t)2;
        ulong a_lo = a[base + 0];
        ulong a_hi = a[base + 1];
        ulong b_lo = b[base + 0];
        ulong b_hi = b[base + 1];

        ulong c_lo, c_hi;
        gf128_mul(a_lo, a_hi, b_lo, b_hi, c_lo, c_hi);

        c[base + 0] = c_lo;
        c[base + 1] = c_hi;
    } else {
        size_t base = (size_t)idx * (size_t)4;
        ulong a0_lo = a[base + 0], a0_hi = a[base + 1];
        ulong a1_lo = a[base + 2], a1_hi = a[base + 3];
        ulong b0_lo = b[base + 0], b0_hi = b[base + 1];
        ulong b1_lo = b[base + 2], b1_hi = b[base + 3];

        ulong m00_lo, m00_hi; gf128_mul(a0_lo, a0_hi, b0_lo, b0_hi, m00_lo, m00_hi);
        ulong m01_lo, m01_hi; gf128_mul(a0_lo, a0_hi, b1_lo, b1_hi, m01_lo, m01_hi);
        ulong m10_lo, m10_hi; gf128_mul(a1_lo, a1_hi, b0_lo, b0_hi, m10_lo, m10_hi);
        ulong m11_lo, m11_hi; gf128_mul(a1_lo, a1_hi, b1_lo, b1_hi, m11_lo, m11_hi);

        ulong am_lo, am_hi;
        gf128_mul(alpha_lo, alpha_hi, m11_lo, m11_hi, am_lo, am_hi);

        ulong c0_lo = m00_lo ^ am_lo;
        ulong c0_hi = m00_hi ^ am_hi;
        ulong c1_lo = m01_lo ^ m10_lo ^ m11_lo;
        ulong c1_hi = m01_hi ^ m10_hi ^ m11_hi;

        c[base + 0] = c0_lo;
        c[base + 1] = c0_hi;
        c[base + 2] = c1_lo;
        c[base + 3] = c1_hi;
    }
}
```

Result of previous attempt:
        gf128_N64K: correct, 0.33 ms, 50.8 Gbitops/s (u64) (8.8% of 577 Gops/s (u64 bitop, est))
       gf128_N256K: correct, 0.72 ms, 93.5 Gbitops/s (u64) (16.2% of 577 Gops/s (u64 bitop, est))
         gf128_N1M: correct, 2.79 ms, 96.1 Gbitops/s (u64) (16.6% of 577 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.1333

## History

- iter  0: compile=OK | correct=True | score=0.09374361344761027
- iter  1: compile=OK | correct=True | score=0.13334647948496645

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
