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

// Build a 16-entry table: T[k] = k (as 4-bit polynomial) * a (128-bit poly).
// Each entry is 192 bits stored as (lo, mid, hi). Stored as flat thread arrays.
inline void build_table(
    ulong a_lo, ulong a_hi,
    thread ulong *Tlo, thread ulong *Tmid, thread ulong *Thi)
{
    // T[0] = 0
    Tlo[0]  = 0ul; Tmid[0] = 0ul; Thi[0] = 0ul;
    // T[1] = a
    Tlo[1]  = a_lo; Tmid[1] = a_hi; Thi[1] = 0ul;
    // T[2] = a << 1
    Tlo[2]  = a_lo << 1;
    Tmid[2] = (a_hi << 1) | (a_lo >> 63);
    Thi[2]  = a_hi >> 63;
    // T[3] = T[2] ^ T[1]
    Tlo[3]  = Tlo[2] ^ Tlo[1];
    Tmid[3] = Tmid[2] ^ Tmid[1];
    Thi[3]  = Thi[2] ^ Thi[1];
    // T[4] = a << 2
    Tlo[4]  = a_lo << 2;
    Tmid[4] = (a_hi << 2) | (a_lo >> 62);
    Thi[4]  = a_hi >> 62;
    // T[5] = T[4] ^ T[1]
    Tlo[5]  = Tlo[4] ^ Tlo[1];
    Tmid[5] = Tmid[4] ^ Tmid[1];
    Thi[5]  = Thi[4] ^ Thi[1];
    // T[6] = T[4] ^ T[2]
    Tlo[6]  = Tlo[4] ^ Tlo[2];
    Tmid[6] = Tmid[4] ^ Tmid[2];
    Thi[6]  = Thi[4] ^ Thi[2];
    // T[7] = T[4] ^ T[3]
    Tlo[7]  = Tlo[4] ^ Tlo[3];
    Tmid[7] = Tmid[4] ^ Tmid[3];
    Thi[7]  = Thi[4] ^ Thi[3];
    // T[8] = a << 3
    Tlo[8]  = a_lo << 3;
    Tmid[8] = (a_hi << 3) | (a_lo >> 61);
    Thi[8]  = a_hi >> 61;
    // T[9..15] = T[8] ^ T[1..7]
    for (uint k = 1u; k < 8u; ++k) {
        Tlo[8u + k]  = Tlo[8] ^ Tlo[k];
        Tmid[8u + k] = Tmid[8] ^ Tmid[k];
        Thi[8u + k]  = Thi[8] ^ Thi[k];
    }
}

// 128x128 -> 256-bit carry-less multiply, nibble-windowed.
inline void clmul128_unreduced(
    ulong a_lo, ulong a_hi, ulong b_lo, ulong b_hi,
    thread ulong &t0, thread ulong &t1,
    thread ulong &t2, thread ulong &t3)
{
    ulong Tlo[16], Tmid[16], Thi[16];
    build_table(a_lo, a_hi, Tlo, Tmid, Thi);

    t0 = 0ul; t1 = 0ul; t2 = 0ul; t3 = 0ul;

    // Process b_lo: 16 nibbles, shifts 0,4,8,...,60
    // Each contributes (Tlo,Tmid,Thi) shifted left by (4*i) into (t0,t1,t2).
    for (uint i = 0u; i < 16u; ++i) {
        uint sh = i * 4u;
        uint nib = (uint)((b_lo >> sh) & 0xFul);
        ulong lo  = Tlo[nib];
        ulong mid = Tmid[nib];
        ulong hi  = Thi[nib];
        if (sh == 0u) {
            t0 ^= lo;
            t1 ^= mid;
            t2 ^= hi;
        } else {
            uint rsh = 64u - sh;
            t0 ^= lo << sh;
            t1 ^= (mid << sh) | (lo  >> rsh);
            t2 ^= (hi  << sh) | (mid >> rsh);
            // hi >> rsh contributes to t3 only when sh > 0 and hi has bits in top
            t3 ^= hi >> rsh;
        }
    }

    // Process b_hi: contributes to (t1,t2,t3) shifted by (4*i)
    for (uint i = 0u; i < 16u; ++i) {
        uint sh = i * 4u;
        uint nib = (uint)((b_hi >> sh) & 0xFul);
        ulong lo  = Tlo[nib];
        ulong mid = Tmid[nib];
        ulong hi  = Thi[nib];
        if (sh == 0u) {
            t1 ^= lo;
            t2 ^= mid;
            t3 ^= hi;
        } else {
            uint rsh = 64u - sh;
            t1 ^= lo << sh;
            t2 ^= (mid << sh) | (lo  >> rsh);
            t3 ^= (hi  << sh) | (mid >> rsh);
        }
    }
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
        gf128_N64K: correct, 0.40 ms, 42.4 Gbitops/s (u64) (7.3% of 577 Gops/s (u64 bitop, est))
       gf128_N256K: correct, 1.45 ms, 46.2 Gbitops/s (u64) (8.0% of 577 Gops/s (u64 bitop, est))
         gf128_N1M: correct, 5.12 ms, 52.4 Gbitops/s (u64) (9.1% of 577 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.0811

## Current best (incumbent)

```metal
// Naive seed for batched binary-field carry-less multiplication (Z11).
//
// Two parameter sets selected by the runtime ``tower`` flag:
//
//   tower = 0  ->  GF(2^128) products. Each element is 2 ``ulong``
//                  limbs (limb 0 = bits 0..63, i.e. the polynomial
//                  coefficients of x^0..x^63; limb 1 = bits 64..127).
//                  The product is computed in GF(2)[x] (carry-less
//                  multiply, no integer multiplies) and reduced
//                  modulo the AES-GCM irreducible polynomial
//                  R(x) = x^128 + x^7 + x^2 + x + 1 via the standard
//                  two-stage fold.
//
//   tower = 1  ->  GF(2^256) products on the Fan-Hasan tower
//                  ``GF(2^128)[v] / (v^2 + v + alpha)`` for a fixed
//                  ``alpha`` supplied via ``alpha_lo`` / ``alpha_hi``.
//                  Each element is 4 ``ulong`` limbs: limbs 0, 1 hold
//                  the v^0 coefficient ``a_0 in GF(2^128)`` and
//                  limbs 2, 3 hold the v^1 coefficient ``a_1``.
//                  The product ``(a_0 + a_1 v) (b_0 + b_1 v) = c_0 + c_1 v``
//                  decomposes (in characteristic 2, using v^2 = v + alpha) as
//
//                    c_0 = a_0 b_0 + alpha * (a_1 b_1)
//                    c_1 = a_0 b_1 + a_1 b_0 + a_1 b_1
//
// Buffer layout (host-fixed; must be preserved by candidate kernels):
//
//   buffer 0: device const ulong *a         (batch * field_words u64s)
//   buffer 1: device const ulong *b         (batch * field_words u64s)
//   buffer 2: device       ulong *c         (batch * field_words u64s)
//   buffer 3: constant ulong &alpha_lo      (low 64 bits of alpha; tower-only)
//   buffer 4: constant ulong &alpha_hi      (high 64 bits of alpha; tower-only)
//   buffer 5: constant uint  &tower         (0 = GF(2^128); 1 = GF(2^256))
//   buffer 6: constant uint  &batch
//
// ``field_words`` is implicit: 2 when ``tower == 0``, 4 when ``tower == 1``.
//
// Dispatch (host-fixed):
//   threadsPerGrid        = (batch, 1, 1)
//   threadsPerThreadgroup = (min(batch, 64), 1, 1)
// Each thread processes ONE product end-to-end; guard against
// ``idx >= batch`` (the grid is rounded up to a multiple of the TG
// width). Threadgroup- or simdgroup-cooperative implementations are
// valid so long as the external buffer layout above and the
// canonical output contract below are preserved.
//
// Canonical output: each output limb is the raw 64-bit polynomial
// coefficient pattern; the host compares bit-exactly against a
// CPU GF(2^128) / GF(2^256)-tower reference.

#include <metal_stdlib>
using namespace metal;


// ----------------------------------------------------------------------
// 128-bit polynomial product into a 256-bit accumulator
// ----------------------------------------------------------------------
//
// Computes the unreduced GF(2)[x] product of two 128-bit operands
// into (t0, t1, t2, t3) where t0 = bits 0..63, t1 = 64..127,
// t2 = 128..191, t3 = 192..255. Shift amounts are guarded against
// the 0 / 64 corner cases so the kernel is portable across
// compilers that treat ``x >> 64`` as either zero or unchanged.

inline void clmul128_unreduced(
    ulong a_lo, ulong a_hi, ulong b_lo, ulong b_hi,
    thread ulong &t0, thread ulong &t1,
    thread ulong &t2, thread ulong &t3)
{
    t0 = 0ul; t1 = 0ul; t2 = 0ul; t3 = 0ul;

    // bit positions 0..63 (from b_lo)
    for (uint i = 0u; i < 64u; ++i) {
        ulong bit = (b_lo >> i) & 1ul;
        ulong mask = 0ul - bit;                  // 0 or all-ones

        ulong sh0, sh1, sh2;
        if (i == 0u) {
            sh0 = a_lo;
            sh1 = a_hi;
            sh2 = 0ul;
        } else {
            sh0 = a_lo << i;
            sh1 = (a_hi << i) | (a_lo >> (64u - i));
            sh2 = a_hi >> (64u - i);
        }
        t0 ^= sh0 & mask;
        t1 ^= sh1 & mask;
        t2 ^= sh2 & mask;
    }

    // bit positions 64..127 (from b_hi)
    for (uint i = 0u; i < 64u; ++i) {
        ulong bit = (b_hi >> i) & 1ul;
        ulong mask = 0ul - bit;

        ulong sh1, sh2, sh3;
        if (i == 0u) {
            sh1 = a_lo;
            sh2 = a_hi;
            sh3 = 0ul;
        } else {
            sh1 = a_lo << i;
            sh2 = (a_hi << i) | (a_lo >> (64u - i));
            sh3 = a_hi >> (64u - i);
        }
        t1 ^= sh1 & mask;
        t2 ^= sh2 & mask;
        t3 ^= sh3 & mask;
    }
}


// ----------------------------------------------------------------------
// Two-stage GCM-style reduction
// ----------------------------------------------------------------------
//
// Input  : 256-bit polynomial (t0, t1, t2, t3).
// Output : 128-bit polynomial (r_lo, r_hi) congruent modulo
//          R(x) = x^128 + x^7 + x^2 + x + 1.
//
// Stage 1: fold (t2, t3) -- the bits at positions 128..255 -- via the
// low pattern of R(x), namely (1 + x + x^2 + x^7). The fold produces
// a 135-bit value whose low 128 bits go into (t0, t1) and whose top
// 7 bits sit in the residual ``d_hi`` (bit positions 128..134).
//
// Stage 2: fold ``d_hi`` once more. With 7 input bits and a 7-bit
// pattern, the result has at most 14 bits and fits entirely in t0.

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
        // GF(2^128): one mul + reduction per thread.
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
        // GF(2^256) Fan-Hasan tower:
        //   c_0 = a_0 b_0 + alpha * (a_1 b_1)
        //   c_1 = a_0 b_1 + a_1 b_0 + a_1 b_1
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

Incumbent result:
        gf128_N64K: correct, 0.61 ms, 27.6 Gbitops/s (u64) (4.8% of 577 Gops/s (u64 bitop, est))
       gf128_N256K: correct, 1.03 ms, 65.0 Gbitops/s (u64) (11.3% of 577 Gops/s (u64 bitop, est))
         gf128_N1M: correct, 3.53 ms, 76.1 Gbitops/s (u64) (13.2% of 577 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.0892

## History

- iter  0: compile=OK | correct=True | score=0.08919958327804488
- iter  1: compile=OK | correct=True | score=0.08109579020171731

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
