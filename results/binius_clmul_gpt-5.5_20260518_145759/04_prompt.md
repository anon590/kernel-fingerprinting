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
// Carry-less 16x16 -> 32 using 3 interleaved bit lanes.
// ----------------------------------------------------------------------

inline uint clmul16_full(uint x, uint y)
{
    const uint A0 = 0x00009249u; // bits 0,3,6,9,12,15
    const uint A1 = 0x00002492u; // bits 1,4,7,10,13
    const uint A2 = 0x00004924u; // bits 2,5,8,11,14

    const uint M0 = 0x49249249u; // bits == 0 mod 3
    const uint M1 = 0x92492492u; // bits == 1 mod 3
    const uint M2 = 0x24924924u; // bits == 2 mod 3

    uint x0 = x & A0;
    uint x1 = x & A1;
    uint x2 = x & A2;
    uint y0 = y & A0;
    uint y1 = y & A1;
    uint y2 = y & A2;

    uint z0 = (x0 * y0) ^ (x1 * y2) ^ (x2 * y1);
    uint z1 = (x0 * y1) ^ (x1 * y0) ^ (x2 * y2);
    uint z2 = (x0 * y2) ^ (x1 * y1) ^ (x2 * y0);

    return (z0 & M0) | (z1 & M1) | (z2 & M2);
}

// ----------------------------------------------------------------------
// Carry-less 32x32 -> 64 by Karatsuba over 16-bit halves.
// ----------------------------------------------------------------------

inline ulong clmul32_full(uint x, uint y)
{
    uint x0 = x & 0xffffu;
    uint x1 = x >> 16u;
    uint y0 = y & 0xffffu;
    uint y1 = y >> 16u;

    uint p0 = clmul16_full(x0, y0);
    uint p2 = clmul16_full(x1, y1);
    uint pm = clmul16_full(x0 ^ x1, y0 ^ y1);
    uint mid = pm ^ p0 ^ p2;

    return ((ulong)p0) ^ (((ulong)mid) << 16u) ^ (((ulong)p2) << 32u);
}

// ----------------------------------------------------------------------
// Carry-less 64x64 -> 128 by Karatsuba over 32-bit halves.
// ----------------------------------------------------------------------

inline void clmul64_full(
    ulong x, ulong y,
    thread ulong &lo, thread ulong &hi)
{
    uint x0 = (uint)x;
    uint x1 = (uint)(x >> 32u);
    uint y0 = (uint)y;
    uint y1 = (uint)(y >> 32u);

    ulong p0 = clmul32_full(x0, y0);
    ulong p2 = clmul32_full(x1, y1);
    ulong pm = clmul32_full(x0 ^ x1, y0 ^ y1);
    ulong mid = pm ^ p0 ^ p2;

    lo = p0 ^ (mid << 32u);
    hi = (mid >> 32u) ^ p2;
}

// ----------------------------------------------------------------------
// Carry-less 128x128 -> 256 by Karatsuba over 64-bit halves.
// ----------------------------------------------------------------------

inline void clmul128_unreduced(
    ulong a_lo, ulong a_hi, ulong b_lo, ulong b_hi,
    thread ulong &t0, thread ulong &t1,
    thread ulong &t2, thread ulong &t3)
{
    ulong z0_lo, z0_hi;
    ulong z2_lo, z2_hi;
    ulong zm_lo, zm_hi;

    clmul64_full(a_lo, b_lo, z0_lo, z0_hi);
    clmul64_full(a_hi, b_hi, z2_lo, z2_hi);
    clmul64_full(a_lo ^ a_hi, b_lo ^ b_hi, zm_lo, zm_hi);

    ulong z1_lo = zm_lo ^ z0_lo ^ z2_lo;
    ulong z1_hi = zm_hi ^ z0_hi ^ z2_hi;

    t0 = z0_lo;
    t1 = z0_hi ^ z1_lo;
    t2 = z2_lo ^ z1_hi;
    t3 = z2_hi;
}

// ----------------------------------------------------------------------
// Sparse carry-less 128x128 -> 256 for a chosen multiplier.
// ----------------------------------------------------------------------

inline void clmul128_unreduced_sparse_multiplier(
    ulong x_lo, ulong x_hi, ulong m_lo, ulong m_hi,
    thread ulong &t0, thread ulong &t1,
    thread ulong &t2, thread ulong &t3)
{
    t0 = 0ul;
    t1 = 0ul;
    t2 = 0ul;
    t3 = 0ul;

    ulong ml = m_lo;
    while (ml != 0ul) {
        uint s = (uint)ctz(ml);
        if (s == 0u) {
            t0 ^= x_lo;
            t1 ^= x_hi;
        } else {
            t0 ^= x_lo << s;
            t1 ^= (x_hi << s) | (x_lo >> (64u - s));
            t2 ^= x_hi >> (64u - s);
        }
        ml &= (ml - 1ul);
    }

    ulong mh = m_hi;
    while (mh != 0ul) {
        uint s = (uint)ctz(mh);
        if (s == 0u) {
            t1 ^= x_lo;
            t2 ^= x_hi;
        } else {
            t1 ^= x_lo << s;
            t2 ^= (x_hi << s) | (x_lo >> (64u - s));
            t3 ^= x_hi >> (64u - s);
        }
        mh &= (mh - 1ul);
    }
}

// ----------------------------------------------------------------------
// Reduction modulo x^128 + x^7 + x^2 + x + 1.
// True 128x128 products have no x^255 term.
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

    ulong d_hi = (t3 >> 62u) ^ (t3 >> 57u);

    t0 ^= d_lo0;
    t1 ^= d_lo1;
    t0 ^= d_hi ^ (d_hi << 1u) ^ (d_hi << 2u) ^ (d_hi << 7u);

    r_lo = t0;
    r_hi = t1;
}

inline void gf128_mul_karat(
    ulong a_lo, ulong a_hi, ulong b_lo, ulong b_hi,
    thread ulong &c_lo, thread ulong &c_hi)
{
    ulong t0, t1, t2, t3;
    clmul128_unreduced(a_lo, a_hi, b_lo, b_hi, t0, t1, t2, t3);
    gcm_reduce(t0, t1, t2, t3, c_lo, c_hi);
}

inline void gf128_mul_sparse_multiplier(
    ulong x_lo, ulong x_hi, ulong m_lo, ulong m_hi,
    thread ulong &c_lo, thread ulong &c_hi)
{
    ulong t0, t1, t2, t3;
    clmul128_unreduced_sparse_multiplier(x_lo, x_hi, m_lo, m_hi, t0, t1, t2, t3);
    gcm_reduce(t0, t1, t2, t3, c_lo, c_hi);
}

// Choose the lower-Hamming-weight operand as the sparse multiplier.
// This path is used only for large batches to avoid hurting the strong
// small/medium-batch Karatsuba timings.
inline void gf128_mul_sparse_minpop(
    ulong a_lo, ulong a_hi, ulong b_lo, ulong b_hi,
    thread ulong &c_lo, thread ulong &c_hi)
{
    uint wa = (uint)popcount(a_lo) + (uint)popcount(a_hi);
    uint wb = (uint)popcount(b_lo) + (uint)popcount(b_hi);

    bool use_a_as_multiplier = (wa < wb);

    ulong x_lo = use_a_as_multiplier ? b_lo : a_lo;
    ulong x_hi = use_a_as_multiplier ? b_hi : a_hi;
    ulong m_lo = use_a_as_multiplier ? a_lo : b_lo;
    ulong m_hi = use_a_as_multiplier ? a_hi : b_hi;

    gf128_mul_sparse_multiplier(x_lo, x_hi, m_lo, m_hi, c_lo, c_hi);
}

inline void gf128_mul_auto(
    ulong a_lo, ulong a_hi, ulong b_lo, ulong b_hi,
    bool use_sparse_large,
    thread ulong &c_lo, thread ulong &c_hi)
{
    if (use_sparse_large) {
        gf128_mul_sparse_minpop(a_lo, a_hi, b_lo, b_hi, c_lo, c_hi);
    } else {
        gf128_mul_karat(a_lo, a_hi, b_lo, b_hi, c_lo, c_hi);
    }
}

inline void gf128_mul_alpha_weighted(
    ulong x_lo, ulong x_hi,
    ulong alpha_lo, ulong alpha_hi,
    uint alpha_weight,
    bool use_sparse_large,
    thread ulong &c_lo, thread ulong &c_hi)
{
    if (alpha_weight <= 64u) {
        gf128_mul_sparse_multiplier(x_lo, x_hi, alpha_lo, alpha_hi, c_lo, c_hi);
    } else {
        gf128_mul_auto(x_lo, x_hi, alpha_lo, alpha_hi, use_sparse_large, c_lo, c_hi);
    }
}

// ----------------------------------------------------------------------
// Required kernel.
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
    if (idx >= batch) {
        return;
    }

    // Preserve the incumbent multiply for the small/medium tests, where it
    // is already very fast; switch only for the long batch.
    bool use_sparse_large = (batch >= 1048576u);

    if (tower == 0u) {
        uint base = idx << 1u;

        ulong a_lo = a[base + 0u];
        ulong a_hi = a[base + 1u];
        ulong b_lo = b[base + 0u];
        ulong b_hi = b[base + 1u];

        ulong c_lo, c_hi;
        gf128_mul_auto(a_lo, a_hi, b_lo, b_hi, use_sparse_large, c_lo, c_hi);

        c[base + 0u] = c_lo;
        c[base + 1u] = c_hi;
        return;
    }

    uint base = idx << 2u;

    ulong a0_lo = a[base + 0u];
    ulong a0_hi = a[base + 1u];
    ulong a1_lo = a[base + 2u];
    ulong a1_hi = a[base + 3u];

    ulong b0_lo = b[base + 0u];
    ulong b0_hi = b[base + 1u];
    ulong b1_lo = b[base + 2u];
    ulong b1_hi = b[base + 3u];

    uint alpha_weight = (uint)popcount(alpha_lo) + (uint)popcount(alpha_hi);

    // Extension-field Karatsuba:
    // mix = (a0+a1)(b0+b1) = m00 + m01 + m10 + m11
    // c1  = m01 + m10 + m11 = mix + m00
    ulong m00_lo, m00_hi;
    ulong m11_lo, m11_hi;
    ulong mix_lo, mix_hi;
    ulong am_lo,  am_hi;

    gf128_mul_auto(a0_lo, a0_hi, b0_lo, b0_hi,
                   use_sparse_large, m00_lo, m00_hi);

    gf128_mul_auto(a1_lo, a1_hi, b1_lo, b1_hi,
                   use_sparse_large, m11_lo, m11_hi);

    gf128_mul_auto(a0_lo ^ a1_lo, a0_hi ^ a1_hi,
                   b0_lo ^ b1_lo, b0_hi ^ b1_hi,
                   use_sparse_large, mix_lo, mix_hi);

    gf128_mul_alpha_weighted(m11_lo, m11_hi,
                             alpha_lo, alpha_hi,
                             alpha_weight,
                             use_sparse_large,
                             am_lo, am_hi);

    c[base + 0u] = m00_lo ^ am_lo;
    c[base + 1u] = m00_hi ^ am_hi;
    c[base + 2u] = mix_lo ^ m00_lo;
    c[base + 3u] = mix_hi ^ m00_hi;
}
```

Result of previous attempt:
        gf128_N64K: correct, 0.14 ms, 121.0 Gbitops/s (u64) (21.0% of 577 Gops/s (u64 bitop, est))
       gf128_N256K: correct, 0.24 ms, 279.2 Gbitops/s (u64) (48.4% of 577 Gops/s (u64 bitop, est))
         gf128_N1M: correct, 2.45 ms, 109.4 Gbitops/s (u64) (18.9% of 577 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.2678

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

// ----------------------------------------------------------------------
// Carry-less 16x16 -> 32 using 3 interleaved bit lanes.
//
// For 16-bit operands each modulo-3 lane has at most 6 bits, so ordinary
// base-8 products have coefficients < 8 and cannot carry between lanes.
// ----------------------------------------------------------------------

inline uint clmul16_full(uint x, uint y)
{
    const uint A0 = 0x00009249u; // bits 0,3,6,9,12,15
    const uint A1 = 0x00002492u; // bits 1,4,7,10,13
    const uint A2 = 0x00004924u; // bits 2,5,8,11,14

    const uint M0 = 0x49249249u; // bits == 0 mod 3
    const uint M1 = 0x92492492u; // bits == 1 mod 3
    const uint M2 = 0x24924924u; // bits == 2 mod 3

    uint x0 = x & A0;
    uint x1 = x & A1;
    uint x2 = x & A2;
    uint y0 = y & A0;
    uint y1 = y & A1;
    uint y2 = y & A2;

    uint z0 = (x0 * y0) ^ (x1 * y2) ^ (x2 * y1);
    uint z1 = (x0 * y1) ^ (x1 * y0) ^ (x2 * y2);
    uint z2 = (x0 * y2) ^ (x1 * y1) ^ (x2 * y0);

    return (z0 & M0) | (z1 & M1) | (z2 & M2);
}

// ----------------------------------------------------------------------
// Carry-less 32x32 -> 64 by Karatsuba over 16-bit halves.
// Uses only low 32-bit integer multiplies inside clmul16_full.
// ----------------------------------------------------------------------

inline ulong clmul32_full(uint x, uint y)
{
    uint x0 = x & 0xffffu;
    uint x1 = x >> 16u;
    uint y0 = y & 0xffffu;
    uint y1 = y >> 16u;

    uint p0 = clmul16_full(x0, y0);
    uint p2 = clmul16_full(x1, y1);
    uint pm = clmul16_full(x0 ^ x1, y0 ^ y1);
    uint mid = pm ^ p0 ^ p2;

    return ((ulong)p0) ^ (((ulong)mid) << 16u) ^ (((ulong)p2) << 32u);
}

// ----------------------------------------------------------------------
// Carry-less 64x64 -> 128 by Karatsuba over 32-bit halves.
// ----------------------------------------------------------------------

inline void clmul64_full(
    ulong x, ulong y,
    thread ulong &lo, thread ulong &hi)
{
    uint x0 = (uint)x;
    uint x1 = (uint)(x >> 32u);
    uint y0 = (uint)y;
    uint y1 = (uint)(y >> 32u);

    ulong p0 = clmul32_full(x0, y0);
    ulong p2 = clmul32_full(x1, y1);
    ulong pm = clmul32_full(x0 ^ x1, y0 ^ y1);
    ulong mid = pm ^ p0 ^ p2;

    lo = p0 ^ (mid << 32u);
    hi = (mid >> 32u) ^ p2;
}

// ----------------------------------------------------------------------
// Carry-less 128x128 -> 256 by Karatsuba over 64-bit halves.
// ----------------------------------------------------------------------

inline void clmul128_unreduced(
    ulong a_lo, ulong a_hi, ulong b_lo, ulong b_hi,
    thread ulong &t0, thread ulong &t1,
    thread ulong &t2, thread ulong &t3)
{
    ulong z0_lo, z0_hi;
    ulong z2_lo, z2_hi;
    ulong zm_lo, zm_hi;

    clmul64_full(a_lo, b_lo, z0_lo, z0_hi);
    clmul64_full(a_hi, b_hi, z2_lo, z2_hi);
    clmul64_full(a_lo ^ a_hi, b_lo ^ b_hi, zm_lo, zm_hi);

    ulong z1_lo = zm_lo ^ z0_lo ^ z2_lo;
    ulong z1_hi = zm_hi ^ z0_hi ^ z2_hi;

    t0 = z0_lo;
    t1 = z0_hi ^ z1_lo;
    t2 = z2_lo ^ z1_hi;
    t3 = z2_hi;
}

// ----------------------------------------------------------------------
// Sparse carry-less 128x128 -> 256 for uniform alpha multipliers.
// ----------------------------------------------------------------------

inline void clmul128_unreduced_sparse_multiplier(
    ulong x_lo, ulong x_hi, ulong m_lo, ulong m_hi,
    thread ulong &t0, thread ulong &t1,
    thread ulong &t2, thread ulong &t3)
{
    t0 = 0ul;
    t1 = 0ul;
    t2 = 0ul;
    t3 = 0ul;

    ulong ml = m_lo;
    while (ml != 0ul) {
        uint s = (uint)ctz(ml);
        if (s == 0u) {
            t0 ^= x_lo;
            t1 ^= x_hi;
        } else {
            t0 ^= x_lo << s;
            t1 ^= (x_hi << s) | (x_lo >> (64u - s));
            t2 ^= x_hi >> (64u - s);
        }
        ml &= (ml - 1ul);
    }

    ulong mh = m_hi;
    while (mh != 0ul) {
        uint s = (uint)ctz(mh);
        if (s == 0u) {
            t1 ^= x_lo;
            t2 ^= x_hi;
        } else {
            t1 ^= x_lo << s;
            t2 ^= (x_hi << s) | (x_lo >> (64u - s));
            t3 ^= x_hi >> (64u - s);
        }
        mh &= (mh - 1ul);
    }
}

// ----------------------------------------------------------------------
// Reduction modulo x^128 + x^7 + x^2 + x + 1.
// True 128x128 products have no x^255 term, so t3 bit 63 is zero.
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

    ulong d_hi = (t3 >> 62u) ^ (t3 >> 57u);

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

inline void gf128_mul_sparse_multiplier(
    ulong x_lo, ulong x_hi, ulong m_lo, ulong m_hi,
    thread ulong &c_lo, thread ulong &c_hi)
{
    ulong t0, t1, t2, t3;
    clmul128_unreduced_sparse_multiplier(x_lo, x_hi, m_lo, m_hi, t0, t1, t2, t3);
    gcm_reduce(t0, t1, t2, t3, c_lo, c_hi);
}

inline void gf128_mul_alpha_weighted(
    ulong x_lo, ulong x_hi,
    ulong alpha_lo, ulong alpha_hi,
    uint alpha_weight,
    thread ulong &c_lo, thread ulong &c_hi)
{
    if (alpha_weight <= 64u) {
        gf128_mul_sparse_multiplier(x_lo, x_hi, alpha_lo, alpha_hi, c_lo, c_hi);
    } else {
        gf128_mul(alpha_lo, alpha_hi, x_lo, x_hi, c_lo, c_hi);
    }
}

// ----------------------------------------------------------------------
// Required kernel.
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
    if (idx >= batch) {
        return;
    }

    if (tower == 0u) {
        uint base = idx << 1u;

        ulong a_lo = a[base + 0u];
        ulong a_hi = a[base + 1u];
        ulong b_lo = b[base + 0u];
        ulong b_hi = b[base + 1u];

        ulong c_lo, c_hi;
        gf128_mul(a_lo, a_hi, b_lo, b_hi, c_lo, c_hi);

        c[base + 0u] = c_lo;
        c[base + 1u] = c_hi;
        return;
    }

    uint base = idx << 2u;

    ulong a0_lo = a[base + 0u];
    ulong a0_hi = a[base + 1u];
    ulong a1_lo = a[base + 2u];
    ulong a1_hi = a[base + 3u];

    ulong b0_lo = b[base + 0u];
    ulong b0_hi = b[base + 1u];
    ulong b1_lo = b[base + 2u];
    ulong b1_hi = b[base + 3u];

    // Compute the uniform alpha popcount once per simdgroup.
    uint alpha_weight = 0u;
    if ((idx & 31u) == 0u) {
        alpha_weight = (uint)popcount(alpha_lo) + (uint)popcount(alpha_hi);
    }
    alpha_weight = simd_broadcast(alpha_weight, 0);

    // Extension-field Karatsuba:
    // mix = (a0+a1)(b0+b1) = m00 + m01 + m10 + m11
    // c1  = m01 + m10 + m11 = mix + m00
    ulong m00_lo, m00_hi;
    ulong m11_lo, m11_hi;
    ulong mix_lo, mix_hi;
    ulong am_lo,  am_hi;

    gf128_mul(a0_lo, a0_hi, b0_lo, b0_hi, m00_lo, m00_hi);
    gf128_mul(a1_lo, a1_hi, b1_lo, b1_hi, m11_lo, m11_hi);
    gf128_mul(a0_lo ^ a1_lo, a0_hi ^ a1_hi,
              b0_lo ^ b1_lo, b0_hi ^ b1_hi,
              mix_lo, mix_hi);

    gf128_mul_alpha_weighted(m11_lo, m11_hi,
                             alpha_lo, alpha_hi,
                             alpha_weight,
                             am_lo, am_hi);

    c[base + 0u] = m00_lo ^ am_lo;
    c[base + 1u] = m00_hi ^ am_hi;
    c[base + 2u] = mix_lo ^ m00_lo;
    c[base + 3u] = mix_hi ^ m00_hi;
}
```

Incumbent result:
        gf128_N64K: correct, 0.14 ms, 119.3 Gbitops/s (u64) (20.7% of 577 Gops/s (u64 bitop, est))
       gf128_N256K: correct, 0.20 ms, 341.7 Gbitops/s (u64) (59.2% of 577 Gops/s (u64 bitop, est))
         gf128_N1M: correct, 1.30 ms, 206.5 Gbitops/s (u64) (35.8% of 577 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.3523

## History

- iter  0: compile=OK | correct=True | score=0.1041251747903846
- iter  1: compile=OK | correct=True | score=0.22872673973641952
- iter  2: compile=OK | correct=True | score=0.3522878982638455
- iter  3: compile=OK | correct=True | score=0.2677757086188393

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
