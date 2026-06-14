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

inline uint clmul16_full(uint x, uint y)
{
    const uint A0 = 0x00009249u;
    const uint A1 = 0x00002492u;
    const uint A2 = 0x00004924u;

    const uint M0 = 0x49249249u;
    const uint M1 = 0x92492492u;
    const uint M2 = 0x24924924u;

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

inline uint2 clmul32_u32(uint x, uint y)
{
    uint x0 = x & 0xffffu;
    uint x1 = x >> 16u;
    uint y0 = y & 0xffffu;
    uint y1 = y >> 16u;

    uint p0 = clmul16_full(x0, y0);
    uint p2 = clmul16_full(x1, y1);
    uint pm = clmul16_full(x0 ^ x1, y0 ^ y1);
    uint mid = pm ^ p0 ^ p2;

    return uint2(p0 ^ (mid << 16u), (mid >> 16u) ^ p2);
}

inline uint4 clmul64_u32(uint a0, uint a1, uint b0, uint b1)
{
    uint2 p0 = clmul32_u32(a0, b0);
    uint2 p2 = clmul32_u32(a1, b1);
    uint2 pm = clmul32_u32(a0 ^ a1, b0 ^ b1);
    uint2 mid = pm ^ p0 ^ p2;

    return uint4(p0.x,
                 p0.y ^ mid.x,
                 mid.y ^ p2.x,
                 p2.y);
}

inline void clmul64_full(
    ulong x, ulong y,
    thread ulong &lo, thread ulong &hi)
{
    uint x0 = (uint)x;
    uint x1 = (uint)(x >> 32u);
    uint y0 = (uint)y;
    uint y1 = (uint)(y >> 32u);

    uint2 p0 = clmul32_u32(x0, y0);
    uint2 p2 = clmul32_u32(x1, y1);
    uint2 pm = clmul32_u32(x0 ^ x1, y0 ^ y1);
    uint2 mid = pm ^ p0 ^ p2;

    lo = ((ulong)p0.x) | (((ulong)(p0.y ^ mid.x)) << 32u);
    hi = ((ulong)(mid.y ^ p2.x)) | (((ulong)p2.y) << 32u);
}

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

    t0 = z0_lo;
    t1 = z0_hi ^ zm_lo ^ z0_lo ^ z2_lo;
    t2 = z2_lo ^ zm_hi ^ z0_hi ^ z2_hi;
    t3 = z2_hi;
}

inline void gcm_reduce64(
    ulong t0, ulong t1, ulong t2, ulong t3,
    thread ulong &r_lo, thread ulong &r_hi)
{
    ulong q = (t3 >> 62u) ^ (t3 >> 57u);

    r_lo = t0
         ^ t2 ^ (t2 << 1u) ^ (t2 << 2u) ^ (t2 << 7u)
         ^ q  ^ (q  << 1u) ^ (q  << 2u) ^ (q  << 7u);

    r_hi = t1
         ^ t3 ^ (t3 << 1u) ^ (t3 << 2u) ^ (t3 << 7u)
         ^ (t2 >> 63u) ^ (t2 >> 62u) ^ (t2 >> 57u);
}

inline void gf128_mul64(
    ulong a_lo, ulong a_hi, ulong b_lo, ulong b_hi,
    thread ulong &c_lo, thread ulong &c_hi)
{
    ulong t0, t1, t2, t3;
    clmul128_unreduced(a_lo, a_hi, b_lo, b_hi, t0, t1, t2, t3);
    gcm_reduce64(t0, t1, t2, t3, c_lo, c_hi);
}

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
            uint r = 64u - s;
            t0 ^= x_lo << s;
            t1 ^= (x_hi << s) | (x_lo >> r);
            t2 ^= x_hi >> r;
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
            uint r = 64u - s;
            t1 ^= x_lo << s;
            t2 ^= (x_hi << s) | (x_lo >> r);
            t3 ^= x_hi >> r;
        }
        mh &= (mh - 1ul);
    }
}

inline void gf128_mul_sparse_multiplier64(
    ulong x_lo, ulong x_hi, ulong m_lo, ulong m_hi,
    thread ulong &c_lo, thread ulong &c_hi)
{
    ulong t0, t1, t2, t3;
    clmul128_unreduced_sparse_multiplier(x_lo, x_hi, m_lo, m_hi, t0, t1, t2, t3);
    gcm_reduce64(t0, t1, t2, t3, c_lo, c_hi);
}

inline void gf128_xtime64(
    ulong x_lo, ulong x_hi,
    thread ulong &r_lo, thread ulong &r_hi)
{
    ulong carry = x_hi >> 63u;
    r_hi = (x_hi << 1u) | (x_lo >> 63u);
    r_lo = (x_lo << 1u) ^ (carry * 0x87ul);
}

inline void gf128_mul_xk64(
    ulong x_lo, ulong x_hi, uint k,
    thread ulong &c_lo, thread ulong &c_hi)
{
    if (k == 0u) {
        c_lo = x_lo;
        c_hi = x_hi;
        return;
    }

    ulong t0 = 0ul;
    ulong t1 = 0ul;
    ulong t2 = 0ul;
    ulong t3 = 0ul;

    if (k < 64u) {
        uint r = 64u - k;
        t0 = x_lo << k;
        t1 = (x_hi << k) | (x_lo >> r);
        t2 = x_hi >> r;
    } else if (k == 64u) {
        t1 = x_lo;
        t2 = x_hi;
    } else {
        uint s = k - 64u;
        uint r = 64u - s;
        t1 = x_lo << s;
        t2 = (x_hi << s) | (x_lo >> r);
        t3 = x_hi >> r;
    }

    gcm_reduce64(t0, t1, t2, t3, c_lo, c_hi);
}

inline void gf128_mul_alpha64(
    ulong x_lo, ulong x_hi,
    ulong alpha_lo_v, ulong alpha_hi_v,
    thread ulong &c_lo, thread ulong &c_hi)
{
    if (alpha_hi_v == 0ul && alpha_lo_v == 0ul) {
        c_lo = 0ul;
        c_hi = 0ul;
        return;
    }

    if (alpha_hi_v == 0ul && alpha_lo_v == 1ul) {
        c_lo = x_lo;
        c_hi = x_hi;
        return;
    }

    if (alpha_hi_v == 0ul && alpha_lo_v == 2ul) {
        gf128_xtime64(x_lo, x_hi, c_lo, c_hi);
        return;
    }

    uint aw = (uint)popcount(alpha_lo_v) + (uint)popcount(alpha_hi_v);

    if (aw == 1u) {
        uint k = (alpha_lo_v != 0ul)
               ? (uint)ctz(alpha_lo_v)
               : (64u + (uint)ctz(alpha_hi_v));
        gf128_mul_xk64(x_lo, x_hi, k, c_lo, c_hi);
    } else if (aw <= 64u) {
        gf128_mul_sparse_multiplier64(x_lo, x_hi, alpha_lo_v, alpha_hi_v, c_lo, c_hi);
    } else {
        gf128_mul64(alpha_lo_v, alpha_hi_v, x_lo, x_hi, c_lo, c_hi);
    }
}

inline uint4 gcm_reduce_u32_words(
    uint l0, uint l1, uint l2, uint l3,
    uint h0, uint h1, uint h2, uint h3)
{
    uint r0 = l0
            ^ h0 ^ (h0 << 1u) ^ (h0 << 2u) ^ (h0 << 7u);

    uint r1 = l1
            ^ h1 ^ (h1 << 1u) ^ (h1 << 2u) ^ (h1 << 7u)
            ^ (h0 >> 31u) ^ (h0 >> 30u) ^ (h0 >> 25u);

    uint r2 = l2
            ^ h2 ^ (h2 << 1u) ^ (h2 << 2u) ^ (h2 << 7u)
            ^ (h1 >> 31u) ^ (h1 >> 30u) ^ (h1 >> 25u);

    uint r3 = l3
            ^ h3 ^ (h3 << 1u) ^ (h3 << 2u) ^ (h3 << 7u)
            ^ (h2 >> 31u) ^ (h2 >> 30u) ^ (h2 >> 25u);

    uint q = (h3 >> 30u) ^ (h3 >> 25u);
    r0 ^= q ^ (q << 1u) ^ (q << 2u) ^ (q << 7u);

    return uint4(r0, r1, r2, r3);
}

inline uint4 gf128_mul_u32_fast(uint4 a, uint4 b)
{
    uint4 z0 = clmul64_u32(a.x, a.y, b.x, b.y);
    uint4 z2 = clmul64_u32(a.z, a.w, b.z, b.w);
    uint4 zm = clmul64_u32(a.x ^ a.z, a.y ^ a.w,
                           b.x ^ b.z, b.y ^ b.w);

    uint l0 = z0.x;
    uint l1 = z0.y;
    uint l2 = z0.z ^ zm.x ^ z0.x ^ z2.x;
    uint l3 = z0.w ^ zm.y ^ z0.y ^ z2.y;

    uint h0 = z2.x ^ zm.z ^ z0.z ^ z2.z;
    uint h1 = z2.y ^ zm.w ^ z0.w ^ z2.w;
    uint h2 = z2.z;
    uint h3 = z2.w;

    return gcm_reduce_u32_words(l0, l1, l2, l3, h0, h1, h2, h3);
}

inline uint4 split_u128(ulong lo, ulong hi)
{
    return uint4((uint)lo,
                 (uint)(lo >> 32u),
                 (uint)hi,
                 (uint)(hi >> 32u));
}

inline ulong pack_u64(uint lo, uint hi)
{
    return ((ulong)lo) | (((ulong)hi) << 32u);
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
    if (idx >= batch) {
        return;
    }

    if (tower == 0u) {
        uint base = idx << 1u;

        ulong a_lo = a[base + 0u];
        ulong a_hi = a[base + 1u];
        ulong b_lo = b[base + 0u];
        ulong b_hi = b[base + 1u];

        if (batch <= 65536u) {
            uint4 cv = gf128_mul_u32_fast(split_u128(a_lo, a_hi),
                                          split_u128(b_lo, b_hi));
            c[base + 0u] = pack_u64(cv.x, cv.y);
            c[base + 1u] = pack_u64(cv.z, cv.w);
        } else {
            ulong c_lo, c_hi;
            gf128_mul64(a_lo, a_hi, b_lo, b_hi, c_lo, c_hi);
            c[base + 0u] = c_lo;
            c[base + 1u] = c_hi;
        }
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

    ulong m00_lo, m00_hi;
    gf128_mul64(a0_lo, a0_hi, b0_lo, b0_hi, m00_lo, m00_hi);

    ulong mix_lo, mix_hi;
    gf128_mul64(a0_lo ^ a1_lo, a0_hi ^ a1_hi,
                b0_lo ^ b1_lo, b0_hi ^ b1_hi,
                mix_lo, mix_hi);

    c[base + 2u] = mix_lo ^ m00_lo;
    c[base + 3u] = mix_hi ^ m00_hi;

    ulong m11_lo, m11_hi;
    gf128_mul64(a1_lo, a1_hi, b1_lo, b1_hi, m11_lo, m11_hi);

    ulong am_lo, am_hi;
    gf128_mul_alpha64(m11_lo, m11_hi, alpha_lo, alpha_hi, am_lo, am_hi);

    c[base + 0u] = m00_lo ^ am_lo;
    c[base + 1u] = m00_hi ^ am_hi;
}
```

Result of previous attempt:
        gf128_N64K: correct, 0.14 ms, 122.4 Gbitops/s (u64) (21.2% of 577 Gops/s (u64 bitop, est))
       gf128_N256K: correct, 0.15 ms, 461.3 Gbitops/s (u64) (79.9% of 577 Gops/s (u64 bitop, est))
         gf128_N1M: correct, 1.28 ms, 209.1 Gbitops/s (u64) (36.2% of 577 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.3944

## History

- iter  0: compile=OK | correct=True | score=0.1041251747903846
- iter  1: compile=OK | correct=True | score=0.22872673973641952
- iter  2: compile=OK | correct=True | score=0.3522878982638455
- iter  3: compile=OK | correct=True | score=0.2677757086188393
- iter  4: compile=OK | correct=True | score=0.2705720037921986
- iter  5: compile=OK | correct=True | score=0.3851904345393261
- iter  6: compile=OK | correct=True | score=0.39435665114649837

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
