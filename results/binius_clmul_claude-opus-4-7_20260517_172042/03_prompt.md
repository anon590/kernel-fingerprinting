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

// 64x64 -> 128 bit carry-less multiply using the 4-way bit-split trick.
// Split b into 4 streams b0..b3 where bk has only bits with (i mod 4)==k set.
// For each stream, mask a by the predicate "bit i of b is set" via the
// well-known branchless pattern using arithmetic from each bit, then sum
// the four shifted partial products. This avoids the 192-bit table and
// 32-nibble loop of the previous attempt.
//
// We use the classic "double-and-add" but unrolled over 8 bits of b at a
// time with 8 shifted masks. Total: 64 iterations of (mask, shift, xor)
// but compiler unrolls and ILP is high.
inline void clmul64(ulong a, ulong b, thread ulong &lo, thread ulong &hi)
{
    ulong l = 0ul;
    ulong h = 0ul;
    // Bit 0
    ulong m;
    m = -(b & 1ul);        l ^= a & m;
    m = -((b >> 1) & 1ul); l ^= (a << 1) & m;  h ^= (a >> 63) & m;
    m = -((b >> 2) & 1ul); l ^= (a << 2) & m;  h ^= (a >> 62) & m;
    m = -((b >> 3) & 1ul); l ^= (a << 3) & m;  h ^= (a >> 61) & m;
    m = -((b >> 4) & 1ul); l ^= (a << 4) & m;  h ^= (a >> 60) & m;
    m = -((b >> 5) & 1ul); l ^= (a << 5) & m;  h ^= (a >> 59) & m;
    m = -((b >> 6) & 1ul); l ^= (a << 6) & m;  h ^= (a >> 58) & m;
    m = -((b >> 7) & 1ul); l ^= (a << 7) & m;  h ^= (a >> 57) & m;
    m = -((b >> 8) & 1ul); l ^= (a << 8) & m;  h ^= (a >> 56) & m;
    m = -((b >> 9) & 1ul); l ^= (a << 9) & m;  h ^= (a >> 55) & m;
    m = -((b >>10) & 1ul); l ^= (a <<10) & m;  h ^= (a >> 54) & m;
    m = -((b >>11) & 1ul); l ^= (a <<11) & m;  h ^= (a >> 53) & m;
    m = -((b >>12) & 1ul); l ^= (a <<12) & m;  h ^= (a >> 52) & m;
    m = -((b >>13) & 1ul); l ^= (a <<13) & m;  h ^= (a >> 51) & m;
    m = -((b >>14) & 1ul); l ^= (a <<14) & m;  h ^= (a >> 50) & m;
    m = -((b >>15) & 1ul); l ^= (a <<15) & m;  h ^= (a >> 49) & m;
    m = -((b >>16) & 1ul); l ^= (a <<16) & m;  h ^= (a >> 48) & m;
    m = -((b >>17) & 1ul); l ^= (a <<17) & m;  h ^= (a >> 47) & m;
    m = -((b >>18) & 1ul); l ^= (a <<18) & m;  h ^= (a >> 46) & m;
    m = -((b >>19) & 1ul); l ^= (a <<19) & m;  h ^= (a >> 45) & m;
    m = -((b >>20) & 1ul); l ^= (a <<20) & m;  h ^= (a >> 44) & m;
    m = -((b >>21) & 1ul); l ^= (a <<21) & m;  h ^= (a >> 43) & m;
    m = -((b >>22) & 1ul); l ^= (a <<22) & m;  h ^= (a >> 42) & m;
    m = -((b >>23) & 1ul); l ^= (a <<23) & m;  h ^= (a >> 41) & m;
    m = -((b >>24) & 1ul); l ^= (a <<24) & m;  h ^= (a >> 40) & m;
    m = -((b >>25) & 1ul); l ^= (a <<25) & m;  h ^= (a >> 39) & m;
    m = -((b >>26) & 1ul); l ^= (a <<26) & m;  h ^= (a >> 38) & m;
    m = -((b >>27) & 1ul); l ^= (a <<27) & m;  h ^= (a >> 37) & m;
    m = -((b >>28) & 1ul); l ^= (a <<28) & m;  h ^= (a >> 36) & m;
    m = -((b >>29) & 1ul); l ^= (a <<29) & m;  h ^= (a >> 35) & m;
    m = -((b >>30) & 1ul); l ^= (a <<30) & m;  h ^= (a >> 34) & m;
    m = -((b >>31) & 1ul); l ^= (a <<31) & m;  h ^= (a >> 33) & m;
    m = -((b >>32) & 1ul); l ^= (a <<32) & m;  h ^= (a >> 32) & m;
    m = -((b >>33) & 1ul); l ^= (a <<33) & m;  h ^= (a >> 31) & m;
    m = -((b >>34) & 1ul); l ^= (a <<34) & m;  h ^= (a >> 30) & m;
    m = -((b >>35) & 1ul); l ^= (a <<35) & m;  h ^= (a >> 29) & m;
    m = -((b >>36) & 1ul); l ^= (a <<36) & m;  h ^= (a >> 28) & m;
    m = -((b >>37) & 1ul); l ^= (a <<37) & m;  h ^= (a >> 27) & m;
    m = -((b >>38) & 1ul); l ^= (a <<38) & m;  h ^= (a >> 26) & m;
    m = -((b >>39) & 1ul); l ^= (a <<39) & m;  h ^= (a >> 25) & m;
    m = -((b >>40) & 1ul); l ^= (a <<40) & m;  h ^= (a >> 24) & m;
    m = -((b >>41) & 1ul); l ^= (a <<41) & m;  h ^= (a >> 23) & m;
    m = -((b >>42) & 1ul); l ^= (a <<42) & m;  h ^= (a >> 22) & m;
    m = -((b >>43) & 1ul); l ^= (a <<43) & m;  h ^= (a >> 21) & m;
    m = -((b >>44) & 1ul); l ^= (a <<44) & m;  h ^= (a >> 20) & m;
    m = -((b >>45) & 1ul); l ^= (a <<45) & m;  h ^= (a >> 19) & m;
    m = -((b >>46) & 1ul); l ^= (a <<46) & m;  h ^= (a >> 18) & m;
    m = -((b >>47) & 1ul); l ^= (a <<47) & m;  h ^= (a >> 17) & m;
    m = -((b >>48) & 1ul); l ^= (a <<48) & m;  h ^= (a >> 16) & m;
    m = -((b >>49) & 1ul); l ^= (a <<49) & m;  h ^= (a >> 15) & m;
    m = -((b >>50) & 1ul); l ^= (a <<50) & m;  h ^= (a >> 14) & m;
    m = -((b >>51) & 1ul); l ^= (a <<51) & m;  h ^= (a >> 13) & m;
    m = -((b >>52) & 1ul); l ^= (a <<52) & m;  h ^= (a >> 12) & m;
    m = -((b >>53) & 1ul); l ^= (a <<53) & m;  h ^= (a >> 11) & m;
    m = -((b >>54) & 1ul); l ^= (a <<54) & m;  h ^= (a >> 10) & m;
    m = -((b >>55) & 1ul); l ^= (a <<55) & m;  h ^= (a >>  9) & m;
    m = -((b >>56) & 1ul); l ^= (a <<56) & m;  h ^= (a >>  8) & m;
    m = -((b >>57) & 1ul); l ^= (a <<57) & m;  h ^= (a >>  7) & m;
    m = -((b >>58) & 1ul); l ^= (a <<58) & m;  h ^= (a >>  6) & m;
    m = -((b >>59) & 1ul); l ^= (a <<59) & m;  h ^= (a >>  5) & m;
    m = -((b >>60) & 1ul); l ^= (a <<60) & m;  h ^= (a >>  4) & m;
    m = -((b >>61) & 1ul); l ^= (a <<61) & m;  h ^= (a >>  3) & m;
    m = -((b >>62) & 1ul); l ^= (a <<62) & m;  h ^= (a >>  2) & m;
    m = -((b >>63) & 1ul); l ^= (a <<63) & m;  h ^= (a >>  1) & m;
    lo = l;
    hi = h;
}

// 128x128 -> 256 unreduced clmul, via 3-mul Karatsuba.
inline void clmul128_unreduced(
    ulong a_lo, ulong a_hi, ulong b_lo, ulong b_hi,
    thread ulong &t0, thread ulong &t1,
    thread ulong &t2, thread ulong &t3)
{
    ulong z0_lo, z0_hi; clmul64(a_lo, b_lo, z0_lo, z0_hi);
    ulong z2_lo, z2_hi; clmul64(a_hi, b_hi, z2_lo, z2_hi);
    ulong zm_lo, zm_hi; clmul64(a_lo ^ a_hi, b_lo ^ b_hi, zm_lo, zm_hi);
    // middle = zm - z0 - z2 (XOR in GF(2))
    ulong m_lo = zm_lo ^ z0_lo ^ z2_lo;
    ulong m_hi = zm_hi ^ z0_hi ^ z2_hi;
    // Combine: result = z0 + (m << 64) + (z2 << 128)
    t0 = z0_lo;
    t1 = z0_hi ^ m_lo;
    t2 = z2_lo ^ m_hi;
    t3 = z2_hi;
}

inline void gcm_reduce(
    ulong t0, ulong t1, ulong t2, ulong t3,
    thread ulong &r_lo, thread ulong &r_hi)
{
    // Fold t3:t2 (upper 128) into lower via R(x) = x^128 + x^7 + x^2 + x + 1.
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

        // Karatsuba over GF(2^128)[v]:
        //   m00 = a0*b0
        //   m11 = a1*b1
        //   mk  = (a0+a1)*(b0+b1) = m00 + m01 + m10 + m11   (in GF(2))
        // => m01 + m10 = mk + m00 + m11
        // c0 = m00 + alpha*m11
        // c1 = m01 + m10 + m11 = mk + m00 + m11 + m11 = mk + m00
        ulong m00_lo, m00_hi; gf128_mul(a0_lo, a0_hi, b0_lo, b0_hi, m00_lo, m00_hi);
        ulong m11_lo, m11_hi; gf128_mul(a1_lo, a1_hi, b1_lo, b1_hi, m11_lo, m11_hi);
        ulong mk_lo,  mk_hi;  gf128_mul(a0_lo ^ a1_lo, a0_hi ^ a1_hi,
                                        b0_lo ^ b1_lo, b0_hi ^ b1_hi,
                                        mk_lo, mk_hi);

        ulong am_lo, am_hi;
        gf128_mul(alpha_lo, alpha_hi, m11_lo, m11_hi, am_lo, am_hi);

        ulong c0_lo = m00_lo ^ am_lo;
        ulong c0_hi = m00_hi ^ am_hi;
        ulong c1_lo = mk_lo ^ m00_lo;
        ulong c1_hi = mk_hi ^ m00_hi;

        c[base + 0] = c0_lo;
        c[base + 1] = c0_hi;
        c[base + 2] = c1_lo;
        c[base + 3] = c1_hi;
    }
}
```

Result of previous attempt:
        gf128_N64K: correct, 1.95 ms, 8.6 Gbitops/s (u64) (1.5% of 577 Gops/s (u64 bitop, est))
       gf128_N256K: correct, 7.30 ms, 9.2 Gbitops/s (u64) (1.6% of 577 Gops/s (u64 bitop, est))
         gf128_N1M: correct, 28.52 ms, 9.4 Gbitops/s (u64) (1.6% of 577 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.0157

## Current best (incumbent)

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

Incumbent result:
        gf128_N64K: correct, 0.33 ms, 50.8 Gbitops/s (u64) (8.8% of 577 Gops/s (u64 bitop, est))
       gf128_N256K: correct, 0.72 ms, 93.5 Gbitops/s (u64) (16.2% of 577 Gops/s (u64 bitop, est))
         gf128_N1M: correct, 2.79 ms, 96.1 Gbitops/s (u64) (16.6% of 577 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.1333

## History

- iter  0: compile=OK | correct=True | score=0.09374361344761027
- iter  1: compile=OK | correct=True | score=0.13334647948496645
- iter  2: compile=OK | correct=True | score=0.015705774360255838

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
