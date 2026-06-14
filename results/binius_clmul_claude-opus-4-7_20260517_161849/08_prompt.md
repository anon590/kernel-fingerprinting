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

// 64x64 -> 128 carry-less multiply via per-bit accumulation.
// For each bit i of b, conditionally XOR (a << i) into the 128-bit result.
// All shifts are compile-time constants (fully unrolled), no tables, no branches.
inline void clmul64(ulong a, ulong b, thread ulong &lo, thread ulong &hi)
{
    ulong rl = 0ul;
    ulong rh = 0ul;

    #define STEP(I) {                                                    \
        ulong m = (ulong)0 - ((b >> (I)) & 1ul);                         \
        rl ^= (a << (I)) & m;                                            \
        rh ^= (((I) == 0 ? 0ul : (a >> (64 - (I)))) ) & m;               \
    }
    // i = 0 separately (no high contribution)
    {
        ulong m = (ulong)0 - (b & 1ul);
        rl ^= a & m;
    }
    STEP( 1) STEP( 2) STEP( 3) STEP( 4) STEP( 5) STEP( 6) STEP( 7)
    STEP( 8) STEP( 9) STEP(10) STEP(11) STEP(12) STEP(13) STEP(14) STEP(15)
    STEP(16) STEP(17) STEP(18) STEP(19) STEP(20) STEP(21) STEP(22) STEP(23)
    STEP(24) STEP(25) STEP(26) STEP(27) STEP(28) STEP(29) STEP(30) STEP(31)
    STEP(32) STEP(33) STEP(34) STEP(35) STEP(36) STEP(37) STEP(38) STEP(39)
    STEP(40) STEP(41) STEP(42) STEP(43) STEP(44) STEP(45) STEP(46) STEP(47)
    STEP(48) STEP(49) STEP(50) STEP(51) STEP(52) STEP(53) STEP(54) STEP(55)
    STEP(56) STEP(57) STEP(58) STEP(59) STEP(60) STEP(61) STEP(62) STEP(63)
    #undef STEP

    lo = rl;
    hi = rh;
}

// 128 x 128 -> 256 carry-less multiply via Karatsuba on 64-bit halves.
inline void clmul128_unreduced(
    ulong a_lo, ulong a_hi, ulong b_lo, ulong b_hi,
    thread ulong &t0, thread ulong &t1,
    thread ulong &t2, thread ulong &t3)
{
    ulong p0_lo, p0_hi;
    ulong p1_lo, p1_hi;
    ulong pm_lo, pm_hi;
    clmul64(a_lo, b_lo, p0_lo, p0_hi);
    clmul64(a_hi, b_hi, p1_lo, p1_hi);
    clmul64(a_lo ^ a_hi, b_lo ^ b_hi, pm_lo, pm_hi);

    ulong mid_lo = pm_lo ^ p0_lo ^ p1_lo;
    ulong mid_hi = pm_hi ^ p0_hi ^ p1_hi;

    t0 = p0_lo;
    t1 = p0_hi ^ mid_lo;
    t2 = p1_lo ^ mid_hi;
    t3 = p1_hi;
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
        ulong m11_lo, m11_hi; gf128_mul(a1_lo, a1_hi, b1_lo, b1_hi, m11_lo, m11_hi);
        ulong msum_lo, msum_hi;
        gf128_mul(a0_lo ^ a1_lo, a0_hi ^ a1_hi,
                  b0_lo ^ b1_lo, b0_hi ^ b1_hi,
                  msum_lo, msum_hi);

        ulong am_lo, am_hi;
        gf128_mul(alpha_lo, alpha_hi, m11_lo, m11_hi, am_lo, am_hi);

        ulong c0_lo = m00_lo ^ am_lo;
        ulong c0_hi = m00_hi ^ am_hi;
        ulong c1_lo = msum_lo ^ m00_lo;
        ulong c1_hi = msum_hi ^ m00_hi;

        c[base + 0] = c0_lo;
        c[base + 1] = c0_hi;
        c[base + 2] = c1_lo;
        c[base + 3] = c1_hi;
    }
}
```

Result of previous attempt:
        gf128_N64K: correct, 2.46 ms, 6.8 Gbitops/s (u64) (1.2% of 577 Gops/s (u64 bitop, est))
       gf128_N256K: correct, 8.94 ms, 7.5 Gbitops/s (u64) (1.3% of 577 Gops/s (u64 bitop, est))
         gf128_N1M: correct, 30.94 ms, 8.7 Gbitops/s (u64) (1.5% of 577 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.0132

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

// Carry-less multiply 64x64 -> 128 using nibble window (16-entry table built on the fly,
// but kept fully unrolled so the compiler can keep table entries in registers).
inline void clmul64(ulong a, ulong b, thread ulong &lo, thread ulong &hi)
{
    // Build 16-entry table T[k] = (k as 4-bit poly) clmul a, stored as 128-bit (Tl, Th).
    ulong T0l = 0ul,            T0h = 0ul;
    ulong T1l = a,              T1h = 0ul;
    ulong T2l = a << 1,         T2h = a >> 63;
    ulong T3l = T2l ^ T1l,      T3h = T2h ^ T1h;
    ulong T4l = a << 2,         T4h = a >> 62;
    ulong T5l = T4l ^ T1l,      T5h = T4h ^ T1h;
    ulong T6l = T4l ^ T2l,      T6h = T4h ^ T2h;
    ulong T7l = T4l ^ T3l,      T7h = T4h ^ T3h;
    ulong T8l = a << 3,         T8h = a >> 61;
    ulong T9l  = T8l ^ T1l,     T9h  = T8h ^ T1h;
    ulong TAl  = T8l ^ T2l,     TAh  = T8h ^ T2h;
    ulong TBl  = T8l ^ T3l,     TBh  = T8h ^ T3h;
    ulong TCl  = T8l ^ T4l,     TCh  = T8h ^ T4h;
    ulong TDl  = T8l ^ T5l,     TDh  = T8h ^ T5h;
    ulong TEl  = T8l ^ T6l,     TEh  = T8h ^ T6h;
    ulong TFl  = T8l ^ T7l,     TFh  = T8h ^ T7h;

    // Helper macro: select table entry by 4-bit index using a chain of selects.
    // Each pick(n) returns (Tl, Th) for n in 0..15.
    ulong rl = 0ul, rh = 0ul;

    // Process 16 nibbles of b. For nibble i (4*i shift), add T[nib] << (4*i).
    // We compute by accumulating into (rl, rh).
    #define PICK_L(n) ( (n)==0 ? T0l : (n)==1 ? T1l : (n)==2 ? T2l : (n)==3 ? T3l : \
                       (n)==4 ? T4l : (n)==5 ? T5l : (n)==6 ? T6l : (n)==7 ? T7l : \
                       (n)==8 ? T8l : (n)==9 ? T9l : (n)==10? TAl : (n)==11? TBl : \
                       (n)==12? TCl : (n)==13? TDl : (n)==14? TEl : TFl )
    #define PICK_H(n) ( (n)==0 ? T0h : (n)==1 ? T1h : (n)==2 ? T2h : (n)==3 ? T3h : \
                       (n)==4 ? T4h : (n)==5 ? T5h : (n)==6 ? T6h : (n)==7 ? T7h : \
                       (n)==8 ? T8h : (n)==9 ? T9h : (n)==10? TAh : (n)==11? TBh : \
                       (n)==12? TCh : (n)==13? TDh : (n)==14? TEh : TFh )

    // Use indexed array lookup (compiler will keep in registers since fully unrolled init).
    ulong Tl[16] = {T0l,T1l,T2l,T3l,T4l,T5l,T6l,T7l,T8l,T9l,TAl,TBl,TCl,TDl,TEl,TFl};
    ulong Th[16] = {T0h,T1h,T2h,T3h,T4h,T5h,T6h,T7h,T8h,T9h,TAh,TBh,TCh,TDh,TEh,TFh};

    // i=0
    {
        uint n = (uint)(b & 0xFul);
        rl ^= Tl[n];
        rh ^= Th[n];
    }
    // i=1..15
    for (uint i = 1u; i < 16u; ++i) {
        uint sh = i * 4u;
        uint nib = (uint)((b >> sh) & 0xFul);
        ulong el = Tl[nib];
        ulong eh = Th[nib];
        rl ^= el << sh;
        rh ^= (eh << sh) | (el >> (64u - sh));
    }

    lo = rl;
    hi = rh;
    #undef PICK_L
    #undef PICK_H
}

// 128 x 128 -> 256 carry-less multiply via Karatsuba on 64-bit halves.
inline void clmul128_unreduced(
    ulong a_lo, ulong a_hi, ulong b_lo, ulong b_hi,
    thread ulong &t0, thread ulong &t1,
    thread ulong &t2, thread ulong &t3)
{
    ulong p0_lo, p0_hi;  // a_lo * b_lo
    ulong p1_lo, p1_hi;  // a_hi * b_hi
    ulong pm_lo, pm_hi;  // (a_lo^a_hi)*(b_lo^b_hi)
    clmul64(a_lo, b_lo, p0_lo, p0_hi);
    clmul64(a_hi, b_hi, p1_lo, p1_hi);
    clmul64(a_lo ^ a_hi, b_lo ^ b_hi, pm_lo, pm_hi);

    // mid = pm ^ p0 ^ p1; result = p0 + (mid << 64) + (p1 << 128)
    ulong mid_lo = pm_lo ^ p0_lo ^ p1_lo;
    ulong mid_hi = pm_hi ^ p0_hi ^ p1_hi;

    t0 = p0_lo;
    t1 = p0_hi ^ mid_lo;
    t2 = p1_lo ^ mid_hi;
    t3 = p1_hi;
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
        ulong m11_lo, m11_hi; gf128_mul(a1_lo, a1_hi, b1_lo, b1_hi, m11_lo, m11_hi);
        // Karatsuba for c_1: (a0+a1)(b0+b1) = a0b0 + a0b1 + a1b0 + a1b1
        // so a0b1 + a1b0 = (a0+a1)(b0+b1) - a0b0 - a1b1 = (sum)(sum) ^ m00 ^ m11
        ulong msum_lo, msum_hi;
        gf128_mul(a0_lo ^ a1_lo, a0_hi ^ a1_hi,
                  b0_lo ^ b1_lo, b0_hi ^ b1_hi,
                  msum_lo, msum_hi);

        ulong am_lo, am_hi;
        gf128_mul(alpha_lo, alpha_hi, m11_lo, m11_hi, am_lo, am_hi);

        ulong c0_lo = m00_lo ^ am_lo;
        ulong c0_hi = m00_hi ^ am_hi;
        // c1 = a0b1 + a1b0 + a1b1 = (msum ^ m00 ^ m11) ^ m11 = msum ^ m00
        ulong c1_lo = msum_lo ^ m00_lo;
        ulong c1_hi = msum_hi ^ m00_hi;

        c[base + 0] = c0_lo;
        c[base + 1] = c0_hi;
        c[base + 2] = c1_lo;
        c[base + 3] = c1_hi;
    }
}
```

Incumbent result:
        gf128_N64K: correct, 0.20 ms, 82.9 Gbitops/s (u64) (14.4% of 577 Gops/s (u64 bitop, est))
       gf128_N256K: correct, 0.97 ms, 69.4 Gbitops/s (u64) (12.0% of 577 Gops/s (u64 bitop, est))
         gf128_N1M: correct, 2.97 ms, 90.4 Gbitops/s (u64) (15.7% of 577 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.1393

## History

- iter  0: compile=OK | correct=True | score=0.08919958327804488
- iter  1: compile=OK | correct=True | score=0.08109579020171731
- iter  2: compile=OK | correct=True | score=0.13928166972625777
- iter  3: compile=OK | correct=True | score=0.044736133327853665
- iter  4: compile=OK | correct=True | score=0.022444094328414148
- iter  5: compile=OK | correct=True | score=0.013731018790946119
- iter  6: compile=OK | correct=True | score=0.06086299418380245
- iter  7: compile=OK | correct=True | score=0.013208865728447476

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
