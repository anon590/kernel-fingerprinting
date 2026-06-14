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

// 64x64 -> 128 carry-less multiply using 4-bit window with register-resident table.
// Table entries are kept as named locals to avoid indirect indexing (which spills).
inline void clmul64(ulong a, ulong b, thread ulong &lo, thread ulong &hi)
{
    // Build T[k] = a * k in GF(2)[x], k = 0..15, as (tl_k, th_k).
    ulong tl1 = a,            th1 = 0ul;
    ulong tl2 = a << 1,       th2 = a >> 63;
    ulong tl4 = a << 2,       th4 = a >> 62;
    ulong tl8 = a << 3,       th8 = a >> 61;
    ulong tl3  = tl1 ^ tl2,   th3  = th1 ^ th2;
    ulong tl5  = tl1 ^ tl4,   th5  = th1 ^ th4;
    ulong tl6  = tl2 ^ tl4,   th6  = th2 ^ th4;
    ulong tl7  = tl3 ^ tl4,   th7  = th3 ^ th4;
    ulong tl9  = tl1 ^ tl8,   th9  = th1 ^ th8;
    ulong tl10 = tl2 ^ tl8,   th10 = th2 ^ th8;
    ulong tl11 = tl3 ^ tl8,   th11 = th3 ^ th8;
    ulong tl12 = tl4 ^ tl8,   th12 = th4 ^ th8;
    ulong tl13 = tl5 ^ tl8,   th13 = th5 ^ th8;
    ulong tl14 = tl6 ^ tl8,   th14 = th6 ^ th8;
    ulong tl15 = tl7 ^ tl8,   th15 = th7 ^ th8;

    ulong rl = 0ul, rh = 0ul;

    // MSB-first scan, 16 nibbles. Use branchless select-by-k via a fully unrolled
    // chain: build sel via bit-tests on k. This keeps everything in registers.
    #pragma clang loop unroll(full)
    for (int s = 60; s >= 0; s -= 4) {
        ulong nh = (rh << 4) | (rl >> 60);
        ulong nl = (rl << 4);
        uint k = (uint)((b >> s) & 0xFul);

        // Select (tl_k, th_k) via mask-based reduction. Branchless: build masks
        // from equality tests, OR-reduced (since masks are disjoint).
        ulong sel_l = 0ul, sel_h = 0ul;
        ulong m;
        m = (ulong)-(long)(k == 1u);  sel_l |= m & tl1;  sel_h |= m & th1;
        m = (ulong)-(long)(k == 2u);  sel_l |= m & tl2;  sel_h |= m & th2;
        m = (ulong)-(long)(k == 3u);  sel_l |= m & tl3;  sel_h |= m & th3;
        m = (ulong)-(long)(k == 4u);  sel_l |= m & tl4;  sel_h |= m & th4;
        m = (ulong)-(long)(k == 5u);  sel_l |= m & tl5;  sel_h |= m & th5;
        m = (ulong)-(long)(k == 6u);  sel_l |= m & tl6;  sel_h |= m & th6;
        m = (ulong)-(long)(k == 7u);  sel_l |= m & tl7;  sel_h |= m & th7;
        m = (ulong)-(long)(k == 8u);  sel_l |= m & tl8;  sel_h |= m & th8;
        m = (ulong)-(long)(k == 9u);  sel_l |= m & tl9;  sel_h |= m & th9;
        m = (ulong)-(long)(k == 10u); sel_l |= m & tl10; sel_h |= m & th10;
        m = (ulong)-(long)(k == 11u); sel_l |= m & tl11; sel_h |= m & th11;
        m = (ulong)-(long)(k == 12u); sel_l |= m & tl12; sel_h |= m & th12;
        m = (ulong)-(long)(k == 13u); sel_l |= m & tl13; sel_h |= m & th13;
        m = (ulong)-(long)(k == 14u); sel_l |= m & tl14; sel_h |= m & th14;
        m = (ulong)-(long)(k == 15u); sel_l |= m & tl15; sel_h |= m & th15;

        rl = nl ^ sel_l;
        rh = nh ^ sel_h;
    }
    lo = rl;
    hi = rh;
}

inline void clmul128_unreduced(
    ulong a_lo, ulong a_hi, ulong b_lo, ulong b_hi,
    thread ulong &t0, thread ulong &t1,
    thread ulong &t2, thread ulong &t3)
{
    ulong p0l, p0h; clmul64(a_lo, b_lo, p0l, p0h);
    ulong p2l, p2h; clmul64(a_hi, b_hi, p2l, p2h);
    ulong pml, pmh; clmul64(a_lo ^ a_hi, b_lo ^ b_hi, pml, pmh);
    ulong p1l = pml ^ p0l ^ p2l;
    ulong p1h = pmh ^ p0h ^ p2h;

    t0 = p0l;
    t1 = p0h ^ p1l;
    t2 = p2l ^ p1h;
    t3 = p2h;
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

        // Karatsuba in v: m11 = a1*b1, m00 = a0*b0, mid = (a0+a1)*(b0+b1).
        // c0 = m00 + alpha*m11, c1 = mid + m00 (since mid - m00 - m11 + m11 = mid - m00, XOR same).
        // Wait: c1 = a0 b1 + a1 b0 + a1 b1 = mid - m00 - m11 + m11 = mid - m00 = mid XOR m00.
        ulong m00_lo, m00_hi; gf128_mul(a0_lo, a0_hi, b0_lo, b0_hi, m00_lo, m00_hi);
        ulong m11_lo, m11_hi; gf128_mul(a1_lo, a1_hi, b1_lo, b1_hi, m11_lo, m11_hi);
        ulong mid_lo, mid_hi; gf128_mul(a0_lo ^ a1_lo, a0_hi ^ a1_hi,
                                        b0_lo ^ b1_lo, b0_hi ^ b1_hi,
                                        mid_lo, mid_hi);

        ulong am_lo, am_hi;
        gf128_mul(alpha_lo, alpha_hi, m11_lo, m11_hi, am_lo, am_hi);

        c[base + 0] = m00_lo ^ am_lo;
        c[base + 1] = m00_hi ^ am_hi;
        c[base + 2] = mid_lo ^ m00_lo;
        c[base + 3] = mid_hi ^ m00_hi;
    }
}
```

Result of previous attempt:
        gf128_N64K: correct, 6.38 ms, 2.6 Gbitops/s (u64) (0.5% of 577 Gops/s (u64 bitop, est))
       gf128_N256K: correct, 22.36 ms, 3.0 Gbitops/s (u64) (0.5% of 577 Gops/s (u64 bitop, est))
         gf128_N1M: correct, 81.10 ms, 3.3 Gbitops/s (u64) (0.6% of 577 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.0051

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

// 64 x 64 -> 128-bit carry-less multiply via 4-bit windowed scan.
// Builds a 16-entry table of (a * k) for k=0..15 as 128-bit pairs,
// then scans b's 16 nibbles from MSB to LSB.
inline void clmul64(ulong a, ulong b, thread ulong &lo, thread ulong &hi)
{
    // Table T[k] = a * k in GF(2)[x], 128-bit (tl, th).
    ulong tl[16], th[16];
    tl[0] = 0ul;            th[0] = 0ul;
    tl[1] = a;              th[1] = 0ul;
    tl[2] = a << 1;         th[2] = a >> 63;
    tl[4] = a << 2;         th[4] = a >> 62;
    tl[8] = a << 3;         th[8] = a >> 61;
    tl[3]  = tl[1]^tl[2];   th[3]  = th[1]^th[2];
    tl[5]  = tl[1]^tl[4];   th[5]  = th[1]^th[4];
    tl[6]  = tl[2]^tl[4];   th[6]  = th[2]^th[4];
    tl[7]  = tl[3]^tl[4];   th[7]  = th[3]^th[4];
    tl[9]  = tl[1]^tl[8];   th[9]  = th[1]^th[8];
    tl[10] = tl[2]^tl[8];   th[10] = th[2]^th[8];
    tl[11] = tl[3]^tl[8];   th[11] = th[3]^th[8];
    tl[12] = tl[4]^tl[8];   th[12] = th[4]^th[8];
    tl[13] = tl[5]^tl[8];   th[13] = th[5]^th[8];
    tl[14] = tl[6]^tl[8];   th[14] = th[6]^th[8];
    tl[15] = tl[7]^tl[8];   th[15] = th[7]^th[8];

    ulong rl = 0ul, rh = 0ul;
    // MSB-first scan: 16 nibbles from bit 60 down to bit 0.
    #pragma clang loop unroll(full)
    for (int s = 60; s >= 0; s -= 4) {
        // shift result left by 4
        ulong nh = (rh << 4) | (rl >> 60);
        ulong nl = (rl << 4);
        uint k = (uint)((b >> s) & 0xFul);
        rl = nl ^ tl[k];
        rh = nh ^ th[k];
    }
    lo = rl;
    hi = rh;
}

// 128 x 128 -> 256-bit unreduced clmul via Karatsuba on 64-bit halves.
// a = a_lo + a_hi * x^64; b = b_lo + b_hi * x^64.
// Let P0 = a_lo*b_lo, P2 = a_hi*b_hi, P1 = (a_lo^a_hi)*(b_lo^b_hi) ^ P0 ^ P2.
// Result = P0 + P1*x^64 + P2*x^128.
inline void clmul128_unreduced(
    ulong a_lo, ulong a_hi, ulong b_lo, ulong b_hi,
    thread ulong &t0, thread ulong &t1,
    thread ulong &t2, thread ulong &t3)
{
    ulong p0l, p0h; clmul64(a_lo, b_lo, p0l, p0h);
    ulong p2l, p2h; clmul64(a_hi, b_hi, p2l, p2h);
    ulong pml, pmh; clmul64(a_lo ^ a_hi, b_lo ^ b_hi, pml, pmh);
    ulong p1l = pml ^ p0l ^ p2l;
    ulong p1h = pmh ^ p0h ^ p2h;

    t0 = p0l;
    t1 = p0h ^ p1l;
    t2 = p2l ^ p1h;
    t3 = p2h;
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

        c[base + 0] = m00_lo ^ am_lo;
        c[base + 1] = m00_hi ^ am_hi;
        c[base + 2] = m01_lo ^ m10_lo ^ m11_lo;
        c[base + 3] = m01_hi ^ m10_hi ^ m11_hi;
    }
}
```

Incumbent result:
        gf128_N64K: correct, 0.17 ms, 101.3 Gbitops/s (u64) (17.6% of 577 Gops/s (u64 bitop, est))
       gf128_N256K: correct, 0.57 ms, 118.5 Gbitops/s (u64) (20.5% of 577 Gops/s (u64 bitop, est))
         gf128_N1M: correct, 2.20 ms, 122.3 Gbitops/s (u64) (21.2% of 577 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.1968

## History

- iter  0: compile=OK | correct=True | score=0.09374361344761027
- iter  1: compile=OK | correct=True | score=0.13334647948496645
- iter  2: compile=OK | correct=True | score=0.015705774360255838
- iter  3: compile=OK | correct=True | score=0.06747492095709441
- iter  4: compile=OK | correct=False | score=N/A
- iter  5: compile=OK | correct=True | score=0.1968326329985548
- iter  6: compile=OK | correct=True | score=0.005137511647774921

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
