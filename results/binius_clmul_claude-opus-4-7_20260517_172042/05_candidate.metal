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