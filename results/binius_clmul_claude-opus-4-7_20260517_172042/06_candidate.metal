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