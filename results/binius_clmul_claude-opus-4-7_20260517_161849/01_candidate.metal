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