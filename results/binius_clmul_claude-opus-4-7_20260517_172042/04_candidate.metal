#include <metal_stdlib>
using namespace metal;

// 128x128 -> 256 bit unreduced clmul.
// Strategy: process b 8 bits at a time. Maintain a "current shifted a" as
// a 192-bit value (a_lo, a_hi, a_top). Each outer step:
//   - for j = 0..7: if bit j of current b-byte set, XOR (a << j) into acc.
//   - then shift the running a by 8 to prepare for next byte.
// All 16 outer iterations and inner 8 steps are unrolled by the compiler.
inline void clmul128_unreduced(
    ulong A0, ulong A1, ulong B0, ulong B1,
    thread ulong &t0, thread ulong &t1,
    thread ulong &t2, thread ulong &t3)
{
    // Running 192-bit "a << (8*byte_index)" represented as (s0,s1,s2).
    ulong s0 = A0;
    ulong s1 = A1;
    ulong s2 = 0ul;

    // 256-bit accumulator (r0 low .. r3 high)
    ulong r0 = 0ul, r1 = 0ul, r2 = 0ul, r3 = 0ul;

    // We need b broken into 16 bytes spanning B0 (low 8 bytes) then B1 (high).
    // Each byte contributes (a << (8*byte)) * byte_value in GF(2)[x].
    // Within a byte, for each bit j, XOR (s << j) into accumulator masked
    // by bit j of the byte.
    //
    // Implement as: for byte=0..15, for j=0..7, mask = -((byte_val>>j)&1).
    // Use a single mask-based XOR per bit, no branches.

    #pragma clang loop unroll(full)
    for (int byte_idx = 0; byte_idx < 16; ++byte_idx) {
        ulong bval = (byte_idx < 8)
            ? ((B0 >> (byte_idx * 8)) & 0xFFul)
            : ((B1 >> ((byte_idx - 8) * 8)) & 0xFFul);

        // Precompute 8 shifted copies of s = (s0,s1,s2) by 0..7.
        // s_j = (s << j) as 256-bit. We only need the low 256 bits.
        // shifted_lo[j..] :
        // j=0 -> (s0, s1, s2, 0)
        // j>0 -> (s0<<j, (s1<<j)|(s0>>(64-j)), (s2<<j)|(s1>>(64-j)), s2>>(64-j))

        // Process each bit; build mask and XOR in.
        #pragma clang loop unroll(full)
        for (int j = 0; j < 8; ++j) {
            ulong mask = (ulong)0 - ((bval >> j) & 1ul);
            ulong x0, x1, x2, x3;
            if (j == 0) {
                x0 = s0; x1 = s1; x2 = s2; x3 = 0ul;
            } else {
                int rj = 64 - j;
                x0 = s0 << j;
                x1 = (s1 << j) | (s0 >> rj);
                x2 = (s2 << j) | (s1 >> rj);
                x3 = s2 >> rj;
            }
            r0 ^= x0 & mask;
            r1 ^= x1 & mask;
            r2 ^= x2 & mask;
            r3 ^= x3 & mask;
        }

        // Advance s by 8 bits: s <<= 8.
        // new s2 = (s2 << 8) | (s1 >> 56)
        // new s1 = (s1 << 8) | (s0 >> 56)
        // new s0 = (s0 << 8)
        // But we also drop the bits that have moved past bit 256 - in practice
        // s grows; we keep 192 bits which is plenty for 128-bit a shifted by
        // up to 120 (= 248 bits total), but the top of s2 could overflow on
        // the last iterations. However we only consume s for the CURRENT byte
        // before shifting, so even if s's top spills after byte 15 it doesn't
        // matter (we don't use it again). The result occupies <=256 bits, and
        // each (s<<j) we XOR-in already accounts for its true position via
        // having shifted s by 8 each previous step.
        ulong ns2 = (s2 << 8) | (s1 >> 56);
        ulong ns1 = (s1 << 8) | (s0 >> 56);
        ulong ns0 = (s0 << 8);
        s0 = ns0; s1 = ns1; s2 = ns2;
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
        ulong m11_lo, m11_hi; gf128_mul(a1_lo, a1_hi, b1_lo, b1_hi, m11_lo, m11_hi);
        ulong mk_lo,  mk_hi;  gf128_mul(a0_lo ^ a1_lo, a0_hi ^ a1_hi,
                                        b0_lo ^ b1_lo, b0_hi ^ b1_hi,
                                        mk_lo, mk_hi);

        ulong am_lo, am_hi;
        gf128_mul(alpha_lo, alpha_hi, m11_lo, m11_hi, am_lo, am_hi);

        c[base + 0] = m00_lo ^ am_lo;
        c[base + 1] = m00_hi ^ am_hi;
        c[base + 2] = mk_lo ^ m00_lo;
        c[base + 3] = mk_hi ^ m00_hi;
    }
}