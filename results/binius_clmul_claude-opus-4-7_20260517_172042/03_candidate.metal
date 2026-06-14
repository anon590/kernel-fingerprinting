#include <metal_stdlib>
using namespace metal;

// 128x128 -> 256 bit unreduced clmul using 4-bit window (16-entry table).
// Two independent accumulator streams (one per 64-bit half of b) then merged.
inline void clmul128_unreduced(
    ulong a_lo, ulong a_hi, ulong b_lo, ulong b_hi,
    thread ulong &t0, thread ulong &t1,
    thread ulong &t2, thread ulong &t3)
{
    // Build 16-entry table T[k] = a * k as 192-bit triple.
    ulong tlo[16], tmd[16], thi[16];

    tlo[0] = 0ul; tmd[0] = 0ul; thi[0] = 0ul;
    tlo[1] = a_lo; tmd[1] = a_hi; thi[1] = 0ul;
    tlo[2] = a_lo << 1; tmd[2] = (a_hi << 1) | (a_lo >> 63); thi[2] = a_hi >> 63;
    tlo[4] = a_lo << 2; tmd[4] = (a_hi << 2) | (a_lo >> 62); thi[4] = a_hi >> 62;
    tlo[8] = a_lo << 3; tmd[8] = (a_hi << 3) | (a_lo >> 61); thi[8] = a_hi >> 61;

    tlo[3]  = tlo[1] ^ tlo[2];  tmd[3]  = tmd[1] ^ tmd[2];  thi[3]  = thi[1] ^ thi[2];
    tlo[5]  = tlo[1] ^ tlo[4];  tmd[5]  = tmd[1] ^ tmd[4];  thi[5]  = thi[1] ^ thi[4];
    tlo[6]  = tlo[2] ^ tlo[4];  tmd[6]  = tmd[2] ^ tmd[4];  thi[6]  = thi[2] ^ thi[4];
    tlo[7]  = tlo[3] ^ tlo[4];  tmd[7]  = tmd[3] ^ tmd[4];  thi[7]  = thi[3] ^ thi[4];
    tlo[9]  = tlo[1] ^ tlo[8];  tmd[9]  = tmd[1] ^ tmd[8];  thi[9]  = thi[1] ^ thi[8];
    tlo[10] = tlo[2] ^ tlo[8];  tmd[10] = tmd[2] ^ tmd[8];  thi[10] = thi[2] ^ thi[8];
    tlo[11] = tlo[3] ^ tlo[8];  tmd[11] = tmd[3] ^ tmd[8];  thi[11] = thi[3] ^ thi[8];
    tlo[12] = tlo[4] ^ tlo[8];  tmd[12] = tmd[4] ^ tmd[8];  thi[12] = thi[4] ^ thi[8];
    tlo[13] = tlo[5] ^ tlo[8];  tmd[13] = tmd[5] ^ tmd[8];  thi[13] = thi[5] ^ thi[8];
    tlo[14] = tlo[6] ^ tlo[8];  tmd[14] = tmd[6] ^ tmd[8];  thi[14] = thi[6] ^ thi[8];
    tlo[15] = tlo[7] ^ tlo[8];  tmd[15] = tmd[7] ^ tmd[8];  thi[15] = thi[7] ^ thi[8];

    // Independent accumulators for b_lo and b_hi streams.
    // Stream-L accumulates contributions of b_lo nibbles (no extra shift).
    // Stream-H accumulates contributions of b_hi nibbles (shifted by 64).
    // Final: result = streamL + (streamH << 64).
    //
    // Within each stream, we still scan high-nibble to low-nibble with
    // a 4-bit shift each step, but the two streams run in parallel for ILP.

    ulong L0 = 0ul, L1 = 0ul, L2 = 0ul;          // 192-bit accumulator for b_lo
    ulong H0 = 0ul, H1 = 0ul, H2 = 0ul;          // 192-bit accumulator for b_hi

    #pragma clang loop unroll(full)
    for (int s = 60; s >= 0; s -= 4) {
        // Shift L left by 4, fold in T[nibble of b_lo]
        ulong nL2 = (L2 << 4) | (L1 >> 60);
        ulong nL1 = (L1 << 4) | (L0 >> 60);
        ulong nL0 = (L0 << 4);
        uint kL = (uint)((b_lo >> s) & 0xFul);
        L0 = nL0 ^ tlo[kL];
        L1 = nL1 ^ tmd[kL];
        L2 = nL2 ^ thi[kL];

        // Shift H left by 4, fold in T[nibble of b_hi]
        ulong nH2 = (H2 << 4) | (H1 >> 60);
        ulong nH1 = (H1 << 4) | (H0 >> 60);
        ulong nH0 = (H0 << 4);
        uint kH = (uint)((b_hi >> s) & 0xFul);
        H0 = nH0 ^ tlo[kH];
        H1 = nH1 ^ tmd[kH];
        H2 = nH2 ^ thi[kH];
    }

    // Combine: result = L (3 limbs at offset 0) ^ H (3 limbs at offset 64)
    // L occupies bits [0..191], H<<64 occupies bits [64..255].
    t0 = L0;
    t1 = L1 ^ H0;
    t2 = L2 ^ H1;
    t3 = H2;
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

        // Karatsuba-style: 3 muls instead of 4 for (a0+a1*v)(b0+b1*v).
        //   m00 = a0*b0
        //   m11 = a1*b1
        //   mk  = (a0+a1)*(b0+b1) = m00 ^ m01 ^ m10 ^ m11   (in GF(2))
        // => m01 ^ m10 = mk ^ m00 ^ m11
        // c0 = m00 ^ alpha*m11
        // c1 = m01 ^ m10 ^ m11 = mk ^ m00 ^ m11 ^ m11 = mk ^ m00
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