#include <metal_stdlib>
using namespace metal;

// 64x64 -> 128 carry-less multiply.
// Strategy: split b into 8 streams (bit j of each byte). For stream s,
// accumulate sum over j of (bit_{8j+s}(b)) * (a << (8j+s)).
// Because s is constant per stream, the shift by (8j+s) decomposes into
// a constant shift (s) followed by a constant shift (8j) — both compile-
// time literals. We then OR the 8 streams together with byte alignment.
inline void clmul64(ulong a, ulong b, thread ulong &lo, thread ulong &hi)
{
    // For each of 8 bit-positions s in 0..7, build an accumulator
    // acc_s = XOR over j in 0..7 of (bit_{8j+s}(b)) ? (a << 8j) : 0
    // Then result += acc_s << s (as 128-bit shift).
    //
    // We expand 8 streams fully.

    ulong rl = 0, rh = 0;

    // Stream s: process 8 bytes (j=0..7)
    // Use mask = -(ulong)bit to AND with (a<<8j).
    #define STREAM(s) {                                                   \
        ulong acc_lo = 0, acc_hi = 0;                                     \
        ulong m;                                                          \
        /* j=0 */ m = (ulong)(0ul - ((b >> (s     )) & 1ul));             \
        acc_lo ^= m & a;                                                  \
        /* j=1 */ m = (ulong)(0ul - ((b >> (s +  8)) & 1ul));             \
        acc_lo ^= m & (a << 8);                                           \
        acc_hi ^= m & (a >> 56);                                          \
        /* j=2 */ m = (ulong)(0ul - ((b >> (s + 16)) & 1ul));             \
        acc_lo ^= m & (a << 16);                                          \
        acc_hi ^= m & (a >> 48);                                          \
        /* j=3 */ m = (ulong)(0ul - ((b >> (s + 24)) & 1ul));             \
        acc_lo ^= m & (a << 24);                                          \
        acc_hi ^= m & (a >> 40);                                          \
        /* j=4 */ m = (ulong)(0ul - ((b >> (s + 32)) & 1ul));             \
        acc_lo ^= m & (a << 32);                                          \
        acc_hi ^= m & (a >> 32);                                          \
        /* j=5 */ m = (ulong)(0ul - ((b >> (s + 40)) & 1ul));             \
        acc_lo ^= m & (a << 40);                                          \
        acc_hi ^= m & (a >> 24);                                          \
        /* j=6 */ m = (ulong)(0ul - ((b >> (s + 48)) & 1ul));             \
        acc_lo ^= m & (a << 48);                                          \
        acc_hi ^= m & (a >> 16);                                          \
        /* j=7 */ m = (ulong)(0ul - ((b >> (s + 56)) & 1ul));             \
        acc_lo ^= m & (a << 56);                                          \
        acc_hi ^= m & (a >>  8);                                          \
        /* shift acc by s and XOR into result */                          \
        if ((s) == 0) {                                                   \
            rl ^= acc_lo;                                                 \
            rh ^= acc_hi;                                                 \
        } else {                                                          \
            rl ^= (acc_lo << (s));                                        \
            rh ^= (acc_hi << (s)) | (acc_lo >> (64 - (s)));               \
        }                                                                 \
    }

    STREAM(0)
    STREAM(1)
    STREAM(2)
    STREAM(3)
    STREAM(4)
    STREAM(5)
    STREAM(6)
    STREAM(7)

    #undef STREAM

    lo = rl;
    hi = rh;
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
    ulong d_lo0 = t2 ^ (t2 << 1) ^ (t2 << 2) ^ (t2 << 7);
    ulong d_lo1 = t3
                ^ ((t3 << 1) | (t2 >> 63))
                ^ ((t3 << 2) | (t2 >> 62))
                ^ ((t3 << 7) | (t2 >> 57));
    ulong d_hi  = (t3 >> 63) ^ (t3 >> 62) ^ (t3 >> 57);

    t0 ^= d_lo0;
    t1 ^= d_lo1;
    t0 ^= d_hi ^ (d_hi << 1) ^ (d_hi << 2) ^ (d_hi << 7);

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