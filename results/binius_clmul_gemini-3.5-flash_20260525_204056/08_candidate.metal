#include <metal_stdlib>
using namespace metal;

inline ulong fold_low(ulong x) {
    return x ^ (x << 1u) ^ (x << 2u) ^ (x << 7u);
}

inline ulong fold_high(ulong x) {
    return (x >> 63u) ^ (x >> 62u) ^ (x >> 57u);
}

inline void gcm_reduce(
    ulong t0, ulong t1, ulong t2, ulong t3,
    thread ulong &r_lo, thread ulong &r_hi)
{
    r_lo = t0 ^ fold_low(t2 ^ fold_high(t3));
    r_hi = t1 ^ fold_low(t3) ^ fold_high(t2);
}

inline ulong2 clmul64_direct(ulong A, ulong B)
{
    long b = (long)B;
    long mask0 = (b << 63) >> 63;
    ulong r_lo = A & (ulong)mask0;
    ulong r_hi = 0;

    #pragma unroll
    for (int j = 1; j < 64; ++j) {
        long mask = (b << (63 - j)) >> 63;
        ulong temp = A & (ulong)mask;
        r_lo ^= temp << j;
        r_hi ^= temp >> (64 - j);
    }
    return ulong2(r_lo, r_hi);
}

inline void clmul128_unreduced(
    ulong a_lo, ulong a_hi, ulong b_lo, ulong b_hi,
    thread ulong &t0, thread ulong &t1,
    thread ulong &t2, thread ulong &t3)
{
    ulong2 low = clmul64_direct(a_lo, b_lo);
    ulong2 high = clmul64_direct(a_hi, b_hi);
    ulong2 mid = clmul64_direct(a_lo ^ a_hi, b_lo ^ b_hi);

    mid ^= low ^ high;

    t0 = low.x;
    t1 = low.y ^ mid.x;
    t2 = high.x ^ mid.y;
    t3 = high.y;
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
        ulong a0 = a[2 * idx];
        ulong a1 = a[2 * idx + 1];
        ulong b0 = b[2 * idx];
        ulong b1 = b[2 * idx + 1];

        ulong c_lo, c_hi;
        gf128_mul(a0, a1, b0, b1, c_lo, c_hi);

        c[2 * idx] = c_lo;
        c[2 * idx + 1] = c_hi;
    } else {
        ulong a0_lo = a[4 * idx];
        ulong a0_hi = a[4 * idx + 1];
        ulong a1_lo = a[4 * idx + 2];
        ulong a1_hi = a[4 * idx + 3];

        ulong b0_lo = b[4 * idx];
        ulong b0_hi = b[4 * idx + 1];
        ulong b1_lo = b[4 * idx + 2];
        ulong b1_hi = b[4 * idx + 3];

        // Unreduced multiplications
        ulong t00_0, t00_1, t00_2, t00_3;
        clmul128_unreduced(a0_lo, a0_hi, b0_lo, b0_hi, t00_0, t00_1, t00_2, t00_3);

        ulong t11_0, t11_1, t11_2, t11_3;
        clmul128_unreduced(a1_lo, a1_hi, b1_lo, b1_hi, t11_0, t11_1, t11_2, t11_3);

        ulong tadd_0, tadd_1, tadd_2, tadd_3;
        clmul128_unreduced(a0_lo ^ a1_lo, a0_hi ^ a1_hi, b0_lo ^ b1_lo, b0_hi ^ b1_hi, tadd_0, tadd_1, tadd_2, tadd_3);

        // We need M11 to multiply by alpha
        ulong m11_lo, m11_hi;
        gcm_reduce(t11_0, t11_1, t11_2, t11_3, m11_lo, m11_hi);

        // Multiply alpha by M11
        ulong tal_0, tal_1, tal_2, tal_3;
        clmul128_unreduced(alpha_lo, alpha_hi, m11_lo, m11_hi, tal_0, tal_1, tal_2, tal_3);

        // c0 = reduce(t00 ^ tal)
        ulong c0_lo, c0_hi;
        gcm_reduce(t00_0 ^ tal_0, t00_1 ^ tal_1, t00_2 ^ tal_2, t00_3 ^ tal_3, c0_lo, c0_hi);

        // c1 = reduce(tadd ^ t00)
        ulong c1_lo, c1_hi;
        gcm_reduce(tadd_0 ^ t00_0, tadd_1 ^ t00_1, tadd_2 ^ t00_2, tadd_3 ^ t00_3, c1_lo, c1_hi);

        c[4 * idx] = c0_lo;
        c[4 * idx + 1] = c0_hi;
        c[4 * idx + 2] = c1_lo;
        c[4 * idx + 3] = c1_hi;
    }
}