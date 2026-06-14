#include <metal_stdlib>
using namespace metal;

inline ulong fold_low(ulong x) {
    ulong r0 = x ^ (x << 1u);
    ulong r1 = (x << 2u) ^ (x << 7u);
    return r0 ^ r1;
}

inline ulong fold_high(ulong x) {
    ulong r0 = (x >> 63u) ^ (x >> 62u);
    ulong r1 = (x >> 57u);
    return r0 ^ r1;
}

inline void gcm_reduce(
    ulong t0, ulong t1, ulong t2, ulong t3,
    thread ulong &r_lo, thread ulong &r_hi)
{
    ulong h3 = fold_high(t3);
    r_lo = t0 ^ fold_low(t2 ^ h3);
    r_hi = (t1 ^ fold_high(t2)) ^ fold_low(t3);
}

inline ulong clmul32(uint A, uint B)
{
    ulong accum0 = 0;
    ulong accum1 = 0;
    ulong accum2 = 0;
    ulong accum3 = 0;
    ulong a = A;
    #pragma unroll
    for (int j = 0; j < 32; j += 4) {
        accum0 ^= ((B & (1u << j)) ? (a << j) : 0ul);
        accum1 ^= ((B & (1u << (j + 1))) ? (a << (j + 1)) : 0ul);
        accum2 ^= ((B & (1u << (j + 2))) ? (a << (j + 2)) : 0ul);
        accum3 ^= ((B & (1u << (j + 3))) ? (a << (j + 3)) : 0ul);
    }
    return (accum0 ^ accum1) ^ (accum2 ^ accum3);
}

inline ulong2 clmul64(ulong A, ulong B)
{
    uint a_lo = (uint)A;
    uint a_hi = (uint)(A >> 32);
    uint b_lo = (uint)B;
    uint b_hi = (uint)(B >> 32);

    ulong low = clmul32(a_lo, b_lo);
    ulong high = clmul32(a_hi, b_hi);
    ulong mid = clmul32(a_lo ^ a_hi, b_lo ^ b_hi);

    mid ^= (low ^ high);

    ulong r_lo = low ^ (mid << 32);
    ulong r_hi = high ^ (mid >> 32);
    return ulong2(r_lo, r_hi);
}

inline void clmul128_unreduced(
    ulong a_lo, ulong a_hi, ulong b_lo, ulong b_hi,
    thread ulong &t0, thread ulong &t1,
    thread ulong &t2, thread ulong &t3)
{
    ulong2 low = clmul64(a_lo, b_lo);
    ulong2 high = clmul64(a_hi, b_hi);
    ulong2 mid = clmul64(a_lo ^ a_hi, b_lo ^ b_hi);

    mid ^= (low ^ high);

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
        device const ulong2 *a2 = (device const ulong2 *)a;
        device const ulong2 *b2 = (device const ulong2 *)b;
        device       ulong2 *c2 = (device       ulong2 *)c;

        ulong2 va = a2[idx];
        ulong2 vb = b2[idx];

        ulong c_lo, c_hi;
        gf128_mul(va.x, va.y, vb.x, vb.y, c_lo, c_hi);

        c2[idx] = ulong2(c_lo, c_hi);
    } else {
        device const ulong4 *a4 = (device const ulong4 *)a;
        device const ulong4 *b4 = (device const ulong4 *)b;
        device       ulong4 *c4 = (device       ulong4 *)c;

        ulong4 va = a4[idx];
        ulong4 vb = b4[idx];

        ulong a0_lo = va.x, a0_hi = va.y;
        ulong a1_lo = va.z, a1_hi = va.w;
        ulong b0_lo = vb.x, b0_hi = vb.y;
        ulong b1_lo = vb.z, b1_hi = vb.w;

        ulong t00_0, t00_1, t00_2, t00_3;
        clmul128_unreduced(a0_lo, a0_hi, b0_lo, b0_hi, t00_0, t00_1, t00_2, t00_3);

        ulong t11_0, t11_1, t11_2, t11_3;
        clmul128_unreduced(a1_lo, a1_hi, b1_lo, b1_hi, t11_0, t11_1, t11_2, t11_3);

        ulong m11_lo, m11_hi;
        gcm_reduce(t11_0, t11_1, t11_2, t11_3, m11_lo, m11_hi);

        ulong tam_0, tam_1, tam_2, tam_3;
        clmul128_unreduced(alpha_lo, alpha_hi, m11_lo, m11_hi, tam_0, tam_1, tam_2, tam_3);

        ulong tadd_0, tadd_1, tadd_2, tadd_3;
        clmul128_unreduced(a0_lo ^ a1_lo, a0_hi ^ a1_hi, b0_lo ^ b1_lo, b0_hi ^ b1_hi, tadd_0, tadd_1, tadd_2, tadd_3);

        ulong tc0_0 = t00_0 ^ tam_0;
        ulong tc0_1 = t00_1 ^ tam_1;
        ulong tc0_2 = t00_2 ^ tam_2;
        ulong tc0_3 = t00_3 ^ tam_3;

        ulong tc1_0 = tadd_0 ^ t00_0;
        ulong tc1_1 = tadd_1 ^ t00_1;
        ulong tc1_2 = tadd_2 ^ t00_2;
        ulong tc1_3 = tadd_3 ^ t00_3;

        ulong c0_lo, c0_hi;
        gcm_reduce(tc0_0, tc0_1, tc0_2, tc0_3, c0_lo, c0_hi);

        ulong c1_lo, c1_hi;
        gcm_reduce(tc1_0, tc1_1, tc1_2, tc1_3, c1_lo, c1_hi);

        c4[idx] = ulong4(c0_lo, c0_hi, c1_lo, c1_hi);
    }
}