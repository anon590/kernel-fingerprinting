#include <metal_stdlib>
using namespace metal;

// 64x64 carry-less multiply using a 16-entry table indexed by 4-bit windows of b.
// Table built from a. Loop fully unrolled so shifts are compile-time constants.
inline void clmul64(ulong a, ulong b, thread ulong &lo, thread ulong &hi)
{
    // T[k] = (k as 4-bit poly) * a, as a 128-bit value (Tl, Th).
    ulong T0l = 0ul,         T0h = 0ul;
    ulong T1l = a,           T1h = 0ul;
    ulong T2l = a << 1,      T2h = a >> 63;
    ulong T3l = T2l ^ T1l,   T3h = T2h ^ T1h;
    ulong T4l = a << 2,      T4h = a >> 62;
    ulong T5l = T4l ^ T1l,   T5h = T4h ^ T1h;
    ulong T6l = T4l ^ T2l,   T6h = T4h ^ T2h;
    ulong T7l = T4l ^ T3l,   T7h = T4h ^ T3h;
    ulong T8l = a << 3,      T8h = a >> 61;
    ulong T9l = T8l ^ T1l,   T9h = T8h ^ T1h;
    ulong TAl = T8l ^ T2l,   TAh = T8h ^ T2h;
    ulong TBl = T8l ^ T3l,   TBh = T8h ^ T3h;
    ulong TCl = T8l ^ T4l,   TCh = T8h ^ T4h;
    ulong TDl = T8l ^ T5l,   TDh = T8h ^ T5h;
    ulong TEl = T8l ^ T6l,   TEh = T8h ^ T6h;
    ulong TFl = T8l ^ T7l,   TFh = T8h ^ T7h;

    ulong Tl[16] = {T0l,T1l,T2l,T3l,T4l,T5l,T6l,T7l,T8l,T9l,TAl,TBl,TCl,TDl,TEl,TFl};
    ulong Th[16] = {T0h,T1h,T2h,T3h,T4h,T5h,T6h,T7h,T8h,T9h,TAh,TBh,TCh,TDh,TEh,TFh};

    // i = 0 (no shift)
    uint n0 = (uint)(b & 0xFul);
    ulong rl = Tl[n0];
    ulong rh = Th[n0];

    // Fully unrolled: each iteration uses compile-time shift amount.
    #define ACC(SH) {                                              \
        uint nib = (uint)((b >> (SH)) & 0xFul);                    \
        ulong el = Tl[nib];                                        \
        ulong eh = Th[nib];                                        \
        rl ^= el << (SH);                                          \
        rh ^= (eh << (SH)) | (el >> (64 - (SH)));                  \
    }
    ACC(4)  ACC(8)  ACC(12) ACC(16) ACC(20) ACC(24) ACC(28) ACC(32)
    ACC(36) ACC(40) ACC(44) ACC(48) ACC(52) ACC(56) ACC(60)
    #undef ACC

    lo = rl;
    hi = rh;
}

// 128 x 128 -> 256 carry-less multiply via Karatsuba on 64-bit halves.
inline void clmul128_unreduced(
    ulong a_lo, ulong a_hi, ulong b_lo, ulong b_hi,
    thread ulong &t0, thread ulong &t1,
    thread ulong &t2, thread ulong &t3)
{
    ulong p0_lo, p0_hi, p1_lo, p1_hi, pm_lo, pm_hi;
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