#include <metal_stdlib>
using namespace metal;

// 64x64 -> 128 carry-less multiply, 4-bit window, fully unrolled.
// Builds a 16-entry table for `a`, then accumulates 16 nibbles of `b` with
// compile-time shift amounts so the compiler emits pure register XORs/shifts.
inline void clmul64(ulong a, ulong b, thread ulong &lo, thread ulong &hi)
{
    // Precompute T[k] = (k as 4-bit polynomial) * a, k = 0..15.
    // Stored as (low64, high64).
    ulong T1l = a;
    ulong T2l = a << 1;            ulong T2h = a >> 63;
    ulong T4l = a << 2;            ulong T4h = a >> 62;
    ulong T8l = a << 3;            ulong T8h = a >> 61;

    ulong T3l = T2l ^ T1l;         ulong T3h = T2h;
    ulong T5l = T4l ^ T1l;         ulong T5h = T4h;
    ulong T6l = T4l ^ T2l;         ulong T6h = T4h ^ T2h;
    ulong T7l = T4l ^ T2l ^ T1l;   ulong T7h = T4h ^ T2h;
    ulong T9l = T8l ^ T1l;         ulong T9h = T8h;
    ulong TAl = T8l ^ T2l;         ulong TAh = T8h ^ T2h;
    ulong TBl = T8l ^ T2l ^ T1l;   ulong TBh = T8h ^ T2h;
    ulong TCl = T8l ^ T4l;         ulong TCh = T8h ^ T4h;
    ulong TDl = T8l ^ T4l ^ T1l;   ulong TDh = T8h ^ T4h;
    ulong TEl = T8l ^ T4l ^ T2l;   ulong TEh = T8h ^ T4h ^ T2h;
    ulong TFl = T8l ^ T4l ^ T2l ^ T1l; ulong TFh = T8h ^ T4h ^ T2h;

    // Select by nibble via chained select; compiler turns this into a mux tree.
    #define SEL_L(n) ( ((n)&8u) ? ( ((n)&4u) ? ( ((n)&2u) ? ( ((n)&1u) ? TFl : TEl ) : ( ((n)&1u) ? TDl : TCl ) ) \
                                              : ( ((n)&2u) ? ( ((n)&1u) ? TBl : TAl ) : ( ((n)&1u) ? T9l : T8l ) ) ) \
                                 : ( ((n)&4u) ? ( ((n)&2u) ? ( ((n)&1u) ? T7l : T6l ) : ( ((n)&1u) ? T5l : T4l ) ) \
                                              : ( ((n)&2u) ? ( ((n)&1u) ? T3l : T2l ) : ( ((n)&1u) ? T1l : 0ul ) ) ) )
    #define SEL_H(n) ( ((n)&8u) ? ( ((n)&4u) ? ( ((n)&2u) ? ( ((n)&1u) ? TFh : TEh ) : ( ((n)&1u) ? TDh : TCh ) ) \
                                              : ( ((n)&2u) ? ( ((n)&1u) ? TBh : TAh ) : ( ((n)&1u) ? T9h : T8h ) ) ) \
                                 : ( ((n)&4u) ? ( ((n)&2u) ? ( ((n)&1u) ? T7h : T6h ) : ( ((n)&1u) ? T5h : T4h ) ) \
                                              : ( ((n)&2u) ? ( ((n)&1u) ? T3h : T2h ) : ( ((n)&1u) ? 0ul : 0ul ) ) ) )

    ulong rl = 0ul, rh = 0ul;

    // Fully unrolled: 16 nibbles of b, each at compile-time shift 4*i.
    #define STEP(I) {                                           \
        uint nib = (uint)((b >> (4u*(I))) & 0xFul);             \
        ulong el = SEL_L(nib);                                  \
        ulong eh = SEL_H(nib);                                  \
        rl ^= (I) == 0 ? el : (el << (4u*(I)));                 \
        rh ^= (I) == 0 ? eh : ((eh << (4u*(I))) | (el >> (64u - 4u*(I)))); \
    }

    STEP(0)  STEP(1)  STEP(2)  STEP(3)
    STEP(4)  STEP(5)  STEP(6)  STEP(7)
    STEP(8)  STEP(9)  STEP(10) STEP(11)
    STEP(12) STEP(13) STEP(14) STEP(15)

    #undef STEP
    #undef SEL_L
    #undef SEL_H

    lo = rl;
    hi = rh;
}

// 128 x 128 -> 256 carry-less multiply via Karatsuba on 64-bit halves.
inline void clmul128_unreduced(
    ulong a_lo, ulong a_hi, ulong b_lo, ulong b_hi,
    thread ulong &t0, thread ulong &t1,
    thread ulong &t2, thread ulong &t3)
{
    ulong p0_lo, p0_hi;
    ulong p1_lo, p1_hi;
    ulong pm_lo, pm_hi;
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