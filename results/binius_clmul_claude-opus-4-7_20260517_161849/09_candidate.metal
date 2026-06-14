#include <metal_stdlib>
using namespace metal;

// 64x64 -> 128 carry-less multiply.
// Strategy: bit-by-bit with 4 independent accumulator chains for ILP.
// Each chain handles every 4th bit of b, accumulating shifted copies of a
// masked by the corresponding b-bit. Chains are independent, then XORed.
inline void clmul64(ulong a, ulong b, thread ulong &lo, thread ulong &hi)
{
    // Build 16-entry nibble table for a: T[k] = (k as 4-bit poly) * a.
    ulong T1l = a;
    ulong T2l = a << 1;             ulong T2h = a >> 63;
    ulong T4l = a << 2;             ulong T4h = a >> 62;
    ulong T8l = a << 3;             ulong T8h = a >> 61;

    ulong T3l = T2l ^ T1l;          ulong T3h = T2h;
    ulong T5l = T4l ^ T1l;          ulong T5h = T4h;
    ulong T6l = T4l ^ T2l;          ulong T6h = T4h ^ T2h;
    ulong T7l = T4l ^ T2l ^ T1l;    ulong T7h = T4h ^ T2h;
    ulong T9l = T8l ^ T1l;          ulong T9h = T8h;
    ulong TAl = T8l ^ T2l;          ulong TAh = T8h ^ T2h;
    ulong TBl = T8l ^ T2l ^ T1l;    ulong TBh = T8h ^ T2h;
    ulong TCl = T8l ^ T4l;          ulong TCh = T8h ^ T4h;
    ulong TDl = T8l ^ T4l ^ T1l;    ulong TDh = T8h ^ T4h;
    ulong TEl = T8l ^ T4l ^ T2l;    ulong TEh = T8h ^ T4h ^ T2h;
    ulong TFl = T8l ^ T4l ^ T2l ^ T1l; ulong TFh = T8h ^ T4h ^ T2h;

    // Branchless select by nibble using bit-masks. For nibble n in 0..15,
    // build (Sl, Sh) via mask-AND-OR. Use signed-shift sign extension for mask.
    // To avoid the mux tree, we precompute all 16 entries in arrays and index.
    // Use uint2 (low,high) packed in a local array; compiler keeps in registers.

    // Four independent accumulators, each handling 4 nibbles (16 nibbles total).
    ulong r0l = 0ul, r0h = 0ul;
    ulong r1l = 0ul, r1h = 0ul;
    ulong r2l = 0ul, r2h = 0ul;
    ulong r3l = 0ul, r3h = 0ul;

    #define PICK(N, OUTL, OUTH) do {                                  \
        uint n = (uint)((b >> (4u*(N))) & 0xFul);                     \
        ulong sl, sh;                                                 \
        switch (n) {                                                  \
            case 0:  sl = 0ul; sh = 0ul; break;                       \
            case 1:  sl = T1l; sh = 0ul;  break;                      \
            case 2:  sl = T2l; sh = T2h;  break;                      \
            case 3:  sl = T3l; sh = T3h;  break;                      \
            case 4:  sl = T4l; sh = T4h;  break;                      \
            case 5:  sl = T5l; sh = T5h;  break;                      \
            case 6:  sl = T6l; sh = T6h;  break;                      \
            case 7:  sl = T7l; sh = T7h;  break;                      \
            case 8:  sl = T8l; sh = T8h;  break;                      \
            case 9:  sl = T9l; sh = T9h;  break;                      \
            case 10: sl = TAl; sh = TAh;  break;                      \
            case 11: sl = TBl; sh = TBh;  break;                      \
            case 12: sl = TCl; sh = TCh;  break;                      \
            case 13: sl = TDl; sh = TDh;  break;                      \
            case 14: sl = TEl; sh = TEh;  break;                      \
            default: sl = TFl; sh = TFh;  break;                      \
        }                                                             \
        OUTL = sl; OUTH = sh;                                         \
    } while(0)

    // For each nibble i, contribution is (Tl[n] << (4i), Th[n] << (4i) | Tl[n] >> (64-4i)).
    // Split into 4 chains by i mod 4.
    ulong sl, sh;

    // Chain 0: i = 0, 4, 8, 12
    PICK(0,  sl, sh); r0l ^= sl;                          r0h ^= sh;
    PICK(4,  sl, sh); r0l ^= sl << 16;                    r0h ^= (sh << 16) | (sl >> 48);
    PICK(8,  sl, sh); r0l ^= sl << 32;                    r0h ^= (sh << 32) | (sl >> 32);
    PICK(12, sl, sh); r0l ^= sl << 48;                    r0h ^= (sh << 48) | (sl >> 16);

    // Chain 1: i = 1, 5, 9, 13
    PICK(1,  sl, sh); r1l ^= sl << 4;                     r1h ^= (sh << 4)  | (sl >> 60);
    PICK(5,  sl, sh); r1l ^= sl << 20;                    r1h ^= (sh << 20) | (sl >> 44);
    PICK(9,  sl, sh); r1l ^= sl << 36;                    r1h ^= (sh << 36) | (sl >> 28);
    PICK(13, sl, sh); r1l ^= sl << 52;                    r1h ^= (sh << 52) | (sl >> 12);

    // Chain 2: i = 2, 6, 10, 14
    PICK(2,  sl, sh); r2l ^= sl << 8;                     r2h ^= (sh << 8)  | (sl >> 56);
    PICK(6,  sl, sh); r2l ^= sl << 24;                    r2h ^= (sh << 24) | (sl >> 40);
    PICK(10, sl, sh); r2l ^= sl << 40;                    r2h ^= (sh << 40) | (sl >> 24);
    PICK(14, sl, sh); r2l ^= sl << 56;                    r2h ^= (sh << 56) | (sl >> 8);

    // Chain 3: i = 3, 7, 11, 15
    PICK(3,  sl, sh); r3l ^= sl << 12;                    r3h ^= (sh << 12) | (sl >> 52);
    PICK(7,  sl, sh); r3l ^= sl << 28;                    r3h ^= (sh << 28) | (sl >> 36);
    PICK(11, sl, sh); r3l ^= sl << 44;                    r3h ^= (sh << 44) | (sl >> 20);
    PICK(15, sl, sh); r3l ^= sl << 60;                    r3h ^= (sh << 60) | (sl >> 4);

    #undef PICK

    lo = r0l ^ r1l ^ r2l ^ r3l;
    hi = r0h ^ r1h ^ r2h ^ r3h;
}

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