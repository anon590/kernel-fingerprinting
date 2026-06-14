### Optimization rationale

The failing iter-9 likely had an issue with `sl >> 48` etc. when `sl` had bits in upper positions... actually the bug was that for chain shifts like `sl >> 48`, when nibble n=0 we had `sl=0, sh=0` so `sl >> 48 = 0` which is fine. Let me reconsider — the issue in iter-9 was a subtle bit: `(sh << 16) | (sl >> 48)` is wrong when shift is 16 because we shift sl by 16 into low 64, and the overflow into high is `sl >> (64-16) = sl >> 48`. That's actually correct. But for the i=0 case, we don't shift at all; the `sl >> (64-0) = sl >> 64` is UB. The incumbent guards i=0 separately, while iter-9's PICK(0) chain entry does `r0l ^= sl` (no shift) — that's fine.

Let me go back to the incumbent and improve it: use a **byte-window (8-bit) table** with 256 entries is too big. Instead, keep the nibble approach but reduce per-nibble overhead by precomputing once and using simple shifts. I'll stick with the working incumbent structure but unroll the loop fully and avoid the array (force registers), plus split the 16 nibbles into 2 independent accumulator chains for ILP.

```metal
#include <metal_stdlib>
using namespace metal;

inline void clmul64(ulong a, ulong b, thread ulong &lo, thread ulong &hi)
{
    // 16-entry table T[k] = k(x) * a in GF(2)[x], 128-bit results.
    ulong T0l = 0ul,            T0h = 0ul;
    ulong T1l = a,              T1h = 0ul;
    ulong T2l = a << 1,         T2h = a >> 63;
    ulong T3l = T2l ^ T1l,      T3h = T2h ^ T1h;
    ulong T4l = a << 2,         T4h = a >> 62;
    ulong T5l = T4l ^ T1l,      T5h = T4h ^ T1h;
    ulong T6l = T4l ^ T2l,      T6h = T4h ^ T2h;
    ulong T7l = T4l ^ T3l,      T7h = T4h ^ T3h;
    ulong T8l = a << 3,         T8h = a >> 61;
    ulong T9l = T8l ^ T1l,      T9h = T8h ^ T1h;
    ulong TAl = T8l ^ T2l,      TAh = T8h ^ T2h;
    ulong TBl = T8l ^ T3l,      TBh = T8h ^ T3h;
    ulong TCl = T8l ^ T4l,      TCh = T8h ^ T4h;
    ulong TDl = T8l ^ T5l,      TDh = T8h ^ T5h;
    ulong TEl = T8l ^ T6l,      TEh = T8h ^ T6h;
    ulong TFl = T8l ^ T7l,      TFh = T8h ^ T7h;

    ulong Tl[16] = {T0l,T1l,T2l,T3l,T4l,T5l,T6l,T7l,T8l,T9l,TAl,TBl,TCl,TDl,TEl,TFl};
    ulong Th[16] = {T0h,T1h,T2h,T3h,T4h,T5h,T6h,T7h,T8h,T9h,TAh,TBh,TCh,TDh,TEh,TFh};

    // Two independent accumulator chains for ILP.
    ulong rlA = 0ul, rhA = 0ul;
    ulong rlB = 0ul, rhB = 0ul;

    // i = 0 (no shift)
    {
        uint n = (uint)(b & 0xFul);
        rlA ^= Tl[n];
        rhA ^= Th[n];
    }
    // Even nibbles -> chain A, odd nibbles -> chain B
    #define DO_NIB(I, CHAIN_L, CHAIN_H) do { \
        uint sh = (uint)(4u * (I)); \
        uint n = (uint)((b >> sh) & 0xFul); \
        ulong el = Tl[n]; \
        ulong eh = Th[n]; \
        CHAIN_L ^= el << sh; \
        CHAIN_H ^= (eh << sh) | (el >> (64u - sh)); \
    } while(0)

    DO_NIB(1,  rlB, rhB);
    DO_NIB(2,  rlA, rhA);
    DO_NIB(3,  rlB, rhB);
    DO_NIB(4,  rlA, rhA);
    DO_NIB(5,  rlB, rhB);
    DO_NIB(6,  rlA, rhA);
    DO_NIB(7,  rlB, rhB);
    DO_NIB(8,  rlA, rhA);
    DO_NIB(9,  rlB, rhB);
    DO_NIB(10, rlA, rhA);
    DO_NIB(11, rlB, rhB);
    DO_NIB(12, rlA, rhA);
    DO_NIB(13, rlB, rhB);
    DO_NIB(14, rlA, rhA);
    DO_NIB(15, rlB, rhB);

    #undef DO_NIB

    lo = rlA ^ rlB;
    hi = rhA ^ rhB;
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
```