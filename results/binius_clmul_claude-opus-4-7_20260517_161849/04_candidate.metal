#include <metal_stdlib>
using namespace metal;

// Select-tree-based 16-entry table lookup, kept entirely in registers.
// Avoids local array which the compiler may spill to threadgroup/stack.
inline void clmul64(ulong a, ulong b, thread ulong &lo, thread ulong &hi)
{
    // Build 16-entry table T[k] = (k as 4-bit poly) clmul a.
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

    // Select-tree lookup macro: produces (el, eh) for nibble n in 0..15.
    // Uses select() which maps to register-only conditional moves.
    #define LOOKUP(n_expr, el_out, eh_out) do {                                  \
        uint _n = (n_expr);                                                      \
        ulong _b0l = select(T0l, T1l, (_n & 1u) != 0u);                          \
        ulong _b0h = select(T0h, T1h, (_n & 1u) != 0u);                          \
        ulong _b2l = select(T2l, T3l, (_n & 1u) != 0u);                          \
        ulong _b2h = select(T2h, T3h, (_n & 1u) != 0u);                          \
        ulong _b4l = select(T4l, T5l, (_n & 1u) != 0u);                          \
        ulong _b4h = select(T4h, T5h, (_n & 1u) != 0u);                          \
        ulong _b6l = select(T6l, T7l, (_n & 1u) != 0u);                          \
        ulong _b6h = select(T6h, T7h, (_n & 1u) != 0u);                          \
        ulong _b8l = select(T8l, T9l, (_n & 1u) != 0u);                          \
        ulong _b8h = select(T8h, T9h, (_n & 1u) != 0u);                          \
        ulong _bAl = select(TAl, TBl, (_n & 1u) != 0u);                          \
        ulong _bAh = select(TAh, TBh, (_n & 1u) != 0u);                          \
        ulong _bCl = select(TCl, TDl, (_n & 1u) != 0u);                          \
        ulong _bCh = select(TCh, TDh, (_n & 1u) != 0u);                          \
        ulong _bEl = select(TEl, TFl, (_n & 1u) != 0u);                          \
        ulong _bEh = select(TEh, TFh, (_n & 1u) != 0u);                          \
        ulong _c0l = select(_b0l, _b2l, (_n & 2u) != 0u);                        \
        ulong _c0h = select(_b0h, _b2h, (_n & 2u) != 0u);                        \
        ulong _c4l = select(_b4l, _b6l, (_n & 2u) != 0u);                        \
        ulong _c4h = select(_b4h, _b6h, (_n & 2u) != 0u);                        \
        ulong _c8l = select(_b8l, _bAl, (_n & 2u) != 0u);                        \
        ulong _c8h = select(_b8h, _bAh, (_n & 2u) != 0u);                        \
        ulong _cCl = select(_bCl, _bEl, (_n & 2u) != 0u);                        \
        ulong _cCh = select(_bCh, _bEh, (_n & 2u) != 0u);                        \
        ulong _d0l = select(_c0l, _c4l, (_n & 4u) != 0u);                        \
        ulong _d0h = select(_c0h, _c4h, (_n & 4u) != 0u);                        \
        ulong _d8l = select(_c8l, _cCl, (_n & 4u) != 0u);                        \
        ulong _d8h = select(_c8h, _cCh, (_n & 4u) != 0u);                        \
        el_out = select(_d0l, _d8l, (_n & 8u) != 0u);                            \
        eh_out = select(_d0h, _d8h, (_n & 8u) != 0u);                            \
    } while(0)

    ulong rl = 0ul, rh = 0ul;
    ulong el, eh;

    // i=0: no shift
    LOOKUP((uint)(b & 0xFul), el, eh);
    rl ^= el;
    rh ^= eh;

    // i=1..15: shift by 4*i
    #define STEP(i) do {                                                         \
        const uint sh = (i) * 4u;                                                \
        LOOKUP((uint)((b >> sh) & 0xFul), el, eh);                               \
        rl ^= el << sh;                                                          \
        rh ^= (eh << sh) | (el >> (64u - sh));                                   \
    } while(0)

    STEP(1);  STEP(2);  STEP(3);  STEP(4);
    STEP(5);  STEP(6);  STEP(7);  STEP(8);
    STEP(9);  STEP(10); STEP(11); STEP(12);
    STEP(13); STEP(14); STEP(15);

    #undef STEP
    #undef LOOKUP

    lo = rl;
    hi = rh;
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