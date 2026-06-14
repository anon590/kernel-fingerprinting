I add a separate fast path for the dominant `t=3, arity=2` case when the external/internal diagonal constants are “easy” (`0..7` or `-1`). This path uses a 32-bit-limb Goldilocks representation, avoiding most 64-bit add/sub operations and replacing diagonal multiplications by short addition/negation chains. It preserves the incumbent generic fallback for all other runtime parameter sets, so held-out `t=4` and non-special matrices remain bit-exact. This should improve over the incumbent when the common Poseidon2 small-diagonal parameters are used, by removing many full modular multiplications in every partial round.

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;

struct G {
    uint lo;
    uint hi;
};

inline G g_zero() {
    G r; r.lo = 0u; r.hi = 0u; return r;
}

inline G g_from_ulong(ulong x) {
    G r;
    r.lo = (uint)x;
    r.hi = (uint)(x >> 32);
    return r;
}

inline ulong g_to_ulong(G x) {
    return ((ulong)x.hi << 32) | (ulong)x.lo;
}

inline bool is_easy_const(ulong c) {
    return (c <= 7ul) || (c == (P_GOLD - 1ul));
}

inline G g_canonical(G x) {
    if (x.hi == 0xFFFFFFFFu && x.lo != 0u) {
        x.lo -= 1u;
        x.hi = 0u;
    }
    return x;
}

inline G g_add(G a, G b) {
    uint lo = a.lo + b.lo;
    uint c0 = (lo < a.lo) ? 1u : 0u;

    uint h = a.hi + b.hi;
    uint c1 = (h < a.hi) ? 1u : 0u;
    uint hi = h + c0;
    uint c2 = (hi < h) ? 1u : 0u;

    if ((c1 | c2) != 0u) {
        uint old = lo;
        lo += 0xFFFFFFFFu;
        hi += (lo < old) ? 1u : 0u;
    }

    G r; r.lo = lo; r.hi = hi;
    return g_canonical(r);
}

inline G g_add_ulong(G a, ulong b) {
    return g_add(a, g_from_ulong(b));
}

inline G g_neg(G x) {
    if ((x.lo | x.hi) == 0u) return x;
    uint lo = 1u - x.lo;
    uint borrow = (1u < x.lo) ? 1u : 0u;
    uint hi = 0xFFFFFFFFu - x.hi - borrow;
    G r; r.lo = lo; r.hi = hi;
    return r;
}

inline G g_mul_easy(ulong c, G x) {
    if (c == 0ul) return g_zero();
    if (c == 1ul) return x;
    if (c == (P_GOLD - 1ul)) return g_neg(x);

    G r2 = g_add(x, x);
    if (c == 2ul) return r2;
    G r3 = g_add(r2, x);
    if (c == 3ul) return r3;
    G r4 = g_add(r2, r2);
    if (c == 4ul) return r4;
    G r5 = g_add(r4, x);
    if (c == 5ul) return r5;
    G r6 = g_add(r4, r2);
    if (c == 6ul) return r6;
    return g_add(r6, x);
}

inline G g_reduce_words(uint w0, uint w1, uint w2, uint w3) {
    G t0; t0.lo = w0; t0.hi = w1;

    uint oldlo = t0.lo;
    t0.lo = oldlo - w3;
    uint b0 = (oldlo < w3) ? 1u : 0u;
    uint oldhi = t0.hi;
    t0.hi = oldhi - b0;
    uint under = (oldhi < b0) ? 1u : 0u;

    if (under != 0u) {
        uint lo2 = t0.lo - 0xFFFFFFFFu;
        uint b = (t0.lo < 0xFFFFFFFFu) ? 1u : 0u;
        t0.lo = lo2;
        t0.hi -= b;
    }

    G t1;
    if (w2 == 0u) {
        t1.lo = 0u;
        t1.hi = 0u;
    } else {
        t1.lo = 0u - w2;
        t1.hi = w2 - 1u;
    }

    return g_add(t0, t1);
}

inline G g_mul(G a, G b) {
    uint a0 = a.lo;
    uint a1 = a.hi;
    uint b0 = b.lo;
    uint b1 = b.hi;

    uint p00_lo = a0 * b0;
    uint p00_hi = mulhi(a0, b0);
    uint p01_lo = a0 * b1;
    uint p01_hi = mulhi(a0, b1);
    uint p10_lo = a1 * b0;
    uint p10_hi = mulhi(a1, b0);
    uint p11_lo = a1 * b1;
    uint p11_hi = mulhi(a1, b1);

    uint s1 = p00_hi + p01_lo;
    uint c1 = (s1 < p00_hi) ? 1u : 0u;
    uint w1 = s1 + p10_lo;
    c1 += (w1 < s1) ? 1u : 0u;

    uint s2 = p01_hi + p10_hi;
    uint c2 = (s2 < p01_hi) ? 1u : 0u;
    uint s3 = s2 + p11_lo;
    c2 += (s3 < s2) ? 1u : 0u;
    uint w2 = s3 + c1;
    c2 += (w2 < s3) ? 1u : 0u;

    uint w3 = p11_hi + c2;

    return g_reduce_words(p00_lo, w1, w2, w3);
}

inline G g_square(G a) {
    uint a0 = a.lo;
    uint a1 = a.hi;

    uint p00_lo = a0 * a0;
    uint p00_hi = mulhi(a0, a0);

    uint q_lo = a0 * a1;
    uint q_hi = mulhi(a0, a1);

    uint p11_lo = a1 * a1;
    uint p11_hi = mulhi(a1, a1);

    uint dbl0 = q_lo << 1;
    uint dbl1 = (q_hi << 1) | (q_lo >> 31);
    uint dbl2 = q_hi >> 31;

    uint s1 = p00_hi + dbl0;
    uint c1 = (s1 < p00_hi) ? 1u : 0u;
    uint w1 = s1;

    uint s2 = dbl1 + p11_lo;
    uint c2 = (s2 < dbl1) ? 1u : 0u;
    uint w2 = s2 + c1;
    c2 += (w2 < s2) ? 1u : 0u;

    uint w3 = p11_hi + dbl2 + c2;

    return g_reduce_words(p00_lo, w1, w2, w3);
}

inline G g_sbox7(G x) {
    G x2 = g_square(x);
    G x4 = g_square(x2);
    G x6 = g_mul(x4, x2);
    return g_mul(x6, x);
}

#define G_APPLY_MDS3_UNIT_EASY() do {                         \
    G ss = g_add(g_add(x0, x1), x2);                          \
    G y0 = g_add(ss, g_mul_easy(e0c, x0));                    \
    G y1 = g_add(ss, g_mul_easy(e1c, x1));                    \
    G y2 = g_add(ss, g_mul_easy(e2c, x2));                    \
    x0 = y0; x1 = y1; x2 = y2;                                \
} while (false)

#define G_APPLY_INT3_EASY() do {                              \
    G ss = g_add(g_add(x0, x1), x2);                          \
    G y0 = g_add(ss, g_mul_easy(d0c, x0));                    \
    G y1 = g_add(ss, g_mul_easy(d1c, x1));                    \
    G y2 = g_add(ss, g_mul_easy(d2c, x2));                    \
    x0 = y0; x1 = y1; x2 = y2;                                \
} while (false)

inline ulong poseidon2_t3_unit_g_easy(
    ulong a0, ulong a1,
    device const ulong *rc_ext,
    device const ulong *rc_int,
    ulong e0c, ulong e1c, ulong e2c,
    ulong d0c, ulong d1c, ulong d2c,
    uint r_f,
    uint r_p)
{
    G x0 = g_from_ulong(a0);
    G x1 = g_from_ulong(a1);
    G x2 = g_zero();

    G_APPLY_MDS3_UNIT_EASY();

    uint half_f = r_f >> 1u;

    for (uint r = 0u; r < half_f; ++r) {
        uint o = r * 3u;
        x0 = g_sbox7(g_add_ulong(x0, rc_ext[o + 0u]));
        x1 = g_sbox7(g_add_ulong(x1, rc_ext[o + 1u]));
        x2 = g_sbox7(g_add_ulong(x2, rc_ext[o + 2u]));
        G_APPLY_MDS3_UNIT_EASY();
    }

    for (uint r = 0u; r < r_p; ++r) {
        x0 = g_sbox7(g_add_ulong(x0, rc_int[r]));
        G_APPLY_INT3_EASY();
    }

    for (uint r = half_f; r < r_f; ++r) {
        uint o = r * 3u;
        x0 = g_sbox7(g_add_ulong(x0, rc_ext[o + 0u]));
        x1 = g_sbox7(g_add_ulong(x1, rc_ext[o + 1u]));
        x2 = g_sbox7(g_add_ulong(x2, rc_ext[o + 2u]));
        G_APPLY_MDS3_UNIT_EASY();
    }

    return g_to_ulong(x0);
}

inline ulong gold_canonical(ulong x) {
    ulong y = x - P_GOLD;
    return (x >= P_GOLD) ? y : x;
}

inline ulong gold_add(ulong a, ulong b) {
    ulong s = a + b;
    s += (s < a) ? EPSILON : 0ul;
    return gold_canonical(s);
}

inline ulong gold_neg(ulong x) {
    return (x == 0ul) ? 0ul : (P_GOLD - x);
}

inline ulong gold_sub_one(ulong x) {
    return (x == 0ul) ? (P_GOLD - 1ul) : (x - 1ul);
}

inline ulong gold_reduce_words(uint w0, uint w1, uint w2, uint w3) {
    ulong x_lo = ((ulong)w1 << 32) | (ulong)w0;

    ulong t0 = x_lo - (ulong)w3;
    t0 -= (t0 > x_lo) ? EPSILON : 0ul;

    ulong t1 = ((ulong)w2 << 32) - (ulong)w2;

    ulong t2 = t0 + t1;
    t2 += (t2 < t0) ? EPSILON : 0ul;

    return gold_canonical(t2);
}

inline ulong gold_mul(ulong a, ulong b) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    uint p00_lo = a0 * b0;
    uint p00_hi = mulhi(a0, b0);
    uint p01_lo = a0 * b1;
    uint p01_hi = mulhi(a0, b1);
    uint p10_lo = a1 * b0;
    uint p10_hi = mulhi(a1, b0);
    uint p11_lo = a1 * b1;
    uint p11_hi = mulhi(a1, b1);

    uint s1 = p00_hi + p01_lo;
    uint c1 = (s1 < p00_hi) ? 1u : 0u;
    uint w1 = s1 + p10_lo;
    c1 += (w1 < s1) ? 1u : 0u;

    uint s2 = p01_hi + p10_hi;
    uint c2 = (s2 < p01_hi) ? 1u : 0u;
    uint s3 = s2 + p11_lo;
    c2 += (s3 < s2) ? 1u : 0u;
    uint w2 = s3 + c1;
    c2 += (w2 < s3) ? 1u : 0u;

    uint w3 = p11_hi + c2;

    return gold_reduce_words(p00_lo, w1, w2, w3);
}

inline ulong gold_square(ulong a) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);

    uint p00_lo = a0 * a0;
    uint p00_hi = mulhi(a0, a0);

    uint q_lo = a0 * a1;
    uint q_hi = mulhi(a0, a1);

    uint p11_lo = a1 * a1;
    uint p11_hi = mulhi(a1, a1);

    uint dbl0 = q_lo << 1;
    uint dbl1 = (q_hi << 1) | (q_lo >> 31);
    uint dbl2 = q_hi >> 31;

    uint s1 = p00_hi + dbl0;
    uint c1 = (s1 < p00_hi) ? 1u : 0u;
    uint w1 = s1;

    uint s2 = dbl1 + p11_lo;
    uint c2 = (s2 < dbl1) ? 1u : 0u;
    uint w2 = s2 + c1;
    c2 += (w2 < s2) ? 1u : 0u;

    uint w3 = p11_hi + dbl2 + c2;

    return gold_reduce_words(p00_lo, w1, w2, w3);
}

inline ulong gold_mul_small_or_full(ulong c, ulong x) {
    if (c <= 7ul) {
        if (c == 0ul) return 0ul;
        if (c == 1ul) return x;
        ulong r2 = gold_add(x, x);
        if (c == 2ul) return r2;
        ulong r3 = gold_add(r2, x);
        if (c == 3ul) return r3;
        ulong r4 = gold_add(r2, r2);
        if (c == 4ul) return r4;
        ulong r5 = gold_add(r4, x);
        if (c == 5ul) return r5;
        ulong r6 = gold_add(r4, r2);
        if (c == 6ul) return r6;
        return gold_add(r6, x);
    }
    if (c == (P_GOLD - 1ul)) return gold_neg(x);
    return gold_mul(c, x);
}

inline ulong sbox7(ulong x) {
    ulong x2 = gold_square(x);
    ulong x4 = gold_square(x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
}

inline bool mds3_is_jplusi(device const ulong *m) {
    return m[0] == 2ul && m[1] == 1ul && m[2] == 1ul &&
           m[3] == 1ul && m[4] == 2ul && m[5] == 1ul &&
           m[6] == 1ul && m[7] == 1ul && m[8] == 2ul;
}

inline bool mds3_has_unit_offdiag(device const ulong *m) {
    return m[1] == 1ul && m[2] == 1ul &&
           m[3] == 1ul && m[5] == 1ul &&
           m[6] == 1ul && m[7] == 1ul;
}

inline bool mds4_is_m4(device const ulong *m) {
    return m[0]  == 5ul && m[1]  == 7ul && m[2]  == 1ul && m[3]  == 3ul &&
           m[4]  == 4ul && m[5]  == 6ul && m[6]  == 1ul && m[7]  == 1ul &&
           m[8]  == 1ul && m[9]  == 3ul && m[10] == 5ul && m[11] == 7ul &&
           m[12] == 1ul && m[13] == 1ul && m[14] == 4ul && m[15] == 6ul;
}

#define APPLY_MDS3() do {                                      \
    ulong y0 = gold_mul(m00, x0);                              \
    y0 = gold_add(y0, gold_mul(m01, x1));                      \
    y0 = gold_add(y0, gold_mul(m02, x2));                      \
    ulong y1 = gold_mul(m10, x0);                              \
    y1 = gold_add(y1, gold_mul(m11, x1));                      \
    y1 = gold_add(y1, gold_mul(m12, x2));                      \
    ulong y2 = gold_mul(m20, x0);                              \
    y2 = gold_add(y2, gold_mul(m21, x1));                      \
    y2 = gold_add(y2, gold_mul(m22, x2));                      \
    x0 = y0; x1 = y1; x2 = y2;                                 \
} while (false)

#define APPLY_MDS3_INIT2() do {                                \
    ulong y0 = gold_mul(m00, x0);                              \
    y0 = gold_add(y0, gold_mul(m01, x1));                      \
    ulong y1 = gold_mul(m10, x0);                              \
    y1 = gold_add(y1, gold_mul(m11, x1));                      \
    ulong y2 = gold_mul(m20, x0);                              \
    y2 = gold_add(y2, gold_mul(m21, x1));                      \
    x0 = y0; x1 = y1; x2 = y2;                                 \
} while (false)

#define APPLY_MDS3_JPLUSI() do {                               \
    ulong s = gold_add(gold_add(x0, x1), x2);                  \
    ulong y0 = gold_add(s, x0);                                \
    ulong y1 = gold_add(s, x1);                                \
    ulong y2 = gold_add(s, x2);                                \
    x0 = y0; x1 = y1; x2 = y2;                                 \
} while (false)

#define APPLY_MDS3_JDIAG() do {                                \
    ulong s = gold_add(gold_add(x0, x1), x2);                  \
    ulong y0 = gold_add(s, gold_mul_small_or_full(e0, x0));    \
    ulong y1 = gold_add(s, gold_mul_small_or_full(e1, x1));    \
    ulong y2 = gold_add(s, gold_mul_small_or_full(e2, x2));    \
    x0 = y0; x1 = y1; x2 = y2;                                 \
} while (false)

#define APPLY_INT3() do {                                      \
    ulong s = gold_add(gold_add(x0, x1), x2);                  \
    ulong y0 = gold_add(s, gold_mul(d0, x0));                  \
    ulong y1 = gold_add(s, gold_mul(d1, x1));                  \
    ulong y2 = gold_add(s, gold_mul(d2, x2));                  \
    x0 = y0; x1 = y1; x2 = y2;                                 \
} while (false)

#define APPLY_MDS4() do {                                      \
    ulong y0 = gold_mul(m00, x0);                              \
    y0 = gold_add(y0, gold_mul(m01, x1));                      \
    y0 = gold_add(y0, gold_mul(m02, x2));                      \
    y0 = gold_add(y0, gold_mul(m03, x3));                      \
    ulong y1 = gold_mul(m10, x0);                              \
    y1 = gold_add(y1, gold_mul(m11, x1));                      \
    y1 = gold_add(y1, gold_mul(m12, x2));                      \
    y1 = gold_add(y1, gold_mul(m13, x3));                      \
    ulong y2 = gold_mul(m20, x0);                              \
    y2 = gold_add(y2, gold_mul(m21, x1));                      \
    y2 = gold_add(y2, gold_mul(m22, x2));                      \
    y2 = gold_add(y2, gold_mul(m23, x3));                      \
    ulong y3 = gold_mul(m30, x0);                              \
    y3 = gold_add(y3, gold_mul(m31, x1));                      \
    y3 = gold_add(y3, gold_mul(m32, x2));                      \
    y3 = gold_add(y3, gold_mul(m33, x3));                      \
    x0 = y0; x1 = y1; x2 = y2; x3 = y3;                        \
} while (false)

#define APPLY_MDS4_INIT2() do {                                \
    ulong y0 = gold_mul(m00, x0);                              \
    y0 = gold_add(y0, gold_mul(m01, x1));                      \
    ulong y1 = gold_mul(m10, x0);                              \
    y1 = gold_add(y1, gold_mul(m11, x1));                      \
    ulong y2 = gold_mul(m20, x0);                              \
    y2 = gold_add(y2, gold_mul(m21, x1));                      \
    ulong y3 = gold_mul(m30, x0);                              \
    y3 = gold_add(y3, gold_mul(m31, x1));                      \
    x0 = y0; x1 = y1; x2 = y2; x3 = y3;                        \
} while (false)

#define APPLY_MDS4_M4() do {                                   \
    ulong a0 = gold_add(x0, x1);                               \
    ulong a1 = gold_add(x2, x3);                               \
    ulong a2 = gold_add(gold_add(x1, x1), a1);                 \
    ulong a3 = gold_add(gold_add(x3, x3), a0);                 \
    ulong b1 = gold_add(a1, a1);                               \
    ulong c1 = gold_add(b1, b1);                               \
    ulong a4 = gold_add(c1, a3);                               \
    ulong b0 = gold_add(a0, a0);                               \
    ulong c0 = gold_add(b0, b0);                               \
    ulong a5 = gold_add(c0, a2);                               \
    ulong y0 = gold_add(a3, a5);                               \
    ulong y1 = a5;                                             \
    ulong y2 = gold_add(a2, a4);                               \
    ulong y3 = a4;                                             \
    x0 = y0; x1 = y1; x2 = y2; x3 = y3;                        \
} while (false)

#define APPLY_INT4() do {                                      \
    ulong s = gold_add(gold_add(gold_add(x0, x1), x2), x3);    \
    ulong y0 = gold_add(s, gold_mul(d0, x0));                  \
    ulong y1 = gold_add(s, gold_mul(d1, x1));                  \
    ulong y2 = gold_add(s, gold_mul(d2, x2));                  \
    ulong y3 = gold_add(s, gold_mul(d3, x3));                  \
    x0 = y0; x1 = y1; x2 = y2; x3 = y3;                        \
} while (false)

inline ulong poseidon2_t3_generic(
    ulong x0, ulong x1, ulong x2,
    device const ulong *rc_ext,
    device const ulong *rc_int,
    device const ulong *ext_mds,
    device const ulong *int_diag,
    uint arity,
    uint r_f,
    uint r_p)
{
    const ulong m00 = ext_mds[0], m01 = ext_mds[1], m02 = ext_mds[2];
    const ulong m10 = ext_mds[3], m11 = ext_mds[4], m12 = ext_mds[5];
    const ulong m20 = ext_mds[6], m21 = ext_mds[7], m22 = ext_mds[8];

    const ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2];

    if (arity <= 2u) {
        APPLY_MDS3_INIT2();
    } else {
        APPLY_MDS3();
    }

    uint half_f = r_f >> 1u;

    for (uint r = 0u; r < half_f; ++r) {
        uint o = r * 3u;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        APPLY_MDS3();
    }

    for (uint r = 0u; r < r_p; ++r) {
        x0 = sbox7(gold_add(x0, rc_int[r]));
        APPLY_INT3();
    }

    for (uint r = half_f; r < r_f; ++r) {
        uint o = r * 3u;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        APPLY_MDS3();
    }

    return x0;
}

inline ulong poseidon2_t3_jplusi(
    ulong x0, ulong x1, ulong x2,
    device const ulong *rc_ext,
    device const ulong *rc_int,
    device const ulong *int_diag,
    uint r_f,
    uint r_p)
{
    const ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2];

    APPLY_MDS3_JPLUSI();

    uint half_f = r_f >> 1u;

    for (uint r = 0u; r < half_f; ++r) {
        uint o = r * 3u;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        APPLY_MDS3_JPLUSI();
    }

    for (uint r = 0u; r < r_p; ++r) {
        x0 = sbox7(gold_add(x0, rc_int[r]));
        APPLY_INT3();
    }

    for (uint r = half_f; r < r_f; ++r) {
        uint o = r * 3u;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        APPLY_MDS3_JPLUSI();
    }

    return x0;
}

inline ulong poseidon2_t3_jdiag(
    ulong x0, ulong x1, ulong x2,
    device const ulong *rc_ext,
    device const ulong *rc_int,
    device const ulong *int_diag,
    ulong e0, ulong e1, ulong e2,
    uint r_f,
    uint r_p)
{
    const ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2];

    APPLY_MDS3_JDIAG();

    uint half_f = r_f >> 1u;

    for (uint r = 0u; r < half_f; ++r) {
        uint o = r * 3u;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        APPLY_MDS3_JDIAG();
    }

    for (uint r = 0u; r < r_p; ++r) {
        x0 = sbox7(gold_add(x0, rc_int[r]));
        APPLY_INT3();
    }

    for (uint r = half_f; r < r_f; ++r) {
        uint o = r * 3u;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        APPLY_MDS3_JDIAG();
    }

    return x0;
}

inline ulong poseidon2_t4_generic(
    ulong x0, ulong x1, ulong x2, ulong x3,
    device const ulong *rc_ext,
    device const ulong *rc_int,
    device const ulong *ext_mds,
    device const ulong *int_diag,
    uint arity,
    uint r_f,
    uint r_p)
{
    const ulong m00 = ext_mds[0],  m01 = ext_mds[1],  m02 = ext_mds[2],  m03 = ext_mds[3];
    const ulong m10 = ext_mds[4],  m11 = ext_mds[5],  m12 = ext_mds[6],  m13 = ext_mds[7];
    const ulong m20 = ext_mds[8],  m21 = ext_mds[9],  m22 = ext_mds[10], m23 = ext_mds[11];
    const ulong m30 = ext_mds[12], m31 = ext_mds[13], m32 = ext_mds[14], m33 = ext_mds[15];

    const ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2], d3 = int_diag[3];

    if (arity <= 2u) {
        APPLY_MDS4_INIT2();
    } else {
        APPLY_MDS4();
    }

    uint half_f = r_f >> 1u;

    for (uint r = 0u; r < half_f; ++r) {
        uint o = r << 2;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        x3 = sbox7(gold_add(x3, rc_ext[o + 3u]));
        APPLY_MDS4();
    }

    for (uint r = 0u; r < r_p; ++r) {
        x0 = sbox7(gold_add(x0, rc_int[r]));
        APPLY_INT4();
    }

    for (uint r = half_f; r < r_f; ++r) {
        uint o = r << 2;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        x3 = sbox7(gold_add(x3, rc_ext[o + 3u]));
        APPLY_MDS4();
    }

    return x0;
}

inline ulong poseidon2_t4_m4(
    ulong x0, ulong x1, ulong x2, ulong x3,
    device const ulong *rc_ext,
    device const ulong *rc_int,
    device const ulong *int_diag,
    uint r_f,
    uint r_p)
{
    const ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2], d3 = int_diag[3];

    APPLY_MDS4_M4();

    uint half_f = r_f >> 1u;

    for (uint r = 0u; r < half_f; ++r) {
        uint o = r << 2;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        x3 = sbox7(gold_add(x3, rc_ext[o + 3u]));
        APPLY_MDS4_M4();
    }

    for (uint r = 0u; r < r_p; ++r) {
        x0 = sbox7(gold_add(x0, rc_int[r]));
        APPLY_INT4();
    }

    for (uint r = half_f; r < r_f; ++r) {
        uint o = r << 2;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        x3 = sbox7(gold_add(x3, rc_ext[o + 3u]));
        APPLY_MDS4_M4();
    }

    return x0;
}

kernel void merkle_build_level(
    device       ulong *tree         [[buffer(0)]],
    device const ulong *rc_ext       [[buffer(1)]],
    device const ulong *rc_int       [[buffer(2)]],
    device const ulong *ext_mds      [[buffer(3)]],
    device const ulong *int_diag     [[buffer(4)]],
    constant uint      &arity        [[buffer(5)]],
    constant uint      &t            [[buffer(6)]],
    constant uint      &r_f          [[buffer(7)]],
    constant uint      &r_p          [[buffer(8)]],
    constant uint      &in_offset    [[buffer(9)]],
    constant uint      &out_offset   [[buffer(10)]],
    constant uint      &child_count  [[buffer(11)]],
    uint p [[thread_position_in_grid]])
{
    if (t == 3u && arity == 2u) {
        uint parent_count = (child_count + 1u) >> 1;
        if (p >= parent_count) return;

        uint base = p << 1;
        uint idx = in_offset + base;

        ulong x0 = tree[idx];
        ulong x1;
        if ((child_count & 1u) == 0u) {
            x1 = tree[idx + 1u];
        } else {
            x1 = (base + 1u < child_count) ? tree[idx + 1u] : 0ul;
        }

        if (mds3_has_unit_offdiag(ext_mds)) {
            ulong e0 = gold_sub_one(ext_mds[0]);
            ulong e1 = gold_sub_one(ext_mds[4]);
            ulong e2 = gold_sub_one(ext_mds[8]);
            ulong d0 = int_diag[0];
            ulong d1 = int_diag[1];
            ulong d2 = int_diag[2];

            if (is_easy_const(e0) && is_easy_const(e1) && is_easy_const(e2) &&
                is_easy_const(d0) && is_easy_const(d1) && is_easy_const(d2)) {
                tree[out_offset + p] = poseidon2_t3_unit_g_easy(
                    x0, x1, rc_ext, rc_int, e0, e1, e2, d0, d1, d2, r_f, r_p);
                return;
            }
        }

        ulong x2 = 0ul;
        ulong out;
        if (mds3_is_jplusi(ext_mds)) {
            out = poseidon2_t3_jplusi(x0, x1, x2, rc_ext, rc_int, int_diag, r_f, r_p);
        } else if (mds3_has_unit_offdiag(ext_mds)) {
            ulong e0 = gold_sub_one(ext_mds[0]);
            ulong e1 = gold_sub_one(ext_mds[4]);
            ulong e2 = gold_sub_one(ext_mds[8]);
            out = poseidon2_t3_jdiag(x0, x1, x2, rc_ext, rc_int, int_diag, e0, e1, e2, r_f, r_p);
        } else {
            out = poseidon2_t3_generic(x0, x1, x2, rc_ext, rc_int, ext_mds, int_diag, 2u, r_f, r_p);
        }

        tree[out_offset + p] = out;
        return;
    }

    uint parent_count;
    if (arity == 2u) {
        parent_count = (child_count + 1u) >> 1;
    } else if (arity == 4u) {
        parent_count = (child_count + 3u) >> 2;
    } else {
        parent_count = (child_count + arity - 1u) / arity;
    }

    if (p >= parent_count) return;

    uint base;
    if (arity == 2u) {
        base = p << 1;
    } else if (arity == 4u) {
        base = p << 2;
    } else {
        base = p * arity;
    }

    uint idx = in_offset + base;

    if (t == 3u) {
        ulong x0 = tree[idx];
        ulong x1 = 0ul;
        ulong x2 = 0ul;

        if (arity > 1u && base + 1u < child_count) x1 = tree[idx + 1u];
        if (arity > 2u && base + 2u < child_count) x2 = tree[idx + 2u];

        ulong out;
        if (mds3_is_jplusi(ext_mds)) {
            out = poseidon2_t3_jplusi(x0, x1, x2, rc_ext, rc_int, int_diag, r_f, r_p);
        } else if (mds3_has_unit_offdiag(ext_mds)) {
            ulong e0 = gold_sub_one(ext_mds[0]);
            ulong e1 = gold_sub_one(ext_mds[4]);
            ulong e2 = gold_sub_one(ext_mds[8]);
            out = poseidon2_t3_jdiag(x0, x1, x2, rc_ext, rc_int, int_diag, e0, e1, e2, r_f, r_p);
        } else {
            out = poseidon2_t3_generic(x0, x1, x2, rc_ext, rc_int, ext_mds, int_diag, arity, r_f, r_p);
        }

        tree[out_offset + p] = out;
    } else {
        ulong x0 = tree[idx];
        ulong x1 = 0ul;
        ulong x2 = 0ul;
        ulong x3 = 0ul;

        if (arity == 4u && ((child_count & 3u) == 0u)) {
            x1 = tree[idx + 1u];
            x2 = tree[idx + 2u];
            x3 = tree[idx + 3u];
        } else if (arity == 4u && base + 3u < child_count) {
            x1 = tree[idx + 1u];
            x2 = tree[idx + 2u];
            x3 = tree[idx + 3u];
        } else {
            if (arity > 1u && base + 1u < child_count) x1 = tree[idx + 1u];
            if (arity > 2u && base + 2u < child_count) x2 = tree[idx + 2u];
            if (arity > 3u && base + 3u < child_count) x3 = tree[idx + 3u];
        }

        ulong out;
        if (mds4_is_m4(ext_mds)) {
            out = poseidon2_t4_m4(x0, x1, x2, x3, rc_ext, rc_int, int_diag, r_f, r_p);
        } else {
            out = poseidon2_t4_generic(x0, x1, x2, x3, rc_ext, rc_int, ext_mds, int_diag, arity, r_f, r_p);
        }

        tree[out_offset + p] = out;
    }
}
```