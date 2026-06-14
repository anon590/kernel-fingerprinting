#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;

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

inline bool is_tiny_signed_const(ulong c) {
    return (c <= 16ul) || (c >= (P_GOLD - 16ul));
}

inline uint tiny_abs_k(ulong c) {
    return (c <= 16ul) ? (uint)c : (uint)(P_GOLD - c);
}

inline bool tiny_is_neg(ulong c) {
    return c > 16ul;
}

inline ulong gold_mul_pos_tiny(uint k, ulong x) {
    if (k == 0u) return 0ul;
    if (k == 1u) return x;

    ulong r2 = gold_add(x, x);
    if (k == 2u) return r2;

    ulong r3 = gold_add(r2, x);
    if (k == 3u) return r3;

    ulong r4 = gold_add(r2, r2);
    if (k == 4u) return r4;

    ulong r5 = gold_add(r4, x);
    if (k == 5u) return r5;

    ulong r6 = gold_add(r4, r2);
    if (k == 6u) return r6;

    ulong r7 = gold_add(r6, x);
    if (k == 7u) return r7;

    ulong r8 = gold_add(r4, r4);
    if (k == 8u) return r8;

    ulong r9 = gold_add(r8, x);
    if (k == 9u) return r9;

    ulong r10 = gold_add(r8, r2);
    if (k == 10u) return r10;

    ulong r11 = gold_add(r10, x);
    if (k == 11u) return r11;

    ulong r12 = gold_add(r8, r4);
    if (k == 12u) return r12;

    ulong r13 = gold_add(r12, x);
    if (k == 13u) return r13;

    ulong r14 = gold_add(r12, r2);
    if (k == 14u) return r14;

    ulong r15 = gold_add(r14, x);
    if (k == 15u) return r15;

    return gold_add(r8, r8);
}

inline ulong gold_mul_tiny_kn(uint k, bool neg, ulong x) {
    ulong r = gold_mul_pos_tiny(k, x);
    return neg ? gold_neg(r) : r;
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

#define APPLY_MDS3_JDIAG_TINY() do {                           \
    ulong s = gold_add(gold_add(x0, x1), x2);                  \
    ulong y0 = gold_add(s, gold_mul_tiny_kn(e0k, e0n, x0));    \
    ulong y1 = gold_add(s, gold_mul_tiny_kn(e1k, e1n, x1));    \
    ulong y2 = gold_add(s, gold_mul_tiny_kn(e2k, e2n, x2));    \
    x0 = y0; x1 = y1; x2 = y2;                                 \
} while (false)

#define APPLY_INT3() do {                                      \
    ulong s = gold_add(gold_add(x0, x1), x2);                  \
    ulong y0 = gold_add(s, gold_mul(d0, x0));                  \
    ulong y1 = gold_add(s, gold_mul(d1, x1));                  \
    ulong y2 = gold_add(s, gold_mul(d2, x2));                  \
    x0 = y0; x1 = y1; x2 = y2;                                 \
} while (false)

#define APPLY_INT3_TINY() do {                                 \
    ulong s = gold_add(gold_add(x0, x1), x2);                  \
    ulong y0 = gold_add(s, gold_mul_tiny_kn(d0k, d0n, x0));    \
    ulong y1 = gold_add(s, gold_mul_tiny_kn(d1k, d1n, x1));    \
    ulong y2 = gold_add(s, gold_mul_tiny_kn(d2k, d2n, x2));    \
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

#define APPLY_INT4_TINY() do {                                 \
    ulong s = gold_add(gold_add(gold_add(x0, x1), x2), x3);    \
    ulong y0 = gold_add(s, gold_mul_tiny_kn(d0k, d0n, x0));    \
    ulong y1 = gold_add(s, gold_mul_tiny_kn(d1k, d1n, x1));    \
    ulong y2 = gold_add(s, gold_mul_tiny_kn(d2k, d2n, x2));    \
    ulong y3 = gold_add(s, gold_mul_tiny_kn(d3k, d3n, x3));    \
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

inline ulong poseidon2_t3_jdiag_tiny(
    ulong x0, ulong x1, ulong x2,
    device const ulong *rc_ext,
    device const ulong *rc_int,
    ulong e0, ulong e1, ulong e2,
    ulong d0v, ulong d1v, ulong d2v,
    uint r_f,
    uint r_p)
{
    const uint e0k = tiny_abs_k(e0), e1k = tiny_abs_k(e1), e2k = tiny_abs_k(e2);
    const bool e0n = tiny_is_neg(e0), e1n = tiny_is_neg(e1), e2n = tiny_is_neg(e2);

    const uint d0k = tiny_abs_k(d0v), d1k = tiny_abs_k(d1v), d2k = tiny_abs_k(d2v);
    const bool d0n = tiny_is_neg(d0v), d1n = tiny_is_neg(d1v), d2n = tiny_is_neg(d2v);

    APPLY_MDS3_JDIAG_TINY();

    uint half_f = r_f >> 1u;

    for (uint r = 0u; r < half_f; ++r) {
        uint o = r * 3u;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        APPLY_MDS3_JDIAG_TINY();
    }

    for (uint r = 0u; r < r_p; ++r) {
        x0 = sbox7(gold_add(x0, rc_int[r]));
        APPLY_INT3_TINY();
    }

    for (uint r = half_f; r < r_f; ++r) {
        uint o = r * 3u;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        APPLY_MDS3_JDIAG_TINY();
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

inline ulong poseidon2_t4_m4_tiny(
    ulong x0, ulong x1, ulong x2, ulong x3,
    device const ulong *rc_ext,
    device const ulong *rc_int,
    ulong d0v, ulong d1v, ulong d2v, ulong d3v,
    uint r_f,
    uint r_p)
{
    const uint d0k = tiny_abs_k(d0v), d1k = tiny_abs_k(d1v), d2k = tiny_abs_k(d2v), d3k = tiny_abs_k(d3v);
    const bool d0n = tiny_is_neg(d0v), d1n = tiny_is_neg(d1v), d2n = tiny_is_neg(d2v), d3n = tiny_is_neg(d3v);

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
        APPLY_INT4_TINY();
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
        ulong x2 = 0ul;

        ulong out;
        if (mds3_has_unit_offdiag(ext_mds)) {
            ulong e0 = gold_sub_one(ext_mds[0]);
            ulong e1 = gold_sub_one(ext_mds[4]);
            ulong e2 = gold_sub_one(ext_mds[8]);
            ulong d0v = int_diag[0];
            ulong d1v = int_diag[1];
            ulong d2v = int_diag[2];

            if (is_tiny_signed_const(e0) && is_tiny_signed_const(e1) && is_tiny_signed_const(e2) &&
                is_tiny_signed_const(d0v) && is_tiny_signed_const(d1v) && is_tiny_signed_const(d2v)) {
                out = poseidon2_t3_jdiag_tiny(x0, x1, x2, rc_ext, rc_int, e0, e1, e2,
                                               d0v, d1v, d2v, r_f, r_p);
                tree[out_offset + p] = out;
                return;
            }
        }

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
        if (mds3_has_unit_offdiag(ext_mds)) {
            ulong e0 = gold_sub_one(ext_mds[0]);
            ulong e1 = gold_sub_one(ext_mds[4]);
            ulong e2 = gold_sub_one(ext_mds[8]);
            ulong d0v = int_diag[0];
            ulong d1v = int_diag[1];
            ulong d2v = int_diag[2];

            if (is_tiny_signed_const(e0) && is_tiny_signed_const(e1) && is_tiny_signed_const(e2) &&
                is_tiny_signed_const(d0v) && is_tiny_signed_const(d1v) && is_tiny_signed_const(d2v)) {
                out = poseidon2_t3_jdiag_tiny(x0, x1, x2, rc_ext, rc_int, e0, e1, e2,
                                               d0v, d1v, d2v, r_f, r_p);
                tree[out_offset + p] = out;
                return;
            }
        }

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
            ulong d0v = int_diag[0], d1v = int_diag[1], d2v = int_diag[2], d3v = int_diag[3];
            if (is_tiny_signed_const(d0v) && is_tiny_signed_const(d1v) &&
                is_tiny_signed_const(d2v) && is_tiny_signed_const(d3v)) {
                out = poseidon2_t4_m4_tiny(x0, x1, x2, x3, rc_ext, rc_int,
                                            d0v, d1v, d2v, d3v, r_f, r_p);
            } else {
                out = poseidon2_t4_m4(x0, x1, x2, x3, rc_ext, rc_int, int_diag, r_f, r_p);
            }
        } else {
            out = poseidon2_t4_generic(x0, x1, x2, x3, rc_ext, rc_int, ext_mds, int_diag, arity, r_f, r_p);
        }

        tree[out_offset + p] = out;
    }
}