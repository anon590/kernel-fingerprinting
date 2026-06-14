#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD      = 0xFFFFFFFF00000001ul;
constant ulong P_MINUS_ONE = 0xFFFFFFFF00000000ul;
constant ulong EPSILON     = 0x00000000FFFFFFFFul;

constant ulong INV2_GOLD   = 0x7FFFFFFF80000001ul;
constant ulong INV4_GOLD   = 0xBFFFFFFF40000001ul;
constant ulong Q4_GOLD     = 0x3FFFFFFFC0000000ul;
constant ulong Q4_2_GOLD   = 0x7FFFFFFF80000000ul;
constant ulong Q4_3_GOLD   = 0xBFFFFFFF40000000ul;

constexpr constant uint POS2_R_F = 8u;
constexpr constant uint POS2_R_P = 22u;

// ----------------------------------------------------------------------
// Goldilocks arithmetic on ulong, used by FRI fold.
// ----------------------------------------------------------------------

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    if (t < a) t += EPSILON;
    if (t >= P_GOLD) t -= P_GOLD;
    return t;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    if (t > a) t -= EPSILON;
    return t;
}

inline ulong gold_div2(ulong x) {
    return (x >> 1) + ((x & 1ul) ? INV2_GOLD : 0ul);
}

inline ulong gold_div4(ulong x) {
    ulong r = x & 3ul;
    ulong k = (0ul - r) & 3ul;

    ulong adj = 0ul;
    if (k == 1ul) {
        adj = Q4_GOLD;
    } else if (k == 2ul) {
        adj = Q4_2_GOLD;
    } else if (k == 3ul) {
        adj = Q4_3_GOLD;
    }

    return (x >> 2) + adj + ((r + k) >> 2);
}

inline ulong gold_reduce_limbs(uint x0, uint x1, uint x2, uint x3) {
    ulong lo = ((ulong)x1 << 32) | (ulong)x0;

    ulong t0 = lo - (ulong)x3;
    if (t0 > lo) t0 -= EPSILON;

    ulong t1 = ((ulong)x2 << 32) - (ulong)x2;

    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;

    return gold_canonical(t2);
}

inline ulong gold_mul(ulong a, ulong b) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    uint p00l = a0 * b0;
    uint p00h = mulhi(a0, b0);
    uint p01l = a0 * b1;
    uint p01h = mulhi(a0, b1);
    uint p10l = a1 * b0;
    uint p10h = mulhi(a1, b0);
    uint p11l = a1 * b1;
    uint p11h = mulhi(a1, b1);

    uint s1 = p00h + p01l;
    uint c1 = (s1 < p00h) ? 1u : 0u;
    uint x1 = s1 + p10l;
    c1 += (x1 < s1) ? 1u : 0u;

    uint s2 = p01h + p10h;
    uint c2 = (s2 < p01h) ? 1u : 0u;
    uint s2b = s2 + p11l;
    c2 += (s2b < s2) ? 1u : 0u;
    uint x2 = s2b + c1;
    c2 += (x2 < s2b) ? 1u : 0u;

    uint x3 = p11h + c2;

    return gold_reduce_limbs(p00l, x1, x2, x3);
}

inline ulong gold_square(ulong a) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);

    uint p00l = a0 * a0;
    uint p00h = mulhi(a0, a0);
    uint p01l = a0 * a1;
    uint p01h = mulhi(a0, a1);
    uint p11l = a1 * a1;
    uint p11h = mulhi(a1, a1);

    uint cross_lo  = p01l << 1;
    uint cross_hi  = (p01h << 1) | (p01l >> 31);
    uint cross_top = p01h >> 31;

    uint x1 = p00h + cross_lo;
    uint c1 = (x1 < p00h) ? 1u : 0u;

    uint s2 = p11l + cross_hi;
    uint c2 = (s2 < p11l) ? 1u : 0u;
    uint x2 = s2 + c1;
    c2 += (x2 < s2) ? 1u : 0u;

    uint x3 = p11h + cross_top + c2;

    return gold_reduce_limbs(p00l, x1, x2, x3);
}

// ----------------------------------------------------------------------
// Goldilocks arithmetic on uint2 limbs, used by Poseidon2.
// uint2.x = low 32 bits, uint2.y = high 32 bits.
// ----------------------------------------------------------------------

inline uint2 fe_from_ulong(ulong x) {
    return uint2((uint)x, (uint)(x >> 32));
}

inline uint2 fe_load_device(device const ulong *p, uint i) {
    ulong x = p[i];
    return uint2((uint)x, (uint)(x >> 32));
}

inline ulong fe_to_ulong(uint2 x) {
    return ((ulong)x.y << 32) | (ulong)x.x;
}

inline bool fe_eq_small(uint2 a, uint v) {
    return (a.x == v) && (a.y == 0u);
}

inline uint2 fe_canonical(uint2 a) {
    uint ge = ((a.y == 0xFFFFFFFFu) && (a.x != 0u)) ? 1u : 0u;
    uint mask = 0u - ge;
    a.x = a.x - (mask & 1u);
    a.y = a.y & ~mask;
    return a;
}

inline uint2 fe_add(uint2 a, uint2 b) {
    uint lo = a.x + b.x;
    uint c0 = (lo < a.x) ? 1u : 0u;

    uint hi = a.y + b.y;
    uint c1 = (hi < a.y) ? 1u : 0u;

    uint hi2 = hi + c0;
    uint c2 = (hi2 < hi) ? 1u : 0u;

    uint carry64 = c1 | c2;

    uint old_lo = lo;
    lo += (0u - carry64);
    hi2 += (lo < old_lo) ? 1u : 0u;

    return fe_canonical(uint2(lo, hi2));
}

inline uint2 fe_add_nc(uint2 a, uint2 b) {
    uint lo = a.x + b.x;
    uint c0 = (lo < a.x) ? 1u : 0u;

    uint hi = a.y + b.y;
    uint c1 = (hi < a.y) ? 1u : 0u;

    uint hi2 = hi + c0;
    uint c2 = (hi2 < hi) ? 1u : 0u;

    uint carry64 = c1 | c2;

    uint old_lo = lo;
    lo += (0u - carry64);
    hi2 += (lo < old_lo) ? 1u : 0u;

    return uint2(lo, hi2);
}

inline uint2 fe_sub(uint2 a, uint2 b) {
    uint lo = a.x - b.x;
    uint b0 = (a.x < b.x) ? 1u : 0u;

    uint hi = a.y - b.y;
    uint b1 = (a.y < b.y) ? 1u : 0u;

    uint hi2 = hi - b0;
    uint b2 = (hi < b0) ? 1u : 0u;

    uint under = b1 | b2;

    uint sub = 0u - under;
    uint old_lo = lo;
    lo -= sub;
    hi2 -= (old_lo < sub) ? 1u : 0u;

    return uint2(lo, hi2);
}

inline uint2 fe_fold_carry_small_nc(uint2 s, uint carry64) {
    uint nz = (carry64 != 0u) ? 1u : 0u;

    uint addlo = 0u - carry64;
    uint addhi = carry64 - nz;

    uint old_lo = s.x;
    s.x += addlo;
    uint c0 = (s.x < old_lo) ? 1u : 0u;

    uint old_hi = s.y;
    s.y += addhi;
    uint overflow64 = (s.y < old_hi) ? 1u : 0u;

    old_hi = s.y;
    s.y += c0;
    overflow64 |= (s.y < old_hi) ? 1u : 0u;

    uint mask = 0u - overflow64;
    old_lo = s.x;
    s.x += mask;
    uint c1 = (s.x < old_lo) ? 1u : 0u;
    s.y += c1;

    return s;
}

inline uint2 fe_fold_carry_small(uint2 s, uint carry64) {
    return fe_canonical(fe_fold_carry_small_nc(s, carry64));
}

inline uint2 fe_add_raw_term_nc(uint2 s, uint carry64, uint2 d) {
    uint old_lo = s.x;
    s.x += d.x;
    uint c_lo = (s.x < old_lo) ? 1u : 0u;

    uint old_hi = s.y;
    s.y += d.y;
    uint ch = (s.y < old_hi) ? 1u : 0u;

    old_hi = s.y;
    s.y += c_lo;
    ch |= (s.y < old_hi) ? 1u : 0u;

    carry64 += ch;
    return fe_fold_carry_small_nc(s, carry64);
}

inline uint2 fe_sum3_nc(uint2 a, uint2 b, uint2 c) {
    uint lo = a.x + b.x;
    uint c_lo = (lo < a.x) ? 1u : 0u;

    uint hi = a.y + b.y;
    uint carry64 = (hi < a.y) ? 1u : 0u;

    uint old_hi = hi;
    hi += c_lo;
    carry64 += (hi < old_hi) ? 1u : 0u;

    uint old_lo = lo;
    lo += c.x;
    c_lo = (lo < old_lo) ? 1u : 0u;

    old_hi = hi;
    hi += c.y;
    uint ch = (hi < old_hi) ? 1u : 0u;

    old_hi = hi;
    hi += c_lo;
    ch |= (hi < old_hi) ? 1u : 0u;

    carry64 += ch;

    return fe_fold_carry_small_nc(uint2(lo, hi), carry64);
}

inline uint2 fe_sum3(uint2 a, uint2 b, uint2 c) {
    return fe_canonical(fe_sum3_nc(a, b, c));
}

inline uint2 fe_sum4_nc(uint2 a, uint2 b, uint2 c, uint2 d) {
    uint lo = a.x + b.x;
    uint c_lo = (lo < a.x) ? 1u : 0u;

    uint hi = a.y + b.y;
    uint carry64 = (hi < a.y) ? 1u : 0u;

    uint old_hi = hi;
    hi += c_lo;
    carry64 += (hi < old_hi) ? 1u : 0u;

    uint old_lo = lo;
    lo += c.x;
    c_lo = (lo < old_lo) ? 1u : 0u;

    old_hi = hi;
    hi += c.y;
    uint ch = (hi < old_hi) ? 1u : 0u;

    old_hi = hi;
    hi += c_lo;
    ch |= (hi < old_hi) ? 1u : 0u;

    carry64 += ch;

    old_lo = lo;
    lo += d.x;
    c_lo = (lo < old_lo) ? 1u : 0u;

    old_hi = hi;
    hi += d.y;
    ch = (hi < old_hi) ? 1u : 0u;

    old_hi = hi;
    hi += c_lo;
    ch |= (hi < old_hi) ? 1u : 0u;

    carry64 += ch;

    return fe_fold_carry_small_nc(uint2(lo, hi), carry64);
}

inline uint2 fe_sum4(uint2 a, uint2 b, uint2 c, uint2 d) {
    return fe_canonical(fe_sum4_nc(a, b, c, d));
}

inline uint2 fe_reduce_limbs_nc(uint x0, uint x1, uint x2, uint x3) {
    uint lo = x0 - x3;
    uint b0 = (x0 < x3) ? 1u : 0u;

    uint hi = x1 - b0;
    uint under = (x1 < b0) ? 1u : 0u;

    uint sub = 0u - under;
    uint old_lo = lo;
    lo -= sub;
    hi -= (old_lo < sub) ? 1u : 0u;

    uint nz = (x2 != 0u) ? 1u : 0u;
    uint t1lo = 0u - x2;
    uint t1hi = x2 - nz;

    uint lo2 = lo + t1lo;
    uint c0 = (lo2 < lo) ? 1u : 0u;

    uint hi2 = hi + t1hi;
    uint c1 = (hi2 < hi) ? 1u : 0u;

    uint hi3 = hi2 + c0;
    uint c2 = (hi3 < hi2) ? 1u : 0u;

    uint carry64 = c1 | c2;

    uint old_lo2 = lo2;
    lo2 += (0u - carry64);
    hi3 += (lo2 < old_lo2) ? 1u : 0u;

    return uint2(lo2, hi3);
}

inline uint2 fe_reduce_limbs(uint x0, uint x1, uint x2, uint x3) {
    return fe_canonical(fe_reduce_limbs_nc(x0, x1, x2, x3));
}

inline uint2 fe_mul_nc(uint2 a, uint2 b) {
    uint a0 = a.x;
    uint a1 = a.y;
    uint b0 = b.x;
    uint b1 = b.y;

    uint p00l = a0 * b0;
    uint p00h = mulhi(a0, b0);
    uint p01l = a0 * b1;
    uint p01h = mulhi(a0, b1);
    uint p10l = a1 * b0;
    uint p10h = mulhi(a1, b0);
    uint p11l = a1 * b1;
    uint p11h = mulhi(a1, b1);

    uint s1 = p00h + p01l;
    uint c1 = (s1 < p00h) ? 1u : 0u;
    uint x1 = s1 + p10l;
    c1 += (x1 < s1) ? 1u : 0u;

    uint s2 = p01h + p10h;
    uint c2 = (s2 < p01h) ? 1u : 0u;
    uint s2b = s2 + p11l;
    c2 += (s2b < s2) ? 1u : 0u;
    uint x2 = s2b + c1;
    c2 += (x2 < s2b) ? 1u : 0u;

    uint x3 = p11h + c2;

    return fe_reduce_limbs_nc(p00l, x1, x2, x3);
}

inline uint2 fe_mul(uint2 a, uint2 b) {
    return fe_canonical(fe_mul_nc(a, b));
}

inline uint2 fe_square_nc(uint2 a) {
    uint a0 = a.x;
    uint a1 = a.y;

    uint p00l = a0 * a0;
    uint p00h = mulhi(a0, a0);
    uint p01l = a0 * a1;
    uint p01h = mulhi(a0, a1);
    uint p11l = a1 * a1;
    uint p11h = mulhi(a1, a1);

    uint cross_lo  = p01l << 1;
    uint cross_hi  = (p01h << 1) | (p01l >> 31);
    uint cross_top = p01h >> 31;

    uint x1 = p00h + cross_lo;
    uint c1 = (x1 < p00h) ? 1u : 0u;

    uint s2 = p11l + cross_hi;
    uint c2 = (s2 < p11l) ? 1u : 0u;
    uint x2 = s2 + c1;
    c2 += (x2 < s2) ? 1u : 0u;

    uint x3 = p11h + cross_top + c2;

    return fe_reduce_limbs_nc(p00l, x1, x2, x3);
}

inline uint2 fe_square(uint2 a) {
    return fe_canonical(fe_square_nc(a));
}

inline uint2 fe_sbox7(uint2 x) {
    uint2 x2 = fe_square_nc(x);
    uint2 x4 = fe_square_nc(x2);
    uint2 x6 = fe_mul_nc(x4, x2);
    return fe_mul_nc(x6, x);
}

// ----------------------------------------------------------------------
// FRI fold
// ----------------------------------------------------------------------

kernel void fri_fold(
    device const ulong *evals_in     [[buffer(0)]],
    device       ulong *evals_out    [[buffer(1)]],
    device const ulong *inv_x_base   [[buffer(2)]],
    device const ulong *zeta_inv_pow [[buffer(3)]],
    constant ulong     &alpha        [[buffer(4)]],
    constant ulong     &inv_fold     [[buffer(5)]],
    constant uint      &fold         [[buffer(6)]],
    constant uint      &n_out        [[buffer(7)]],
    uint j [[thread_position_in_grid]])
{
    if (j >= n_out) return;

    ulong ax = gold_mul(alpha, inv_x_base[j]);

    if (fold == 2u) {
        ulong e0 = evals_in[j];
        ulong e1 = evals_in[j + n_out];

        ulong c0 = gold_add(e0, e1);
        ulong c1 = gold_sub(e0, e1);
        ulong acc = gold_add(c0, gold_mul(ax, c1));

        evals_out[j] = gold_div2(acc);
        return;
    }

    if (fold == 4u) {
        ulong z1 = zeta_inv_pow[1];

        ulong e0 = evals_in[j];
        ulong e1 = evals_in[j + n_out];
        ulong e2 = evals_in[j + 2u * n_out];
        ulong e3 = evals_in[j + 3u * n_out];

        ulong e1z = gold_mul(e1, z1);
        ulong e3z = gold_mul(e3, z1);

        ulong c0 = gold_add(gold_add(e0, e1), gold_add(e2, e3));
        ulong c1 = gold_sub(gold_sub(gold_add(e0, e1z), e2), e3z);
        ulong c2 = gold_add(gold_sub(e0, e1), gold_sub(e2, e3));
        ulong c3 = gold_add(gold_sub(gold_sub(e0, e1z), e2), e3z);

        ulong ax2 = gold_square(ax);
        ulong ax3 = gold_mul(ax2, ax);

        ulong acc = c0;
        acc = gold_add(acc, gold_mul(ax,  c1));
        acc = gold_add(acc, gold_mul(ax2, c2));
        acc = gold_add(acc, gold_mul(ax3, c3));

        evals_out[j] = gold_div4(acc);
        return;
    }

    ulong acc = 0ul;
    for (uint m = 0u; m < fold; ++m) {
        ulong rm = (m == 0u) ? ax : gold_mul(ax, zeta_inv_pow[m]);

        ulong sm = 1ul;
        ulong rpow = rm;
        for (uint q = 1u; q < fold; ++q) {
            sm = gold_add(sm, rpow);
            if (q + 1u < fold) rpow = gold_mul(rpow, rm);
        }

        acc = gold_add(acc, gold_mul(evals_in[j + m * n_out], sm));
    }

    if (inv_fold == 1ul) {
        evals_out[j] = acc;
    } else if (fold == 2u && inv_fold == INV2_GOLD) {
        evals_out[j] = gold_div2(acc);
    } else if (fold == 4u && inv_fold == INV4_GOLD) {
        evals_out[j] = gold_div4(acc);
    } else {
        evals_out[j] = gold_mul(acc, inv_fold);
    }
}

// ----------------------------------------------------------------------
// Poseidon2-t=3 Merkle commit.
// ----------------------------------------------------------------------

#define FE_MDS_JPLUSI_FAST() do {                       \
    uint2 ma0 = s0;                                     \
    uint2 ma1 = s1;                                     \
    uint2 ma2 = s2;                                     \
    uint lo = ma0.x + ma1.x;                            \
    uint c_lo = (lo < ma0.x) ? 1u : 0u;                 \
    uint hi = ma0.y + ma1.y;                            \
    uint carry64 = (hi < ma0.y) ? 1u : 0u;              \
    uint old_hi = hi;                                   \
    hi += c_lo;                                         \
    carry64 += (hi < old_hi) ? 1u : 0u;                 \
    uint old_lo = lo;                                   \
    lo += ma2.x;                                        \
    c_lo = (lo < old_lo) ? 1u : 0u;                     \
    old_hi = hi;                                        \
    hi += ma2.y;                                        \
    uint ch = (hi < old_hi) ? 1u : 0u;                  \
    old_hi = hi;                                        \
    hi += c_lo;                                         \
    ch |= (hi < old_hi) ? 1u : 0u;                      \
    carry64 += ch;                                      \
    uint2 raw = uint2(lo, hi);                          \
    uint rawc = carry64;                                \
    s0 = fe_add_raw_term_nc(raw, rawc, ma0);            \
    s1 = fe_add_raw_term_nc(raw, rawc, ma1);            \
    s2 = fe_add_raw_term_nc(raw, rawc, ma2);            \
} while (0)

#define FE_MDS_EXT_JPLUSI()                             \
    FE_MDS_JPLUSI_FAST()

#define FE_MDS_EXT_JPLUSI_S0()                          \
    (fe_sum4(s0, s0, s1, s2))

#define FE_MDS_EXT_FALLBACK() do {                      \
    uint2 ma0 = s0;                                     \
    uint2 ma1 = s1;                                     \
    uint2 ma2 = s2;                                     \
    if (ext_is_jdiag) {                                 \
        uint2 msum = fe_sum3_nc(ma0, ma1, ma2);         \
        s0 = fe_add_nc(msum, fe_mul_nc(ed0, ma0));      \
        s1 = fe_add_nc(msum, fe_mul_nc(ed1, ma1));      \
        s2 = fe_add_nc(msum, fe_mul_nc(ed2, ma2));      \
    } else {                                            \
        s0 = fe_sum3_nc(fe_mul_nc(m00f, ma0),           \
                        fe_mul_nc(m01f, ma1),           \
                        fe_mul_nc(m02f, ma2));          \
        s1 = fe_sum3_nc(fe_mul_nc(m10f, ma0),           \
                        fe_mul_nc(m11f, ma1),           \
                        fe_mul_nc(m12f, ma2));          \
        s2 = fe_sum3_nc(fe_mul_nc(m20f, ma0),           \
                        fe_mul_nc(m21f, ma1),           \
                        fe_mul_nc(m22f, ma2));          \
    }                                                   \
} while (0)

#define FE_MDS_EXT_FALLBACK_S0()                         \
    (ext_is_jdiag ?                                      \
        fe_add(fe_sum3(s0, s1, s2), fe_mul_nc(ed0, s0)) :\
        fe_sum3(fe_mul_nc(m00f, s0),                    \
                fe_mul_nc(m01f, s1),                    \
                fe_mul_nc(m02f, s2)))

#define FE_MDS_INT_GENERIC() do {                       \
    uint2 ma0 = s0;                                     \
    uint2 ma1 = s1;                                     \
    uint2 ma2 = s2;                                     \
    uint2 msum = fe_sum3_nc(ma0, ma1, ma2);             \
    s0 = fe_add_nc(msum, fe_mul_nc(d0, ma0));           \
    s1 = fe_add_nc(msum, fe_mul_nc(d1, ma1));           \
    s2 = fe_add_nc(msum, fe_mul_nc(d2, ma2));           \
} while (0)

#define FE_MDS_INT_JPLUSI()                             \
    FE_MDS_JPLUSI_FAST()

#define FE_MDS_INT_J() do {                             \
    uint2 msum = fe_sum3_nc(s0, s1, s2);                \
    s0 = msum; s1 = msum; s2 = msum;                    \
} while (0)

inline ulong poseidon2_hash_t3_fe(
    ulong in0,
    ulong in1,
    threadgroup const uint2 *rc_ext,
    threadgroup const uint2 *rc_int,
    threadgroup const uint2 *ext_mds,
    threadgroup const uint2 *int_diag)
{
    uint2 s0 = fe_from_ulong(in0);
    uint2 s1 = fe_from_ulong(in1);
    uint2 s2 = uint2(0u, 0u);

    uint2 m00f = ext_mds[0];
    uint2 m01f = ext_mds[1];
    uint2 m02f = ext_mds[2];
    uint2 m10f = ext_mds[3];
    uint2 m11f = ext_mds[4];
    uint2 m12f = ext_mds[5];
    uint2 m20f = ext_mds[6];
    uint2 m21f = ext_mds[7];
    uint2 m22f = ext_mds[8];

    bool offdiag_ones =
        fe_eq_small(m01f, 1u) && fe_eq_small(m02f, 1u) &&
        fe_eq_small(m10f, 1u) && fe_eq_small(m12f, 1u) &&
        fe_eq_small(m20f, 1u) && fe_eq_small(m21f, 1u);

    bool ext_is_jplusi =
        offdiag_ones &&
        fe_eq_small(m00f, 2u) && fe_eq_small(m11f, 2u) && fe_eq_small(m22f, 2u);

    uint2 d0 = int_diag[0];
    uint2 d1 = int_diag[1];
    uint2 d2 = int_diag[2];

    bool int_is_jplusi = fe_eq_small(d0, 1u) && fe_eq_small(d1, 1u) && fe_eq_small(d2, 1u);
    bool int_is_j      = fe_eq_small(d0, 0u) && fe_eq_small(d1, 0u) && fe_eq_small(d2, 0u);

    if (ext_is_jplusi) {
        FE_MDS_EXT_JPLUSI();

#pragma unroll
        for (uint r = 0u; r < POS2_R_F / 2u; ++r) {
            uint k = r * 3u;
            s0 = fe_sbox7(fe_add_nc(s0, rc_ext[k + 0u]));
            s1 = fe_sbox7(fe_add_nc(s1, rc_ext[k + 1u]));
            s2 = fe_sbox7(fe_add_nc(s2, rc_ext[k + 2u]));
            FE_MDS_EXT_JPLUSI();
        }

        if (int_is_jplusi) {
#pragma unroll
            for (uint r = 0u; r < POS2_R_P; ++r) {
                s0 = fe_sbox7(fe_add_nc(s0, rc_int[r]));
                FE_MDS_INT_JPLUSI();
            }
        } else if (int_is_j) {
#pragma unroll
            for (uint r = 0u; r < POS2_R_P; ++r) {
                s0 = fe_sbox7(fe_add_nc(s0, rc_int[r]));
                FE_MDS_INT_J();
            }
        } else {
#pragma unroll
            for (uint r = 0u; r < POS2_R_P; ++r) {
                s0 = fe_sbox7(fe_add_nc(s0, rc_int[r]));
                FE_MDS_INT_GENERIC();
            }
        }

#pragma unroll
        for (uint r = POS2_R_F / 2u; r < POS2_R_F - 1u; ++r) {
            uint k = r * 3u;
            s0 = fe_sbox7(fe_add_nc(s0, rc_ext[k + 0u]));
            s1 = fe_sbox7(fe_add_nc(s1, rc_ext[k + 1u]));
            s2 = fe_sbox7(fe_add_nc(s2, rc_ext[k + 2u]));
            FE_MDS_EXT_JPLUSI();
        }

        uint k = (POS2_R_F - 1u) * 3u;
        s0 = fe_sbox7(fe_add_nc(s0, rc_ext[k + 0u]));
        s1 = fe_sbox7(fe_add_nc(s1, rc_ext[k + 1u]));
        s2 = fe_sbox7(fe_add_nc(s2, rc_ext[k + 2u]));
        return fe_to_ulong(FE_MDS_EXT_JPLUSI_S0());
    }

    bool ext_is_jdiag = offdiag_ones;

    uint2 one = uint2(1u, 0u);
    uint2 ed0 = fe_sub(m00f, one);
    uint2 ed1 = fe_sub(m11f, one);
    uint2 ed2 = fe_sub(m22f, one);

    FE_MDS_EXT_FALLBACK();

#pragma unroll
    for (uint r = 0u; r < POS2_R_F / 2u; ++r) {
        uint k = r * 3u;
        s0 = fe_sbox7(fe_add_nc(s0, rc_ext[k + 0u]));
        s1 = fe_sbox7(fe_add_nc(s1, rc_ext[k + 1u]));
        s2 = fe_sbox7(fe_add_nc(s2, rc_ext[k + 2u]));
        FE_MDS_EXT_FALLBACK();
    }

    if (int_is_jplusi) {
#pragma unroll
        for (uint r = 0u; r < POS2_R_P; ++r) {
            s0 = fe_sbox7(fe_add_nc(s0, rc_int[r]));
            FE_MDS_INT_JPLUSI();
        }
    } else if (int_is_j) {
#pragma unroll
        for (uint r = 0u; r < POS2_R_P; ++r) {
            s0 = fe_sbox7(fe_add_nc(s0, rc_int[r]));
            FE_MDS_INT_J();
        }
    } else {
#pragma unroll
        for (uint r = 0u; r < POS2_R_P; ++r) {
            s0 = fe_sbox7(fe_add_nc(s0, rc_int[r]));
            FE_MDS_INT_GENERIC();
        }
    }

#pragma unroll
    for (uint r = POS2_R_F / 2u; r < POS2_R_F - 1u; ++r) {
        uint k = r * 3u;
        s0 = fe_sbox7(fe_add_nc(s0, rc_ext[k + 0u]));
        s1 = fe_sbox7(fe_add_nc(s1, rc_ext[k + 1u]));
        s2 = fe_sbox7(fe_add_nc(s2, rc_ext[k + 2u]));
        FE_MDS_EXT_FALLBACK();
    }

    uint k = (POS2_R_F - 1u) * 3u;
    s0 = fe_sbox7(fe_add_nc(s0, rc_ext[k + 0u]));
    s1 = fe_sbox7(fe_add_nc(s1, rc_ext[k + 1u]));
    s2 = fe_sbox7(fe_add_nc(s2, rc_ext[k + 2u]));
    return fe_to_ulong(FE_MDS_EXT_FALLBACK_S0());
}

inline ulong poseidon2_hash_t3_dev(
    ulong in0,
    ulong in1,
    device const ulong *rc_ext,
    device const ulong *rc_int,
    device const ulong *ext_mds,
    device const ulong *int_diag)
{
    uint2 s0 = fe_from_ulong(in0);
    uint2 s1 = fe_from_ulong(in1);
    uint2 s2 = uint2(0u, 0u);

    uint2 m00f = fe_load_device(ext_mds, 0u);
    uint2 m01f = fe_load_device(ext_mds, 1u);
    uint2 m02f = fe_load_device(ext_mds, 2u);
    uint2 m10f = fe_load_device(ext_mds, 3u);
    uint2 m11f = fe_load_device(ext_mds, 4u);
    uint2 m12f = fe_load_device(ext_mds, 5u);
    uint2 m20f = fe_load_device(ext_mds, 6u);
    uint2 m21f = fe_load_device(ext_mds, 7u);
    uint2 m22f = fe_load_device(ext_mds, 8u);

    bool offdiag_ones =
        fe_eq_small(m01f, 1u) && fe_eq_small(m02f, 1u) &&
        fe_eq_small(m10f, 1u) && fe_eq_small(m12f, 1u) &&
        fe_eq_small(m20f, 1u) && fe_eq_small(m21f, 1u);

    bool ext_is_jplusi =
        offdiag_ones &&
        fe_eq_small(m00f, 2u) && fe_eq_small(m11f, 2u) && fe_eq_small(m22f, 2u);

    uint2 d0 = fe_load_device(int_diag, 0u);
    uint2 d1 = fe_load_device(int_diag, 1u);
    uint2 d2 = fe_load_device(int_diag, 2u);

    bool int_is_jplusi = fe_eq_small(d0, 1u) && fe_eq_small(d1, 1u) && fe_eq_small(d2, 1u);
    bool int_is_j      = fe_eq_small(d0, 0u) && fe_eq_small(d1, 0u) && fe_eq_small(d2, 0u);

    if (ext_is_jplusi) {
        FE_MDS_EXT_JPLUSI();

#pragma unroll
        for (uint r = 0u; r < POS2_R_F / 2u; ++r) {
            uint k = r * 3u;
            s0 = fe_sbox7(fe_add_nc(s0, fe_load_device(rc_ext, k + 0u)));
            s1 = fe_sbox7(fe_add_nc(s1, fe_load_device(rc_ext, k + 1u)));
            s2 = fe_sbox7(fe_add_nc(s2, fe_load_device(rc_ext, k + 2u)));
            FE_MDS_EXT_JPLUSI();
        }

        if (int_is_jplusi) {
#pragma unroll
            for (uint r = 0u; r < POS2_R_P; ++r) {
                s0 = fe_sbox7(fe_add_nc(s0, fe_load_device(rc_int, r)));
                FE_MDS_INT_JPLUSI();
            }
        } else if (int_is_j) {
#pragma unroll
            for (uint r = 0u; r < POS2_R_P; ++r) {
                s0 = fe_sbox7(fe_add_nc(s0, fe_load_device(rc_int, r)));
                FE_MDS_INT_J();
            }
        } else {
#pragma unroll
            for (uint r = 0u; r < POS2_R_P; ++r) {
                s0 = fe_sbox7(fe_add_nc(s0, fe_load_device(rc_int, r)));
                FE_MDS_INT_GENERIC();
            }
        }

#pragma unroll
        for (uint r = POS2_R_F / 2u; r < POS2_R_F - 1u; ++r) {
            uint k = r * 3u;
            s0 = fe_sbox7(fe_add_nc(s0, fe_load_device(rc_ext, k + 0u)));
            s1 = fe_sbox7(fe_add_nc(s1, fe_load_device(rc_ext, k + 1u)));
            s2 = fe_sbox7(fe_add_nc(s2, fe_load_device(rc_ext, k + 2u)));
            FE_MDS_EXT_JPLUSI();
        }

        uint k = (POS2_R_F - 1u) * 3u;
        s0 = fe_sbox7(fe_add_nc(s0, fe_load_device(rc_ext, k + 0u)));
        s1 = fe_sbox7(fe_add_nc(s1, fe_load_device(rc_ext, k + 1u)));
        s2 = fe_sbox7(fe_add_nc(s2, fe_load_device(rc_ext, k + 2u)));
        return fe_to_ulong(FE_MDS_EXT_JPLUSI_S0());
    }

    bool ext_is_jdiag = offdiag_ones;

    uint2 one = uint2(1u, 0u);
    uint2 ed0 = fe_sub(m00f, one);
    uint2 ed1 = fe_sub(m11f, one);
    uint2 ed2 = fe_sub(m22f, one);

    FE_MDS_EXT_FALLBACK();

#pragma unroll
    for (uint r = 0u; r < POS2_R_F / 2u; ++r) {
        uint k = r * 3u;
        s0 = fe_sbox7(fe_add_nc(s0, fe_load_device(rc_ext, k + 0u)));
        s1 = fe_sbox7(fe_add_nc(s1, fe_load_device(rc_ext, k + 1u)));
        s2 = fe_sbox7(fe_add_nc(s2, fe_load_device(rc_ext, k + 2u)));
        FE_MDS_EXT_FALLBACK();
    }

    if (int_is_jplusi) {
#pragma unroll
        for (uint r = 0u; r < POS2_R_P; ++r) {
            s0 = fe_sbox7(fe_add_nc(s0, fe_load_device(rc_int, r)));
            FE_MDS_INT_JPLUSI();
        }
    } else if (int_is_j) {
#pragma unroll
        for (uint r = 0u; r < POS2_R_P; ++r) {
            s0 = fe_sbox7(fe_add_nc(s0, fe_load_device(rc_int, r)));
            FE_MDS_INT_J();
        }
    } else {
#pragma unroll
        for (uint r = 0u; r < POS2_R_P; ++r) {
            s0 = fe_sbox7(fe_add_nc(s0, fe_load_device(rc_int, r)));
            FE_MDS_INT_GENERIC();
        }
    }

#pragma unroll
    for (uint r = POS2_R_F / 2u; r < POS2_R_F - 1u; ++r) {
        uint k = r * 3u;
        s0 = fe_sbox7(fe_add_nc(s0, fe_load_device(rc_ext, k + 0u)));
        s1 = fe_sbox7(fe_add_nc(s1, fe_load_device(rc_ext, k + 1u)));
        s2 = fe_sbox7(fe_add_nc(s2, fe_load_device(rc_ext, k + 2u)));
        FE_MDS_EXT_FALLBACK();
    }

    uint k = (POS2_R_F - 1u) * 3u;
    s0 = fe_sbox7(fe_add_nc(s0, fe_load_device(rc_ext, k + 0u)));
    s1 = fe_sbox7(fe_add_nc(s1, fe_load_device(rc_ext, k + 1u)));
    s2 = fe_sbox7(fe_add_nc(s2, fe_load_device(rc_ext, k + 2u)));
    return fe_to_ulong(FE_MDS_EXT_FALLBACK_S0());
}

#undef FE_MDS_JPLUSI_FAST
#undef FE_MDS_EXT_JPLUSI
#undef FE_MDS_EXT_JPLUSI_S0
#undef FE_MDS_EXT_FALLBACK
#undef FE_MDS_EXT_FALLBACK_S0
#undef FE_MDS_INT_GENERIC
#undef FE_MDS_INT_JPLUSI
#undef FE_MDS_INT_J

kernel void fri_commit_level(
    device       ulong *tree         [[buffer(0)]],
    device const ulong *rc_ext       [[buffer(1)]],
    device const ulong *rc_int       [[buffer(2)]],
    device const ulong *ext_mds      [[buffer(3)]],
    device const ulong *int_diag     [[buffer(4)]],
    constant uint      &in_offset    [[buffer(5)]],
    constant uint      &out_offset   [[buffer(6)]],
    constant uint      &child_count  [[buffer(7)]],
    uint p [[thread_position_in_grid]])
{
    uint parent_count = (child_count + 1u) >> 1u;

    if (parent_count < 64u) {
        if (p >= parent_count) return;

        uint base = p << 1u;
        ulong left = tree[in_offset + base];
        ulong right = 0ul;
        if (base + 1u < child_count) {
            right = tree[in_offset + base + 1u];
        }

        tree[out_offset + p] = poseidon2_hash_t3_dev(left, right,
                                                     rc_ext, rc_int,
                                                     ext_mds, int_diag);
        return;
    }

    threadgroup uint2 tg_rc_ext[POS2_R_F * 3u];
    threadgroup uint2 tg_rc_int[POS2_R_P];
    threadgroup uint2 tg_ext_mds[9];
    threadgroup uint2 tg_int_diag[3];

    uint lid = p & 63u;

    for (uint i = lid; i < POS2_R_F * 3u; i += 64u) {
        tg_rc_ext[i] = fe_load_device(rc_ext, i);
    }
    for (uint i = lid; i < POS2_R_P; i += 64u) {
        tg_rc_int[i] = fe_load_device(rc_int, i);
    }
    for (uint i = lid; i < 9u; i += 64u) {
        tg_ext_mds[i] = fe_load_device(ext_mds, i);
    }
    for (uint i = lid; i < 3u; i += 64u) {
        tg_int_diag[i] = fe_load_device(int_diag, i);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (p >= parent_count) return;

    uint base = p << 1u;

    ulong left = tree[in_offset + base];
    ulong right = 0ul;
    if (base + 1u < child_count) {
        right = tree[in_offset + base + 1u];
    }

    tree[out_offset + p] = poseidon2_hash_t3_fe(left, right,
                                                tg_rc_ext, tg_rc_int,
                                                tg_ext_mds, tg_int_diag);
}