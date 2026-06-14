#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_LIMBS = 6u;
constexpr constant uint POINT_LIMBS = 18u;
constexpr constant uint SCALAR_BITS = 256u;
constexpr constant uint W4_TABLE = 8u;

// ------------------------------------------------------------------
// 64x64 -> 128 using 32-bit products and mulhi(uint,uint).
// ulong2.x = low 64, ulong2.y = high 64.
// ------------------------------------------------------------------
inline ulong2 umul128(ulong a, ulong b) {
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

    ulong mid = (ulong)p00h + (ulong)p01l + (ulong)p10l;
    ulong lo  = (ulong)p00l | (((ulong)((uint)mid)) << 32);
    ulong p11 = ((ulong)p11h << 32) | (ulong)p11l;
    ulong hi  = p11 + (ulong)p01h + (ulong)p10h + (mid >> 32);
    return ulong2(lo, hi);
}

inline ulong2 fma_add128(ulong a, ulong b, ulong t, ulong c) {
    ulong2 prod = umul128(a, b);
    ulong lo1 = prod.x + t;
    ulong cy1 = (lo1 < prod.x) ? 1ul : 0ul;
    ulong hi1 = prod.y + cy1;
    ulong lo2 = lo1 + c;
    ulong cy2 = (lo2 < lo1) ? 1ul : 0ul;
    ulong hi2 = hi1 + cy2;
    return ulong2(lo2, hi2);
}

// ------------------------------------------------------------------
// Limb helpers.
// ------------------------------------------------------------------
inline void copy_n(thread ulong *dst, thread const ulong *src) {
    for (uint i = 0u; i < N_LIMBS; ++i) dst[i] = src[i];
}

inline bool is_zero_n(thread const ulong *a) {
    for (uint i = 0u; i < N_LIMBS; ++i) {
        if (a[i] != 0ul) return false;
    }
    return true;
}

inline bool eq_n(thread const ulong *a, thread const ulong *b) {
    for (uint i = 0u; i < N_LIMBS; ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

inline void mod_add(thread ulong *c,
                    thread const ulong *a,
                    thread const ulong *b,
                    thread const ulong *q)
{
    ulong sum[N_LIMBS];
    ulong carry = 0ul;
    for (uint i = 0u; i < N_LIMBS; ++i) {
        ulong s = a[i] + carry;
        ulong cy1 = (s < a[i]) ? 1ul : 0ul;
        ulong t = s + b[i];
        ulong cy2 = (t < s) ? 1ul : 0ul;
        sum[i] = t;
        carry = cy1 + cy2;
    }

    ulong diff[N_LIMBS];
    ulong borrow = 0ul;
    for (uint i = 0u; i < N_LIMBS; ++i) {
        ulong tv = sum[i] - q[i];
        ulong b1 = (tv > sum[i]) ? 1ul : 0ul;
        ulong d = tv - borrow;
        ulong b2 = (d > tv) ? 1ul : 0ul;
        diff[i] = d;
        borrow = b1 + b2;
    }

    bool use_diff = (carry != 0ul) || (borrow == 0ul);
    for (uint i = 0u; i < N_LIMBS; ++i) {
        c[i] = use_diff ? diff[i] : sum[i];
    }
}

inline void mod_sub(thread ulong *c,
                    thread const ulong *a,
                    thread const ulong *b,
                    thread const ulong *q)
{
    ulong diff[N_LIMBS];
    ulong borrow = 0ul;
    for (uint i = 0u; i < N_LIMBS; ++i) {
        ulong tv = a[i] - b[i];
        ulong b1 = (tv > a[i]) ? 1ul : 0ul;
        ulong d = tv - borrow;
        ulong b2 = (d > tv) ? 1ul : 0ul;
        diff[i] = d;
        borrow = b1 + b2;
    }

    ulong added[N_LIMBS];
    ulong carry = 0ul;
    for (uint i = 0u; i < N_LIMBS; ++i) {
        ulong s = diff[i] + carry;
        ulong cy1 = (s < diff[i]) ? 1ul : 0ul;
        ulong t = s + q[i];
        ulong cy2 = (t < s) ? 1ul : 0ul;
        added[i] = t;
        carry = cy1 + cy2;
    }

    bool use_added = (borrow != 0ul);
    for (uint i = 0u; i < N_LIMBS; ++i) {
        c[i] = use_added ? added[i] : diff[i];
    }
}

inline void mod_neg(thread ulong *c,
                    thread const ulong *a,
                    thread const ulong *q)
{
    ulong diff[N_LIMBS];
    ulong borrow = 0ul;
    bool z = true;

    for (uint i = 0u; i < N_LIMBS; ++i) {
        if (a[i] != 0ul) z = false;
        ulong tv = q[i] - a[i];
        ulong b1 = (tv > q[i]) ? 1ul : 0ul;
        ulong d = tv - borrow;
        ulong b2 = (d > tv) ? 1ul : 0ul;
        diff[i] = d;
        borrow = b1 + b2;
    }

    for (uint i = 0u; i < N_LIMBS; ++i) {
        c[i] = z ? 0ul : diff[i];
    }
}

#define CIOS_ROUND6(BI) do { \
    ulong bi_m = (BI); \
    ulong C_m = 0ul; \
    ulong2 r_m; \
    r_m = fma_add128(a0, bi_m, t0, C_m); t0 = r_m.x; C_m = r_m.y; \
    r_m = fma_add128(a1, bi_m, t1, C_m); t1 = r_m.x; C_m = r_m.y; \
    r_m = fma_add128(a2, bi_m, t2, C_m); t2 = r_m.x; C_m = r_m.y; \
    r_m = fma_add128(a3, bi_m, t3, C_m); t3 = r_m.x; C_m = r_m.y; \
    r_m = fma_add128(a4, bi_m, t4, C_m); t4 = r_m.x; C_m = r_m.y; \
    r_m = fma_add128(a5, bi_m, t5, C_m); t5 = r_m.x; C_m = r_m.y; \
    ulong s_m = t6 + C_m; \
    ulong cy_m = (s_m < t6) ? 1ul : 0ul; \
    t6 = s_m; \
    t7 += cy_m; \
    ulong m_m = t0 * q_inv_neg; \
    C_m = 0ul; \
    r_m = fma_add128(m_m, q0, t0, C_m); t0 = r_m.x; C_m = r_m.y; \
    r_m = fma_add128(m_m, q1, t1, C_m); t1 = r_m.x; C_m = r_m.y; \
    r_m = fma_add128(m_m, q2, t2, C_m); t2 = r_m.x; C_m = r_m.y; \
    r_m = fma_add128(m_m, q3, t3, C_m); t3 = r_m.x; C_m = r_m.y; \
    r_m = fma_add128(m_m, q4, t4, C_m); t4 = r_m.x; C_m = r_m.y; \
    r_m = fma_add128(m_m, q5, t5, C_m); t5 = r_m.x; C_m = r_m.y; \
    s_m = t6 + C_m; \
    cy_m = (s_m < t6) ? 1ul : 0ul; \
    t6 = s_m; \
    t7 += cy_m; \
    t0 = t1; t1 = t2; t2 = t3; t3 = t4; t4 = t5; t5 = t6; t6 = t7; t7 = 0ul; \
} while(false)

// Fixed 6-limb CIOS Montgomery multiplication: out = a*b*R^-1 mod q.
inline void mont_mul(thread ulong *out,
                     thread const ulong *a,
                     thread const ulong *b,
                     thread const ulong *q,
                     ulong q_inv_neg)
{
    ulong a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5];
    ulong b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3], b4 = b[4], b5 = b[5];
    ulong q0 = q[0], q1 = q[1], q2 = q[2], q3 = q[3], q4 = q[4], q5 = q[5];

    ulong t0 = 0ul, t1 = 0ul, t2 = 0ul, t3 = 0ul;
    ulong t4 = 0ul, t5 = 0ul, t6 = 0ul, t7 = 0ul;

    CIOS_ROUND6(b0);
    CIOS_ROUND6(b1);
    CIOS_ROUND6(b2);
    CIOS_ROUND6(b3);
    CIOS_ROUND6(b4);
    CIOS_ROUND6(b5);

    ulong borrow = 0ul;

    ulong tv0 = t0 - q0;
    ulong b10 = (tv0 > t0) ? 1ul : 0ul;
    ulong d0 = tv0 - borrow;
    ulong b20 = (d0 > tv0) ? 1ul : 0ul;
    borrow = b10 + b20;

    ulong tv1 = t1 - q1;
    ulong b11 = (tv1 > t1) ? 1ul : 0ul;
    ulong d1 = tv1 - borrow;
    ulong b21 = (d1 > tv1) ? 1ul : 0ul;
    borrow = b11 + b21;

    ulong tv2 = t2 - q2;
    ulong b12 = (tv2 > t2) ? 1ul : 0ul;
    ulong d2 = tv2 - borrow;
    ulong b22 = (d2 > tv2) ? 1ul : 0ul;
    borrow = b12 + b22;

    ulong tv3 = t3 - q3;
    ulong b13 = (tv3 > t3) ? 1ul : 0ul;
    ulong d3 = tv3 - borrow;
    ulong b23 = (d3 > tv3) ? 1ul : 0ul;
    borrow = b13 + b23;

    ulong tv4 = t4 - q4;
    ulong b14 = (tv4 > t4) ? 1ul : 0ul;
    ulong d4 = tv4 - borrow;
    ulong b24 = (d4 > tv4) ? 1ul : 0ul;
    borrow = b14 + b24;

    ulong tv5 = t5 - q5;
    ulong b15 = (tv5 > t5) ? 1ul : 0ul;
    ulong d5 = tv5 - borrow;
    ulong b25 = (d5 > tv5) ? 1ul : 0ul;
    borrow = b15 + b25;

    bool use_diff = (t6 != 0ul) || (borrow == 0ul);

    out[0] = use_diff ? d0 : t0;
    out[1] = use_diff ? d1 : t1;
    out[2] = use_diff ? d2 : t2;
    out[3] = use_diff ? d3 : t3;
    out[4] = use_diff ? d4 : t4;
    out[5] = use_diff ? d5 : t5;
}

#undef CIOS_ROUND6

inline void add_limb14(thread ulong *t, uint idx, ulong v) {
    if (v == 0ul) return;
    ulong old = t[idx];
    ulong nv = old + v;
    t[idx] = nv;
    ulong carry = (nv < old) ? 1ul : 0ul;
    idx++;
    while (carry != 0ul && idx < 14u) {
        old = t[idx];
        nv = old + 1ul;
        t[idx] = nv;
        carry = (nv == 0ul) ? 1ul : 0ul;
        idx++;
    }
}

inline void acc_add_192(thread ulong &c0,
                        thread ulong &c1,
                        thread ulong &c2,
                        ulong lo,
                        ulong hi,
                        ulong extra)
{
    ulong old0 = c0;
    ulong n0 = old0 + lo;
    ulong carry0 = (n0 < old0) ? 1ul : 0ul;
    c0 = n0;

    ulong old1 = c1;
    ulong n1 = old1 + hi;
    ulong carry1 = (n1 < old1) ? 1ul : 0ul;
    c1 = n1;

    old1 = c1;
    n1 = old1 + carry0;
    ulong carry2 = (n1 < old1) ? 1ul : 0ul;
    c1 = n1;

    c2 += extra + carry1 + carry2;
}

inline void sqr_add_diag(thread ulong &c0,
                         thread ulong &c1,
                         thread ulong &c2,
                         ulong x)
{
    ulong2 p = umul128(x, x);
    acc_add_192(c0, c1, c2, p.x, p.y, 0ul);
}

inline void sqr_add_cross(thread ulong &c0,
                          thread ulong &c1,
                          thread ulong &c2,
                          ulong x,
                          ulong y)
{
    ulong2 p = umul128(x, y);
    ulong lo = p.x << 1;
    ulong hi = (p.y << 1) | (p.x >> 63);
    ulong ex = p.y >> 63;
    acc_add_192(c0, c1, c2, lo, hi, ex);
}

// Unrolled symmetric Comba square + Montgomery REDC.
// out = a^2*R^-1 mod q.
inline void mont_sqr(thread ulong *out,
                     thread const ulong *a,
                     thread const ulong *q,
                     ulong q_inv_neg)
{
    ulong a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5];
    ulong t[14];

    ulong c0 = 0ul, c1 = 0ul, c2 = 0ul;

#define EMIT_SQR(K) do { t[(K)] = c0; c0 = c1; c1 = c2; c2 = 0ul; } while(false)

    sqr_add_diag(c0, c1, c2, a0);
    EMIT_SQR(0);

    sqr_add_cross(c0, c1, c2, a0, a1);
    EMIT_SQR(1);

    sqr_add_cross(c0, c1, c2, a0, a2);
    sqr_add_diag(c0, c1, c2, a1);
    EMIT_SQR(2);

    sqr_add_cross(c0, c1, c2, a0, a3);
    sqr_add_cross(c0, c1, c2, a1, a2);
    EMIT_SQR(3);

    sqr_add_cross(c0, c1, c2, a0, a4);
    sqr_add_cross(c0, c1, c2, a1, a3);
    sqr_add_diag(c0, c1, c2, a2);
    EMIT_SQR(4);

    sqr_add_cross(c0, c1, c2, a0, a5);
    sqr_add_cross(c0, c1, c2, a1, a4);
    sqr_add_cross(c0, c1, c2, a2, a3);
    EMIT_SQR(5);

    sqr_add_cross(c0, c1, c2, a1, a5);
    sqr_add_cross(c0, c1, c2, a2, a4);
    sqr_add_diag(c0, c1, c2, a3);
    EMIT_SQR(6);

    sqr_add_cross(c0, c1, c2, a2, a5);
    sqr_add_cross(c0, c1, c2, a3, a4);
    EMIT_SQR(7);

    sqr_add_cross(c0, c1, c2, a3, a5);
    sqr_add_diag(c0, c1, c2, a4);
    EMIT_SQR(8);

    sqr_add_cross(c0, c1, c2, a4, a5);
    EMIT_SQR(9);

    sqr_add_diag(c0, c1, c2, a5);
    EMIT_SQR(10);

    t[11] = c0;
    t[12] = c1;
    t[13] = c2;

#undef EMIT_SQR

    for (uint i = 0u; i < N_LIMBS; ++i) {
        ulong m = t[i] * q_inv_neg;

        ulong C = 0ul;
        for (uint j = 0u; j < N_LIMBS; ++j) {
            ulong2 r = fma_add128(m, q[j], t[i + j], C);
            t[i + j] = r.x;
            C = r.y;
        }
        add_limb14(t, i + N_LIMBS, C);
    }

    ulong diff[N_LIMBS];
    ulong borrow = 0ul;
    for (uint i = 0u; i < N_LIMBS; ++i) {
        ulong v = t[N_LIMBS + i];
        ulong tv = v - q[i];
        ulong b1 = (tv > v) ? 1ul : 0ul;
        ulong d = tv - borrow;
        ulong b2 = (d > tv) ? 1ul : 0ul;
        diff[i] = d;
        borrow = b1 + b2;
    }

    bool use_diff = (t[12] != 0ul) || (t[13] != 0ul) || (borrow == 0ul);
    for (uint i = 0u; i < N_LIMBS; ++i) {
        out[i] = use_diff ? diff[i] : t[N_LIMBS + i];
    }
}

// ------------------------------------------------------------------
// Point helpers.
// ------------------------------------------------------------------
inline void load_point(thread ulong *X, thread ulong *Y, thread ulong *Z,
                       device const ulong *src)
{
    for (uint i = 0u; i < N_LIMBS; ++i) X[i] = src[i];
    for (uint i = 0u; i < N_LIMBS; ++i) Y[i] = src[N_LIMBS + i];
    for (uint i = 0u; i < N_LIMBS; ++i) Z[i] = src[2u * N_LIMBS + i];
}

inline void store_point(device ulong *dst,
                        thread const ulong *X,
                        thread const ulong *Y,
                        thread const ulong *Z)
{
    for (uint i = 0u; i < N_LIMBS; ++i) dst[i] = X[i];
    for (uint i = 0u; i < N_LIMBS; ++i) dst[N_LIMBS + i] = Y[i];
    for (uint i = 0u; i < N_LIMBS; ++i) dst[2u * N_LIMBS + i] = Z[i];
}

inline void zero_point(thread ulong *X, thread ulong *Y, thread ulong *Z) {
    for (uint i = 0u; i < N_LIMBS; ++i) X[i] = 0ul;
    for (uint i = 0u; i < N_LIMBS; ++i) Y[i] = 0ul;
    for (uint i = 0u; i < N_LIMBS; ++i) Z[i] = 0ul;
}

inline void copy_point(thread ulong *X, thread ulong *Y, thread ulong *Z,
                       thread const ulong *A,
                       thread const ulong *B,
                       thread const ulong *C)
{
    copy_n(X, A);
    copy_n(Y, B);
    copy_n(Z, C);
}

// ------------------------------------------------------------------
// Jacobian formulas for a = 0 short-Weierstrass curves.
// ------------------------------------------------------------------
inline void jac_double_pt(thread ulong *oX, thread ulong *oY, thread ulong *oZ,
                          thread const ulong *X,
                          thread const ulong *Y,
                          thread const ulong *Z,
                          thread const ulong *q,
                          ulong q_inv_neg)
{
    if (is_zero_n(Z) || is_zero_n(Y)) {
        zero_point(oX, oY, oZ);
        return;
    }

    ulong A[N_LIMBS], B[N_LIMBS], C[N_LIMBS];
    ulong D[N_LIMBS], E[N_LIMBS], F[N_LIMBS];
    ulong tmp[N_LIMBS], tmp2[N_LIMBS];

    mont_sqr(A, X, q, q_inv_neg);
    mont_sqr(B, Y, q, q_inv_neg);
    mont_sqr(C, B, q, q_inv_neg);

    mod_add(tmp, X, B, q);
    mont_sqr(D, tmp, q, q_inv_neg);
    mod_sub(D, D, A, q);
    mod_sub(D, D, C, q);
    mod_add(D, D, D, q);

    mod_add(E, A, A, q);
    mod_add(E, E, A, q);

    mont_sqr(F, E, q, q_inv_neg);

    mod_add(tmp, D, D, q);
    mod_sub(oX, F, tmp, q);

    mod_sub(tmp, D, oX, q);
    mont_mul(tmp, E, tmp, q, q_inv_neg);

    mod_add(tmp2, C, C, q);
    mod_add(tmp2, tmp2, tmp2, q);
    mod_add(tmp2, tmp2, tmp2, q);
    mod_sub(oY, tmp, tmp2, q);

    mont_mul(tmp, Y, Z, q, q_inv_neg);
    mod_add(oZ, tmp, tmp, q);
}

inline void jac_add_pt(thread ulong *oX, thread ulong *oY, thread ulong *oZ,
                       thread const ulong *X1,
                       thread const ulong *Y1,
                       thread const ulong *Z1,
                       thread const ulong *X2,
                       thread const ulong *Y2,
                       thread const ulong *Z2,
                       thread const ulong *q,
                       ulong q_inv_neg)
{
    if (is_zero_n(Z1)) {
        copy_n(oX, X2); copy_n(oY, Y2); copy_n(oZ, Z2);
        return;
    }
    if (is_zero_n(Z2)) {
        copy_n(oX, X1); copy_n(oY, Y1); copy_n(oZ, Z1);
        return;
    }

    ulong Z1Z1[N_LIMBS], Z2Z2[N_LIMBS];
    ulong U1[N_LIMBS], U2[N_LIMBS], S1[N_LIMBS], S2[N_LIMBS];
    ulong H[N_LIMBS], R[N_LIMBS];
    ulong HH[N_LIMBS], HHH[N_LIMBS], V[N_LIMBS];
    ulong tmp[N_LIMBS], tmp2[N_LIMBS];

    mont_sqr(Z1Z1, Z1, q, q_inv_neg);
    mont_sqr(Z2Z2, Z2, q, q_inv_neg);
    mont_mul(U1,   X1, Z2Z2, q, q_inv_neg);
    mont_mul(U2,   X2, Z1Z1, q, q_inv_neg);
    mont_mul(tmp,  Y1, Z2,   q, q_inv_neg);
    mont_mul(S1,   tmp, Z2Z2, q, q_inv_neg);
    mont_mul(tmp,  Y2, Z1,   q, q_inv_neg);
    mont_mul(S2,   tmp, Z1Z1, q, q_inv_neg);

    if (eq_n(U1, U2)) {
        if (eq_n(S1, S2)) {
            jac_double_pt(oX, oY, oZ, X1, Y1, Z1, q, q_inv_neg);
        } else {
            zero_point(oX, oY, oZ);
        }
        return;
    }

    mod_sub(H, U2, U1, q);
    mod_sub(R, S2, S1, q);

    mont_sqr(HH, H, q, q_inv_neg);
    mont_mul(HHH, H, HH, q, q_inv_neg);
    mont_mul(V,   U1, HH, q, q_inv_neg);

    mont_sqr(oX, R, q, q_inv_neg);
    mod_sub(oX, oX, HHH, q);
    mod_add(tmp, V, V, q);
    mod_sub(oX, oX, tmp, q);

    mod_sub(tmp, V, oX, q);
    mont_mul(tmp, R, tmp, q, q_inv_neg);
    mont_mul(tmp2, S1, HHH, q, q_inv_neg);
    mod_sub(oY, tmp, tmp2, q);

    mont_mul(tmp, Z1, Z2, q, q_inv_neg);
    mont_mul(oZ, tmp, H, q, q_inv_neg);
}

// ------------------------------------------------------------------
// Kernel A: per-pair scalar multiplication using signed radix-16.
// Digits are in {-7..8}; final carry is an optional +1 at window 64.
// ------------------------------------------------------------------
kernel void montgomery_msm_pair(
    device const ulong *scalars      [[buffer(0)]],
    device const ulong *points_in    [[buffer(1)]],
    device       ulong *scratch      [[buffer(2)]],
    device const ulong *q            [[buffer(3)]],
    constant ulong     &q_inv_neg    [[buffer(4)]],
    constant uint      &n_pairs      [[buffer(5)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n_pairs) return;

    ulong qloc[N_LIMBS];
    for (uint i = 0u; i < N_LIMBS; ++i) qloc[i] = q[i];

    ulong s0 = scalars[idx * 4u + 0u];
    ulong s1 = scalars[idx * 4u + 1u];
    ulong s2 = scalars[idx * 4u + 2u];
    ulong s3 = scalars[idx * 4u + 3u];

    // Signed radix-16 Booth recoding, LSB to MSB.
    short digs[64];
    uint carry = 0u;
    for (uint w = 0u; w < 64u; ++w) {
        uint word = w >> 4;
        uint sh = (w & 15u) << 2;
        ulong sw = (word == 0u) ? s0 :
                   (word == 1u) ? s1 :
                   (word == 2u) ? s2 : s3;
        uint nib = (uint)((sw >> sh) & 15ul);
        uint u = nib + carry;

        int d;
        if (u > 8u) {
            d = (int)u - 16;
            carry = 1u;
        } else {
            d = (int)u;
            carry = 0u;
        }
        digs[w] = (short)d;
    }

    // Table[0..7] = 1P..8P.
    ulong tblX[W4_TABLE][N_LIMBS];
    ulong tblY[W4_TABLE][N_LIMBS];
    ulong tblZ[W4_TABLE][N_LIMBS];

    load_point(tblX[0], tblY[0], tblZ[0], points_in + idx * POINT_LIMBS);

    // 2P
    jac_double_pt(tblX[1], tblY[1], tblZ[1],
                  tblX[0], tblY[0], tblZ[0],
                  qloc, q_inv_neg);

    // 3P = 2P + P
    jac_add_pt(tblX[2], tblY[2], tblZ[2],
               tblX[1], tblY[1], tblZ[1],
               tblX[0], tblY[0], tblZ[0],
               qloc, q_inv_neg);

    // 4P = 2*(2P)
    jac_double_pt(tblX[3], tblY[3], tblZ[3],
                  tblX[1], tblY[1], tblZ[1],
                  qloc, q_inv_neg);

    // 5P = 4P + P
    jac_add_pt(tblX[4], tblY[4], tblZ[4],
               tblX[3], tblY[3], tblZ[3],
               tblX[0], tblY[0], tblZ[0],
               qloc, q_inv_neg);

    // 6P = 2*(3P)
    jac_double_pt(tblX[5], tblY[5], tblZ[5],
                  tblX[2], tblY[2], tblZ[2],
                  qloc, q_inv_neg);

    // 7P = 6P + P
    jac_add_pt(tblX[6], tblY[6], tblZ[6],
               tblX[5], tblY[5], tblZ[5],
               tblX[0], tblY[0], tblZ[0],
               qloc, q_inv_neg);

    // 8P = 2*(4P)
    jac_double_pt(tblX[7], tblY[7], tblZ[7],
                  tblX[3], tblY[3], tblZ[3],
                  qloc, q_inv_neg);

    ulong AX[N_LIMBS], AY[N_LIMBS], AZ[N_LIMBS];
    ulong TX[N_LIMBS], TY[N_LIMBS], TZ[N_LIMBS];

    zero_point(AX, AY, AZ);

    // Optional top digit: carry * 16^64.
    if (carry != 0u) {
        copy_point(AX, AY, AZ, tblX[0], tblY[0], tblZ[0]);
    }

    // Fixed 65-digit radix-16 recurrence:
    // A starts as d64, then for w=63..0: A = 16*A + d_w.
    for (int win = 63; win >= 0; --win) {
        jac_double_pt(TX, TY, TZ, AX, AY, AZ, qloc, q_inv_neg);
        copy_point(AX, AY, AZ, TX, TY, TZ);
        jac_double_pt(TX, TY, TZ, AX, AY, AZ, qloc, q_inv_neg);
        copy_point(AX, AY, AZ, TX, TY, TZ);
        jac_double_pt(TX, TY, TZ, AX, AY, AZ, qloc, q_inv_neg);
        copy_point(AX, AY, AZ, TX, TY, TZ);
        jac_double_pt(TX, TY, TZ, AX, AY, AZ, qloc, q_inv_neg);
        copy_point(AX, AY, AZ, TX, TY, TZ);

        short sd = digs[(uint)win];
        if (sd != (short)0) {
            bool neg = (sd < (short)0);
            uint ad = neg ? (uint)(-((int)sd)) : (uint)sd;
            uint ti = ad - 1u;

            for (uint i = 0u; i < N_LIMBS; ++i) {
                TX[i] = tblX[ti][i];
                TY[i] = tblY[ti][i];
                TZ[i] = tblZ[ti][i];
            }

            if (neg) {
                mod_neg(TY, TY, qloc);
            }

            jac_add_pt(TX, TY, TZ,
                       AX, AY, AZ,
                       TX, TY, TZ,
                       qloc, q_inv_neg);
            copy_point(AX, AY, AZ, TX, TY, TZ);
        }
    }

    store_point(scratch + idx * POINT_LIMBS, AX, AY, AZ);
}

// ------------------------------------------------------------------
// Kernel B: one tree-reduction level.
// ------------------------------------------------------------------
kernel void montgomery_msm_reduce(
    device       ulong *scratch      [[buffer(0)]],
    device const ulong *q            [[buffer(1)]],
    constant ulong     &q_inv_neg    [[buffer(2)]],
    constant uint      &half_count   [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= half_count) return;

    ulong qloc[N_LIMBS];
    for (uint i = 0u; i < N_LIMBS; ++i) qloc[i] = q[i];

    ulong AX[N_LIMBS], AY[N_LIMBS], AZ[N_LIMBS];
    ulong BX[N_LIMBS], BY[N_LIMBS], BZ[N_LIMBS];

    load_point(AX, AY, AZ, scratch + idx * POINT_LIMBS);
    load_point(BX, BY, BZ, scratch + (idx + half_count) * POINT_LIMBS);

    ulong RX[N_LIMBS], RY[N_LIMBS], RZ[N_LIMBS];
    jac_add_pt(RX, RY, RZ,
               AX, AY, AZ,
               BX, BY, BZ,
               qloc, q_inv_neg);

    store_point(scratch + idx * POINT_LIMBS, RX, RY, RZ);
}