#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_LIMBS = 6u;
constexpr constant uint POINT_LIMBS = 18u;
constexpr constant uint WNAF_BITS = 257u;
constexpr constant uint WNAF_TABLE = 8u;
constexpr constant uint WNAF_MAX = 60u;

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

// CIOS Montgomery multiplication: out = a*b*R^-1 mod q.
inline void mont_mul(thread ulong *out,
                     thread const ulong *a,
                     thread const ulong *b,
                     thread const ulong *q,
                     ulong q_inv_neg)
{
    ulong t[N_LIMBS + 2u];
    for (uint i = 0u; i < N_LIMBS + 2u; ++i) t[i] = 0ul;

    for (uint i = 0u; i < N_LIMBS; ++i) {
        ulong C = 0ul;
        for (uint j = 0u; j < N_LIMBS; ++j) {
            ulong2 r = fma_add128(a[j], b[i], t[j], C);
            t[j] = r.x;
            C = r.y;
        }
        {
            ulong s = t[N_LIMBS] + C;
            ulong cy = (s < t[N_LIMBS]) ? 1ul : 0ul;
            t[N_LIMBS] = s;
            t[N_LIMBS + 1u] += cy;
        }

        ulong m = t[0] * q_inv_neg;

        C = 0ul;
        for (uint j = 0u; j < N_LIMBS; ++j) {
            ulong2 r = fma_add128(m, q[j], t[j], C);
            t[j] = r.x;
            C = r.y;
        }
        {
            ulong s = t[N_LIMBS] + C;
            ulong cy = (s < t[N_LIMBS]) ? 1ul : 0ul;
            t[N_LIMBS] = s;
            t[N_LIMBS + 1u] += cy;
        }

        for (uint j = 0u; j < N_LIMBS + 1u; ++j) {
            t[j] = t[j + 1u];
        }
        t[N_LIMBS + 1u] = 0ul;
    }

    ulong diff[N_LIMBS];
    ulong borrow = 0ul;
    for (uint i = 0u; i < N_LIMBS; ++i) {
        ulong tv = t[i] - q[i];
        ulong b1 = (tv > t[i]) ? 1ul : 0ul;
        ulong d = tv - borrow;
        ulong b2 = (d > tv) ? 1ul : 0ul;
        diff[i] = d;
        borrow = b1 + b2;
    }

    bool use_diff = (t[N_LIMBS] != 0ul) || (borrow == 0ul);
    for (uint i = 0u; i < N_LIMBS; ++i) {
        out[i] = use_diff ? diff[i] : t[i];
    }
}

// Specialized Montgomery squaring using symmetric Comba product + REDC.
// out = a^2*R^-1 mod q.
inline void mont_sqr(thread ulong *out,
                     thread const ulong *a,
                     thread const ulong *q,
                     ulong q_inv_neg)
{
    ulong t[14];

    ulong c0 = 0ul, c1 = 0ul, c2 = 0ul;

    for (uint k = 0u; k <= 10u; ++k) {
        uint ilo = (k > 5u) ? (k - 5u) : 0u;
        uint ihi = (k < 5u) ? k : 5u;

        for (uint i = ilo; i <= ihi; ++i) {
            uint j = k - i;
            if (i > j) continue;

            ulong2 p = umul128(a[i], a[j]);
            if (i == j) {
                acc_add_192(c0, c1, c2, p.x, p.y, 0ul);
            } else {
                ulong lo = p.x << 1;
                ulong hi = (p.y << 1) | (p.x >> 63);
                ulong ex = p.y >> 63;
                acc_add_192(c0, c1, c2, lo, hi, ex);
            }
        }

        t[k] = c0;
        c0 = c1;
        c1 = c2;
        c2 = 0ul;
    }

    t[11] = c0;
    t[12] = c1;
    t[13] = c2;

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

// Add where the second point has cached Z2^2 and Z2^3.
inline void jac_add_cached_pt(thread ulong *oX, thread ulong *oY, thread ulong *oZ,
                              thread const ulong *X1,
                              thread const ulong *Y1,
                              thread const ulong *Z1,
                              thread const ulong *X2,
                              thread const ulong *Y2,
                              thread const ulong *Z2,
                              thread const ulong *Z2Z2_cached,
                              thread const ulong *Z2Z3_cached,
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

    ulong Z1Z1[N_LIMBS];
    ulong U1[N_LIMBS], U2[N_LIMBS], S1[N_LIMBS], S2[N_LIMBS];
    ulong H[N_LIMBS], R[N_LIMBS];
    ulong HH[N_LIMBS], HHH[N_LIMBS], V[N_LIMBS];
    ulong tmp[N_LIMBS], tmp2[N_LIMBS];

    mont_sqr(Z1Z1, Z1, q, q_inv_neg);
    mont_mul(U1,   X1, Z2Z2_cached, q, q_inv_neg);
    mont_mul(U2,   X2, Z1Z1, q, q_inv_neg);
    mont_mul(S1,   Y1, Z2Z3_cached, q, q_inv_neg);
    mont_mul(tmp,  Y2, Z1, q, q_inv_neg);
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
// Kernel A: per-pair scalar multiplication using compact width-5 wNAF.
// Table entries are odd multiples 1P,3P,...,15P.
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

    ulong s0 = scalars[idx * 4u + 0u];
    ulong s1 = scalars[idx * 4u + 1u];
    ulong s2 = scalars[idx * 4u + 2u];
    ulong s3 = scalars[idx * 4u + 3u];

    if ((s0 | s1 | s2 | s3) == 0ul) {
        device ulong *dst = scratch + idx * POINT_LIMBS;
        for (uint i = 0u; i < POINT_LIMBS; ++i) dst[i] = 0ul;
        return;
    }

    // Width-5 non-adjacent form. Store only non-zero digits.
    uint pos_arr[WNAF_MAX];
    short dig_arr[WNAF_MAX];
    uint n_digs = 0u;

    ulong k0 = s0, k1 = s1, k2 = s2, k3 = s3, k4 = 0ul;

    for (uint pos = 0u; pos < WNAF_BITS; ++pos) {
        bool nz = ((k0 | k1 | k2 | k3 | k4) != 0ul);

        if (nz && ((k0 & 1ul) != 0ul)) {
            uint rem = (uint)(k0 & 31ul);
            int u = (rem > 16u) ? ((int)rem - 32) : (int)rem;

            pos_arr[n_digs] = pos;
            dig_arr[n_digs] = (short)u;
            n_digs++;

            if (u > 0) {
                ulong sub = (ulong)u;
                ulong tv = k0 - sub;
                ulong borrow = (tv > k0) ? 1ul : 0ul;
                k0 = tv;
                tv = k1 - borrow; borrow = (tv > k1) ? 1ul : 0ul; k1 = tv;
                tv = k2 - borrow; borrow = (tv > k2) ? 1ul : 0ul; k2 = tv;
                tv = k3 - borrow; borrow = (tv > k3) ? 1ul : 0ul; k3 = tv;
                k4 -= borrow;
            } else {
                ulong add = (ulong)(-u);
                ulong tv = k0 + add;
                ulong carry = (tv < k0) ? 1ul : 0ul;
                k0 = tv;
                tv = k1 + carry; carry = (tv < k1) ? 1ul : 0ul; k1 = tv;
                tv = k2 + carry; carry = (tv < k2) ? 1ul : 0ul; k2 = tv;
                tv = k3 + carry; carry = (tv < k3) ? 1ul : 0ul; k3 = tv;
                k4 += carry;
            }
        }

        k0 = (k0 >> 1) | (k1 << 63);
        k1 = (k1 >> 1) | (k2 << 63);
        k2 = (k2 >> 1) | (k3 << 63);
        k3 = (k3 >> 1) | (k4 << 63);
        k4 = (k4 >> 1);
    }

    if (n_digs == 0u) {
        device ulong *dst = scratch + idx * POINT_LIMBS;
        for (uint i = 0u; i < POINT_LIMBS; ++i) dst[i] = 0ul;
        return;
    }

    ulong qloc[N_LIMBS];
    for (uint i = 0u; i < N_LIMBS; ++i) qloc[i] = q[i];

    ulong tblX[WNAF_TABLE][N_LIMBS];
    ulong tblY[WNAF_TABLE][N_LIMBS];
    ulong tblZ[WNAF_TABLE][N_LIMBS];

    load_point(tblX[0], tblY[0], tblZ[0], points_in + idx * POINT_LIMBS);

    // D = 2P, then odd table T[i] = T[i-1] + D.
    ulong DX[N_LIMBS], DY[N_LIMBS], DZ[N_LIMBS];
    ulong DZZ[N_LIMBS], DZZZ[N_LIMBS];

    jac_double_pt(DX, DY, DZ,
                  tblX[0], tblY[0], tblZ[0],
                  qloc, q_inv_neg);

    mont_sqr(DZZ, DZ, qloc, q_inv_neg);
    mont_mul(DZZZ, DZZ, DZ, qloc, q_inv_neg);

    for (uint t = 1u; t < WNAF_TABLE; ++t) {
        jac_add_cached_pt(tblX[t], tblY[t], tblZ[t],
                          tblX[t - 1u], tblY[t - 1u], tblZ[t - 1u],
                          DX, DY, DZ,
                          DZZ, DZZZ,
                          qloc, q_inv_neg);
    }

    ulong AX[N_LIMBS], AY[N_LIMBS], AZ[N_LIMBS];
    ulong TX[N_LIMBS], TY[N_LIMBS], TZ[N_LIMBS];

    uint top_i = n_digs - 1u;
    short top = dig_arr[top_i];
    bool top_neg = (top < (short)0);
    uint top_abs = top_neg ? (uint)(-((int)top)) : (uint)top;
    uint top_ti = top_abs >> 1; // 1,3,...,15 -> 0..7

    for (uint i = 0u; i < N_LIMBS; ++i) {
        AX[i] = tblX[top_ti][i];
        AY[i] = tblY[top_ti][i];
        AZ[i] = tblZ[top_ti][i];
    }
    if (top_neg) {
        mod_neg(AY, AY, qloc);
    }

    uint cur_pos = pos_arr[top_i];

    for (int ii = (int)n_digs - 2; ii >= 0; --ii) {
        uint next_pos = pos_arr[(uint)ii];
        uint steps = cur_pos - next_pos;

        for (uint d = 0u; d < steps; ++d) {
            jac_double_pt(TX, TY, TZ, AX, AY, AZ, qloc, q_inv_neg);
            copy_point(AX, AY, AZ, TX, TY, TZ);
        }

        short sd = dig_arr[(uint)ii];
        bool neg = (sd < (short)0);
        uint ad = neg ? (uint)(-((int)sd)) : (uint)sd;
        uint ti = ad >> 1;

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

        cur_pos = next_pos;
    }

    // Account for trailing zero bits below the last non-zero wNAF digit.
    for (uint d = 0u; d < cur_pos; ++d) {
        jac_double_pt(TX, TY, TZ, AX, AY, AZ, qloc, q_inv_neg);
        copy_point(AX, AY, AZ, TX, TY, TZ);
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