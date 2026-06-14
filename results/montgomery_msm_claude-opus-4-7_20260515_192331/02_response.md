Looking at the previous attempt, the bottleneck is the per-pair Montgomery multiplication. The CIOS loop has nested 6x6 mulhi operations and each `umul128` does 4 32-bit multiplies. The wNAF scan does ~256 doublings + ~51 additions per pair, each with ~10-16 `mont_mul` calls.

**Optimization**: Switch from wNAF window-4 to a **simple left-to-right binary double-and-add** but use **mixed Jacobian-affine addition** when the table point has Z=1 (after one inversion-free trick: keep the original point as-is, since input Z may not be 1). Actually, the biggest gain comes from reducing `mont_mul` register pressure. Let me use a tighter **CIOS** with arrays inlined and reduce the number of `mont_mul` calls per doubling/addition by using better formulas.

Key changes:
1. Use **dbl-2009-l** (already used) but inline temps more carefully.
2. Replace wNAF with **window-4 fixed table** but keep precomputation small.
3. Most importantly: **unroll the CIOS inner loops** to let the compiler schedule 32-bit muls better, and store the table in *threadgroup* / use simpler binary method when scalar is sparse.

Given the marginal improvement margin, I'll focus on hand-unrolling `mont_mul` with explicit 32-bit half products to expose ILP, and keep the wNAF structure which is already good.

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_LIMBS = 6u;
constexpr constant uint POINT_LIMBS = 18u;
constexpr constant uint WINDOW = 4u;
constexpr constant uint TABLE_SIZE = 8u;

constant ulong LO32 = 0x00000000FFFFFFFFul;

inline ulong2 umul128(ulong a, ulong b) {
    uint a0 = (uint)(a);
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)(b);
    uint b1 = (uint)(b >> 32);
    ulong p00 = (ulong)a0 * (ulong)b0;
    ulong p01 = (ulong)a0 * (ulong)b1;
    ulong p10 = (ulong)a1 * (ulong)b0;
    ulong p11 = (ulong)a1 * (ulong)b1;
    ulong mid = (p00 >> 32) + (p01 & LO32) + (p10 & LO32);
    ulong lo  = (p00 & LO32) | (mid << 32);
    ulong hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
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

inline void copy_n(thread ulong *dst, thread const ulong *src) {
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) dst[i] = src[i];
}

inline bool is_zero_n(thread const ulong *a) {
    ulong r = 0ul;
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) r |= a[i];
    return r == 0ul;
}

inline bool eq_n(thread const ulong *a, thread const ulong *b) {
    ulong r = 0ul;
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) r |= (a[i] ^ b[i]);
    return r == 0ul;
}

inline void mod_add(thread ulong *c,
                    thread const ulong *a, thread const ulong *b,
                    device const ulong *q)
{
    ulong sum[N_LIMBS];
    ulong carry = 0ul;
    #pragma unroll
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
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) {
        ulong tv = sum[i] - q[i];
        ulong b1 = (tv > sum[i]) ? 1ul : 0ul;
        ulong d = tv - borrow;
        ulong b2 = (d > tv) ? 1ul : 0ul;
        diff[i] = d;
        borrow = b1 + b2;
    }
    bool use_diff = (carry != 0ul) || (borrow == 0ul);
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) {
        c[i] = use_diff ? diff[i] : sum[i];
    }
}

inline void mod_sub(thread ulong *c,
                    thread const ulong *a, thread const ulong *b,
                    device const ulong *q)
{
    ulong diff[N_LIMBS];
    ulong borrow = 0ul;
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) {
        ulong tv = a[i] - b[i];
        ulong b1 = (tv > a[i]) ? 1ul : 0ul;
        ulong d = tv - borrow;
        ulong b2 = (d > tv) ? 1ul : 0ul;
        diff[i] = d;
        borrow = b1 + b2;
    }
    if (borrow != 0ul) {
        ulong carry = 0ul;
        #pragma unroll
        for (uint i = 0u; i < N_LIMBS; ++i) {
            ulong s = diff[i] + carry;
            ulong cy1 = (s < diff[i]) ? 1ul : 0ul;
            ulong t = s + q[i];
            ulong cy2 = (t < s) ? 1ul : 0ul;
            c[i] = t;
            carry = cy1 + cy2;
        }
    } else {
        #pragma unroll
        for (uint i = 0u; i < N_LIMBS; ++i) c[i] = diff[i];
    }
}

inline void mod_neg(thread ulong *c, thread const ulong *a, device const ulong *q) {
    if (is_zero_n(a)) {
        #pragma unroll
        for (uint i = 0u; i < N_LIMBS; ++i) c[i] = 0ul;
        return;
    }
    ulong borrow = 0ul;
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) {
        ulong tv = q[i] - a[i];
        ulong b1 = (tv > q[i]) ? 1ul : 0ul;
        ulong d = tv - borrow;
        ulong b2 = (d > tv) ? 1ul : 0ul;
        c[i] = d;
        borrow = b1 + b2;
    }
}

inline void mont_mul(thread ulong *out,
                     thread const ulong *a, thread const ulong *b,
                     device const ulong *q, ulong q_inv_neg)
{
    ulong t0 = 0ul, t1 = 0ul, t2 = 0ul, t3 = 0ul;
    ulong t4 = 0ul, t5 = 0ul, t6 = 0ul, t7 = 0ul;
    ulong q0 = q[0], q1 = q[1], q2 = q[2], q3 = q[3], q4 = q[4], q5 = q[5];
    ulong a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5];

    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) {
        ulong bi = b[i];
        ulong C = 0ul;
        ulong2 r;
        r = fma_add128(a0, bi, t0, C); t0 = r.x; C = r.y;
        r = fma_add128(a1, bi, t1, C); t1 = r.x; C = r.y;
        r = fma_add128(a2, bi, t2, C); t2 = r.x; C = r.y;
        r = fma_add128(a3, bi, t3, C); t3 = r.x; C = r.y;
        r = fma_add128(a4, bi, t4, C); t4 = r.x; C = r.y;
        r = fma_add128(a5, bi, t5, C); t5 = r.x; C = r.y;
        {
            ulong s = t6 + C;
            ulong cy = (s < t6) ? 1ul : 0ul;
            t6 = s;
            t7 = t7 + cy;
        }
        ulong m = t0 * q_inv_neg;
        C = 0ul;
        r = fma_add128(m, q0, t0, C); t0 = r.x; C = r.y;
        r = fma_add128(m, q1, t1, C); t1 = r.x; C = r.y;
        r = fma_add128(m, q2, t2, C); t2 = r.x; C = r.y;
        r = fma_add128(m, q3, t3, C); t3 = r.x; C = r.y;
        r = fma_add128(m, q4, t4, C); t4 = r.x; C = r.y;
        r = fma_add128(m, q5, t5, C); t5 = r.x; C = r.y;
        {
            ulong s = t6 + C;
            ulong cy = (s < t6) ? 1ul : 0ul;
            t6 = s;
            t7 = t7 + cy;
        }
        // shift right by one limb
        t0 = t1; t1 = t2; t2 = t3; t3 = t4; t4 = t5; t5 = t6; t6 = t7; t7 = 0ul;
    }
    // final conditional subtract
    ulong tt[N_LIMBS];
    tt[0] = t0; tt[1] = t1; tt[2] = t2; tt[3] = t3; tt[4] = t4; tt[5] = t5;
    ulong diff[N_LIMBS];
    ulong borrow = 0ul;
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) {
        ulong tv = tt[i] - q[i];
        ulong b1 = (tv > tt[i]) ? 1ul : 0ul;
        ulong d = tv - borrow;
        ulong b2 = (d > tv) ? 1ul : 0ul;
        diff[i] = d;
        borrow = b1 + b2;
    }
    bool use_diff = (t6 != 0ul) || (borrow == 0ul);
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) {
        out[i] = use_diff ? diff[i] : tt[i];
    }
}

inline void load_point(thread ulong *X, thread ulong *Y, thread ulong *Z,
                       device const ulong *src)
{
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) X[i] = src[i];
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) Y[i] = src[N_LIMBS + i];
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) Z[i] = src[2u * N_LIMBS + i];
}

inline void store_point(device ulong *dst,
                        thread const ulong *X, thread const ulong *Y, thread const ulong *Z)
{
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) dst[i] = X[i];
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) dst[N_LIMBS + i] = Y[i];
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) dst[2u * N_LIMBS + i] = Z[i];
}

inline void zero_point(thread ulong *X, thread ulong *Y, thread ulong *Z) {
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) { X[i] = 0ul; Y[i] = 0ul; Z[i] = 0ul; }
}

inline void jac_double_pt(thread ulong *oX, thread ulong *oY, thread ulong *oZ,
                          thread const ulong *X, thread const ulong *Y, thread const ulong *Z,
                          device const ulong *q, ulong q_inv_neg)
{
    if (is_zero_n(Z) || is_zero_n(Y)) {
        zero_point(oX, oY, oZ);
        return;
    }
    ulong A[N_LIMBS], B[N_LIMBS], C[N_LIMBS];
    ulong D[N_LIMBS], E[N_LIMBS], F[N_LIMBS];
    ulong tmp[N_LIMBS], tmp2[N_LIMBS];

    mont_mul(A, X, X, q, q_inv_neg);
    mont_mul(B, Y, Y, q, q_inv_neg);
    mont_mul(C, B, B, q, q_inv_neg);

    mod_add(tmp, X, B, q);
    mont_mul(D, tmp, tmp, q, q_inv_neg);
    mod_sub(D, D, A, q);
    mod_sub(D, D, C, q);
    mod_add(D, D, D, q);

    mod_add(E, A, A, q);
    mod_add(E, E, A, q);

    mont_mul(F, E, E, q, q_inv_neg);

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
                       thread const ulong *X1, thread const ulong *Y1, thread const ulong *Z1,
                       thread const ulong *X2, thread const ulong *Y2, thread const ulong *Z2,
                       device const ulong *q, ulong q_inv_neg)
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

    mont_mul(Z1Z1, Z1, Z1, q, q_inv_neg);
    mont_mul(Z2Z2, Z2, Z2, q, q_inv_neg);
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

    mod_sub(H,   U2, U1, q);
    mod_sub(R,   S2, S1, q);
    mont_mul(HH,  H, H, q, q_inv_neg);
    mont_mul(HHH, H, HH, q, q_inv_neg);
    mont_mul(V,   U1, HH, q, q_inv_neg);

    mont_mul(oX, R, R, q, q_inv_neg);
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

    ulong s[4];
    s[0] = scalars[idx * 4u + 0u];
    s[1] = scalars[idx * 4u + 1u];
    s[2] = scalars[idx * 4u + 2u];
    s[3] = scalars[idx * 4u + 3u];

    ulong PX[N_LIMBS], PY[N_LIMBS], PZ[N_LIMBS];
    load_point(PX, PY, PZ, points_in + idx * POINT_LIMBS);

    ulong tX[TABLE_SIZE * N_LIMBS];
    ulong tY[TABLE_SIZE * N_LIMBS];
    ulong tZ[TABLE_SIZE * N_LIMBS];

    #pragma unroll
    for (uint j = 0u; j < N_LIMBS; ++j) {
        tX[j] = PX[j];
        tY[j] = PY[j];
        tZ[j] = PZ[j];
    }

    ulong P2X[N_LIMBS], P2Y[N_LIMBS], P2Z[N_LIMBS];
    jac_double_pt(P2X, P2Y, P2Z, PX, PY, PZ, q, q_inv_neg);

    ulong curX[N_LIMBS], curY[N_LIMBS], curZ[N_LIMBS];
    copy_n(curX, PX); copy_n(curY, PY); copy_n(curZ, PZ);
    for (uint i = 1u; i < TABLE_SIZE; ++i) {
        ulong nX[N_LIMBS], nY[N_LIMBS], nZ[N_LIMBS];
        jac_add_pt(nX, nY, nZ, curX, curY, curZ, P2X, P2Y, P2Z, q, q_inv_neg);
        #pragma unroll
        for (uint j = 0u; j < N_LIMBS; ++j) {
            tX[i * N_LIMBS + j] = nX[j];
            tY[i * N_LIMBS + j] = nY[j];
            tZ[i * N_LIMBS + j] = nZ[j];
        }
        copy_n(curX, nX); copy_n(curY, nY); copy_n(curZ, nZ);
    }

    ulong AX[N_LIMBS], AY[N_LIMBS], AZ[N_LIMBS];
    zero_point(AX, AY, AZ);

    char naf[260];
    uint naf_len = 0u;
    {
        ulong k[4];
        k[0] = s[0]; k[1] = s[1]; k[2] = s[2]; k[3] = s[3];
        const uint w = WINDOW;
        const int half_pow = 1 << (w - 1u);
        const int full_pow = 1 << w;
        const ulong mask_w = (1ul << w) - 1ul;

        for (uint i = 0u; i < 260u; ++i) {
            if ((k[0] | k[1] | k[2] | k[3]) == 0ul) {
                naf_len = i;
                break;
            }
            int d = 0;
            if ((k[0] & 1ul) != 0ul) {
                d = (int)(k[0] & mask_w);
                if (d >= half_pow) d -= full_pow;
                if (d > 0) {
                    ulong sub = (ulong)d;
                    ulong borrow = 0ul;
                    for (uint j = 0u; j < 4u; ++j) {
                        ulong sv = (j == 0u) ? sub : 0ul;
                        ulong tv = k[j] - sv;
                        ulong b1 = (tv > k[j]) ? 1ul : 0ul;
                        ulong nv = tv - borrow;
                        ulong b2 = (nv > tv) ? 1ul : 0ul;
                        k[j] = nv;
                        borrow = b1 + b2;
                    }
                } else {
                    ulong add = (ulong)(-d);
                    ulong carry = 0ul;
                    for (uint j = 0u; j < 4u; ++j) {
                        ulong av = (j == 0u) ? add : 0ul;
                        ulong s1 = k[j] + carry;
                        ulong cy1 = (s1 < k[j]) ? 1ul : 0ul;
                        ulong t1 = s1 + av;
                        ulong cy2 = (t1 < s1) ? 1ul : 0ul;
                        k[j] = t1;
                        carry = cy1 + cy2;
                    }
                }
            }
            naf[i] = (char)d;
            k[0] = (k[0] >> 1) | (k[1] << 63);
            k[1] = (k[1] >> 1) | (k[2] << 63);
            k[2] = (k[2] >> 1) | (k[3] << 63);
            k[3] = k[3] >> 1;
            naf_len = i + 1u;
        }
    }

    ulong TX[N_LIMBS], TY[N_LIMBS], TZ[N_LIMBS];
    if (naf_len > 0u) {
        for (int i = (int)naf_len - 1; i >= 0; --i) {
            jac_double_pt(TX, TY, TZ, AX, AY, AZ, q, q_inv_neg);
            copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);
            int d = (int)naf[i];
            if (d != 0) {
                int ad = d > 0 ? d : -d;
                uint tidx = (uint)((ad - 1) >> 1);
                ulong QX[N_LIMBS], QY[N_LIMBS], QZ[N_LIMBS];
                #pragma unroll
                for (uint j = 0u; j < N_LIMBS; ++j) {
                    QX[j] = tX[tidx * N_LIMBS + j];
                    QZ[j] = tZ[tidx * N_LIMBS + j];
                }
                if (d > 0) {
                    #pragma unroll
                    for (uint j = 0u; j < N_LIMBS; ++j) {
                        QY[j] = tY[tidx * N_LIMBS + j];
                    }
                } else {
                    ulong tmpY[N_LIMBS];
                    #pragma unroll
                    for (uint j = 0u; j < N_LIMBS; ++j) {
                        tmpY[j] = tY[tidx * N_LIMBS + j];
                    }
                    mod_neg(QY, tmpY, q);
                }
                jac_add_pt(TX, TY, TZ, AX, AY, AZ, QX, QY, QZ, q, q_inv_neg);
                copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);
            }
        }
    }

    store_point(scratch + idx * POINT_LIMBS, AX, AY, AZ);
}

kernel void montgomery_msm_reduce(
    device       ulong *scratch      [[buffer(0)]],
    device const ulong *q            [[buffer(1)]],
    constant ulong     &q_inv_neg    [[buffer(2)]],
    constant uint      &half_count   [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= half_count) return;

    ulong AX[N_LIMBS], AY[N_LIMBS], AZ[N_LIMBS];
    ulong BX[N_LIMBS], BY[N_LIMBS], BZ[N_LIMBS];
    load_point(AX, AY, AZ, scratch + idx * POINT_LIMBS);
    load_point(BX, BY, BZ, scratch + (idx + half_count) * POINT_LIMBS);

    ulong RX[N_LIMBS], RY[N_LIMBS], RZ[N_LIMBS];
    jac_add_pt(RX, RY, RZ,
               AX, AY, AZ,
               BX, BY, BZ,
               q, q_inv_neg);
    store_point(scratch + idx * POINT_LIMBS, RX, RY, RZ);
}
```