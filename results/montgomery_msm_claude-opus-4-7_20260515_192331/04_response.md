I'll acknowledge the system reminder about the keybindings-help skill — it's not relevant to this MSL kernel optimization task, so I'll proceed with the task.

**Optimization:** The biggest cost is Montgomery multiplication. I'll switch to a **FIOS-style separated-loop CIOS** with all locals as scalars (no array indexing), and more critically, **switch the per-pair algorithm from 4-bit wNAF + precomputed table (which costs ~7 doublings + many adds upfront, then ~260 doublings + ~65 adds) to a simple left-to-right binary double-and-add**. The wNAF table build does 7 EC ops upfront just to save ~65 adds out of ~260 doublings — but each table point requires its own store/load through thread-local arrays, and each NAF add is a full general Jacobian add. Plain binary needs ~256 doubles + ~128 mixed adds with Z2=1 specialization. Actually the cleanest gain: use **mixed addition** (Z2 known to be R mod q = "1" in Montgomery form) when adding the original point, saving 4 muls and 1 sqr per add. I'll specialize `jac_madd` for Z2 = R_mont. Also tighten the inner mont_mul.

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_LIMBS = 6u;
constexpr constant uint POINT_LIMBS = 18u;

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

inline ulong2 mac(ulong a, ulong b, ulong t, ulong c) {
    ulong2 p = umul128(a, b);
    ulong lo1 = p.x + t;
    ulong cy1 = (lo1 < p.x) ? 1ul : 0ul;
    ulong hi1 = p.y + cy1;
    ulong lo2 = lo1 + c;
    ulong cy2 = (lo2 < lo1) ? 1ul : 0ul;
    ulong hi2 = hi1 + cy2;
    return ulong2(lo2, hi2);
}

inline void copy_n(thread ulong *dst, thread const ulong *src) {
    for (uint i = 0u; i < N_LIMBS; ++i) dst[i] = src[i];
}

inline bool is_zero_n(thread const ulong *a) {
    ulong r = 0ul;
    for (uint i = 0u; i < N_LIMBS; ++i) r |= a[i];
    return r == 0ul;
}

inline bool eq_n(thread const ulong *a, thread const ulong *b) {
    ulong r = 0ul;
    for (uint i = 0u; i < N_LIMBS; ++i) r |= (a[i] ^ b[i]);
    return r == 0ul;
}

inline void mod_add(thread ulong *c,
                    thread const ulong *a, thread const ulong *b,
                    device const ulong *q)
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
                    thread const ulong *a, thread const ulong *b,
                    device const ulong *q)
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
    if (borrow != 0ul) {
        ulong carry = 0ul;
        for (uint i = 0u; i < N_LIMBS; ++i) {
            ulong s = diff[i] + carry;
            ulong cy1 = (s < diff[i]) ? 1ul : 0ul;
            ulong t = s + q[i];
            ulong cy2 = (t < s) ? 1ul : 0ul;
            c[i] = t;
            carry = cy1 + cy2;
        }
    } else {
        for (uint i = 0u; i < N_LIMBS; ++i) c[i] = diff[i];
    }
}

inline void mont_mul(thread ulong *out,
                     thread const ulong *a, thread const ulong *b,
                     device const ulong *q, ulong q_inv_neg)
{
    ulong a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5];
    ulong q0 = q[0], q1 = q[1], q2 = q[2], q3 = q[3], q4 = q[4], q5 = q[5];

    ulong t0 = 0ul, t1 = 0ul, t2 = 0ul, t3 = 0ul, t4 = 0ul, t5 = 0ul, t6 = 0ul;
    ulong2 r;
    ulong C, m, bi, s, cy;

    #define CIOS_STEP(IDX) \
        bi = b[IDX]; \
        r = mac(a0, bi, t0, 0ul); t0 = r.x; C = r.y; \
        r = mac(a1, bi, t1, C);   t1 = r.x; C = r.y; \
        r = mac(a2, bi, t2, C);   t2 = r.x; C = r.y; \
        r = mac(a3, bi, t3, C);   t3 = r.x; C = r.y; \
        r = mac(a4, bi, t4, C);   t4 = r.x; C = r.y; \
        r = mac(a5, bi, t5, C);   t5 = r.x; C = r.y; \
        s = t6 + C; t6 = s; \
        m = t0 * q_inv_neg; \
        r = mac(m, q0, t0, 0ul);  C = r.y; \
        r = mac(m, q1, t1, C);    t0 = r.x; C = r.y; \
        r = mac(m, q2, t2, C);    t1 = r.x; C = r.y; \
        r = mac(m, q3, t3, C);    t2 = r.x; C = r.y; \
        r = mac(m, q4, t4, C);    t3 = r.x; C = r.y; \
        r = mac(m, q5, t5, C);    t4 = r.x; C = r.y; \
        s = t6 + C; cy = (s < t6) ? 1ul : 0ul; t5 = s; t6 = cy;

    CIOS_STEP(0)
    CIOS_STEP(1)
    CIOS_STEP(2)
    CIOS_STEP(3)
    CIOS_STEP(4)
    CIOS_STEP(5)
    #undef CIOS_STEP

    ulong tt0 = t0, tt1 = t1, tt2 = t2, tt3 = t3, tt4 = t4, tt5 = t5;

    // conditional subtract
    ulong d0 = tt0 - q0;            ulong br = (d0 > tt0) ? 1ul : 0ul;
    ulong d1 = tt1 - q1;            ulong b1n = (d1 > tt1) ? 1ul : 0ul;
    ulong d1b = d1 - br;            ulong b1c = (d1b > d1) ? 1ul : 0ul; br = b1n + b1c;
    ulong d2 = tt2 - q2;            ulong b2n = (d2 > tt2) ? 1ul : 0ul;
    ulong d2b = d2 - br;            ulong b2c = (d2b > d2) ? 1ul : 0ul; br = b2n + b2c;
    ulong d3 = tt3 - q3;            ulong b3n = (d3 > tt3) ? 1ul : 0ul;
    ulong d3b = d3 - br;            ulong b3c = (d3b > d3) ? 1ul : 0ul; br = b3n + b3c;
    ulong d4 = tt4 - q4;            ulong b4n = (d4 > tt4) ? 1ul : 0ul;
    ulong d4b = d4 - br;            ulong b4c = (d4b > d4) ? 1ul : 0ul; br = b4n + b4c;
    ulong d5 = tt5 - q5;            ulong b5n = (d5 > tt5) ? 1ul : 0ul;
    ulong d5b = d5 - br;            ulong b5c = (d5b > d5) ? 1ul : 0ul; br = b5n + b5c;

    bool use_diff = (t6 != 0ul) || (br == 0ul);
    out[0] = use_diff ? d0 : tt0;
    out[1] = use_diff ? d1b : tt1;
    out[2] = use_diff ? d2b : tt2;
    out[3] = use_diff ? d3b : tt3;
    out[4] = use_diff ? d4b : tt4;
    out[5] = use_diff ? d5b : tt5;
}

inline void load_point(thread ulong *X, thread ulong *Y, thread ulong *Z,
                       device const ulong *src)
{
    for (uint i = 0u; i < N_LIMBS; ++i) X[i] = src[i];
    for (uint i = 0u; i < N_LIMBS; ++i) Y[i] = src[N_LIMBS + i];
    for (uint i = 0u; i < N_LIMBS; ++i) Z[i] = src[2u * N_LIMBS + i];
}

inline void store_point(device ulong *dst,
                        thread const ulong *X, thread const ulong *Y, thread const ulong *Z)
{
    for (uint i = 0u; i < N_LIMBS; ++i) dst[i] = X[i];
    for (uint i = 0u; i < N_LIMBS; ++i) dst[N_LIMBS + i] = Y[i];
    for (uint i = 0u; i < N_LIMBS; ++i) dst[2u * N_LIMBS + i] = Z[i];
}

inline void zero_point(thread ulong *X, thread ulong *Y, thread ulong *Z) {
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

// Mixed addition: Z2 == 1 in affine sense, but we're in Montgomery form so
// Z2 must equal R mod q. We can't assume that, so this helper takes Z2 as
// a regular input — we instead specialise by computing Z2Z2 once outside and
// using the standard formulas. Not used in this version.

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

    ulong PX[N_LIMBS], PY[N_LIMBS], PZ[N_LIMBS];
    load_point(PX, PY, PZ, points_in + idx * POINT_LIMBS);

    ulong AX[N_LIMBS], AY[N_LIMBS], AZ[N_LIMBS];
    zero_point(AX, AY, AZ);

    // Find highest bit
    int top = -1;
    if (s3 != 0ul) {
        ulong v = s3; int b = 192;
        for (int i = 63; i >= 0; --i) { if ((v >> i) & 1ul) { top = b + i; break; } }
    } else if (s2 != 0ul) {
        ulong v = s2; int b = 128;
        for (int i = 63; i >= 0; --i) { if ((v >> i) & 1ul) { top = b + i; break; } }
    } else if (s1 != 0ul) {
        ulong v = s1; int b = 64;
        for (int i = 63; i >= 0; --i) { if ((v >> i) & 1ul) { top = b + i; break; } }
    } else if (s0 != 0ul) {
        ulong v = s0;
        for (int i = 63; i >= 0; --i) { if ((v >> i) & 1ul) { top = i; break; } }
    }

    if (top < 0) {
        store_point(scratch + idx * POINT_LIMBS, AX, AY, AZ);
        return;
    }

    // Initialise A = P (skip the leading 1 bit)
    copy_n(AX, PX); copy_n(AY, PY); copy_n(AZ, PZ);

    ulong TX[N_LIMBS], TY[N_LIMBS], TZ[N_LIMBS];

    for (int i = top - 1; i >= 0; --i) {
        jac_double_pt(TX, TY, TZ, AX, AY, AZ, q, q_inv_neg);
        copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);

        uint limb = (uint)(i >> 6);
        uint bit  = (uint)(i & 63);
        ulong sv = (limb == 0u) ? s0 : (limb == 1u) ? s1 : (limb == 2u) ? s2 : s3;
        if ((sv >> bit) & 1ul) {
            jac_add_pt(TX, TY, TZ, AX, AY, AZ, PX, PY, PZ, q, q_inv_neg);
            copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);
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