## Task: montgomery_msm

Multi-scalar multiplication on a short-Weierstrass elliptic curve. Given ``n_pairs`` pairs of (256-bit scalar ``s_i``, Jacobian point ``P_i``), compute the single curve point ``R = sum_i s_i * P_i`` and emit it in Jacobian Montgomery form. The in-distribution sizes use BLS12-381 G1 (q ~ 381 bits, b = 4); the held-out size uses BN254 G1 (q ~ 254 bits, b = 3).

Field representation: all elements live in Montgomery form with R = 2^384, six 64-bit limbs. The base-field modulus ``q`` (6 ulongs, little-endian) and the CIOS scalar ``q_inv_neg`` (``-q^-1 mod 2^64``) are bound as device / constant buffers; both **must** be read at runtime. A candidate that hardcodes the in-distribution modulus or its Montgomery constants silently produces wrong output on the held-out probe.

Coordinate convention: 6-limb Jacobian ``(X, Y, Z)``, little-endian limbs, affine point is ``(X / Z^2, Y / Z^3)``, ``Z = 0`` represents the point at infinity. Per point: 18 ulongs.

Scalars: 4-ulong little-endian limbs (both curves' scalar fields fit in 256 bits).

Bit-exact correctness: the host normalizes the GPU Jacobian output to affine Montgomery form via one base-field inversion, then compares the (X_aff_mont, Y_aff_mont) pair against the algebraic reference. A non-canonical limb (>= q) counts as a mismatch even if the residue class agrees.

Threadgroup-cooperative and simdgroup-cooperative implementations are valid so long as the external buffer layout above is preserved and the ``pair`` + ``log2(n_pairs)`` x ``reduce`` dispatch schedule is honored (the pair kernel sees each (scalar, point) pair exactly once; each reduce dispatch sees the current tree level via ``half_count``).

## Required kernel signature(s)

```
kernel void montgomery_msm_pair(
    device const ulong *scalars      [[buffer(0)]],
    device const ulong *points_in    [[buffer(1)]],
    device       ulong *scratch      [[buffer(2)]],
    device const ulong *q            [[buffer(3)]],
    constant ulong     &q_inv_neg    [[buffer(4)]],
    constant uint      &n_pairs      [[buffer(5)]],
    uint idx [[thread_position_in_grid]]);

kernel void montgomery_msm_reduce(
    device       ulong *scratch      [[buffer(0)]],
    device const ulong *q            [[buffer(1)]],
    constant ulong     &q_inv_neg    [[buffer(2)]],
    constant uint      &half_count   [[buffer(3)]],
    uint idx [[thread_position_in_grid]]);

Dispatch (host-fixed):
  montgomery_msm_pair: one thread per (s_i, P_i); guard against idx >= n_pairs; grid rounded up to a multiple of the TG width.
  montgomery_msm_reduce: invoked log2(n_pairs) times in a single compute command encoder with ``half_count`` successively halving (n_pairs/2, n_pairs/4, ..., 1). One thread per active slot; thread t reads scratch[t] and scratch[t + half_count], adds them in Jacobian form, and writes the sum back to scratch[t]. The serial command encoder gives read-after-write between levels with no explicit barriers required.
  threadsPerThreadgroup = (min(grid_w, 64), 1, 1) for both kernels in the seed; cooperative implementations may pick a different tile width but must honor the buffer layout and the half_count contract in reduce.
```

## Your previous attempt

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_LIMBS = 6u;
constexpr constant uint POINT_LIMBS = 18u;
constexpr constant uint WNAF_TABLE = 8u;

// ------------------------------------------------------------------
// 64x64 -> 128 using 32-bit products.
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
    ulong acc = 0ul;
    for (uint i = 0u; i < N_LIMBS; ++i) acc |= a[i];
    return acc == 0ul;
}

inline bool eq_n(thread const ulong *a, thread const ulong *b) {
    ulong acc = 0ul;
    for (uint i = 0u; i < N_LIMBS; ++i) acc |= (a[i] ^ b[i]);
    return acc == 0ul;
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
    for (uint i = 0u; i < N_LIMBS; ++i) c[i] = use_diff ? diff[i] : sum[i];
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
    for (uint i = 0u; i < N_LIMBS; ++i) c[i] = use_added ? added[i] : diff[i];
}

inline void mod_neg(thread ulong *c,
                    thread const ulong *a,
                    thread const ulong *q)
{
    ulong diff[N_LIMBS];
    ulong borrow = 0ul;
    ulong nz = 0ul;

    for (uint i = 0u; i < N_LIMBS; ++i) {
        nz |= a[i];
        ulong tv = q[i] - a[i];
        ulong b1 = (tv > q[i]) ? 1ul : 0ul;
        ulong d = tv - borrow;
        ulong b2 = (d > tv) ? 1ul : 0ul;
        diff[i] = d;
        borrow = b1 + b2;
    }

    bool z = (nz == 0ul);
    for (uint i = 0u; i < N_LIMBS; ++i) c[i] = z ? 0ul : diff[i];
}

// ------------------------------------------------------------------
// Unrolled 6-limb CIOS Montgomery multiplication:
// out = a*b*R^-1 mod q, R = 2^384.
// ------------------------------------------------------------------
#define MM_FMA(A, B, T, C)                                      \
    {                                                           \
        ulong2 rr = fma_add128((A), (B), (T), (C));             \
        (T) = rr.x;                                             \
        (C) = rr.y;                                             \
    }

#define MM_STEP(BI)                                             \
    {                                                           \
        ulong C = 0ul;                                          \
        MM_FMA(a0, (BI), t0, C)                                 \
        MM_FMA(a1, (BI), t1, C)                                 \
        MM_FMA(a2, (BI), t2, C)                                 \
        MM_FMA(a3, (BI), t3, C)                                 \
        MM_FMA(a4, (BI), t4, C)                                 \
        MM_FMA(a5, (BI), t5, C)                                 \
        ulong sA = t6 + C;                                      \
        ulong cyA = (sA < t6) ? 1ul : 0ul;                      \
        t6 = sA;                                                \
        t7 += cyA;                                              \
                                                                \
        ulong m = t0 * q_inv_neg;                               \
        C = 0ul;                                                \
        MM_FMA(m, q0, t0, C)                                    \
        MM_FMA(m, q1, t1, C)                                    \
        MM_FMA(m, q2, t2, C)                                    \
        MM_FMA(m, q3, t3, C)                                    \
        MM_FMA(m, q4, t4, C)                                    \
        MM_FMA(m, q5, t5, C)                                    \
        ulong sB = t6 + C;                                      \
        ulong cyB = (sB < t6) ? 1ul : 0ul;                      \
        t6 = sB;                                                \
        t7 += cyB;                                              \
                                                                \
        t0 = t1; t1 = t2; t2 = t3;                              \
        t3 = t4; t4 = t5; t5 = t6;                              \
        t6 = t7; t7 = 0ul;                                      \
    }

inline void mont_mul(thread ulong *out,
                     thread const ulong *a,
                     thread const ulong *b,
                     thread const ulong *q,
                     ulong q_inv_neg)
{
    ulong a0 = a[0], a1 = a[1], a2 = a[2];
    ulong a3 = a[3], a4 = a[4], a5 = a[5];
    ulong b0 = b[0], b1 = b[1], b2 = b[2];
    ulong b3 = b[3], b4 = b[4], b5 = b[5];
    ulong q0 = q[0], q1 = q[1], q2 = q[2];
    ulong q3 = q[3], q4 = q[4], q5 = q[5];

    ulong t0 = 0ul, t1 = 0ul, t2 = 0ul, t3 = 0ul;
    ulong t4 = 0ul, t5 = 0ul, t6 = 0ul, t7 = 0ul;

    MM_STEP(b0)
    MM_STEP(b1)
    MM_STEP(b2)
    MM_STEP(b3)
    MM_STEP(b4)
    MM_STEP(b5)

    ulong borrow = 0ul;

    ulong tv0 = t0 - q0;
    ulong brr0 = (tv0 > t0) ? 1ul : 0ul;
    ulong d0 = tv0 - borrow;
    ulong brr0b = (d0 > tv0) ? 1ul : 0ul;
    borrow = brr0 + brr0b;

    ulong tv1 = t1 - q1;
    ulong brr1 = (tv1 > t1) ? 1ul : 0ul;
    ulong d1 = tv1 - borrow;
    ulong brr1b = (d1 > tv1) ? 1ul : 0ul;
    borrow = brr1 + brr1b;

    ulong tv2 = t2 - q2;
    ulong brr2 = (tv2 > t2) ? 1ul : 0ul;
    ulong d2 = tv2 - borrow;
    ulong brr2b = (d2 > tv2) ? 1ul : 0ul;
    borrow = brr2 + brr2b;

    ulong tv3 = t3 - q3;
    ulong brr3 = (tv3 > t3) ? 1ul : 0ul;
    ulong d3 = tv3 - borrow;
    ulong brr3b = (d3 > tv3) ? 1ul : 0ul;
    borrow = brr3 + brr3b;

    ulong tv4 = t4 - q4;
    ulong brr4 = (tv4 > t4) ? 1ul : 0ul;
    ulong d4 = tv4 - borrow;
    ulong brr4b = (d4 > tv4) ? 1ul : 0ul;
    borrow = brr4 + brr4b;

    ulong tv5 = t5 - q5;
    ulong brr5 = (tv5 > t5) ? 1ul : 0ul;
    ulong d5 = tv5 - borrow;
    ulong brr5b = (d5 > tv5) ? 1ul : 0ul;
    borrow = brr5 + brr5b;

    bool use_diff = (t6 != 0ul) || (borrow == 0ul);
    out[0] = use_diff ? d0 : t0;
    out[1] = use_diff ? d1 : t1;
    out[2] = use_diff ? d2 : t2;
    out[3] = use_diff ? d3 : t3;
    out[4] = use_diff ? d4 : t4;
    out[5] = use_diff ? d5 : t5;
}

#undef MM_STEP
#undef MM_FMA

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

    mont_mul(Z1Z1, Z1, Z1, q, q_inv_neg);
    mont_mul(Z2Z2, Z2, Z2, q, q_inv_neg);
    mont_mul(U1, X1, Z2Z2, q, q_inv_neg);
    mont_mul(U2, X2, Z1Z1, q, q_inv_neg);
    mont_mul(tmp, Y1, Z2, q, q_inv_neg);
    mont_mul(S1, tmp, Z2Z2, q, q_inv_neg);
    mont_mul(tmp, Y2, Z1, q, q_inv_neg);
    mont_mul(S2, tmp, Z1Z1, q, q_inv_neg);

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

    mont_mul(HH, H, H, q, q_inv_neg);
    mont_mul(HHH, H, HH, q, q_inv_neg);
    mont_mul(V, U1, HH, q, q_inv_neg);

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

// ------------------------------------------------------------------
// Kernel A: per-pair scalar multiplication using width-5 wNAF.
// Table stores odd multiples 1P,3P,...,15P.
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

    ulong k0 = scalars[idx * 4u + 0u];
    ulong k1 = scalars[idx * 4u + 1u];
    ulong k2 = scalars[idx * 4u + 2u];
    ulong k3 = scalars[idx * 4u + 3u];
    ulong k4 = 0ul;

    ushort nz_pos[64];
    short  nz_dig[64];
    uint nz = 0u;

    for (uint bit = 0u; bit < 260u; ++bit) {
        if ((k0 | k1 | k2 | k3 | k4) == 0ul) break;

        if ((k0 & 1ul) != 0ul) {
            uint u = (uint)(k0 & 31ul);
            int d = (u > 16u) ? ((int)u - 32) : (int)u;

            nz_pos[nz] = (ushort)bit;
            nz_dig[nz] = (short)d;
            ++nz;

            if (d > 0) {
                k0 -= (ulong)d;
            } else {
                ulong addv = (ulong)(-d);
                ulong old = k0;
                k0 = old + addv;
                ulong cy = (k0 < old) ? 1ul : 0ul;
                if (cy != 0ul) {
                    old = k1; k1 = old + 1ul; cy = (k1 < old) ? 1ul : 0ul;
                    if (cy != 0ul) {
                        old = k2; k2 = old + 1ul; cy = (k2 < old) ? 1ul : 0ul;
                        if (cy != 0ul) {
                            old = k3; k3 = old + 1ul; cy = (k3 < old) ? 1ul : 0ul;
                            if (cy != 0ul) k4 += 1ul;
                        }
                    }
                }
            }
        }

        k0 = (k0 >> 1) | (k1 << 63);
        k1 = (k1 >> 1) | (k2 << 63);
        k2 = (k2 >> 1) | (k3 << 63);
        k3 = (k3 >> 1) | (k4 << 63);
        k4 = (k4 >> 1);
    }

    ulong AX[N_LIMBS], AY[N_LIMBS], AZ[N_LIMBS];

    if (nz == 0u) {
        zero_point(AX, AY, AZ);
        store_point(scratch + idx * POINT_LIMBS, AX, AY, AZ);
        return;
    }

    ulong tblX[WNAF_TABLE][N_LIMBS];
    ulong tblY[WNAF_TABLE][N_LIMBS];
    ulong tblZ[WNAF_TABLE][N_LIMBS];

    ulong TX[N_LIMBS], TY[N_LIMBS], TZ[N_LIMBS];

    load_point(tblX[0], tblY[0], tblZ[0], points_in + idx * POINT_LIMBS);

    // TX/TY/TZ temporarily hold 2P.
    jac_double_pt(TX, TY, TZ,
                  tblX[0], tblY[0], tblZ[0],
                  qloc, q_inv_neg);

    for (uint t = 1u; t < WNAF_TABLE; ++t) {
        jac_add_pt(tblX[t], tblY[t], tblZ[t],
                   tblX[t - 1u], tblY[t - 1u], tblZ[t - 1u],
                   TX, TY, TZ,
                   qloc, q_inv_neg);
    }

    int top_i = (int)nz - 1;
    short sd = nz_dig[(uint)top_i];
    bool neg = (sd < (short)0);
    uint ad = neg ? (uint)(-((int)sd)) : (uint)sd;
    uint ti = (ad - 1u) >> 1;

    for (uint i = 0u; i < N_LIMBS; ++i) {
        AX[i] = tblX[ti][i];
        AY[i] = tblY[ti][i];
        AZ[i] = tblZ[ti][i];
    }
    if (neg) mod_neg(AY, AY, qloc);

    uint prev_pos = (uint)nz_pos[(uint)top_i];

    for (int ii = top_i - 1; ii >= 0; --ii) {
        uint pos = (uint)nz_pos[(uint)ii];
        uint gap = prev_pos - pos;

        for (uint d = 0u; d < gap; ++d) {
            jac_double_pt(TX, TY, TZ, AX, AY, AZ, qloc, q_inv_neg);
            copy_point(AX, AY, AZ, TX, TY, TZ);
        }

        sd = nz_dig[(uint)ii];
        neg = (sd < (short)0);
        ad = neg ? (uint)(-((int)sd)) : (uint)sd;
        ti = (ad - 1u) >> 1;

        for (uint i = 0u; i < N_LIMBS; ++i) {
            TX[i] = tblX[ti][i];
            TY[i] = tblY[ti][i];
            TZ[i] = tblZ[ti][i];
        }
        if (neg) mod_neg(TY, TY, qloc);

        jac_add_pt(TX, TY, TZ,
                   AX, AY, AZ,
                   TX, TY, TZ,
                   qloc, q_inv_neg);
        copy_point(AX, AY, AZ, TX, TY, TZ);

        prev_pos = pos;
    }

    for (uint d = 0u; d < prev_pos; ++d) {
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
```

Result of previous attempt:
           bls_N4K: correct, 113.53 ms, 0.2 Gmodmul/s (int64) (0.3% of 53 Gops/s (int64 mul, est))
          bls_N16K: correct, 423.13 ms, 0.2 Gmodmul/s (int64) (0.3% of 53 Gops/s (int64 mul, est))
          bls_N64K: correct, 1528.80 ms, 0.2 Gmodmul/s (int64) (0.4% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0034

## Current best (incumbent)

```metal
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

inline ulong mullo64(ulong a, ulong b) {
    return a * b;
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

// Branchless conditional add-back to avoid SIMD-lane divergence.
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

        ulong m = mullo64(t[0], q_inv_neg);

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

    mod_sub(H, U2, U1, q);
    mod_sub(R, S2, S1, q);

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
```

Incumbent result:
           bls_N4K: correct, 54.91 ms, 0.3 Gmodmul/s (int64) (0.6% of 53 Gops/s (int64 mul, est))
          bls_N16K: correct, 207.88 ms, 0.4 Gmodmul/s (int64) (0.7% of 53 Gops/s (int64 mul, est))
          bls_N64K: correct, 784.44 ms, 0.4 Gmodmul/s (int64) (0.7% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0068

## History

- iter  0: compile=OK | correct=True | score=0.003965992898013797
- iter  1: compile=OK | correct=True | score=0.005376858798713791
- iter  2: compile=OK | correct=True | score=0.006839194989824701
- iter  3: compile=OK | correct=True | score=0.0033912607923107137

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
