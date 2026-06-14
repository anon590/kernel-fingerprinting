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

inline void store_zero_point(device ulong *dst) {
    for (uint i = 0u; i < POINT_LIMBS; ++i) dst[i] = 0ul;
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

// Build, cooperatively over one SIMD group, the window sum:
//   W = sum_lanes signed_digit_lane * P_lane
// Digits must be in [-7, 8].  Only lane 0 receives nonzero W.
inline void simd_window_sum_signed(thread ulong *WX,
                                   thread ulong *WY,
                                   thread ulong *WZ,
                                   thread const ulong *PX,
                                   thread const ulong *PY,
                                   thread const ulong *PZ,
                                   int digit,
                                   uint lane,
                                   thread const ulong *q,
                                   ulong q_inv_neg)
{
    ulong BX[N_LIMBS], BY[N_LIMBS], BZ[N_LIMBS];
    ulong SX[N_LIMBS], SY[N_LIMBS], SZ[N_LIMBS];
    ulong RX[N_LIMBS], RY[N_LIMBS], RZ[N_LIMBS];
    ulong TX[N_LIMBS], TY[N_LIMBS], TZ[N_LIMBS];

    zero_point(BX, BY, BZ);

    // Bucket lanes 1..8 hold signed buckets for absolute digits 1..8.
    for (uint src = 0u; src < 32u; ++src) {
        int sd = simd_shuffle(digit, (ushort)src);
        if (sd != 0) {
            for (uint i = 0u; i < N_LIMBS; ++i) {
                SX[i] = simd_shuffle(PX[i], (ushort)src);
                SY[i] = simd_shuffle(PY[i], (ushort)src);
                SZ[i] = simd_shuffle(PZ[i], (ushort)src);
            }

            if (sd < 0) {
                mod_neg(SY, SY, q);
            }

            uint ad = (sd < 0) ? (uint)(-sd) : (uint)sd;
            if (lane == ad) {
                jac_add_pt(TX, TY, TZ,
                           BX, BY, BZ,
                           SX, SY, SZ,
                           q, q_inv_neg);
                copy_point(BX, BY, BZ, TX, TY, TZ);
            }
        }
    }

    zero_point(WX, WY, WZ);
    zero_point(RX, RY, RZ);

    // W = 8*B8 + 7*B7 + ... + B1 via running bucket sum.
    for (int k = 8; k >= 1; --k) {
        ushort src_lane = (ushort)k;
        for (uint i = 0u; i < N_LIMBS; ++i) {
            SX[i] = simd_shuffle(BX[i], src_lane);
            SY[i] = simd_shuffle(BY[i], src_lane);
            SZ[i] = simd_shuffle(BZ[i], src_lane);
        }

        if (lane == 0u) {
            jac_add_pt(TX, TY, TZ,
                       RX, RY, RZ,
                       SX, SY, SZ,
                       q, q_inv_neg);
            copy_point(RX, RY, RZ, TX, TY, TZ);

            jac_add_pt(TX, TY, TZ,
                       WX, WY, WZ,
                       RX, RY, RZ,
                       q, q_inv_neg);
            copy_point(WX, WY, WZ, TX, TY, TZ);
        }
    }
}

// ------------------------------------------------------------------
// Kernel A: SIMD-group cooperative 32-pair signed radix-16 MSM.
// Lane 0 writes the SIMD-group partial sum; other valid lanes write zero.
// The following tree reduction then sums these sparse partials.
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
    uint lane = idx & 31u;
    bool valid = (idx < n_pairs);

    ulong qloc[N_LIMBS];
    for (uint i = 0u; i < N_LIMBS; ++i) qloc[i] = q[i];

    ulong s0 = 0ul, s1 = 0ul, s2 = 0ul, s3 = 0ul;
    ulong PX[N_LIMBS], PY[N_LIMBS], PZ[N_LIMBS];
    zero_point(PX, PY, PZ);

    if (valid) {
        uint soff = idx * 4u;
        s0 = scalars[soff + 0u];
        s1 = scalars[soff + 1u];
        s2 = scalars[soff + 2u];
        s3 = scalars[soff + 3u];

        load_point(PX, PY, PZ, points_in + idx * POINT_LIMBS);
    }

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

    ulong AX[N_LIMBS], AY[N_LIMBS], AZ[N_LIMBS];
    ulong WX[N_LIMBS], WY[N_LIMBS], WZ[N_LIMBS];
    ulong TX[N_LIMBS], TY[N_LIMBS], TZ[N_LIMBS];

    // Optional top signed-radix carry at digit 64.
    simd_window_sum_signed(AX, AY, AZ,
                           PX, PY, PZ,
                           (int)carry,
                           lane,
                           qloc, q_inv_neg);

    for (int win = 63; win >= 0; --win) {
        if (lane == 0u) {
            jac_double_pt(TX, TY, TZ, AX, AY, AZ, qloc, q_inv_neg);
            copy_point(AX, AY, AZ, TX, TY, TZ);
            jac_double_pt(TX, TY, TZ, AX, AY, AZ, qloc, q_inv_neg);
            copy_point(AX, AY, AZ, TX, TY, TZ);
            jac_double_pt(TX, TY, TZ, AX, AY, AZ, qloc, q_inv_neg);
            copy_point(AX, AY, AZ, TX, TY, TZ);
            jac_double_pt(TX, TY, TZ, AX, AY, AZ, qloc, q_inv_neg);
            copy_point(AX, AY, AZ, TX, TY, TZ);
        }

        int d = (int)digs[(uint)win];
        simd_window_sum_signed(WX, WY, WZ,
                               PX, PY, PZ,
                               d,
                               lane,
                               qloc, q_inv_neg);

        if (lane == 0u) {
            jac_add_pt(TX, TY, TZ,
                       AX, AY, AZ,
                       WX, WY, WZ,
                       qloc, q_inv_neg);
            copy_point(AX, AY, AZ, TX, TY, TZ);
        }
    }

    if (valid) {
        device ulong *dst = scratch + idx * POINT_LIMBS;
        if (lane == 0u) {
            store_point(dst, AX, AY, AZ);
        } else {
            store_zero_point(dst);
        }
    }
}

// ------------------------------------------------------------------
// Kernel B: one tree-reduction level, with sparse-zero fast paths.
// ------------------------------------------------------------------
kernel void montgomery_msm_reduce(
    device       ulong *scratch      [[buffer(0)]],
    device const ulong *q            [[buffer(1)]],
    constant ulong     &q_inv_neg    [[buffer(2)]],
    constant uint      &half_count   [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= half_count) return;

    device ulong *Aptr = scratch + idx * POINT_LIMBS;
    device ulong *Bptr = scratch + (idx + half_count) * POINT_LIMBS;

    ulong AZ0[N_LIMBS], BZ0[N_LIMBS];
    bool z1 = true;
    bool z2 = true;
    for (uint i = 0u; i < N_LIMBS; ++i) {
        AZ0[i] = Aptr[2u * N_LIMBS + i];
        BZ0[i] = Bptr[2u * N_LIMBS + i];
        if (AZ0[i] != 0ul) z1 = false;
        if (BZ0[i] != 0ul) z2 = false;
    }

    if (z2) {
        return;
    }

    if (z1) {
        for (uint i = 0u; i < POINT_LIMBS; ++i) {
            Aptr[i] = Bptr[i];
        }
        return;
    }

    ulong qloc[N_LIMBS];
    for (uint i = 0u; i < N_LIMBS; ++i) qloc[i] = q[i];

    ulong AX[N_LIMBS], AY[N_LIMBS], AZ[N_LIMBS];
    ulong BX[N_LIMBS], BY[N_LIMBS], BZ[N_LIMBS];

    for (uint i = 0u; i < N_LIMBS; ++i) {
        AX[i] = Aptr[i];
        AY[i] = Aptr[N_LIMBS + i];
        AZ[i] = AZ0[i];

        BX[i] = Bptr[i];
        BY[i] = Bptr[N_LIMBS + i];
        BZ[i] = BZ0[i];
    }

    ulong RX[N_LIMBS], RY[N_LIMBS], RZ[N_LIMBS];
    jac_add_pt(RX, RY, RZ,
               AX, AY, AZ,
               BX, BY, BZ,
               qloc, q_inv_neg);

    store_point(Aptr, RX, RY, RZ);
}
```

Result of previous attempt:
  COMPILE FAILED: Error Domain=MTLLibraryErrorDomain Code=3 "program_source:400:25: error: no matching function for call to 'simd_shuffle'
                SX[i] = simd_shuffle(PX[i], (ushort)src);
                        ^~~~~~~~~~~~
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_simdgroup:207:14: note: candidate template ignored: requirement '__is_valid_simdgroup_type<unsigned long, void>::value' was not satisfied [with T = unsigned long]
METAL_FUNC T simd_shuffle(T data METAL_MAYBE_UNDEF, ushort simd_lane_id)
             ^
program_source:401:25: error: no matching function for call to 'simd_shuffle'
                SY[i] = simd_shuffle(PY[i], (ushort)src);
                        ^~~~~~~~~~~~
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_simdgroup:207:14: note: candidate template ignored: requirement '__is_valid_simdgroup_type<unsigned long, void>::value' was not satisfied [with T = unsigned long]
METAL_FUNC T simd_shuffle(T data METAL_MAYBE_UNDEF, ushort simd_lane_id)
             ^
program_source:402:25: error: no matching function for call to 'simd_shuffle'
                SZ[i] = simd_shuffle(PZ[i], (ushort)src);
                        ^~~~~~~~~~~~
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_simdgroup:207:14: note: candidate template ignored: requirement '__is_valid_simdgroup_type<unsigned long, void>::value' was not satisfied [with T = unsigned long]
METAL_FUNC T simd_shuffle(T data METAL_MAYBE_UNDEF, ushort simd_lane_id)
             ^
program_source:427:21: error: no matching function for call to 'simd_shuffle'
            SX[i] = simd_shuffle(BX[i], src_lane);
                    ^~~~~~~~~~~~
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_simdgroup:207:14: note: candidate template ignored: requirement '__is_valid_simdgroup_type<unsigned long, void>::value' was not satisfied [with T = unsigned long]
METAL_FUNC T simd_shuffle(T data METAL_MAYBE_UNDEF, ushort simd_lane_id)
             ^
program_source:428:21: error: no matching function for call to 'simd_shuffle'
            SY[i] = simd_shuffle(BY[i], src_lane);
                    ^~~~~~~~~~~~
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_simdgroup:207:14: note: candidate template ignored: requirement '__is_valid_simdgroup_type<unsigned long, void>::value' was not satisfied [with T = unsigned long]
METAL_FUNC T simd_shuffle(T data METAL_MAYBE_UNDEF, ushort simd_lane_id)
             ^
program_source:429:21: error: no matching function for call to 'simd_shuffle'
            SZ[i] = simd_shuffle(BZ[i], src_lane);
                    ^~~~~~~~~~~~
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_simdgroup:207:14: note: candidate template ignored: requirement '__is_valid_simdgroup_type<unsigned long, void>::value' was not satisfied [with T = unsigned long]
METAL_FUNC T simd_shuffle(T data METAL_MAYBE_UNDEF, ushort simd_lane_id)
             ^
" UserInfo={NSLocalizedDescription=program_source:400:25: error: no matching function for call to 'simd_shuffle'
                SX[i] = simd_shuffle(PX[i], (ushort)src);
                        ^~~~~~~~~~~~
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_simdgroup:207:14: note: candidate template ignored: requirement '__is_valid_simdgroup_type<unsigned long, void>::value' was not satisfied [with T = unsigned long]
METAL_FUNC T simd_shuffle(T data METAL_MAYBE_UNDEF, ushort simd_lane_id)
             ^
program_source:401:25: error: no matching function for call to 'simd_shuffle'
                SY[i] = simd_shuffle(PY[i], (ushort)src);
                        ^~~~~~~~~~~~
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_simdgroup:207:14: note: candidate template ignored: requirement '__is_valid_simdgroup_type<unsigned long, void>::value' was not satisfied [with T = unsigned long]
METAL_FUNC T simd_shuffle(T data METAL_MAYBE_UNDEF, ushort simd_lane_id)
             ^
program_source:402:25: error: no matching function for call to 'simd_shuffle'
                SZ[i] = simd_shuffle(PZ[i], (ushort)src);
                        ^~~~~~~~~~~~
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_simdgroup:207:14: note: candidate template ignored: requirement '__is_valid_simdgroup_type<unsigned long, void>::value' was not satisfied [with T = unsigned long]
METAL_FUNC T simd_shuffle(T data METAL_MAYBE_UNDEF, ushort simd_lane_id)
             ^
program_source:427:21: error: no matching function for call to 'simd_shuffle'
            SX[i] = simd_shuffle(BX[i], src_lane);
                    ^~~~~~~~~~~~
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_simdgroup:207:14: note: candidate template ignored: requirement '__is_valid_simdgroup_type<unsigned long, void>::value' was not satisfied [with T = unsigned long]
METAL_FUNC T simd_shuffle(T data METAL_MAYBE_UNDEF, ushort simd_lane_id)
             ^
program_source:428:21: error: no matching function for call to 'simd_shuffle'
            SY[i] = simd_shuffle(BY[i], src_lane);
                    ^~~~~~~~~~~~
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_simdgroup:207:14: note: candidate template ignored: requirement '__is_valid_simdgroup_type<unsigned long, void>::value' was not satisfied [with T = unsigned long]
METAL_FUNC T simd_shuffle(T data METAL_MAYBE_UNDEF, ushort simd_lane_id)
             ^
program_source:429:21: error: no matching function for call to 'simd_shuffle'
            SZ[i] = simd_shuffle(BZ[i], src_lane);
                    ^~~~~~~~~~~~
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_simdgroup:207:14: note: candidate template ignored: requirement '__is_valid_simdgroup_type<unsigned long, void>::value' was not satisfied [with T = unsigned long]
METAL_FUNC T simd_shuffle(T data METAL_MAYBE_UNDEF, ushort simd_lane_id)
             ^
}

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
- iter  4: compile=OK | correct=True | score=0.0047044960548237955
- iter  5: compile=FAIL | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
