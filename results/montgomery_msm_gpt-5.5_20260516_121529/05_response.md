I’m replacing per-thread scalar multiplication with a SIMD-group cooperative signed-window MSM over each 32-lane group. Each lane still reads its assigned pair, but the group accumulates a 32-pair partial MSM using radix-16 signed buckets, writes the group sum in lane 0, and zeros other slots. This removes most per-pair doublings/table precomputation and leaves the existing tree reduce to sum sparse partials. I also add zero-aware reduce fast paths to avoid field arithmetic for sparse scratch entries.

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