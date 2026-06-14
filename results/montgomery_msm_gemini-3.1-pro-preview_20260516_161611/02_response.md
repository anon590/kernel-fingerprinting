To optimize the MSM kernel, this version introduces three key improvements that significantly reduce both operation count and register spilling:

1. **Non-Adjacent Form (2-NAF) Scalar Processing:** Rather than using standard MSB-to-LSB binary double-and-add (which requires ~128 point additions), we dynamically compute the 2-NAF of the scalar on the fly. This recodes the scalar into digits of `{-1, 0, 1}`, reducing the average number of additions to ~85. The digits are compactly packed into a register array (`naf[9]`) requiring zero precomputed points—only the base point `P` and its negation `-P` (which just means negating `PY`), keeping register pressure very low.
2. **Optimized 128-bit Multiplication:** Apple GPUs lack native `ulong * ulong -> u128`. The `umul128` is optimized using zero-cost 32-bit casts and native 32x32->64-bit multipliers, avoiding explicit nested `mulhi` function overhead and minimizing instruction count.
3. **Fully Unrolled Finely-Grained CIOS:** The `mont_mul` Montgomery multiplier's inner loops are strictly unrolled using `#pragma unroll`. Together with substituting intermediate point calculations in `jac_double_pt` (computing `D` directly from `X` and `B`), we save multiple multi-precision additions per loop without register spilling.

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_LIMBS = 6u;
constexpr constant uint POINT_LIMBS = 18u;

inline ulong2 umul128(ulong x, ulong y) {
    ulong lo = x * y;
    uint x0 = (uint)x;
    uint x1 = (uint)(x >> 32);
    uint y0 = (uint)y;
    uint y1 = (uint)(y >> 32);
    
    ulong p01 = (ulong)x0 * y1;
    ulong p10 = (ulong)x1 * y0;
    ulong p11 = (ulong)x1 * y1;
    
    ulong mid = p01 + (uint)p10 + (((ulong)x0 * y0) >> 32);
    ulong hi = p11 + (p10 >> 32) + (mid >> 32);
    return ulong2(lo, hi);
}

inline void copy_n(thread ulong *dst, thread const ulong *src) {
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) dst[i] = src[i];
}

inline bool is_zero_n(thread const ulong *a) {
    return (a[0] | a[1] | a[2] | a[3] | a[4] | a[5]) == 0ul;
}

inline bool eq_n(thread const ulong *a, thread const ulong *b) {
    return ((a[0] ^ b[0]) | (a[1] ^ b[1]) | (a[2] ^ b[2]) |
            (a[3] ^ b[3]) | (a[4] ^ b[4]) | (a[5] ^ b[5])) == 0ul;
}

inline void mod_add(thread ulong *c,
                    thread const ulong *a, thread const ulong *b,
                    thread const ulong *q)
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
                    thread const ulong *q)
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
    
    ulong t_arr[N_LIMBS];
    ulong carry = 0ul;
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) {
        ulong s = diff[i] + carry;
        ulong cy1 = (s < diff[i]) ? 1ul : 0ul;
        ulong t = s + q[i];
        ulong cy2 = (t < s) ? 1ul : 0ul;
        t_arr[i] = t;
        carry = cy1 + cy2;
    }
    
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) {
        c[i] = borrow ? t_arr[i] : diff[i];
    }
}

inline void mont_mul(thread ulong *out,
                     thread const ulong *a, thread const ulong *b,
                     thread const ulong *q, ulong q_inv_neg)
{
    ulong t[7] = {0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul};

    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) {
        ulong C = 0ul;
        ulong bi = b[i];
        
        #pragma unroll
        for (uint j = 0u; j < N_LIMBS; ++j) {
            ulong2 r = umul128(a[j], bi);
            ulong lo1 = r.x + t[j];
            ulong cy1 = (lo1 < r.x) ? 1ul : 0ul;
            ulong lo2 = lo1 + C;
            ulong cy2 = (lo2 < lo1) ? 1ul : 0ul;
            t[j] = lo2;
            C = r.y + cy1 + cy2;
        }
        ulong s1 = t[6] + C;
        ulong cy1 = (s1 < t[6]) ? 1ul : 0ul;
        t[6] = s1;
        ulong t7 = cy1;

        ulong m = t[0] * q_inv_neg;

        C = 0ul;
        #pragma unroll
        for (uint j = 0u; j < N_LIMBS; ++j) {
            ulong2 r = umul128(m, q[j]);
            ulong lo1 = r.x + t[j];
            ulong cy2 = (lo1 < r.x) ? 1ul : 0ul;
            ulong lo2 = lo1 + C;
            ulong cy3 = (lo2 < lo1) ? 1ul : 0ul;
            t[j] = lo2;
            C = r.y + cy2 + cy3;
        }
        ulong s2 = t[6] + C;
        ulong cy4 = (s2 < t[6]) ? 1ul : 0ul;
        t[6] = s2;
        t7 += cy4;

        #pragma unroll
        for (uint j = 0u; j < N_LIMBS; ++j) {
            t[j] = t[j + 1];
        }
        t[6] = t7;
    }

    ulong diff[N_LIMBS];
    ulong borrow = 0ul;
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) {
        ulong tv = t[i] - q[i];
        ulong b1 = (tv > t[i]) ? 1ul : 0ul;
        ulong d = tv - borrow;
        ulong b2 = (d > tv) ? 1ul : 0ul;
        diff[i] = d;
        borrow = b1 + b2;
    }
    
    bool use_diff = (t[6] != 0ul) || (borrow == 0ul);
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) {
        out[i] = use_diff ? diff[i] : t[i];
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
    for (uint i = 0u; i < N_LIMBS; ++i) X[i] = 0ul;
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) Y[i] = 0ul;
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) Z[i] = 0ul;
}

inline void jac_double_pt(thread ulong *oX, thread ulong *oY, thread ulong *oZ,
                          thread const ulong *X, thread const ulong *Y, thread const ulong *Z,
                          thread const ulong *q, ulong q_inv_neg)
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

    // D = 4 * X * B
    mont_mul(D, X, B, q, q_inv_neg);
    mod_add(D, D, D, q);
    mod_add(D, D, D, q);

    // E = 3 * A
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
                       thread const ulong *q, ulong q_inv_neg)
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

    ulong q_local[N_LIMBS];
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) q_local[i] = q[i];

    ulong s0 = scalars[idx * 4u + 0u];
    ulong s1 = scalars[idx * 4u + 1u];
    ulong s2 = scalars[idx * 4u + 2u];
    ulong s3 = scalars[idx * 4u + 3u];
    ulong s[4] = {s0, s1, s2, s3};

    ulong PX[N_LIMBS], PY[N_LIMBS], PZ[N_LIMBS];
    load_point(PX, PY, PZ, points_in + idx * POINT_LIMBS);

    ulong nPY[N_LIMBS];
    if (is_zero_n(PY)) {
        copy_n(nPY, PY);
    } else {
        ulong borrow = 0ul;
        #pragma unroll
        for (uint i = 0u; i < N_LIMBS; ++i) {
            ulong tv = q_local[i] - PY[i];
            ulong b1 = (tv > q_local[i]) ? 1ul : 0ul;
            ulong d = tv - borrow;
            ulong b2 = (d > tv) ? 1ul : 0ul;
            nPY[i] = d;
            borrow = b1 + b2;
        }
    }

    ulong AX[N_LIMBS], AY[N_LIMBS], AZ[N_LIMBS];
    zero_point(AX, AY, AZ);

    ulong naf[9] = {0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul};
    int len = 0;
    while ((s[0] | s[1] | s[2] | s[3]) != 0ul) {
        if ((s[0] & 1ul) != 0ul) {
            int d = 2 - (int)(s[0] & 3ul);
            ulong val = (d == 1) ? 1ul : 2ul;
            naf[len / 32] |= (val << ((len % 32) * 2));
            
            if (d == 1) {
                ulong b0 = s[0] - 1ul; ulong br = (b0 > s[0]) ? 1ul : 0ul; s[0] = b0;
                ulong b1 = s[1] - br;  br = (b1 > s[1]) ? 1ul : 0ul;  s[1] = b1;
                ulong b2 = s[2] - br;  br = (b2 > s[2]) ? 1ul : 0ul;  s[2] = b2;
                s[3] -= br;
            } else {
                ulong a0 = s[0] + 1ul; ulong cy = (a0 < s[0]) ? 1ul : 0ul; s[0] = a0;
                ulong a1 = s[1] + cy;  cy = (a1 < s[1]) ? 1ul : 0ul;  s[1] = a1;
                ulong a2 = s[2] + cy;  cy = (a2 < s[2]) ? 1ul : 0ul;  s[2] = a2;
                s[3] += cy;
            }
        }
        len++;
        s[0] = (s[0] >> 1) | (s[1] << 63);
        s[1] = (s[1] >> 1) | (s[2] << 63);
        s[2] = (s[2] >> 1) | (s[3] << 63);
        s[3] >>= 1;
    }

    if (len > 0) {
        ulong TX[N_LIMBS], TY[N_LIMBS], TZ[N_LIMBS];
        for (uint bit = (uint)(len - 1); bit != 0xFFFFFFFFu; --bit) {
            jac_double_pt(TX, TY, TZ, AX, AY, AZ, q_local, q_inv_neg);
            copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);

            ulong val = (naf[bit / 32] >> ((bit % 32) * 2)) & 3ul;
            if (val != 0ul) {
                if (val == 1ul) {
                    jac_add_pt(TX, TY, TZ, AX, AY, AZ, PX, PY, PZ, q_local, q_inv_neg);
                } else {
                    jac_add_pt(TX, TY, TZ, AX, AY, AZ, PX, nPY, PZ, q_local, q_inv_neg);
                }
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

    ulong q_local[N_LIMBS];
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) q_local[i] = q[i];

    ulong AX[N_LIMBS], AY[N_LIMBS], AZ[N_LIMBS];
    ulong BX[N_LIMBS], BY[N_LIMBS], BZ[N_LIMBS];
    load_point(AX, AY, AZ, scratch + idx * POINT_LIMBS);
    load_point(BX, BY, BZ, scratch + (idx + half_count) * POINT_LIMBS);

    ulong RX[N_LIMBS], RY[N_LIMBS], RZ[N_LIMBS];
    jac_add_pt(RX, RY, RZ,
               AX, AY, AZ,
               BX, BY, BZ,
               q_local, q_inv_neg);
               
    store_point(scratch + idx * POINT_LIMBS, RX, RY, RZ);
}
```