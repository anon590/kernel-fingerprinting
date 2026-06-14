This optimization significantly boosts performance through several techniques:
1. **w-NAF (Windowed Non-Adjacent Form) Scalar Multiplication:** Implements a windowed NAF with $w=5$, dynamically packing the scalar bits. This slashes the average number of point additions from ~128 down to ~42.
2. **Algorithm Tweak:** Replaces $2 \times ((X+B)^2 - A - C)$ with the mathematically equivalent $4 \times X \times B$ in point doubling, saving multiple expensive modular operations.
3. **Loop Unrolling & Memory:** Unrolls the CIOS Montgomery multiplication entirely. The 384-bit modulus `q` is loaded into thread registers (`q_local`), avoiding hundreds of redundant device memory fetches per thread. 
4. **Branchless Arithmetic:** Refactors `mod_add` and `mod_sub` into branchless instruction flows, perfectly suited for the GPU pipeline.
5. **Instruction Efficiency:** Employs optimized `umul128` explicitly decomposing into 32-bit sub-words to match native hardware performance.

```metal
#include <metal_stdlib>
using namespace metal;

inline ulong2 umul128(ulong a, ulong b) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    ulong p00 = (ulong)a0 * b0;
    ulong p01 = (ulong)a0 * b1;
    ulong p10 = (ulong)a1 * b0;
    ulong p11 = (ulong)a1 * b1;

    ulong mid = (p00 >> 32) + (uint)p01 + (uint)p10;
    ulong lo = (p00 & 0xFFFFFFFFul) | (mid << 32);
    ulong hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return ulong2(lo, hi);
}

inline void copy_n(thread ulong *dst, thread const ulong *src) {
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) dst[i] = src[i];
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
    ulong sum[6];
    ulong carry = 0ul;
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) {
        ulong s = a[i] + carry;
        ulong cy1 = s < a[i] ? 1ul : 0ul;
        ulong t = s + b[i];
        ulong cy2 = t < s ? 1ul : 0ul;
        sum[i] = t;
        carry = cy1 + cy2;
    }
    
    ulong diff[6];
    ulong borrow = 0ul;
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) {
        ulong tv = sum[i] - q[i];
        ulong b1 = tv > sum[i] ? 1ul : 0ul;
        ulong d = tv - borrow;
        ulong b2 = d > tv ? 1ul : 0ul;
        diff[i] = d;
        borrow = b1 + b2;
    }
    
    bool use_diff = (carry != 0ul) || (borrow == 0ul);
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) {
        c[i] = use_diff ? diff[i] : sum[i];
    }
}

inline void mod_sub(thread ulong *c,
                    thread const ulong *a, thread const ulong *b,
                    thread const ulong *q)
{
    ulong diff[6];
    ulong borrow = 0ul;
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) {
        ulong tv = a[i] - b[i];
        ulong b1 = tv > a[i] ? 1ul : 0ul;
        ulong d = tv - borrow;
        ulong b2 = d > tv ? 1ul : 0ul;
        diff[i] = d;
        borrow = b1 + b2;
    }
    
    ulong t_arr[6];
    ulong carry = 0ul;
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) {
        ulong s = diff[i] + carry;
        ulong cy1 = s < diff[i] ? 1ul : 0ul;
        ulong t = s + q[i];
        ulong cy2 = t < s ? 1ul : 0ul;
        t_arr[i] = t;
        carry = cy1 + cy2;
    }
    
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) {
        c[i] = borrow ? t_arr[i] : diff[i];
    }
}

inline void mont_mul(thread ulong *out,
                     thread const ulong *a, thread const ulong *b,
                     thread const ulong *q, ulong q_inv_neg)
{
    ulong t[7] = {0, 0, 0, 0, 0, 0, 0};

    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) {
        ulong C = 0ul;
        ulong bi = b[i];
        
        #pragma unroll
        for (uint j = 0u; j < 6u; ++j) {
            ulong2 r = umul128(a[j], bi);
            ulong lo1 = r.x + t[j];
            ulong cy1 = lo1 < r.x ? 1ul : 0ul;
            ulong lo2 = lo1 + C;
            ulong cy2 = lo2 < lo1 ? 1ul : 0ul;
            t[j] = lo2;
            C = r.y + cy1 + cy2;
        }
        ulong s1 = t[6] + C;
        ulong cy1 = s1 < t[6] ? 1ul : 0ul;
        t[6] = s1;
        ulong next_C = cy1;

        ulong m = t[0] * q_inv_neg;

        C = 0ul;
        #pragma unroll
        for (uint j = 0u; j < 6u; ++j) {
            ulong2 r = umul128(m, q[j]);
            ulong lo1 = r.x + t[j];
            ulong cy1 = lo1 < r.x ? 1ul : 0ul;
            ulong lo2 = lo1 + C;
            ulong cy2 = lo2 < lo1 ? 1ul : 0ul;
            t[j] = lo2;
            C = r.y + cy1 + cy2;
        }
        ulong s2 = t[6] + C;
        ulong cy2 = s2 < t[6] ? 1ul : 0ul;
        t[6] = s2;
        next_C += cy2;

        #pragma unroll
        for (uint j = 0u; j < 6u; ++j) {
            t[j] = t[j + 1];
        }
        t[6] = next_C;
    }

    ulong diff[6];
    ulong borrow = 0ul;
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) {
        ulong tv = t[i] - q[i];
        ulong b1 = tv > t[i] ? 1ul : 0ul;
        ulong d = tv - borrow;
        ulong b2 = d > tv ? 1ul : 0ul;
        diff[i] = d;
        borrow = b1 + b2;
    }
    
    bool use_diff = (t[6] != 0ul) || (borrow == 0ul);
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) {
        out[i] = use_diff ? diff[i] : t[i];
    }
}

inline void load_point(thread ulong *X, thread ulong *Y, thread ulong *Z,
                       device const ulong *src)
{
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) X[i] = src[i];
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) Y[i] = src[6u + i];
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) Z[i] = src[12u + i];
}

inline void store_point(device ulong *dst,
                        thread const ulong *X, thread const ulong *Y, thread const ulong *Z)
{
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) dst[i] = X[i];
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) dst[6u + i] = Y[i];
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) dst[12u + i] = Z[i];
}

inline void zero_point(thread ulong *X, thread ulong *Y, thread ulong *Z) {
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) X[i] = 0ul;
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) Y[i] = 0ul;
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) Z[i] = 0ul;
}

inline void jac_double_pt(thread ulong *oX, thread ulong *oY, thread ulong *oZ,
                          thread const ulong *X, thread const ulong *Y, thread const ulong *Z,
                          thread const ulong *q, ulong q_inv_neg)
{
    if (is_zero_n(Z) || is_zero_n(Y)) {
        zero_point(oX, oY, oZ);
        return;
    }
    
    ulong A[6], B[6], C[6];
    ulong D[6], E[6], F[6];
    ulong tmp[6], tmp2[6];

    mont_mul(A, X, X, q, q_inv_neg);
    mont_mul(B, Y, Y, q, q_inv_neg);
    mont_mul(C, B, B, q, q_inv_neg);

    // D = 4 * X * Y^2 = 4 * X * B
    mont_mul(D, X, B, q, q_inv_neg);
    mod_add(D, D, D, q);
    mod_add(D, D, D, q);

    // E = 3 * X^2 = 3 * A
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
    
    ulong Z1Z1[6], Z2Z2[6];
    ulong U1[6], U2[6], S1[6], S2[6];
    ulong H[6], R[6];
    ulong HH[6], HHH[6], V[6];
    ulong tmp[6], tmp2[6];

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

    ulong q_local[6];
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) q_local[i] = q[i];

    ulong s0 = scalars[idx * 4u + 0u];
    ulong s1 = scalars[idx * 4u + 1u];
    ulong s2 = scalars[idx * 4u + 2u];
    ulong s3 = scalars[idx * 4u + 3u];

    ulong PX[6], PY[6], PZ[6];
    load_point(PX, PY, PZ, points_in + idx * 18u);

    ulong AX[6], AY[6], AZ[6];
    zero_point(AX, AY, AZ);

    char naf[260];
    int len = 0;
    ulong s[4] = {s0, s1, s2, s3};
    while ((s[0] | s[1] | s[2] | s[3]) != 0ul) {
        if ((s[0] & 1ul) != 0ul) {
            int d = (int)(s[0] & 31ul);
            if (d > 15) d -= 32;
            naf[len] = (char)d;
            
            if (d > 0) {
                ulong b0 = s[0] - d; ulong br = b0 > s[0] ? 1ul : 0ul; s[0] = b0;
                ulong b1 = s[1] - br; br = b1 > s[1] ? 1ul : 0ul; s[1] = b1;
                ulong b2 = s[2] - br; br = b2 > s[2] ? 1ul : 0ul; s[2] = b2;
                s[3] -= br;
            } else {
                ulong ad = -d;
                ulong a0 = s[0] + ad; ulong cy = a0 < s[0] ? 1ul : 0ul; s[0] = a0;
                ulong a1 = s[1] + cy; cy = a1 < s[1] ? 1ul : 0ul; s[1] = a1;
                ulong a2 = s[2] + cy; cy = a2 < s[2] ? 1ul : 0ul; s[2] = a2;
                s[3] += cy;
            }
        } else {
            naf[len] = 0;
        }
        len++;
        s[0] = (s[0] >> 1) | (s[1] << 63);
        s[1] = (s[1] >> 1) | (s[2] << 63);
        s[2] = (s[2] >> 1) | (s[3] << 63);
        s[3] >>= 1;
    }

    if (len > 0) {
        ulong P_X[8][6], P_Y[8][6], P_Z[8][6];
        copy_n(P_X[0], PX); copy_n(P_Y[0], PY); copy_n(P_Z[0], PZ);
        ulong P2X[6], P2Y[6], P2Z[6];
        jac_double_pt(P2X, P2Y, P2Z, PX, PY, PZ, q_local, q_inv_neg);
        for(int i = 1; i < 8; ++i) {
            jac_add_pt(P_X[i], P_Y[i], P_Z[i], P_X[i-1], P_Y[i-1], P_Z[i-1], P2X, P2Y, P2Z, q_local, q_inv_neg);
        }

        ulong TX[6], TY[6], TZ[6];
        for (int bit = len - 1; bit >= 0; --bit) {
            jac_double_pt(TX, TY, TZ, AX, AY, AZ, q_local, q_inv_neg);
            copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);

            int d = naf[bit];
            if (d != 0) {
                ulong SX[6], SY[6], SZ[6];
                int abs_d = d < 0 ? -d : d;
                int pt_idx = (abs_d - 1) / 2;
                
                copy_n(SX, P_X[pt_idx]);
                copy_n(SZ, P_Z[pt_idx]);
                
                if (d < 0) {
                    if (is_zero_n(P_Y[pt_idx])) {
                        copy_n(SY, P_Y[pt_idx]);
                    } else {
                        ulong borrow = 0ul;
                        #pragma unroll
                        for (uint j = 0u; j < 6u; ++j) {
                            ulong tv = q_local[j] - P_Y[pt_idx][j];
                            ulong b1 = tv > q_local[j] ? 1ul : 0ul;
                            ulong sub_d = tv - borrow;
                            ulong b2 = sub_d > tv ? 1ul : 0ul;
                            SY[j] = sub_d;
                            borrow = b1 + b2;
                        }
                    }
                } else {
                    copy_n(SY, P_Y[pt_idx]);
                }
                
                jac_add_pt(TX, TY, TZ, AX, AY, AZ, SX, SY, SZ, q_local, q_inv_neg);
                copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);
            }
        }
    }

    store_point(scratch + idx * 18u, AX, AY, AZ);
}

kernel void montgomery_msm_reduce(
    device       ulong *scratch      [[buffer(0)]],
    device const ulong *q            [[buffer(1)]],
    constant ulong     &q_inv_neg    [[buffer(2)]],
    constant uint      &half_count   [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= half_count) return;

    ulong q_local[6];
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) q_local[i] = q[i];

    ulong AX[6], AY[6], AZ[6];
    ulong BX[6], BY[6], BZ[6];
    load_point(AX, AY, AZ, scratch + idx * 18u);
    load_point(BX, BY, BZ, scratch + (idx + half_count) * 18u);

    ulong RX[6], RY[6], RZ[6];
    jac_add_pt(RX, RY, RZ,
               AX, AY, AZ,
               BX, BY, BZ,
               q_local, q_inv_neg);
               
    store_point(scratch + idx * 18u, RX, RY, RZ);
}
```