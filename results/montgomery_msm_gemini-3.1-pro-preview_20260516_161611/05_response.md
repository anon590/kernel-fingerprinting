This optimization applies **Radix-8 Booth Encoding** (a 3-bit window) to the scalar multiplication, reading the scalar from MSB to LSB. By precomputing just three additional points ($2P$, $3P$, $4P$) — which comfortably fit in registers without spilling to local memory — we reduce the number of costly point additions by ~50% compared to standard double-and-add. 

Additionally, the base field modulus `q` is loaded from device memory into thread-local registers at the start of the kernel, avoiding thousands of global memory L1 loads during the field arithmetic. Helper functions and Montgomery operations are strictly `#pragma unroll`ed to maximize SIMD throughput, while keeping the main 86-iteration loop dynamic to avoid instruction cache exhaustion.

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_LIMBS = 6u;
constexpr constant uint POINT_LIMBS = 18u;

constant ulong LIMB_MASK_LO32 = 0x00000000FFFFFFFFul;

inline ulong2 umul128(ulong a, ulong b) {
    uint a0 = (uint)(a);
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)(b);
    uint b1 = (uint)(b >> 32);

    ulong p00 = (ulong)a0 * (ulong)b0;
    ulong p01 = (ulong)a0 * (ulong)b1;
    ulong p10 = (ulong)a1 * (ulong)b0;
    ulong p11 = (ulong)a1 * (ulong)b1;

    ulong mid = (p00 >> 32) + (p01 & LIMB_MASK_LO32) + (p10 & LIMB_MASK_LO32);
    ulong lo  = (p00 & LIMB_MASK_LO32) | (mid << 32);
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
    bool z = true;
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) {
        if (a[i] != 0ul) z = false;
    }
    return z;
}

inline bool eq_n(thread const ulong *a, thread const ulong *b) {
    bool e = true;
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) {
        if (a[i] != b[i]) e = false;
    }
    return e;
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

inline void mont_mul(thread ulong *out,
                     thread const ulong *a, thread const ulong *b,
                     thread const ulong *q, ulong q_inv_neg)
{
    ulong t[N_LIMBS + 2] = {0,0,0,0,0,0,0,0};

    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) {
        ulong C = 0ul;
        #pragma unroll
        for (uint j = 0u; j < N_LIMBS; ++j) {
            ulong2 r = fma_add128(a[j], b[i], t[j], C);
            t[j] = r.x;
            C = r.y;
        }
        ulong s = t[N_LIMBS] + C;
        ulong cy = (s < t[N_LIMBS]) ? 1ul : 0ul;
        t[N_LIMBS] = s;
        t[N_LIMBS + 1] += cy;

        ulong m = t[0] * q_inv_neg;

        C = 0ul;
        #pragma unroll
        for (uint j = 0u; j < N_LIMBS; ++j) {
            ulong2 r = fma_add128(m, q[j], t[j], C);
            t[j] = r.x;
            C = r.y;
        }
        s = t[N_LIMBS] + C;
        cy = (s < t[N_LIMBS]) ? 1ul : 0ul;
        t[N_LIMBS] = s;
        t[N_LIMBS + 1] += cy;

        #pragma unroll
        for (uint j = 0u; j < N_LIMBS + 1u; ++j) {
            t[j] = t[j + 1];
        }
        t[N_LIMBS + 1] = 0ul;
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
    bool use_diff = (t[N_LIMBS] != 0ul) || (borrow == 0ul);
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
    for (uint i = 0u; i < N_LIMBS; ++i) {
        X[i] = 0ul;
        Y[i] = 0ul;
        Z[i] = 0ul;
    }
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

inline int get_bit(ulong s0, ulong s1, ulong s2, ulong s3, int bit_idx) {
    ulong w = 0ul;
    w = (bit_idx >= 0   && bit_idx < 64)  ? s0 : w;
    w = (bit_idx >= 64  && bit_idx < 128) ? s1 : w;
    w = (bit_idx >= 128 && bit_idx < 192) ? s2 : w;
    w = (bit_idx >= 192 && bit_idx < 256) ? s3 : w;
    return (int)((w >> (bit_idx & 63)) & 1ul);
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

    ulong T1X[N_LIMBS], T1Y[N_LIMBS], T1Z[N_LIMBS];
    load_point(T1X, T1Y, T1Z, points_in + idx * POINT_LIMBS);

    ulong T2X[N_LIMBS], T2Y[N_LIMBS], T2Z[N_LIMBS];
    jac_double_pt(T2X, T2Y, T2Z, T1X, T1Y, T1Z, q_local, q_inv_neg);

    ulong T3X[N_LIMBS], T3Y[N_LIMBS], T3Z[N_LIMBS];
    jac_add_pt(T3X, T3Y, T3Z, T1X, T1Y, T1Z, T2X, T2Y, T2Z, q_local, q_inv_neg);

    ulong T4X[N_LIMBS], T4Y[N_LIMBS], T4Z[N_LIMBS];
    jac_double_pt(T4X, T4Y, T4Z, T2X, T2Y, T2Z, q_local, q_inv_neg);

    ulong AX[N_LIMBS], AY[N_LIMBS], AZ[N_LIMBS];
    zero_point(AX, AY, AZ);

    ulong TX[N_LIMBS], TY[N_LIMBS], TZ[N_LIMBS];

    for (int k = 85; k >= 0; --k) {
        if (k != 85) {
            jac_double_pt(TX, TY, TZ, AX, AY, AZ, q_local, q_inv_neg);
            copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);
            
            jac_double_pt(TX, TY, TZ, AX, AY, AZ, q_local, q_inv_neg);
            copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);
            
            jac_double_pt(TX, TY, TZ, AX, AY, AZ, q_local, q_inv_neg);
            copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);
        }

        int bit_idx = k * 3;
        int b_m1 = get_bit(s0, s1, s2, s3, bit_idx - 1);
        int b_0  = get_bit(s0, s1, s2, s3, bit_idx);
        int b_1  = get_bit(s0, s1, s2, s3, bit_idx + 1);
        int b_2  = get_bit(s0, s1, s2, s3, bit_idx + 2);

        int digit = -4 * b_2 + 2 * b_1 + b_0 + b_m1;

        if (digit != 0) {
            int abs_D = digit < 0 ? -digit : digit;
            ulong SX[N_LIMBS], SY[N_LIMBS], SZ[N_LIMBS];
            
            #pragma unroll
            for (uint i = 0u; i < N_LIMBS; ++i) {
                ulong x = 0, y = 0, z = 0;
                x = (abs_D == 1) ? T1X[i] : x;
                y = (abs_D == 1) ? T1Y[i] : y;
                z = (abs_D == 1) ? T1Z[i] : z;
                
                x = (abs_D == 2) ? T2X[i] : x;
                y = (abs_D == 2) ? T2Y[i] : y;
                z = (abs_D == 2) ? T2Z[i] : z;
                
                x = (abs_D == 3) ? T3X[i] : x;
                y = (abs_D == 3) ? T3Y[i] : y;
                z = (abs_D == 3) ? T3Z[i] : z;
                
                x = (abs_D == 4) ? T4X[i] : x;
                y = (abs_D == 4) ? T4Y[i] : y;
                z = (abs_D == 4) ? T4Z[i] : z;
                
                SX[i] = x; SY[i] = y; SZ[i] = z;
            }

            if (digit < 0) {
                ulong diff[N_LIMBS];
                ulong borrow = 0ul;
                #pragma unroll
                for (uint i = 0; i < N_LIMBS; ++i) {
                    ulong tv = q_local[i] - SY[i];
                    ulong b1 = (tv > q_local[i]) ? 1ul : 0ul;
                    ulong d = tv - borrow;
                    ulong b2 = (d > tv) ? 1ul : 0ul;
                    diff[i] = d;
                    borrow = b1 + b2;
                }
                #pragma unroll
                for (uint i = 0; i < N_LIMBS; ++i) SY[i] = diff[i];
            }

            jac_add_pt(TX, TY, TZ, AX, AY, AZ, SX, SY, SZ, q_local, q_inv_neg);
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