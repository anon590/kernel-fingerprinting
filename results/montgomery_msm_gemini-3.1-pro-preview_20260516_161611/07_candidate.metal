#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_LIMBS = 6u;
constexpr constant uint POINT_LIMBS = 18u;

inline ulong2 fma_add_carry_ret(ulong a, ulong b, ulong t, ulong c) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);
    
    ulong p00 = (ulong)a0 * b0;
    ulong p01 = (ulong)a0 * b1;
    ulong p10 = (ulong)a1 * b0;
    ulong p11 = (ulong)a1 * b1;
    
    ulong mid = (p00 >> 32) + (uint)p01 + (uint)p10;
    ulong hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    ulong lo = (ulong)(uint)p00 | (mid << 32);
    
    ulong lo1 = lo + t;
    hi += (lo1 < lo) ? 1ul : 0ul;
    ulong lo2 = lo1 + c;
    hi += (lo2 < lo1) ? 1ul : 0ul;
    
    return ulong2(lo2, hi);
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
    ulong t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, t7 = 0;
    ulong2 r;

    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) {
        ulong bi = b[i];
        ulong c = 0ul;
        
        r = fma_add_carry_ret(a[0], bi, t0, c); t0 = r.x; c = r.y;
        r = fma_add_carry_ret(a[1], bi, t1, c); t1 = r.x; c = r.y;
        r = fma_add_carry_ret(a[2], bi, t2, c); t2 = r.x; c = r.y;
        r = fma_add_carry_ret(a[3], bi, t3, c); t3 = r.x; c = r.y;
        r = fma_add_carry_ret(a[4], bi, t4, c); t4 = r.x; c = r.y;
        r = fma_add_carry_ret(a[5], bi, t5, c); t5 = r.x; c = r.y;
        
        ulong s = t6 + c;
        t7 += (s < t6) ? 1ul : 0ul;
        t6 = s;

        ulong m = t0 * q_inv_neg;
        c = 0ul;
        
        r = fma_add_carry_ret(m, q[0], t0, c); t0 = r.x; c = r.y;
        r = fma_add_carry_ret(m, q[1], t1, c); t1 = r.x; c = r.y;
        r = fma_add_carry_ret(m, q[2], t2, c); t2 = r.x; c = r.y;
        r = fma_add_carry_ret(m, q[3], t3, c); t3 = r.x; c = r.y;
        r = fma_add_carry_ret(m, q[4], t4, c); t4 = r.x; c = r.y;
        r = fma_add_carry_ret(m, q[5], t5, c); t5 = r.x; c = r.y;

        s = t6 + c;
        t7 += (s < t6) ? 1ul : 0ul;
        t6 = s;

        t0 = t1; t1 = t2; t2 = t3; t3 = t4; t4 = t5; t5 = t6; t6 = t7; t7 = 0ul;
    }

    ulong diff[N_LIMBS];
    ulong borrow = 0ul;
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) {
        ulong ti = (i==0?t0: i==1?t1: i==2?t2: i==3?t3: i==4?t4: t5);
        ulong tv = ti - q[i];
        ulong b1 = (tv > ti) ? 1ul : 0ul;
        ulong d = tv - borrow;
        ulong b2 = (d > tv) ? 1ul : 0ul;
        diff[i] = d;
        borrow = b1 + b2;
    }

    bool use_diff = (t6 != 0ul) || (borrow == 0ul);
    out[0] = use_diff ? diff[0] : t0;
    out[1] = use_diff ? diff[1] : t1;
    out[2] = use_diff ? diff[2] : t2;
    out[3] = use_diff ? diff[3] : t3;
    out[4] = use_diff ? diff[4] : t4;
    out[5] = use_diff ? diff[5] : t5;
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

// 100% In-place safe doubling kernel
inline void jac_double_pt(thread ulong *oX, thread ulong *oY, thread ulong *oZ,
                          thread const ulong *X, thread const ulong *Y, thread const ulong *Z,
                          thread const ulong *q, ulong q_inv_neg)
{
    if (is_zero_n(Z) || is_zero_n(Y)) {
        zero_point(oX, oY, oZ);
        return;
    }
    ulong T1[N_LIMBS], T2[N_LIMBS], T3[N_LIMBS];

    mont_mul(T1, Y, Z, q, q_inv_neg);
    mod_add(oZ, T1, T1, q); // oZ = 2YZ (safe to overwrite oZ as Z is no longer used)

    mont_mul(T1, X, X, q, q_inv_neg); // T1 = X^2 (A)
    mont_mul(T2, Y, Y, q, q_inv_neg); // T2 = Y^2 (B)
    mont_mul(T3, T2, T2, q, q_inv_neg); // T3 = B^2 (C)

    mod_add(oX, X, T2, q); 
    mont_mul(oY, oX, oX, q, q_inv_neg); // oY = (X+B)^2
    mod_sub(oY, oY, T1, q); // - A
    mod_sub(oY, oY, T3, q); // - C
    mod_add(T2, oY, oY, q); // T2 = S

    mod_add(oX, T1, T1, q);
    mod_add(T1, oX, T1, q); // T1 = M

    mont_mul(oX, T1, T1, q, q_inv_neg); // oX = M^2
    mod_add(oY, T2, T2, q); // 2S
    mod_sub(oX, oX, oY, q); // oX = X3

    mod_sub(oY, T2, oX, q); // S - X3
    mont_mul(oY, T1, oY, q, q_inv_neg); // M * (S - X3)
    
    mod_add(T3, T3, T3, q); // 2C
    mod_add(T3, T3, T3, q); // 4C
    mod_add(T3, T3, T3, q); // 8C
    mod_sub(oY, oY, T3, q); // oY = Y3
}

// 100% In-place safe add kernel
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
    ulong T1[N_LIMBS], T2[N_LIMBS], T3[N_LIMBS], T4[N_LIMBS];

    mont_mul(T1, Z1, Z1, q, q_inv_neg); // Z1Z1
    mont_mul(T2, Z2, Z2, q, q_inv_neg); // Z2Z2

    mont_mul(T3, X1, T2, q, q_inv_neg); // U1
    mont_mul(T4, X2, T1, q, q_inv_neg); // U2

    ulong S1[N_LIMBS], S2[N_LIMBS];
    mont_mul(S1, Y1, Z2, q, q_inv_neg);
    mont_mul(S1, S1, T2, q, q_inv_neg); // S1

    mont_mul(S2, Y2, Z1, q, q_inv_neg);
    mont_mul(S2, S2, T1, q, q_inv_neg); // S2

    if (eq_n(T3, T4)) {
        if (eq_n(S1, S2)) {
            jac_double_pt(oX, oY, oZ, X1, Y1, Z1, q, q_inv_neg);
        } else {
            zero_point(oX, oY, oZ);
        }
        return;
    }

    mod_sub(T4, T4, T3, q); // H is T4
    mod_sub(S2, S2, S1, q); // R is S2

    mont_mul(oZ, Z1, Z2, q, q_inv_neg);
    mont_mul(oZ, oZ, T4, q, q_inv_neg); // oZ = Z3 (Z1 and Z2 no longer needed)

    mont_mul(T2, T4, T4, q, q_inv_neg); // HH
    mont_mul(T1, T4, T2, q, q_inv_neg); // HHH
    mont_mul(T3, T3, T2, q, q_inv_neg); // V

    mont_mul(T2, S2, S2, q, q_inv_neg); // R^2
    mod_sub(T2, T2, T1, q); // R^2 - HHH
    mod_add(T4, T3, T3, q); // 2V
    mod_sub(T2, T2, T4, q); // X3 (stored in T2)

    mod_sub(T3, T3, T2, q); // V - X3
    mont_mul(T3, S2, T3, q, q_inv_neg); // R * (V - X3)
    mont_mul(T4, S1, T1, q, q_inv_neg); // S1 * HHH
    mod_sub(oY, T3, T4, q); // oY = Y3

    copy_n(oX, T2); // oX = X3
}

inline int get_bit(ulong s0, ulong s1, ulong s2, ulong s3, int bit_idx) {
    if (bit_idx < 0 || bit_idx >= 256) return 0;
    ulong w = 0ul;
    if (bit_idx < 64) w = s0;
    else if (bit_idx < 128) w = s1;
    else if (bit_idx < 192) w = s2;
    else w = s3;
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

    ulong T5X[N_LIMBS], T5Y[N_LIMBS], T5Z[N_LIMBS];
    jac_add_pt(T5X, T5Y, T5Z, T4X, T4Y, T4Z, T1X, T1Y, T1Z, q_local, q_inv_neg);

    ulong T6X[N_LIMBS], T6Y[N_LIMBS], T6Z[N_LIMBS];
    jac_double_pt(T6X, T6Y, T6Z, T3X, T3Y, T3Z, q_local, q_inv_neg);

    ulong T7X[N_LIMBS], T7Y[N_LIMBS], T7Z[N_LIMBS];
    jac_add_pt(T7X, T7Y, T7Z, T6X, T6Y, T6Z, T1X, T1Y, T1Z, q_local, q_inv_neg);

    ulong T8X[N_LIMBS], T8Y[N_LIMBS], T8Z[N_LIMBS];
    jac_double_pt(T8X, T8Y, T8Z, T4X, T4Y, T4Z, q_local, q_inv_neg);

    ulong AX[N_LIMBS], AY[N_LIMBS], AZ[N_LIMBS];
    zero_point(AX, AY, AZ);

    for (int k = 64; k >= 0; --k) {
        if (k != 64) {
            jac_double_pt(AX, AY, AZ, AX, AY, AZ, q_local, q_inv_neg);
            jac_double_pt(AX, AY, AZ, AX, AY, AZ, q_local, q_inv_neg);
            jac_double_pt(AX, AY, AZ, AX, AY, AZ, q_local, q_inv_neg);
            jac_double_pt(AX, AY, AZ, AX, AY, AZ, q_local, q_inv_neg);
        }

        int bit_idx = k * 4;
        int b_m1 = get_bit(s0, s1, s2, s3, bit_idx - 1);
        int b_0  = get_bit(s0, s1, s2, s3, bit_idx);
        int b_1  = get_bit(s0, s1, s2, s3, bit_idx + 1);
        int b_2  = get_bit(s0, s1, s2, s3, bit_idx + 2);
        int b_3  = get_bit(s0, s1, s2, s3, bit_idx + 3);

        int digit = -8 * b_3 + 4 * b_2 + 2 * b_1 + b_0 + b_m1;

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
                
                x = (abs_D == 5) ? T5X[i] : x;
                y = (abs_D == 5) ? T5Y[i] : y;
                z = (abs_D == 5) ? T5Z[i] : z;
                
                x = (abs_D == 6) ? T6X[i] : x;
                y = (abs_D == 6) ? T6Y[i] : y;
                z = (abs_D == 6) ? T6Z[i] : z;
                
                x = (abs_D == 7) ? T7X[i] : x;
                y = (abs_D == 7) ? T7Y[i] : y;
                z = (abs_D == 7) ? T7Z[i] : z;
                
                x = (abs_D == 8) ? T8X[i] : x;
                y = (abs_D == 8) ? T8Y[i] : y;
                z = (abs_D == 8) ? T8Z[i] : z;
                
                SX[i] = x; SY[i] = y; SZ[i] = z;
            }

            if (digit < 0) {
                ulong zero[N_LIMBS] = {0};
                mod_sub(SY, zero, SY, q_local);
            }

            jac_add_pt(AX, AY, AZ, AX, AY, AZ, SX, SY, SZ, q_local, q_inv_neg);
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

    jac_add_pt(AX, AY, AZ, AX, AY, AZ, BX, BY, BZ, q_local, q_inv_neg);
    store_point(scratch + idx * POINT_LIMBS, AX, AY, AZ);
}