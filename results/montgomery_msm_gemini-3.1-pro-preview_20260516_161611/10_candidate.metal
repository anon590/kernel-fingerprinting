#include <metal_stdlib>
using namespace metal;

inline void copy_n(thread uint *dst, thread const uint *src) {
    #pragma unroll
    for (uint i = 0u; i < 12u; ++i) dst[i] = src[i];
}

inline bool is_zero_n(thread const uint *a) {
    bool z = true;
    #pragma unroll
    for (uint i = 0u; i < 12u; ++i) {
        if (a[i] != 0u) z = false;
    }
    return z;
}

inline bool eq_n(thread const uint *a, thread const uint *b) {
    bool e = true;
    #pragma unroll
    for (uint i = 0u; i < 12u; ++i) {
        if (a[i] != b[i]) e = false;
    }
    return e;
}

inline void mod_add(thread uint *c,
                    thread const uint *a, thread const uint *b,
                    thread const uint *q)
{
    uint sum[12];
    uint carry = 0u;
    #pragma unroll
    for (uint i = 0u; i < 12u; ++i) {
        ulong s = (ulong)a[i] + b[i] + carry;
        sum[i] = (uint)s;
        carry = (uint)(s >> 32);
    }
    uint diff[12];
    uint borrow = 0u;
    #pragma unroll
    for (uint i = 0u; i < 12u; ++i) {
        uint tv = sum[i] - q[i];
        uint b1 = (tv > sum[i]) ? 1u : 0u;
        uint d = tv - borrow;
        uint b2 = (d > tv) ? 1u : 0u;
        diff[i] = d;
        borrow = b1 + b2;
    }
    bool use_diff = (carry != 0u) || (borrow == 0u);
    #pragma unroll
    for (uint i = 0u; i < 12u; ++i) {
        c[i] = use_diff ? diff[i] : sum[i];
    }
}

inline void mod_sub(thread uint *c,
                    thread const uint *a, thread const uint *b,
                    thread const uint *q)
{
    uint diff[12];
    uint borrow = 0u;
    #pragma unroll
    for (uint i = 0u; i < 12u; ++i) {
        uint tv = a[i] - b[i];
        uint b1 = (tv > a[i]) ? 1u : 0u;
        uint d = tv - borrow;
        uint b2 = (d > tv) ? 1u : 0u;
        diff[i] = d;
        borrow = b1 + b2;
    }
    if (borrow != 0u) {
        uint carry = 0u;
        #pragma unroll
        for (uint i = 0u; i < 12u; ++i) {
            ulong s = (ulong)diff[i] + q[i] + carry;
            c[i] = (uint)s;
            carry = (uint)(s >> 32);
        }
    } else {
        #pragma unroll
        for (uint i = 0u; i < 12u; ++i) c[i] = diff[i];
    }
}

inline void mont_mul(thread uint *out,
                     thread const uint *a, thread const uint *b,
                     thread const uint *q, uint q_inv_neg)
{
    uint t[14] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    #pragma unroll
    for (uint i = 0u; i < 12u; ++i) {
        uint C = 0u;
        #pragma unroll
        for (uint j = 0u; j < 12u; ++j) {
            ulong p = (ulong)a[j] * b[i] + t[j] + C;
            t[j] = (uint)p;
            C = (uint)(p >> 32);
        }
        ulong s_a = (ulong)t[12] + C;
        t[12] = (uint)s_a;
        uint cy_a = (uint)(s_a >> 32);

        uint m = t[0] * q_inv_neg;

        C = 0u;
        #pragma unroll
        for (uint j = 0u; j < 12u; ++j) {
            ulong p = (ulong)m * q[j] + t[j] + C;
            t[j] = (uint)p;
            C = (uint)(p >> 32);
        }
        ulong s_b = (ulong)t[12] + C;
        t[12] = (uint)s_b;
        t[13] = cy_a + (uint)(s_b >> 32);

        #pragma unroll
        for (uint j = 0u; j < 13u; ++j) {
            t[j] = t[j + 1];
        }
        t[13] = 0u;
    }

    uint diff[12];
    uint borrow = 0u;
    #pragma unroll
    for (uint i = 0u; i < 12u; ++i) {
        uint tv = t[i] - q[i];
        uint b1 = (tv > t[i]) ? 1u : 0u;
        uint d = tv - borrow;
        uint b2 = (d > tv) ? 1u : 0u;
        diff[i] = d;
        borrow = b1 + b2;
    }
    bool use_diff = (t[12] != 0u) || (borrow == 0u);
    #pragma unroll
    for (uint i = 0u; i < 12u; ++i) {
        out[i] = use_diff ? diff[i] : t[i];
    }
}

inline void load_point(thread uint *X, thread uint *Y, thread uint *Z,
                       device const ulong *src)
{
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) {
        ulong v = src[i];
        X[2*i] = (uint)v;
        X[2*i+1] = (uint)(v >> 32);
        
        v = src[6u + i];
        Y[2*i] = (uint)v;
        Y[2*i+1] = (uint)(v >> 32);
        
        v = src[12u + i];
        Z[2*i] = (uint)v;
        Z[2*i+1] = (uint)(v >> 32);
    }
}

inline void store_point(device ulong *dst,
                        thread const uint *X, thread const uint *Y, thread const uint *Z)
{
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) {
        dst[i]       = ((ulong)X[2*i+1] << 32) | X[2*i];
        dst[6u + i]  = ((ulong)Y[2*i+1] << 32) | Y[2*i];
        dst[12u + i] = ((ulong)Z[2*i+1] << 32) | Z[2*i];
    }
}

inline void zero_point(thread uint *X, thread uint *Y, thread uint *Z) {
    #pragma unroll
    for (uint i = 0u; i < 12u; ++i) {
        X[i] = 0u;
        Y[i] = 0u;
        Z[i] = 0u;
    }
}

inline void jac_double_pt(thread uint *oX, thread uint *oY, thread uint *oZ,
                          thread const uint *X, thread const uint *Y, thread const uint *Z,
                          thread const uint *q, uint q_inv_neg)
{
    if (is_zero_n(Z) || is_zero_n(Y)) {
        zero_point(oX, oY, oZ);
        return;
    }
    uint A[12], B[12], C[12];
    uint D[12], E[12], F[12];
    uint tmp[12], tmp2[12];

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

inline void jac_add_pt(thread uint *oX, thread uint *oY, thread uint *oZ,
                       thread const uint *X1, thread const uint *Y1, thread const uint *Z1,
                       thread const uint *X2, thread const uint *Y2, thread const uint *Z2,
                       thread const uint *q, uint q_inv_neg)
{
    if (is_zero_n(Z1)) {
        copy_n(oX, X2); copy_n(oY, Y2); copy_n(oZ, Z2);
        return;
    }
    if (is_zero_n(Z2)) {
        copy_n(oX, X1); copy_n(oY, Y1); copy_n(oZ, Z1);
        return;
    }
    uint Z1Z1[12], Z2Z2[12];
    uint U1[12], U2[12], S1[12], S2[12];
    uint H[12], R[12];
    uint HH[12], HHH[12], V[12];
    uint tmp[12], tmp2[12];

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
    if (bit_idx < 0 || bit_idx >= 256) return 0;
    ulong w = 0ul;
    w = (bit_idx < 64)  ? s0 : w;
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

    uint q_local[12];
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) {
        ulong v = q[i];
        q_local[2*i]   = (uint)v;
        q_local[2*i+1] = (uint)(v >> 32);
    }
    uint q_inv_neg32 = (uint)q_inv_neg;

    ulong s0 = scalars[idx * 4u + 0u];
    ulong s1 = scalars[idx * 4u + 1u];
    ulong s2 = scalars[idx * 4u + 2u];
    ulong s3 = scalars[idx * 4u + 3u];

    uint T1X[12], T1Y[12], T1Z[12];
    load_point(T1X, T1Y, T1Z, points_in + idx * 18u);

    uint T2X[12], T2Y[12], T2Z[12];
    jac_double_pt(T2X, T2Y, T2Z, T1X, T1Y, T1Z, q_local, q_inv_neg32);

    uint T3X[12], T3Y[12], T3Z[12];
    jac_add_pt(T3X, T3Y, T3Z, T2X, T2Y, T2Z, T1X, T1Y, T1Z, q_local, q_inv_neg32);

    uint TX_temp[12], TY_temp[12], TZ_temp[12];
    jac_double_pt(TX_temp, TY_temp, TZ_temp, T2X, T2Y, T2Z, q_local, q_inv_neg32);

    uint T5X[12], T5Y[12], T5Z[12];
    jac_add_pt(T5X, T5Y, T5Z, TX_temp, TY_temp, TZ_temp, T1X, T1Y, T1Z, q_local, q_inv_neg32);

    jac_double_pt(TX_temp, TY_temp, TZ_temp, T3X, T3Y, T3Z, q_local, q_inv_neg32);

    uint T7X[12], T7Y[12], T7Z[12];
    jac_add_pt(T7X, T7Y, T7Z, TX_temp, TY_temp, TZ_temp, T1X, T1Y, T1Z, q_local, q_inv_neg32);

    uint AX[12], AY[12], AZ[12];
    zero_point(AX, AY, AZ);

    uint TX[12], TY[12], TZ[12];

    for (int k = 64; k >= 0; --k) {
        int bit_idx = k * 4;
        int b_m1 = get_bit(s0, s1, s2, s3, bit_idx - 1);
        int b_0  = get_bit(s0, s1, s2, s3, bit_idx);
        int b_1  = get_bit(s0, s1, s2, s3, bit_idx + 1);
        int b_2  = get_bit(s0, s1, s2, s3, bit_idx + 2);
        int b_3  = get_bit(s0, s1, s2, s3, bit_idx + 3);

        int digit = b_m1 + b_0 + 2 * b_1 + 4 * b_2 - 8 * b_3;

        int abs_D = digit < 0 ? -digit : digit;
        int val = 0;
        int c = 0;
        if (abs_D != 0) {
            int tz = __builtin_ctz(abs_D);
            val = abs_D >> tz;
            c = tz;
        }

        uint SX[12], SY[12], SZ[12];
        if (abs_D != 0) {
            #pragma unroll
            for (uint i = 0u; i < 12u; ++i) {
                uint x = 0, y = 0, z = 0;
                x = (val == 1) ? T1X[i] : x;
                y = (val == 1) ? T1Y[i] : y;
                z = (val == 1) ? T1Z[i] : z;
                
                x = (val == 3) ? T3X[i] : x;
                y = (val == 3) ? T3Y[i] : y;
                z = (val == 3) ? T3Z[i] : z;
                
                x = (val == 5) ? T5X[i] : x;
                y = (val == 5) ? T5Y[i] : y;
                z = (val == 5) ? T5Z[i] : z;
                
                x = (val == 7) ? T7X[i] : x;
                y = (val == 7) ? T7Y[i] : y;
                z = (val == 7) ? T7Z[i] : z;
                
                SX[i] = x; SY[i] = y; SZ[i] = z;
            }

            if (digit < 0) {
                uint diff[12];
                uint borrow = 0u;
                #pragma unroll
                for (uint i = 0u; i < 12u; ++i) {
                    uint tv = q_local[i] - SY[i];
                    uint b1 = (tv > q_local[i]) ? 1u : 0u;
                    uint d = tv - borrow;
                    uint b2 = (d > tv) ? 1u : 0u;
                    diff[i] = d;
                    borrow = b1 + b2;
                }
                bool is_y_zero = true;
                #pragma unroll
                for (uint i = 0u; i < 12u; ++i) {
                    if (SY[i] != 0) is_y_zero = false;
                }
                if (!is_y_zero) {
                    #pragma unroll
                    for (uint i = 0u; i < 12u; ++i) SY[i] = diff[i];
                }
            }
        }

        if (k != 64) {
            jac_double_pt(TX, TY, TZ, AX, AY, AZ, q_local, q_inv_neg32);
            if (abs_D != 0 && c == 3) {
                jac_add_pt(AX, AY, AZ, TX, TY, TZ, SX, SY, SZ, q_local, q_inv_neg32);
            } else {
                copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);
            }

            jac_double_pt(TX, TY, TZ, AX, AY, AZ, q_local, q_inv_neg32);
            if (abs_D != 0 && c == 2) {
                jac_add_pt(AX, AY, AZ, TX, TY, TZ, SX, SY, SZ, q_local, q_inv_neg32);
            } else {
                copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);
            }

            jac_double_pt(TX, TY, TZ, AX, AY, AZ, q_local, q_inv_neg32);
            if (abs_D != 0 && c == 1) {
                jac_add_pt(AX, AY, AZ, TX, TY, TZ, SX, SY, SZ, q_local, q_inv_neg32);
            } else {
                copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);
            }

            jac_double_pt(TX, TY, TZ, AX, AY, AZ, q_local, q_inv_neg32);
            if (abs_D != 0 && c == 0) {
                jac_add_pt(AX, AY, AZ, TX, TY, TZ, SX, SY, SZ, q_local, q_inv_neg32);
            } else {
                copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);
            }
        } else {
            if (abs_D != 0) {
                jac_add_pt(TX, TY, TZ, AX, AY, AZ, SX, SY, SZ, q_local, q_inv_neg32);
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

    uint q_local[12];
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) {
        ulong v = q[i];
        q_local[2*i]   = (uint)v;
        q_local[2*i+1] = (uint)(v >> 32);
    }
    uint q_inv_neg32 = (uint)q_inv_neg;

    uint AX[12], AY[12], AZ[12];
    uint BX[12], BY[12], BZ[12];
    load_point(AX, AY, AZ, scratch + idx * 18u);
    load_point(BX, BY, BZ, scratch + (idx + half_count) * 18u);

    uint RX[12], RY[12], RZ[12];
    jac_add_pt(RX, RY, RZ,
               AX, AY, AZ,
               BX, BY, BZ,
               q_local, q_inv_neg32);
    store_point(scratch + idx * 18u, RX, RY, RZ);
}