#include <metal_stdlib>
using namespace metal;

inline ulong2 umul128(ulong a, ulong b) {
    ulong lo = a * b;
    ulong a0 = (uint)a, a1 = a >> 32;
    ulong b0 = (uint)b, b1 = b >> 32;
    ulong p01 = a0 * b1;
    ulong p10 = a1 * b0;
    ulong mid = (p01 & 0xFFFFFFFFul) + (p10 & 0xFFFFFFFFul) + ((a0 * b0) >> 32);
    ulong hi = (a1 * b1) + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return ulong2(lo, hi);
}

inline ulong2 fma_add128(ulong a, ulong b, ulong t, ulong c) {
    ulong2 prod = umul128(a, b);
    ulong lo1 = prod.x + t;
    ulong cy1 = (lo1 < prod.x) ? 1ul : 0ul;
    ulong lo2 = lo1 + c;
    ulong cy2 = (lo2 < lo1) ? 1ul : 0ul;
    return ulong2(lo2, prod.y + cy1 + cy2);
}

inline void copy_n(thread ulong *dst, thread const ulong *src) {
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) dst[i] = src[i];
}

inline bool is_zero_n(thread const ulong *a) {
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) {
        if (a[i] != 0ul) return false;
    }
    return true;
}

inline bool eq_n(thread const ulong *a, thread const ulong *b) {
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
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
        ulong cy1 = (s < a[i]) ? 1ul : 0ul;
        ulong t = s + b[i];
        ulong cy2 = (t < s) ? 1ul : 0ul;
        sum[i] = t;
        carry = cy1 + cy2;
    }
    ulong diff[6];
    ulong borrow = 0ul;
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) {
        ulong tv = sum[i] - q[i];
        ulong b1 = (tv > sum[i]) ? 1ul : 0ul;
        ulong d = tv - borrow;
        ulong b2 = (d > tv) ? 1ul : 0ul;
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
        ulong b1 = (tv > a[i]) ? 1ul : 0ul;
        ulong d = tv - borrow;
        ulong b2 = (d > tv) ? 1ul : 0ul;
        diff[i] = d;
        borrow = b1 + b2;
    }
    if (borrow != 0ul) {
        ulong carry = 0ul;
        #pragma unroll
        for (uint i = 0u; i < 6u; ++i) {
            ulong s = diff[i] + carry;
            ulong cy1 = (s < diff[i]) ? 1ul : 0ul;
            ulong t = s + q[i];
            ulong cy2 = (t < s) ? 1ul : 0ul;
            c[i] = t;
            carry = cy1 + cy2;
        }
    } else {
        #pragma unroll
        for (uint i = 0u; i < 6u; ++i) c[i] = diff[i];
    }
}

inline void mont_mul(thread ulong *out,
                     thread const ulong *a, thread const ulong *b,
                     thread const ulong *q, ulong q_inv_neg)
{
    ulong t[8];
    #pragma unroll
    for (uint i = 0u; i < 8u; ++i) t[i] = 0ul;

    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) {
        ulong C = 0ul;
        #pragma unroll
        for (uint j = 0u; j < 6u; ++j) {
            ulong2 r = fma_add128(a[j], b[i], t[j], C);
            t[j] = r.x;
            C = r.y;
        }
        {
            ulong s = t[6] + C;
            ulong cy = (s < t[6]) ? 1ul : 0ul;
            t[6] = s;
            t[7] += cy;
        }
        ulong m = t[0] * q_inv_neg;
        C = 0ul;
        #pragma unroll
        for (uint j = 0u; j < 6u; ++j) {
            ulong2 r = fma_add128(m, q[j], t[j], C);
            t[j] = r.x;
            C = r.y;
        }
        {
            ulong s = t[6] + C;
            ulong cy = (s < t[6]) ? 1ul : 0ul;
            t[6] = s;
            t[7] += cy;
        }
        #pragma unroll
        for (uint j = 0u; j < 7u; ++j) {
            t[j] = t[j + 1];
        }
        t[7] = 0ul;
    }

    ulong diff[6];
    ulong borrow = 0ul;
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) {
        ulong tv = t[i] - q[i];
        ulong b1 = (tv > t[i]) ? 1ul : 0ul;
        ulong d = tv - borrow;
        ulong b2 = (d > tv) ? 1ul : 0ul;
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
    for (uint i = 0u; i < 6u; ++i) {
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
    
    ulong T1[6], T2[6], T3[6];
    mont_mul(T1, X, X, q, q_inv_neg);
    mont_mul(T2, Y, Y, q, q_inv_neg);
    mont_mul(T3, T2, T2, q, q_inv_neg);
    
    ulong D[6];
    mod_add(D, X, T2, q);
    mont_mul(D, D, D, q, q_inv_neg);
    mod_sub(D, D, T1, q);
    mod_sub(D, D, T3, q);
    mod_add(D, D, D, q);
    
    ulong E[6];
    mod_add(E, T1, T1, q);
    mod_add(E, E, T1, q);
    
    ulong F[6];
    mont_mul(F, E, E, q, q_inv_neg);
    
    mod_add(T1, D, D, q);
    mod_sub(oX, F, T1, q);
    
    mod_sub(T2, D, oX, q);
    mont_mul(oY, E, T2, q, q_inv_neg);
    
    mod_add(T3, T3, T3, q);
    mod_add(T3, T3, T3, q);
    mod_add(T3, T3, T3, q);
    mod_sub(oY, oY, T3, q);
    
    mont_mul(oZ, Y, Z, q, q_inv_neg);
    mod_add(oZ, oZ, oZ, q);
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
    
    ulong T1[6], T2[6], T3[6], T4[6];
    
    mont_mul(T1, Z1, Z1, q, q_inv_neg);
    mont_mul(T2, Z2, Z2, q, q_inv_neg);
    
    ulong U1[6];
    mont_mul(U1, X1, T2, q, q_inv_neg);
    
    ulong U2[6];
    mont_mul(U2, X2, T1, q, q_inv_neg);
    
    ulong S1[6];
    mont_mul(S1, Y1, Z2, q, q_inv_neg);
    mont_mul(S1, S1, T2, q, q_inv_neg);
    
    ulong S2[6];
    mont_mul(S2, Y2, Z1, q, q_inv_neg);
    mont_mul(S2, S2, T1, q, q_inv_neg);
    
    if (eq_n(U1, U2)) {
        if (eq_n(S1, S2)) {
            jac_double_pt(oX, oY, oZ, X1, Y1, Z1, q, q_inv_neg);
            return;
        } else {
            zero_point(oX, oY, oZ);
            return;
        }
    }
    
    mod_sub(T1, U2, U1, q);
    mod_sub(T2, S2, S1, q);
    
    mont_mul(T3, T1, T1, q, q_inv_neg);
    mont_mul(T4, T1, T3, q, q_inv_neg);
    
    mont_mul(U2, U1, T3, q, q_inv_neg);
    
    mont_mul(oX, T2, T2, q, q_inv_neg);
    mod_sub(oX, oX, T4, q);
    mod_sub(oX, oX, U2, q);
    mod_sub(oX, oX, U2, q);
    
    mod_sub(U1, U2, oX, q);
    mont_mul(oY, T2, U1, q, q_inv_neg);
    mont_mul(S2, S1, T4, q, q_inv_neg);
    mod_sub(oY, oY, S2, q);
    
    mont_mul(oZ, Z1, Z2, q, q_inv_neg);
    mont_mul(oZ, oZ, T1, q, q_inv_neg);
}

kernel void pippenger_bucket_scatter(
    device const ulong *scalars     [[buffer(0)]],
    device const ulong *points_in   [[buffer(1)]],
    device       ulong *buckets     [[buffer(2)]],
    device const ulong *q           [[buffer(3)]],
    constant ulong     &q_inv_neg   [[buffer(4)]],
    constant uint      &n_pairs     [[buffer(5)]],
    constant uint      &num_windows [[buffer(6)]],
    constant uint      &window_bits [[buffer(7)]],
    uint idx [[thread_position_in_grid]])
{
    uint num_buckets = (1u << window_bits) - 1u;
    if (idx >= num_windows * num_buckets) return;

    uint window_idx = idx / num_buckets;
    uint bucket_minus_1 = idx % num_buckets;
    
    uint shift = (window_idx * window_bits) & 63u;
    ulong mask = (1ul << window_bits) - 1ul;

    ulong AX[6], AY[6], AZ[6];
    zero_point(AX, AY, AZ);

    ulong q_reg[6];
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) q_reg[i] = q[i];

    for (uint i = 0u; i < n_pairs; ++i) {
        ulong S = scalars[i * 4u];
        uint w_val = (uint)((S >> shift) & mask);
        
        if (w_val != 0u && (w_val - 1u) == bucket_minus_1) {
            ulong PX[6], PY[6], PZ[6];
            load_point(PX, PY, PZ, points_in + i * 18u);
            
            ulong TX[6], TY[6], TZ[6];
            jac_add_pt(TX, TY, TZ, AX, AY, AZ, PX, PY, PZ, q_reg, q_inv_neg);
            copy_n(AX, TX);
            copy_n(AY, TY);
            copy_n(AZ, TZ);
        }
    }

    store_point(buckets + idx * 18u, AX, AY, AZ);
}