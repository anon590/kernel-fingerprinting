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
    ulong v = a[0];
    #pragma unroll
    for (uint i = 1u; i < N_LIMBS; ++i) v |= a[i];
    return v == 0ul;
}

inline bool eq_n(thread const ulong *a, thread const ulong *b) {
    ulong v = a[0] ^ b[0];
    #pragma unroll
    for (uint i = 1u; i < N_LIMBS; ++i) v |= (a[i] ^ b[i]);
    return v == 0ul;
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
    ulong t[N_LIMBS + 2];
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS + 2u; ++i) t[i] = 0ul;

    for (uint i = 0u; i < N_LIMBS; ++i) {
        ulong C = 0ul;
        #pragma unroll
        for (uint j = 0u; j < N_LIMBS; ++j) {
            ulong2 r = fma_add128(a[j], b[i], t[j], C);
            t[j] = r.x;
            C = r.y;
        }
        {
            ulong s = t[N_LIMBS] + C;
            ulong cy = (s < t[N_LIMBS]) ? 1ul : 0ul;
            t[N_LIMBS] = s;
            t[N_LIMBS + 1] += cy;
        }
        ulong m = t[0] * q_inv_neg;
        C = 0ul;
        #pragma unroll
        for (uint j = 0u; j < N_LIMBS; ++j) {
            ulong2 r = fma_add128(m, q[j], t[j], C);
            t[j] = r.x;
            C = r.y;
        }
        {
            ulong s = t[N_LIMBS] + C;
            ulong cy = (s < t[N_LIMBS]) ? 1ul : 0ul;
            t[N_LIMBS] = s;
            t[N_LIMBS + 1] += cy;
        }
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
    uint w = window_bits;
    uint num_buckets = (1u << w) - 1u;
    uint total = num_windows * num_buckets;

    if (idx >= total) return;

    uint window_idx = idx / num_buckets;
    uint bucket_minus_1 = idx % num_buckets;
    uint target_w = bucket_minus_1 + 1u;

    ulong q_reg[N_LIMBS];
    #pragma unroll
    for (uint i = 0; i < N_LIMBS; ++i) {
        q_reg[i] = q[i];
    }

    ulong AX[N_LIMBS], AY[N_LIMBS], AZ[N_LIMBS];
    zero_point(AX, AY, AZ);

    uint shift = window_idx * w;
    uint word_idx = shift / 32u;
    uint bit_shift = shift % 32u;
    bool cross_boundary = (bit_shift + w > 32u);
    uint mask = (1u << w) - 1u;

    device const uint *scalars_uint = (device const uint *)scalars;
    
    // Size the chunks such that lambda = matches per thread per chunk ~= 0.5.
    // This perfectly limits the expected variance per thread to safely fit within 8.
    uint chunk_size = max((1u << w) >> 1, 32u);

    for (uint chunk_start = 0; chunk_start < n_pairs; chunk_start += chunk_size) {
        uint chunk_end = min(chunk_start + chunk_size, n_pairs);
        uint my_indices[8];
        uint my_count = 0;

        // Broadcast scalar pass: all active threads scan identically.
        for (uint i = chunk_start; i < chunk_end; ++i) {
            uint S_val = scalars_uint[i * 8u + word_idx];
            uint w_val = (S_val >> bit_shift);
            if (cross_boundary) {
                uint S_val_hi = scalars_uint[i * 8u + word_idx + 1u];
                w_val |= (S_val_hi << (32u - bit_shift));
            }
            w_val &= mask;
            
            if (w_val == target_w) {
                if (my_count < 8u) {
                    my_indices[my_count] = i;
                }
                my_count++;
            }
        }

        // Amortize actual point addition sequentially within each thread locally 
        // but completely in PARALLEL over the active warp threads.
        uint max_c = simd_max(min(my_count, 8u));
        for (uint k = 0; k < max_c; ++k) {
            if (k < my_count) {
                uint idx_pt = my_indices[k];
                ulong PX[N_LIMBS], PY[N_LIMBS], PZ[N_LIMBS];
                load_point(PX, PY, PZ, points_in + idx_pt * POINT_LIMBS);
                
                ulong TX[N_LIMBS], TY[N_LIMBS], TZ[N_LIMBS];
                jac_add_pt(TX, TY, TZ, AX, AY, AZ, PX, PY, PZ, q_reg, q_inv_neg);
                copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);
            }
        }

        // Handle mathematically infinitesimal chances of warp variance exceeding our buffer.
        if (simd_max(my_count) > 8u) {
            if (my_count > 8u) {
                uint found = 0;
                for (uint i = chunk_start; i < chunk_end; ++i) {
                    uint S_val = scalars_uint[i * 8u + word_idx];
                    uint w_val = (S_val >> bit_shift);
                    if (cross_boundary) {
                        uint S_val_hi = scalars_uint[i * 8u + word_idx + 1u];
                        w_val |= (S_val_hi << (32u - bit_shift));
                    }
                    w_val &= mask;
                    
                    if (w_val == target_w) {
                        if (found >= 8u) {
                            ulong PX[N_LIMBS], PY[N_LIMBS], PZ[N_LIMBS];
                            load_point(PX, PY, PZ, points_in + i * POINT_LIMBS);
                            
                            ulong TX[N_LIMBS], TY[N_LIMBS], TZ[N_LIMBS];
                            jac_add_pt(TX, TY, TZ, AX, AY, AZ, PX, PY, PZ, q_reg, q_inv_neg);
                            copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);
                        }
                        found++;
                    }
                }
            }
        }
    }

    store_point(buckets + idx * POINT_LIMBS, AX, AY, AZ);
}