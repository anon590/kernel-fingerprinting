An analysis of the previous implementation reveals three major bottlenecks causing the severe performance degradation at higher `n_pairs`:

1. **Memory Divergence:** Threads were conditionally executing `load_point`, causing uncoalesced and divergent 18-ulong global memory reads. Pulling this load outside the divergent `if` block allows the compiler to issue a uniform load, broadcasting the data directly from the L1 cache to all lanes.
2. **Bandwidth Over-fetching:** Each thread loaded a full 64-bit `ulong` to check 16-bit window values. Narrowing this to a 32-bit `uint` read halves the L1 cache footprint and memory bandwidth.
3. **Boundary SIMD Groups:** SIMD groups spanning two windows fell back to an $O(N)$ thread-local loop, independently scanning all 64K points. They are now refactored to use a 32-lane cooperative `simd_broadcast` loop, which brings them inline with the fast path. 

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
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) {
        if (a[i] != 0ul) return false;
    }
    return true;
}

inline bool eq_n(thread const ulong *a, thread const ulong *b) {
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
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
    uint idx [[thread_position_in_grid]],
    uint lane_id [[thread_index_in_simdgroup]])
{
    uint w = window_bits;
    uint num_buckets = (1u << w) - 1u;
    uint total = num_windows * num_buckets;
    uint mask = (1u << w) - 1u;

    ulong q_reg[N_LIMBS];
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) {
        q_reg[i] = q[i];
    }

    uint window_idx = idx / num_buckets;
    uint bucket_minus_1 = idx - window_idx * num_buckets;
    
    uint simd_start_idx = idx - lane_id;
    uint window_start = simd_start_idx / num_buckets;
    uint window_end = (simd_start_idx + 31u) / num_buckets;

    bool same_window = (window_start == window_end);

    ulong AX[N_LIMBS], AY[N_LIMBS], AZ[N_LIMBS];
    zero_point(AX, AY, AZ);

    if (same_window && window_start < num_windows) {
        uint bucket_start = simd_start_idx % num_buckets;
        uint bucket_start_plus_1 = bucket_start + 1u;
        
        uint shift = (window_start * w) & 63u;
        uint word_offset = (shift >= 32u) ? 1u : 0u;
        uint local_shift = shift & 31u;
        
        device const uint* scalars_uint = (device const uint*)scalars;

        uint chunk = 0;
        for (; chunk + 511u < n_pairs; chunk += 512u) {
            uint p0 = chunk + lane_id;
            uint my_mask = 0;
            
            #pragma unroll
            for (uint i = 0; i < 16; ++i) {
                uint S_val = scalars_uint[(p0 + i * 32u) * 8u + word_offset];
                uint w_curr = (S_val >> local_shift) & mask;
                if ((w_curr - bucket_start_plus_1) < 32u) {
                    my_mask |= (1u << i);
                }
            }

            uint active_j = simd_or(my_mask);

            while (active_j != 0u) {
                uint j = ctz(active_j);
                active_j &= active_j - 1u;
                
                uint m_j = simd_or((my_mask & (1u << j)) ? (1u << lane_id) : 0u);
                while (m_j != 0u) {
                    uint src_lane = ctz(m_j);
                    m_j &= m_j - 1u;
                    
                    uint S_src = scalars_uint[(chunk + j * 32u + src_lane) * 8u + word_offset];
                    uint w_src = (S_src >> local_shift) & mask;
                    uint target_lane = w_src - bucket_start_plus_1;
                    
                    ulong PX[N_LIMBS], PY[N_LIMBS], PZ[N_LIMBS];
                    load_point(PX, PY, PZ, points_in + (chunk + j * 32u + src_lane) * POINT_LIMBS);
                    
                    if (lane_id == target_lane) {
                        ulong TX[N_LIMBS], TY[N_LIMBS], TZ[N_LIMBS];
                        jac_add_pt(TX, TY, TZ, AX, AY, AZ, PX, PY, PZ, q_reg, q_inv_neg);
                        copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);
                    }
                }
            }
        }
        
        for (; chunk < n_pairs; chunk += 32u) {
            uint p = chunk + lane_id;
            uint S_val = (p < n_pairs) ? scalars_uint[p * 8u + word_offset] : 0u;
            uint w_curr = (S_val >> local_shift) & mask;
            bool m = (p < n_pairs) && ((w_curr - bucket_start_plus_1) < 32u);
            uint m_mask = simd_or(m ? (1u << lane_id) : 0u);
            
            while (m_mask != 0u) {
                uint src_lane = ctz(m_mask);
                m_mask &= m_mask - 1u;
                
                uint S_src = scalars_uint[(chunk + src_lane) * 8u + word_offset];
                uint w_src = (S_src >> local_shift) & mask;
                uint target_lane = w_src - bucket_start_plus_1;
                
                ulong PX[N_LIMBS], PY[N_LIMBS], PZ[N_LIMBS];
                load_point(PX, PY, PZ, points_in + (chunk + src_lane) * POINT_LIMBS);
                
                if (lane_id == target_lane) {
                    ulong TX[N_LIMBS], TY[N_LIMBS], TZ[N_LIMBS];
                    jac_add_pt(TX, TY, TZ, AX, AY, AZ, PX, PY, PZ, q_reg, q_inv_neg);
                    copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);
                }
            }
        }
    } else {
        uint shift = 0;
        uint target_w = 0xFFFFFFFFu;
        if (idx < total) {
            shift = (window_idx * w) & 63u;
            target_w = bucket_minus_1 + 1u;
        }
        
        uint word_offset = (shift >= 32u) ? 1u : 0u;
        uint local_shift = shift & 31u;
        
        device const uint* scalars_uint = (device const uint*)scalars;
        
        uint chunk = 0;
        for (; chunk < n_pairs; chunk += 32u) {
            uint p = chunk + lane_id;
            uint S_val = (p < n_pairs) ? scalars_uint[p * 8u + word_offset] : 0u;
            
            for (uint j = 0; j < 32u; ++j) {
                uint p_j = chunk + j;
                if (p_j >= n_pairs) break;
                
                uint S_j = simd_broadcast(S_val, j);
                uint w_val = (S_j >> local_shift) & mask;
                bool match = (w_val == target_w);
                
                uint match_mask = simd_or(match ? (1u << lane_id) : 0u);
                if (match_mask != 0u) {
                    ulong PX[N_LIMBS], PY[N_LIMBS], PZ[N_LIMBS];
                    load_point(PX, PY, PZ, points_in + p_j * POINT_LIMBS);
                    
                    if (match) {
                        ulong TX[N_LIMBS], TY[N_LIMBS], TZ[N_LIMBS];
                        jac_add_pt(TX, TY, TZ, AX, AY, AZ, PX, PY, PZ, q_reg, q_inv_neg);
                        copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);
                    }
                }
            }
        }
    }

    if (idx < total) {
        store_point(buckets + idx * POINT_LIMBS, AX, AY, AZ);
    }
}
```