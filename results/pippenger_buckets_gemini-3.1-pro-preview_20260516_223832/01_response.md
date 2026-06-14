To optimize Pippenger's bucket-scatter, we can aggressively minimize both memory bandwidth and execution divergence by leveraging **SIMD-level cooperation**. 

1. **Coalesced Tile Loading**: The baseline code has every thread load all $N$ pairs independently, resulting in massive L1 cache thrashing. We partition the pairs into 32-element chunks. Within each chunk, each lane in the 32-thread SIMD group performs exactly one load, fetching all 32 pairs perfectly coalesced. These pairs are then distributed via `simd_broadcast`, cutting global scalar memory reads by a factor of 32.
2. **SIMD Divergence Bypass**: Bucket additions are extremely rare per thread (~1 in 65,535). Instead of each thread independently evaluating and branching, we compute matches for 8 pairs at a time and use `simd_any` to skip the expensive `jac_add_pt` block across the entire SIMD group if no lane matches. This bypasses point loading and arithmetic for 99.8% of the loop iterations.
3. **Register Caching**: The base-field modulus `q` is loaded from `device` to thread-private registers once, avoiding repeated memory lookups during Montgomery reductions.

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
                    thread const ulong *a, thread const ulong *b,
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
                    thread const ulong *a, thread const ulong *b,
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
    if (borrow != 0ul) {
        ulong carry = 0ul;
        for (uint i = 0u; i < N_LIMBS; ++i) {
            ulong s = diff[i] + carry;
            ulong cy1 = (s < diff[i]) ? 1ul : 0ul;
            ulong t = s + q[i];
            ulong cy2 = (t < s) ? 1ul : 0ul;
            c[i] = t;
            carry = cy1 + cy2;
        }
    } else {
        for (uint i = 0u; i < N_LIMBS; ++i) c[i] = diff[i];
    }
}

inline void mont_mul(thread ulong *out,
                     thread const ulong *a, thread const ulong *b,
                     thread const ulong *q, ulong q_inv_neg)
{
    ulong t[N_LIMBS + 2];
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
            t[N_LIMBS + 1] += cy;
        }
        ulong m = t[0] * q_inv_neg;
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
            t[N_LIMBS + 1] += cy;
        }
        for (uint j = 0u; j < N_LIMBS + 1u; ++j) {
            t[j] = t[j + 1];
        }
        t[N_LIMBS + 1] = 0ul;
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

inline void load_point(thread ulong *X, thread ulong *Y, thread ulong *Z,
                       device const ulong *src)
{
    for (uint i = 0u; i < N_LIMBS; ++i) X[i] = src[i];
    for (uint i = 0u; i < N_LIMBS; ++i) Y[i] = src[N_LIMBS + i];
    for (uint i = 0u; i < N_LIMBS; ++i) Z[i] = src[2u * N_LIMBS + i];
}

inline void store_point(device ulong *dst,
                        thread const ulong *X, thread const ulong *Y, thread const ulong *Z)
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
    if (idx >= total) return;

    // Localize `q` into registers for repeated fast access
    ulong q_reg[N_LIMBS];
    for (uint i = 0u; i < N_LIMBS; ++i) {
        q_reg[i] = q[i];
    }

    uint window_idx = idx / num_buckets;
    uint bucket_minus_1 = idx - window_idx * num_buckets;
    uint bucket_val = bucket_minus_1 + 1u;
    uint bit_offset = window_idx * w;
    ulong mask = (1ul << w) - 1ul;

    ulong AX[N_LIMBS], AY[N_LIMBS], AZ[N_LIMBS];
    zero_point(AX, AY, AZ);

    // SIMD Tiling: Distribute sequential reads across 32 lanes
    for (uint chunk = 0; chunk < n_pairs; chunk += 32u) {
        uint pair_idx = chunk + lane_id;
        ulong my_s = 0ul;
        
        if (pair_idx < n_pairs) {
            my_s = scalars[pair_idx * 4u];
        }

        uint chunk_limit = min(32u, n_pairs - chunk);
        uint chunk_limit_8 = chunk_limit & ~7u;
        
        // Broadcast evaluated pairs down to every thread
        for (uint j = 0; j < chunk_limit_8; j += 8u) {
            ulong s0 = simd_broadcast(my_s, (ushort)(j + 0u));
            ulong s1 = simd_broadcast(my_s, (ushort)(j + 1u));
            ulong s2 = simd_broadcast(my_s, (ushort)(j + 2u));
            ulong s3 = simd_broadcast(my_s, (ushort)(j + 3u));
            ulong s4 = simd_broadcast(my_s, (ushort)(j + 4u));
            ulong s5 = simd_broadcast(my_s, (ushort)(j + 5u));
            ulong s6 = simd_broadcast(my_s, (ushort)(j + 6u));
            ulong s7 = simd_broadcast(my_s, (ushort)(j + 7u));

            uint b0 = (uint)((s0 >> bit_offset) & mask);
            uint b1 = (uint)((s1 >> bit_offset) & mask);
            uint b2 = (uint)((s2 >> bit_offset) & mask);
            uint b3 = (uint)((s3 >> bit_offset) & mask);
            uint b4 = (uint)((s4 >> bit_offset) & mask);
            uint b5 = (uint)((s5 >> bit_offset) & mask);
            uint b6 = (uint)((s6 >> bit_offset) & mask);
            uint b7 = (uint)((s7 >> bit_offset) & mask);

            bool m0 = (b0 == bucket_val);
            bool m1 = (b1 == bucket_val);
            bool m2 = (b2 == bucket_val);
            bool m3 = (b3 == bucket_val);
            bool m4 = (b4 == bucket_val);
            bool m5 = (b5 == bucket_val);
            bool m6 = (b6 == bucket_val);
            bool m7 = (b7 == bucket_val);

            // Vast majority of points bypass evaluation here (~99.8%)
            bool any_m = m0 | m1 | m2 | m3 | m4 | m5 | m6 | m7;
            if (simd_any(any_m)) {
                bool m_arr[8] = {m0, m1, m2, m3, m4, m5, m6, m7};
                
                #pragma unroll
                for (uint k = 0; k < 8; ++k) {
                    bool m = m_arr[k];
                    if (simd_any(m)) {
                        if (m) {
                            ulong PX[N_LIMBS], PY[N_LIMBS], PZ[N_LIMBS];
                            load_point(PX, PY, PZ, points_in + (chunk + j + k) * POINT_LIMBS);
                            ulong TX[N_LIMBS], TY[N_LIMBS], TZ[N_LIMBS];
                            jac_add_pt(TX, TY, TZ, AX, AY, AZ, PX, PY, PZ, q_reg, q_inv_neg);
                            copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);
                        }
                    }
                }
            }
        }
        
        // Remainder pair loop processing block
        for (uint j = chunk_limit_8; j < chunk_limit; ++j) {
            ulong s0 = simd_broadcast(my_s, (ushort)j);
            uint b = (uint)((s0 >> bit_offset) & mask);
            bool m = (b == bucket_val);
            if (simd_any(m)) {
                if (m) {
                    ulong PX[N_LIMBS], PY[N_LIMBS], PZ[N_LIMBS];
                    load_point(PX, PY, PZ, points_in + (chunk + j) * POINT_LIMBS);
                    ulong TX[N_LIMBS], TY[N_LIMBS], TZ[N_LIMBS];
                    jac_add_pt(TX, TY, TZ, AX, AY, AZ, PX, PY, PZ, q_reg, q_inv_neg);
                    copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);
                }
            }
        }
    }

    uint bucket_idx = window_idx * num_buckets + bucket_minus_1;
    store_point(buckets + bucket_idx * POINT_LIMBS, AX, AY, AZ);
}
```