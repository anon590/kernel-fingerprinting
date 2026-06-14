An elegant, highly optimized two-kernel pipeline for `fri_fold` and `fri_commit_level`.

### Optimizations Applied

1. **Unrolled & Factorized FRI Folding**: Branchless specialization of `fri_fold` for `fold == 2` and `fold == 4` eliminates loop overhead. The $S_m(j)$ geometric sum for `fold == 4` is factorized as $(1+r)(1+r^2)$ to save multiplications.
2. **Zero-Cost Bit Splitting**: Replaced costly 64-bit shifts and masks with `as_type<uint2>` bitcasts, allowing zero-overhead register reinterpretation on Apple Silicon.
3. **Multiplication-Free EPSILON Reduction**: Optimized $h_0 \times \text{EPSILON}$ to $(h_0 \ll 32) - h_0$, replacing a 64-bit multiplication with high-speed shift/subtract instructions.
4. **Specialized Squaring**: Added `gold_square` to save 1 multiplication per squaring, removing 92 multiplications per thread in the Poseidon permutation.
5. **Vectorized Constants & Coalesced Memory**: Loaded MDS/diagonal constants using `ulong4`/`ulong2` vector reads into registers. Paired leaf reads in the binary Merkle build are loaded via `ulong2` to maximize memory bandwidth.

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;   // 2^64 - 2^32 + 1
constant ulong EPSILON = 0x00000000FFFFFFFFul;   // 2^32 - 1

// ----------------------------------------------------------------------
// Optimized Goldilocks Arithmetic
// ----------------------------------------------------------------------

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    t = (t < a) ? (t + EPSILON) : t;
    return (t >= P_GOLD) ? (t - P_GOLD) : t;
}

inline ulong gold_add_one(ulong x) {
    ulong t = x + 1ul;
    return (t >= P_GOLD) ? (t - P_GOLD) : t;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    return (t > a) ? (t - EPSILON) : t;
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong lo = a * b;
    
    uint2 a_parts = as_type<uint2>(a);
    uint a0 = a_parts.x;
    uint a1 = a_parts.y;
    
    uint2 b_parts = as_type<uint2>(b);
    uint b0 = b_parts.x;
    uint b1 = b_parts.y;
    
    ulong p00 = (ulong)a0 * b0;
    ulong p01 = (ulong)a0 * b1;
    ulong p10 = (ulong)a1 * b0;
    ulong p11 = (ulong)a1 * b1;
    
    ulong sum_mid = p01 + p10;
    ulong carry_mid = (sum_mid < p01) ? 1ul : 0ul;
    ulong carry_to_64 = (lo < p00) ? 1ul : 0ul;
    
    ulong hi = p11 + (sum_mid >> 32) + (carry_mid << 32) + carry_to_64;
    
    ulong h0 = (uint)hi;
    ulong h1 = hi >> 32;
    
    ulong t0 = lo - h1;
    t0 = (t0 > lo) ? (t0 - EPSILON) : t0;
    
    ulong t1 = (h0 << 32) - h0;
    ulong t2 = t0 + t1;
    t2 = (t2 < t0) ? (t2 + EPSILON) : t2;
    
    return (t2 >= P_GOLD) ? (t2 - P_GOLD) : t2;
}

inline ulong gold_square(ulong x) {
    ulong lo = x * x;
    
    uint2 x_parts = as_type<uint2>(x);
    uint x0 = x_parts.x;
    uint x1 = x_parts.y;
    
    ulong p00 = (ulong)x0 * x0;
    ulong p01 = (ulong)x0 * x1;
    ulong p11 = (ulong)x1 * x1;
    
    ulong sum_mid = p01 << 1;
    ulong carry_mid = (sum_mid < p01) ? 1ul : 0ul;
    ulong carry_to_64 = (lo < p00) ? 1ul : 0ul;
    
    ulong hi = p11 + (sum_mid >> 32) + (carry_mid << 32) + carry_to_64;
    
    ulong h0 = (uint)hi;
    ulong h1 = hi >> 32;
    
    ulong t0 = lo - h1;
    t0 = (t0 > lo) ? (t0 - EPSILON) : t0;
    
    ulong t1 = (h0 << 32) - h0;
    ulong t2 = t0 + t1;
    t2 = (t2 < t0) ? (t2 + EPSILON) : t2;
    
    return (t2 >= P_GOLD) ? (t2 - P_GOLD) : t2;
}

// ----------------------------------------------------------------------
// FRI Fold Specializations
// ----------------------------------------------------------------------

kernel void fri_fold(
    device const ulong *evals_in     [[buffer(0)]],
    device       ulong *evals_out    [[buffer(1)]],
    device const ulong *inv_x_base   [[buffer(2)]],
    device const ulong *zeta_inv_pow [[buffer(3)]],
    constant ulong     &alpha        [[buffer(4)]],
    constant ulong     &inv_fold     [[buffer(5)]],
    constant uint      &fold         [[buffer(6)]],
    constant uint      &n_out        [[buffer(7)]],
    uint j [[thread_position_in_grid]])
{
    if (j >= n_out) return;

    ulong ax = gold_mul(alpha, inv_x_base[j]);

    if (fold == 2u) {
        ulong rm0 = gold_mul(ax, zeta_inv_pow[0]);
        ulong sm0 = gold_add_one(rm0);
        ulong term0 = gold_mul(evals_in[j], sm0);

        ulong rm1 = gold_mul(ax, zeta_inv_pow[1]);
        ulong sm1 = gold_add_one(rm1);
        ulong term1 = gold_mul(evals_in[j + n_out], sm1);

        ulong acc = gold_add(term0, term1);
        evals_out[j] = gold_mul(acc, inv_fold);
    } 
    else if (fold == 4u) {
        ulong rm0 = gold_mul(ax, zeta_inv_pow[0]);
        ulong rm0_2 = gold_square(rm0);
        ulong sm0 = gold_mul(gold_add_one(rm0), gold_add_one(rm0_2));
        ulong term0 = gold_mul(evals_in[j], sm0);

        ulong rm1 = gold_mul(ax, zeta_inv_pow[1]);
        ulong rm1_2 = gold_square(rm1);
        ulong sm1 = gold_mul(gold_add_one(rm1), gold_add_one(rm1_2));
        ulong term1 = gold_mul(evals_in[j + n_out], sm1);

        ulong rm2 = gold_mul(ax, zeta_inv_pow[2]);
        ulong rm2_2 = gold_square(rm2);
        ulong sm2 = gold_mul(gold_add_one(rm2), gold_add_one(rm2_2));
        ulong term2 = gold_mul(evals_in[j + 2u * n_out], sm2);

        ulong rm3 = gold_mul(ax, zeta_inv_pow[3]);
        ulong rm3_2 = gold_square(rm3);
        ulong sm3 = gold_mul(gold_add_one(rm3), gold_add_one(rm3_2));
        ulong term3 = gold_mul(evals_in[j + 3u * n_out], sm3);

        ulong acc = gold_add(gold_add(term0, term1), gold_add(term2, term3));
        evals_out[j] = gold_mul(acc, inv_fold);
    } 
    else {
        ulong acc = 0ul;
        for (uint m = 0u; m < fold; ++m) {
            ulong rm = gold_mul(ax, zeta_inv_pow[m]);
            ulong sm = 0ul;
            ulong rpow = 1ul;
            for (uint p = 0u; p < fold; ++p) {
                sm = gold_add(sm, rpow);
                rpow = gold_mul(rpow, rm);
            }
            acc = gold_add(acc, gold_mul(evals_in[j + m * n_out], sm));
        }
        evals_out[j] = gold_mul(acc, inv_fold);
    }
}

// ----------------------------------------------------------------------
// Poseidon2-t=3 Commitment
// ----------------------------------------------------------------------

inline ulong sbox7(ulong x) {
    ulong x2 = gold_square(x);
    ulong x4 = gold_square(x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
}

inline void poseidon2_permute_t3(thread ulong *state,
                                 device const ulong *rc_ext,
                                 device const ulong *rc_int,
                                 device const ulong *ext_mds,
                                 device const ulong *int_diag)
{
    device const ulong4 *ext_mds_u4 = (device const ulong4 *)ext_mds;
    ulong4 m_row0_1 = ext_mds_u4[0];
    ulong4 m_row1_2 = ext_mds_u4[1];
    ulong m22 = ext_mds[8];
    
    ulong m00 = m_row0_1.x, m01 = m_row0_1.y, m02 = m_row0_1.z;
    ulong m10 = m_row0_1.w, m11 = m_row1_2.x, m12 = m_row1_2.y;
    ulong m20 = m_row1_2.z, m21 = m_row1_2.w;

    device const ulong2 *int_diag_u2 = (device const ulong2 *)int_diag;
    ulong2 d_pair = int_diag_u2[0];
    ulong d0 = d_pair.x, d1 = d_pair.y, d2 = int_diag[2];

    auto matvec_ext = [&](thread ulong *st) {
        ulong s0 = st[0], s1 = st[1], s2 = st[2];
        st[0] = gold_add(gold_add(gold_mul(m00, s0), gold_mul(m01, s1)), gold_mul(m02, s2));
        st[1] = gold_add(gold_add(gold_mul(m10, s0), gold_mul(m11, s1)), gold_mul(m12, s2));
        st[2] = gold_add(gold_add(gold_mul(m20, s0), gold_mul(m21, s1)), gold_mul(m22, s2));
    };

    auto matvec_int = [&](thread ulong *st) {
        ulong s = gold_add(gold_add(st[0], st[1]), st[2]);
        st[0] = gold_add(s, gold_mul(d0, st[0]));
        st[1] = gold_add(s, gold_mul(d1, st[1]));
        st[2] = gold_add(s, gold_mul(d2, st[2]));
    };

    matvec_ext(state);

    // First half full rounds
    for (uint r = 0u; r < 4u; ++r) {
        state[0] = sbox7(gold_add(state[0], rc_ext[r * 3u + 0u]));
        state[1] = sbox7(gold_add(state[1], rc_ext[r * 3u + 1u]));
        state[2] = sbox7(gold_add(state[2], rc_ext[r * 3u + 2u]));
        matvec_ext(state);
    }

    // Partial rounds
    for (uint r = 0u; r < 22u; ++r) {
        state[0] = sbox7(gold_add(state[0], rc_int[r]));
        matvec_int(state);
    }

    // Second half full rounds
    for (uint r = 4u; r < 8u; ++r) {
        state[0] = sbox7(gold_add(state[0], rc_ext[r * 3u + 0u]));
        state[1] = sbox7(gold_add(state[1], rc_ext[r * 3u + 1u]));
        state[2] = sbox7(gold_add(state[2], rc_ext[r * 3u + 2u]));
        matvec_ext(state);
    }
}

kernel void fri_commit_level(
    device       ulong *tree         [[buffer(0)]],
    device const ulong *rc_ext       [[buffer(1)]],
    device const ulong *rc_int       [[buffer(2)]],
    device const ulong *ext_mds      [[buffer(3)]],
    device const ulong *int_diag     [[buffer(4)]],
    constant uint      &in_offset    [[buffer(5)]],
    constant uint      &out_offset   [[buffer(6)]],
    constant uint      &child_count  [[buffer(7)]],
    uint p [[thread_position_in_grid]])
{
    uint parent_count = (child_count + 1u) >> 1u;
    if (p >= parent_count) return;

    ulong s0 = 0ul;
    ulong s1 = 0ul;
    uint base = p << 1u;

    if (base + 1u < child_count) {
        device const ulong2 *ptr = (device const ulong2 *)(tree + in_offset + base);
        ulong2 pair = *ptr;
        s0 = pair.x;
        s1 = pair.y;
    } else {
        s0 = tree[in_offset + base];
    }

    thread ulong state[3];
    state[0] = s0;
    state[1] = s1;
    state[2] = 0ul;

    poseidon2_permute_t3(state, rc_ext, rc_int, ext_mds, int_diag);

    tree[out_offset + p] = state[0];
}
```