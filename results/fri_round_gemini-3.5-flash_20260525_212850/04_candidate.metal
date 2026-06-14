#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;   // 2^64 - 2^32 + 1
constant ulong EPSILON = 0x00000000FFFFFFFFul;   // 2^32 - 1

constexpr constant uint T_FIXED  = 3u;           // Poseidon2 width

// ----------------------------------------------------------------------
// Branchless Goldilocks Arithmetic
// ----------------------------------------------------------------------

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    t += (t < a) ? EPSILON : 0ul;
    t -= (t >= P_GOLD) ? P_GOLD : 0ul;
    return t;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    t -= (t > a) ? EPSILON : 0ul;
    return t;
}

inline ulong2 umul128(ulong a, ulong b) {
    uint2 a_32 = as_type<uint2>(a);
    uint2 b_32 = as_type<uint2>(b);

    ulong p00 = (ulong)a_32.x * b_32.x;
    ulong p01 = (ulong)a_32.x * b_32.y;
    ulong p10 = (ulong)a_32.y * b_32.x;
    ulong p11 = (ulong)a_32.y * b_32.y;

    ulong mid = (p00 >> 32) + (uint)p01 + (uint)p10;
    ulong lo  = (uint)p00 | (mid << 32);
    ulong hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return ulong2(lo, hi);
}

inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong x_hi_lo = x_hi & EPSILON;
    ulong x_hi_hi = x_hi >> 32;

    ulong t0 = x_lo - x_hi_hi;
    t0 -= (t0 > x_lo) ? EPSILON : 0ul;

    ulong t1 = (x_hi_lo << 32) - x_hi_lo;

    ulong t2 = t0 + t1;
    t2 += (t2 < t0) ? EPSILON : 0ul;

    t2 -= (t2 >= P_GOLD) ? P_GOLD : 0ul;
    return t2;
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong2 prod = umul128(a, b);
    return gold_reduce128(prod.x, prod.y);
}

// 3-cycle height-optimized sbox7 (parallelized dependency paths)
inline ulong sbox7(ulong x) {
    ulong x2 = gold_mul(x, x);
    ulong x4 = gold_mul(x2, x2);
    ulong x3 = gold_mul(x2, x);
    return gold_mul(x4, x3);
}

// ----------------------------------------------------------------------
// FRI fold (Optimized with Grouped Algebra)
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
        ulong E0 = evals_in[j];
        ulong E1 = evals_in[j + n_out];

        ulong E0_plus_E1  = gold_add(E0, E1);
        ulong E0_minus_E1 = gold_sub(E0, E1);

        ulong acc = gold_add(E0_plus_E1, gold_mul(ax, E0_minus_E1));
        evals_out[j] = gold_mul(acc, inv_fold);
    } else if (fold == 4u) {
        ulong E0 = evals_in[j];
        ulong E1 = evals_in[j + n_out];
        ulong E2 = evals_in[j + 2u * n_out];
        ulong E3 = evals_in[j + 3u * n_out];

        ulong z1 = zeta_inv_pow[1];

        ulong rm1 = gold_mul(ax, z1);
        ulong ax_sq = gold_mul(ax, ax);

        ulong ax_sq_plus1     = gold_add(1ul, ax_sq);
        ulong ax_sq_neg_plus1 = gold_sub(1ul, ax_sq);

        ulong E0_plus_E2  = gold_add(E0, E2);
        ulong E0_minus_E2 = gold_sub(E0, E2);
        ulong E1_plus_E3  = gold_add(E1, E3);
        ulong E1_minus_E3 = gold_sub(E1, E3);

        ulong part0 = gold_add(E0_plus_E2, gold_mul(ax, E0_minus_E2));
        ulong part1 = gold_add(E1_plus_E3, gold_mul(rm1, E1_minus_E3));

        ulong term0_2 = gold_mul(part0, ax_sq_plus1);
        ulong term1_3 = gold_mul(part1, ax_sq_neg_plus1);

        ulong acc = gold_add(term0_2, term1_3);
        evals_out[j] = gold_mul(acc, inv_fold);
    } else {
        ulong acc = 0ul;
        for (uint m = 0u; m < fold; ++m) {
            ulong rm = gold_mul(ax, zeta_inv_pow[m]);
            ulong sm = 0ul;
            ulong rpow = 1ul;
            for (uint p = 0u; p < fold; ++p) {
                sm   = gold_add(sm, rpow);
                rpow = gold_mul(rpow, rm);
            }
            acc = gold_add(acc, gold_mul(evals_in[j + m * n_out], sm));
        }
        evals_out[j] = gold_mul(acc, inv_fold);
    }
}

// ----------------------------------------------------------------------
// Poseidon2-t=3 Merkle Commit
// ----------------------------------------------------------------------

#define MATVEC_EXT(s0, s1, s2) \
{ \
    ulong t0 = gold_add(gold_add(gold_mul(m0, s0), gold_mul(m1, s1)), gold_mul(m2, s2)); \
    ulong t1 = gold_add(gold_add(gold_mul(m3, s0), gold_mul(m4, s1)), gold_mul(m5, s2)); \
    ulong t2 = gold_add(gold_add(gold_mul(m6, s0), gold_mul(m7, s1)), gold_mul(m8, s2)); \
    s0 = t0; s1 = t1; s2 = t2; \
}

#define MATVEC_INT(s0, s1, s2) \
{ \
    ulong sum_s = gold_add(gold_add(s0, s1), s2); \
    ulong t0 = gold_add(sum_s, gold_mul(d0, s0)); \
    ulong t1 = gold_add(sum_s, gold_mul(d1, s1)); \
    ulong t2 = gold_add(sum_s, gold_mul(d2, s2)); \
    s0 = t0; s1 = t1; s2 = t2; \
}

inline void poseidon2_permute_t3_direct(
    thread ulong &s0, thread ulong &s1, thread ulong &s2,
    device const ulong *rc_ext,
    device const ulong *rc_int,
    device const ulong *ext_mds,
    device const ulong *int_diag)
{
    // Load MDS matrix and diagonal elements to thread registers
    ulong m0 = ext_mds[0], m1 = ext_mds[1], m2 = ext_mds[2];
    ulong m3 = ext_mds[3], m4 = ext_mds[4], m5 = ext_mds[5];
    ulong m6 = ext_mds[6], m7 = ext_mds[7], m8 = ext_mds[8];

    ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2];

    MATVEC_EXT(s0, s1, s2);

    // First half full rounds (4 rounds)
    #pragma unroll
    for (uint r = 0u; r < 4u; ++r) {
        s0 = sbox7(gold_add(s0, rc_ext[r * 3u + 0u]));
        s1 = sbox7(gold_add(s1, rc_ext[r * 3u + 1u]));
        s2 = sbox7(gold_add(s2, rc_ext[r * 3u + 2u]));
        MATVEC_EXT(s0, s1, s2);
    }

    // Partial rounds (22 rounds)
    #pragma unroll
    for (uint r = 0u; r < 22u; ++r) {
        s0 = sbox7(gold_add(s0, rc_int[r]));
        MATVEC_INT(s0, s1, s2);
    }

    // Second half full rounds (4 rounds)
    #pragma unroll
    for (uint r = 4u; r < 8u; ++r) {
        s0 = sbox7(gold_add(s0, rc_ext[r * 3u + 0u]));
        s1 = sbox7(gold_add(s1, rc_ext[r * 3u + 1u]));
        s2 = sbox7(gold_add(s2, rc_ext[r * 3u + 2u]));
        MATVEC_EXT(s0, s1, s2);
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
    ulong s2 = 0ul;

    uint base = p << 1u;
    if (base + 1u < child_count) {
        // Coalesced 16-byte load
        ulong2 loaded = ((device const ulong2*)(tree + in_offset))[p];
        s0 = loaded.x;
        s1 = loaded.y;
    } else {
        s0 = tree[in_offset + base];
    }

    poseidon2_permute_t3_direct(s0, s1, s2, rc_ext, rc_int, ext_mds, int_diag);

    tree[out_offset + p] = s0;
}