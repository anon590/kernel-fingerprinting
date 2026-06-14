#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;   // 2^64 - 2^32 + 1
constant ulong EPSILON = 0x00000000FFFFFFFFul;   // 2^32 - 1

constexpr constant uint T_FIXED  = 3u;           // Poseidon2 width
constexpr constant uint POS2_R_F = 8u;           // Poseidon2 full rounds (4+4)
constexpr constant uint POS2_R_P = 22u;          // Poseidon2 partial rounds

// ----------------------------------------------------------------------
// Branchless Goldilocks Arithmetic
// ----------------------------------------------------------------------

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    t += (t < a) ? EPSILON : 0ul;
    return (t >= P_GOLD) ? (t - P_GOLD) : t;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    t -= (t > a) ? EPSILON : 0ul;
    return t;
}

inline ulong2 umul128(ulong a, ulong b) {
    return ulong2(a * b, mulhi(a, b));
}

inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong x_hi_lo = x_hi & EPSILON;
    ulong x_hi_hi = x_hi >> 32;

    ulong t0 = x_lo - x_hi_hi;
    t0 -= (t0 > x_lo) ? EPSILON : 0ul;

    ulong t1 = (x_hi_lo << 32) - x_hi_lo;

    ulong t2 = t0 + t1;
    t2 += (t2 < t0) ? EPSILON : 0ul;

    return gold_canonical(t2);
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong2 prod = umul128(a, b);
    return gold_reduce128(prod.x, prod.y);
}

// ----------------------------------------------------------------------
// FRI fold (highly specialized per fold factor)
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
        ulong z0 = zeta_inv_pow[0];
        ulong z1 = zeta_inv_pow[1];

        ulong rm0 = gold_mul(ax, z0);
        ulong sm0 = gold_add(1ul, rm0);
        ulong term0 = gold_mul(evals_in[j], sm0);

        ulong rm1 = gold_mul(ax, z1);
        ulong sm1 = gold_add(1ul, rm1);
        ulong term1 = gold_mul(evals_in[j + n_out], sm1);

        ulong acc = gold_add(term0, term1);
        evals_out[j] = gold_mul(acc, inv_fold);
    } else if (fold == 4u) {
        ulong z0 = zeta_inv_pow[0];
        ulong z1 = zeta_inv_pow[1];
        ulong z2 = zeta_inv_pow[2];
        ulong z3 = zeta_inv_pow[3];

        ulong ax_sq = gold_mul(ax, ax);
        ulong ax_sq_neg = gold_sub(0ul, ax_sq);

        ulong rm0 = gold_mul(ax, z0);
        ulong sm0 = gold_mul(gold_add(1ul, rm0), gold_add(1ul, ax_sq));
        ulong term0 = gold_mul(evals_in[j], sm0);

        ulong rm1 = gold_mul(ax, z1);
        ulong sm1 = gold_mul(gold_add(1ul, rm1), gold_add(1ul, ax_sq_neg));
        ulong term1 = gold_mul(evals_in[j + n_out], sm1);

        ulong rm2 = gold_mul(ax, z2);
        ulong sm2 = gold_mul(gold_add(1ul, rm2), gold_add(1ul, ax_sq));
        ulong term2 = gold_mul(evals_in[j + 2u * n_out], sm2);

        ulong rm3 = gold_mul(ax, z3);
        ulong sm3 = gold_mul(gold_add(1ul, rm3), gold_add(1ul, ax_sq_neg));
        ulong term3 = gold_mul(evals_in[j + 3u * n_out], sm3);

        ulong acc = gold_add(gold_add(term0, term1), gold_add(term2, term3));
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
// Poseidon2-t=3 (binary Merkle commit)
// ----------------------------------------------------------------------

inline ulong sbox7(ulong x) {
    ulong x2 = gold_mul(x, x);
    ulong x4 = gold_mul(x2, x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
}

inline void matvec_ext_t3(thread ulong *state,
                          ulong m0, ulong m1, ulong m2,
                          ulong m3, ulong m4, ulong m5,
                          ulong m6, ulong m7, ulong m8)
{
    ulong s0 = state[0], s1 = state[1], s2 = state[2];
    state[0] = gold_add(gold_add(gold_mul(m0, s0), gold_mul(m1, s1)), gold_mul(m2, s2));
    state[1] = gold_add(gold_add(gold_mul(m3, s0), gold_mul(m4, s1)), gold_mul(m5, s2));
    state[2] = gold_add(gold_add(gold_mul(m6, s0), gold_mul(m7, s1)), gold_mul(m8, s2));
}

inline void matvec_int_t3(thread ulong *state,
                          ulong d0, ulong d1, ulong d2)
{
    ulong s = gold_add(gold_add(state[0], state[1]), state[2]);
    state[0] = gold_add(s, gold_mul(d0, state[0]));
    state[1] = gold_add(s, gold_mul(d1, state[1]));
    state[2] = gold_add(s, gold_mul(d2, state[2]));
}

inline void poseidon2_permute_t3(thread ulong *state,
                                 device const ulong *rc_ext,
                                 device const ulong *rc_int,
                                 device const ulong *ext_mds,
                                 device const ulong *int_diag)
{
    ulong m0 = ext_mds[0], m1 = ext_mds[1], m2 = ext_mds[2];
    ulong m3 = ext_mds[3], m4 = ext_mds[4], m5 = ext_mds[5];
    ulong m6 = ext_mds[6], m7 = ext_mds[7], m8 = ext_mds[8];

    ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2];

    matvec_ext_t3(state, m0, m1, m2, m3, m4, m5, m6, m7, m8);

    // First half full rounds (4 rounds)
    #pragma unroll
    for (uint r = 0u; r < 4u; ++r) {
        state[0] = sbox7(gold_add(state[0], rc_ext[r * 3u + 0u]));
        state[1] = sbox7(gold_add(state[1], rc_ext[r * 3u + 1u]));
        state[2] = sbox7(gold_add(state[2], rc_ext[r * 3u + 2u]));
        matvec_ext_t3(state, m0, m1, m2, m3, m4, m5, m6, m7, m8);
    }

    // Partial rounds (22 rounds)
    #pragma unroll
    for (uint r = 0u; r < 22u; ++r) {
        state[0] = sbox7(gold_add(state[0], rc_int[r]));
        matvec_int_t3(state, d0, d1, d2);
    }

    // Second half full rounds (4 rounds)
    #pragma unroll
    for (uint r = 4u; r < 8u; ++r) {
        state[0] = sbox7(gold_add(state[0], rc_ext[r * 3u + 0u]));
        state[1] = sbox7(gold_add(state[1], rc_ext[r * 3u + 1u]));
        state[2] = sbox7(gold_add(state[2], rc_ext[r * 3u + 2u]));
        matvec_ext_t3(state, m0, m1, m2, m3, m4, m5, m6, m7, m8);
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

    thread ulong state[T_FIXED];
    state[0] = 0ul; state[1] = 0ul; state[2] = 0ul;

    uint base = p << 1u;
    state[0] = tree[in_offset + base];
    state[1] = (base + 1u < child_count) ? tree[in_offset + base + 1u] : 0ul;

    poseidon2_permute_t3(state, rc_ext, rc_int, ext_mds, int_diag);

    tree[out_offset + p] = state[0];
}