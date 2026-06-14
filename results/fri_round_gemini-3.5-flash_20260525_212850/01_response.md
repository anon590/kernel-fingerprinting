### Optimization Techniques

1. **Branchless Goldilocks Arithmetic**: Rewrote modular addition, subtraction, and reduction using conditional ternary operators that compile directly to high-throughput hardware conditional selects (`csel`), entirely eliminating branch penalties.
2. **Multiplication-Free Reduction**: Replaced `x_hi_lo * EPSILON` in `gold_reduce128` with `(x_hi_lo << 32) - x_hi_lo`, saving one 64-bit multiply per modular multiplication.
3. **Register-Promoted Poseidon2**: Loaded MDS matrices, diagonal elements, and round constants into local thread registers at the beginning of the permutation, avoiding redundant high-latency global memory transactions.
4. **Algebraic Simplification and Loop-Free Folding**: Completely unrolled the `fold == 2` and `fold == 4` paths. Optimized $1 + r + r^2 + r^3$ to $(1 + r)(1 + r^2)$ to reduce the multiplication count for calculating $S_m$ from 3 to 2.

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;   // 2^64 - 2^32 + 1
constant ulong EPSILON = 0x00000000FFFFFFFFul;   // 2^32 - 1

// ----------------------------------------------------------------------
// Branchless Goldilocks Arithmetic
// ----------------------------------------------------------------------

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    ulong carry = (t < a) ? EPSILON : 0ul;
    t += carry;
    return (t >= P_GOLD) ? (t - P_GOLD) : t;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    ulong borrow = (t > a) ? EPSILON : 0ul;
    return t - borrow;
}

inline ulong gold_reduce128(ulong lo, ulong hi) {
    ulong x_hi_lo = hi & EPSILON;
    ulong x_hi_hi = hi >> 32;

    ulong t0 = lo - x_hi_hi;
    ulong borrow = (t0 > lo) ? EPSILON : 0ul;
    t0 -= borrow;

    ulong t1 = (x_hi_lo << 32) - x_hi_lo;

    ulong t2 = t0 + t1;
    ulong carry = (t2 < t0) ? EPSILON : 0ul;
    t2 += carry;

    return gold_canonical(t2);
}

inline ulong gold_mul(ulong a, ulong b) {
    uint a0 = (uint)(a);
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)(b);
    uint b1 = (uint)(b >> 32);

    ulong p00 = (ulong)a0 * b0;
    ulong p01 = (ulong)a0 * b1;
    ulong p10 = (ulong)a1 * b0;
    ulong p11 = (ulong)a1 * b1;

    ulong mid = (p00 >> 32) + (p01 & EPSILON) + (p10 & EPSILON);
    ulong lo  = (p00 & EPSILON) | (mid << 32);
    ulong hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);

    return gold_reduce128(lo, hi);
}

// ----------------------------------------------------------------------
// FRI fold (one kernel dispatch per round)
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

        ulong rm0 = gold_mul(ax, z0);
        ulong rm0_sq = gold_mul(rm0, rm0);
        ulong sm0 = gold_mul(gold_add(1ul, rm0), gold_add(1ul, rm0_sq));
        ulong acc = gold_mul(evals_in[j], sm0);

        ulong rm1 = gold_mul(ax, z1);
        ulong rm1_sq = gold_mul(rm1, rm1);
        ulong sm1 = gold_mul(gold_add(1ul, rm1), gold_add(1ul, rm1_sq));
        acc = gold_add(acc, gold_mul(evals_in[j + n_out], sm1));

        ulong rm2 = gold_mul(ax, z2);
        ulong rm2_sq = gold_mul(rm2, rm2);
        ulong sm2 = gold_mul(gold_add(1ul, rm2), gold_add(1ul, rm2_sq));
        acc = gold_add(acc, gold_mul(evals_in[j + 2u * n_out], sm2));

        ulong rm3 = gold_mul(ax, z3);
        ulong rm3_sq = gold_mul(rm3, rm3);
        ulong sm3 = gold_mul(gold_add(1ul, rm3), gold_add(1ul, rm3_sq));
        acc = gold_add(acc, gold_mul(evals_in[j + 3u * n_out], sm3));

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

    ulong rx[24];
    #pragma unroll
    for (uint i = 0u; i < 24u; ++i) rx[i] = rc_ext[i];

    ulong ri[22];
    #pragma unroll
    for (uint i = 0u; i < 22u; ++i) ri[i] = rc_int[i];

    matvec_ext_t3(state, m0, m1, m2, m3, m4, m5, m6, m7, m8);

    // First half full rounds (4 rounds)
    #pragma unroll
    for (uint r = 0u; r < 4u; ++r) {
        state[0] = sbox7(gold_add(state[0], rx[r * 3u + 0u]));
        state[1] = sbox7(gold_add(state[1], rx[r * 3u + 1u]));
        state[2] = sbox7(gold_add(state[2], rx[r * 3u + 2u]));
        matvec_ext_t3(state, m0, m1, m2, m3, m4, m5, m6, m7, m8);
    }

    // Partial rounds (22 rounds)
    #pragma unroll
    for (uint r = 0u; r < 22u; ++r) {
        state[0] = sbox7(gold_add(state[0], ri[r]));
        matvec_int_t3(state, d0, d1, d2);
    }

    // Second half full rounds (4 rounds)
    #pragma unroll
    for (uint r = 4u; r < 8u; ++r) {
        state[0] = sbox7(gold_add(state[0], rx[r * 3u + 0u]));
        state[1] = sbox7(gold_add(state[1], rx[r * 3u + 1u]));
        state[2] = sbox7(gold_add(state[2], rx[r * 3u + 2u]));
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

    thread ulong state[3];
    uint base = p << 1u;
    state[0] = tree[in_offset + base];
    state[1] = (base + 1u < child_count) ? tree[in_offset + base + 1u] : 0ul;
    state[2] = 0ul;

    poseidon2_permute_t3(state, rc_ext, rc_int, ext_mds, int_diag);

    tree[out_offset + p] = state[0];
}
```