To maximize performance on Apple Silicon, we apply several key optimizations:

1. **Specialized FRI Folding Paths**: We fully specialize `fri_fold` for the runtime bounds `fold == 2` and `fold == 4`. By mathematically simplifying the coset evaluations, we reduce the number of modular multiplications from 22 to only 8 for `fold == 4`.
2. **Custom Multi-less Squaring (`gold_sqr`)**: We design an exact, custom squaring function using only 3 32-bit multiplications instead of 4, saving 92 multiplications per thread in `fri_commit_level` S-boxes.
3. **Multiplication-Free Reduction**: In `gold_reduce128`, we replace the costly 64-bit multiplication `x_hi_lo * EPSILON` with `(x_hi_lo << 32) - x_hi_lo`.
4. **Register-Promoted Parameters**: All MDS matrices, constants, and diagonals are preloaded into registers at the kernel's entry to avoid global memory latency during hot loops.
5. **Coalesced 128-bit Loads**: Vectorized `ulong2` reads are used for tree elements.

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;   // 2^64 - 2^32 + 1
constant ulong EPSILON = 0x00000000FFFFFFFFul;   // 2^32 - 1

constexpr constant uint T_FIXED  = 3u;           // Poseidon2 width
constexpr constant uint POS2_R_F = 8u;           // Poseidon2 full rounds (4+4)
constexpr constant uint POS2_R_P = 22u;          // Poseidon2 partial rounds

// ----------------------------------------------------------------------
// Goldilocks arithmetic
// ----------------------------------------------------------------------

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    if (t < a) t += EPSILON;
    if (t >= P_GOLD) t -= P_GOLD;
    return t;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    if (t > a) t -= EPSILON;
    return t;
}

inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong x_hi_lo = (uint)x_hi;
    ulong x_hi_hi = x_hi >> 32;

    ulong t0 = x_lo - x_hi_hi;
    if (t0 > x_lo) t0 -= EPSILON;

    ulong t1 = (x_hi_lo << 32) - x_hi_lo;

    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;

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

    ulong mid = (p00 >> 32) + (uint)p01 + (uint)p10;
    ulong lo  = (uint)p00 | (mid << 32);
    ulong hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);

    return gold_reduce128(lo, hi);
}

inline ulong gold_sqr(ulong a) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);

    ulong p00 = (ulong)a0 * a0;
    ulong p01 = (ulong)a0 * a1;
    ulong p11 = (ulong)a1 * a1;

    ulong p01_2 = p01 << 1;
    ulong carry_hi = p01 >> 63;

    ulong mid = (p00 >> 32) + (uint)p01_2;
    ulong lo  = (uint)p00 | (mid << 32);
    ulong hi  = p11 + (p01_2 >> 32) + (carry_hi << 32) + (mid >> 32);

    return gold_reduce128(lo, hi);
}

inline ulong sbox7(ulong x) {
    ulong x2 = gold_sqr(x);
    ulong x4 = gold_sqr(x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
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
        ulong E0 = evals_in[j];
        ulong E1 = evals_in[j + n_out];
        ulong E_sum = gold_add(E0, E1);
        ulong E_diff = gold_sub(E0, E1);
        ulong acc = gold_add(E_sum, gold_mul(ax, E_diff));
        evals_out[j] = gold_mul(acc, inv_fold);
    } 
    else if (fold == 4u) {
        ulong E0 = evals_in[j];
        ulong E1 = evals_in[j + n_out];
        ulong E2 = evals_in[j + 2u * n_out];
        ulong E3 = evals_in[j + 3u * n_out];

        ulong ax2 = gold_sqr(ax);
        ulong ax_z1 = gold_mul(ax, zeta_inv_pow[1]);

        ulong part0_diff = gold_sub(E0, E2);
        ulong part0_sum  = gold_add(E0, E2);
        ulong part0      = gold_add(part0_sum, gold_mul(ax, part0_diff));

        ulong part1_diff = gold_sub(E1, E3);
        ulong part1_sum  = gold_add(E1, E3);
        ulong part1      = gold_add(part1_sum, gold_mul(ax_z1, part1_diff));

        ulong term0 = gold_mul(part0, gold_add(1ul, ax2));
        ulong term1 = gold_mul(part1, gold_sub(1ul, ax2));

        ulong acc = gold_add(term0, term1);
        evals_out[j] = gold_mul(acc, inv_fold);
    } 
    else {
        ulong acc = 0ul;
        for (uint m = 0u; m < fold; ++m) {
            ulong rm = gold_mul(ax, zeta_inv_pow[m]);
            ulong sm   = 0ul;
            ulong rpow = 1ul;
            for (uint p = 0u; p < fold; ++p) {
                sm   = gold_add(sm, rpow);
                rpow = gold_mul(rpow, rm);
            }
            uint src = j + m * n_out;
            acc = gold_add(acc, gold_mul(evals_in[src], sm));
        }
        evals_out[j] = gold_mul(acc, inv_fold);
    }
}

// ----------------------------------------------------------------------
// Poseidon2-t=3 Merkle commit
// ----------------------------------------------------------------------

inline void matvec_ext_t3_local(thread ulong *state,
                                ulong m0, ulong m1, ulong m2,
                                ulong m3, ulong m4, ulong m5,
                                ulong m6, ulong m7, ulong m8)
{
    ulong t0 = gold_add(gold_add(gold_mul(m0, state[0]), gold_mul(m1, state[1])), gold_mul(m2, state[2]));
    ulong t1 = gold_add(gold_add(gold_mul(m3, state[0]), gold_mul(m4, state[1])), gold_mul(m5, state[2]));
    ulong t2 = gold_add(gold_add(gold_mul(m6, state[0]), gold_mul(m7, state[1])), gold_mul(m8, state[2]));
    state[0] = t0;
    state[1] = t1;
    state[2] = t2;
}

inline void matvec_int_t3_local(thread ulong *state, ulong d0, ulong d1, ulong d2) {
    ulong s = gold_add(gold_add(state[0], state[1]), state[2]);
    state[0] = gold_add(s, gold_mul(d0, state[0]));
    state[1] = gold_add(s, gold_mul(d1, state[1]));
    state[2] = gold_add(s, gold_mul(d2, state[2]));
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

    thread ulong state[3] = {0ul, 0ul, 0ul};
    uint base = p << 1u;

    if (base + 1u < child_count) {
        ulong2 loaded = ((device const ulong2 *)(tree + in_offset))[p];
        state[0] = loaded.x;
        state[1] = loaded.y;
    } else {
        state[0] = tree[in_offset + base];
    }

    // Load MDS matrix & Int Diag parameters directly into registers to avoid pointer arithmetic
    ulong m0 = ext_mds[0], m1 = ext_mds[1], m2 = ext_mds[2];
    ulong m3 = ext_mds[3], m4 = ext_mds[4], m5 = ext_mds[5];
    ulong m6 = ext_mds[6], m7 = ext_mds[7], m8 = ext_mds[8];

    ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2];

    // Specialized first step of Poseidon2 where state[2] is known to be 0
    ulong t0 = gold_add(gold_mul(m0, state[0]), gold_mul(m1, state[1]));
    ulong t1 = gold_add(gold_mul(m3, state[0]), gold_mul(m4, state[1]));
    ulong t2 = gold_add(gold_mul(m6, state[0]), gold_mul(m7, state[1]));
    state[0] = t0;
    state[1] = t1;
    state[2] = t2;

    // First half full rounds (4 rounds)
    #pragma unroll
    for (uint r = 0u; r < 4u; ++r) {
        state[0] = sbox7(gold_add(state[0], rc_ext[r * 3u + 0u]));
        state[1] = sbox7(gold_add(state[1], rc_ext[r * 3u + 1u]));
        state[2] = sbox7(gold_add(state[2], rc_ext[r * 3u + 2u]));
        matvec_ext_t3_local(state, m0, m1, m2, m3, m4, m5, m6, m7, m8);
    }

    // Partial rounds (22 rounds)
    #pragma unroll
    for (uint r = 0u; r < 22u; ++r) {
        state[0] = sbox7(gold_add(state[0], rc_int[r]));
        matvec_int_t3_local(state, d0, d1, d2);
    }

    // Second half full rounds (4 rounds)
    #pragma unroll
    for (uint r = 4u; r < 8u; ++r) {
        state[0] = sbox7(gold_add(state[0], rc_ext[r * 3u + 0u]));
        state[1] = sbox7(gold_add(state[1], rc_ext[r * 3u + 1u]));
        state[2] = sbox7(gold_add(state[2], rc_ext[r * 3u + 2u]));
        matvec_ext_t3_local(state, m0, m1, m2, m3, m4, m5, m6, m7, m8);
    }

    tree[out_offset + p] = state[0];
}
```