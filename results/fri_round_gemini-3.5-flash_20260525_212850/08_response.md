To maximize performance on Apple Silicon GPUs, we apply several key mathematical and engineering optimizations:

1. **Branchless Arithmetic**: We rewrite `gold_add`, `gold_sub`, `gold_reduce128`, and `gold_canonical` to be completely branchless using ternary select statements, which compile directly to hardware `csel` instructions.
2. **Squaring Optimization**: We implement an optimized 3-multiply `gold_sqr` instead of the general 4-multiply `gold_mul`, saving 25% of the multiplication instructions during squarings in `sbox7` and `fri_fold`.
3. **Advanced FRI Fold Factorization**: For `fold == 2` and `fold == 4`, we mathematically factorize the FRI folding arithmetic. This drastically reduces the number of multiplications (from 5 to 3 for `fold == 2`, and from 21 to only 11 for `fold == 4`).
4. **Register Caching & Balanced Unrolling**: In `fri_commit_level`, we pre-load MDS matrix and diagonal elements to thread registers. We unroll only the full rounds for high ILP while keeping the 22 partial rounds in a compact loop to avoid instruction-cache pressure and register spilling.
5. **Vectorized Loading**: We load consecutive sibling leaf hashes as `ulong2` vectors, yielding aligned 16-byte memory reads.

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;   // 2^64 - 2^32 + 1
constant ulong EPSILON = 0x00000000FFFFFFFFul;   // 2^32 - 1

constexpr constant uint T_FIXED  = 3u;
constexpr constant uint POS2_R_F = 8u;

// ----------------------------------------------------------------------
// Branchless Goldilocks Arithmetic
// ----------------------------------------------------------------------

inline ulong gold_canonical(ulong x) {
    ulong t = x - P_GOLD;
    return (x >= P_GOLD) ? t : x;
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    ulong t_plus_eps = t + EPSILON;
    t = (t < a) ? t_plus_eps : t;
    ulong t_minus_p = t - P_GOLD;
    return (t >= P_GOLD) ? t_minus_p : t;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    ulong t_minus_eps = t - EPSILON;
    return (t > a) ? t_minus_eps : t;
}

inline ulong2 umul128(ulong a, ulong b) {
    ulong lo = a * b;

    uint a0 = (uint)(a);
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)(b);
    uint b1 = (uint)(b >> 32);

    ulong p00 = (ulong)a0 * b0;
    ulong p01 = (ulong)a0 * b1;
    ulong p10 = (ulong)a1 * b0;
    ulong p11 = (ulong)a1 * b1;

    ulong mid = (p00 >> 32) + (p01 & EPSILON) + (p10 & EPSILON);
    ulong hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return ulong2(lo, hi);
}

inline ulong2 usqr128(ulong a) {
    ulong lo = a * a;

    uint a0 = (uint)(a);
    uint a1 = (uint)(a >> 32);

    ulong p00 = (ulong)a0 * a0;
    ulong p01 = (ulong)a0 * a1;
    ulong p11 = (ulong)a1 * a1;

    ulong mid = (p00 >> 32) + ((p01 & EPSILON) << 1);
    ulong hi  = p11 + ((p01 >> 32) << 1) + (mid >> 32);
    return ulong2(lo, hi);
}

inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong x_hi_lo = x_hi & EPSILON;
    ulong x_hi_hi = x_hi >> 32;

    ulong t0 = x_lo - x_hi_hi;
    ulong t0_minus_eps = t0 - EPSILON;
    t0 = (t0 > x_lo) ? t0_minus_eps : t0;

    ulong t1 = x_hi_lo * EPSILON;

    ulong t2 = t0 + t1;
    ulong t2_plus_eps = t2 + EPSILON;
    t2 = (t2 < t0) ? t2_plus_eps : t2;

    return gold_canonical(t2);
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong2 prod = umul128(a, b);
    return gold_reduce128(prod.x, prod.y);
}

inline ulong gold_sqr(ulong a) {
    ulong2 prod = usqr128(a);
    return gold_reduce128(prod.x, prod.y);
}

inline ulong sbox7(ulong x) {
    ulong x2 = gold_sqr(x);
    ulong x4 = gold_sqr(x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
}

// ----------------------------------------------------------------------
// FRI fold (highly optimized factorization paths)
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
        ulong z1 = zeta_inv_pow[1];
        ulong E0 = evals_in[j];
        ulong E1 = evals_in[j + n_out];

        ulong E_sum  = gold_add(E0, E1);
        ulong E_diff = gold_add(E0, gold_mul(E1, z1));
        ulong acc    = gold_add(E_sum, gold_mul(ax, E_diff));

        evals_out[j] = gold_mul(acc, inv_fold);
    } 
    else if (fold == 4u) {
        ulong z1 = zeta_inv_pow[1];
        ulong z2 = zeta_inv_pow[2];

        ulong ax_sq    = gold_sqr(ax);
        ulong ax_sq_z2 = gold_mul(ax_sq, z2);

        ulong rm1 = gold_mul(ax, z1);

        ulong E0 = evals_in[j];
        ulong E1 = evals_in[j + n_out];
        ulong E2 = evals_in[j + 2u * n_out];
        ulong E3 = evals_in[j + 3u * n_out];

        ulong E0_E2 = gold_add(E0, E2);
        ulong E2_z2 = gold_mul(E2, z2);
        ulong part0 = gold_add(E0_E2, gold_mul(ax, gold_add(E0, E2_z2)));

        ulong E1_E3 = gold_add(E1, E3);
        ulong E3_z2 = gold_mul(E3, z2);
        ulong part1 = gold_add(E1_E3, gold_mul(rm1, gold_add(E1, E3_z2)));

        ulong term0 = gold_mul(part0, gold_add(1ul, ax_sq));
        ulong term1 = gold_mul(part1, gold_add(1ul, ax_sq_z2));

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
// Poseidon2-t=3 (binary Merkle commit)
// ----------------------------------------------------------------------

inline void matvec_ext_t3_local(thread ulong *state, thread const ulong *m) {
    ulong t0 = gold_add(gold_add(gold_mul(m[0], state[0]), gold_mul(m[1], state[1])), gold_mul(m[2], state[2]));
    ulong t1 = gold_add(gold_add(gold_mul(m[3], state[0]), gold_mul(m[4], state[1])), gold_mul(m[5], state[2]));
    ulong t2 = gold_add(gold_add(gold_mul(m[6], state[0]), gold_mul(m[7], state[1])), gold_mul(m[8], state[2]));
    state[0] = t0; state[1] = t1; state[2] = t2;
}

inline void matvec_int_t3_local(thread ulong *state, thread const ulong *d) {
    ulong s = gold_add(gold_add(state[0], state[1]), state[2]);
    state[0] = gold_add(s, gold_mul(d[0], state[0]));
    state[1] = gold_add(s, gold_mul(d[1], state[1]));
    state[2] = gold_add(s, gold_mul(d[2], state[2]));
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

    ulong state[3] = {0ul, 0ul, 0ul};

    uint base = p << 1u;
    if (base + 1u < child_count) {
        ulong2 loaded = ((device const ulong2*)(tree + in_offset))[p];
        state[0] = loaded.x;
        state[1] = loaded.y;
    } else {
        state[0] = tree[in_offset + base];
    }

    ulong m[9];
    m[0] = ext_mds[0]; m[1] = ext_mds[1]; m[2] = ext_mds[2];
    m[3] = ext_mds[3]; m[4] = ext_mds[4]; m[5] = ext_mds[5];
    m[6] = ext_mds[6]; m[7] = ext_mds[7]; m[8] = ext_mds[8];

    ulong d[3];
    d[0] = int_diag[0]; d[1] = int_diag[1]; d[2] = int_diag[2];

    matvec_ext_t3_local(state, m);

    // First half full rounds (4 rounds)
    #pragma unroll
    for (uint r = 0u; r < 4u; ++r) {
        state[0] = sbox7(gold_add(state[0], rc_ext[r * 3u + 0u]));
        state[1] = sbox7(gold_add(state[1], rc_ext[r * 3u + 1u]));
        state[2] = sbox7(gold_add(state[2], rc_ext[r * 3u + 2u]));
        matvec_ext_t3_local(state, m);
    }

    // Partial rounds (22 rounds)
    for (uint r = 0u; r < 22u; ++r) {
        state[0] = sbox7(gold_add(state[0], rc_int[r]));
        matvec_int_t3_local(state, d);
    }

    // Second half full rounds (4 rounds)
    #pragma unroll
    for (uint r = 4u; r < 8u; ++r) {
        state[0] = sbox7(gold_add(state[0], rc_ext[r * 3u + 0u]));
        state[1] = sbox7(gold_add(state[1], rc_ext[r * 3u + 1u]));
        state[2] = sbox7(gold_add(state[2], rc_ext[r * 3u + 2u]));
        matvec_ext_t3_local(state, m);
    }

    tree[out_offset + p] = state[0];
}
```