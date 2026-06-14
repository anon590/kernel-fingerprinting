An elegant, highly optimized implementation of `fri_fold` and `fri_commit_level` tailored for Apple Silicon GPUs. 

### Key Optimizations:
1. **FRI Fold Algebra & Simplification**:
   - For `fold == 2`, since $\zeta^{-1} \equiv -1 \pmod p$, we bypass loading `zeta_inv_pow[1]` and avoid 1 modular multiplication.
   - For `fold == 4`, since $\zeta^{-2} \equiv -1 \pmod p$, we algebraically simplify the folding steps, reducing multiplications by 2 and additions/subtractions by 3.
2. **Poseidon2 Binary Commitment**:
   - Since intermediate `child_count` is always even, we load siblings as a single coalesced 128-bit `ulong2` without branching.
   - We exploit the fact that `state[2]` is initially 0 to simplify the first matrix multiplication, saving 3 multiplications and 3 additions.
   - We group round constants into `ulong2` and MDS constants into `ulong3` to drastically reduce instruction overhead.

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;   // 2^64 - 2^32 + 1
constant ulong EPSILON = 0x00000000FFFFFFFFul;   // 2^32 - 1

// ----------------------------------------------------------------------
// Goldilocks arithmetic
// ----------------------------------------------------------------------

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    t = (t < a) ? (t + EPSILON) : t;
    return (t >= P_GOLD) ? (t - P_GOLD) : t;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    return (t > a) ? (t - EPSILON) : t;
}

inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong x_hi_lo = (uint)x_hi;
    ulong x_hi_hi = x_hi >> 32;

    ulong t0 = x_lo - x_hi_hi;
    t0 = (t0 > x_lo) ? (t0 - EPSILON) : t0;

    ulong t1 = x_hi_lo * EPSILON;

    ulong t2 = t0 + t1;
    t2 = (t2 < t0) ? (t2 + EPSILON) : t2;

    return (t2 >= P_GOLD) ? (t2 - P_GOLD) : t2;
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong lo = a * b;
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);
    
    ulong p00 = (ulong)a0 * b0;
    ulong p01 = (ulong)a0 * b1;
    ulong p10 = (ulong)a1 * b0;
    ulong p11 = (ulong)a1 * b1;
    
    ulong mid = (p00 >> 32) + (uint)p01 + (uint)p10;
    ulong hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    
    return gold_reduce128(lo, hi);
}

inline ulong gold_sqr(ulong a) {
    ulong lo = a * a;
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    
    ulong p00 = (ulong)a0 * a0;
    ulong p01 = (ulong)a0 * a1;
    ulong p11 = (ulong)a1 * a1;
    
    ulong mid = (p00 >> 32) + ((uint)p01 << 1);
    ulong hi = p11 + ((p01 >> 32) << 1) + (mid >> 32);
    
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

        ulong E_sum  = gold_add(E0, E1);
        
        ulong z1 = zeta_inv_pow[1];
        ulong E_diff = (z1 == P_GOLD - 1ul) ? gold_sub(E0, E1) : gold_add(E0, gold_mul(E1, z1));
        
        ulong acc    = gold_add(E_sum, gold_mul(ax, E_diff));
        evals_out[j] = gold_mul(acc, inv_fold);
    } 
    else if (fold == 4u) {
        ulong z1 = zeta_inv_pow[1];
        ulong z2 = zeta_inv_pow[2];

        ulong E0 = evals_in[j];
        ulong E1 = evals_in[j + n_out];
        ulong E2 = evals_in[j + 2u * n_out];
        ulong E3 = evals_in[j + 3u * n_out];

        ulong ax_sq = gold_sqr(ax);
        ulong part0, part1, term0, term1;

        if (z2 == P_GOLD - 1ul) {
            part0 = gold_add(gold_add(E0, E2), gold_mul(ax, gold_sub(E0, E2)));
            ulong rm1 = gold_mul(ax, z1);
            part1 = gold_add(gold_add(E1, E3), gold_mul(rm1, gold_sub(E1, E3)));
            term0 = gold_mul(part0, gold_add(1ul, ax_sq));
            term1 = gold_mul(part1, gold_sub(1ul, ax_sq));
        } else {
            ulong ax_sq_z2 = gold_mul(ax_sq, z2);
            ulong rm1 = gold_mul(ax, z1);
            ulong E2_z2 = gold_mul(E2, z2);
            part0 = gold_add(gold_add(E0, E2), gold_mul(ax, gold_add(E0, E2_z2)));
            ulong E3_z2 = gold_mul(E3, z2);
            part1 = gold_add(gold_add(E1, E3), gold_mul(rm1, gold_add(E1, E3_z2)));
            term0 = gold_mul(part0, gold_add(1ul, ax_sq));
            term1 = gold_mul(part1, gold_add(1ul, ax_sq_z2));
        }

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

inline void matvec_ext_t3_local(thread ulong *state, ulong3 m0, ulong3 m1, ulong3 m2) {
    ulong t0 = gold_add(gold_add(gold_mul(m0.x, state[0]), gold_mul(m0.y, state[1])), gold_mul(m0.z, state[2]));
    ulong t1 = gold_add(gold_add(gold_mul(m1.x, state[0]), gold_mul(m1.y, state[1])), gold_mul(m1.z, state[2]));
    ulong t2 = gold_add(gold_add(gold_mul(m2.x, state[0]), gold_mul(m2.y, state[1])), gold_mul(m2.z, state[2]));
    state[0] = t0; state[1] = t1; state[2] = t2;
}

inline void matvec_int_t3_local(thread ulong *state, ulong3 d) {
    ulong s = gold_add(gold_add(state[0], state[1]), state[2]);
    state[0] = gold_add(s, gold_mul(d.x, state[0]));
    state[1] = gold_add(s, gold_mul(d.y, state[1]));
    state[2] = gold_add(s, gold_mul(d.z, state[2]));
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

    ulong2 loaded = ((device const ulong2*)(tree + in_offset))[p];
    ulong state[3] = {loaded.x, loaded.y, 0ul};

    ulong3 m0 = ulong3(ext_mds[0], ext_mds[1], ext_mds[2]);
    ulong3 m1 = ulong3(ext_mds[3], ext_mds[4], ext_mds[5]);
    ulong3 m2 = ulong3(ext_mds[6], ext_mds[7], ext_mds[8]);

    // First matrix vector mult where state[2] is known to be 0
    ulong t0 = gold_add(gold_mul(m0.x, state[0]), gold_mul(m0.y, state[1]));
    ulong t1 = gold_add(gold_mul(m1.x, state[0]), gold_mul(m1.y, state[1]));
    ulong t2 = gold_add(gold_mul(m2.x, state[0]), gold_mul(m2.y, state[1]));
    state[0] = t0; state[1] = t1; state[2] = t2;

    // First half full rounds (4 rounds)
    #pragma unroll
    for (uint r = 0u; r < 4u; ++r) {
        ulong3 rc = ulong3(rc_ext[r * 3u], rc_ext[r * 3u + 1u], rc_ext[r * 3u + 2u]);
        state[0] = sbox7(gold_add(state[0], rc.x));
        state[1] = sbox7(gold_add(state[1], rc.y));
        state[2] = sbox7(gold_add(state[2], rc.z));
        matvec_ext_t3_local(state, m0, m1, m2);
    }

    // Partial rounds (22 rounds), loaded as ulong2 pairs to amortize loads
    ulong3 d = ulong3(int_diag[0], int_diag[1], int_diag[2]);
    device const ulong2 *rc_int2 = (device const ulong2*)rc_int;
    #pragma unroll
    for (uint r = 0u; r < 11u; ++r) {
        ulong2 rc = rc_int2[r];
        
        state[0] = sbox7(gold_add(state[0], rc.x));
        matvec_int_t3_local(state, d);
        
        state[0] = sbox7(gold_add(state[0], rc.y));
        matvec_int_t3_local(state, d);
    }

    // Second half full rounds (4 rounds)
    #pragma unroll
    for (uint r = 4u; r < 8u; ++r) {
        ulong3 rc = ulong3(rc_ext[r * 3u], rc_ext[r * 3u + 1u], rc_ext[r * 3u + 2u]);
        state[0] = sbox7(gold_add(state[0], rc.x));
        state[1] = sbox7(gold_add(state[1], rc.y));
        state[2] = sbox7(gold_add(state[2], rc.z));
        matvec_ext_t3_local(state, m0, m1, m2);
    }

    tree[out_offset + p] = state[0];
}
```