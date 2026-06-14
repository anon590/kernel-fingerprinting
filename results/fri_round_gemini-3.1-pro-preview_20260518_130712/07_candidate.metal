#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD = 0xFFFFFFFF00000001ul;

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    if (t < a) t += 0xFFFFFFFFul;
    if (t >= P_GOLD) t -= P_GOLD;
    return t;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    // Breaking dependency on `t` for the condition allows parallel borrow evaluation
    if (a < b) t -= 0xFFFFFFFFul;
    return t;
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong lo = a * b;
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);
    
    ulong p01 = (ulong)a0 * b1;
    ulong p10 = (ulong)a1 * b0;
    
    // Explicit upcasting guarantees 64-bit addition
    ulong mid = (ulong)mulhi(a0, b0) + (ulong)(uint)p01 + (ulong)(uint)p10;
    ulong hi = (ulong)a1 * b1 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    
    uint hi_lo = (uint)hi;
    uint hi_hi = (uint)(hi >> 32);

    ulong t0 = lo - hi_hi;
    if (lo < hi_hi) t0 -= 0xFFFFFFFFul;

    // Replaces hi_lo * 0xFFFFFFFFul with a faster shift/subtract identity
    ulong t1 = ((ulong)hi_lo << 32) - hi_lo;

    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += 0xFFFFFFFFul;

    return (t2 >= P_GOLD) ? (t2 - P_GOLD) : t2;
}

inline ulong sbox7(ulong x) {
    ulong x2 = gold_mul(x, x);
    ulong x3 = gold_mul(x2, x);
    ulong x4 = gold_mul(x2, x2);
    return gold_mul(x4, x3);
}

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
    ulong acc = 0ul;

    if (fold == 2u) {
        ulong e0 = evals_in[j];
        ulong e1 = evals_in[j + n_out];
        
        ulong e0_plus_e1 = gold_add(e0, e1);
        ulong e0_minus_e1 = gold_sub(e0, e1);
        
        acc = gold_add(e0_plus_e1, gold_mul(ax, e0_minus_e1));
    } else if (fold == 4u) {
        ulong e0 = evals_in[j];
        ulong e1 = evals_in[j + n_out];
        ulong e2 = evals_in[j + 2u * n_out];
        ulong e3 = evals_in[j + 3u * n_out];

        ulong e0_plus_e2 = gold_add(e0, e2);
        ulong e0_minus_e2 = gold_sub(e0, e2);
        ulong e1_plus_e3 = gold_add(e1, e3);
        ulong e1_minus_e3 = gold_sub(e1, e3);

        ulong r0 = ax;
        ulong r1 = gold_mul(ax, zeta_inv_pow[1]);

        ulong term0 = gold_add(e0_plus_e2, gold_mul(r0, e0_minus_e2));
        ulong term1 = gold_add(e1_plus_e3, gold_mul(r1, e1_minus_e3));

        ulong r0_2 = gold_mul(r0, r0);
        ulong T0 = gold_add(1ul, r0_2);

        // Mathematical factoring reduces standard fold polynomial from 7 to 4 multiplications
        ulong diff = gold_sub(term0, term1);
        ulong term1_x2 = gold_add(term1, term1);

        acc = gold_add(gold_mul(T0, diff), term1_x2);
    } else {
        for (uint m = 0u; m < fold; ++m) {
            ulong rm = (m == 0u) ? ax : gold_mul(ax, zeta_inv_pow[m]);
            ulong sm = 0ul;
            ulong rpow = 1ul;
            for (uint p = 0u; p < fold; ++p) {
                sm = gold_add(sm, rpow);
                rpow = gold_mul(rpow, rm);
            }
            uint src = j + m * n_out;
            acc = gold_add(acc, gold_mul(evals_in[src], sm));
        }
    }
    
    evals_out[j] = gold_mul(acc, inv_fold);
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

    ulong state0 = 0ul;
    ulong state1 = 0ul;
    ulong state2 = 0ul;

    uint base = p << 1u;
    state0 = tree[in_offset + base];
    if (base + 1u < child_count) {
        state1 = tree[in_offset + base + 1u];
    }

    ulong m0 = ext_mds[0], m1 = ext_mds[1], m2 = ext_mds[2];
    ulong m3 = ext_mds[3], m4 = ext_mds[4], m5 = ext_mds[5];
    ulong m6 = ext_mds[6], m7 = ext_mds[7], m8 = ext_mds[8];
    ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2];

    {
        ulong t0 = state0, t1 = state1, t2 = state2;
        state0 = gold_add(gold_add(gold_mul(m0, t0), gold_mul(m1, t1)), gold_mul(m2, t2));
        state1 = gold_add(gold_add(gold_mul(m3, t0), gold_mul(m4, t1)), gold_mul(m5, t2));
        state2 = gold_add(gold_add(gold_mul(m6, t0), gold_mul(m7, t1)), gold_mul(m8, t2));
    }

    uint rc_idx = 0u;
    for (uint r = 0u; r < 4u; ++r) {
        state0 = sbox7(gold_add(state0, rc_ext[rc_idx]));
        state1 = sbox7(gold_add(state1, rc_ext[rc_idx + 1u]));
        state2 = sbox7(gold_add(state2, rc_ext[rc_idx + 2u]));
        rc_idx += 3u;

        ulong t0 = state0, t1 = state1, t2 = state2;
        state0 = gold_add(gold_add(gold_mul(m0, t0), gold_mul(m1, t1)), gold_mul(m2, t2));
        state1 = gold_add(gold_add(gold_mul(m3, t0), gold_mul(m4, t1)), gold_mul(m5, t2));
        state2 = gold_add(gold_add(gold_mul(m6, t0), gold_mul(m7, t1)), gold_mul(m8, t2));
    }

    // Unrolling by exactly 2 overlaps `sbox7` dependencies properly without register spilling
    #pragma clang loop unroll_count(2)
    for (uint r = 0u; r < 22u; ++r) {
        ulong s_rem = gold_add(state1, state2);
        state0 = sbox7(gold_add(state0, rc_int[r]));
        
        ulong s = gold_add(state0, s_rem);
        state0 = gold_add(s, gold_mul(d0, state0));
        state1 = gold_add(s, gold_mul(d1, state1));
        state2 = gold_add(s, gold_mul(d2, state2));
    }

    // rc_idx accurately resumes tracking here for full round chunk 2
    for (uint r = 4u; r < 8u; ++r) {
        state0 = sbox7(gold_add(state0, rc_ext[rc_idx]));
        state1 = sbox7(gold_add(state1, rc_ext[rc_idx + 1u]));
        state2 = sbox7(gold_add(state2, rc_ext[rc_idx + 2u]));
        rc_idx += 3u;

        ulong t0 = state0, t1 = state1, t2 = state2;
        state0 = gold_add(gold_add(gold_mul(m0, t0), gold_mul(m1, t1)), gold_mul(m2, t2));
        state1 = gold_add(gold_add(gold_mul(m3, t0), gold_mul(m4, t1)), gold_mul(m5, t2));
        state2 = gold_add(gold_add(gold_mul(m6, t0), gold_mul(m7, t1)), gold_mul(m8, t2));
    }

    tree[out_offset + p] = state0;
}