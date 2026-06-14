#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD = 0xFFFFFFFF00000001ul;

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    t += (t < a) ? 0xFFFFFFFFul : 0ul;
    return (t >= P_GOLD) ? (t - P_GOLD) : t;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    t -= (a < b) ? 0xFFFFFFFFul : 0ul;
    return t;
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong lo = a * b;
    uint2 av = as_type<uint2>(a);
    uint2 bv = as_type<uint2>(b);
    
    ulong p01 = (ulong)av.x * bv.y;
    ulong p10 = (ulong)av.y * bv.x;
    ulong p11 = (ulong)av.y * bv.y;
    
    // Extracted directly from 32-bit registers, bypassing ALUs for shifts/ANDs
    ulong mid = (ulong)mulhi(av.x, bv.x) + as_type<uint2>(p01).x + as_type<uint2>(p10).x;
    ulong hi = p11 + as_type<uint2>(p01).y + as_type<uint2>(p10).y + as_type<uint2>(mid).y;
    
    uint2 hiv = as_type<uint2>(hi);

    ulong t0 = lo - hiv.y;
    t0 -= (lo < hiv.y) ? 0xFFFFFFFFul : 0ul;

    ulong t1 = (ulong)hiv.x * 0xFFFFFFFFul;

    ulong t2 = t0 + t1;
    t2 += (t2 < t0) ? 0xFFFFFFFFul : 0ul;

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

        ulong r0_2 = gold_mul(r0, r0);

        ulong M0 = gold_add(e0_plus_e2, gold_mul(r0, e0_minus_e2));
        ulong M1 = gold_add(e1_plus_e3, gold_mul(r1, e1_minus_e3));

        ulong M0_plus_M1 = gold_add(M0, M1);
        ulong M0_minus_M1 = gold_sub(M0, M1);

        // Highly minimized sum operation - cuts down mult count vs explicitly expanding constants
        acc = gold_add(M0_plus_M1, gold_mul(r0_2, M0_minus_M1));
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

    #pragma clang loop unroll(full)
    for (uint r = 0u; r < 4u; ++r) {
        state0 = sbox7(gold_add(state0, rc_ext[r * 3u + 0u]));
        state1 = sbox7(gold_add(state1, rc_ext[r * 3u + 1u]));
        state2 = sbox7(gold_add(state2, rc_ext[r * 3u + 2u]));

        ulong t0 = state0, t1 = state1, t2 = state2;
        state0 = gold_add(gold_add(gold_mul(m0, t0), gold_mul(m1, t1)), gold_mul(m2, t2));
        state1 = gold_add(gold_add(gold_mul(m3, t0), gold_mul(m4, t1)), gold_mul(m5, t2));
        state2 = gold_add(gold_add(gold_mul(m6, t0), gold_mul(m7, t1)), gold_mul(m8, t2));
    }

    #pragma clang loop unroll_count(2)
    for (uint r = 0u; r < 22u; ++r) {
        ulong s_rem = gold_add(state1, state2);
        state0 = sbox7(gold_add(state0, rc_int[r]));
        
        ulong s = gold_add(state0, s_rem);
        state0 = gold_add(s, gold_mul(d0, state0));
        state1 = gold_add(s, gold_mul(d1, state1));
        state2 = gold_add(s, gold_mul(d2, state2));
    }

    #pragma clang loop unroll(full)
    for (uint r = 4u; r < 8u; ++r) {
        state0 = sbox7(gold_add(state0, rc_ext[r * 3u + 0u]));
        state1 = sbox7(gold_add(state1, rc_ext[r * 3u + 1u]));
        state2 = sbox7(gold_add(state2, rc_ext[r * 3u + 2u]));

        ulong t0 = state0, t1 = state1, t2 = state2;
        state0 = gold_add(gold_add(gold_mul(m0, t0), gold_mul(m1, t1)), gold_mul(m2, t2));
        state1 = gold_add(gold_add(gold_mul(m3, t0), gold_mul(m4, t1)), gold_mul(m5, t2));
        state2 = gold_add(gold_add(gold_mul(m6, t0), gold_mul(m7, t1)), gold_mul(m8, t2));
    }

    tree[out_offset + p] = state0;
}