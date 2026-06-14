#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    if (t < a) t += 0xFFFFFFFFul;
    if (t >= P_GOLD) t -= P_GOLD;
    return t;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    if (t > a) t -= 0xFFFFFFFFul;
    return t;
}

inline ulong gold_mul(ulong a, ulong b) {
    uint a0 = (uint)a, a1 = (uint)(a >> 32);
    uint b0 = (uint)b, b1 = (uint)(b >> 32);

    ulong p00 = (ulong)a0 * b0;
    ulong p01 = (ulong)a0 * b1;
    ulong p10 = (ulong)a1 * b0;
    ulong p11 = (ulong)a1 * b1;

    ulong mid = (p00 >> 32) + (uint)p01 + (uint)p10;
    ulong x_lo = (p00 & 0xFFFFFFFFul) | (mid << 32);
    ulong x_hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);

    ulong x_hi_lo = x_hi & 0xFFFFFFFFul;
    ulong x_hi_hi = x_hi >> 32;

    ulong t0 = x_lo - x_hi_hi;
    if (t0 > x_lo) t0 -= 0xFFFFFFFFul;

    ulong t1 = (x_hi_lo << 32) - x_hi_lo;

    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += 0xFFFFFFFFul;

    return (t2 >= P_GOLD) ? (t2 - P_GOLD) : t2;
}

inline ulong sbox7(ulong x) {
    ulong x2 = gold_mul(x, x);
    ulong x3 = gold_mul(x2, x);
    ulong x6 = gold_mul(x3, x3);
    return gold_mul(x6, x);
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

    if (fold == 2) {
        ulong r0 = ax;
        ulong S0 = gold_add(1ul, r0);
        ulong S1 = gold_sub(1ul, r0);
        
        ulong e0 = evals_in[j];
        ulong e1 = evals_in[j + n_out];
        
        acc = gold_add(gold_mul(e0, S0), gold_mul(e1, S1));
    } else if (fold == 4) {
        ulong r0 = ax;
        ulong r1 = gold_mul(ax, zeta_inv_pow[1]);
        
        ulong r0_2 = gold_mul(r0, r0);
        ulong r1_2 = gold_mul(r1, r1);
        
        ulong T0 = gold_add(1ul, r0_2);
        ulong T1 = gold_add(1ul, r1_2);
        
        ulong U0 = gold_mul(r0, T0);
        ulong U1 = gold_mul(r1, T1);
        
        ulong S0 = gold_add(T0, U0);
        ulong S2 = gold_sub(T0, U0);
        ulong S1 = gold_add(T1, U1);
        ulong S3 = gold_sub(T1, U1);
        
        ulong e0 = evals_in[j];
        ulong e1 = evals_in[j + n_out];
        ulong e2 = evals_in[j + 2*n_out];
        ulong e3 = evals_in[j + 3*n_out];
        
        acc = gold_add(gold_mul(e0, S0), gold_mul(e1, S1));
        acc = gold_add(acc, gold_mul(e2, S2));
        acc = gold_add(acc, gold_mul(e3, S3));
    } else {
        for (uint m = 0u; m < fold; ++m) {
            ulong rm = gold_mul(ax, zeta_inv_pow[m]);
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

    ulong mds[9];
    #pragma unroll
    for (int i = 0; i < 9; ++i) mds[i] = ext_mds[i];

    ulong diag[3];
    #pragma unroll
    for (int i = 0; i < 3; ++i) diag[i] = int_diag[i];

    ulong state[3];
    state[0] = 0ul; state[1] = 0ul; state[2] = 0ul;

    uint base = p << 1u;
    state[0] = tree[in_offset + base];
    if (base + 1u < child_count) {
        state[1] = tree[in_offset + base + 1u];
    }

    {
        ulong s0 = state[0], s1 = state[1], s2 = state[2];
        state[0] = gold_add(gold_add(gold_mul(mds[0], s0), gold_mul(mds[1], s1)), gold_mul(mds[2], s2));
        state[1] = gold_add(gold_add(gold_mul(mds[3], s0), gold_mul(mds[4], s1)), gold_mul(mds[5], s2));
        state[2] = gold_add(gold_add(gold_mul(mds[6], s0), gold_mul(mds[7], s1)), gold_mul(mds[8], s2));
    }

    #pragma unroll
    for (uint r = 0u; r < 4u; ++r) {
        state[0] = sbox7(gold_add(state[0], rc_ext[r * 3 + 0]));
        state[1] = sbox7(gold_add(state[1], rc_ext[r * 3 + 1]));
        state[2] = sbox7(gold_add(state[2], rc_ext[r * 3 + 2]));
        
        ulong s0 = state[0], s1 = state[1], s2 = state[2];
        state[0] = gold_add(gold_add(gold_mul(mds[0], s0), gold_mul(mds[1], s1)), gold_mul(mds[2], s2));
        state[1] = gold_add(gold_add(gold_mul(mds[3], s0), gold_mul(mds[4], s1)), gold_mul(mds[5], s2));
        state[2] = gold_add(gold_add(gold_mul(mds[6], s0), gold_mul(mds[7], s1)), gold_mul(mds[8], s2));
    }

    #pragma unroll
    for (uint r = 0u; r < 22u; ++r) {
        state[0] = sbox7(gold_add(state[0], rc_int[r]));
        
        ulong s = gold_add(gold_add(state[0], state[1]), state[2]);
        state[0] = gold_add(s, gold_mul(diag[0], state[0]));
        state[1] = gold_add(s, gold_mul(diag[1], state[1]));
        state[2] = gold_add(s, gold_mul(diag[2], state[2]));
    }

    #pragma unroll
    for (uint r = 4u; r < 8u; ++r) {
        state[0] = sbox7(gold_add(state[0], rc_ext[r * 3 + 0]));
        state[1] = sbox7(gold_add(state[1], rc_ext[r * 3 + 1]));
        state[2] = sbox7(gold_add(state[2], rc_ext[r * 3 + 2]));
        
        ulong s0 = state[0], s1 = state[1], s2 = state[2];
        state[0] = gold_add(gold_add(gold_mul(mds[0], s0), gold_mul(mds[1], s1)), gold_mul(mds[2], s2));
        state[1] = gold_add(gold_add(gold_mul(mds[3], s0), gold_mul(mds[4], s1)), gold_mul(mds[5], s2));
        state[2] = gold_add(gold_add(gold_mul(mds[6], s0), gold_mul(mds[7], s1)), gold_mul(mds[8], s2));
    }

    tree[out_offset + p] = state[0];
}