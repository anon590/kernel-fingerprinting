#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;   // 2^64 - 2^32 + 1
constant ulong EPSILON = 0x00000000FFFFFFFFul;   // 2^32 - 1

// ----------------------------------------------------------------------
// Branchless Goldilocks Arithmetic
// ----------------------------------------------------------------------

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

inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong h0 = x_hi & EPSILON;
    ulong h1 = x_hi >> 32;

    ulong t0 = x_lo - h1;
    t0 -= (t0 > x_lo) ? EPSILON : 0ul;

    ulong t1 = (h0 << 32) - h0;

    ulong t2 = t0 + t1;
    t2 += (t2 < t0) ? EPSILON : 0ul;

    return (t2 >= P_GOLD) ? (t2 - P_GOLD) : t2;
}

inline ulong gold_mul(ulong a, ulong b) {
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
    
    return gold_reduce128(lo, hi);
}

inline ulong gold_sqr(ulong a) {
    return gold_mul(a, a);
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
        ulong z3 = zeta_inv_pow[3];

        ulong rm0 = ax;
        ulong rm1 = gold_mul(ax, z1);
        ulong rm2 = gold_mul(ax, z2);
        ulong rm3 = gold_mul(ax, z3);

        ulong ax_sq    = gold_sqr(ax);
        ulong ax_sq_z2 = gold_mul(ax_sq, z2);

        ulong E0 = evals_in[j];
        ulong E1 = evals_in[j + n_out];
        ulong E2 = evals_in[j + 2u * n_out];
        ulong E3 = evals_in[j + 3u * n_out];

        ulong part0 = gold_add(gold_mul(E0, gold_add(1ul, rm0)), gold_mul(E2, gold_add(1ul, rm2)));
        ulong part1 = gold_add(gold_mul(E1, gold_add(1ul, rm1)), gold_mul(E3, gold_add(1ul, rm3)));

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
// Poseidon2-t=3 (Fully register-allocated & unrolled)
// ----------------------------------------------------------------------

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
        ulong2 loaded = ((device const ulong2*)(tree + in_offset))[p];
        s0 = loaded.x;
        s1 = loaded.y;
    } else if (base < child_count) {
        s0 = tree[in_offset + base];
    }

    // Load MDS matrix elements to registers
    const ulong m0 = ext_mds[0], m1 = ext_mds[1], m2 = ext_mds[2];
    const ulong m3 = ext_mds[3], m4 = ext_mds[4], m5 = ext_mds[5];
    const ulong m6 = ext_mds[6], m7 = ext_mds[7], m8 = ext_mds[8];

    const ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2];

    // Initial MATVEC_EXT
    {
        ulong t0 = gold_add(gold_add(gold_mul(m0, s0), gold_mul(m1, s1)), gold_mul(m2, s2));
        ulong t1 = gold_add(gold_add(gold_mul(m3, s0), gold_mul(m4, s1)), gold_mul(m5, s2));
        ulong t2 = gold_add(gold_add(gold_mul(m6, s0), gold_mul(m7, s1)), gold_mul(m8, s2));
        s0 = t0; s1 = t1; s2 = t2;
    }

    // First half full rounds (4 rounds)
    #pragma unroll
    for (uint r = 0u; r < 4u; ++r) {
        s0 = sbox7(gold_add(s0, rc_ext[r * 3u + 0u]));
        s1 = sbox7(gold_add(s1, rc_ext[r * 3u + 1u]));
        s2 = sbox7(gold_add(s2, rc_ext[r * 3u + 2u]));
        
        ulong t0 = gold_add(gold_add(gold_mul(m0, s0), gold_mul(m1, s1)), gold_mul(m2, s2));
        ulong t1 = gold_add(gold_add(gold_mul(m3, s0), gold_mul(m4, s1)), gold_mul(m5, s2));
        ulong t2 = gold_add(gold_add(gold_mul(m6, s0), gold_mul(m7, s1)), gold_mul(m8, s2));
        s0 = t0; s1 = t1; s2 = t2;
    }

    // Partial rounds (22 rounds)
    #pragma unroll
    for (uint r = 0u; r < 22u; ++r) {
        s0 = sbox7(gold_add(s0, rc_int[r]));
        
        ulong sum_s = gold_add(gold_add(s0, s1), s2);
        s0 = gold_add(sum_s, gold_mul(d0, s0));
        s1 = gold_add(sum_s, gold_mul(d1, s1));
        s2 = gold_add(sum_s, gold_mul(d2, s2));
    }

    // Second half full rounds (4 rounds)
    #pragma unroll
    for (uint r = 4u; r < 8u; ++r) {
        s0 = sbox7(gold_add(s0, rc_ext[r * 3u + 0u]));
        s1 = sbox7(gold_add(s1, rc_ext[r * 3u + 1u]));
        s2 = sbox7(gold_add(s2, rc_ext[r * 3u + 2u]));
        
        ulong t0 = gold_add(gold_add(gold_mul(m0, s0), gold_mul(m1, s1)), gold_mul(m2, s2));
        ulong t1 = gold_add(gold_add(gold_mul(m3, s0), gold_mul(m4, s1)), gold_mul(m5, s2));
        ulong t2 = gold_add(gold_add(gold_mul(m6, s0), gold_mul(m7, s1)), gold_mul(m8, s2));
        s0 = t0; s1 = t1; s2 = t2;
    }

    tree[out_offset + p] = s0;
}