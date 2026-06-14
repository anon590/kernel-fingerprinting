#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;

constexpr constant uint POS2_R_F = 8u;
constexpr constant uint POS2_R_P = 22u;

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

inline ulong2 umul128(ulong a, ulong b) {
    uint a0 = (uint)(a);
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)(b);
    uint b1 = (uint)(b >> 32);

    ulong p00 = (ulong)a0 * (ulong)b0;
    ulong p01 = (ulong)a0 * (ulong)b1;
    ulong p10 = (ulong)a1 * (ulong)b0;
    ulong p11 = (ulong)a1 * (ulong)b1;

    ulong mid = (p00 >> 32) + (p01 & EPSILON) + (p10 & EPSILON);
    ulong lo  = (p00 & EPSILON) | (mid << 32);
    ulong hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return ulong2(lo, hi);
}

// Reduce (lo, hi) mod p = 2^64 - 2^32 + 1.
// x = lo + hi_lo*2^32 + hi_hi*2^64
//   ≡ lo + hi_lo*2^32 + hi_hi*(2^32 - 1)   (mod p)
//   = lo + (hi << 32) - hi_hi              (since hi*2^32 = hi_lo*2^32 + hi_hi*2^64,
//                                                 and 2^64 ≡ 2^32-1, so hi_hi*2^64 ≡ hi_hi*(2^32-1))
// Actually: hi*(2^32-1) wrapped + lo, equivalent to lo + (hi_lo*EPSILON) - hi_hi.
inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong x_hi_lo = x_hi & EPSILON;
    ulong x_hi_hi = x_hi >> 32;

    // Step 1: subtract hi_hi from lo (mod 2^64), with single fixup.
    ulong t0 = x_lo - x_hi_hi;
    if (t0 > x_lo) t0 -= EPSILON;

    // Step 2: add hi_lo*EPSILON = (hi_lo<<32) - hi_lo, both fit in u64.
    ulong t1 = (x_hi_lo << 32) - x_hi_lo;

    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;

    return gold_canonical(t2);
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong2 prod = umul128(a, b);
    return gold_reduce128(prod.x, prod.y);
}

inline ulong sbox7(ulong x) {
    ulong x2 = gold_mul(x, x);
    ulong x4 = gold_mul(x2, x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
}

// ----------------------------------------------------------------------
// FRI fold
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

    uint F = fold;
    uint N = n_out;
    ulong ax = gold_mul(alpha, inv_x_base[j]);

    if (F == 2u) {
        ulong z1 = zeta_inv_pow[1];
        ulong e0 = evals_in[j];
        ulong e1 = evals_in[j + N];
        ulong r0 = ax;
        ulong r1 = gold_mul(ax, z1);
        ulong s0 = gold_add(1ul, r0);
        ulong s1 = gold_add(1ul, r1);
        ulong acc = gold_add(gold_mul(e0, s0), gold_mul(e1, s1));
        evals_out[j] = gold_mul(acc, inv_fold);
        return;
    }

    if (F == 4u) {
        ulong z1 = zeta_inv_pow[1];
        ulong z2 = zeta_inv_pow[2];
        ulong z3 = zeta_inv_pow[3];

        ulong e0 = evals_in[j];
        ulong e1 = evals_in[j + N];
        ulong e2 = evals_in[j + 2u * N];
        ulong e3 = evals_in[j + 3u * N];

        ulong r0 = ax;
        ulong r1 = gold_mul(ax, z1);
        ulong r2 = gold_mul(ax, z2);
        ulong r3 = gold_mul(ax, z3);

        ulong r0_2 = gold_mul(r0, r0);
        ulong r1_2 = gold_mul(r1, r1);
        ulong r2_2 = gold_mul(r2, r2);
        ulong r3_2 = gold_mul(r3, r3);

        ulong s0 = gold_mul(gold_add(1ul, r0), gold_add(1ul, r0_2));
        ulong s1 = gold_mul(gold_add(1ul, r1), gold_add(1ul, r1_2));
        ulong s2 = gold_mul(gold_add(1ul, r2), gold_add(1ul, r2_2));
        ulong s3 = gold_mul(gold_add(1ul, r3), gold_add(1ul, r3_2));

        ulong a01 = gold_add(gold_mul(e0, s0), gold_mul(e1, s1));
        ulong a23 = gold_add(gold_mul(e2, s2), gold_mul(e3, s3));
        ulong acc = gold_add(a01, a23);
        evals_out[j] = gold_mul(acc, inv_fold);
        return;
    }

    // Generic fallback.
    ulong acc = 0ul;
    for (uint m = 0u; m < F; ++m) {
        ulong rm = gold_mul(ax, zeta_inv_pow[m]);
        ulong sm   = 0ul;
        ulong rpow = 1ul;
        for (uint pp = 0u; pp < F; ++pp) {
            sm = gold_add(sm, rpow);
            rpow = gold_mul(rpow, rm);
        }
        acc = gold_add(acc, gold_mul(evals_in[j + m * N], sm));
    }
    evals_out[j] = gold_mul(acc, inv_fold);
}

// ----------------------------------------------------------------------
// Poseidon2-t=3 Merkle commit
// ----------------------------------------------------------------------

inline void matvec_ext_t3_generic(thread ulong *state,
                                  device const ulong *ext_mds)
{
    ulong t0 = gold_add(gold_add(gold_mul(ext_mds[0], state[0]),
                                 gold_mul(ext_mds[1], state[1])),
                        gold_mul(ext_mds[2], state[2]));
    ulong t1 = gold_add(gold_add(gold_mul(ext_mds[3], state[0]),
                                 gold_mul(ext_mds[4], state[1])),
                        gold_mul(ext_mds[5], state[2]));
    ulong t2 = gold_add(gold_add(gold_mul(ext_mds[6], state[0]),
                                 gold_mul(ext_mds[7], state[1])),
                        gold_mul(ext_mds[8], state[2]));
    state[0] = t0; state[1] = t1; state[2] = t2;
}

inline void matvec_ext_t3_fused(thread ulong *state)
{
    ulong s = gold_add(gold_add(state[0], state[1]), state[2]);
    state[0] = gold_add(s, state[0]);
    state[1] = gold_add(s, state[1]);
    state[2] = gold_add(s, state[2]);
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

    ulong m00 = ext_mds[0];
    ulong m01 = ext_mds[1];
    ulong m02 = ext_mds[2];
    bool fused = (m00 == 2ul) && (m01 == 1ul) && (m02 == 1ul);

    ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2];

    uint base = p << 1u;
    ulong s0 = tree[in_offset + base];
    ulong s1 = (base + 1u < child_count) ? tree[in_offset + base + 1u] : 0ul;

    thread ulong st[3];
    st[0] = s0; st[1] = s1; st[2] = 0ul;

    if (fused) {
        matvec_ext_t3_fused(st);
    } else {
        matvec_ext_t3_generic(st, ext_mds);
    }

    for (uint r = 0u; r < POS2_R_F / 2u; ++r) {
        ulong c0 = rc_ext[r * 3u + 0u];
        ulong c1 = rc_ext[r * 3u + 1u];
        ulong c2 = rc_ext[r * 3u + 2u];
        st[0] = sbox7(gold_add(st[0], c0));
        st[1] = sbox7(gold_add(st[1], c1));
        st[2] = sbox7(gold_add(st[2], c2));
        if (fused) {
            matvec_ext_t3_fused(st);
        } else {
            matvec_ext_t3_generic(st, ext_mds);
        }
    }

    for (uint r = 0u; r < POS2_R_P; ++r) {
        st[0] = sbox7(gold_add(st[0], rc_int[r]));
        ulong s = gold_add(gold_add(st[0], st[1]), st[2]);
        ulong t0 = gold_add(s, gold_mul(d0, st[0]));
        ulong t1 = gold_add(s, gold_mul(d1, st[1]));
        ulong t2 = gold_add(s, gold_mul(d2, st[2]));
        st[0] = t0; st[1] = t1; st[2] = t2;
    }

    for (uint r = POS2_R_F / 2u; r < POS2_R_F; ++r) {
        ulong c0 = rc_ext[r * 3u + 0u];
        ulong c1 = rc_ext[r * 3u + 1u];
        ulong c2 = rc_ext[r * 3u + 2u];
        st[0] = sbox7(gold_add(st[0], c0));
        st[1] = sbox7(gold_add(st[1], c1));
        st[2] = sbox7(gold_add(st[2], c2));
        if (fused) {
            matvec_ext_t3_fused(st);
        } else {
            matvec_ext_t3_generic(st, ext_mds);
        }
    }

    tree[out_offset + p] = st[0];
}