#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;

constexpr constant uint T_FIXED  = 3u;
constexpr constant uint POS2_R_F = 8u;
constexpr constant uint POS2_R_P = 22u;

// Canonical add: inputs in [0, p), output in [0, p).
inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    // If wrap occurred OR t >= p, subtract p.
    bool wrap = (t < a);
    ulong adj = wrap ? EPSILON : 0ul;   // adding EPSILON = subtracting p mod 2^64
    t += adj;
    if (t >= P_GOLD) t -= P_GOLD;
    return t;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    if (b > a) t -= EPSILON;
    return t;
}

// 64x64 -> reduce mod p. Tighter reduction path.
inline ulong gold_mul(ulong a, ulong b) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    ulong p00 = (ulong)a0 * (ulong)b0;
    ulong p01 = (ulong)a0 * (ulong)b1;
    ulong p10 = (ulong)a1 * (ulong)b0;
    ulong p11 = (ulong)a1 * (ulong)b1;

    ulong mid = (p00 >> 32) + (p01 & EPSILON) + (p10 & EPSILON);
    ulong lo  = (p00 & EPSILON) | (mid << 32);
    ulong hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);

    // Reduce: x = lo + hi * 2^64; 2^64 ≡ 2^32 - 1 (mod p).
    // So x ≡ lo + hi*2^32 - hi (mod p).
    // Split hi = hi_hi*2^32 + hi_lo.
    // hi*2^32 - hi = hi_lo*2^64 + hi_hi*2^32*2^32 - hi
    //             ≡ hi_lo*(2^32-1) + hi_hi*(2^32-1)*2^32 - hi ... messy.
    // Use standard form: result = lo - hi_hi + hi_lo*(2^32 - 1), all mod p, handling borrows/carries.
    uint hi_hi = (uint)(hi >> 32);
    ulong hi_lo = hi & EPSILON;

    // t0 = lo - hi_hi (mod p), via 2^64 path.
    ulong t0 = lo - (ulong)hi_hi;
    if (hi_hi > (uint)(lo >> 32) || (hi_hi == 0 && false)) {
        // borrow check: lo < hi_hi only if upper word zero and lower < hi_hi, generally use t0 > lo
    }
    if (t0 > lo) t0 -= EPSILON;

    // Add hi_lo * (2^32 - 1) = (hi_lo << 32) - hi_lo.
    ulong t1 = (hi_lo << 32) - hi_lo;

    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;
    if (t2 >= P_GOLD) t2 -= P_GOLD;
    return t2;
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
        // r0 = ax (zeta_inv_pow[0]=1), r1 = ax * zeta_inv_pow[1].
        // S_m = 1 + r_m.
        ulong e0 = evals_in[j];
        ulong e1 = evals_in[j + N];
        ulong z1 = zeta_inv_pow[1];

        ulong r1 = gold_mul(ax, z1);
        ulong s0 = gold_add(1ul, ax);
        ulong s1 = gold_add(1ul, r1);
        ulong acc = gold_add(gold_mul(e0, s0), gold_mul(e1, s1));
        evals_out[j] = gold_mul(acc, inv_fold);
        return;
    }

    if (F == 4u) {
        ulong z1 = zeta_inv_pow[1];
        ulong z2 = zeta_inv_pow[2];
        ulong z3 = zeta_inv_pow[3];

        // Prefetch evals.
        ulong e0 = evals_in[j];
        ulong e1 = evals_in[j + N];
        ulong e2 = evals_in[j + 2u * N];
        ulong e3 = evals_in[j + 3u * N];

        ulong r0 = ax;
        ulong r1 = gold_mul(ax, z1);
        ulong r2 = gold_mul(ax, z2);
        ulong r3 = gold_mul(ax, z3);

        // S_m = (1+r_m)(1+r_m^2).
        ulong r0_2 = gold_mul(r0, r0);
        ulong r1_2 = gold_mul(r1, r1);
        ulong r2_2 = gold_mul(r2, r2);
        ulong r3_2 = gold_mul(r3, r3);

        ulong s0 = gold_mul(gold_add(1ul, r0), gold_add(1ul, r0_2));
        ulong s1 = gold_mul(gold_add(1ul, r1), gold_add(1ul, r1_2));
        ulong s2 = gold_mul(gold_add(1ul, r2), gold_add(1ul, r2_2));
        ulong s3 = gold_mul(gold_add(1ul, r3), gold_add(1ul, r3_2));

        ulong acc = gold_add(gold_mul(e0, s0), gold_mul(e1, s1));
        acc = gold_add(acc, gold_mul(e2, s2));
        acc = gold_add(acc, gold_mul(e3, s3));
        evals_out[j] = gold_mul(acc, inv_fold);
        return;
    }

    // Generic fallback.
    ulong acc = 0ul;
    for (uint m = 0u; m < F; ++m) {
        ulong rm = (m == 0u) ? ax : gold_mul(ax, zeta_inv_pow[m]);
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
// Poseidon2-t=3
// ----------------------------------------------------------------------

inline ulong sbox7(ulong x) {
    ulong x2 = gold_mul(x, x);
    ulong x4 = gold_mul(x2, x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
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

    // Load MDS / diag into registers once.
    ulong m00 = ext_mds[0], m01 = ext_mds[1], m02 = ext_mds[2];
    ulong m10 = ext_mds[3], m11 = ext_mds[4], m12 = ext_mds[5];
    ulong m20 = ext_mds[6], m21 = ext_mds[7], m22 = ext_mds[8];
    ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2];

    uint base = p << 1u;
    ulong s0 = tree[in_offset + base];
    ulong s1 = (base + 1u < child_count) ? tree[in_offset + base + 1u] : 0ul;
    ulong s2 = 0ul;

    // Initial external mat.
    {
        ulong t0 = gold_add(gold_add(gold_mul(m00, s0), gold_mul(m01, s1)), gold_mul(m02, s2));
        ulong t1 = gold_add(gold_add(gold_mul(m10, s0), gold_mul(m11, s1)), gold_mul(m12, s2));
        ulong t2 = gold_add(gold_add(gold_mul(m20, s0), gold_mul(m21, s1)), gold_mul(m22, s2));
        s0 = t0; s1 = t1; s2 = t2;
    }

    // First half full rounds.
    for (uint r = 0u; r < POS2_R_F / 2u; ++r) {
        s0 = sbox7(gold_add(s0, rc_ext[r * 3u + 0u]));
        s1 = sbox7(gold_add(s1, rc_ext[r * 3u + 1u]));
        s2 = sbox7(gold_add(s2, rc_ext[r * 3u + 2u]));
        ulong t0 = gold_add(gold_add(gold_mul(m00, s0), gold_mul(m01, s1)), gold_mul(m02, s2));
        ulong t1 = gold_add(gold_add(gold_mul(m10, s0), gold_mul(m11, s1)), gold_mul(m12, s2));
        ulong t2 = gold_add(gold_add(gold_mul(m20, s0), gold_mul(m21, s1)), gold_mul(m22, s2));
        s0 = t0; s1 = t1; s2 = t2;
    }

    // Partial rounds.
    for (uint r = 0u; r < POS2_R_P; ++r) {
        s0 = sbox7(gold_add(s0, rc_int[r]));
        ulong s = gold_add(gold_add(s0, s1), s2);
        ulong t0 = gold_add(s, gold_mul(d0, s0));
        ulong t1 = gold_add(s, gold_mul(d1, s1));
        ulong t2 = gold_add(s, gold_mul(d2, s2));
        s0 = t0; s1 = t1; s2 = t2;
    }

    // Second half full rounds.
    for (uint r = POS2_R_F / 2u; r < POS2_R_F; ++r) {
        s0 = sbox7(gold_add(s0, rc_ext[r * 3u + 0u]));
        s1 = sbox7(gold_add(s1, rc_ext[r * 3u + 1u]));
        s2 = sbox7(gold_add(s2, rc_ext[r * 3u + 2u]));
        ulong t0 = gold_add(gold_add(gold_mul(m00, s0), gold_mul(m01, s1)), gold_mul(m02, s2));
        ulong t1 = gold_add(gold_add(gold_mul(m10, s0), gold_mul(m11, s1)), gold_mul(m12, s2));
        ulong t2 = gold_add(gold_add(gold_mul(m20, s0), gold_mul(m21, s1)), gold_mul(m22, s2));
        s0 = t0; s1 = t1; s2 = t2;
    }

    tree[out_offset + p] = s0;
}