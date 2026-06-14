#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;   // 2^64 - 2^32 + 1
constant ulong EPSILON = 0x00000000FFFFFFFFul;   // 2^32 - 1

constexpr constant uint T_FIXED  = 3u;
constexpr constant uint POS2_R_F = 8u;
constexpr constant uint POS2_R_P = 22u;

// ----------------------------------------------------------------------
// Goldilocks arithmetic — fast reduction using 2^64 ≡ 2^32 - 1 (mod p)
// ----------------------------------------------------------------------

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    if (t < a) t += EPSILON;   // wrapped: add 2^64 mod p = EPSILON
    if (t >= P_GOLD) t -= P_GOLD;
    return t;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    if (t > a) t -= EPSILON;
    return t;
}

// Reduce a 128-bit value (lo, hi) mod p, where p = 2^64 - 2^32 + 1.
// Split hi = hi_hi * 2^32 + hi_lo.
//   (lo, hi) = lo + hi * 2^64
//            ≡ lo + hi*(2^32 - 1)  (since 2^64 ≡ 2^32 - 1)
//            = lo + hi_hi*2^96 + hi_lo*2^64 - hi_hi*2^64 - hi_lo*2^32 ... 
// Standard plonky2-style reduction:
//   res = lo - hi_hi  (with EPSILON correction on borrow)
//   res += hi_lo * EPSILON  (with EPSILON correction on carry)
inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong x_hi_lo = x_hi & EPSILON;
    ulong x_hi_hi = x_hi >> 32;

    ulong t0 = x_lo - x_hi_hi;
    if (t0 > x_lo) t0 -= EPSILON;

    // hi_lo * EPSILON = hi_lo * (2^32 - 1) = (hi_lo << 32) - hi_lo.
    // hi_lo fits in 32 bits, so (hi_lo << 32) - hi_lo fits in 64 bits exactly.
    ulong t1 = (x_hi_lo << 32) - x_hi_lo;

    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;

    return gold_canonical(t2);
}

// 64x64 -> 128 multiply using Metal's mulhi/mul for 32-bit halves.
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

inline ulong gold_mul(ulong a, ulong b) {
    ulong2 prod = umul128(a, b);
    return gold_reduce128(prod.x, prod.y);
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

    ulong ax = gold_mul(alpha, inv_x_base[j]);

    // Cache zeta_inv_pow[m] in registers (fold <= 4).
    ulong zpow[4];
    uint F = fold;
    for (uint m = 0u; m < F; ++m) zpow[m] = zeta_inv_pow[m];

    ulong acc = 0ul;
    for (uint m = 0u; m < F; ++m) {
        ulong rm = gold_mul(ax, zpow[m]);

        // S_m = sum_{p=0..F-1} rm^p  (Horner-style: 1 + rm*(1 + rm*(1 + ...)))
        ulong sm = 1ul;
        for (uint p = 1u; p < F; ++p) {
            sm = gold_add(1ul, gold_mul(sm, rm));
        }

        uint src = j + m * n_out;
        acc = gold_add(acc, gold_mul(evals_in[src], sm));
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

inline void matvec_ext_t3(thread ulong *state,
                          thread const ulong *mds)
{
    ulong s0 = state[0], s1 = state[1], s2 = state[2];
    ulong t0 = gold_add(gold_add(gold_mul(mds[0], s0), gold_mul(mds[1], s1)), gold_mul(mds[2], s2));
    ulong t1 = gold_add(gold_add(gold_mul(mds[3], s0), gold_mul(mds[4], s1)), gold_mul(mds[5], s2));
    ulong t2 = gold_add(gold_add(gold_mul(mds[6], s0), gold_mul(mds[7], s1)), gold_mul(mds[8], s2));
    state[0] = t0; state[1] = t1; state[2] = t2;
}

inline void matvec_int_t3(thread ulong *state,
                          thread const ulong *diag)
{
    ulong s0 = state[0], s1 = state[1], s2 = state[2];
    ulong s  = gold_add(gold_add(s0, s1), s2);
    state[0] = gold_add(s, gold_mul(diag[0], s0));
    state[1] = gold_add(s, gold_mul(diag[1], s1));
    state[2] = gold_add(s, gold_mul(diag[2], s2));
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

    // Load MDS and diag into thread registers (small: 9 + 3 ulongs).
    ulong mds[9];
    for (uint i = 0u; i < 9u; ++i) mds[i] = ext_mds[i];
    ulong diag[3];
    diag[0] = int_diag[0]; diag[1] = int_diag[1]; diag[2] = int_diag[2];

    thread ulong state[T_FIXED];
    state[0] = 0ul; state[1] = 0ul; state[2] = 0ul;

    uint base = p << 1u;
    state[0] = tree[in_offset + base];
    if (base + 1u < child_count) {
        state[1] = tree[in_offset + base + 1u];
    }

    // Initial external matrix multiplication.
    matvec_ext_t3(state, mds);

    // First half full rounds.
    for (uint r = 0u; r < POS2_R_F / 2u; ++r) {
        state[0] = sbox7(gold_add(state[0], rc_ext[r * T_FIXED + 0u]));
        state[1] = sbox7(gold_add(state[1], rc_ext[r * T_FIXED + 1u]));
        state[2] = sbox7(gold_add(state[2], rc_ext[r * T_FIXED + 2u]));
        matvec_ext_t3(state, mds);
    }

    // Partial rounds.
    for (uint r = 0u; r < POS2_R_P; ++r) {
        state[0] = sbox7(gold_add(state[0], rc_int[r]));
        matvec_int_t3(state, diag);
    }

    // Second half full rounds.
    for (uint r = POS2_R_F / 2u; r < POS2_R_F; ++r) {
        state[0] = sbox7(gold_add(state[0], rc_ext[r * T_FIXED + 0u]));
        state[1] = sbox7(gold_add(state[1], rc_ext[r * T_FIXED + 1u]));
        state[2] = sbox7(gold_add(state[2], rc_ext[r * T_FIXED + 2u]));
        matvec_ext_t3(state, mds);
    }

    tree[out_offset + p] = state[0];
}