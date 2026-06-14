An elegant, structurally superior optimization has been engineered for both `fri_fold` and `fri_commit_level` to break out of the performance plateau:

1. **Squaring-Specific Arithmetic (`gold_sqr`)**: A specialized 128-bit squaring helper (`usqr128`) reduces general-purpose multiplications from 4 to 3 by exploiting symmetry. This is integrated into `sbox7` (reducing its complexity to 2 multiplications and 2 squarings) and the `fold == 4u` algebra.
2. **Optimal Register Load Trade-off**: In `fri_commit_level`, MDS and diagonal coefficients are loaded into registers once using vectorized 256-bit (`ulong4`) and 128-bit (`ulong2`) memory operations to eliminate 147 device-memory reads. In contrast, round constants are read on the fly to conserve register files and avoid spilling.
3. **Coalesced Vectorized Tree Reads**: Merkle tree sibling pairs are read as single 16-byte `ulong2` loads.
4. **Compiler-Friendly Goldilocks primitives**: Reverted ternary operations back to clean `if` statements, allowing the MSL compiler to perfectly match single-cycle hardware select instructions (`csel`).

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;   // 2^64 - 2^32 + 1
constant ulong EPSILON = 0x00000000FFFFFFFFul;   // 2^32 - 1

constexpr constant uint T_FIXED  = 3u;           // Poseidon2 width
constexpr constant uint POS2_R_F = 8u;           // Poseidon2 full rounds (4+4)
constexpr constant uint POS2_R_P = 22u;          // Poseidon2 partial rounds

// ----------------------------------------------------------------------
// Branchless-compatible Goldilocks Arithmetic
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

inline ulong2 umul128(ulong a, ulong b) {
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
    return ulong2(lo, hi);
}

inline ulong2 usqr128(ulong a) {
    uint a0 = (uint)(a);
    uint a1 = (uint)(a >> 32);

    ulong p00 = (ulong)a0 * a0;
    ulong p01 = (ulong)a0 * a1;
    ulong p11 = (ulong)a1 * a1;

    ulong mid = (p00 >> 32) + (uint)p01 + (uint)p01;
    ulong lo  = (uint)p00 | (mid << 32);
    ulong p01_hi = p01 >> 32;
    ulong hi  = p11 + p01_hi + p01_hi + (mid >> 32);
    return ulong2(lo, hi);
}

inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong x_hi_lo = x_hi & EPSILON;
    ulong x_hi_hi = x_hi >> 32;

    ulong t0 = x_lo - x_hi_hi;
    if (t0 > x_lo) t0 -= EPSILON;

    ulong t1 = (x_hi_lo << 32) - x_hi_lo;

    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;

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

// Parallelized, sbox7 optimized for register-to-register latency
inline ulong sbox7(ulong x) {
    ulong x2 = gold_sqr(x);
    ulong x4 = gold_sqr(x2);
    ulong x3 = gold_mul(x2, x);
    return gold_mul(x4, x3);
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

    if (fold == 2u) {
        ulong E0 = evals_in[j];
        ulong E1 = evals_in[j + n_out];

        ulong E0_plus_E1  = gold_add(E0, E1);
        ulong E0_minus_E1 = gold_sub(E0, E1);

        ulong acc = gold_add(E0_plus_E1, gold_mul(ax, E0_minus_E1));
        evals_out[j] = gold_mul(acc, inv_fold);
    } else if (fold == 4u) {
        ulong E0 = evals_in[j];
        ulong E1 = evals_in[j + n_out];
        ulong E2 = evals_in[j + 2u * n_out];
        ulong E3 = evals_in[j + 3u * n_out];

        ulong z1 = zeta_inv_pow[1];

        ulong rm1 = gold_mul(ax, z1);
        ulong ax_sq = gold_sqr(ax);

        ulong ax_sq_plus1     = gold_add(1ul, ax_sq);
        ulong ax_sq_neg_plus1 = gold_sub(1ul, ax_sq);

        ulong E0_plus_E2  = gold_add(E0, E2);
        ulong E0_minus_E2 = gold_sub(E0, E2);
        ulong E1_plus_E3  = gold_add(E1, E3);
        ulong E1_minus_E3 = gold_sub(E1, E3);

        ulong part0 = gold_add(E0_plus_E2, gold_mul(ax, E0_minus_E2));
        ulong part1 = gold_add(E1_plus_E3, gold_mul(rm1, E1_minus_E3));

        ulong term0_2 = gold_mul(part0, ax_sq_plus1);
        ulong term1_3 = gold_mul(part1, ax_sq_neg_plus1);

        ulong acc = gold_add(term0_2, term1_3);
        evals_out[j] = gold_mul(acc, inv_fold);
    } else {
        ulong acc = 0ul;
        for (uint m = 0u; m < fold; ++m) {
            ulong rm = gold_mul(ax, zeta_inv_pow[m]);
            ulong sm = 0ul;
            ulong rpow = 1ul;
            for (uint p = 0u; p < fold; ++p) {
                sm   = gold_add(sm, rpow);
                rpow = gold_mul(rpow, rm);
            }
            acc = gold_add(acc, gold_mul(evals_in[j + m * n_out], sm));
        }
        evals_out[j] = gold_mul(acc, inv_fold);
    }
}

// ----------------------------------------------------------------------
// Poseidon2-t=3 Merkle Commit
// ----------------------------------------------------------------------

#define MATVEC_EXT(s0, s1, s2, m0, m1, m2, m3, m4, m5, m6, m7, m8) \
{ \
    ulong t0 = gold_add(gold_add(gold_mul(m0, s0), gold_mul(m1, s1)), gold_mul(m2, s2)); \
    ulong t1 = gold_add(gold_add(gold_mul(m3, s0), gold_mul(m4, s1)), gold_mul(m5, s2)); \
    ulong t2 = gold_add(gold_add(gold_mul(m6, s0), gold_mul(m7, s1)), gold_mul(m8, s2)); \
    s0 = t0; s1 = t1; s2 = t2; \
}

#define MATVEC_INT(s0, s1, s2, d0, d1, d2) \
{ \
    ulong sum_s = gold_add(gold_add(s0, s1), s2); \
    ulong t0 = gold_add(sum_s, gold_mul(d0, s0)); \
    ulong t1 = gold_add(sum_s, gold_mul(d1, s1)); \
    ulong t2 = gold_add(sum_s, gold_mul(d2, s2)); \
    s0 = t0; s1 = t1; s2 = t2; \
}

inline void poseidon2_permute_t3_optimized(
    thread ulong &s0, thread ulong &s1, thread ulong &s2,
    device const ulong *rc_ext,
    device const ulong *rc_int,
    device const ulong *ext_mds,
    device const ulong *int_diag)
{
    // Coalesced 256-bit and 128-bit loads for matrix constants
    device const ulong4* mds4 = (device const ulong4*)ext_mds;
    ulong4 row01 = mds4[0];
    ulong4 row23 = mds4[1];
    const ulong m8 = ext_mds[8];

    const ulong m0 = row01.x, m1 = row01.y, m2 = row01.z, m3 = row01.w;
    const ulong m4 = row23.x, m5 = row23.y, m6 = row23.z, m7 = row23.w;

    device const ulong2* diag2 = (device const ulong2*)int_diag;
    ulong2 d01 = diag2[0];
    const ulong d2 = int_diag[2];
    const ulong d0 = d01.x, d1 = d01.y;

    MATVEC_EXT(s0, s1, s2, m0, m1, m2, m3, m4, m5, m6, m7, m8);

    // First half full rounds (4 rounds)
    #pragma unroll
    for (uint r = 0u; r < 4u; ++r) {
        s0 = sbox7(gold_add(s0, rc_ext[r * 3u + 0u]));
        s1 = sbox7(gold_add(s1, rc_ext[r * 3u + 1u]));
        s2 = sbox7(gold_add(s2, rc_ext[r * 3u + 2u]));
        MATVEC_EXT(s0, s1, s2, m0, m1, m2, m3, m4, m5, m6, m7, m8);
    }

    // Partial rounds (22 rounds)
    #pragma unroll
    for (uint r = 0u; r < 22u; ++r) {
        s0 = sbox7(gold_add(s0, rc_int[r]));
        MATVEC_INT(s0, s1, s2, d0, d1, d2);
    }

    // Second half full rounds (4 rounds)
    #pragma unroll
    for (uint r = 4u; r < 8u; ++r) {
        s0 = sbox7(gold_add(s0, rc_ext[r * 3u + 0u]));
        s1 = sbox7(gold_add(s1, rc_ext[r * 3u + 1u]));
        s2 = sbox7(gold_add(s2, rc_ext[r * 3u + 2u]));
        MATVEC_EXT(s0, s1, s2, m0, m1, m2, m3, m4, m5, m6, m7, m8);
    }
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

    ulong s0 = 0ul;
    ulong s1 = 0ul;
    ulong s2 = 0ul;

    uint base = p << 1u;
    if (base + 1u < child_count) {
        device const ulong2* ptr = (device const ulong2*)(tree + in_offset);
        ulong2 loaded = ptr[p];
        s0 = loaded.x;
        s1 = loaded.y;
    } else if (base < child_count) {
        s0 = tree[in_offset + base];
    }

    poseidon2_permute_t3_optimized(s0, s1, s2, rc_ext, rc_int, ext_mds, int_diag);

    tree[out_offset + p] = s0;
}
```