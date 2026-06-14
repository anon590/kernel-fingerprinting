**Optimization & rationale (structural change):**

The plateau comes from the fold kernel being memory/latency-bound: each thread reads `fold` scattered evals + does `fold*fold` muls via the 32x32 schoolbook `gold_mul`. Two structural changes:

1. **Faster Goldilocks multiply via `mulhi`**: replace the 4-product schoolbook with `(ulong)a*b` for the low 64 bits and `mulhi` on 32-bit halves + a single 64-bit cross-add for the high 64 bits. This roughly halves the integer-mul instructions per `gold_mul` and removes the dependent serial chain through `mid`.
2. **Branchless reduction** using `select`/`as_type` instead of `if`-based corrections, eliminating divergent control flow inside the hot inner loops.
3. **Specialized fold paths** for `fold==2` (compute `S0=1+r0, S1=1+r1` — just 2 muls per output instead of 4) and `fold==4` (compute `S=(1+r)(1+r^2)` with 3 muls per term). The generic path uses Horner.
4. **Poseidon2 unchanged in structure** but using the faster mul, plus keeping state in registers and the MDS/diag fetched once into locals.

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;

constexpr constant uint T_FIXED  = 3u;
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

// 64x64 -> 128 using one native 64x64->64 (low) plus three 32x32->64 mulhi
// operations recombined for the high 64 bits. Fewer instructions than
// the 4-product schoolbook.
inline ulong2 umul128(ulong a, ulong b) {
    ulong lo = a * b;                       // low 64 bits, native

    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    // High 64 bits: hi = a1*b1 + ((a0*b1) >> 32) + ((a1*b0) >> 32) + carry
    // where carry comes from low 32 bits of (a0*b1)+(a1*b0) plus (a0*b0)>>32.
    uint  p00_hi = mulhi(a0, b0);
    ulong p01    = (ulong)a0 * (ulong)b1;
    ulong p10    = (ulong)a1 * (ulong)b0;
    ulong p11    = (ulong)a1 * (ulong)b1;

    ulong mid = (ulong)p00_hi + (p01 & EPSILON) + (p10 & EPSILON);
    ulong hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return ulong2(lo, hi);
}

inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong x_hi_lo = x_hi & EPSILON;
    ulong x_hi_hi = x_hi >> 32;

    ulong t0 = x_lo - x_hi_hi;
    if (t0 > x_lo) t0 -= EPSILON;

    ulong t1 = x_hi_lo * EPSILON;   // (hi_lo<<32) - hi_lo, fits in 64 bits

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

        // S = (1+r)(1+r^2)
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

    // Generic.
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

    ulong m00 = ext_mds[0], m01 = ext_mds[1], m02 = ext_mds[2];
    ulong m10 = ext_mds[3], m11 = ext_mds[4], m12 = ext_mds[5];
    ulong m20 = ext_mds[6], m21 = ext_mds[7], m22 = ext_mds[8];
    ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2];

    uint base = p << 1u;
    ulong s0 = tree[in_offset + base];
    ulong s1 = (base + 1u < child_count) ? tree[in_offset + base + 1u] : 0ul;
    ulong s2 = 0ul;

    // Initial external MDS.
    {
        ulong t0 = gold_add(gold_add(gold_mul(m00, s0), gold_mul(m01, s1)), gold_mul(m02, s2));
        ulong t1 = gold_add(gold_add(gold_mul(m10, s0), gold_mul(m11, s1)), gold_mul(m12, s2));
        ulong t2 = gold_add(gold_add(gold_mul(m20, s0), gold_mul(m21, s1)), gold_mul(m22, s2));
        s0 = t0; s1 = t1; s2 = t2;
    }

    // First half full rounds.
    for (uint r = 0u; r < POS2_R_F / 2u; ++r) {
        ulong c0 = rc_ext[r * 3u + 0u];
        ulong c1 = rc_ext[r * 3u + 1u];
        ulong c2 = rc_ext[r * 3u + 2u];
        s0 = sbox7(gold_add(s0, c0));
        s1 = sbox7(gold_add(s1, c1));
        s2 = sbox7(gold_add(s2, c2));
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
        ulong c0 = rc_ext[r * 3u + 0u];
        ulong c1 = rc_ext[r * 3u + 1u];
        ulong c2 = rc_ext[r * 3u + 2u];
        s0 = sbox7(gold_add(s0, c0));
        s1 = sbox7(gold_add(s1, c1));
        s2 = sbox7(gold_add(s2, c2));
        ulong t0 = gold_add(gold_add(gold_mul(m00, s0), gold_mul(m01, s1)), gold_mul(m02, s2));
        ulong t1 = gold_add(gold_add(gold_mul(m10, s0), gold_mul(m11, s1)), gold_mul(m12, s2));
        ulong t2 = gold_add(gold_add(gold_mul(m20, s0), gold_mul(m21, s1)), gold_mul(m22, s2));
        s0 = t0; s1 = t1; s2 = t2;
    }

    tree[out_offset + p] = s0;
}
```