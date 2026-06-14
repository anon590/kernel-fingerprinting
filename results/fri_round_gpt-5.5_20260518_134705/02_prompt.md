## Task: fri_round

One FRI folding round on a polynomial committed via evaluations over a Goldilocks coset, followed by a binary Poseidon2-t=3 Merkle commit of the folded evaluations. The folding factor and N are bound at runtime through constant buffers; the kernel must use those runtime values rather than compile-time constants.

Algebra of one round (closed-form FRI fold over a coset domain):
  D  = { coset_g * omega_N^i  : i in [0, N)         }
  D' = { coset_g^fold * omega_N^(j*fold) : j in [0, n_out) }
  E'[j] = inv_fold * sum_{m=0..fold-1} S_m(j) * E[j + m * n_out]
  r_m(j) = alpha / (coset_g * omega_N^{j + m * n_out})
  S_m(j) = sum_{p=0..fold-1} r_m(j)^p
with omega_N the primitive N-th root of unity in Goldilocks (p = 2^64 - 2^32 + 1, derived from the plonky2 / risc0 generator g_root_2^32 = 1753635133440165772) and coset_g = 7. n_out = N / fold.

Host-side precomputation (uploaded to device buffers):
  inv_x_base[j]   = 1 / (coset_g * omega_N^j)   (length n_out)
  zeta_inv_pow[m] = zeta^{-m}, zeta = omega_N^n_out  (length fold)
  alpha           = round challenge (canonical [0, p))
  inv_fold        = pow(fold, -1, p)
so that r_m(j) = alpha * inv_x_base[j] * zeta_inv_pow[m].

Two-kernel pipeline (host issues both in one compute command encoder; the serial encoder gives read-after-write ordering across dispatches with no explicit barrier):
  1) fri_fold              : one dispatch over n_out threads; the host binds the tree buffer to fri_fold's evals_out slot, so the fold writes land in the level-0 (leaves) slice of the Merkle tree.
  2) fri_commit_level (xL) : one dispatch per Merkle level over n_out folded leaves; produces the binary Poseidon2-t=3 Merkle root. The Merkle commit is binary Poseidon2-t=3 across every test size.

Tree layout: a single contiguous ulong buffer holds ALL Merkle levels concatenated -- folded leaves first, then each parent level in order, finally the 1-element root. Total length = sum of binary level_counts(n_out). Per-level scalars (in_offset, out_offset, child_count) are bound at dispatch time via per-level uint offsets into a small constants buffer (mirrors the merkle_build task's host pattern).

Correctness is bit-exact against the Python bigint reference:
  * folded leaves slice (length n_out) must match the reference folded evaluations element-wise;
  * the full tree (all levels, including every intermediate digest, not just the root) must match the reference Merkle commitment.
Outputs MUST be canonical ([0, p)); a non-canonical value with the same residue class still counts as a mismatch. All test sizes satisfy fold <= 4 and t = 3; thread-private scratch arrays of size 4 are sufficient.

## Required kernel signature(s)

```
kernel void fri_fold(
    device const ulong *evals_in     [[buffer(0)]],
    device       ulong *evals_out    [[buffer(1)]],
    device const ulong *inv_x_base   [[buffer(2)]],
    device const ulong *zeta_inv_pow [[buffer(3)]],
    constant ulong     &alpha        [[buffer(4)]],
    constant ulong     &inv_fold     [[buffer(5)]],
    constant uint      &fold         [[buffer(6)]],
    constant uint      &n_out        [[buffer(7)]],
    uint j [[thread_position_in_grid]]);

kernel void fri_commit_level(
    device       ulong *tree         [[buffer(0)]],
    device const ulong *rc_ext       [[buffer(1)]],
    device const ulong *rc_int       [[buffer(2)]],
    device const ulong *ext_mds      [[buffer(3)]],
    device const ulong *int_diag     [[buffer(4)]],
    constant uint      &in_offset    [[buffer(5)]],
    constant uint      &out_offset   [[buffer(6)]],
    constant uint      &child_count  [[buffer(7)]],
    uint p [[thread_position_in_grid]]);

Dispatch geometry (host-fixed):
  fri_fold:
    threadsPerGrid        = (n_out rounded up to TG width, 1, 1)
    threadsPerThreadgroup = (min(n_out, 256), 1, 1)
  fri_commit_level (one call per parent level):
    threadsPerGrid        = (parent_count rounded up, 1, 1)
    threadsPerThreadgroup = (min(parent_count, 64), 1, 1)

fri_fold: each thread owns one output index j; guard against j >= n_out (the grid is rounded up to a multiple of the TG width). The same tree buffer is bound to evals_out (offset 0); the commit kernel reads from that buffer on the next dispatch.
fri_commit_level: each thread owns one parent node at the current level; guard against p >= parent_count = ceil(child_count / 2). The host pre-binds rc_ext / rc_int / ext_mds / int_diag once and rebinds only the three per-level uint scalars per dispatch.
```

## Your previous attempt

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD      = 0xFFFFFFFF00000001ul;
constant ulong P_MINUS_ONE = 0xFFFFFFFF00000000ul;
constant ulong EPSILON     = 0x00000000FFFFFFFFul;

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

inline ulong gold_neg(ulong x) {
    return (x == 0ul) ? 0ul : (P_GOLD - x);
}

inline ulong gold_reduce_limbs(uint x0, uint x1, uint x2, uint x3) {
    ulong lo = ((ulong)x1 << 32) | (ulong)x0;

    ulong t0 = lo - (ulong)x3;
    if (t0 > lo) t0 -= EPSILON;

    ulong t1 = ((ulong)x2 << 32) - (ulong)x2;

    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;

    return gold_canonical(t2);
}

inline ulong gold_mul(ulong a, ulong b) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    uint p00l = a0 * b0;
    uint p00h = mulhi(a0, b0);
    uint p01l = a0 * b1;
    uint p01h = mulhi(a0, b1);
    uint p10l = a1 * b0;
    uint p10h = mulhi(a1, b0);
    uint p11l = a1 * b1;
    uint p11h = mulhi(a1, b1);

    uint s1 = p00h + p01l;
    uint c1 = (s1 < p00h) ? 1u : 0u;
    uint x1 = s1 + p10l;
    c1 += (x1 < s1) ? 1u : 0u;

    uint s2 = p01h + p10h;
    uint c2 = (s2 < p01h) ? 1u : 0u;
    uint s2b = s2 + p11l;
    c2 += (s2b < s2) ? 1u : 0u;
    uint x2 = s2b + c1;
    c2 += (x2 < s2b) ? 1u : 0u;

    uint x3 = p11h + c2;

    return gold_reduce_limbs(p00l, x1, x2, x3);
}

inline ulong gold_mul_const_common(ulong c, ulong x) {
    if (c == 0ul) return 0ul;
    if (c == 1ul) return x;
    if (c == 2ul) return gold_add(x, x);
    if (c == 3ul) {
        ulong y = gold_add(x, x);
        return gold_add(y, x);
    }
    if (c == 4ul) {
        ulong y = gold_add(x, x);
        return gold_add(y, y);
    }
    if (c == P_MINUS_ONE) return gold_neg(x);
    return gold_mul(c, x);
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

    ulong ax = gold_mul(alpha, inv_x_base[j]);

    if (fold == 2u) {
        ulong e0 = evals_in[j];
        ulong e1 = evals_in[j + n_out];
        ulong z1 = zeta_inv_pow[1];

        if (z1 == P_MINUS_ONE) {
            // inv2 * ((e0 + e1) + ax * (e0 - e1))
            ulong c0 = gold_add(e0, e1);
            ulong c1 = gold_sub(e0, e1);
            ulong acc = gold_add(c0, gold_mul(ax, c1));
            evals_out[j] = gold_mul(acc, inv_fold);
            return;
        } else {
            ulong s0 = gold_add(1ul, ax);
            ulong r1 = gold_mul(ax, z1);
            ulong s1 = gold_add(1ul, r1);
            ulong acc = gold_add(gold_mul(e0, s0), gold_mul(e1, s1));
            evals_out[j] = gold_mul(acc, inv_fold);
            return;
        }
    }

    if (fold == 4u) {
        ulong z1 = zeta_inv_pow[1];
        ulong z2 = zeta_inv_pow[2];
        ulong z3 = zeta_inv_pow[3];

        if (z2 == P_MINUS_ONE && gold_add(z1, z3) == 0ul) {
            ulong e0 = evals_in[j];
            ulong e1 = evals_in[j + n_out];
            ulong e2 = evals_in[j + 2u * n_out];
            ulong e3 = evals_in[j + 3u * n_out];

            // 4-point DFT over zeta_inv_pow:
            // sum_m E_m * sum_p (ax*z_m)^p
            // = c0 + ax*c1 + ax^2*c2 + ax^3*c3.
            ulong e1z = gold_mul(e1, z1);
            ulong e3z = gold_mul(e3, z1);

            ulong c0 = gold_add(gold_add(e0, e1), gold_add(e2, e3));
            ulong c1 = gold_sub(gold_sub(gold_add(e0, e1z), e2), e3z);
            ulong c2 = gold_add(gold_sub(e0, e1), gold_sub(e2, e3));
            ulong c3 = gold_add(gold_sub(gold_sub(e0, e1z), e2), e3z);

            ulong ax2 = gold_mul(ax, ax);
            ulong ax3 = gold_mul(ax2, ax);

            ulong acc = c0;
            acc = gold_add(acc, gold_mul(ax,  c1));
            acc = gold_add(acc, gold_mul(ax2, c2));
            acc = gold_add(acc, gold_mul(ax3, c3));

            evals_out[j] = gold_mul(acc, inv_fold);
            return;
        }
    }

    // Generic safe fallback for any fold <= 4.
    ulong acc = 0ul;
    for (uint m = 0u; m < fold; ++m) {
        ulong rm = (m == 0u) ? ax : gold_mul(ax, zeta_inv_pow[m]);

        ulong sm = 1ul;
        ulong rpow = rm;
        for (uint q = 1u; q < fold; ++q) {
            sm = gold_add(sm, rpow);
            if (q + 1u < fold) rpow = gold_mul(rpow, rm);
        }

        acc = gold_add(acc, gold_mul(evals_in[j + m * n_out], sm));
    }

    evals_out[j] = gold_mul(acc, inv_fold);
}

// ----------------------------------------------------------------------
// Poseidon2-t=3 Merkle commit
// ----------------------------------------------------------------------

#define FRI_MDS_EXT_SCALAR() do {                                      \
    ulong ma0 = s0;                                                     \
    ulong ma1 = s1;                                                     \
    ulong ma2 = s2;                                                     \
    if (ext_is_jplusi) {                                                \
        ulong msum = gold_add(gold_add(ma0, ma1), ma2);                 \
        s0 = gold_add(msum, ma0);                                       \
        s1 = gold_add(msum, ma1);                                       \
        s2 = gold_add(msum, ma2);                                       \
    } else if (ext_is_jdiag) {                                          \
        ulong msum = gold_add(gold_add(ma0, ma1), ma2);                 \
        s0 = gold_add(msum, gold_mul_const_common(ed0, ma0));           \
        s1 = gold_add(msum, gold_mul_const_common(ed1, ma1));           \
        s2 = gold_add(msum, gold_mul_const_common(ed2, ma2));           \
    } else {                                                           \
        ulong ny0 = gold_add(gold_add(gold_mul(m00, ma0),               \
                                      gold_mul(m01, ma1)),              \
                                      gold_mul(m02, ma2));              \
        ulong ny1 = gold_add(gold_add(gold_mul(m10, ma0),               \
                                      gold_mul(m11, ma1)),              \
                                      gold_mul(m12, ma2));              \
        ulong ny2 = gold_add(gold_add(gold_mul(m20, ma0),               \
                                      gold_mul(m21, ma1)),              \
                                      gold_mul(m22, ma2));              \
        s0 = ny0; s1 = ny1; s2 = ny2;                                   \
    }                                                                  \
} while (0)

#define FRI_MDS_INT_GENERIC() do {                                      \
    ulong ma0 = s0;                                                     \
    ulong ma1 = s1;                                                     \
    ulong ma2 = s2;                                                     \
    ulong msum = gold_add(gold_add(ma0, ma1), ma2);                     \
    s0 = gold_add(msum, gold_mul(d0, ma0));                             \
    s1 = gold_add(msum, gold_mul(d1, ma1));                             \
    s2 = gold_add(msum, gold_mul(d2, ma2));                             \
} while (0)

#define FRI_MDS_INT_JPLUSI() do {                                       \
    ulong ma0 = s0;                                                     \
    ulong ma1 = s1;                                                     \
    ulong ma2 = s2;                                                     \
    ulong msum = gold_add(gold_add(ma0, ma1), ma2);                     \
    s0 = gold_add(msum, ma0);                                           \
    s1 = gold_add(msum, ma1);                                           \
    s2 = gold_add(msum, ma2);                                           \
} while (0)

#define FRI_MDS_INT_J() do {                                            \
    ulong msum = gold_add(gold_add(s0, s1), s2);                        \
    s0 = msum; s1 = msum; s2 = msum;                                    \
} while (0)

inline ulong poseidon2_hash_t3(ulong s0,
                               ulong s1,
                               device const ulong *rc_ext,
                               device const ulong *rc_int,
                               device const ulong *ext_mds,
                               device const ulong *int_diag)
{
    ulong s2 = 0ul;

    ulong m00 = ext_mds[0];
    ulong m01 = ext_mds[1];
    ulong m02 = ext_mds[2];
    ulong m10 = ext_mds[3];
    ulong m11 = ext_mds[4];
    ulong m12 = ext_mds[5];
    ulong m20 = ext_mds[6];
    ulong m21 = ext_mds[7];
    ulong m22 = ext_mds[8];

    ulong d0 = int_diag[0];
    ulong d1 = int_diag[1];
    ulong d2 = int_diag[2];

    bool offdiag_ones =
        (m01 == 1ul) && (m02 == 1ul) &&
        (m10 == 1ul) && (m12 == 1ul) &&
        (m20 == 1ul) && (m21 == 1ul);

    bool ext_is_jplusi =
        offdiag_ones &&
        (m00 == 2ul) && (m11 == 2ul) && (m22 == 2ul);

    bool ext_is_jdiag = offdiag_ones;

    ulong ed0 = gold_sub(m00, 1ul);
    ulong ed1 = gold_sub(m11, 1ul);
    ulong ed2 = gold_sub(m22, 1ul);

    bool int_is_jplusi = (d0 == 1ul) && (d1 == 1ul) && (d2 == 1ul);
    bool int_is_j      = (d0 == 0ul) && (d1 == 0ul) && (d2 == 0ul);

    FRI_MDS_EXT_SCALAR();

#pragma unroll
    for (uint r = 0u; r < POS2_R_F / 2u; ++r) {
        uint k = r * 3u;
        s0 = sbox7(gold_add(s0, rc_ext[k + 0u]));
        s1 = sbox7(gold_add(s1, rc_ext[k + 1u]));
        s2 = sbox7(gold_add(s2, rc_ext[k + 2u]));
        FRI_MDS_EXT_SCALAR();
    }

    if (int_is_jplusi) {
#pragma unroll
        for (uint r = 0u; r < POS2_R_P; ++r) {
            s0 = sbox7(gold_add(s0, rc_int[r]));
            FRI_MDS_INT_JPLUSI();
        }
    } else if (int_is_j) {
#pragma unroll
        for (uint r = 0u; r < POS2_R_P; ++r) {
            s0 = sbox7(gold_add(s0, rc_int[r]));
            FRI_MDS_INT_J();
        }
    } else {
#pragma unroll
        for (uint r = 0u; r < POS2_R_P; ++r) {
            s0 = sbox7(gold_add(s0, rc_int[r]));
            FRI_MDS_INT_GENERIC();
        }
    }

#pragma unroll
    for (uint r = POS2_R_F / 2u; r < POS2_R_F; ++r) {
        uint k = r * 3u;
        s0 = sbox7(gold_add(s0, rc_ext[k + 0u]));
        s1 = sbox7(gold_add(s1, rc_ext[k + 1u]));
        s2 = sbox7(gold_add(s2, rc_ext[k + 2u]));
        FRI_MDS_EXT_SCALAR();
    }

    return s0;
}

#undef FRI_MDS_EXT_SCALAR
#undef FRI_MDS_INT_GENERIC
#undef FRI_MDS_INT_JPLUSI
#undef FRI_MDS_INT_J

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

    uint base = p << 1u;

    ulong left = tree[in_offset + base];
    ulong right = 0ul;
    if (base + 1u < child_count) {
        right = tree[in_offset + base + 1u];
    }

    tree[out_offset + p] = poseidon2_hash_t3(left, right,
                                             rc_ext, rc_int,
                                             ext_mds, int_diag);
}
```

Result of previous attempt:
           f2_N64K: correct, 1.63 ms, 6.8 Gmodmul/s (int64) (12.8% of 53 Gops/s (int64 mul, est))
          f2_N256K: correct, 2.81 ms, 15.8 Gmodmul/s (int64) (29.6% of 53 Gops/s (int64 mul, est))
            f2_N1M: correct, 7.44 ms, 23.9 Gmodmul/s (int64) (44.8% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.2571

## History

- iter  0: compile=OK | correct=True | score=0.252303523944502
- iter  1: compile=OK | correct=True | score=0.2570927702957165

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
