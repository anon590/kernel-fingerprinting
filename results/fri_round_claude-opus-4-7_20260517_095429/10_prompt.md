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

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;

constexpr constant uint POS2_R_F = 8u;
constexpr constant uint POS2_R_P = 22u;

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    // overflow if t < a
    ulong of = (t < a) ? EPSILON : 0ul;
    t += of;
    return (t >= P_GOLD) ? (t - P_GOLD) : t;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    ulong uf = (t > a) ? EPSILON : 0ul;
    return t - uf;
}

// 64x64 -> 128 multiply via 32-bit halves.
inline ulong2 umul128(ulong a, ulong b) {
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
    return ulong2(lo, hi);
}

// Reduce (lo, hi) mod p = 2^64 - 2^32 + 1.
// hi = hi_hi*2^32 + hi_lo
// x = lo + hi_lo * 2^32 + hi_hi * 2^64
//   = lo + hi_lo * 2^32 + hi_hi * (2^32 - 1)
// Use: t = lo - hi_hi (mod p); then add hi_lo*(2^32-1) = hi_lo*EPSILON
inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong x_hi_lo = x_hi & EPSILON;
    ulong x_hi_hi = x_hi >> 32;

    ulong t0 = x_lo - x_hi_hi;
    ulong uf0 = (t0 > x_lo) ? EPSILON : 0ul;
    t0 -= uf0;

    // hi_lo < 2^32, so hi_lo*EPSILON < 2^64 - 2^32 < p, fits in 64 bits.
    ulong t1 = x_hi_lo * EPSILON;

    ulong t2 = t0 + t1;
    ulong of = (t2 < t0) ? EPSILON : 0ul;
    t2 += of;
    return (t2 >= P_GOLD) ? (t2 - P_GOLD) : t2;
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
        // r0 = ax, r1 = ax * z1
        // s_m = 1 + r_m
        // out = inv_fold * (e0*(1+r0) + e1*(1+r1))
        //     = inv_fold * ((e0+e1) + e0*r0 + e1*r1)
        ulong z1 = zeta_inv_pow[1];
        ulong e0 = evals_in[j];
        ulong e1 = evals_in[j + N];
        ulong r0 = ax;
        ulong r1 = gold_mul(ax, z1);
        ulong e0r0 = gold_mul(e0, r0);
        ulong e1r1 = gold_mul(e1, r1);
        ulong sum_e = gold_add(e0, e1);
        ulong acc  = gold_add(sum_e, gold_add(e0r0, e1r1));
        evals_out[j] = gold_mul(acc, inv_fold);
        return;
    }

    if (F == 4u) {
        ulong z1 = zeta_inv_pow[1];
        ulong z2 = zeta_inv_pow[2];
        ulong z3 = zeta_inv_pow[3];

        // Load evals first to overlap with arithmetic
        ulong e0 = evals_in[j];
        ulong e1 = evals_in[j + N];
        ulong e2 = evals_in[j + 2u * N];
        ulong e3 = evals_in[j + 3u * N];

        // All r's independent
        ulong r0 = ax;
        ulong r1 = gold_mul(ax, z1);
        ulong r2 = gold_mul(ax, z2);
        ulong r3 = gold_mul(ax, z3);

        // All r^2's independent
        ulong r0_2 = gold_mul(r0, r0);
        ulong r1_2 = gold_mul(r1, r1);
        ulong r2_2 = gold_mul(r2, r2);
        ulong r3_2 = gold_mul(r3, r3);

        // s_m = (1 + r_m) * (1 + r_m^2)
        ulong s0 = gold_mul(gold_add(1ul, r0), gold_add(1ul, r0_2));
        ulong s1 = gold_mul(gold_add(1ul, r1), gold_add(1ul, r1_2));
        ulong s2 = gold_mul(gold_add(1ul, r2), gold_add(1ul, r2_2));
        ulong s3 = gold_mul(gold_add(1ul, r3), gold_add(1ul, r3_2));

        // e*s in 4 independent streams
        ulong p0 = gold_mul(e0, s0);
        ulong p1 = gold_mul(e1, s1);
        ulong p2 = gold_mul(e2, s2);
        ulong p3 = gold_mul(e3, s3);

        // pair-tree reduction
        ulong a01 = gold_add(p0, p1);
        ulong a23 = gold_add(p2, p3);
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
    ulong s0 = state[0], s1 = state[1], s2 = state[2];
    ulong t0 = gold_add(gold_add(gold_mul(ext_mds[0], s0),
                                 gold_mul(ext_mds[1], s1)),
                        gold_mul(ext_mds[2], s2));
    ulong t1 = gold_add(gold_add(gold_mul(ext_mds[3], s0),
                                 gold_mul(ext_mds[4], s1)),
                        gold_mul(ext_mds[5], s2));
    ulong t2 = gold_add(gold_add(gold_mul(ext_mds[6], s0),
                                 gold_mul(ext_mds[7], s1)),
                        gold_mul(ext_mds[8], s2));
    state[0] = t0; state[1] = t1; state[2] = t2;
}

inline void matvec_ext_t3_fused(thread ulong *state)
{
    // M_E = J + I  =>  y_i = sum(s) + s_i.
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
```

Result of previous attempt:
           f2_N64K: correct, 1.28 ms, 8.7 Gmodmul/s (int64) (16.2% of 53 Gops/s (int64 mul, est))
          f2_N256K: correct, 1.92 ms, 23.1 Gmodmul/s (int64) (43.3% of 53 Gops/s (int64 mul, est))
            f2_N1M: correct, 5.94 ms, 29.9 Gmodmul/s (int64) (56.1% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.3403

## Current best (incumbent)

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

inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong x_hi_lo = x_hi & EPSILON;
    ulong x_hi_hi = x_hi >> 32;

    ulong t0 = x_lo - x_hi_hi;
    if (t0 > x_lo) t0 -= EPSILON;

    ulong t1 = x_hi_lo * EPSILON;

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
//
// KEY optimization: Poseidon2 external MDS for t=3 is
//   M_E = [[2,1,1],[1,2,1],[1,1,2]] = J + I
// so M_E * s = (s0+s1+s2) + s_i for component i, which is just adds.
// We verify by reading ext_mds[0]; if it equals 2 we use the fused
// add-only path; otherwise we fall back to the generic matvec.
// The Plonky3/Horizen Labs Poseidon2 spec uses exactly this matrix
// for t=3, so this is the common case. The fallback keeps us
// correct for any other matrix the host might bind.
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
    // M_E = J + I  =>  y_i = sum(s) + s_i.
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

    // Detect M_E = [[2,1,1],[1,2,1],[1,1,2]] vs anything else, in
    // canonical form. Since round constants and MDS are bound once
    // per-pipeline, branching on ext_mds[0] is uniform across the grid.
    ulong m00 = ext_mds[0];
    ulong m01 = ext_mds[1];
    ulong m02 = ext_mds[2];
    bool fused = (m00 == 2ul) && (m01 == 1ul) && (m02 == 1ul);

    ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2];

    uint base = p << 1u;
    ulong s0 = tree[in_offset + base];
    ulong s1 = (base + 1u < child_count) ? tree[in_offset + base + 1u] : 0ul;
    ulong s2 = 0ul;

    thread ulong st[3];
    st[0] = s0; st[1] = s1; st[2] = s2;

    // Initial external MDS.
    if (fused) {
        matvec_ext_t3_fused(st);
    } else {
        matvec_ext_t3_generic(st, ext_mds);
    }

    // First half full rounds.
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

    // Partial rounds: y_i = sum(s) + d_i * s_i.
    for (uint r = 0u; r < POS2_R_P; ++r) {
        st[0] = sbox7(gold_add(st[0], rc_int[r]));
        ulong s = gold_add(gold_add(st[0], st[1]), st[2]);
        ulong t0 = gold_add(s, gold_mul(d0, st[0]));
        ulong t1 = gold_add(s, gold_mul(d1, st[1]));
        ulong t2 = gold_add(s, gold_mul(d2, st[2]));
        st[0] = t0; st[1] = t1; st[2] = t2;
    }

    // Second half full rounds.
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
```

Incumbent result:
           f2_N64K: correct, 1.20 ms, 9.3 Gmodmul/s (int64) (17.4% of 53 Gops/s (int64 mul, est))
          f2_N256K: correct, 1.94 ms, 22.9 Gmodmul/s (int64) (42.8% of 53 Gops/s (int64 mul, est))
            f2_N1M: correct, 6.07 ms, 29.3 Gmodmul/s (int64) (54.9% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.3443

## History

- iter  2: compile=OK | correct=True | score=0.26091115284017186
- iter  3: compile=OK | correct=True | score=0.26021751240299773
- iter  4: compile=OK | correct=True | score=0.2612601633088594
- iter  5: compile=OK | correct=True | score=0.25377464507822434
- iter  6: compile=OK | correct=True | score=0.24329202285533494
- iter  7: compile=OK | correct=True | score=0.3442978844039952
- iter  8: compile=OK | correct=True | score=0.23564698205018722
- iter  9: compile=OK | correct=True | score=0.3403158596895179

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
