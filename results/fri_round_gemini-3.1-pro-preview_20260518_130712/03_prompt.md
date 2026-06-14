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

constant ulong P_GOLD = 0xFFFFFFFF00000001ul;

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
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    ulong p00 = (ulong)a0 * b0;
    ulong p01 = (ulong)a0 * b1;
    ulong p10 = (ulong)a1 * b0;
    ulong p11 = (ulong)a1 * b1;

    ulong mid = (p00 >> 32) + (p01 & 0xFFFFFFFFul) + (p10 & 0xFFFFFFFFul);
    ulong lo  = (p00 & 0xFFFFFFFFul) | (mid << 32);
    ulong hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);

    ulong t0 = lo - (hi >> 32);
    if (t0 > lo) t0 -= 0xFFFFFFFFul;

    ulong hi_lo = hi & 0xFFFFFFFFul;
    ulong t1 = (hi_lo << 32) - hi_lo;

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

    if (fold == 2u) {
        ulong S0 = gold_add(1ul, ax);
        ulong S1 = gold_sub(1ul, ax);
        
        ulong e0 = evals_in[j];
        ulong e1 = evals_in[j + n_out];
        
        acc = gold_add(gold_mul(e0, S0), gold_mul(e1, S1));
    } else if (fold == 4u) {
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
        ulong e2 = evals_in[j + 2u * n_out];
        ulong e3 = evals_in[j + 3u * n_out];
        
        ulong acc02 = gold_add(gold_mul(e0, S0), gold_mul(e2, S2));
        ulong acc13 = gold_add(gold_mul(e1, S1), gold_mul(e3, S3));
        
        acc = gold_add(acc02, acc13);
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

    ulong state[3];
    uint base = p << 1u;
    state[0] = tree[in_offset + base];
    state[1] = (base + 1u < child_count) ? tree[in_offset + base + 1u] : 0ul;
    state[2] = 0ul;

    ulong mds0 = ext_mds[0], mds1 = ext_mds[1], mds2 = ext_mds[2];
    ulong mds3 = ext_mds[3], mds4 = ext_mds[4], mds5 = ext_mds[5];
    ulong mds6 = ext_mds[6], mds7 = ext_mds[7], mds8 = ext_mds[8];

    // Initial external matvec
    {
        ulong t0 = state[0], t1 = state[1], t2 = state[2];
        state[0] = gold_add(gold_add(gold_mul(mds0, t0), gold_mul(mds1, t1)), gold_mul(mds2, t2));
        state[1] = gold_add(gold_add(gold_mul(mds3, t0), gold_mul(mds4, t1)), gold_mul(mds5, t2));
        state[2] = gold_add(gold_add(gold_mul(mds6, t0), gold_mul(mds7, t1)), gold_mul(mds8, t2));
    }

    // First half full rounds
    #pragma unroll
    for (uint r = 0u; r < 4u; ++r) {
        state[0] = sbox7(gold_add(state[0], rc_ext[r * 3 + 0]));
        state[1] = sbox7(gold_add(state[1], rc_ext[r * 3 + 1]));
        state[2] = sbox7(gold_add(state[2], rc_ext[r * 3 + 2]));
        
        ulong t0 = state[0], t1 = state[1], t2 = state[2];
        state[0] = gold_add(gold_add(gold_mul(mds0, t0), gold_mul(mds1, t1)), gold_mul(mds2, t2));
        state[1] = gold_add(gold_add(gold_mul(mds3, t0), gold_mul(mds4, t1)), gold_mul(mds5, t2));
        state[2] = gold_add(gold_add(gold_mul(mds6, t0), gold_mul(mds7, t1)), gold_mul(mds8, t2));
    }

    ulong diag0 = int_diag[0], diag1 = int_diag[1], diag2 = int_diag[2];

    // Partial rounds (independent state[1]+state[2] runs simultaneously to sbox7)
    #pragma unroll
    for (uint r = 0u; r < 22u; ++r) {
        ulong s_rem = gold_add(state[1], state[2]);
        state[0] = sbox7(gold_add(state[0], rc_int[r]));
        
        ulong s = gold_add(state[0], s_rem);
        state[0] = gold_add(s, gold_mul(diag0, state[0]));
        state[1] = gold_add(s, gold_mul(diag1, state[1]));
        state[2] = gold_add(s, gold_mul(diag2, state[2]));
    }

    // Second half full rounds
    #pragma unroll
    for (uint r = 4u; r < 8u; ++r) {
        state[0] = sbox7(gold_add(state[0], rc_ext[r * 3 + 0]));
        state[1] = sbox7(gold_add(state[1], rc_ext[r * 3 + 1]));
        state[2] = sbox7(gold_add(state[2], rc_ext[r * 3 + 2]));
        
        ulong t0 = state[0], t1 = state[1], t2 = state[2];
        state[0] = gold_add(gold_add(gold_mul(mds0, t0), gold_mul(mds1, t1)), gold_mul(mds2, t2));
        state[1] = gold_add(gold_add(gold_mul(mds3, t0), gold_mul(mds4, t1)), gold_mul(mds5, t2));
        state[2] = gold_add(gold_add(gold_mul(mds6, t0), gold_mul(mds7, t1)), gold_mul(mds8, t2));
    }

    tree[out_offset + p] = state[0];
}
```

Result of previous attempt:
           f2_N64K: correct, 2.12 ms, 5.2 Gmodmul/s (int64) (9.8% of 53 Gops/s (int64 mul, est))
          f2_N256K: correct, 3.59 ms, 12.4 Gmodmul/s (int64) (23.2% of 53 Gops/s (int64 mul, est))
            f2_N1M: correct, 9.32 ms, 19.1 Gmodmul/s (int64) (35.8% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.2013

## Current best (incumbent)

```metal
// Naive seed for one FRI folding round (Z5).
//
// Pipeline (host-dispatched in this order inside a single compute
// command encoder; the serial encoder gives read-after-write
// ordering between dispatches with no explicit barriers needed):
//
//   1) fri_fold              : one dispatch over n_out = N / fold
//                              threads; produces the folded
//                              evaluations into the leaves slice of
//                              the tree buffer.
//   2) fri_commit_level (xN) : one dispatch per Merkle level over
//                              the folded evaluations; produces a
//                              binary Poseidon2-t=3 Merkle root.
//
// Algebra of one round (closed-form FRI fold over a coset domain):
//
//   D  = { coset_g * omega_N^i : i in [0, N) }      // input domain
//   D' = { (coset_g)^fold * omega_N^(j*fold) :
//                                 j in [0, N/fold) } // output domain
//
//   E'[j] = inv_fold * sum_{m=0..fold-1} S_m(j) * E[j + m * n_out]
//
//   r_m(j) = alpha / (coset_g * omega_N^{j + m * n_out})
//   S_m(j) = sum_{p=0..fold-1} r_m(j)^p
//
// The host precomputes:
//   inv_x_base[j]   = 1 / (coset_g * omega_N^j)        // length n_out
//   zeta_inv_pow[m] = zeta^{-m}, zeta = omega_N^n_out  // length fold
// so that r_m(j) = alpha * inv_x_base[j] * zeta_inv_pow[m].
//
// Buffer layouts (host-fixed; must be preserved by candidate kernels):
//
// fri_fold:
//   buffer 0: device const ulong *evals_in       (length N)
//   buffer 1: device       ulong *evals_out      (length n_out;
//             the host binds this slot to the leaves slice of the
//             tree buffer used by fri_commit_level)
//   buffer 2: device const ulong *inv_x_base     (length n_out)
//   buffer 3: device const ulong *zeta_inv_pow   (length fold)
//   buffer 4: constant ulong &alpha
//   buffer 5: constant ulong &inv_fold           (1/fold mod p)
//   buffer 6: constant uint  &fold               (2 or 4)
//   buffer 7: constant uint  &n_out              (N / fold)
//
// Dispatch (host-provided):
//   threadsPerGrid        = (n_out rounded up to TG width, 1, 1)
//   threadsPerThreadgroup = (min(n_out, 256), 1, 1)
//   Each thread owns one output index j; guard against j >= n_out.
//
// fri_commit_level (binary Poseidon2-t=3 Merkle build; the host
// issues one dispatch per parent level after the fold):
//   buffer 0: device       ulong *tree           (length sum of
//             binary level_counts(n_out), with the level-0 slice
//             carrying the folded evaluations written by fri_fold)
//   buffer 1: device const ulong *rc_ext         (r_f * t = 8 * 3 = 24)
//   buffer 2: device const ulong *rc_int         (r_p     = 22)
//   buffer 3: device const ulong *ext_mds        (t * t   = 9)
//   buffer 4: device const ulong *int_diag       (t       = 3)
//   buffer 5: constant uint &in_offset           (start of input slice)
//   buffer 6: constant uint &out_offset          (start of output slice)
//   buffer 7: constant uint &child_count         (#nodes at input level)
//
// Boundary policy: if child_count at a level is odd, the lone
// trailing child is paired with a **zero** sibling (matches the CPU
// reference). With n_out and fold both powers of two, every level
// before the root has an even node count.
//
// Dispatch (host-provided, once per parent level):
//   threadsPerGrid        = (parent_count rounded up, 1, 1)
//   threadsPerThreadgroup = (min(parent_count, 64), 1, 1)
//
// Poseidon2 parameters for the commit:
//   t = 3, arity = 2, alpha = 7 S-box, r_f = 8, r_p = 22.
// All round constants and MDS coefficients are read at runtime from
// the bound device buffers.

#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;   // 2^64 - 2^32 + 1
constant ulong EPSILON = 0x00000000FFFFFFFFul;   // 2^32 - 1

constexpr constant uint T_FIXED  = 3u;           // Poseidon2 width
constexpr constant uint FOLD_MAX = 4u;           // upper bound on fold
constexpr constant uint POS2_R_F = 8u;           // Poseidon2 full rounds (4+4)
constexpr constant uint POS2_R_P = 22u;          // Poseidon2 partial rounds

// ----------------------------------------------------------------------
// Goldilocks arithmetic (same as the goldilocks_ntt / merkle_build seeds)
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

// ----------------------------------------------------------------------
// FRI fold (one kernel dispatch per round)
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

    // alpha / x where x = coset_g * omega_N^j.
    ulong ax = gold_mul(alpha, inv_x_base[j]);

    ulong acc = 0ul;
    for (uint m = 0u; m < fold; ++m) {
        // r_m = alpha / (coset_g * omega_N^{j + m * n_out})
        //     = (alpha / x) * zeta^{-m}
        ulong rm = gold_mul(ax, zeta_inv_pow[m]);

        // S_m = sum_{p=0..fold-1} r_m^p
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

// ----------------------------------------------------------------------
// Poseidon2-t=3 (binary Merkle commit; matches the merkle_build seed)
// ----------------------------------------------------------------------

inline ulong sbox7(ulong x) {
    ulong x2 = gold_mul(x, x);
    ulong x4 = gold_mul(x2, x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
}

inline void matvec_ext_t3(thread ulong *state,
                          device const ulong *ext_mds)
{
    ulong tmp[T_FIXED];
    for (uint i = 0u; i < T_FIXED; ++i) {
        ulong acc = 0ul;
        for (uint k = 0u; k < T_FIXED; ++k) {
            acc = gold_add(acc, gold_mul(ext_mds[i * T_FIXED + k], state[k]));
        }
        tmp[i] = acc;
    }
    for (uint i = 0u; i < T_FIXED; ++i) state[i] = tmp[i];
}

inline void matvec_int_t3(thread ulong *state,
                          device const ulong *int_diag)
{
    // M_I = J + diag(int_diag); equivalently y[i] = sum(state) + d[i] * state[i].
    ulong s = 0ul;
    for (uint i = 0u; i < T_FIXED; ++i) s = gold_add(s, state[i]);
    ulong tmp[T_FIXED];
    for (uint i = 0u; i < T_FIXED; ++i) {
        tmp[i] = gold_add(s, gold_mul(int_diag[i], state[i]));
    }
    for (uint i = 0u; i < T_FIXED; ++i) state[i] = tmp[i];
}

inline void poseidon2_permute_t3(thread ulong *state,
                                 device const ulong *rc_ext,
                                 device const ulong *rc_int,
                                 device const ulong *ext_mds,
                                 device const ulong *int_diag)
{
    matvec_ext_t3(state, ext_mds);

    // First half full rounds.
    for (uint r = 0u; r < POS2_R_F / 2u; ++r) {
        for (uint i = 0u; i < T_FIXED; ++i) {
            state[i] = gold_add(state[i], rc_ext[r * T_FIXED + i]);
            state[i] = sbox7(state[i]);
        }
        matvec_ext_t3(state, ext_mds);
    }

    // Partial rounds.
    for (uint r = 0u; r < POS2_R_P; ++r) {
        state[0] = gold_add(state[0], rc_int[r]);
        state[0] = sbox7(state[0]);
        matvec_int_t3(state, int_diag);
    }

    // Second half full rounds.
    for (uint r = POS2_R_F / 2u; r < POS2_R_F; ++r) {
        for (uint i = 0u; i < T_FIXED; ++i) {
            state[i] = gold_add(state[i], rc_ext[r * T_FIXED + i]);
            state[i] = sbox7(state[i]);
        }
        matvec_ext_t3(state, ext_mds);
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

    thread ulong state[T_FIXED];
    state[0] = 0ul; state[1] = 0ul; state[2] = 0ul;

    uint base = p << 1u;
    state[0] = tree[in_offset + base];
    if (base + 1u < child_count) {
        state[1] = tree[in_offset + base + 1u];
    }

    poseidon2_permute_t3(state, rc_ext, rc_int, ext_mds, int_diag);

    tree[out_offset + p] = state[0];
}
```

Incumbent result:
           f2_N64K: correct, 1.56 ms, 7.1 Gmodmul/s (int64) (13.4% of 53 Gops/s (int64 mul, est))
          f2_N256K: correct, 2.70 ms, 16.4 Gmodmul/s (int64) (30.8% of 53 Gops/s (int64 mul, est))
            f2_N1M: correct, 8.47 ms, 21.0 Gmodmul/s (int64) (39.3% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.2530

## History

- iter  0: compile=OK | correct=True | score=0.2530444956927559
- iter  1: compile=OK | correct=True | score=0.2014186337120031
- iter  2: compile=OK | correct=True | score=0.20126028888524133

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
