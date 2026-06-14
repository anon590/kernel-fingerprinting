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
    t += (t < a) ? 0xFFFFFFFFul : 0;
    t -= (t >= P_GOLD) ? P_GOLD : 0;
    return t;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    t -= (t > a) ? 0xFFFFFFFFul : 0;
    return t;
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong lo = a * b;
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);
    
    ulong p01 = (ulong)a0 * b1;
    ulong p10 = (ulong)a1 * b0;
    
    // Explicit 64-bit addition avoids 32-bit carry truncation
    ulong mid = (ulong)mulhi(a0, b0) + (ulong)(uint)p01 + (ulong)(uint)p10;
    ulong hi = (ulong)a1 * b1 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    
    ulong t = lo - (hi >> 32);
    t -= (t > lo) ? 0xFFFFFFFFul : 0;
    
    ulong t1 = ((ulong)(uint)hi << 32) - (uint)hi;
    ulong t2 = t + t1;
    t2 += (t2 < t) ? 0xFFFFFFFFul : 0;
    
    t2 -= (t2 >= P_GOLD) ? P_GOLD : 0;
    return t2;
}

inline ulong sbox7(ulong x) {
    ulong x2 = gold_mul(x, x);
    ulong x3 = gold_mul(x2, x);
    ulong x4 = gold_mul(x2, x2);
    return gold_mul(x4, x3); // Independent execution of x4 and x3
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
        ulong e0 = evals_in[j];
        ulong e1 = evals_in[j + n_out];
        
        ulong e0_plus_e1 = gold_add(e0, e1);
        ulong e0_minus_e1 = gold_sub(e0, e1);
        
        acc = gold_add(e0_plus_e1, gold_mul(ax, e0_minus_e1));
    } else if (fold == 4u) {
        ulong e0 = evals_in[j];
        ulong e1 = evals_in[j + n_out];
        ulong e2 = evals_in[j + 2u * n_out];
        ulong e3 = evals_in[j + 3u * n_out];

        ulong e0_plus_e2 = gold_add(e0, e2);
        ulong e0_minus_e2 = gold_sub(e0, e2);
        ulong e1_plus_e3 = gold_add(e1, e3);
        ulong e1_minus_e3 = gold_sub(e1, e3);

        ulong r0 = ax;
        ulong r1 = gold_mul(ax, zeta_inv_pow[1]);

        ulong r0_2 = gold_mul(r0, r0);
        ulong T0 = gold_add(1ul, r0_2);

        ulong term0 = gold_add(e0_plus_e2, gold_mul(r0, e0_minus_e2));
        ulong term1 = gold_add(e1_plus_e3, gold_mul(r1, e1_minus_e3));

        // Math simplifications evaluating fold polynomial reduces 8 multiplications to 5
        ulong diff = gold_sub(term0, term1);
        ulong term1_x2 = gold_add(term1, term1);

        acc = gold_add(gold_mul(T0, diff), term1_x2);
    } else {
        for (uint m = 0u; m < fold; ++m) {
            ulong rm = (m == 0u) ? ax : gold_mul(ax, zeta_inv_pow[m]);
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

    ulong state0 = 0ul;
    ulong state1 = 0ul;
    ulong state2 = 0ul;

    uint base = p << 1u;
    state0 = tree[in_offset + base];
    if (base + 1u < child_count) {
        state1 = tree[in_offset + base + 1u];
    }

    ulong m0 = ext_mds[0], m1 = ext_mds[1], m2 = ext_mds[2];
    ulong m3 = ext_mds[3], m4 = ext_mds[4], m5 = ext_mds[5];
    ulong m6 = ext_mds[6], m7 = ext_mds[7], m8 = ext_mds[8];
    ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2];

    {
        ulong t0 = state0, t1 = state1, t2 = state2;
        state0 = gold_add(gold_add(gold_mul(m0, t0), gold_mul(m1, t1)), gold_mul(m2, t2));
        state1 = gold_add(gold_add(gold_mul(m3, t0), gold_mul(m4, t1)), gold_mul(m5, t2));
        state2 = gold_add(gold_add(gold_mul(m6, t0), gold_mul(m7, t1)), gold_mul(m8, t2));
    }

    #pragma clang loop unroll(full)
    for (uint r = 0u; r < 4u; ++r) {
        ulong st0 = sbox7(gold_add(state0, rc_ext[r * 3u + 0u]));
        ulong st1 = sbox7(gold_add(state1, rc_ext[r * 3u + 1u]));
        ulong st2 = sbox7(gold_add(state2, rc_ext[r * 3u + 2u]));

        state0 = gold_add(gold_add(gold_mul(m0, st0), gold_mul(m1, st1)), gold_mul(m2, st2));
        state1 = gold_add(gold_add(gold_mul(m3, st0), gold_mul(m4, st1)), gold_mul(m5, st2));
        state2 = gold_add(gold_add(gold_mul(m6, st0), gold_mul(m7, st1)), gold_mul(m8, st2));
    }

    for (uint r = 0u; r < 22u; ++r) {
        ulong s_rem = gold_add(state1, state2);
        
        // ILP Improvement: Calculate disjoint branches natively parallel with S-box latency
        ulong d1_s1 = gold_mul(d1, state1);
        ulong d2_s2 = gold_mul(d2, state2);

        ulong st0_sbox = sbox7(gold_add(state0, rc_int[r]));
        
        ulong s = gold_add(st0_sbox, s_rem);
        
        state0 = gold_add(s, gold_mul(d0, st0_sbox));
        state1 = gold_add(s, d1_s1);
        state2 = gold_add(s, d2_s2);
    }

    #pragma clang loop unroll(full)
    for (uint r = 4u; r < 8u; ++r) {
        ulong st0 = sbox7(gold_add(state0, rc_ext[r * 3u + 0u]));
        ulong st1 = sbox7(gold_add(state1, rc_ext[r * 3u + 1u]));
        ulong st2 = sbox7(gold_add(state2, rc_ext[r * 3u + 2u]));

        state0 = gold_add(gold_add(gold_mul(m0, st0), gold_mul(m1, st1)), gold_mul(m2, st2));
        state1 = gold_add(gold_add(gold_mul(m3, st0), gold_mul(m4, st1)), gold_mul(m5, st2));
        state2 = gold_add(gold_add(gold_mul(m6, st0), gold_mul(m7, st1)), gold_mul(m8, st2));
    }

    tree[out_offset + p] = state0;
}
```

Result of previous attempt:
           f2_N64K: correct, 1.89 ms, 5.9 Gmodmul/s (int64) (11.0% of 53 Gops/s (int64 mul, est))
          f2_N256K: correct, 3.61 ms, 12.3 Gmodmul/s (int64) (23.1% of 53 Gops/s (int64 mul, est))
            f2_N1M: correct, 10.05 ms, 17.7 Gmodmul/s (int64) (33.1% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.2035

## Current best (incumbent)

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
    ulong lo = a * b;
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);
    
    ulong p01 = (ulong)a0 * b1;
    ulong p10 = (ulong)a1 * b0;
    ulong p11 = (ulong)a1 * b1;
    
    ulong mid = mulhi(a0, b0) + (p01 & 0xFFFFFFFFul) + (p10 & 0xFFFFFFFFul);
    ulong hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    
    ulong hi_lo = (uint)hi;
    ulong hi_hi = hi >> 32;

    ulong t0 = lo - hi_hi;
    if (t0 > lo) t0 -= 0xFFFFFFFFul;

    ulong t1 = hi_lo * 0xFFFFFFFFul;

    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += 0xFFFFFFFFul;

    return (t2 >= P_GOLD) ? (t2 - P_GOLD) : t2;
}

inline ulong sbox7(ulong x) {
    ulong x2 = gold_mul(x, x);
    ulong x3 = gold_mul(x2, x);
    ulong x4 = gold_mul(x2, x2);
    return gold_mul(x4, x3);
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
        ulong e0 = evals_in[j];
        ulong e1 = evals_in[j + n_out];
        
        ulong e0_plus_e1 = gold_add(e0, e1);
        ulong e0_minus_e1 = gold_sub(e0, e1);
        
        acc = gold_add(e0_plus_e1, gold_mul(ax, e0_minus_e1));
    } else if (fold == 4u) {
        ulong e0 = evals_in[j];
        ulong e1 = evals_in[j + n_out];
        ulong e2 = evals_in[j + 2u * n_out];
        ulong e3 = evals_in[j + 3u * n_out];

        ulong e0_plus_e2 = gold_add(e0, e2);
        ulong e0_minus_e2 = gold_sub(e0, e2);
        ulong e1_plus_e3 = gold_add(e1, e3);
        ulong e1_minus_e3 = gold_sub(e1, e3);

        ulong r0 = ax;
        ulong r1 = gold_mul(ax, zeta_inv_pow[1]);

        ulong r0_2 = gold_mul(r0, r0);

        ulong T0 = gold_add(1ul, r0_2);
        ulong T1 = gold_sub(1ul, r0_2);

        ulong U0 = gold_mul(r0, T0);
        ulong U1 = gold_mul(r1, T1);

        ulong acc02 = gold_add(gold_mul(T0, e0_plus_e2), gold_mul(U0, e0_minus_e2));
        ulong acc13 = gold_add(gold_mul(T1, e1_plus_e3), gold_mul(U1, e1_minus_e3));

        acc = gold_add(acc02, acc13);
    } else {
        for (uint m = 0u; m < fold; ++m) {
            ulong rm = (m == 0u) ? ax : gold_mul(ax, zeta_inv_pow[m]);
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

    ulong state0 = 0ul;
    ulong state1 = 0ul;
    ulong state2 = 0ul;

    uint base = p << 1u;
    state0 = tree[in_offset + base];
    if (base + 1u < child_count) {
        state1 = tree[in_offset + base + 1u];
    }

    ulong m0 = ext_mds[0], m1 = ext_mds[1], m2 = ext_mds[2];
    ulong m3 = ext_mds[3], m4 = ext_mds[4], m5 = ext_mds[5];
    ulong m6 = ext_mds[6], m7 = ext_mds[7], m8 = ext_mds[8];
    ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2];

    {
        ulong t0 = state0, t1 = state1, t2 = state2;
        state0 = gold_add(gold_add(gold_mul(m0, t0), gold_mul(m1, t1)), gold_mul(m2, t2));
        state1 = gold_add(gold_add(gold_mul(m3, t0), gold_mul(m4, t1)), gold_mul(m5, t2));
        state2 = gold_add(gold_add(gold_mul(m6, t0), gold_mul(m7, t1)), gold_mul(m8, t2));
    }

    for (uint r = 0u; r < 4u; ++r) {
        state0 = sbox7(gold_add(state0, rc_ext[r * 3u + 0u]));
        state1 = sbox7(gold_add(state1, rc_ext[r * 3u + 1u]));
        state2 = sbox7(gold_add(state2, rc_ext[r * 3u + 2u]));

        ulong t0 = state0, t1 = state1, t2 = state2;
        state0 = gold_add(gold_add(gold_mul(m0, t0), gold_mul(m1, t1)), gold_mul(m2, t2));
        state1 = gold_add(gold_add(gold_mul(m3, t0), gold_mul(m4, t1)), gold_mul(m5, t2));
        state2 = gold_add(gold_add(gold_mul(m6, t0), gold_mul(m7, t1)), gold_mul(m8, t2));
    }

    for (uint r = 0u; r < 22u; ++r) {
        ulong s_rem = gold_add(state1, state2);
        state0 = sbox7(gold_add(state0, rc_int[r]));
        
        ulong s = gold_add(state0, s_rem);
        state0 = gold_add(s, gold_mul(d0, state0));
        state1 = gold_add(s, gold_mul(d1, state1));
        state2 = gold_add(s, gold_mul(d2, state2));
    }

    for (uint r = 4u; r < 8u; ++r) {
        state0 = sbox7(gold_add(state0, rc_ext[r * 3u + 0u]));
        state1 = sbox7(gold_add(state1, rc_ext[r * 3u + 1u]));
        state2 = sbox7(gold_add(state2, rc_ext[r * 3u + 2u]));

        ulong t0 = state0, t1 = state1, t2 = state2;
        state0 = gold_add(gold_add(gold_mul(m0, t0), gold_mul(m1, t1)), gold_mul(m2, t2));
        state1 = gold_add(gold_add(gold_mul(m3, t0), gold_mul(m4, t1)), gold_mul(m5, t2));
        state2 = gold_add(gold_add(gold_mul(m6, t0), gold_mul(m7, t1)), gold_mul(m8, t2));
    }

    tree[out_offset + p] = state0;
}
```

Incumbent result:
           f2_N64K: correct, 1.29 ms, 8.6 Gmodmul/s (int64) (16.2% of 53 Gops/s (int64 mul, est))
          f2_N256K: correct, 2.85 ms, 15.6 Gmodmul/s (int64) (29.2% of 53 Gops/s (int64 mul, est))
            f2_N1M: correct, 8.97 ms, 19.8 Gmodmul/s (int64) (37.2% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.2599

## History

- iter  0: compile=OK | correct=True | score=0.2530444956927559
- iter  1: compile=OK | correct=True | score=0.2014186337120031
- iter  2: compile=OK | correct=True | score=0.20126028888524133
- iter  3: compile=OK | correct=True | score=0.2598657192160378
- iter  4: compile=OK | correct=False | score=N/A
- iter  5: compile=OK | correct=True | score=0.20285617538622955
- iter  6: compile=OK | correct=True | score=0.20350859132709612

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
