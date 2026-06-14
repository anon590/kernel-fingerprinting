## Task: poseidon2_hash

Batched Poseidon2 permutation over the Goldilocks field (p = 2^64 - 2^32 + 1, S-box alpha = 7, R_F = 8 full rounds split 4+4, R_P = 22 partial rounds). Each of ``batch`` independent sponges runs the same permutation on its own length-t state vector. The output is the full permuted state (NOT a sponge truncation): out_state[idx, :] = Permute(in_state[idx, :]).

The arity ``t``, the round-count parameters, and the round constants / MDS coefficients are all bound as device or constant buffers (see the buffer layout below); the kernel must use the runtime values rather than compile-time constants. The host always passes a t-square ``ext_mds`` and a t-length ``int_diag`` in row-major order; the internal-MDS convention is M_I = J + diag(int_diag) where J is the all-ones matrix, i.e. the per-thread internal matvec is
  y[i] = sum(state) + int_diag[i] * state[i].

The external matvec is the generic dense form: y[i] = sum_j ext_mds[i * t + j] * state[j].

Algorithm (executed by the seed):
  state <- ext_mds * state
  for r in 0..R_F/2:        # first half-full rounds
    state[i] += rc_ext[r, i] for all i
    state[i] = state[i]^7  for all i
    state <- ext_mds * state
  for r in 0..R_P:           # partial rounds
    state[0] += rc_int[r]
    state[0] = state[0]^7
    state <- (J + diag(int_diag)) * state
  for r in R_F/2..R_F:       # second half-full rounds
    (same shape as first half)

All arithmetic is in Goldilocks; bit-exact correctness against a Python bigint reference. Outputs MUST be canonical (< p); a non-canonical value with the same residue class still counts as a mismatch.

## Required kernel signature(s)

```
kernel void poseidon2_hash(
    device const ulong *in_state    [[buffer(0)]],
    device       ulong *out_state   [[buffer(1)]],
    device const ulong *rc_ext      [[buffer(2)]],
    device const ulong *rc_int      [[buffer(3)]],
    device const ulong *ext_mds     [[buffer(4)]],
    device const ulong *int_diag    [[buffer(5)]],
    constant uint      &t           [[buffer(6)]],
    constant uint      &r_f         [[buffer(7)]],
    constant uint      &r_p         [[buffer(8)]],
    constant uint      &batch       [[buffer(9)]],
    uint idx [[thread_position_in_grid]]);

Dispatch (host-fixed):
  threadsPerGrid        = (batch, 1, 1)
  threadsPerThreadgroup = (min(batch, 64), 1, 1)
Each thread runs ONE sponge end-to-end; guard against idx >= batch (the grid is rounded up to a multiple of the TG width).

All test sizes satisfy t <= 4 and R_F <= 8, R_P <= 32; thread-private state arrays of size 4 and round-constant tables of size 32 are sufficient. Threadgroup-cooperative and simdgroup schemes are valid as long as the external buffer layout above is preserved.
```

## Baseline: naive seed kernel

```metal
// Naive seed for the Poseidon2 permutation over Goldilocks (one thread per sponge).
//
// State width `t` is read at runtime from buffer(6). All round constants
// and MDS coefficients are loaded from device buffers (see layout below).
//
// Algorithm: standard Poseidon2 (Grassi-Khovratovich-Lueftenegger 2023):
//
//   state <- ext_mds * state                              # pre-mat
//   for r in 0..R_F/2:                                    # first half-full rounds
//       state[i] += rc_ext[r, i]   for i in 0..t
//       state[i]  = sbox(state[i]) for i in 0..t           # x^7
//       state    <- ext_mds * state
//   for r in 0..R_P:                                      # partial rounds
//       state[0] += rc_int[r]
//       state[0]  = sbox(state[0])
//       state    <- int_mds * state          # M_I = J + diag(d), J all-ones
//   for r in R_F/2..R_F:                                  # second half-full rounds
//       (same as first half, with rc_ext[r])
//
// The internal MDS multiplication is:
//   y[i] = sum(state) + d[i] * state[i]
//
// Buffer layout (host-fixed; preserved by candidate):
//   buffer 0: device const ulong *in_state        (batch * t)
//   buffer 1: device       ulong *out_state       (batch * t)
//   buffer 2: device const ulong *rc_ext          (r_f * t, row-major)
//   buffer 3: device const ulong *rc_int          (r_p)
//   buffer 4: device const ulong *ext_mds         (t * t, row-major)
//   buffer 5: device const ulong *int_diag        (t)
//   buffer 6: constant uint &t                    (state width, t <= 4)
//   buffer 7: constant uint &r_f                  (total full rounds, even)
//   buffer 8: constant uint &r_p                  (partial rounds)
//   buffer 9: constant uint &batch                (number of independent sponges)
//
// Dispatch (host-provided):
//   threadsPerGrid        = (batch, 1, 1)
//   threadsPerThreadgroup = (min(batch, 64), 1, 1)

#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;

constexpr constant uint T_MAX = 4u;

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    if (t < a) t += EPSILON;
    if (t >= P_GOLD) t -= P_GOLD;
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

inline ulong sbox(ulong x) {
    // x^7 = x^4 * x^2 * x
    ulong x2 = gold_mul(x, x);
    ulong x4 = gold_mul(x2, x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
}

inline void matvec_ext(thread ulong *state,
                       device const ulong *ext_mds,
                       uint t)
{
    ulong tmp[T_MAX];
    for (uint i = 0u; i < t; ++i) {
        ulong acc = 0ul;
        for (uint j = 0u; j < t; ++j) {
            acc = gold_add(acc, gold_mul(ext_mds[i * t + j], state[j]));
        }
        tmp[i] = acc;
    }
    for (uint i = 0u; i < t; ++i) state[i] = tmp[i];
}

inline void matvec_int(thread ulong *state,
                       device const ulong *int_diag,
                       uint t)
{
    // y[i] = sum(state) + d[i] * state[i]
    ulong s = 0ul;
    for (uint i = 0u; i < t; ++i) s = gold_add(s, state[i]);
    ulong tmp[T_MAX];
    for (uint i = 0u; i < t; ++i) {
        tmp[i] = gold_add(s, gold_mul(int_diag[i], state[i]));
    }
    for (uint i = 0u; i < t; ++i) state[i] = tmp[i];
}

kernel void poseidon2_hash(
    device const ulong *in_state        [[buffer(0)]],
    device       ulong *out_state       [[buffer(1)]],
    device const ulong *rc_ext          [[buffer(2)]],
    device const ulong *rc_int          [[buffer(3)]],
    device const ulong *ext_mds         [[buffer(4)]],
    device const ulong *int_diag        [[buffer(5)]],
    constant uint      &t               [[buffer(6)]],
    constant uint      &r_f             [[buffer(7)]],
    constant uint      &r_p             [[buffer(8)]],
    constant uint      &batch           [[buffer(9)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= batch) return;

    thread ulong state[T_MAX];
    for (uint i = 0u; i < t; ++i) {
        state[i] = in_state[idx * t + i];
    }

    // Pre-multiply by external MDS.
    matvec_ext(state, ext_mds, t);

    uint half_f = r_f >> 1u;

    // First half full rounds.
    for (uint r = 0u; r < half_f; ++r) {
        for (uint i = 0u; i < t; ++i) {
            state[i] = gold_add(state[i], rc_ext[r * t + i]);
            state[i] = sbox(state[i]);
        }
        matvec_ext(state, ext_mds, t);
    }

    // Partial rounds.
    for (uint r = 0u; r < r_p; ++r) {
        state[0] = gold_add(state[0], rc_int[r]);
        state[0] = sbox(state[0]);
        matvec_int(state, int_diag, t);
    }

    // Second half full rounds.
    for (uint r = half_f; r < r_f; ++r) {
        for (uint i = 0u; i < t; ++i) {
            state[i] = gold_add(state[i], rc_ext[r * t + i]);
            state[i] = sbox(state[i]);
        }
        matvec_ext(state, ext_mds, t);
    }

    for (uint i = 0u; i < t; ++i) {
        out_state[idx * t + i] = state[i];
    }
}
```

Measured baseline (seed):
            t3_B4K: correct, 0.30 ms, 4.5 Gmodmul/s (int64) (8.5% of 53 Gops/s (int64 mul, est))
           t3_B64K: correct, 0.99 ms, 21.8 Gmodmul/s (int64) (41.0% of 53 Gops/s (int64 mul, est))
            t3_B1M: correct, 15.33 ms, 22.6 Gmodmul/s (int64) (42.4% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.2456

## Your task

Write an improved Metal kernel that produces bit-exact-correct
results AND runs faster than the seed across all problem sizes. The
fitness score is the geometric mean of `achieved / ceiling` across
sizes; score 0 if any size fails bit-exact correctness.

Output ONE fenced ```metal``` code block containing the kernel(s).
Preserve the kernel name(s) and buffer indices exactly.
