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

## Your previous attempt

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0xFFFFFFFFul;

constexpr constant uint T_MAX = 4u;

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    ulong carry = (t < a) ? EPSILON : 0ul;
    t += carry;
    ulong over = (t >= P_GOLD) ? P_GOLD : 0ul;
    t -= over;
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

    ulong mid = (p00 >> 32) + (uint)p01 + (uint)p10;
    ulong lo  = ((mid & EPSILON) << 32) | (uint)p00;
    ulong hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);

    ulong x_hi_lo = (uint)hi;
    ulong x_hi_hi = hi >> 32;

    ulong t0 = lo - x_hi_hi;
    ulong under = (t0 > lo) ? EPSILON : 0ul;
    t0 -= under;

    ulong t1 = (x_hi_lo << 32) - x_hi_lo;

    ulong t2 = t0 + t1;
    ulong carry = (t2 < t0) ? EPSILON : 0ul;
    t2 += carry;

    ulong over = (t2 >= P_GOLD) ? P_GOLD : 0ul;
    t2 -= over;

    return t2;
}

inline ulong sbox(ulong x) {
    ulong x2 = gold_mul(x, x);
    ulong x4 = gold_mul(x2, x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
}

template <uint t>
inline void matvec_ext_unrolled(thread ulong& s0, thread ulong& s1, thread ulong& s2, thread ulong& s3,
                                threadgroup const ulong* mds) {
    if (t == 4) {
        ulong n0 = gold_add(gold_add(gold_mul(mds[0], s0), gold_mul(mds[1], s1)), gold_add(gold_mul(mds[2], s2), gold_mul(mds[3], s3)));
        ulong n1 = gold_add(gold_add(gold_mul(mds[4], s0), gold_mul(mds[5], s1)), gold_add(gold_mul(mds[6], s2), gold_mul(mds[7], s3)));
        ulong n2 = gold_add(gold_add(gold_mul(mds[8], s0), gold_mul(mds[9], s1)), gold_add(gold_mul(mds[10], s2), gold_mul(mds[11], s3)));
        ulong n3 = gold_add(gold_add(gold_mul(mds[12], s0), gold_mul(mds[13], s1)), gold_add(gold_mul(mds[14], s2), gold_mul(mds[15], s3)));
        s0 = n0; s1 = n1; s2 = n2; s3 = n3;
    } else if (t == 3) {
        ulong n0 = gold_add(gold_add(gold_mul(mds[0], s0), gold_mul(mds[1], s1)), gold_mul(mds[2], s2));
        ulong n1 = gold_add(gold_add(gold_mul(mds[3], s0), gold_mul(mds[4], s1)), gold_mul(mds[5], s2));
        ulong n2 = gold_add(gold_add(gold_mul(mds[6], s0), gold_mul(mds[7], s1)), gold_mul(mds[8], s2));
        s0 = n0; s1 = n1; s2 = n2;
    } else if (t == 2) {
        ulong n0 = gold_add(gold_mul(mds[0], s0), gold_mul(mds[1], s1));
        ulong n1 = gold_add(gold_mul(mds[2], s0), gold_mul(mds[3], s1));
        s0 = n0; s1 = n1;
    } else if (t == 1) {
        s0 = gold_mul(mds[0], s0);
    }
}

template <uint t>
inline void matvec_int_unrolled(thread ulong& s0, thread ulong& s1, thread ulong& s2, thread ulong& s3,
                                threadgroup const ulong* diag) {
    if (t == 4) {
        ulong sum = gold_add(gold_add(s0, s1), gold_add(s2, s3));
        ulong n0 = gold_add(sum, gold_mul(diag[0], s0));
        ulong n1 = gold_add(sum, gold_mul(diag[1], s1));
        ulong n2 = gold_add(sum, gold_mul(diag[2], s2));
        ulong n3 = gold_add(sum, gold_mul(diag[3], s3));
        s0 = n0; s1 = n1; s2 = n2; s3 = n3;
    } else if (t == 3) {
        ulong sum = gold_add(gold_add(s0, s1), s2);
        ulong n0 = gold_add(sum, gold_mul(diag[0], s0));
        ulong n1 = gold_add(sum, gold_mul(diag[1], s1));
        ulong n2 = gold_add(sum, gold_mul(diag[2], s2));
        s0 = n0; s1 = n1; s2 = n2;
    } else if (t == 2) {
        ulong sum = gold_add(s0, s1);
        ulong n0 = gold_add(sum, gold_mul(diag[0], s0));
        ulong n1 = gold_add(sum, gold_mul(diag[1], s1));
        s0 = n0; s1 = n1;
    } else if (t == 1) {
        ulong sum = s0;
        s0 = gold_add(sum, gold_mul(diag[0], s0));
    }
}

template <uint t>
inline void process_sponge(device const ulong* in_state,
                           device       ulong* out_state,
                           threadgroup const ulong* tg_rc_ext,
                           threadgroup const ulong* tg_rc_int,
                           threadgroup const ulong* tg_ext_mds,
                           threadgroup const ulong* tg_int_diag,
                           uint r_f, uint r_p)
{
    ulong s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    if (t > 0) s0 = in_state[0];
    if (t > 1) s1 = in_state[1];
    if (t > 2) s2 = in_state[2];
    if (t > 3) s3 = in_state[3];

    matvec_ext_unrolled<t>(s0, s1, s2, s3, tg_ext_mds);

    uint half_f = r_f >> 1;

    for (uint r = 0; r < half_f; ++r) {
        if (t > 0) s0 = sbox(gold_add(s0, tg_rc_ext[r * t + 0]));
        if (t > 1) s1 = sbox(gold_add(s1, tg_rc_ext[r * t + 1]));
        if (t > 2) s2 = sbox(gold_add(s2, tg_rc_ext[r * t + 2]));
        if (t > 3) s3 = sbox(gold_add(s3, tg_rc_ext[r * t + 3]));
        matvec_ext_unrolled<t>(s0, s1, s2, s3, tg_ext_mds);
    }

    for (uint r = 0; r < r_p; ++r) {
        s0 = sbox(gold_add(s0, tg_rc_int[r]));
        matvec_int_unrolled<t>(s0, s1, s2, s3, tg_int_diag);
    }

    for (uint r = half_f; r < r_f; ++r) {
        if (t > 0) s0 = sbox(gold_add(s0, tg_rc_ext[r * t + 0]));
        if (t > 1) s1 = sbox(gold_add(s1, tg_rc_ext[r * t + 1]));
        if (t > 2) s2 = sbox(gold_add(s2, tg_rc_ext[r * t + 2]));
        if (t > 3) s3 = sbox(gold_add(s3, tg_rc_ext[r * t + 3]));
        matvec_ext_unrolled<t>(s0, s1, s2, s3, tg_ext_mds);
    }

    if (t > 0) out_state[0] = s0;
    if (t > 1) out_state[1] = s1;
    if (t > 2) out_state[2] = s2;
    if (t > 3) out_state[3] = s3;
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
    uint idx [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgsz [[threads_per_threadgroup]])
{
    uint t_loc = t;
    uint r_f_loc = r_f;
    uint r_p_loc = r_p;

    threadgroup ulong tg_ext_mds[T_MAX * T_MAX];
    threadgroup ulong tg_int_diag[T_MAX];
    threadgroup ulong tg_rc_ext[8 * T_MAX];
    threadgroup ulong tg_rc_int[32];

    for (uint i = tid; i < t_loc * t_loc; i += tgsz) tg_ext_mds[i] = ext_mds[i];
    for (uint i = tid; i < t_loc; i += tgsz)         tg_int_diag[i] = int_diag[i];
    for (uint i = tid; i < r_f_loc * t_loc; i += tgsz) tg_rc_ext[i] = rc_ext[i];
    for (uint i = tid; i < r_p_loc; i += tgsz)         tg_rc_int[i] = rc_int[i];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (idx >= batch) return;

    if (t_loc == 4) {
        process_sponge<4>(in_state + idx * 4, out_state + idx * 4, tg_rc_ext, tg_rc_int, tg_ext_mds, tg_int_diag, r_f_loc, r_p_loc);
    } else if (t_loc == 3) {
        process_sponge<3>(in_state + idx * 3, out_state + idx * 3, tg_rc_ext, tg_rc_int, tg_ext_mds, tg_int_diag, r_f_loc, r_p_loc);
    } else if (t_loc == 2) {
        process_sponge<2>(in_state + idx * 2, out_state + idx * 2, tg_rc_ext, tg_rc_int, tg_ext_mds, tg_int_diag, r_f_loc, r_p_loc);
    } else if (t_loc == 1) {
        process_sponge<1>(in_state + idx * 1, out_state + idx * 1, tg_rc_ext, tg_rc_int, tg_ext_mds, tg_int_diag, r_f_loc, r_p_loc);
    }
}
```

Result of previous attempt:
            t3_B4K: correct, 0.24 ms, 5.6 Gmodmul/s (int64) (1.0% of 562 Gops/s (int64 mul, est))
           t3_B64K: correct, 1.44 ms, 15.0 Gmodmul/s (int64) (2.7% of 562 Gops/s (int64 mul, est))
            t3_B1M: correct, 13.36 ms, 26.0 Gmodmul/s (int64) (4.6% of 562 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0230

## Current best (incumbent)

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

Incumbent result:
            t3_B4K: correct, 0.15 ms, 8.8 Gmodmul/s (int64) (1.6% of 562 Gops/s (int64 mul, est))
           t3_B64K: correct, 1.01 ms, 21.4 Gmodmul/s (int64) (3.8% of 562 Gops/s (int64 mul, est))
            t3_B1M: correct, 15.28 ms, 22.7 Gmodmul/s (int64) (4.0% of 562 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0289

## History

- iter  0: compile=OK | correct=True | score=0.028874123021347668
- iter  1: compile=OK | correct=True | score=0.023034023271217643

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
