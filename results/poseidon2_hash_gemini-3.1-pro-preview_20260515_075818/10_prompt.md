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

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    t += (t < a) ? 0xFFFFFFFFul : 0ul;
    return (t >= P_GOLD) ? (t - P_GOLD) : t;
}

inline ulong2 umul128(ulong a, ulong b) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    ulong p00 = (ulong)a0 * b0;
    ulong p01 = (ulong)a0 * b1;
    ulong p10 = (ulong)a1 * b0;
    ulong p11 = (ulong)a1 * b1;

    ulong mid = p10 + (uint)(p00 >> 32);
    ulong hi = p11 + (uint)(mid >> 32);
    mid = (uint)mid + p01;
    hi += (uint)(mid >> 32);
    ulong lo = (mid << 32) | (uint)p00;
    
    return ulong2(lo, hi);
}

inline ulong2 usqr128(ulong a) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);

    ulong p00 = (ulong)a0 * a0;
    ulong p01 = (ulong)a0 * a1;
    ulong p11 = (ulong)a1 * a1;

    ulong mid = p01 + (uint)(p00 >> 32);
    ulong hi = p11 + (uint)(mid >> 32);
    mid = (uint)mid + p01;
    hi += (uint)(mid >> 32);
    ulong lo = (mid << 32) | (uint)p00;
    
    return ulong2(lo, hi);
}

inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong x_hi_lo = (uint)x_hi;
    ulong x_hi_hi = x_hi >> 32;

    ulong t0 = x_lo - x_hi_hi;
    t0 -= (t0 > x_lo) ? 0xFFFFFFFFul : 0ul;

    ulong t1 = (x_hi_lo << 32) - x_hi_lo;

    ulong t2 = t0 + t1;
    t2 += (t2 < t0) ? 0xFFFFFFFFul : 0ul;

    return (t2 >= P_GOLD) ? (t2 - P_GOLD) : t2;
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong2 p = umul128(a, b);
    return gold_reduce128(p.x, p.y);
}

inline ulong gold_sqr(ulong a) {
    ulong2 p = usqr128(a);
    return gold_reduce128(p.x, p.y);
}

inline ulong sbox(ulong x) {
    ulong x2 = gold_sqr(x);
    ulong x4 = gold_sqr(x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
}

template <uint T>
inline void apply_matvec_ext(thread ulong *state, device const ulong *mds) {
    ulong tmp[T];
    #pragma unroll
    for (uint i = 0; i < T; ++i) {
        ulong sum_px = 0;
        ulong sum_t1 = 0;
        ulong sum_hi_hi = 0;
        
        #pragma unroll
        for (uint j = 0; j < T; ++j) {
            ulong2 p = umul128(mds[i * T + j], state[j]);
            
            ulong px = p.x;
            sum_px += px;
            sum_px += (sum_px < px) ? 0xFFFFFFFFul : 0ul;
            
            ulong py = p.y;
            sum_hi_hi += py >> 32;
            
            ulong x_hi_lo = (uint)py;
            ulong t1 = (x_hi_lo << 32) - x_hi_lo;
            sum_t1 += t1;
            sum_t1 += (sum_t1 < t1) ? 0xFFFFFFFFul : 0ul;
        }
        
        ulong total = sum_px + sum_t1;
        total += (total < sum_px) ? 0xFFFFFFFFul : 0ul;
        
        ulong prev_total = total;
        total -= sum_hi_hi;
        total -= (total > prev_total) ? 0xFFFFFFFFul : 0ul;
        
        tmp[i] = (total >= P_GOLD) ? (total - P_GOLD) : total;
    }
    #pragma unroll
    for (uint i = 0; i < T; ++i) state[i] = tmp[i];
}

template <uint T>
inline void apply_matvec_int(thread ulong *state, device const ulong *diag) {
    ulong sum_state = 0;
    #pragma unroll
    for (uint i = 0; i < T; ++i) {
        ulong st = state[i];
        sum_state += st;
        sum_state += (sum_state < st) ? 0xFFFFFFFFul : 0ul;
    }
    
    ulong tmp[T];
    #pragma unroll
    for (uint i = 0; i < T; ++i) {
        ulong2 p = umul128(diag[i], state[i]);
        
        ulong px = p.x;
        ulong sum_px = px + sum_state;
        sum_px += (sum_px < px) ? 0xFFFFFFFFul : 0ul;
        
        ulong py = p.y;
        ulong x_hi_lo = (uint)py;
        ulong t1 = (x_hi_lo << 32) - x_hi_lo;
        
        ulong total = sum_px + t1;
        total += (total < sum_px) ? 0xFFFFFFFFul : 0ul;
        
        ulong hi_hi = py >> 32;
        ulong prev_total = total;
        total -= hi_hi;
        total -= (total > prev_total) ? 0xFFFFFFFFul : 0ul;
        
        tmp[i] = (total >= P_GOLD) ? (total - P_GOLD) : total;
    }
    #pragma unroll
    for (uint i = 0; i < T; ++i) state[i] = tmp[i];
}

template <uint T>
inline void process_sponge(
    device const ulong *in_state,
    device       ulong *out_state,
    device const ulong *rc_ext,
    device const ulong *rc_int,
    device const ulong *ext_mds,
    device const ulong *int_diag,
    uint r_f,
    uint r_p,
    uint idx)
{
    ulong state[T];
    #pragma unroll
    for (uint i = 0; i < T; ++i) {
        state[i] = in_state[idx * T + i];
    }

    // Pre-matvec directly consumes from device const memory 
    apply_matvec_ext<T>(state, ext_mds);

    uint half_f = r_f >> 1;
    device const ulong *rc_ext_ptr = rc_ext;

    // Guaranteed unrolling of R_F <= 8
    #pragma unroll(4)
    for (uint r = 0; r < 4; ++r) {
        if (r >= half_f) break;
        #pragma unroll
        for (uint i = 0; i < T; ++i) {
            state[i] = sbox(gold_add(state[i], rc_ext_ptr[i]));
        }
        rc_ext_ptr += T;
        apply_matvec_ext<T>(state, ext_mds);
    }

    device const ulong *rc_int_ptr = rc_int;
    
    // Guaranteed unrolling of R_P <= 32
    #pragma unroll(32)
    for (uint r = 0; r < 32; ++r) {
        if (r >= r_p) break;
        state[0] = sbox(gold_add(state[0], rc_int_ptr[r]));
        apply_matvec_int<T>(state, int_diag);
    }

    // Unroll remainder of R_F
    #pragma unroll(4)
    for (uint r = 0; r < 4; ++r) {
        if (r >= r_f - half_f) break;
        #pragma unroll
        for (uint i = 0; i < T; ++i) {
            state[i] = sbox(gold_add(state[i], rc_ext_ptr[i]));
        }
        rc_ext_ptr += T;
        apply_matvec_ext<T>(state, ext_mds);
    }

    #pragma unroll
    for (uint i = 0; i < T; ++i) {
        out_state[idx * T + i] = state[i];
    }
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

    // Exploit template instantiations to fix `T` per invocation
    switch (t) {
        case 1: process_sponge<1>(in_state, out_state, rc_ext, rc_int, ext_mds, int_diag, r_f, r_p, idx); break;
        case 2: process_sponge<2>(in_state, out_state, rc_ext, rc_int, ext_mds, int_diag, r_f, r_p, idx); break;
        case 3: process_sponge<3>(in_state, out_state, rc_ext, rc_int, ext_mds, int_diag, r_f, r_p, idx); break;
        case 4: process_sponge<4>(in_state, out_state, rc_ext, rc_int, ext_mds, int_diag, r_f, r_p, idx); break;
    }
}
```

Result of previous attempt:
            t3_B4K: correct, 0.39 ms, 3.5 Gmodmul/s (int64) (0.6% of 562 Gops/s (int64 mul, est))
           t3_B64K: correct, 0.89 ms, 24.4 Gmodmul/s (int64) (4.3% of 562 Gops/s (int64 mul, est))
            t3_B1M: correct, 13.16 ms, 26.4 Gmodmul/s (int64) (4.7% of 562 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0233

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;

constexpr constant uint T_MAX = 4u;

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    t += (t < a) ? 0xFFFFFFFFul : 0ul;
    return (t >= P_GOLD) ? (t - P_GOLD) : t;
}

inline ulong2 umul128(ulong a, ulong b) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    ulong p00 = (ulong)a0 * b0;
    ulong p01 = (ulong)a0 * b1;
    ulong p10 = (ulong)a1 * b0;
    ulong p11 = (ulong)a1 * b1;

    ulong mid = p10 + (uint)(p00 >> 32);
    ulong hi = p11 + (uint)(mid >> 32);
    mid = (uint)mid + p01;
    hi += (uint)(mid >> 32);
    ulong lo = (mid << 32) | (uint)p00;
    
    return ulong2(lo, hi);
}

// Optimized 128-bit squaring: removes one cross-multiplication
inline ulong2 usqr128(ulong a) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);

    ulong p00 = (ulong)a0 * a0;
    ulong p01 = (ulong)a0 * a1;
    ulong p11 = (ulong)a1 * a1;

    ulong mid = p01 + (uint)(p00 >> 32);
    ulong hi = p11 + (uint)(mid >> 32);
    mid = (uint)mid + p01;
    hi += (uint)(mid >> 32);
    ulong lo = (mid << 32) | (uint)p00;
    
    return ulong2(lo, hi);
}

inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong x_hi_lo = (uint)x_hi;
    ulong x_hi_hi = x_hi >> 32;

    ulong t0 = x_lo - x_hi_hi;
    t0 -= (t0 > x_lo) ? 0xFFFFFFFFul : 0ul;

    ulong t1 = (x_hi_lo << 32) - x_hi_lo;

    ulong t2 = t0 + t1;
    t2 += (t2 < t0) ? 0xFFFFFFFFul : 0ul;

    return (t2 >= P_GOLD) ? (t2 - P_GOLD) : t2;
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong2 p = umul128(a, b);
    return gold_reduce128(p.x, p.y);
}

inline ulong gold_sqr(ulong a) {
    ulong2 p = usqr128(a);
    return gold_reduce128(p.x, p.y);
}

inline ulong sbox(ulong x) {
    // x^7 = x^4 * x^2 * x
    ulong x2 = gold_sqr(x);
    ulong x4 = gold_sqr(x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
}

template <uint T>
inline void apply_matvec_ext(thread ulong *state, thread const ulong *mds) {
    ulong tmp[T];
    #pragma unroll
    for (uint i = 0; i < T; ++i) {
        ulong acc_lo = 0;
        ulong acc_hi = 0;
        uint acc_top = 0;
        
        #pragma unroll
        for (uint j = 0; j < T; ++j) {
            ulong2 p = umul128(mds[i * T + j], state[j]);
            
            ulong next_lo = acc_lo + p.x;
            uint c1 = (next_lo < acc_lo) ? 1 : 0;
            acc_lo = next_lo;
            
            ulong next_hi = acc_hi + p.y;
            uint c2 = (next_hi < acc_hi) ? 1 : 0;
            next_hi += c1;
            c2 += (next_hi < (ulong)c1) ? 1 : 0;
            acc_hi = next_hi;
            
            acc_top += c2;
        }
        
        // 2^128 = -2^32 mod P. Adjust single reduction with accumulated overflow wraps.
        ulong rem = gold_reduce128(acc_lo, acc_hi);
        ulong sub = (ulong)acc_top << 32;
        if (rem >= sub) {
            rem -= sub;
        } else {
            rem = rem + P_GOLD - sub;
        }
        tmp[i] = rem;
    }
    #pragma unroll
    for (uint i = 0; i < T; ++i) state[i] = tmp[i];
}

template <uint T>
inline void apply_matvec_int(thread ulong *state, thread const ulong *diag) {
    ulong sum = 0;
    #pragma unroll
    for (uint i = 0; i < T; ++i) {
        sum = gold_add(sum, state[i]);
    }
    
    ulong tmp[T];
    #pragma unroll
    for (uint i = 0; i < T; ++i) {
        ulong2 p = umul128(diag[i], state[i]);
        // Fold the row sum before fully reducing (safely fits in 128 bits)
        ulong lo = p.x + sum;
        ulong hi = p.y + ((lo < p.x) ? 1 : 0);
        tmp[i] = gold_reduce128(lo, hi);
    }
    #pragma unroll
    for (uint i = 0; i < T; ++i) state[i] = tmp[i];
}

template <uint T>
inline void process_sponge(
    device const ulong *in_state,
    device       ulong *out_state,
    device const ulong *rc_ext,
    device const ulong *rc_int,
    device const ulong *ext_mds,
    device const ulong *int_diag,
    uint r_f,
    uint r_p,
    uint idx)
{
    ulong state[T];
    #pragma unroll
    for (uint i = 0; i < T; ++i) {
        state[i] = in_state[idx * T + i];
    }

    // Cache to registers for tight access spanning multiple rounds
    ulong mds[T * T];
    #pragma unroll
    for (uint i = 0; i < T * T; ++i) {
        mds[i] = ext_mds[i];
    }

    ulong diag[T];
    #pragma unroll
    for (uint i = 0; i < T; ++i) {
        diag[i] = int_diag[i];
    }

    // Pre-matvec
    apply_matvec_ext<T>(state, mds);

    uint half_f = r_f >> 1;
    device const ulong *rc_ext_ptr = rc_ext;

    // First half full rounds
    for (uint r = 0; r < half_f; ++r) {
        #pragma unroll
        for (uint i = 0; i < T; ++i) {
            state[i] = sbox(gold_add(state[i], rc_ext_ptr[i]));
        }
        rc_ext_ptr += T;
        apply_matvec_ext<T>(state, mds);
    }

    device const ulong *rc_int_ptr = rc_int;
    
    // Partial rounds
    for (uint r = 0; r < r_p; ++r) {
        state[0] = sbox(gold_add(state[0], rc_int_ptr[r]));
        apply_matvec_int<T>(state, diag);
    }

    // Second half full rounds
    for (uint r = half_f; r < r_f; ++r) {
        #pragma unroll
        for (uint i = 0; i < T; ++i) {
            state[i] = sbox(gold_add(state[i], rc_ext_ptr[i]));
        }
        rc_ext_ptr += T;
        apply_matvec_ext<T>(state, mds);
    }

    #pragma unroll
    for (uint i = 0; i < T; ++i) {
        out_state[idx * T + i] = state[i];
    }
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

    // Compile-time unrolling via branching
    switch (t) {
        case 1: process_sponge<1>(in_state, out_state, rc_ext, rc_int, ext_mds, int_diag, r_f, r_p, idx); break;
        case 2: process_sponge<2>(in_state, out_state, rc_ext, rc_int, ext_mds, int_diag, r_f, r_p, idx); break;
        case 3: process_sponge<3>(in_state, out_state, rc_ext, rc_int, ext_mds, int_diag, r_f, r_p, idx); break;
        case 4: process_sponge<4>(in_state, out_state, rc_ext, rc_int, ext_mds, int_diag, r_f, r_p, idx); break;
    }
}
```

Incumbent result:
            t3_B4K: correct, 0.21 ms, 6.6 Gmodmul/s (int64) (1.2% of 562 Gops/s (int64 mul, est))
           t3_B64K: correct, 0.76 ms, 28.4 Gmodmul/s (int64) (5.0% of 562 Gops/s (int64 mul, est))
            t3_B1M: correct, 11.79 ms, 29.4 Gmodmul/s (int64) (5.2% of 562 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0314

## History

- iter  2: compile=OK | correct=True | score=0.025127508237680556
- iter  3: compile=OK | correct=True | score=0.024308146166597093
- iter  4: compile=OK | correct=True | score=0.025078272171242803
- iter  5: compile=OK | correct=True | score=0.024563899554657292
- iter  6: compile=OK | correct=True | score=0.028415924429327454
- iter  7: compile=OK | correct=True | score=0.03139211244518381
- iter  8: compile=OK | correct=False | score=N/A
- iter  9: compile=OK | correct=True | score=0.02329137946717296

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
