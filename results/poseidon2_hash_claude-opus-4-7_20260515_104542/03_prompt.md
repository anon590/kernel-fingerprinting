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
constant ulong EPSILON = 0x00000000FFFFFFFFul;

constexpr constant uint T_MAX  = 4u;
constexpr constant uint RF_MAX = 8u;
constexpr constant uint RP_MAX = 32u;

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    if (t < a) t += EPSILON;          // wrap by 2^64 -> add 2^32-1
    if (t >= P_GOLD) t -= P_GOLD;
    return t;
}

// Reduce a 128-bit value (lo, hi) modulo Goldilocks. Result canonical.
inline ulong gold_reduce_128(ulong lo, ulong hi) {
    ulong x_hi_lo = hi & EPSILON;     // low 32 of hi
    ulong x_hi_hi = hi >> 32;         // high 32 of hi

    // t0 = lo - x_hi_hi  (mod 2^64), borrow handling
    ulong t0 = lo - x_hi_hi;
    if (t0 > lo) t0 -= EPSILON;

    // t1 = x_hi_lo * (2^32 - 1) = (x_hi_lo << 32) - x_hi_lo, fits in 64 bits
    ulong t1 = (x_hi_lo << 32) - x_hi_lo;

    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;       // carry

    return gold_canonical(t2);
}

// Full 64x64 -> 128 multiply, then Goldilocks reduce.
inline ulong gold_mul(ulong a, ulong b) {
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

    return gold_reduce_128(lo, hi);
}

// Multiply a*b producing (lo, hi) 128-bit product without reducing.
// Inputs: a, b in [0, 2^64). Output: (lo, hi) with lo + hi*2^64 = a*b.
inline void mul_full_128(ulong a, ulong b, thread ulong &lo, thread ulong &hi) {
    uint a0 = (uint)(a);
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)(b);
    uint b1 = (uint)(b >> 32);

    ulong p00 = (ulong)a0 * (ulong)b0;
    ulong p01 = (ulong)a0 * (ulong)b1;
    ulong p10 = (ulong)a1 * (ulong)b0;
    ulong p11 = (ulong)a1 * (ulong)b1;

    ulong mid = (p00 >> 32) + (p01 & EPSILON) + (p10 & EPSILON);
    lo = (p00 & EPSILON) | (mid << 32);
    hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}

// Add a 128-bit (alo, ahi) into accumulator (lo, hi).
inline void add128(thread ulong &lo, thread ulong &hi, ulong alo, ulong ahi) {
    ulong nl = lo + alo;
    ulong carry = (nl < lo) ? 1ul : 0ul;
    lo = nl;
    hi = hi + ahi + carry;
}

// Compute (a*b mod p) where the result is then added to acc using gold_add.
// Used for the rare case where we need a fully reduced single product.
inline ulong sbox(ulong x) {
    ulong x2 = gold_mul(x, x);
    ulong x4 = gold_mul(x2, x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
}

// Dot product of (a[0..n], b[0..n]) with lazy reduction.
// n <= 4. Each operand < p < 2^64. Each product < p^2 < 2^128.
// Sum of up to 4 such products fits in 130 bits; the top 2 bits would
// require care. p^2 < 2^127.something? p = 2^64 - 2^32 + 1, so p^2 ~ 2^128
// minus a bit. To be safe, we accumulate up to t=4 products by handling
// carry into hi properly (hi can be up to ~4*(p^2 >> 64) which still fits).
inline ulong dot4_reduce(ulong a0, ulong b0, ulong a1, ulong b1,
                         ulong a2, ulong b2, ulong a3, ulong b3, uint n) {
    ulong lo = 0ul, hi = 0ul;
    ulong pl, ph;
    if (n > 0u) { mul_full_128(a0, b0, pl, ph); add128(lo, hi, pl, ph); }
    if (n > 1u) { mul_full_128(a1, b1, pl, ph); add128(lo, hi, pl, ph); }
    if (n > 2u) { mul_full_128(a2, b2, pl, ph); add128(lo, hi, pl, ph); }
    if (n > 3u) { mul_full_128(a3, b3, pl, ph); add128(lo, hi, pl, ph); }
    // Reduce (lo, hi). hi may now have up to ~66 bits. Reduce in two steps:
    // First fold hi's upper part by repeating reduction.
    // hi < 4 * 2^64 effectively bounded; gold_reduce_128 assumes hi < 2^64.
    // Since each ph < 2^64 and we add up to 4 of them, hi sum < 4*2^64,
    // i.e. hi as a ulong overflowed at most 2 times. Track that:
    // Simpler: reduce iteratively. Each reduce_128 takes a 128-bit input and
    // outputs < p < 2^64. We process additions one product at a time.
    return gold_reduce_128(lo, hi);
}

// Safer per-product accumulator: reduce after each MAC.
// y += a*b  where y is a Goldilocks element, returning new y in [0, p).
inline ulong mac_reduce(ulong y, ulong a, ulong b) {
    ulong pl, ph;
    mul_full_128(a, b, pl, ph);
    // Add y (which is < p < 2^64) into (pl, ph).
    ulong nl = pl + y;
    ulong c  = (nl < pl) ? 1ul : 0ul;
    pl = nl;
    ph = ph + c;
    return gold_reduce_128(pl, ph);
}

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
    uint idx [[thread_position_in_grid]])
{
    if (idx >= batch) return;

    uint tt = t;
    uint rf = r_f;
    uint rp = r_p;
    uint half_f = rf >> 1u;

    // Load MDS, diag, RCs into registers.
    ulong mds[T_MAX * T_MAX];
    #pragma unroll
    for (uint i = 0u; i < T_MAX * T_MAX; ++i) {
        mds[i] = (i < tt * tt) ? ext_mds[i] : 0ul;
    }
    ulong diag[T_MAX];
    #pragma unroll
    for (uint i = 0u; i < T_MAX; ++i) {
        diag[i] = (i < tt) ? int_diag[i] : 0ul;
    }
    ulong rce[RF_MAX * T_MAX];
    for (uint i = 0u; i < rf * tt; ++i) rce[i] = rc_ext[i];
    ulong rci[RP_MAX];
    for (uint i = 0u; i < rp; ++i) rci[i] = rc_int[i];

    // Specialize t = 3.
    if (tt == 3u) {
        ulong s0 = in_state[idx * 3u + 0u];
        ulong s1 = in_state[idx * 3u + 1u];
        ulong s2 = in_state[idx * 3u + 2u];

        ulong m00 = mds[0], m01 = mds[1], m02 = mds[2];
        ulong m10 = mds[3], m11 = mds[4], m12 = mds[5];
        ulong m20 = mds[6], m21 = mds[7], m22 = mds[8];
        ulong d0 = diag[0], d1 = diag[1], d2 = diag[2];

        // Lazy-reduction matvec for t=3 using sequential mac_reduce.
        // Each row: y = m_i0*s0; y += m_i1*s1; y += m_i2*s2.
        // mac_reduce keeps the running scalar < p so the next MAC's
        // 128-bit accumulator can hold (a*b + y) safely.

        // Pre-multiply by external MDS.
        {
            ulong n0 = mac_reduce(mac_reduce(gold_mul(m00, s0), m01, s1), m02, s2);
            ulong n1 = mac_reduce(mac_reduce(gold_mul(m10, s0), m11, s1), m12, s2);
            ulong n2 = mac_reduce(mac_reduce(gold_mul(m20, s0), m21, s1), m22, s2);
            s0 = n0; s1 = n1; s2 = n2;
        }

        // First half full rounds.
        for (uint r = 0u; r < half_f; ++r) {
            s0 = sbox(gold_add(s0, rce[r * 3u + 0u]));
            s1 = sbox(gold_add(s1, rce[r * 3u + 1u]));
            s2 = sbox(gold_add(s2, rce[r * 3u + 2u]));
            ulong n0 = mac_reduce(mac_reduce(gold_mul(m00, s0), m01, s1), m02, s2);
            ulong n1 = mac_reduce(mac_reduce(gold_mul(m10, s0), m11, s1), m12, s2);
            ulong n2 = mac_reduce(mac_reduce(gold_mul(m20, s0), m21, s1), m22, s2);
            s0 = n0; s1 = n1; s2 = n2;
        }

        // Partial rounds: y_i = sum + diag_i * s_i.
        for (uint r = 0u; r < rp; ++r) {
            s0 = sbox(gold_add(s0, rci[r]));
            ulong sum = gold_add(gold_add(s0, s1), s2);
            ulong n0 = mac_reduce(sum, d0, s0);
            ulong n1 = mac_reduce(sum, d1, s1);
            ulong n2 = mac_reduce(sum, d2, s2);
            s0 = n0; s1 = n1; s2 = n2;
        }

        // Second half full rounds.
        for (uint r = half_f; r < rf; ++r) {
            s0 = sbox(gold_add(s0, rce[r * 3u + 0u]));
            s1 = sbox(gold_add(s1, rce[r * 3u + 1u]));
            s2 = sbox(gold_add(s2, rce[r * 3u + 2u]));
            ulong n0 = mac_reduce(mac_reduce(gold_mul(m00, s0), m01, s1), m02, s2);
            ulong n1 = mac_reduce(mac_reduce(gold_mul(m10, s0), m11, s1), m12, s2);
            ulong n2 = mac_reduce(mac_reduce(gold_mul(m20, s0), m21, s1), m22, s2);
            s0 = n0; s1 = n1; s2 = n2;
        }

        out_state[idx * 3u + 0u] = s0;
        out_state[idx * 3u + 1u] = s1;
        out_state[idx * 3u + 2u] = s2;
        return;
    }

    // Generic path (t != 3).
    thread ulong state[T_MAX];
    #pragma unroll
    for (uint i = 0u; i < T_MAX; ++i) {
        state[i] = (i < tt) ? in_state[idx * tt + i] : 0ul;
    }

    // Pre-multiply by external MDS.
    {
        ulong tmp[T_MAX];
        for (uint i = 0u; i < tt; ++i) {
            ulong y = gold_mul(mds[i * tt + 0u], state[0]);
            for (uint j = 1u; j < tt; ++j) {
                y = mac_reduce(y, mds[i * tt + j], state[j]);
            }
            tmp[i] = y;
        }
        for (uint i = 0u; i < tt; ++i) state[i] = tmp[i];
    }

    for (uint r = 0u; r < half_f; ++r) {
        for (uint i = 0u; i < tt; ++i) {
            state[i] = sbox(gold_add(state[i], rce[r * tt + i]));
        }
        ulong tmp[T_MAX];
        for (uint i = 0u; i < tt; ++i) {
            ulong y = gold_mul(mds[i * tt + 0u], state[0]);
            for (uint j = 1u; j < tt; ++j) {
                y = mac_reduce(y, mds[i * tt + j], state[j]);
            }
            tmp[i] = y;
        }
        for (uint i = 0u; i < tt; ++i) state[i] = tmp[i];
    }

    for (uint r = 0u; r < rp; ++r) {
        state[0] = sbox(gold_add(state[0], rci[r]));
        ulong s = 0ul;
        for (uint i = 0u; i < tt; ++i) s = gold_add(s, state[i]);
        ulong tmp[T_MAX];
        for (uint i = 0u; i < tt; ++i) {
            tmp[i] = mac_reduce(s, diag[i], state[i]);
        }
        for (uint i = 0u; i < tt; ++i) state[i] = tmp[i];
    }

    for (uint r = half_f; r < rf; ++r) {
        for (uint i = 0u; i < tt; ++i) {
            state[i] = sbox(gold_add(state[i], rce[r * tt + i]));
        }
        ulong tmp[T_MAX];
        for (uint i = 0u; i < tt; ++i) {
            ulong y = gold_mul(mds[i * tt + 0u], state[0]);
            for (uint j = 1u; j < tt; ++j) {
                y = mac_reduce(y, mds[i * tt + j], state[j]);
            }
            tmp[i] = y;
        }
        for (uint i = 0u; i < tt; ++i) state[i] = tmp[i];
    }

    for (uint i = 0u; i < tt; ++i) {
        out_state[idx * tt + i] = state[i];
    }
}
```

Result of previous attempt:
            t3_B4K: correct, 0.28 ms, 4.8 Gmodmul/s (int64) (9.1% of 53 Gops/s (int64 mul, est))
           t3_B64K: correct, 1.18 ms, 18.3 Gmodmul/s (int64) (34.4% of 53 Gops/s (int64 mul, est))
            t3_B1M: correct, 13.48 ms, 25.8 Gmodmul/s (int64) (48.3% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.2469

## History

- iter  0: compile=OK | correct=True | score=0.22313100350851453
- iter  1: compile=OK | correct=True | score=0.22945717366315468
- iter  2: compile=OK | correct=True | score=0.2468564646184756

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
