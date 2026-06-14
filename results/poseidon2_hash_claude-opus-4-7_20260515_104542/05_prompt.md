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

constexpr constant uint T_MAX = 4u;

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    if (t < a) t += EPSILON;          // wrapped: add 2^64 mod p = EPSILON
    if (t >= P_GOLD) t -= P_GOLD;
    return t;
}

// Reduce (lo + hi*2^64) mod p. Result canonical (< p).
inline ulong gold_reduce_128(ulong lo, ulong hi) {
    ulong x_hi_lo = hi & EPSILON;
    ulong x_hi_hi = hi >> 32;

    ulong t0 = lo - x_hi_hi;
    if (t0 > lo) t0 -= EPSILON;

    ulong t1 = (x_hi_lo << 32) - x_hi_lo;

    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;

    return gold_canonical(t2);
}

inline void mul_full_128(ulong a, ulong b, thread ulong &lo, thread ulong &hi) {
    uint a0 = (uint)(a);
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)(b);
    uint b1 = (uint)(b >> 32);

    ulong p00 = (ulong)a0 * (ulong)b0;
    ulong p01 = (ulong)a0 * (ulong)b1;
    ulong p10 = (ulong)a1 * (ulong)b0;
    ulong p11 = (ulong)a1 * (ulong)b1;

    ulong mid_ = (p00 >> 32) + (p01 & EPSILON) + (p10 & EPSILON);
    lo = (p00 & EPSILON) | (mid_ << 32);
    hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid_ >> 32);
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong lo, hi;
    mul_full_128(a, b, lo, hi);
    return gold_reduce_128(lo, hi);
}

inline ulong sbox(ulong x) {
    ulong x2 = gold_mul(x, x);
    ulong x4 = gold_mul(x2, x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
}

// y = sum + diag*s, where sum < p. One full mul + add then reduce.
inline ulong sum_plus_mul(ulong sum, ulong diag_i, ulong s_i) {
    ulong pl, ph;
    mul_full_128(diag_i, s_i, pl, ph);
    ulong nl = pl + sum;
    if (nl < pl) ph += 1ul;
    return gold_reduce_128(nl, ph);
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

    // ============== Specialized t=3 ==============
    if (tt == 3u) {
        ulong m00 = ext_mds[0], m01 = ext_mds[1], m02 = ext_mds[2];
        ulong m10 = ext_mds[3], m11 = ext_mds[4], m12 = ext_mds[5];
        ulong m20 = ext_mds[6], m21 = ext_mds[7], m22 = ext_mds[8];
        ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2];

        ulong s0 = in_state[idx * 3u + 0u];
        ulong s1 = in_state[idx * 3u + 1u];
        ulong s2 = in_state[idx * 3u + 2u];

        // ext_mds matvec via reduced products + gold_add
        {
            ulong p00 = gold_mul(m00, s0), p01 = gold_mul(m01, s1), p02 = gold_mul(m02, s2);
            ulong p10 = gold_mul(m10, s0), p11 = gold_mul(m11, s1), p12 = gold_mul(m12, s2);
            ulong p20 = gold_mul(m20, s0), p21 = gold_mul(m21, s1), p22 = gold_mul(m22, s2);
            s0 = gold_add(gold_add(p00, p01), p02);
            s1 = gold_add(gold_add(p10, p11), p12);
            s2 = gold_add(gold_add(p20, p21), p22);
        }

        for (uint r = 0u; r < half_f; ++r) {
            s0 = sbox(gold_add(s0, rc_ext[r * 3u + 0u]));
            s1 = sbox(gold_add(s1, rc_ext[r * 3u + 1u]));
            s2 = sbox(gold_add(s2, rc_ext[r * 3u + 2u]));
            ulong p00 = gold_mul(m00, s0), p01 = gold_mul(m01, s1), p02 = gold_mul(m02, s2);
            ulong p10 = gold_mul(m10, s0), p11 = gold_mul(m11, s1), p12 = gold_mul(m12, s2);
            ulong p20 = gold_mul(m20, s0), p21 = gold_mul(m21, s1), p22 = gold_mul(m22, s2);
            s0 = gold_add(gold_add(p00, p01), p02);
            s1 = gold_add(gold_add(p10, p11), p12);
            s2 = gold_add(gold_add(p20, p21), p22);
        }

        for (uint r = 0u; r < rp; ++r) {
            s0 = sbox(gold_add(s0, rc_int[r]));
            ulong sum = gold_add(gold_add(s0, s1), s2);
            ulong n0 = sum_plus_mul(sum, d0, s0);
            ulong n1 = sum_plus_mul(sum, d1, s1);
            ulong n2 = sum_plus_mul(sum, d2, s2);
            s0 = n0; s1 = n1; s2 = n2;
        }

        for (uint r = half_f; r < rf; ++r) {
            s0 = sbox(gold_add(s0, rc_ext[r * 3u + 0u]));
            s1 = sbox(gold_add(s1, rc_ext[r * 3u + 1u]));
            s2 = sbox(gold_add(s2, rc_ext[r * 3u + 2u]));
            ulong p00 = gold_mul(m00, s0), p01 = gold_mul(m01, s1), p02 = gold_mul(m02, s2);
            ulong p10 = gold_mul(m10, s0), p11 = gold_mul(m11, s1), p12 = gold_mul(m12, s2);
            ulong p20 = gold_mul(m20, s0), p21 = gold_mul(m21, s1), p22 = gold_mul(m22, s2);
            s0 = gold_add(gold_add(p00, p01), p02);
            s1 = gold_add(gold_add(p10, p11), p12);
            s2 = gold_add(gold_add(p20, p21), p22);
        }

        out_state[idx * 3u + 0u] = s0;
        out_state[idx * 3u + 1u] = s1;
        out_state[idx * 3u + 2u] = s2;
        return;
    }

    // ============== Specialized t=2 ==============
    if (tt == 2u) {
        ulong m00 = ext_mds[0], m01 = ext_mds[1];
        ulong m10 = ext_mds[2], m11 = ext_mds[3];
        ulong d0 = int_diag[0], d1 = int_diag[1];

        ulong s0 = in_state[idx * 2u + 0u];
        ulong s1 = in_state[idx * 2u + 1u];

        {
            ulong p00 = gold_mul(m00, s0), p01 = gold_mul(m01, s1);
            ulong p10 = gold_mul(m10, s0), p11 = gold_mul(m11, s1);
            s0 = gold_add(p00, p01);
            s1 = gold_add(p10, p11);
        }
        for (uint r = 0u; r < half_f; ++r) {
            s0 = sbox(gold_add(s0, rc_ext[r * 2u + 0u]));
            s1 = sbox(gold_add(s1, rc_ext[r * 2u + 1u]));
            ulong p00 = gold_mul(m00, s0), p01 = gold_mul(m01, s1);
            ulong p10 = gold_mul(m10, s0), p11 = gold_mul(m11, s1);
            s0 = gold_add(p00, p01);
            s1 = gold_add(p10, p11);
        }
        for (uint r = 0u; r < rp; ++r) {
            s0 = sbox(gold_add(s0, rc_int[r]));
            ulong sum = gold_add(s0, s1);
            ulong n0 = sum_plus_mul(sum, d0, s0);
            ulong n1 = sum_plus_mul(sum, d1, s1);
            s0 = n0; s1 = n1;
        }
        for (uint r = half_f; r < rf; ++r) {
            s0 = sbox(gold_add(s0, rc_ext[r * 2u + 0u]));
            s1 = sbox(gold_add(s1, rc_ext[r * 2u + 1u]));
            ulong p00 = gold_mul(m00, s0), p01 = gold_mul(m01, s1);
            ulong p10 = gold_mul(m10, s0), p11 = gold_mul(m11, s1);
            s0 = gold_add(p00, p01);
            s1 = gold_add(p10, p11);
        }
        out_state[idx * 2u + 0u] = s0;
        out_state[idx * 2u + 1u] = s1;
        return;
    }

    // ============== Specialized t=4 ==============
    if (tt == 4u) {
        ulong m00 = ext_mds[0],  m01 = ext_mds[1],  m02 = ext_mds[2],  m03 = ext_mds[3];
        ulong m10 = ext_mds[4],  m11 = ext_mds[5],  m12 = ext_mds[6],  m13 = ext_mds[7];
        ulong m20 = ext_mds[8],  m21 = ext_mds[9],  m22 = ext_mds[10], m23 = ext_mds[11];
        ulong m30 = ext_mds[12], m31 = ext_mds[13], m32 = ext_mds[14], m33 = ext_mds[15];
        ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2], d3 = int_diag[3];

        ulong s0 = in_state[idx * 4u + 0u];
        ulong s1 = in_state[idx * 4u + 1u];
        ulong s2 = in_state[idx * 4u + 2u];
        ulong s3 = in_state[idx * 4u + 3u];

        {
            ulong a0 = gold_add(gold_mul(m00, s0), gold_mul(m01, s1));
            ulong b0 = gold_add(gold_mul(m02, s2), gold_mul(m03, s3));
            ulong a1 = gold_add(gold_mul(m10, s0), gold_mul(m11, s1));
            ulong b1 = gold_add(gold_mul(m12, s2), gold_mul(m13, s3));
            ulong a2 = gold_add(gold_mul(m20, s0), gold_mul(m21, s1));
            ulong b2 = gold_add(gold_mul(m22, s2), gold_mul(m23, s3));
            ulong a3 = gold_add(gold_mul(m30, s0), gold_mul(m31, s1));
            ulong b3 = gold_add(gold_mul(m32, s2), gold_mul(m33, s3));
            s0 = gold_add(a0, b0);
            s1 = gold_add(a1, b1);
            s2 = gold_add(a2, b2);
            s3 = gold_add(a3, b3);
        }
        for (uint r = 0u; r < half_f; ++r) {
            s0 = sbox(gold_add(s0, rc_ext[r * 4u + 0u]));
            s1 = sbox(gold_add(s1, rc_ext[r * 4u + 1u]));
            s2 = sbox(gold_add(s2, rc_ext[r * 4u + 2u]));
            s3 = sbox(gold_add(s3, rc_ext[r * 4u + 3u]));
            ulong a0 = gold_add(gold_mul(m00, s0), gold_mul(m01, s1));
            ulong b0 = gold_add(gold_mul(m02, s2), gold_mul(m03, s3));
            ulong a1 = gold_add(gold_mul(m10, s0), gold_mul(m11, s1));
            ulong b1 = gold_add(gold_mul(m12, s2), gold_mul(m13, s3));
            ulong a2 = gold_add(gold_mul(m20, s0), gold_mul(m21, s1));
            ulong b2 = gold_add(gold_mul(m22, s2), gold_mul(m23, s3));
            ulong a3 = gold_add(gold_mul(m30, s0), gold_mul(m31, s1));
            ulong b3 = gold_add(gold_mul(m32, s2), gold_mul(m33, s3));
            s0 = gold_add(a0, b0);
            s1 = gold_add(a1, b1);
            s2 = gold_add(a2, b2);
            s3 = gold_add(a3, b3);
        }
        for (uint r = 0u; r < rp; ++r) {
            s0 = sbox(gold_add(s0, rc_int[r]));
            ulong sum = gold_add(gold_add(s0, s1), gold_add(s2, s3));
            ulong n0 = sum_plus_mul(sum, d0, s0);
            ulong n1 = sum_plus_mul(sum, d1, s1);
            ulong n2 = sum_plus_mul(sum, d2, s2);
            ulong n3 = sum_plus_mul(sum, d3, s3);
            s0 = n0; s1 = n1; s2 = n2; s3 = n3;
        }
        for (uint r = half_f; r < rf; ++r) {
            s0 = sbox(gold_add(s0, rc_ext[r * 4u + 0u]));
            s1 = sbox(gold_add(s1, rc_ext[r * 4u + 1u]));
            s2 = sbox(gold_add(s2, rc_ext[r * 4u + 2u]));
            s3 = sbox(gold_add(s3, rc_ext[r * 4u + 3u]));
            ulong a0 = gold_add(gold_mul(m00, s0), gold_mul(m01, s1));
            ulong b0 = gold_add(gold_mul(m02, s2), gold_mul(m03, s3));
            ulong a1 = gold_add(gold_mul(m10, s0), gold_mul(m11, s1));
            ulong b1 = gold_add(gold_mul(m12, s2), gold_mul(m13, s3));
            ulong a2 = gold_add(gold_mul(m20, s0), gold_mul(m21, s1));
            ulong b2 = gold_add(gold_mul(m22, s2), gold_mul(m23, s3));
            ulong a3 = gold_add(gold_mul(m30, s0), gold_mul(m31, s1));
            ulong b3 = gold_add(gold_mul(m32, s2), gold_mul(m33, s3));
            s0 = gold_add(a0, b0);
            s1 = gold_add(a1, b1);
            s2 = gold_add(a2, b2);
            s3 = gold_add(a3, b3);
        }
        out_state[idx * 4u + 0u] = s0;
        out_state[idx * 4u + 1u] = s1;
        out_state[idx * 4u + 2u] = s2;
        out_state[idx * 4u + 3u] = s3;
        return;
    }

    // ============== Generic fallback ==============
    ulong state[T_MAX];
    for (uint i = 0u; i < tt; ++i) state[i] = in_state[idx * tt + i];

    {
        ulong tmp[T_MAX];
        for (uint i = 0u; i < tt; ++i) {
            ulong acc = 0ul;
            for (uint j = 0u; j < tt; ++j) {
                acc = gold_add(acc, gold_mul(ext_mds[i * tt + j], state[j]));
            }
            tmp[i] = acc;
        }
        for (uint i = 0u; i < tt; ++i) state[i] = tmp[i];
    }
    for (uint r = 0u; r < half_f; ++r) {
        for (uint i = 0u; i < tt; ++i) state[i] = sbox(gold_add(state[i], rc_ext[r * tt + i]));
        ulong tmp[T_MAX];
        for (uint i = 0u; i < tt; ++i) {
            ulong acc = 0ul;
            for (uint j = 0u; j < tt; ++j) {
                acc = gold_add(acc, gold_mul(ext_mds[i * tt + j], state[j]));
            }
            tmp[i] = acc;
        }
        for (uint i = 0u; i < tt; ++i) state[i] = tmp[i];
    }
    for (uint r = 0u; r < rp; ++r) {
        state[0] = sbox(gold_add(state[0], rc_int[r]));
        ulong s = 0ul;
        for (uint i = 0u; i < tt; ++i) s = gold_add(s, state[i]);
        ulong tmp[T_MAX];
        for (uint i = 0u; i < tt; ++i) tmp[i] = sum_plus_mul(s, int_diag[i], state[i]);
        for (uint i = 0u; i < tt; ++i) state[i] = tmp[i];
    }
    for (uint r = half_f; r < rf; ++r) {
        for (uint i = 0u; i < tt; ++i) state[i] = sbox(gold_add(state[i], rc_ext[r * tt + i]));
        ulong tmp[T_MAX];
        for (uint i = 0u; i < tt; ++i) {
            ulong acc = 0ul;
            for (uint j = 0u; j < tt; ++j) {
                acc = gold_add(acc, gold_mul(ext_mds[i * tt + j], state[j]));
            }
            tmp[i] = acc;
        }
        for (uint i = 0u; i < tt; ++i) state[i] = tmp[i];
    }
    for (uint i = 0u; i < tt; ++i) out_state[idx * tt + i] = state[i];
}
```

Result of previous attempt:
            t3_B4K: correct, 0.10 ms, 13.0 Gmodmul/s (int64) (24.4% of 53 Gops/s (int64 mul, est))
           t3_B64K: correct, 1.26 ms, 17.2 Gmodmul/s (int64) (32.3% of 53 Gops/s (int64 mul, est))
            t3_B1M: correct, 14.62 ms, 23.7 Gmodmul/s (int64) (44.5% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.3272

## History

- iter  0: compile=OK | correct=True | score=0.22313100350851453
- iter  1: compile=OK | correct=True | score=0.22945717366315468
- iter  2: compile=OK | correct=True | score=0.2468564646184756
- iter  3: compile=OK | correct=True | score=0.28001545247299503
- iter  4: compile=OK | correct=True | score=0.32724556471757404

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
