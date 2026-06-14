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
    ulong s = a + b;
    s += (s < a) ? EPSILON : 0ul;
    return gold_canonical(s);
}

inline ulong gold_fold_sum_carry(ulong s, uint carry) {
    ulong corr = (((ulong)carry) << 32) - (ulong)carry;
    ulong old = s;
    s += corr;
    s += (s < old) ? EPSILON : 0ul;
    return gold_canonical(s);
}

inline ulong gold_add3(ulong a, ulong b, ulong c) {
    ulong s = a + b;
    uint carry = (s < a) ? 1u : 0u;

    ulong old = s;
    s += c;
    carry += (s < old) ? 1u : 0u;

    return gold_fold_sum_carry(s, carry);
}

inline ulong gold_reduce_limbs(uint lo0, uint lo1, uint hi0, uint hi1) {
    ulong x_lo = ((ulong)lo1 << 32) | (ulong)lo0;

    ulong t0 = x_lo - (ulong)hi1;
    t0 -= (t0 > x_lo) ? EPSILON : 0ul;

    ulong t1 = ((ulong)hi0 << 32) - (ulong)hi0;

    ulong t2 = t0 + t1;
    t2 += (t2 < t0) ? EPSILON : 0ul;

    return gold_canonical(t2);
}

inline ulong gold_mul(ulong a, ulong b) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    uint p00lo = a0 * b0;
    uint p00hi = mulhi(a0, b0);

    uint p01lo = a0 * b1;
    uint p01hi = mulhi(a0, b1);

    uint p10lo = a1 * b0;
    uint p10hi = mulhi(a1, b0);

    uint p11lo = a1 * b1;
    uint p11hi = mulhi(a1, b1);

    uint low_sum = p01lo + p10lo;
    uint carry = (low_sum < p01lo) ? 1u : 0u;

    uint lo1 = p00hi + low_sum;
    carry += (lo1 < p00hi) ? 1u : 0u;

    uint high_sum = p01hi + p10hi;
    uint high_carry = (high_sum < p01hi) ? 1u : 0u;

    uint hi0 = p11lo + high_sum;
    uint hi1 = p11hi + high_carry + ((hi0 < p11lo) ? 1u : 0u);

    uint old = hi0;
    hi0 += carry;
    hi1 += (hi0 < old) ? 1u : 0u;

    return gold_reduce_limbs(p00lo, lo1, hi0, hi1);
}

inline ulong gold_mul_u32(ulong a, uint b0) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);

    uint p00lo = a0 * b0;
    uint p00hi = mulhi(a0, b0);

    uint p10lo = a1 * b0;
    uint p10hi = mulhi(a1, b0);

    uint lo1 = p00hi + p10lo;
    uint carry = (lo1 < p00hi) ? 1u : 0u;

    uint hi0 = p10hi + carry;

    return gold_reduce_limbs(p00lo, lo1, hi0, 0u);
}

inline ulong gold_mul_u32_coeff(uint c, ulong a) {
    if (c > 8u) {
        return gold_mul_u32(a, c);
    }

    switch (c) {
        case 0u:
            return 0ul;
        case 1u:
            return a;
        case 2u:
            return gold_add(a, a);
        case 3u:
            return gold_add3(a, a, a);
        case 4u: {
            ulong a2 = gold_add(a, a);
            return gold_add(a2, a2);
        }
        case 5u: {
            ulong a2 = gold_add(a, a);
            ulong a4 = gold_add(a2, a2);
            return gold_add(a4, a);
        }
        case 6u: {
            ulong a2 = gold_add(a, a);
            ulong a4 = gold_add(a2, a2);
            return gold_add(a4, a2);
        }
        case 7u: {
            ulong a2 = gold_add(a, a);
            ulong a4 = gold_add(a2, a2);
            return gold_add3(a4, a2, a);
        }
        default: {
            ulong a2 = gold_add(a, a);
            ulong a4 = gold_add(a2, a2);
            return gold_add(a4, a4);
        }
    }
}

inline ulong gold_square(ulong a) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);

    uint p00lo = a0 * a0;
    uint p00hi = mulhi(a0, a0);

    uint p01lo = a0 * a1;
    uint p01hi = mulhi(a0, a1);

    uint p11lo = a1 * a1;
    uint p11hi = mulhi(a1, a1);

    uint dbl_lo = p01lo << 1;
    uint carry = p01lo >> 31;

    uint lo1 = p00hi + dbl_lo;
    carry += (lo1 < p00hi) ? 1u : 0u;

    uint dbl_hi = p01hi << 1;
    uint high_carry = p01hi >> 31;

    uint hi0 = p11lo + dbl_hi;
    uint hi1 = p11hi + high_carry + ((hi0 < p11lo) ? 1u : 0u);

    uint old = hi0;
    hi0 += carry;
    hi1 += (hi0 < old) ? 1u : 0u;

    return gold_reduce_limbs(p00lo, lo1, hi0, hi1);
}

inline ulong sbox7(ulong x) {
    ulong x2 = gold_square(x);
    ulong x4 = gold_square(x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
}

inline ulong gold_dot3(ulong c0, ulong x0,
                       ulong c1, ulong x1,
                       ulong c2, ulong x2)
{
    ulong s = gold_mul(c0, x0);

    ulong term = gold_mul(c1, x1);
    ulong old = s;
    s += term;
    uint carry = (s < old) ? 1u : 0u;

    term = gold_mul(c2, x2);
    old = s;
    s += term;
    carry += (s < old) ? 1u : 0u;

    return gold_fold_sum_carry(s, carry);
}

inline ulong gold_dot3_u32(uint c0, ulong x0,
                           uint c1, ulong x1,
                           uint c2, ulong x2)
{
    ulong s = gold_mul_u32_coeff(c0, x0);

    ulong term = gold_mul_u32_coeff(c1, x1);
    ulong old = s;
    s += term;
    uint carry = (s < old) ? 1u : 0u;

    term = gold_mul_u32_coeff(c2, x2);
    old = s;
    s += term;
    carry += (s < old) ? 1u : 0u;

    return gold_fold_sum_carry(s, carry);
}

inline void matvec_ext_generic(thread ulong *state,
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
    for (uint i = 0u; i < t; ++i) {
        state[i] = tmp[i];
    }
}

inline void matvec_int_generic(thread ulong *state,
                               device const ulong *int_diag,
                               uint t)
{
    ulong s = 0ul;
    for (uint i = 0u; i < t; ++i) {
        s = gold_add(s, state[i]);
    }

    ulong tmp[T_MAX];
    for (uint i = 0u; i < t; ++i) {
        tmp[i] = gold_add(s, gold_mul(int_diag[i], state[i]));
    }
    for (uint i = 0u; i < t; ++i) {
        state[i] = tmp[i];
    }
}

#define P2_EXT3_STEP_SIMPLE() do {                                      \
    ulong e0_ = gold_add(gold_add(gold_mul(m00, s0),                    \
                                  gold_mul(m01, s1)),                   \
                         gold_mul(m02, s2));                            \
    ulong e1_ = gold_add(gold_add(gold_mul(m10, s0),                    \
                                  gold_mul(m11, s1)),                   \
                         gold_mul(m12, s2));                            \
    ulong e2_ = gold_add(gold_add(gold_mul(m20, s0),                    \
                                  gold_mul(m21, s1)),                   \
                         gold_mul(m22, s2));                            \
    s0 = e0_;                                                           \
    s1 = e1_;                                                           \
    s2 = e2_;                                                           \
} while (0)

#define P2_EXT3_STEP_DOT() do {                                         \
    ulong e0_ = gold_dot3(m00, s0, m01, s1, m02, s2);                   \
    ulong e1_ = gold_dot3(m10, s0, m11, s1, m12, s2);                   \
    ulong e2_ = gold_dot3(m20, s0, m21, s1, m22, s2);                   \
    s0 = e0_;                                                           \
    s1 = e1_;                                                           \
    s2 = e2_;                                                           \
} while (0)

#define P2_EXT3_STEP_U32() do {                                         \
    ulong e0_ = gold_dot3_u32(cm00, s0, cm01, s1, cm02, s2);            \
    ulong e1_ = gold_dot3_u32(cm10, s0, cm11, s1, cm12, s2);            \
    ulong e2_ = gold_dot3_u32(cm20, s0, cm21, s1, cm22, s2);            \
    s0 = e0_;                                                           \
    s1 = e1_;                                                           \
    s2 = e2_;                                                           \
} while (0)

#define P2_FULL3_STEP_SIMPLE_PTR() do {                                 \
    s0 = sbox7(gold_add(s0, rcp[0]));                                   \
    s1 = sbox7(gold_add(s1, rcp[1]));                                   \
    s2 = sbox7(gold_add(s2, rcp[2]));                                   \
    P2_EXT3_STEP_SIMPLE();                                              \
    rcp += 3;                                                           \
} while (0)

#define P2_FULL3_STEP_DOT_PTR() do {                                    \
    s0 = sbox7(gold_add(s0, rcp[0]));                                   \
    s1 = sbox7(gold_add(s1, rcp[1]));                                   \
    s2 = sbox7(gold_add(s2, rcp[2]));                                   \
    P2_EXT3_STEP_DOT();                                                 \
    rcp += 3;                                                           \
} while (0)

#define P2_FULL3_STEP_U32_PTR() do {                                    \
    s0 = sbox7(gold_add(s0, rcp[0]));                                   \
    s1 = sbox7(gold_add(s1, rcp[1]));                                   \
    s2 = sbox7(gold_add(s2, rcp[2]));                                   \
    P2_EXT3_STEP_U32();                                                 \
    rcp += 3;                                                           \
} while (0)

#define P2_PART3_STEP_SIMPLE_PTR() do {                                 \
    s0 = sbox7(gold_add(s0, rip[0]));                                   \
    ulong sum_ = gold_add(gold_add(s0, s1), s2);                        \
    ulong p0_ = gold_add(sum_, gold_mul(d0, s0));                       \
    ulong p1_ = gold_add(sum_, gold_mul(d1, s1));                       \
    ulong p2_ = gold_add(sum_, gold_mul(d2, s2));                       \
    s0 = p0_;                                                           \
    s1 = p1_;                                                           \
    s2 = p2_;                                                           \
    rip += 1;                                                           \
} while (0)

#define P2_PART3_STEP_DOT_PTR() do {                                    \
    s0 = sbox7(gold_add(s0, rip[0]));                                   \
    ulong sum_ = gold_add3(s0, s1, s2);                                 \
    ulong p0_ = gold_add(sum_, gold_mul(d0, s0));                       \
    ulong p1_ = gold_add(sum_, gold_mul(d1, s1));                       \
    ulong p2_ = gold_add(sum_, gold_mul(d2, s2));                       \
    s0 = p0_;                                                           \
    s1 = p1_;                                                           \
    s2 = p2_;                                                           \
    rip += 1;                                                           \
} while (0)

#define P2_PART3_STEP_U32_PTR() do {                                    \
    s0 = sbox7(gold_add(s0, rip[0]));                                   \
    ulong sum_ = gold_add3(s0, s1, s2);                                 \
    ulong p0_ = gold_add(sum_, gold_mul_u32_coeff(cd0, s0));            \
    ulong p1_ = gold_add(sum_, gold_mul_u32_coeff(cd1, s1));            \
    ulong p2_ = gold_add(sum_, gold_mul_u32_coeff(cd2, s2));            \
    s0 = p0_;                                                           \
    s1 = p1_;                                                           \
    s2 = p2_;                                                           \
    rip += 1;                                                           \
} while (0)

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

    if (t == 3u) {
        uint base = idx * 3u;

        ulong s0 = in_state[base + 0u];
        ulong s1 = in_state[base + 1u];
        ulong s2 = in_state[base + 2u];

        ulong m00 = ext_mds[0u];
        ulong m01 = ext_mds[1u];
        ulong m02 = ext_mds[2u];
        ulong m10 = ext_mds[3u];
        ulong m11 = ext_mds[4u];
        ulong m12 = ext_mds[5u];
        ulong m20 = ext_mds[6u];
        ulong m21 = ext_mds[7u];
        ulong m22 = ext_mds[8u];

        ulong d0 = int_diag[0u];
        ulong d1 = int_diag[1u];
        ulong d2 = int_diag[2u];

        uint half_f = r_f >> 1u;

        if (batch <= 8192u) {
            P2_EXT3_STEP_SIMPLE();

            device const ulong *rcp = rc_ext;
            for (uint r = 0u; r < half_f; ++r) {
                P2_FULL3_STEP_SIMPLE_PTR();
            }

            device const ulong *rip = rc_int;
            for (uint r = 0u; r < r_p; ++r) {
                P2_PART3_STEP_SIMPLE_PTR();
            }

            rcp = rc_ext + half_f * 3u;
            for (uint r = half_f; r < r_f; ++r) {
                P2_FULL3_STEP_SIMPLE_PTR();
            }

            out_state[base + 0u] = s0;
            out_state[base + 1u] = s1;
            out_state[base + 2u] = s2;
            return;
        }

        bool ext32 = (((m00 | m01 | m02 |
                        m10 | m11 | m12 |
                        m20 | m21 | m22) >> 32) == 0ul);
        bool diag32 = (((d0 | d1 | d2) >> 32) == 0ul);

        uint cm00 = (uint)m00;
        uint cm01 = (uint)m01;
        uint cm02 = (uint)m02;
        uint cm10 = (uint)m10;
        uint cm11 = (uint)m11;
        uint cm12 = (uint)m12;
        uint cm20 = (uint)m20;
        uint cm21 = (uint)m21;
        uint cm22 = (uint)m22;

        uint cd0 = (uint)d0;
        uint cd1 = (uint)d1;
        uint cd2 = (uint)d2;

        device const ulong *rcp;
        device const ulong *rip;

        if (ext32) {
            P2_EXT3_STEP_U32();

            rcp = rc_ext;
            for (uint r = 0u; r < half_f; ++r) {
                P2_FULL3_STEP_U32_PTR();
            }

            rip = rc_int;
            if (diag32) {
                for (uint r = 0u; r < r_p; ++r) {
                    P2_PART3_STEP_U32_PTR();
                }
            } else {
                for (uint r = 0u; r < r_p; ++r) {
                    P2_PART3_STEP_DOT_PTR();
                }
            }

            rcp = rc_ext + half_f * 3u;
            for (uint r = half_f; r < r_f; ++r) {
                P2_FULL3_STEP_U32_PTR();
            }

            out_state[base + 0u] = s0;
            out_state[base + 1u] = s1;
            out_state[base + 2u] = s2;
            return;
        }

        P2_EXT3_STEP_DOT();

        rcp = rc_ext;
        for (uint r = 0u; r < half_f; ++r) {
            P2_FULL3_STEP_DOT_PTR();
        }

        rip = rc_int;
        if (diag32) {
            for (uint r = 0u; r < r_p; ++r) {
                P2_PART3_STEP_U32_PTR();
            }
        } else {
            for (uint r = 0u; r < r_p; ++r) {
                P2_PART3_STEP_DOT_PTR();
            }
        }

        rcp = rc_ext + half_f * 3u;
        for (uint r = half_f; r < r_f; ++r) {
            P2_FULL3_STEP_DOT_PTR();
        }

        out_state[base + 0u] = s0;
        out_state[base + 1u] = s1;
        out_state[base + 2u] = s2;
        return;
    }

    thread ulong state[T_MAX];
    uint base = idx * t;

    for (uint i = 0u; i < t; ++i) {
        state[i] = in_state[base + i];
    }

    matvec_ext_generic(state, ext_mds, t);

    uint half_f = r_f >> 1u;

    for (uint r = 0u; r < half_f; ++r) {
        uint off = r * t;
        for (uint i = 0u; i < t; ++i) {
            state[i] = sbox7(gold_add(state[i], rc_ext[off + i]));
        }
        matvec_ext_generic(state, ext_mds, t);
    }

    for (uint r = 0u; r < r_p; ++r) {
        state[0] = sbox7(gold_add(state[0], rc_int[r]));
        matvec_int_generic(state, int_diag, t);
    }

    for (uint r = half_f; r < r_f; ++r) {
        uint off = r * t;
        for (uint i = 0u; i < t; ++i) {
            state[i] = sbox7(gold_add(state[i], rc_ext[off + i]));
        }
        matvec_ext_generic(state, ext_mds, t);
    }

    for (uint i = 0u; i < t; ++i) {
        out_state[base + i] = state[i];
    }
}

#undef P2_EXT3_STEP_SIMPLE
#undef P2_EXT3_STEP_DOT
#undef P2_EXT3_STEP_U32
#undef P2_FULL3_STEP_SIMPLE_PTR
#undef P2_FULL3_STEP_DOT_PTR
#undef P2_FULL3_STEP_U32_PTR
#undef P2_PART3_STEP_SIMPLE_PTR
#undef P2_PART3_STEP_DOT_PTR
#undef P2_PART3_STEP_U32_PTR
```

Result of previous attempt:
            t3_B4K: correct, 0.28 ms, 4.9 Gmodmul/s (int64) (9.2% of 53 Gops/s (int64 mul, est))
           t3_B64K: correct, 1.20 ms, 18.1 Gmodmul/s (int64) (34.0% of 53 Gops/s (int64 mul, est))
            t3_B1M: correct, 13.60 ms, 25.5 Gmodmul/s (int64) (47.8% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.2466

## Current best (incumbent)

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
    ulong s = a + b;
    if (s < a) s += EPSILON;
    if (s >= P_GOLD) s -= P_GOLD;
    return s;
}

inline ulong gold_reduce_limbs(uint lo0, uint lo1, uint hi0, uint hi1) {
    ulong x_lo = ((ulong)lo1 << 32) | (ulong)lo0;

    ulong t0 = x_lo - (ulong)hi1;
    if (t0 > x_lo) t0 -= EPSILON;

    ulong t1 = ((ulong)hi0 << 32) - (ulong)hi0;

    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;

    return gold_canonical(t2);
}

inline ulong gold_mul(ulong a, ulong b) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    uint p00lo = a0 * b0;
    uint p00hi = mulhi(a0, b0);

    uint p01lo = a0 * b1;
    uint p01hi = mulhi(a0, b1);

    uint p10lo = a1 * b0;
    uint p10hi = mulhi(a1, b0);

    uint p11lo = a1 * b1;
    uint p11hi = mulhi(a1, b1);

    uint mid = p00hi + p01lo;
    uint carry = (mid < p00hi) ? 1u : 0u;
    uint lo1 = mid + p10lo;
    carry += (lo1 < mid) ? 1u : 0u;

    uint hi0 = p11lo;
    uint hi1 = p11hi;

    uint old = hi0;
    hi0 += p01hi;
    hi1 += (hi0 < old) ? 1u : 0u;

    old = hi0;
    hi0 += p10hi;
    hi1 += (hi0 < old) ? 1u : 0u;

    old = hi0;
    hi0 += carry;
    hi1 += (hi0 < old) ? 1u : 0u;

    return gold_reduce_limbs(p00lo, lo1, hi0, hi1);
}

inline ulong gold_square(ulong a) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);

    uint p00lo = a0 * a0;
    uint p00hi = mulhi(a0, a0);

    uint p01lo = a0 * a1;
    uint p01hi = mulhi(a0, a1);

    uint p11lo = a1 * a1;
    uint p11hi = mulhi(a1, a1);

    uint dbl_lo = p01lo << 1;
    uint carry = p01lo >> 31;

    uint lo1 = p00hi + dbl_lo;
    carry += (lo1 < p00hi) ? 1u : 0u;

    uint hi0 = p11lo;
    uint hi1 = p11hi;

    uint old = hi0;
    hi0 += p01hi;
    hi1 += (hi0 < old) ? 1u : 0u;

    old = hi0;
    hi0 += p01hi;
    hi1 += (hi0 < old) ? 1u : 0u;

    old = hi0;
    hi0 += carry;
    hi1 += (hi0 < old) ? 1u : 0u;

    return gold_reduce_limbs(p00lo, lo1, hi0, hi1);
}

inline ulong sbox7(ulong x) {
    ulong x2 = gold_square(x);
    ulong x4 = gold_square(x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
}

inline void matvec_ext_generic(thread ulong *state,
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
    for (uint i = 0u; i < t; ++i) {
        state[i] = tmp[i];
    }
}

inline void matvec_int_generic(thread ulong *state,
                               device const ulong *int_diag,
                               uint t)
{
    ulong s = 0ul;
    for (uint i = 0u; i < t; ++i) {
        s = gold_add(s, state[i]);
    }

    ulong tmp[T_MAX];
    for (uint i = 0u; i < t; ++i) {
        tmp[i] = gold_add(s, gold_mul(int_diag[i], state[i]));
    }
    for (uint i = 0u; i < t; ++i) {
        state[i] = tmp[i];
    }
}

#define P2_EXT3_STEP() do {                                                     \
    ulong e0_ = gold_add(gold_add(gold_mul(m00, s0), gold_mul(m01, s1)),        \
                          gold_mul(m02, s2));                                   \
    ulong e1_ = gold_add(gold_add(gold_mul(m10, s0), gold_mul(m11, s1)),        \
                          gold_mul(m12, s2));                                   \
    ulong e2_ = gold_add(gold_add(gold_mul(m20, s0), gold_mul(m21, s1)),        \
                          gold_mul(m22, s2));                                   \
    s0 = e0_;                                                                   \
    s1 = e1_;                                                                   \
    s2 = e2_;                                                                   \
} while (0)

#define P2_FULL3_STEP_PTR() do {                                                \
    s0 = sbox7(gold_add(s0, rcp[0]));                                           \
    s1 = sbox7(gold_add(s1, rcp[1]));                                           \
    s2 = sbox7(gold_add(s2, rcp[2]));                                           \
    P2_EXT3_STEP();                                                             \
    rcp += 3;                                                                   \
} while (0)

#define P2_PART3_STEP_PTR() do {                                                \
    s0 = sbox7(gold_add(s0, rip[0]));                                           \
    ulong sum_ = gold_add(gold_add(s0, s1), s2);                                \
    ulong p0_ = gold_add(sum_, gold_mul(d0, s0));                               \
    ulong p1_ = gold_add(sum_, gold_mul(d1, s1));                               \
    ulong p2_ = gold_add(sum_, gold_mul(d2, s2));                               \
    s0 = p0_;                                                                   \
    s1 = p1_;                                                                   \
    s2 = p2_;                                                                   \
    rip += 1;                                                                   \
} while (0)

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

    if (t == 3u) {
        uint base = idx * 3u;

        ulong s0 = in_state[base + 0u];
        ulong s1 = in_state[base + 1u];
        ulong s2 = in_state[base + 2u];

        ulong m00 = ext_mds[0u];
        ulong m01 = ext_mds[1u];
        ulong m02 = ext_mds[2u];
        ulong m10 = ext_mds[3u];
        ulong m11 = ext_mds[4u];
        ulong m12 = ext_mds[5u];
        ulong m20 = ext_mds[6u];
        ulong m21 = ext_mds[7u];
        ulong m22 = ext_mds[8u];

        ulong d0 = int_diag[0u];
        ulong d1 = int_diag[1u];
        ulong d2 = int_diag[2u];

        P2_EXT3_STEP();

        uint half_f = r_f >> 1u;

        device const ulong *rcp = rc_ext;
        for (uint r = 0u; r < half_f; ++r) {
            P2_FULL3_STEP_PTR();
        }

        device const ulong *rip = rc_int;
        for (uint r = 0u; r < r_p; ++r) {
            P2_PART3_STEP_PTR();
        }

        rcp = rc_ext + half_f * 3u;
        for (uint r = half_f; r < r_f; ++r) {
            P2_FULL3_STEP_PTR();
        }

        out_state[base + 0u] = s0;
        out_state[base + 1u] = s1;
        out_state[base + 2u] = s2;
        return;
    }

    thread ulong state[T_MAX];
    uint base = idx * t;

    for (uint i = 0u; i < t; ++i) {
        state[i] = in_state[base + i];
    }

    matvec_ext_generic(state, ext_mds, t);

    uint half_f = r_f >> 1u;

    for (uint r = 0u; r < half_f; ++r) {
        uint off = r * t;
        for (uint i = 0u; i < t; ++i) {
            state[i] = sbox7(gold_add(state[i], rc_ext[off + i]));
        }
        matvec_ext_generic(state, ext_mds, t);
    }

    for (uint r = 0u; r < r_p; ++r) {
        state[0] = sbox7(gold_add(state[0], rc_int[r]));
        matvec_int_generic(state, int_diag, t);
    }

    for (uint r = half_f; r < r_f; ++r) {
        uint off = r * t;
        for (uint i = 0u; i < t; ++i) {
            state[i] = sbox7(gold_add(state[i], rc_ext[off + i]));
        }
        matvec_ext_generic(state, ext_mds, t);
    }

    for (uint i = 0u; i < t; ++i) {
        out_state[base + i] = state[i];
    }
}

#undef P2_EXT3_STEP
#undef P2_FULL3_STEP_PTR
#undef P2_PART3_STEP_PTR
```

Incumbent result:
            t3_B4K: correct, 0.24 ms, 5.6 Gmodmul/s (int64) (10.5% of 53 Gops/s (int64 mul, est))
           t3_B64K: correct, 0.93 ms, 23.3 Gmodmul/s (int64) (43.8% of 53 Gops/s (int64 mul, est))
            t3_B1M: correct, 14.41 ms, 24.1 Gmodmul/s (int64) (45.1% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.2744

## History

- iter  0: compile=OK | correct=True | score=0.24563303919915822
- iter  1: compile=OK | correct=True | score=0.27442333933299823
- iter  2: compile=OK | correct=True | score=0.19962569094325608
- iter  3: compile=OK | correct=True | score=0.2597024198513166
- iter  4: compile=OK | correct=True | score=0.2465666634612418

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
