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
    if (s < a) s += EPSILON;
    if (s >= P_GOLD) s -= P_GOLD;
    return s;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong s = a - b;
    if (a < b) s -= EPSILON;
    return s;
}

inline ulong gold_fold_sum_carry(ulong s, uint carry) {
    ulong corr = (((ulong)carry) << 32) - (ulong)carry;
    ulong old = s;
    s += corr;
    if (s < old) s += EPSILON;
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

inline ulong gold_sub_one(ulong x) {
    return (x == 0ul) ? (P_GOLD - 1ul) : (x - 1ul);
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

inline ulong gold_mul_limb(uint a0, uint a1, ulong b) {
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

inline ulong gold_mul_add_limb(uint a0, uint a1, ulong b, ulong addend) {
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

    uint old32 = hi0;
    hi0 += p01hi;
    hi1 += (hi0 < old32) ? 1u : 0u;

    old32 = hi0;
    hi0 += p10hi;
    hi1 += (hi0 < old32) ? 1u : 0u;

    old32 = hi0;
    hi0 += carry;
    hi1 += (hi0 < old32) ? 1u : 0u;

    ulong lo = ((ulong)lo1 << 32) | (ulong)p00lo;
    ulong hi = ((ulong)hi1 << 32) | (ulong)hi0;

    ulong old = lo;
    lo += addend;
    hi += (lo < old) ? 1ul : 0ul;

    return gold_reduce_limbs((uint)lo, (uint)(lo >> 32),
                             (uint)hi, (uint)(hi >> 32));
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

inline ulong gold_mul_add(ulong a, ulong b, ulong addend) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    return gold_mul_add_limb(a0, a1, b, addend);
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

inline bool gold_small_const(ulong c) {
    return (c <= 4ul) || (c == (P_GOLD - 1ul));
}

inline ulong gold_mul_small_add(ulong c, ulong x, ulong addend) {
    if (c == 0ul) return addend;
    if (c == 1ul) return gold_add(addend, x);
    if (c == 2ul) {
        ulong x2 = gold_add(x, x);
        return gold_add(addend, x2);
    }
    if (c == 3ul) {
        ulong x2 = gold_add(x, x);
        ulong x3 = gold_add(x2, x);
        return gold_add(addend, x3);
    }
    if (c == 4ul) {
        ulong x2 = gold_add(x, x);
        ulong x4 = gold_add(x2, x2);
        return gold_add(addend, x4);
    }
    if (c == (P_GOLD - 1ul)) {
        return gold_sub(addend, x);
    }
    return gold_mul_add(c, x, addend);
}

inline void matvec_ext_generic(thread ulong *state,
                               device const ulong *ext_mds,
                               uint t)
{
    ulong tmp[T_MAX];
    for (uint i = 0u; i < t; ++i) {
        ulong acc = gold_mul(ext_mds[i * t], state[0]);
        for (uint j = 1u; j < t; ++j) {
            acc = gold_mul_add(ext_mds[i * t + j], state[j], acc);
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
        tmp[i] = gold_mul_add(int_diag[i], state[i], s);
    }
    for (uint i = 0u; i < t; ++i) {
        state[i] = tmp[i];
    }
}

#define P2_STORE3() do {                                                        \
    out_state[base + 0u] = s0;                                                  \
    out_state[base + 1u] = s1;                                                  \
    out_state[base + 2u] = s2;                                                  \
} while (0)

#define P2_EXT3_JI_STEP() do {                                                  \
    ulong a0_ = s0;                                                             \
    ulong a1_ = s1;                                                             \
    ulong a2_ = s2;                                                             \
    ulong sum_ = gold_add3(a0_, a1_, a2_);                                      \
    s0 = gold_add(sum_, a0_);                                                   \
    s1 = gold_add(sum_, a1_);                                                   \
    s2 = gold_add(sum_, a2_);                                                   \
} while (0)

#define P2_EXT3_JDIAG_STEP() do {                                               \
    ulong a0_ = s0;                                                             \
    ulong a1_ = s1;                                                             \
    ulong a2_ = s2;                                                             \
    ulong sum_ = gold_add3(a0_, a1_, a2_);                                      \
    s0 = gold_mul_add(e0, a0_, sum_);                                           \
    s1 = gold_mul_add(e1, a1_, sum_);                                           \
    s2 = gold_mul_add(e2, a2_, sum_);                                           \
} while (0)

#define P2_EXT3_DENSE_STEP() do {                                               \
    ulong a0_ = s0;                                                             \
    ulong a1_ = s1;                                                             \
    ulong a2_ = s2;                                                             \
    ulong e0_ = gold_mul(m00, a0_);                                             \
    e0_ = gold_mul_add(m01, a1_, e0_);                                          \
    e0_ = gold_mul_add(m02, a2_, e0_);                                          \
    ulong e1_ = gold_mul(m10, a0_);                                             \
    e1_ = gold_mul_add(m11, a1_, e1_);                                          \
    e1_ = gold_mul_add(m12, a2_, e1_);                                          \
    ulong e2_ = gold_mul(m20, a0_);                                             \
    e2_ = gold_mul_add(m21, a1_, e2_);                                          \
    e2_ = gold_mul_add(m22, a2_, e2_);                                          \
    s0 = e0_;                                                                   \
    s1 = e1_;                                                                   \
    s2 = e2_;                                                                   \
} while (0)

#define P2_PART3_DIAG1_PTR() do {                                               \
    s0 = sbox7(gold_add(s0, rip[0]));                                           \
    P2_EXT3_JI_STEP();                                                          \
    rip += 1;                                                                   \
} while (0)

#define P2_PART3_SMALL_PTR() do {                                               \
    s0 = sbox7(gold_add(s0, rip[0]));                                           \
    ulong sum_ = gold_add3(s0, s1, s2);                                         \
    ulong p0_ = gold_mul_small_add(d0, s0, sum_);                               \
    ulong p1_ = gold_mul_small_add(d1, s1, sum_);                               \
    ulong p2_ = gold_mul_small_add(d2, s2, sum_);                               \
    s0 = p0_;                                                                   \
    s1 = p1_;                                                                   \
    s2 = p2_;                                                                   \
    rip += 1;                                                                   \
} while (0)

#define P2_PART3_LIMB_PTR() do {                                                \
    s0 = sbox7(gold_add(s0, rip[0]));                                           \
    ulong sum_ = gold_add3(s0, s1, s2);                                         \
    ulong p0_ = gold_mul_add_limb(d0l, d0h, s0, sum_);                          \
    ulong p1_ = gold_mul_add_limb(d1l, d1h, s1, sum_);                          \
    ulong p2_ = gold_mul_add_limb(d2l, d2h, s2, sum_);                          \
    s0 = p0_;                                                                   \
    s1 = p1_;                                                                   \
    s2 = p2_;                                                                   \
    rip += 1;                                                                   \
} while (0)

#define P2_PART3_GENERIC_PTR() do {                                             \
    s0 = sbox7(gold_add(s0, rip[0]));                                           \
    ulong sum_ = gold_add3(s0, s1, s2);                                         \
    ulong p0_ = gold_mul_add(d0, s0, sum_);                                     \
    ulong p1_ = gold_mul_add(d1, s1, sum_);                                     \
    ulong p2_ = gold_mul_add(d2, s2, sum_);                                     \
    s0 = p0_;                                                                   \
    s1 = p1_;                                                                   \
    s2 = p2_;                                                                   \
    rip += 1;                                                                   \
} while (0)

#define P2_PART3_DIAG1_K(K) do {                                                \
    s0 = sbox7(gold_add(s0, rc_int[(K)]));                                      \
    P2_EXT3_JI_STEP();                                                          \
} while (0)

#define P2_PART3_SMALL_K(K) do {                                                \
    s0 = sbox7(gold_add(s0, rc_int[(K)]));                                      \
    ulong sum_ = gold_add3(s0, s1, s2);                                         \
    ulong p0_ = gold_mul_small_add(d0, s0, sum_);                               \
    ulong p1_ = gold_mul_small_add(d1, s1, sum_);                               \
    ulong p2_ = gold_mul_small_add(d2, s2, sum_);                               \
    s0 = p0_;                                                                   \
    s1 = p1_;                                                                   \
    s2 = p2_;                                                                   \
} while (0)

#define P2_PART3_LIMB_K(K) do {                                                 \
    s0 = sbox7(gold_add(s0, rc_int[(K)]));                                      \
    ulong sum_ = gold_add3(s0, s1, s2);                                         \
    ulong p0_ = gold_mul_add_limb(d0l, d0h, s0, sum_);                          \
    ulong p1_ = gold_mul_add_limb(d1l, d1h, s1, sum_);                          \
    ulong p2_ = gold_mul_add_limb(d2l, d2h, s2, sum_);                          \
    s0 = p0_;                                                                   \
    s1 = p1_;                                                                   \
    s2 = p2_;                                                                   \
} while (0)

#define P2_FULL3_JI_PTR() do {                                                  \
    s0 = sbox7(gold_add(s0, rcp[0]));                                           \
    s1 = sbox7(gold_add(s1, rcp[1]));                                           \
    s2 = sbox7(gold_add(s2, rcp[2]));                                           \
    P2_EXT3_JI_STEP();                                                          \
    rcp += 3;                                                                   \
} while (0)

#define P2_FULL3_JDIAG_PTR() do {                                               \
    s0 = sbox7(gold_add(s0, rcp[0]));                                           \
    s1 = sbox7(gold_add(s1, rcp[1]));                                           \
    s2 = sbox7(gold_add(s2, rcp[2]));                                           \
    P2_EXT3_JDIAG_STEP();                                                       \
    rcp += 3;                                                                   \
} while (0)

#define P2_FULL3_DENSE_PTR() do {                                               \
    s0 = sbox7(gold_add(s0, rcp[0]));                                           \
    s1 = sbox7(gold_add(s1, rcp[1]));                                           \
    s2 = sbox7(gold_add(s2, rcp[2]));                                           \
    P2_EXT3_DENSE_STEP();                                                       \
    rcp += 3;                                                                   \
} while (0)

#define P2_FULL3_JI_K(K) do {                                                   \
    s0 = sbox7(gold_add(s0, rc_ext[(K) * 3u + 0u]));                            \
    s1 = sbox7(gold_add(s1, rc_ext[(K) * 3u + 1u]));                            \
    s2 = sbox7(gold_add(s2, rc_ext[(K) * 3u + 2u]));                            \
    P2_EXT3_JI_STEP();                                                          \
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

        bool offdiag_one = (m01 == 1ul) & (m02 == 1ul) &
                           (m10 == 1ul) & (m12 == 1ul) &
                           (m20 == 1ul) & (m21 == 1ul);

        bool d_all_one = (d0 == 1ul) & (d1 == 1ul) & (d2 == 1ul);
        bool d_small = gold_small_const(d0) & gold_small_const(d1) & gold_small_const(d2);

        uint half_f = r_f >> 1u;

        if (offdiag_one && (m00 == 2ul) && (m11 == 2ul) && (m22 == 2ul)) {
            if ((r_f == 8u) & (r_p == 22u)) {
                P2_EXT3_JI_STEP();

                P2_FULL3_JI_K(0u);
                P2_FULL3_JI_K(1u);
                P2_FULL3_JI_K(2u);
                P2_FULL3_JI_K(3u);

                if (d_all_one) {
                    P2_PART3_DIAG1_K(0u);
                    P2_PART3_DIAG1_K(1u);
                    P2_PART3_DIAG1_K(2u);
                    P2_PART3_DIAG1_K(3u);
                    P2_PART3_DIAG1_K(4u);
                    P2_PART3_DIAG1_K(5u);
                    P2_PART3_DIAG1_K(6u);
                    P2_PART3_DIAG1_K(7u);
                    P2_PART3_DIAG1_K(8u);
                    P2_PART3_DIAG1_K(9u);
                    P2_PART3_DIAG1_K(10u);
                    P2_PART3_DIAG1_K(11u);
                    P2_PART3_DIAG1_K(12u);
                    P2_PART3_DIAG1_K(13u);
                    P2_PART3_DIAG1_K(14u);
                    P2_PART3_DIAG1_K(15u);
                    P2_PART3_DIAG1_K(16u);
                    P2_PART3_DIAG1_K(17u);
                    P2_PART3_DIAG1_K(18u);
                    P2_PART3_DIAG1_K(19u);
                    P2_PART3_DIAG1_K(20u);
                    P2_PART3_DIAG1_K(21u);
                } else if (d_small) {
                    P2_PART3_SMALL_K(0u);
                    P2_PART3_SMALL_K(1u);
                    P2_PART3_SMALL_K(2u);
                    P2_PART3_SMALL_K(3u);
                    P2_PART3_SMALL_K(4u);
                    P2_PART3_SMALL_K(5u);
                    P2_PART3_SMALL_K(6u);
                    P2_PART3_SMALL_K(7u);
                    P2_PART3_SMALL_K(8u);
                    P2_PART3_SMALL_K(9u);
                    P2_PART3_SMALL_K(10u);
                    P2_PART3_SMALL_K(11u);
                    P2_PART3_SMALL_K(12u);
                    P2_PART3_SMALL_K(13u);
                    P2_PART3_SMALL_K(14u);
                    P2_PART3_SMALL_K(15u);
                    P2_PART3_SMALL_K(16u);
                    P2_PART3_SMALL_K(17u);
                    P2_PART3_SMALL_K(18u);
                    P2_PART3_SMALL_K(19u);
                    P2_PART3_SMALL_K(20u);
                    P2_PART3_SMALL_K(21u);
                } else {
                    uint d0l = (uint)d0, d0h = (uint)(d0 >> 32);
                    uint d1l = (uint)d1, d1h = (uint)(d1 >> 32);
                    uint d2l = (uint)d2, d2h = (uint)(d2 >> 32);

                    P2_PART3_LIMB_K(0u);
                    P2_PART3_LIMB_K(1u);
                    P2_PART3_LIMB_K(2u);
                    P2_PART3_LIMB_K(3u);
                    P2_PART3_LIMB_K(4u);
                    P2_PART3_LIMB_K(5u);
                    P2_PART3_LIMB_K(6u);
                    P2_PART3_LIMB_K(7u);
                    P2_PART3_LIMB_K(8u);
                    P2_PART3_LIMB_K(9u);
                    P2_PART3_LIMB_K(10u);
                    P2_PART3_LIMB_K(11u);
                    P2_PART3_LIMB_K(12u);
                    P2_PART3_LIMB_K(13u);
                    P2_PART3_LIMB_K(14u);
                    P2_PART3_LIMB_K(15u);
                    P2_PART3_LIMB_K(16u);
                    P2_PART3_LIMB_K(17u);
                    P2_PART3_LIMB_K(18u);
                    P2_PART3_LIMB_K(19u);
                    P2_PART3_LIMB_K(20u);
                    P2_PART3_LIMB_K(21u);
                }

                P2_FULL3_JI_K(4u);
                P2_FULL3_JI_K(5u);
                P2_FULL3_JI_K(6u);
                P2_FULL3_JI_K(7u);

                P2_STORE3();
                return;
            }

            P2_EXT3_JI_STEP();

            device const ulong *rcp = rc_ext;
            for (uint r = 0u; r < half_f; ++r) {
                P2_FULL3_JI_PTR();
            }

            device const ulong *rip = rc_int;
            if (d_all_one) {
                for (uint r = 0u; r < r_p; ++r) {
                    P2_PART3_DIAG1_PTR();
                }
            } else if (d_small) {
                for (uint r = 0u; r < r_p; ++r) {
                    P2_PART3_SMALL_PTR();
                }
            } else {
                uint d0l = (uint)d0, d0h = (uint)(d0 >> 32);
                uint d1l = (uint)d1, d1h = (uint)(d1 >> 32);
                uint d2l = (uint)d2, d2h = (uint)(d2 >> 32);
                for (uint r = 0u; r < r_p; ++r) {
                    P2_PART3_LIMB_PTR();
                }
            }

            rcp = rc_ext + half_f * 3u;
            for (uint r = half_f; r < r_f; ++r) {
                P2_FULL3_JI_PTR();
            }

            P2_STORE3();
            return;
        }

        if (offdiag_one) {
            ulong e0 = gold_sub_one(m00);
            ulong e1 = gold_sub_one(m11);
            ulong e2 = gold_sub_one(m22);

            P2_EXT3_JDIAG_STEP();

            device const ulong *rcp = rc_ext;
            for (uint r = 0u; r < half_f; ++r) {
                P2_FULL3_JDIAG_PTR();
            }

            device const ulong *rip = rc_int;
            if (d_all_one) {
                for (uint r = 0u; r < r_p; ++r) {
                    P2_PART3_DIAG1_PTR();
                }
            } else if (d_small) {
                for (uint r = 0u; r < r_p; ++r) {
                    P2_PART3_SMALL_PTR();
                }
            } else {
                uint d0l = (uint)d0, d0h = (uint)(d0 >> 32);
                uint d1l = (uint)d1, d1h = (uint)(d1 >> 32);
                uint d2l = (uint)d2, d2h = (uint)(d2 >> 32);
                for (uint r = 0u; r < r_p; ++r) {
                    P2_PART3_LIMB_PTR();
                }
            }

            rcp = rc_ext + half_f * 3u;
            for (uint r = half_f; r < r_f; ++r) {
                P2_FULL3_JDIAG_PTR();
            }

            P2_STORE3();
            return;
        }

        P2_EXT3_DENSE_STEP();

        device const ulong *rcp = rc_ext;
        for (uint r = 0u; r < half_f; ++r) {
            P2_FULL3_DENSE_PTR();
        }

        device const ulong *rip = rc_int;
        if (d_all_one) {
            for (uint r = 0u; r < r_p; ++r) {
                P2_PART3_DIAG1_PTR();
            }
        } else if (d_small) {
            for (uint r = 0u; r < r_p; ++r) {
                P2_PART3_SMALL_PTR();
            }
        } else {
            uint d0l = (uint)d0, d0h = (uint)(d0 >> 32);
            uint d1l = (uint)d1, d1h = (uint)(d1 >> 32);
            uint d2l = (uint)d2, d2h = (uint)(d2 >> 32);
            for (uint r = 0u; r < r_p; ++r) {
                P2_PART3_LIMB_PTR();
            }
        }

        rcp = rc_ext + half_f * 3u;
        for (uint r = half_f; r < r_f; ++r) {
            P2_FULL3_DENSE_PTR();
        }

        P2_STORE3();
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

#undef P2_STORE3
#undef P2_EXT3_JI_STEP
#undef P2_EXT3_JDIAG_STEP
#undef P2_EXT3_DENSE_STEP
#undef P2_PART3_DIAG1_PTR
#undef P2_PART3_SMALL_PTR
#undef P2_PART3_LIMB_PTR
#undef P2_PART3_GENERIC_PTR
#undef P2_PART3_DIAG1_K
#undef P2_PART3_SMALL_K
#undef P2_PART3_LIMB_K
#undef P2_FULL3_JI_PTR
#undef P2_FULL3_JDIAG_PTR
#undef P2_FULL3_DENSE_PTR
#undef P2_FULL3_JI_K
```

Result of previous attempt:
            t3_B4K: correct, 0.18 ms, 7.5 Gmodmul/s (int64) (14.1% of 53 Gops/s (int64 mul, est))
           t3_B64K: correct, 1.52 ms, 14.3 Gmodmul/s (int64) (26.7% of 53 Gops/s (int64 mul, est))
            t3_B1M: correct, 12.70 ms, 27.3 Gmodmul/s (int64) (51.2% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.2681

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

inline ulong gold_fold_sum_carry(ulong s, uint carry) {
    ulong corr = (((ulong)carry) << 32) - (ulong)carry;
    ulong old = s;
    s += corr;
    if (s < old) s += EPSILON;
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

inline ulong gold_sub_one(ulong x) {
    return (x == 0ul) ? (P_GOLD - 1ul) : (x - 1ul);
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

inline ulong gold_mul_add(ulong a, ulong b, ulong addend) {
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

    uint old32 = hi0;
    hi0 += p01hi;
    hi1 += (hi0 < old32) ? 1u : 0u;

    old32 = hi0;
    hi0 += p10hi;
    hi1 += (hi0 < old32) ? 1u : 0u;

    old32 = hi0;
    hi0 += carry;
    hi1 += (hi0 < old32) ? 1u : 0u;

    ulong lo = ((ulong)lo1 << 32) | (ulong)p00lo;
    ulong hi = ((ulong)hi1 << 32) | (ulong)hi0;

    ulong old = lo;
    lo += addend;
    hi += (lo < old) ? 1ul : 0ul;

    return gold_reduce_limbs((uint)lo, (uint)(lo >> 32),
                             (uint)hi, (uint)(hi >> 32));
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
        ulong acc = gold_mul(ext_mds[i * t], state[0]);
        for (uint j = 1u; j < t; ++j) {
            acc = gold_mul_add(ext_mds[i * t + j], state[j], acc);
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
        tmp[i] = gold_mul_add(int_diag[i], state[i], s);
    }
    for (uint i = 0u; i < t; ++i) {
        state[i] = tmp[i];
    }
}

#define P2_EXT3_JI_STEP() do {                                                  \
    ulong a0_ = s0;                                                             \
    ulong a1_ = s1;                                                             \
    ulong a2_ = s2;                                                             \
    ulong sum_ = gold_add3(a0_, a1_, a2_);                                      \
    s0 = gold_add(sum_, a0_);                                                   \
    s1 = gold_add(sum_, a1_);                                                   \
    s2 = gold_add(sum_, a2_);                                                   \
} while (0)

#define P2_EXT3_JDIAG_STEP() do {                                               \
    ulong a0_ = s0;                                                             \
    ulong a1_ = s1;                                                             \
    ulong a2_ = s2;                                                             \
    ulong sum_ = gold_add3(a0_, a1_, a2_);                                      \
    s0 = gold_mul_add(e0, a0_, sum_);                                           \
    s1 = gold_mul_add(e1, a1_, sum_);                                           \
    s2 = gold_mul_add(e2, a2_, sum_);                                           \
} while (0)

#define P2_EXT3_DENSE_STEP() do {                                               \
    ulong a0_ = s0;                                                             \
    ulong a1_ = s1;                                                             \
    ulong a2_ = s2;                                                             \
    ulong e0_ = gold_mul(m00, a0_);                                             \
    e0_ = gold_mul_add(m01, a1_, e0_);                                          \
    e0_ = gold_mul_add(m02, a2_, e0_);                                          \
    ulong e1_ = gold_mul(m10, a0_);                                             \
    e1_ = gold_mul_add(m11, a1_, e1_);                                          \
    e1_ = gold_mul_add(m12, a2_, e1_);                                          \
    ulong e2_ = gold_mul(m20, a0_);                                             \
    e2_ = gold_mul_add(m21, a1_, e2_);                                          \
    e2_ = gold_mul_add(m22, a2_, e2_);                                          \
    s0 = e0_;                                                                   \
    s1 = e1_;                                                                   \
    s2 = e2_;                                                                   \
} while (0)

#define P2_PART3_STEP_PTR() do {                                                \
    s0 = sbox7(gold_add(s0, rip[0]));                                           \
    ulong sum_ = gold_add3(s0, s1, s2);                                         \
    ulong p0_ = gold_mul_add(d0, s0, sum_);                                     \
    ulong p1_ = gold_mul_add(d1, s1, sum_);                                     \
    ulong p2_ = gold_mul_add(d2, s2, sum_);                                     \
    s0 = p0_;                                                                   \
    s1 = p1_;                                                                   \
    s2 = p2_;                                                                   \
    rip += 1;                                                                   \
} while (0)

#define P2_FULL3_JI_PTR() do {                                                  \
    s0 = sbox7(gold_add(s0, rcp[0]));                                           \
    s1 = sbox7(gold_add(s1, rcp[1]));                                           \
    s2 = sbox7(gold_add(s2, rcp[2]));                                           \
    P2_EXT3_JI_STEP();                                                          \
    rcp += 3;                                                                   \
} while (0)

#define P2_FULL3_JDIAG_PTR() do {                                               \
    s0 = sbox7(gold_add(s0, rcp[0]));                                           \
    s1 = sbox7(gold_add(s1, rcp[1]));                                           \
    s2 = sbox7(gold_add(s2, rcp[2]));                                           \
    P2_EXT3_JDIAG_STEP();                                                       \
    rcp += 3;                                                                   \
} while (0)

#define P2_FULL3_DENSE_PTR() do {                                               \
    s0 = sbox7(gold_add(s0, rcp[0]));                                           \
    s1 = sbox7(gold_add(s1, rcp[1]));                                           \
    s2 = sbox7(gold_add(s2, rcp[2]));                                           \
    P2_EXT3_DENSE_STEP();                                                       \
    rcp += 3;                                                                   \
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

        bool offdiag_one = (m01 == 1ul) & (m02 == 1ul) &
                           (m10 == 1ul) & (m12 == 1ul) &
                           (m20 == 1ul) & (m21 == 1ul);

        uint half_f = r_f >> 1u;

        if (offdiag_one && (m00 == 2ul) && (m11 == 2ul) && (m22 == 2ul)) {
            P2_EXT3_JI_STEP();

            device const ulong *rcp = rc_ext;
            for (uint r = 0u; r < half_f; ++r) {
                P2_FULL3_JI_PTR();
            }

            device const ulong *rip = rc_int;
            for (uint r = 0u; r < r_p; ++r) {
                P2_PART3_STEP_PTR();
            }

            rcp = rc_ext + half_f * 3u;
            for (uint r = half_f; r < r_f; ++r) {
                P2_FULL3_JI_PTR();
            }

            out_state[base + 0u] = s0;
            out_state[base + 1u] = s1;
            out_state[base + 2u] = s2;
            return;
        }

        if (offdiag_one) {
            ulong e0 = gold_sub_one(m00);
            ulong e1 = gold_sub_one(m11);
            ulong e2 = gold_sub_one(m22);

            P2_EXT3_JDIAG_STEP();

            device const ulong *rcp = rc_ext;
            for (uint r = 0u; r < half_f; ++r) {
                P2_FULL3_JDIAG_PTR();
            }

            device const ulong *rip = rc_int;
            for (uint r = 0u; r < r_p; ++r) {
                P2_PART3_STEP_PTR();
            }

            rcp = rc_ext + half_f * 3u;
            for (uint r = half_f; r < r_f; ++r) {
                P2_FULL3_JDIAG_PTR();
            }

            out_state[base + 0u] = s0;
            out_state[base + 1u] = s1;
            out_state[base + 2u] = s2;
            return;
        }

        P2_EXT3_DENSE_STEP();

        device const ulong *rcp = rc_ext;
        for (uint r = 0u; r < half_f; ++r) {
            P2_FULL3_DENSE_PTR();
        }

        device const ulong *rip = rc_int;
        for (uint r = 0u; r < r_p; ++r) {
            P2_PART3_STEP_PTR();
        }

        rcp = rc_ext + half_f * 3u;
        for (uint r = half_f; r < r_f; ++r) {
            P2_FULL3_DENSE_PTR();
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

#undef P2_EXT3_JI_STEP
#undef P2_EXT3_JDIAG_STEP
#undef P2_EXT3_DENSE_STEP
#undef P2_PART3_STEP_PTR
#undef P2_FULL3_JI_PTR
#undef P2_FULL3_JDIAG_PTR
#undef P2_FULL3_DENSE_PTR
```

Incumbent result:
            t3_B4K: correct, 0.19 ms, 7.3 Gmodmul/s (int64) (13.7% of 53 Gops/s (int64 mul, est))
           t3_B64K: correct, 1.16 ms, 18.7 Gmodmul/s (int64) (35.1% of 53 Gops/s (int64 mul, est))
            t3_B1M: correct, 10.73 ms, 32.3 Gmodmul/s (int64) (60.6% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.3076

## History

- iter  2: compile=OK | correct=True | score=0.19962569094325608
- iter  3: compile=OK | correct=True | score=0.2597024198513166
- iter  4: compile=OK | correct=True | score=0.2465666634612418
- iter  5: compile=OK | correct=True | score=0.24208128837843862
- iter  6: compile=OK | correct=True | score=0.21567426650798874
- iter  7: compile=OK | correct=True | score=0.30761132018665177
- iter  8: compile=OK | correct=True | score=0.2508078890177279
- iter  9: compile=OK | correct=True | score=0.2680695272641571

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
