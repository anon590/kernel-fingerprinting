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

static inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

// a, b canonical (< p)
static inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    if (t < a) t += EPSILON;
    if (t >= P_GOLD) t -= P_GOLD;
    return t;
}

// Reduce (lo + hi*2^64) mod p. hi arbitrary 64-bit. Result canonical.
static inline ulong gold_reduce_128(ulong lo, ulong hi) {
    ulong x_hi_lo = hi & EPSILON;     // low 32 of hi
    ulong x_hi_hi = hi >> 32;         // high 32 of hi

    ulong t0 = lo - x_hi_hi;
    if (t0 > lo) t0 -= EPSILON;       // borrow

    ulong t1 = (x_hi_lo << 32) - x_hi_lo;

    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;       // carry

    return gold_canonical(t2);
}

static inline void mul_full_128(ulong a, ulong b, thread ulong &lo, thread ulong &hi) {
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

static inline ulong gold_mul(ulong a, ulong b) {
    ulong lo, hi;
    mul_full_128(a, b, lo, hi);
    return gold_reduce_128(lo, hi);
}

static inline void mac_128(ulong a, ulong b, thread ulong &lo, thread ulong &hi) {
    ulong pl, ph;
    mul_full_128(a, b, pl, ph);
    ulong nl = lo + pl;
    ulong c  = (nl < lo) ? 1ul : 0ul;
    lo = nl;
    hi = hi + ph + c;
}

static inline ulong sbox(ulong x) {
    ulong x2 = gold_mul(x, x);
    ulong x4 = gold_mul(x2, x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
}

// y = sum + diag*s_i, sum canonical
static inline ulong sum_plus_mul(ulong sum, ulong diag_i, ulong s_i) {
    ulong pl, ph;
    mul_full_128(diag_i, s_i, pl, ph);
    ulong nl = pl + sum;
    if (nl < pl) ph += 1ul;
    return gold_reduce_128(nl, ph);
}

// ----- Interleaved 3-row matvec for t=3 -----
static inline void matvec3_interleaved(
    ulong s0, ulong s1, ulong s2,
    ulong m00, ulong m01, ulong m02,
    ulong m10, ulong m11, ulong m12,
    ulong m20, ulong m21, ulong m22,
    thread ulong &n0, thread ulong &n1, thread ulong &n2)
{
    ulong l0=0,h0=0,l1=0,h1=0,l2=0,h2=0;
    mac_128(m00,s0,l0,h0); mac_128(m10,s0,l1,h1); mac_128(m20,s0,l2,h2);
    mac_128(m01,s1,l0,h0); mac_128(m11,s1,l1,h1); mac_128(m21,s1,l2,h2);
    mac_128(m02,s2,l0,h0); mac_128(m12,s2,l1,h1); mac_128(m22,s2,l2,h2);
    n0 = gold_reduce_128(l0,h0);
    n1 = gold_reduce_128(l1,h1);
    n2 = gold_reduce_128(l2,h2);
}

static inline void matvec4_interleaved(
    ulong s0, ulong s1, ulong s2, ulong s3,
    ulong m00, ulong m01, ulong m02, ulong m03,
    ulong m10, ulong m11, ulong m12, ulong m13,
    ulong m20, ulong m21, ulong m22, ulong m23,
    ulong m30, ulong m31, ulong m32, ulong m33,
    thread ulong &n0, thread ulong &n1, thread ulong &n2, thread ulong &n3)
{
    ulong l0=0,h0=0,l1=0,h1=0,l2=0,h2=0,l3=0,h3=0;
    mac_128(m00,s0,l0,h0); mac_128(m10,s0,l1,h1); mac_128(m20,s0,l2,h2); mac_128(m30,s0,l3,h3);
    mac_128(m01,s1,l0,h0); mac_128(m11,s1,l1,h1); mac_128(m21,s1,l2,h2); mac_128(m31,s1,l3,h3);
    mac_128(m02,s2,l0,h0); mac_128(m12,s2,l1,h1); mac_128(m22,s2,l2,h2); mac_128(m32,s2,l3,h3);
    mac_128(m03,s3,l0,h0); mac_128(m13,s3,l1,h1); mac_128(m23,s3,l2,h2); mac_128(m33,s3,l3,h3);
    n0 = gold_reduce_128(l0,h0);
    n1 = gold_reduce_128(l1,h1);
    n2 = gold_reduce_128(l2,h2);
    n3 = gold_reduce_128(l3,h3);
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

        ulong n0, n1, n2;
        matvec3_interleaved(s0,s1,s2, m00,m01,m02, m10,m11,m12, m20,m21,m22, n0,n1,n2);
        s0=n0; s1=n1; s2=n2;

        for (uint r = 0u; r < half_f; ++r) {
            s0 = sbox(gold_add(s0, rc_ext[r * 3u + 0u]));
            s1 = sbox(gold_add(s1, rc_ext[r * 3u + 1u]));
            s2 = sbox(gold_add(s2, rc_ext[r * 3u + 2u]));
            matvec3_interleaved(s0,s1,s2, m00,m01,m02, m10,m11,m12, m20,m21,m22, n0,n1,n2);
            s0=n0; s1=n1; s2=n2;
        }

        for (uint r = 0u; r < rp; ++r) {
            s0 = sbox(gold_add(s0, rc_int[r]));
            ulong sum = gold_add(gold_add(s0, s1), s2);
            ulong p0 = sum_plus_mul(sum, d0, s0);
            ulong p1 = sum_plus_mul(sum, d1, s1);
            ulong p2 = sum_plus_mul(sum, d2, s2);
            s0 = p0; s1 = p1; s2 = p2;
        }

        for (uint r = half_f; r < rf; ++r) {
            s0 = sbox(gold_add(s0, rc_ext[r * 3u + 0u]));
            s1 = sbox(gold_add(s1, rc_ext[r * 3u + 1u]));
            s2 = sbox(gold_add(s2, rc_ext[r * 3u + 2u]));
            matvec3_interleaved(s0,s1,s2, m00,m01,m02, m10,m11,m12, m20,m21,m22, n0,n1,n2);
            s0=n0; s1=n1; s2=n2;
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
            ulong l0=0,h0=0,l1=0,h1=0;
            mac_128(m00,s0,l0,h0); mac_128(m10,s0,l1,h1);
            mac_128(m01,s1,l0,h0); mac_128(m11,s1,l1,h1);
            ulong n0=gold_reduce_128(l0,h0), n1=gold_reduce_128(l1,h1);
            s0=n0; s1=n1;
        }
        for (uint r = 0u; r < half_f; ++r) {
            s0 = sbox(gold_add(s0, rc_ext[r * 2u + 0u]));
            s1 = sbox(gold_add(s1, rc_ext[r * 2u + 1u]));
            ulong l0=0,h0=0,l1=0,h1=0;
            mac_128(m00,s0,l0,h0); mac_128(m10,s0,l1,h1);
            mac_128(m01,s1,l0,h0); mac_128(m11,s1,l1,h1);
            ulong n0=gold_reduce_128(l0,h0), n1=gold_reduce_128(l1,h1);
            s0=n0; s1=n1;
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
            ulong l0=0,h0=0,l1=0,h1=0;
            mac_128(m00,s0,l0,h0); mac_128(m10,s0,l1,h1);
            mac_128(m01,s1,l0,h0); mac_128(m11,s1,l1,h1);
            ulong n0=gold_reduce_128(l0,h0), n1=gold_reduce_128(l1,h1);
            s0=n0; s1=n1;
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

        ulong n0,n1,n2,n3;
        matvec4_interleaved(s0,s1,s2,s3,
            m00,m01,m02,m03, m10,m11,m12,m13, m20,m21,m22,m23, m30,m31,m32,m33,
            n0,n1,n2,n3);
        s0=n0; s1=n1; s2=n2; s3=n3;

        for (uint r = 0u; r < half_f; ++r) {
            s0 = sbox(gold_add(s0, rc_ext[r * 4u + 0u]));
            s1 = sbox(gold_add(s1, rc_ext[r * 4u + 1u]));
            s2 = sbox(gold_add(s2, rc_ext[r * 4u + 2u]));
            s3 = sbox(gold_add(s3, rc_ext[r * 4u + 3u]));
            matvec4_interleaved(s0,s1,s2,s3,
                m00,m01,m02,m03, m10,m11,m12,m13, m20,m21,m22,m23, m30,m31,m32,m33,
                n0,n1,n2,n3);
            s0=n0; s1=n1; s2=n2; s3=n3;
        }
        for (uint r = 0u; r < rp; ++r) {
            s0 = sbox(gold_add(s0, rc_int[r]));
            ulong sum = gold_add(gold_add(s0, s1), gold_add(s2, s3));
            ulong p0 = sum_plus_mul(sum, d0, s0);
            ulong p1 = sum_plus_mul(sum, d1, s1);
            ulong p2 = sum_plus_mul(sum, d2, s2);
            ulong p3 = sum_plus_mul(sum, d3, s3);
            s0=p0; s1=p1; s2=p2; s3=p3;
        }
        for (uint r = half_f; r < rf; ++r) {
            s0 = sbox(gold_add(s0, rc_ext[r * 4u + 0u]));
            s1 = sbox(gold_add(s1, rc_ext[r * 4u + 1u]));
            s2 = sbox(gold_add(s2, rc_ext[r * 4u + 2u]));
            s3 = sbox(gold_add(s3, rc_ext[r * 4u + 3u]));
            matvec4_interleaved(s0,s1,s2,s3,
                m00,m01,m02,m03, m10,m11,m12,m13, m20,m21,m22,m23, m30,m31,m32,m33,
                n0,n1,n2,n3);
            s0=n0; s1=n1; s2=n2; s3=n3;
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
            ulong l = 0ul, h = 0ul;
            for (uint j = 0u; j < tt; ++j) {
                mac_128(ext_mds[i * tt + j], state[j], l, h);
            }
            tmp[i] = gold_reduce_128(l, h);
        }
        for (uint i = 0u; i < tt; ++i) state[i] = tmp[i];
    }
    for (uint r = 0u; r < half_f; ++r) {
        for (uint i = 0u; i < tt; ++i) state[i] = sbox(gold_add(state[i], rc_ext[r * tt + i]));
        ulong tmp[T_MAX];
        for (uint i = 0u; i < tt; ++i) {
            ulong l = 0ul, h = 0ul;
            for (uint j = 0u; j < tt; ++j) {
                mac_128(ext_mds[i * tt + j], state[j], l, h);
            }
            tmp[i] = gold_reduce_128(l, h);
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
            ulong l = 0ul, h = 0ul;
            for (uint j = 0u; j < tt; ++j) {
                mac_128(ext_mds[i * tt + j], state[j], l, h);
            }
            tmp[i] = gold_reduce_128(l, h);
        }
        for (uint i = 0u; i < tt; ++i) state[i] = tmp[i];
    }
    for (uint i = 0u; i < tt; ++i) out_state[idx * tt + i] = state[i];
}
```

Result of previous attempt:
            t3_B4K: correct, 0.20 ms, 6.8 Gmodmul/s (int64) (12.7% of 53 Gops/s (int64 mul, est))
           t3_B64K: correct, 1.24 ms, 17.6 Gmodmul/s (int64) (32.9% of 53 Gops/s (int64 mul, est))
            t3_B1M: correct, 11.77 ms, 29.5 Gmodmul/s (int64) (55.3% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.2850

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

// a, b canonical (< p). a+b < 2p, single subtraction suffices.
inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    // wrap if a+b >= 2^64; since a,b < p < 2^64, wrap means t = a+b - 2^64,
    // and true sum >= 2^64 > p, so we need t + EPSILON (= -p mod 2^64),
    // followed by canonical-clamp.
    if (t < a) t += EPSILON;
    if (t >= P_GOLD) t -= P_GOLD;
    return t;
}

// Reduce (lo + hi*2^64) mod p, hi < 2^64. Result canonical.
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

// 128-bit fused multiply-accumulate: (lo,hi) += a*b
inline void mac_128(ulong a, ulong b, thread ulong &lo, thread ulong &hi) {
    ulong pl, ph;
    mul_full_128(a, b, pl, ph);
    ulong nl = lo + pl;
    ulong c  = (nl < lo) ? 1ul : 0ul;
    lo = nl;
    hi = hi + ph + c;
}

// y = sum + diag*s, sum canonical (< p).
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

        // Fused matvec: accumulate 3 full products in 128-bit, reduce once per row.
        {
            ulong l, h;
            ulong n0, n1, n2;
            l=0; h=0; mac_128(m00,s0,l,h); mac_128(m01,s1,l,h); mac_128(m02,s2,l,h);
            n0 = gold_reduce_128(l, h);
            l=0; h=0; mac_128(m10,s0,l,h); mac_128(m11,s1,l,h); mac_128(m12,s2,l,h);
            n1 = gold_reduce_128(l, h);
            l=0; h=0; mac_128(m20,s0,l,h); mac_128(m21,s1,l,h); mac_128(m22,s2,l,h);
            n2 = gold_reduce_128(l, h);
            s0 = n0; s1 = n1; s2 = n2;
        }

        for (uint r = 0u; r < half_f; ++r) {
            s0 = sbox(gold_add(s0, rc_ext[r * 3u + 0u]));
            s1 = sbox(gold_add(s1, rc_ext[r * 3u + 1u]));
            s2 = sbox(gold_add(s2, rc_ext[r * 3u + 2u]));
            ulong l, h, n0, n1, n2;
            l=0; h=0; mac_128(m00,s0,l,h); mac_128(m01,s1,l,h); mac_128(m02,s2,l,h);
            n0 = gold_reduce_128(l, h);
            l=0; h=0; mac_128(m10,s0,l,h); mac_128(m11,s1,l,h); mac_128(m12,s2,l,h);
            n1 = gold_reduce_128(l, h);
            l=0; h=0; mac_128(m20,s0,l,h); mac_128(m21,s1,l,h); mac_128(m22,s2,l,h);
            n2 = gold_reduce_128(l, h);
            s0 = n0; s1 = n1; s2 = n2;
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
            ulong l, h, n0, n1, n2;
            l=0; h=0; mac_128(m00,s0,l,h); mac_128(m01,s1,l,h); mac_128(m02,s2,l,h);
            n0 = gold_reduce_128(l, h);
            l=0; h=0; mac_128(m10,s0,l,h); mac_128(m11,s1,l,h); mac_128(m12,s2,l,h);
            n1 = gold_reduce_128(l, h);
            l=0; h=0; mac_128(m20,s0,l,h); mac_128(m21,s1,l,h); mac_128(m22,s2,l,h);
            n2 = gold_reduce_128(l, h);
            s0 = n0; s1 = n1; s2 = n2;
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
            ulong l, h, n0, n1;
            l=0; h=0; mac_128(m00,s0,l,h); mac_128(m01,s1,l,h); n0 = gold_reduce_128(l,h);
            l=0; h=0; mac_128(m10,s0,l,h); mac_128(m11,s1,l,h); n1 = gold_reduce_128(l,h);
            s0 = n0; s1 = n1;
        }
        for (uint r = 0u; r < half_f; ++r) {
            s0 = sbox(gold_add(s0, rc_ext[r * 2u + 0u]));
            s1 = sbox(gold_add(s1, rc_ext[r * 2u + 1u]));
            ulong l, h, n0, n1;
            l=0; h=0; mac_128(m00,s0,l,h); mac_128(m01,s1,l,h); n0 = gold_reduce_128(l,h);
            l=0; h=0; mac_128(m10,s0,l,h); mac_128(m11,s1,l,h); n1 = gold_reduce_128(l,h);
            s0 = n0; s1 = n1;
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
            ulong l, h, n0, n1;
            l=0; h=0; mac_128(m00,s0,l,h); mac_128(m01,s1,l,h); n0 = gold_reduce_128(l,h);
            l=0; h=0; mac_128(m10,s0,l,h); mac_128(m11,s1,l,h); n1 = gold_reduce_128(l,h);
            s0 = n0; s1 = n1;
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
            ulong l, h, n0, n1, n2, n3;
            l=0; h=0; mac_128(m00,s0,l,h); mac_128(m01,s1,l,h); mac_128(m02,s2,l,h); mac_128(m03,s3,l,h); n0=gold_reduce_128(l,h);
            l=0; h=0; mac_128(m10,s0,l,h); mac_128(m11,s1,l,h); mac_128(m12,s2,l,h); mac_128(m13,s3,l,h); n1=gold_reduce_128(l,h);
            l=0; h=0; mac_128(m20,s0,l,h); mac_128(m21,s1,l,h); mac_128(m22,s2,l,h); mac_128(m23,s3,l,h); n2=gold_reduce_128(l,h);
            l=0; h=0; mac_128(m30,s0,l,h); mac_128(m31,s1,l,h); mac_128(m32,s2,l,h); mac_128(m33,s3,l,h); n3=gold_reduce_128(l,h);
            s0=n0; s1=n1; s2=n2; s3=n3;
        }
        for (uint r = 0u; r < half_f; ++r) {
            s0 = sbox(gold_add(s0, rc_ext[r * 4u + 0u]));
            s1 = sbox(gold_add(s1, rc_ext[r * 4u + 1u]));
            s2 = sbox(gold_add(s2, rc_ext[r * 4u + 2u]));
            s3 = sbox(gold_add(s3, rc_ext[r * 4u + 3u]));
            ulong l, h, n0, n1, n2, n3;
            l=0; h=0; mac_128(m00,s0,l,h); mac_128(m01,s1,l,h); mac_128(m02,s2,l,h); mac_128(m03,s3,l,h); n0=gold_reduce_128(l,h);
            l=0; h=0; mac_128(m10,s0,l,h); mac_128(m11,s1,l,h); mac_128(m12,s2,l,h); mac_128(m13,s3,l,h); n1=gold_reduce_128(l,h);
            l=0; h=0; mac_128(m20,s0,l,h); mac_128(m21,s1,l,h); mac_128(m22,s2,l,h); mac_128(m23,s3,l,h); n2=gold_reduce_128(l,h);
            l=0; h=0; mac_128(m30,s0,l,h); mac_128(m31,s1,l,h); mac_128(m32,s2,l,h); mac_128(m33,s3,l,h); n3=gold_reduce_128(l,h);
            s0=n0; s1=n1; s2=n2; s3=n3;
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
            ulong l, h, n0, n1, n2, n3;
            l=0; h=0; mac_128(m00,s0,l,h); mac_128(m01,s1,l,h); mac_128(m02,s2,l,h); mac_128(m03,s3,l,h); n0=gold_reduce_128(l,h);
            l=0; h=0; mac_128(m10,s0,l,h); mac_128(m11,s1,l,h); mac_128(m12,s2,l,h); mac_128(m13,s3,l,h); n1=gold_reduce_128(l,h);
            l=0; h=0; mac_128(m20,s0,l,h); mac_128(m21,s1,l,h); mac_128(m22,s2,l,h); mac_128(m23,s3,l,h); n2=gold_reduce_128(l,h);
            l=0; h=0; mac_128(m30,s0,l,h); mac_128(m31,s1,l,h); mac_128(m32,s2,l,h); mac_128(m33,s3,l,h); n3=gold_reduce_128(l,h);
            s0=n0; s1=n1; s2=n2; s3=n3;
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
            ulong l = 0ul, h = 0ul;
            for (uint j = 0u; j < tt; ++j) {
                mac_128(ext_mds[i * tt + j], state[j], l, h);
            }
            tmp[i] = gold_reduce_128(l, h);
        }
        for (uint i = 0u; i < tt; ++i) state[i] = tmp[i];
    }
    for (uint r = 0u; r < half_f; ++r) {
        for (uint i = 0u; i < tt; ++i) state[i] = sbox(gold_add(state[i], rc_ext[r * tt + i]));
        ulong tmp[T_MAX];
        for (uint i = 0u; i < tt; ++i) {
            ulong l = 0ul, h = 0ul;
            for (uint j = 0u; j < tt; ++j) {
                mac_128(ext_mds[i * tt + j], state[j], l, h);
            }
            tmp[i] = gold_reduce_128(l, h);
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
            ulong l = 0ul, h = 0ul;
            for (uint j = 0u; j < tt; ++j) {
                mac_128(ext_mds[i * tt + j], state[j], l, h);
            }
            tmp[i] = gold_reduce_128(l, h);
        }
        for (uint i = 0u; i < tt; ++i) state[i] = tmp[i];
    }
    for (uint i = 0u; i < tt; ++i) out_state[idx * tt + i] = state[i];
}
```

Incumbent result:
            t3_B4K: correct, 0.10 ms, 13.7 Gmodmul/s (int64) (25.7% of 53 Gops/s (int64 mul, est))
           t3_B64K: correct, 1.11 ms, 19.6 Gmodmul/s (int64) (36.8% of 53 Gops/s (int64 mul, est))
            t3_B1M: correct, 12.79 ms, 27.1 Gmodmul/s (int64) (50.9% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.3635

## History

- iter  0: compile=OK | correct=True | score=0.22313100350851453
- iter  1: compile=OK | correct=True | score=0.22945717366315468
- iter  2: compile=OK | correct=True | score=0.2468564646184756
- iter  3: compile=OK | correct=True | score=0.28001545247299503
- iter  4: compile=OK | correct=True | score=0.32724556471757404
- iter  5: compile=OK | correct=True | score=0.3634876990761484
- iter  6: compile=OK | correct=True | score=0.28497758875930695

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
