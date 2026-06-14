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
    ulong s = a + b;
    if (s < a) s += EPSILON;
    if (s >= P_GOLD) s -= P_GOLD;
    return s;
}

// 64x64 -> 128 (lo, hi)
inline void mul_64_128(ulong a, ulong b, thread ulong &lo, thread ulong &hi) {
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

// Add (lo_b, hi_b) into (lo_a, hi_a) accumulator; 128-bit add (carry out discarded;
// we keep hi small by reducing periodically).
inline void add128(thread ulong &lo_a, thread ulong &hi_a, ulong lo_b, ulong hi_b) {
    ulong s = lo_a + lo_b;
    ulong c = (s < lo_a) ? 1ul : 0ul;
    lo_a = s;
    hi_a = hi_a + hi_b + c;
}

// Reduce 128-bit (lo, hi) -> canonical Goldilocks.
// Using:  x_hi splits to hi_lo (low 32) + hi_hi (high 32).
//         2^64 = 2^32 - 1 mod p ; 2^96 = -1 mod p
//   x ≡ lo + hi_lo*(2^32 - 1) - hi_hi   (mod p)
inline ulong gold_reduce128(ulong lo, ulong hi) {
    ulong hi_lo = hi & EPSILON;
    uint  hi_hi = (uint)(hi >> 32);

    // t0 = lo - hi_hi  (mod p)
    ulong t0 = lo - (ulong)hi_hi;
    if (t0 > lo) t0 -= EPSILON;

    // hi_lo*(2^32 - 1) = (hi_lo<<32) - hi_lo  (fits in 64 bits since hi_lo<2^32)
    ulong t1 = (hi_lo << 32) - hi_lo;

    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;
    if (t2 >= P_GOLD) t2 -= P_GOLD;
    return t2;
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong lo, hi;
    mul_64_128(a, b, lo, hi);
    return gold_reduce128(lo, hi);
}

inline ulong sbox(ulong x) {
    ulong x2 = gold_mul(x, x);
    ulong x4 = gold_mul(x2, x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
}

// Compute a*b + accumulator (128-bit) lazily.
// Three-term dot: returns canonical (a0*b0 + a1*b1 + a2*b2) mod p.
// Hi accumulates at most 3 * (2^64 - 1) carries; fits in u64 safely (hi values
// are < 2^64 each so sum < 3*2^64 which doesn't fit in 64 bits in general).
// To stay safe, we reduce mid-way: reduce after 2 products, then accumulate the 3rd.
inline ulong dot3(ulong a0, ulong b0, ulong a1, ulong b1, ulong a2, ulong b2) {
    ulong lo, hi, lo1, hi1;
    mul_64_128(a0, b0, lo, hi);
    mul_64_128(a1, b1, lo1, hi1);
    add128(lo, hi, lo1, hi1);
    // (lo, hi) fits in 128 bits since 2*(2^64-1)^2 < 2^129; hi may use 1 extra bit
    // but we stored carry into hi so hi could be up to 2*(2^64-1) which overflows.
    // To be safe, reduce now.
    ulong r01 = gold_reduce128(lo, hi);
    ulong lo2, hi2;
    mul_64_128(a2, b2, lo2, hi2);
    // add r01 to (lo2, hi2) at low position
    ulong s = lo2 + r01;
    ulong c = (s < lo2) ? 1ul : 0ul;
    lo2 = s;
    hi2 = hi2 + c;
    return gold_reduce128(lo2, hi2);
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
    uint idx   [[thread_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint tg_sz [[threads_per_threadgroup]])
{
    threadgroup ulong tg_ext_mds[T_MAX * T_MAX];
    threadgroup ulong tg_int_diag[T_MAX];
    threadgroup ulong tg_rc_ext[RF_MAX * T_MAX];
    threadgroup ulong tg_rc_int[RP_MAX];

    uint tl       = t;
    uint rf_local = r_f;
    uint rp_local = r_p;
    uint half_f   = rf_local >> 1u;

    uint ext_mds_sz = tl * tl;
    uint rc_ext_sz  = rf_local * tl;

    for (uint i = lid; i < ext_mds_sz; i += tg_sz) tg_ext_mds[i]  = ext_mds[i];
    for (uint i = lid; i < tl;         i += tg_sz) tg_int_diag[i] = int_diag[i];
    for (uint i = lid; i < rc_ext_sz;  i += tg_sz) tg_rc_ext[i]   = rc_ext[i];
    for (uint i = lid; i < rp_local;   i += tg_sz) tg_rc_int[i]   = rc_int[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (idx >= batch) return;

    // Load MDS / diag
    ulong m00=0,m01=0,m02=0,m03=0;
    ulong m10=0,m11=0,m12=0,m13=0;
    ulong m20=0,m21=0,m22=0,m23=0;
    ulong m30=0,m31=0,m32=0,m33=0;
    ulong d0=0,d1=0,d2=0,d3=0;

    m00 = tg_ext_mds[0];
    if (tl > 1) m01 = tg_ext_mds[0*tl + 1];
    if (tl > 2) m02 = tg_ext_mds[0*tl + 2];
    if (tl > 3) m03 = tg_ext_mds[0*tl + 3];
    if (tl > 1) {
        m10 = tg_ext_mds[1*tl + 0];
        m11 = tg_ext_mds[1*tl + 1];
        if (tl > 2) m12 = tg_ext_mds[1*tl + 2];
        if (tl > 3) m13 = tg_ext_mds[1*tl + 3];
    }
    if (tl > 2) {
        m20 = tg_ext_mds[2*tl + 0];
        m21 = tg_ext_mds[2*tl + 1];
        m22 = tg_ext_mds[2*tl + 2];
        if (tl > 3) m23 = tg_ext_mds[2*tl + 3];
    }
    if (tl > 3) {
        m30 = tg_ext_mds[3*tl + 0];
        m31 = tg_ext_mds[3*tl + 1];
        m32 = tg_ext_mds[3*tl + 2];
        m33 = tg_ext_mds[3*tl + 3];
    }
    d0 = tg_int_diag[0];
    if (tl > 1) d1 = tg_int_diag[1];
    if (tl > 2) d2 = tg_int_diag[2];
    if (tl > 3) d3 = tg_int_diag[3];

    // Load state
    ulong s0=0, s1=0, s2=0, s3=0;
    s0 = in_state[idx*tl + 0];
    if (tl > 1) s1 = in_state[idx*tl + 1];
    if (tl > 2) s2 = in_state[idx*tl + 2];
    if (tl > 3) s3 = in_state[idx*tl + 3];

    // ----- t=3 specialized path -----
    if (tl == 3) {
        // External MDS using dot3
        {
            ulong n0 = dot3(m00,s0, m01,s1, m02,s2);
            ulong n1 = dot3(m10,s0, m11,s1, m12,s2);
            ulong n2 = dot3(m20,s0, m21,s1, m22,s2);
            s0=n0; s1=n1; s2=n2;
        }

        for (uint r = 0u; r < half_f; ++r) {
            s0 = sbox(gold_add(s0, tg_rc_ext[r*3 + 0]));
            s1 = sbox(gold_add(s1, tg_rc_ext[r*3 + 1]));
            s2 = sbox(gold_add(s2, tg_rc_ext[r*3 + 2]));
            ulong n0 = dot3(m00,s0, m01,s1, m02,s2);
            ulong n1 = dot3(m10,s0, m11,s1, m12,s2);
            ulong n2 = dot3(m20,s0, m21,s1, m22,s2);
            s0=n0; s1=n1; s2=n2;
        }

        for (uint r = 0u; r < rp_local; ++r) {
            s0 = sbox(gold_add(s0, tg_rc_int[r]));
            ulong sum = gold_add(gold_add(s0, s1), s2);
            ulong n0 = gold_add(sum, gold_mul(d0, s0));
            ulong n1 = gold_add(sum, gold_mul(d1, s1));
            ulong n2 = gold_add(sum, gold_mul(d2, s2));
            s0=n0; s1=n1; s2=n2;
        }

        for (uint r = half_f; r < rf_local; ++r) {
            s0 = sbox(gold_add(s0, tg_rc_ext[r*3 + 0]));
            s1 = sbox(gold_add(s1, tg_rc_ext[r*3 + 1]));
            s2 = sbox(gold_add(s2, tg_rc_ext[r*3 + 2]));
            ulong n0 = dot3(m00,s0, m01,s1, m02,s2);
            ulong n1 = dot3(m10,s0, m11,s1, m12,s2);
            ulong n2 = dot3(m20,s0, m21,s1, m22,s2);
            s0=n0; s1=n1; s2=n2;
        }

        out_state[idx*3 + 0] = s0;
        out_state[idx*3 + 1] = s1;
        out_state[idx*3 + 2] = s2;
        return;
    }

    // ----- Generic path -----
    {
        ulong n0=0,n1=0,n2=0,n3=0;
        n0 = gold_mul(m00, s0);
        if (tl > 1) n0 = gold_add(n0, gold_mul(m01, s1));
        if (tl > 2) n0 = gold_add(n0, gold_mul(m02, s2));
        if (tl > 3) n0 = gold_add(n0, gold_mul(m03, s3));
        if (tl > 1) {
            n1 = gold_mul(m10, s0);
            n1 = gold_add(n1, gold_mul(m11, s1));
            if (tl > 2) n1 = gold_add(n1, gold_mul(m12, s2));
            if (tl > 3) n1 = gold_add(n1, gold_mul(m13, s3));
        }
        if (tl > 2) {
            n2 = gold_mul(m20, s0);
            n2 = gold_add(n2, gold_mul(m21, s1));
            n2 = gold_add(n2, gold_mul(m22, s2));
            if (tl > 3) n2 = gold_add(n2, gold_mul(m23, s3));
        }
        if (tl > 3) {
            n3 = gold_mul(m30, s0);
            n3 = gold_add(n3, gold_mul(m31, s1));
            n3 = gold_add(n3, gold_mul(m32, s2));
            n3 = gold_add(n3, gold_mul(m33, s3));
        }
        s0=n0; s1=n1; s2=n2; s3=n3;
    }

    for (uint r = 0u; r < half_f; ++r) {
        s0 = sbox(gold_add(s0, tg_rc_ext[r*tl + 0]));
        if (tl > 1) s1 = sbox(gold_add(s1, tg_rc_ext[r*tl + 1]));
        if (tl > 2) s2 = sbox(gold_add(s2, tg_rc_ext[r*tl + 2]));
        if (tl > 3) s3 = sbox(gold_add(s3, tg_rc_ext[r*tl + 3]));

        ulong n0=0,n1=0,n2=0,n3=0;
        n0 = gold_mul(m00, s0);
        if (tl > 1) n0 = gold_add(n0, gold_mul(m01, s1));
        if (tl > 2) n0 = gold_add(n0, gold_mul(m02, s2));
        if (tl > 3) n0 = gold_add(n0, gold_mul(m03, s3));
        if (tl > 1) {
            n1 = gold_mul(m10, s0);
            n1 = gold_add(n1, gold_mul(m11, s1));
            if (tl > 2) n1 = gold_add(n1, gold_mul(m12, s2));
            if (tl > 3) n1 = gold_add(n1, gold_mul(m13, s3));
        }
        if (tl > 2) {
            n2 = gold_mul(m20, s0);
            n2 = gold_add(n2, gold_mul(m21, s1));
            n2 = gold_add(n2, gold_mul(m22, s2));
            if (tl > 3) n2 = gold_add(n2, gold_mul(m23, s3));
        }
        if (tl > 3) {
            n3 = gold_mul(m30, s0);
            n3 = gold_add(n3, gold_mul(m31, s1));
            n3 = gold_add(n3, gold_mul(m32, s2));
            n3 = gold_add(n3, gold_mul(m33, s3));
        }
        s0=n0; s1=n1; s2=n2; s3=n3;
    }

    for (uint r = 0u; r < rp_local; ++r) {
        s0 = sbox(gold_add(s0, tg_rc_int[r]));
        ulong sum = s0;
        if (tl > 1) sum = gold_add(sum, s1);
        if (tl > 2) sum = gold_add(sum, s2);
        if (tl > 3) sum = gold_add(sum, s3);
        ulong n0 = gold_add(sum, gold_mul(d0, s0));
        ulong n1 = (tl > 1) ? gold_add(sum, gold_mul(d1, s1)) : 0ul;
        ulong n2 = (tl > 2) ? gold_add(sum, gold_mul(d2, s2)) : 0ul;
        ulong n3 = (tl > 3) ? gold_add(sum, gold_mul(d3, s3)) : 0ul;
        s0=n0; s1=n1; s2=n2; s3=n3;
    }

    for (uint r = half_f; r < rf_local; ++r) {
        s0 = sbox(gold_add(s0, tg_rc_ext[r*tl + 0]));
        if (tl > 1) s1 = sbox(gold_add(s1, tg_rc_ext[r*tl + 1]));
        if (tl > 2) s2 = sbox(gold_add(s2, tg_rc_ext[r*tl + 2]));
        if (tl > 3) s3 = sbox(gold_add(s3, tg_rc_ext[r*tl + 3]));

        ulong n0=0,n1=0,n2=0,n3=0;
        n0 = gold_mul(m00, s0);
        if (tl > 1) n0 = gold_add(n0, gold_mul(m01, s1));
        if (tl > 2) n0 = gold_add(n0, gold_mul(m02, s2));
        if (tl > 3) n0 = gold_add(n0, gold_mul(m03, s3));
        if (tl > 1) {
            n1 = gold_mul(m10, s0);
            n1 = gold_add(n1, gold_mul(m11, s1));
            if (tl > 2) n1 = gold_add(n1, gold_mul(m12, s2));
            if (tl > 3) n1 = gold_add(n1, gold_mul(m13, s3));
        }
        if (tl > 2) {
            n2 = gold_mul(m20, s0);
            n2 = gold_add(n2, gold_mul(m21, s1));
            n2 = gold_add(n2, gold_mul(m22, s2));
            if (tl > 3) n2 = gold_add(n2, gold_mul(m23, s3));
        }
        if (tl > 3) {
            n3 = gold_mul(m30, s0);
            n3 = gold_add(n3, gold_mul(m31, s1));
            n3 = gold_add(n3, gold_mul(m32, s2));
            n3 = gold_add(n3, gold_mul(m33, s3));
        }
        s0=n0; s1=n1; s2=n2; s3=n3;
    }

    out_state[idx*tl + 0] = s0;
    if (tl > 1) out_state[idx*tl + 1] = s1;
    if (tl > 2) out_state[idx*tl + 2] = s2;
    if (tl > 3) out_state[idx*tl + 3] = s3;
}
```

Result of previous attempt:
            t3_B4K: correct, 0.22 ms, 6.2 Gmodmul/s (int64) (1.1% of 562 Gops/s (int64 mul, est))
           t3_B64K: correct, 1.13 ms, 19.2 Gmodmul/s (int64) (3.4% of 562 Gops/s (int64 mul, est))
            t3_B1M: correct, 12.87 ms, 27.0 Gmodmul/s (int64) (4.8% of 562 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0262

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;

constexpr constant uint T_MAX = 4u;
constexpr constant uint RF_MAX = 8u;
constexpr constant uint RP_MAX = 32u;

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    if (t < a) t += EPSILON;
    if (t >= P_GOLD) t -= P_GOLD;
    return t;
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
    return gold_reduce128(lo, hi);
}

inline ulong sbox(ulong x) {
    ulong x2 = gold_mul(x, x);
    ulong x4 = gold_mul(x2, x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
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
    uint idx        [[thread_position_in_grid]],
    uint lid        [[thread_position_in_threadgroup]],
    uint tg_sz      [[threads_per_threadgroup]])
{
    threadgroup ulong tg_ext_mds[T_MAX * T_MAX];
    threadgroup ulong tg_int_diag[T_MAX];
    threadgroup ulong tg_rc_ext[RF_MAX * T_MAX];
    threadgroup ulong tg_rc_int[RP_MAX];

    uint tl       = t;
    uint rf_local = r_f;
    uint rp_local = r_p;
    uint half_f   = rf_local >> 1u;

    uint ext_mds_sz = tl * tl;
    uint rc_ext_sz  = rf_local * tl;

    for (uint i = lid; i < ext_mds_sz; i += tg_sz) tg_ext_mds[i]  = ext_mds[i];
    for (uint i = lid; i < tl;         i += tg_sz) tg_int_diag[i] = int_diag[i];
    for (uint i = lid; i < rc_ext_sz;  i += tg_sz) tg_rc_ext[i]   = rc_ext[i];
    for (uint i = lid; i < rp_local;   i += tg_sz) tg_rc_int[i]   = rc_int[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (idx >= batch) return;

    // Load MDS / diag into registers
    ulong m00=0,m01=0,m02=0,m03=0;
    ulong m10=0,m11=0,m12=0,m13=0;
    ulong m20=0,m21=0,m22=0,m23=0;
    ulong m30=0,m31=0,m32=0,m33=0;
    ulong d0=0,d1=0,d2=0,d3=0;

    m00 = tg_ext_mds[0];
    if (tl > 1) m01 = tg_ext_mds[0*tl + 1];
    if (tl > 2) m02 = tg_ext_mds[0*tl + 2];
    if (tl > 3) m03 = tg_ext_mds[0*tl + 3];
    if (tl > 1) {
        m10 = tg_ext_mds[1*tl + 0];
        m11 = tg_ext_mds[1*tl + 1];
        if (tl > 2) m12 = tg_ext_mds[1*tl + 2];
        if (tl > 3) m13 = tg_ext_mds[1*tl + 3];
    }
    if (tl > 2) {
        m20 = tg_ext_mds[2*tl + 0];
        m21 = tg_ext_mds[2*tl + 1];
        m22 = tg_ext_mds[2*tl + 2];
        if (tl > 3) m23 = tg_ext_mds[2*tl + 3];
    }
    if (tl > 3) {
        m30 = tg_ext_mds[3*tl + 0];
        m31 = tg_ext_mds[3*tl + 1];
        m32 = tg_ext_mds[3*tl + 2];
        m33 = tg_ext_mds[3*tl + 3];
    }
    d0 = tg_int_diag[0];
    if (tl > 1) d1 = tg_int_diag[1];
    if (tl > 2) d2 = tg_int_diag[2];
    if (tl > 3) d3 = tg_int_diag[3];

    // Load state
    ulong s0=0, s1=0, s2=0, s3=0;
    s0 = in_state[idx*tl + 0];
    if (tl > 1) s1 = in_state[idx*tl + 1];
    if (tl > 2) s2 = in_state[idx*tl + 2];
    if (tl > 3) s3 = in_state[idx*tl + 3];

    // Specialized t=3 fast path
    if (tl == 3) {
        // Pre-mul by external MDS
        {
            ulong n0 = gold_add(gold_add(gold_mul(m00,s0), gold_mul(m01,s1)), gold_mul(m02,s2));
            ulong n1 = gold_add(gold_add(gold_mul(m10,s0), gold_mul(m11,s1)), gold_mul(m12,s2));
            ulong n2 = gold_add(gold_add(gold_mul(m20,s0), gold_mul(m21,s1)), gold_mul(m22,s2));
            s0 = n0; s1 = n1; s2 = n2;
        }

        // First half full rounds
        for (uint r = 0u; r < half_f; ++r) {
            s0 = sbox(gold_add(s0, tg_rc_ext[r*3 + 0]));
            s1 = sbox(gold_add(s1, tg_rc_ext[r*3 + 1]));
            s2 = sbox(gold_add(s2, tg_rc_ext[r*3 + 2]));
            ulong n0 = gold_add(gold_add(gold_mul(m00,s0), gold_mul(m01,s1)), gold_mul(m02,s2));
            ulong n1 = gold_add(gold_add(gold_mul(m10,s0), gold_mul(m11,s1)), gold_mul(m12,s2));
            ulong n2 = gold_add(gold_add(gold_mul(m20,s0), gold_mul(m21,s1)), gold_mul(m22,s2));
            s0 = n0; s1 = n1; s2 = n2;
        }

        // Partial rounds: y[i] = sum + d[i]*s[i]
        for (uint r = 0u; r < rp_local; ++r) {
            s0 = sbox(gold_add(s0, tg_rc_int[r]));
            ulong sum = gold_add(gold_add(s0, s1), s2);
            ulong n0 = gold_add(sum, gold_mul(d0, s0));
            ulong n1 = gold_add(sum, gold_mul(d1, s1));
            ulong n2 = gold_add(sum, gold_mul(d2, s2));
            s0 = n0; s1 = n1; s2 = n2;
        }

        // Second half full rounds
        for (uint r = half_f; r < rf_local; ++r) {
            s0 = sbox(gold_add(s0, tg_rc_ext[r*3 + 0]));
            s1 = sbox(gold_add(s1, tg_rc_ext[r*3 + 1]));
            s2 = sbox(gold_add(s2, tg_rc_ext[r*3 + 2]));
            ulong n0 = gold_add(gold_add(gold_mul(m00,s0), gold_mul(m01,s1)), gold_mul(m02,s2));
            ulong n1 = gold_add(gold_add(gold_mul(m10,s0), gold_mul(m11,s1)), gold_mul(m12,s2));
            ulong n2 = gold_add(gold_add(gold_mul(m20,s0), gold_mul(m21,s1)), gold_mul(m22,s2));
            s0 = n0; s1 = n1; s2 = n2;
        }

        out_state[idx*3 + 0] = s0;
        out_state[idx*3 + 1] = s1;
        out_state[idx*3 + 2] = s2;
        return;
    }

    // Generic path
    {
        ulong n0=0,n1=0,n2=0,n3=0;
        n0 = gold_mul(m00, s0);
        if (tl > 1) n0 = gold_add(n0, gold_mul(m01, s1));
        if (tl > 2) n0 = gold_add(n0, gold_mul(m02, s2));
        if (tl > 3) n0 = gold_add(n0, gold_mul(m03, s3));
        if (tl > 1) {
            n1 = gold_mul(m10, s0);
            n1 = gold_add(n1, gold_mul(m11, s1));
            if (tl > 2) n1 = gold_add(n1, gold_mul(m12, s2));
            if (tl > 3) n1 = gold_add(n1, gold_mul(m13, s3));
        }
        if (tl > 2) {
            n2 = gold_mul(m20, s0);
            n2 = gold_add(n2, gold_mul(m21, s1));
            n2 = gold_add(n2, gold_mul(m22, s2));
            if (tl > 3) n2 = gold_add(n2, gold_mul(m23, s3));
        }
        if (tl > 3) {
            n3 = gold_mul(m30, s0);
            n3 = gold_add(n3, gold_mul(m31, s1));
            n3 = gold_add(n3, gold_mul(m32, s2));
            n3 = gold_add(n3, gold_mul(m33, s3));
        }
        s0=n0; s1=n1; s2=n2; s3=n3;
    }

    for (uint r = 0u; r < half_f; ++r) {
        s0 = sbox(gold_add(s0, tg_rc_ext[r*tl + 0]));
        if (tl > 1) s1 = sbox(gold_add(s1, tg_rc_ext[r*tl + 1]));
        if (tl > 2) s2 = sbox(gold_add(s2, tg_rc_ext[r*tl + 2]));
        if (tl > 3) s3 = sbox(gold_add(s3, tg_rc_ext[r*tl + 3]));

        ulong n0=0,n1=0,n2=0,n3=0;
        n0 = gold_mul(m00, s0);
        if (tl > 1) n0 = gold_add(n0, gold_mul(m01, s1));
        if (tl > 2) n0 = gold_add(n0, gold_mul(m02, s2));
        if (tl > 3) n0 = gold_add(n0, gold_mul(m03, s3));
        if (tl > 1) {
            n1 = gold_mul(m10, s0);
            n1 = gold_add(n1, gold_mul(m11, s1));
            if (tl > 2) n1 = gold_add(n1, gold_mul(m12, s2));
            if (tl > 3) n1 = gold_add(n1, gold_mul(m13, s3));
        }
        if (tl > 2) {
            n2 = gold_mul(m20, s0);
            n2 = gold_add(n2, gold_mul(m21, s1));
            n2 = gold_add(n2, gold_mul(m22, s2));
            if (tl > 3) n2 = gold_add(n2, gold_mul(m23, s3));
        }
        if (tl > 3) {
            n3 = gold_mul(m30, s0);
            n3 = gold_add(n3, gold_mul(m31, s1));
            n3 = gold_add(n3, gold_mul(m32, s2));
            n3 = gold_add(n3, gold_mul(m33, s3));
        }
        s0=n0; s1=n1; s2=n2; s3=n3;
    }

    for (uint r = 0u; r < rp_local; ++r) {
        s0 = sbox(gold_add(s0, tg_rc_int[r]));
        ulong sum = s0;
        if (tl > 1) sum = gold_add(sum, s1);
        if (tl > 2) sum = gold_add(sum, s2);
        if (tl > 3) sum = gold_add(sum, s3);
        ulong n0 = gold_add(sum, gold_mul(d0, s0));
        ulong n1 = (tl > 1) ? gold_add(sum, gold_mul(d1, s1)) : 0ul;
        ulong n2 = (tl > 2) ? gold_add(sum, gold_mul(d2, s2)) : 0ul;
        ulong n3 = (tl > 3) ? gold_add(sum, gold_mul(d3, s3)) : 0ul;
        s0=n0; s1=n1; s2=n2; s3=n3;
    }

    for (uint r = half_f; r < rf_local; ++r) {
        s0 = sbox(gold_add(s0, tg_rc_ext[r*tl + 0]));
        if (tl > 1) s1 = sbox(gold_add(s1, tg_rc_ext[r*tl + 1]));
        if (tl > 2) s2 = sbox(gold_add(s2, tg_rc_ext[r*tl + 2]));
        if (tl > 3) s3 = sbox(gold_add(s3, tg_rc_ext[r*tl + 3]));

        ulong n0=0,n1=0,n2=0,n3=0;
        n0 = gold_mul(m00, s0);
        if (tl > 1) n0 = gold_add(n0, gold_mul(m01, s1));
        if (tl > 2) n0 = gold_add(n0, gold_mul(m02, s2));
        if (tl > 3) n0 = gold_add(n0, gold_mul(m03, s3));
        if (tl > 1) {
            n1 = gold_mul(m10, s0);
            n1 = gold_add(n1, gold_mul(m11, s1));
            if (tl > 2) n1 = gold_add(n1, gold_mul(m12, s2));
            if (tl > 3) n1 = gold_add(n1, gold_mul(m13, s3));
        }
        if (tl > 2) {
            n2 = gold_mul(m20, s0);
            n2 = gold_add(n2, gold_mul(m21, s1));
            n2 = gold_add(n2, gold_mul(m22, s2));
            if (tl > 3) n2 = gold_add(n2, gold_mul(m23, s3));
        }
        if (tl > 3) {
            n3 = gold_mul(m30, s0);
            n3 = gold_add(n3, gold_mul(m31, s1));
            n3 = gold_add(n3, gold_mul(m32, s2));
            n3 = gold_add(n3, gold_mul(m33, s3));
        }
        s0=n0; s1=n1; s2=n2; s3=n3;
    }

    out_state[idx*tl + 0] = s0;
    if (tl > 1) out_state[idx*tl + 1] = s1;
    if (tl > 2) out_state[idx*tl + 2] = s2;
    if (tl > 3) out_state[idx*tl + 3] = s3;
}
```

Incumbent result:
            t3_B4K: correct, 0.24 ms, 5.8 Gmodmul/s (int64) (1.0% of 562 Gops/s (int64 mul, est))
           t3_B64K: correct, 0.91 ms, 23.8 Gmodmul/s (int64) (4.2% of 562 Gops/s (int64 mul, est))
            t3_B1M: correct, 14.09 ms, 24.6 Gmodmul/s (int64) (4.4% of 562 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0267

## History

- iter  0: compile=OK | correct=True | score=0.021099312837649573
- iter  1: compile=OK | correct=True | score=0.026071433111135436
- iter  2: compile=FAIL | correct=False | score=N/A
- iter  3: compile=OK | correct=True | score=0.02667441386821328
- iter  4: compile=OK | correct=True | score=0.02370255014602819
- iter  5: compile=OK | correct=True | score=0.026177034865347528

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
