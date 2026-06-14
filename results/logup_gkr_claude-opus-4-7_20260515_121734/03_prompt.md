## Task: logup_gkr

Batched LogUp lookup-argument core (Haebock 2022; back-end of Jolt / Lasso / Plonkish v2). Given a table T[M] and a witness column w[N] where each w_i := T[witness_idx[i]], compute (1) multiplicities m[j] = #{ i : witness_idx[i] == j } for j in [0, M), and (2) the running product
  P = prod_{i=0..N-1} 1/(alpha - w_i)
     * prod_{j=0..M-1} m_j / (alpha - T_j)   (mod p)
where alpha is a verifier challenge. The host fixes alpha so that alpha is outside the set of table values -- no zero denominators arise.

Combined fingerprint stream of length N + M:
  k <  N:   x_k = T[witness_idx[k]],  num_k = 1
  k >= N:   x_k = T[k - N],            num_k = m[k - N]

The host issues two dispatches in a single compute command encoder. Their serial ordering provides the implicit barrier so the second dispatch sees the first's atomic writes:
  Dispatch 1 (logup_count_mult): one thread per witness     row; atomically increments multiplicities[witness_idx[i]].
  Dispatch 2 (logup_partial_product): each threadgroup     of TG_WIDTH = 256 threads owns 256 consecutive indices     in [0, N+M). Each thread computes num_k * 1/(alpha -     x_k); threadgroup-cooperatively reduces the 256 terms     into one tile product written to partial[tgid]. Threads     with k >= N+M contribute the multiplicative identity     (1). The host then multiplies partial[0..K-1] (K =     ceil((N+M)/256)) on the CPU to obtain the final running     product (the sub-millisecond host fold is intentionally     untimed).

Field selection (constant prime_kind):
  0 = Goldilocks  p = 2^64 - 2^32 + 1
  1 = BabyBear    p = 2^31 - 2^27 + 1 = 2013265921
Both reductions are runtime-dispatched on prime_kind; a candidate that hardcodes the Goldilocks reduction macro, or assumes 64-bit limbs are needed, silently fails the held-out BabyBear probe.

All field elements (table, alpha, partial[]) are canonical uint64 in [0, p); a non-canonical output element is a correctness failure even if its residue class matches. Multiplicities are canonical uint32 counts (promoted to ulong only when used as the numerator).

## Required kernel signature(s)

```
kernel void logup_count_mult(
    device const uint  *witness_idx    [[buffer(0)]],
    device atomic_uint *multiplicities [[buffer(1)]],
    constant uint      &N              [[buffer(2)]],
    uint i [[thread_position_in_grid]]);

kernel void logup_partial_product(
    device const ulong *table          [[buffer(0)]],
    device const uint  *witness_idx    [[buffer(1)]],
    device const uint  *multiplicities [[buffer(2)]],
    device       ulong *partial        [[buffer(3)]],
    constant uint      &N              [[buffer(4)]],
    constant uint      &M              [[buffer(5)]],
    constant uint      &prime_kind     [[buffer(6)]],
    constant ulong     &alpha          [[buffer(7)]],
    uint gid  [[thread_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]);

Dispatch geometry (host-fixed):
  logup_count_mult:
    threadsPerGrid        = (N rounded up to TG width, 1, 1)
    threadsPerThreadgroup = (min(N, 256), 1, 1)
  logup_partial_product:
    threadsPerGrid        = (K * 256, 1, 1)   K = ceil((N+M)/256)
    threadsPerThreadgroup = (256, 1, 1)        // FIXED at TG_WIDTH=256

The 256-wide threadgroup is part of the host-kernel contract for logup_partial_product: K = ceil((N+M) / 256) is baked into the host-side partial[] allocation, so the kernel must emit exactly one tile-product per 256 consecutive indices. The host pre-zeroes multiplicities[M] before every dispatch. The two dispatches share a single MTLComputeCommandEncoder; the implicit cross-dispatch barrier in serial mode gives logup_partial_product a coherent view of multiplicities written by logup_count_mult.
```

## Your previous attempt

```metal
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;
constant ulong P_BB    = 2013265921ul;
constant ulong GOLD_INV_EXP = 0xFFFFFFFEFFFFFFFFul;
constant ulong BB_INV_EXP   = 2013265919ul;
constant uint  TG_WIDTH = 256u;
constant uint  SIMD_W   = 32u;
constant uint  NUM_SIMDS = TG_WIDTH / SIMD_W; // 8

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    if (t > a) t -= EPSILON;
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

// BabyBear: inputs in [0, P_BB), p < 2^31, so product < 2^62 — fits in ulong.
// Reduction via Barrett-like trick: q = (x * mu) >> 62, mu = floor(2^62 / p).
// But simple `% P_BB` on ulong compiles to integer divide; let's use it
// but avoid in tight loops. Since p ~ 2^31 we can do explicit reduction.
inline ulong bb_reduce(ulong x) {
    // x < 2^62; subtract multiples of P_BB.
    // mu = floor(2^62 / P_BB) ~ 2147483647 (since P_BB ~ 2^31)
    // Use: q = x / P_BB (hw divide), or shift-based estimate.
    // For correctness, simple modulo is fine; just inline.
    return x % P_BB;
}

inline ulong bb_sub(ulong a, ulong b) {
    return (a >= b) ? (a - b) : (a + P_BB - b);
}

inline ulong bb_mul(ulong a, ulong b) {
    return (a * b) % P_BB;
}

inline ulong mod_sub(ulong a, ulong b, uint prime_kind) {
    return (prime_kind == 0u) ? gold_sub(a, b) : bb_sub(a, b);
}

inline ulong mod_mul(ulong a, ulong b, uint prime_kind) {
    return (prime_kind == 0u) ? gold_mul(a, b) : bb_mul(a, b);
}

inline ulong mod_pow(ulong base, ulong exp, uint prime_kind) {
    ulong r = 1ul;
    while (exp != 0ul) {
        if ((exp & 1ul) != 0ul) {
            r = mod_mul(r, base, prime_kind);
        }
        base = mod_mul(base, base, prime_kind);
        exp >>= 1;
    }
    return r;
}

inline ulong mod_inv(ulong x, uint prime_kind) {
    ulong exp = (prime_kind == 0u) ? GOLD_INV_EXP : BB_INV_EXP;
    return mod_pow(x, exp, prime_kind);
}

// Simdgroup product reduction for ulong (split into two uint halves).
inline ulong simd_product(ulong v, uint prime_kind) {
    // Butterfly reduction across 32 lanes.
    for (uint offset = 16u; offset > 0u; offset >>= 1u) {
        uint lo = (uint)v;
        uint hi = (uint)(v >> 32);
        uint olo = simd_shuffle_xor(lo, offset);
        uint ohi = simd_shuffle_xor(hi, offset);
        ulong other = ((ulong)ohi << 32) | (ulong)olo;
        v = mod_mul(v, other, prime_kind);
    }
    return v;
}

kernel void logup_count_mult(
    device const uint  *witness_idx    [[buffer(0)]],
    device atomic_uint *multiplicities [[buffer(1)]],
    constant uint      &N              [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= N) return;
    uint j = witness_idx[i];
    atomic_fetch_add_explicit(&multiplicities[j], 1u, memory_order_relaxed);
}

kernel void logup_partial_product(
    device const ulong *table         [[buffer(0)]],
    device const uint  *witness_idx   [[buffer(1)]],
    device const uint  *multiplicities[[buffer(2)]],
    device       ulong *partial       [[buffer(3)]],
    constant uint      &N             [[buffer(4)]],
    constant uint      &M             [[buffer(5)]],
    constant uint      &prime_kind    [[buffer(6)]],
    constant ulong     &alpha         [[buffer(7)]],
    uint gid  [[thread_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    threadgroup ulong num_simd[NUM_SIMDS];
    threadgroup ulong den_simd[NUM_SIMDS];

    uint total = N + M;
    ulong num_term = 1ul;
    ulong den_term = 1ul;

    if (gid < total) {
        ulong x;
        ulong num;
        if (gid < N) {
            x   = table[witness_idx[gid]];
            num = 1ul;
        } else {
            x   = table[gid - N];
            num = (ulong)multiplicities[gid - N];
        }
        num_term = num;
        den_term = mod_sub(alpha, x, prime_kind);
    }

    // Stage 1: simdgroup-wide product reduction (no threadgroup memory).
    uint simd_lane = tid & 31u;
    uint simd_id   = tid >> 5;

    ulong num_simd_prod = simd_product(num_term, prime_kind);
    ulong den_simd_prod = simd_product(den_term, prime_kind);

    if (simd_lane == 0u) {
        num_simd[simd_id] = num_simd_prod;
        den_simd[simd_id] = den_simd_prod;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 2: final fold of 8 simd products — done by simd 0.
    if (simd_id == 0u) {
        ulong nv, dv;
        if (simd_lane < NUM_SIMDS) {
            nv = num_simd[simd_lane];
            dv = den_simd[simd_lane];
        } else {
            nv = 1ul;
            dv = 1ul;
        }
        // Reduce across first 8 lanes (others are 1).
        for (uint offset = 4u; offset > 0u; offset >>= 1u) {
            uint lo = (uint)nv;
            uint hi = (uint)(nv >> 32);
            uint olo = simd_shuffle_xor(lo, offset);
            uint ohi = simd_shuffle_xor(hi, offset);
            ulong onv = ((ulong)ohi << 32) | (ulong)olo;
            nv = mod_mul(nv, onv, prime_kind);

            uint dlo = (uint)dv;
            uint dhi = (uint)(dv >> 32);
            uint odlo = simd_shuffle_xor(dlo, offset);
            uint odhi = simd_shuffle_xor(dhi, offset);
            ulong odv = ((ulong)odhi << 32) | (ulong)odlo;
            dv = mod_mul(dv, odv, prime_kind);
        }
        if (simd_lane == 0u) {
            ulong result;
            if (nv == 0ul) {
                result = 0ul;
            } else {
                ulong inv_den = mod_inv(dv, prime_kind);
                result = mod_mul(nv, inv_den, prime_kind);
            }
            partial[tgid] = result;
        }
    }
}
```

Result of previous attempt:
          gold_M4K: correct, 0.93 ms, 0.1 Gmodmul/s (int64) (0.1% of 53 Gops/s (int64 mul, est))
         gold_M64K: correct, 2.22 ms, 0.4 Gmodmul/s (int64) (0.8% of 53 Gops/s (int64 mul, est))
          gold_M1M: correct, 34.72 ms, 0.5 Gmodmul/s (int64) (0.8% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0044

## Current best (incumbent)

```metal
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;
constant ulong P_BB    = 2013265921ul;
constant ulong GOLD_INV_EXP = 0xFFFFFFFEFFFFFFFFul;
constant ulong BB_INV_EXP   = 2013265919ul;
constant uint  TG_WIDTH = 256u;

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    if (t > a) t -= EPSILON;
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

inline ulong bb_sub(ulong a, ulong b) {
    return (a >= b) ? (a - b) : (a + P_BB - b);
}

inline ulong bb_mul(ulong a, ulong b) {
    return (a * b) % P_BB;
}

inline ulong mod_sub(ulong a, ulong b, uint prime_kind) {
    return (prime_kind == 0u) ? gold_sub(a, b) : bb_sub(a, b);
}

inline ulong mod_mul(ulong a, ulong b, uint prime_kind) {
    return (prime_kind == 0u) ? gold_mul(a, b) : bb_mul(a, b);
}

inline ulong mod_pow(ulong base, ulong exp, uint prime_kind) {
    ulong r = 1ul;
    while (exp != 0ul) {
        if ((exp & 1ul) != 0ul) {
            r = mod_mul(r, base, prime_kind);
        }
        base = mod_mul(base, base, prime_kind);
        exp >>= 1;
    }
    return r;
}

inline ulong mod_inv(ulong x, uint prime_kind) {
    ulong exp = (prime_kind == 0u) ? GOLD_INV_EXP : BB_INV_EXP;
    return mod_pow(x, exp, prime_kind);
}

kernel void logup_count_mult(
    device const uint  *witness_idx    [[buffer(0)]],
    device atomic_uint *multiplicities [[buffer(1)]],
    constant uint      &N              [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= N) return;
    uint j = witness_idx[i];
    atomic_fetch_add_explicit(&multiplicities[j], 1u, memory_order_relaxed);
}

kernel void logup_partial_product(
    device const ulong *table         [[buffer(0)]],
    device const uint  *witness_idx   [[buffer(1)]],
    device const uint  *multiplicities[[buffer(2)]],
    device       ulong *partial       [[buffer(3)]],
    constant uint      &N             [[buffer(4)]],
    constant uint      &M             [[buffer(5)]],
    constant uint      &prime_kind    [[buffer(6)]],
    constant ulong     &alpha         [[buffer(7)]],
    uint gid  [[thread_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    threadgroup ulong num_scratch[TG_WIDTH];
    threadgroup ulong den_scratch[TG_WIDTH];

    uint total = N + M;
    ulong num_term;
    ulong den_term;

    if (gid < total) {
        ulong x;
        ulong num;
        if (gid < N) {
            x   = table[witness_idx[gid]];
            num = 1ul;
        } else {
            x   = table[gid - N];
            num = (ulong)multiplicities[gid - N];
        }
        num_term = num;
        den_term = mod_sub(alpha, x, prime_kind);
    } else {
        num_term = 1ul;
        den_term = 1ul;
    }

    // Special case: if num == 0 (a multiplicity is 0), the term is 0,
    // and the whole tile product becomes 0. We must propagate that.
    // We'll detect this by computing num_product separately; if it's 0,
    // we skip inversion. But mod_inv(den_product) is well-defined since
    // alpha avoids table values, so denom != 0.

    num_scratch[tid] = num_term;
    den_scratch[tid] = den_term;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = TG_WIDTH >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            num_scratch[tid] = mod_mul(num_scratch[tid], num_scratch[tid + stride], prime_kind);
            den_scratch[tid] = mod_mul(den_scratch[tid], den_scratch[tid + stride], prime_kind);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        ulong num_prod = num_scratch[0];
        ulong den_prod = den_scratch[0];
        ulong result;
        if (num_prod == 0ul) {
            result = 0ul;
        } else {
            ulong inv_den = mod_inv(den_prod, prime_kind);
            result = mod_mul(num_prod, inv_den, prime_kind);
        }
        partial[tgid] = result;
    }
}
```

Incumbent result:
          gold_M4K: correct, 0.59 ms, 0.1 Gmodmul/s (int64) (0.2% of 53 Gops/s (int64 mul, est))
         gold_M64K: correct, 2.39 ms, 0.4 Gmodmul/s (int64) (0.8% of 53 Gops/s (int64 mul, est))
          gold_M1M: correct, 37.39 ms, 0.4 Gmodmul/s (int64) (0.8% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0049

## History

- iter  0: compile=OK | correct=True | score=0.001014074531666877
- iter  1: compile=OK | correct=True | score=0.004901798097017582
- iter  2: compile=OK | correct=True | score=0.004431301440059308

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
