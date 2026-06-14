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

constant ulong P_GOLD       = 0xFFFFFFFF00000001ul;
constant ulong EPSILON      = 0x00000000FFFFFFFFul;
constant ulong P_BB         = 2013265921ul;
constant ulong GOLD_INV_EXP = 0xFFFFFFFEFFFFFFFFul;
constant ulong BB_INV_EXP   = 2013265919ul;

// Helper for 64-bit SIMD shuffle
inline ulong simd_shuffle_down_ulong(ulong val, ushort delta) {
    uint2 v = as_type<uint2>(val);
    uint2 res;
    res.x = simd_shuffle_down(v.x, delta);
    res.y = simd_shuffle_down(v.y, delta);
    return as_type<ulong>(res);
}

// ---------------------- Goldilocks helpers ----------------------------

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    if (t > a) t -= EPSILON;
    return t;
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong lo = a * b; // directly leverages native 64-bit multiply
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);
    
    ulong p01 = (ulong)a0 * b1;
    ulong p10 = (ulong)a1 * b0;
    
    uint p00_hi = mulhi(a0, b0);
    ulong mid = (ulong)p00_hi + (uint)p01 + (uint)p10;
    ulong hi = ((ulong)a1 * b1) + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    
    ulong hi_lo = (uint)hi;
    ulong hi_hi = hi >> 32;
    ulong t0 = lo - hi_hi;
    if (t0 > lo) t0 -= EPSILON;
    ulong t1 = hi_lo * EPSILON;
    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;
    
    return gold_canonical(t2);
}

inline ulong gold_pow(ulong base, ulong exp) {
    ulong r = 1ul;
    while (exp != 0ul) {
        if ((exp & 1ul) != 0ul) {
            r = gold_mul(r, base);
        }
        base = gold_mul(base, base);
        exp >>= 1;
    }
    return r;
}

// ---------------------- BabyBear helpers ------------------------------

inline ulong bb_sub(ulong a, ulong b) {
    return (a >= b) ? (a - b) : (a + P_BB - b);
}

inline ulong bb_mul(ulong a, ulong b) {
    return (a * b) % P_BB;
}

inline ulong bb_pow(ulong base, ulong exp) {
    ulong r = 1ul;
    while (exp != 0ul) {
        if ((exp & 1ul) != 0ul) {
            r = bb_mul(r, base);
        }
        base = bb_mul(base, base);
        exp >>= 1;
    }
    return r;
}

// ----------------------------------------------------------------------
// Kernel A: Count Multiplicities
// ----------------------------------------------------------------------

kernel void logup_count_mult(
    device const uint  *witness_idx    [[buffer(0)]],
    device atomic_uint *multiplicities [[buffer(1)]],
    constant uint      &N              [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i < N) {
        uint j = witness_idx[i];
        atomic_fetch_add_explicit(&multiplicities[j], 1u, memory_order_relaxed);
    }
}

// ----------------------------------------------------------------------
// Kernel B: Partial Product (Deferred Inversion via Dual SIMD Reduction)
// ----------------------------------------------------------------------

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
    uint tgid [[threadgroup_position_in_grid]])
{
    uint total = N + M;
    ulong num = 1ul;
    ulong denom = 1ul;

    if (gid < total) {
        ulong x;
        if (gid < N) {
            x = table[witness_idx[gid]];
        } else {
            x = table[gid - N];
            num = (ulong)multiplicities[gid - N];
        }
        
        if (prime_kind == 0u) {
            denom = gold_sub(alpha, x);
        } else {
            denom = bb_sub(alpha, x);
        }
    }

    // 8 active SIMD lanes will export to these scratch arrays
    threadgroup ulong scratch_num[8];
    threadgroup ulong scratch_denom[8];

    // Split branches completely to avoid divergent branch execution costs
    if (prime_kind == 0u) {
        // SIMD intra-group reduction
        for (ushort offset = 16; offset > 0; offset /= 2) {
            num = gold_mul(num, simd_shuffle_down_ulong(num, offset));
            denom = gold_mul(denom, simd_shuffle_down_ulong(denom, offset));
        }
        
        if (tid % 32 == 0) {
            scratch_num[tid / 32] = num;
            scratch_denom[tid / 32] = denom;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Final reduction on threads 0..31 to reduce the 8 SIMD group results
        if (tid < 32) {
            num = (tid < 8) ? scratch_num[tid] : 1ul;
            denom = (tid < 8) ? scratch_denom[tid] : 1ul;
            
            for (ushort offset = 4; offset > 0; offset /= 2) {
                num = gold_mul(num, simd_shuffle_down_ulong(num, offset));
                denom = gold_mul(denom, simd_shuffle_down_ulong(denom, offset));
            }
            
            // Execute the single deferred inversion
            if (tid == 0) {
                ulong inv = gold_pow(denom, GOLD_INV_EXP);
                partial[tgid] = gold_mul(num, inv);
            }
        }
    } else {
        // SIMD intra-group reduction
        for (ushort offset = 16; offset > 0; offset /= 2) {
            num = bb_mul(num, simd_shuffle_down_ulong(num, offset));
            denom = bb_mul(denom, simd_shuffle_down_ulong(denom, offset));
        }
        
        if (tid % 32 == 0) {
            scratch_num[tid / 32] = num;
            scratch_denom[tid / 32] = denom;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tid < 32) {
            num = (tid < 8) ? scratch_num[tid] : 1ul;
            denom = (tid < 8) ? scratch_denom[tid] : 1ul;
            
            for (ushort offset = 4; offset > 0; offset /= 2) {
                num = bb_mul(num, simd_shuffle_down_ulong(num, offset));
                denom = bb_mul(denom, simd_shuffle_down_ulong(denom, offset));
            }
            
            if (tid == 0) {
                ulong inv = bb_pow(denom, BB_INV_EXP);
                partial[tgid] = bb_mul(num, inv);
            }
        }
    }
}
```

Result of previous attempt:
          gold_M4K: correct, 0.09 ms, 0.7 Gmodmul/s (int64) (1.3% of 53 Gops/s (int64 mul, est))
         gold_M64K: correct, 0.47 ms, 2.1 Gmodmul/s (int64) (3.9% of 53 Gops/s (int64 mul, est))
          gold_M1M: correct, 3.94 ms, 4.0 Gmodmul/s (int64) (7.5% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0335

## History

- iter  0: compile=OK | correct=True | score=0.0010692109512541036
- iter  1: compile=FAIL | correct=False | score=N/A
- iter  2: compile=OK | correct=True | score=0.03352834430210124

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
