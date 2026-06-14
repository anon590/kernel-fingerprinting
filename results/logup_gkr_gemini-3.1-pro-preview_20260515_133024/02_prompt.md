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

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    if (t > a) t -= EPSILON;
    return t;
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong lo = a * b;
    uint x0 = (uint)a;
    uint x1 = (uint)(a >> 32);
    uint y0 = (uint)b;
    uint y1 = (uint)(b >> 32);
    
    ulong p01 = (ulong)x0 * y1;
    ulong p10 = (ulong)x1 * y0;
    
    uint p00_hi = mulhi(x0, y0);
    // Explicitly cast to ulong first to prevent 32-bit addition overflow
    ulong mid = (ulong)p00_hi + (uint)p01 + (uint)p10;
    ulong hi = ((ulong)x1 * y1) + (p01 >> 32) + (p10 >> 32) + (mid >> 32);

    ulong hi_lo = (uint)hi;
    ulong hi_hi = hi >> 32;
    ulong t0 = lo - hi_hi;
    if (t0 > lo) t0 -= EPSILON;
    ulong t1 = hi_lo * EPSILON;
    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;
    
    return gold_canonical(t2);
}

inline ulong bb_sub(ulong a, ulong b) {
    return (a >= b) ? (a - b) : (a + P_BB - b);
}

inline ulong bb_mul(ulong a, ulong b) {
    return (a * b) % P_BB;
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
            num = 1ul;
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

    threadgroup ulong scratch_num[8];
    threadgroup ulong scratch_denom[8];

    // Branching hoisted outside the loop for optimal performance
    if (prime_kind == 0u) {
        // SIMD 32 tree reduction
        for (uint offset = 16; offset > 0; offset /= 2) {
            num = gold_mul(num, simd_shuffle_down(num, offset));
            denom = gold_mul(denom, simd_shuffle_down(denom, offset));
        }
        
        if (tid % 32 == 0) {
            scratch_num[tid / 32] = num;
            scratch_denom[tid / 32] = denom;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tid == 0) {
            ulong total_num = scratch_num[0];
            ulong total_denom = scratch_denom[0];
            for (uint i = 1; i < 8; i++) {
                total_num = gold_mul(total_num, scratch_num[i]);
                total_denom = gold_mul(total_denom, scratch_denom[i]);
            }
            // Execute the single inversion and combine
            ulong inv = gold_pow(total_denom, GOLD_INV_EXP);
            partial[tgid] = gold_mul(total_num, inv);
        }
    } else {
        // SIMD 32 tree reduction for BabyBear
        for (uint offset = 16; offset > 0; offset /= 2) {
            num = bb_mul(num, simd_shuffle_down(num, offset));
            denom = bb_mul(denom, simd_shuffle_down(denom, offset));
        }
        
        if (tid % 32 == 0) {
            scratch_num[tid / 32] = num;
            scratch_denom[tid / 32] = denom;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tid == 0) {
            ulong total_num = scratch_num[0];
            ulong total_denom = scratch_denom[0];
            for (uint i = 1; i < 8; i++) {
                total_num = bb_mul(total_num, scratch_num[i]);
                total_denom = bb_mul(total_denom, scratch_denom[i]);
            }
            // Execute the single inversion and combine
            ulong inv = bb_pow(total_denom, BB_INV_EXP);
            partial[tgid] = bb_mul(total_num, inv);
        }
    }
}
```

Result of previous attempt:
  COMPILE FAILED: Error Domain=MTLLibraryErrorDomain Code=3 "program_source:134:33: error: no matching function for call to 'simd_shuffle_down'
            num = gold_mul(num, simd_shuffle_down(num, offset));
                                ^~~~~~~~~~~~~~~~~
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_simdgroup:239:14: note: candidate template ignored: requirement '__is_valid_simdgroup_type<unsigned long, void>::value' was not satisfied [with T = unsigned long]
METAL_FUNC T simd_shuffle_down(T data METAL_MAYBE_UNDEF, ushort delta)
             ^
program_source:135:37: error: no matching function for call to 'simd_shuffle_down'
            denom = gold_mul(denom, simd_shuffle_down(denom, offset));
                                    ^~~~~~~~~~~~~~~~~
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_simdgroup:239:14: note: candidate template ignored: requirement '__is_valid_simdgroup_type<unsigned long, void>::value' was not satisfied [with T = unsigned long]
METAL_FUNC T simd_shuffle_down(T data METAL_MAYBE_UNDEF, ushort delta)
             ^
program_source:159:31: error: no matching function for call to 'simd_shuffle_down'
            num = bb_mul(num, simd_shuffle_down(num, offset));
                              ^~~~~~~~~~~~~~~~~
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_simdgroup:239:14: note: candidate template ignored: requirement '__is_valid_simdgroup_type<unsigned long, void>::value' was not satisfied [with T = unsigned long]
METAL_FUNC T simd_shuffle_down(T data METAL_MAYBE_UNDEF, ushort delta)
             ^
program_source:160:35: error: no matching function for call to 'simd_shuffle_down'
            denom = bb_mul(denom, simd_shuffle_down(denom, offset));
                                  ^~~~~~~~~~~~~~~~~
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_simdgroup:239:14: note: candidate template ignored: requirement '__is_valid_simdgroup_type<unsigned long, void>::value' was not satisfied [with T = unsigned long]
METAL_FUNC T simd_shuffle_down(T data METAL_MAYBE_UNDEF, ushort delta)
             ^
" UserInfo={NSLocalizedDescription=program_source:134:33: error: no matching function for call to 'simd_shuffle_down'
            num = gold_mul(num, simd_shuffle_down(num, offset));
                                ^~~~~~~~~~~~~~~~~
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_simdgroup:239:14: note: candidate template ignored: requirement '__is_valid_simdgroup_type<unsigned long, void>::value' was not satisfied [with T = unsigned long]
METAL_FUNC T simd_shuffle_down(T data METAL_MAYBE_UNDEF, ushort delta)
             ^
program_source:135:37: error: no matching function for call to 'simd_shuffle_down'
            denom = gold_mul(denom, simd_shuffle_down(denom, offset));
                                    ^~~~~~~~~~~~~~~~~
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_simdgroup:239:14: note: candidate template ignored: requirement '__is_valid_simdgroup_type<unsigned long, void>::value' was not satisfied [with T = unsigned long]
METAL_FUNC T simd_shuffle_down(T data METAL_MAYBE_UNDEF, ushort delta)
             ^
program_source:159:31: error: no matching function for call to 'simd_shuffle_down'
            num = bb_mul(num, simd_shuffle_down(num, offset));
                              ^~~~~~~~~~~~~~~~~
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_simdgroup:239:14: note: candidate template ignored: requirement '__is_valid_simdgroup_type<unsigned long, void>::value' was not satisfied [with T = unsigned long]
METAL_FUNC T simd_shuffle_down(T data METAL_MAYBE_UNDEF, ushort delta)
             ^
program_source:160:35: error: no matching function for call to 'simd_shuffle_down'
            denom = bb_mul(denom, simd_shuffle_down(denom, offset));
                                  ^~~~~~~~~~~~~~~~~
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_simdgroup:239:14: note: candidate template ignored: requirement '__is_valid_simdgroup_type<unsigned long, void>::value' was not satisfied [with T = unsigned long]
METAL_FUNC T simd_shuffle_down(T data METAL_MAYBE_UNDEF, ushort delta)
             ^
}

## Current best (incumbent)

```metal
// Naive seed for the LogUp lookup-argument running product (Z7).
//
// Computes:
//   (1) multiplicities[j] = #{ i : witness_idx[i] == j }   for j in [0, M)
//   (2) the running product
//
//       P = prod_{i=0..N-1} 1/(alpha - w_i)
//          * prod_{j=0..M-1} m_j / (alpha - T_j)            (mod p)
//
//       where w_i := T[witness_idx[i]].
//
// Combined fingerprint stream of length N + M:
//   index k < N:   x_k = T[witness_idx[k]],   numerator num_k = 1
//   index k >= N:  x_k = T[k - N],            numerator num_k = m[k - N]
//
// The host issues TWO dispatches in this order, within ONE compute
// command encoder (so kernel B sees kernel A's atomic writes):
//
//   1) logup_count_mult
//        threadsPerGrid       = (N, 1, 1)         rounded up to TG width
//        threadsPerThreadgroup= (min(N, 256), 1, 1)
//        One thread per witness row; atomically increments
//        multiplicities[witness_idx[i]] by 1.
//
//   2) logup_partial_product
//        threadsPerGrid       = (ceil((N+M)/256) * 256, 1, 1)
//        threadsPerThreadgroup= (TG_WIDTH = 256, 1, 1)            // FIXED
//        Each threadgroup owns 256 consecutive indices in [0, N+M).
//        Each thread computes num_k * 1/(alpha - x_k) for its index
//        (or 1 for k >= N+M, the multiplicative identity), then the
//        threadgroup tree-reduces 256 terms into one tile product
//        written to partial[tgid].
//
// The host then reads partial[0..K-1] (K = ceil((N+M)/256)) and
// multiplies them on the CPU to obtain the final running product.
// (This sub-millisecond host-side fold is intentionally not timed.)
//
// Field selection (constant prime_kind):
//   0 = Goldilocks   p = 2^64 - 2^32 + 1
//   1 = BabyBear     p = 2^31 - 2^27 + 1 = 2013265921
// Both reductions are runtime-dispatched on prime_kind; a candidate
// that hardcodes the Goldilocks reduction silently fails the held-out
// BabyBear probe.
//
// Buffer layout (host-fixed, must be preserved by candidate kernels):
//
//   logup_count_mult:
//     buffer 0: device const uint  *witness_idx   (length N)
//     buffer 1: device atomic_uint *multiplicities(length M, zero-initialized by host)
//     buffer 2: constant uint &N
//
//   logup_partial_product:
//     buffer 0: device const ulong *table         (length M)
//     buffer 1: device const uint  *witness_idx   (length N)
//     buffer 2: device const uint  *multiplicities(length M)
//     buffer 3: device       ulong *partial       (length K = ceil((N+M)/256))
//     buffer 4: constant uint &N
//     buffer 5: constant uint &M
//     buffer 6: constant uint &prime_kind         (0 = Goldilocks, 1 = BabyBear)
//     buffer 7: constant ulong &alpha             (canonical, < p)
//
// All field elements (table, alpha, partial[]) are canonical uint64 in
// [0, p); a non-canonical output element is treated as a correctness
// failure even if its residue class matches the reference.

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;       // 2^64 - 2^32 + 1
constant ulong EPSILON = 0x00000000FFFFFFFFul;       // 2^32 - 1
constant ulong P_BB    = 2013265921ul;               // 2^31 - 2^27 + 1
constant ulong GOLD_INV_EXP = 0xFFFFFFFEFFFFFFFFul;  // P_GOLD - 2
constant ulong BB_INV_EXP   = 2013265919ul;          // P_BB - 2

constant uint  TG_WIDTH = 256u;

// ---------------------- Goldilocks helpers ----------------------------

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    if (t > a) t -= EPSILON;       // borrow -> fix with -= (2^32 - 1) mod p
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

// ---------------------- BabyBear helpers ------------------------------
// All BabyBear elements fit in 31 bits, so a*b fits in 62 bits and the
// % operator is well-defined on uint64.

inline ulong bb_sub(ulong a, ulong b) {
    return (a >= b) ? (a - b) : (a + P_BB - b);
}

inline ulong bb_mul(ulong a, ulong b) {
    return (a * b) % P_BB;
}

// ---------------------- Generic dispatch ------------------------------

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

// ----------------------------------------------------------------------
// Kernel A: count multiplicities (atomic add).
// ----------------------------------------------------------------------

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

// ----------------------------------------------------------------------
// Kernel B: per-tile fingerprint inversion + threadgroup product
// reduction.
//
// Per-thread: term = num_k * inv(alpha - x_k) when k < N + M, else 1.
// Per-threadgroup: log2(TG_WIDTH)-step in-place tree reduction over
// scratch[]; thread 0 writes the tile product to partial[tgid].
// ----------------------------------------------------------------------

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
    threadgroup ulong scratch[TG_WIDTH];

    uint total = N + M;
    ulong term;
    if (gid < total) {
        ulong x, num;
        if (gid < N) {
            x   = table[witness_idx[gid]];
            num = 1ul;
        } else {
            x   = table[gid - N];
            num = (ulong)multiplicities[gid - N];
        }
        ulong denom = mod_sub(alpha, x, prime_kind);
        ulong inv   = mod_inv(denom, prime_kind);
        term = mod_mul(num, inv, prime_kind);
    } else {
        term = 1ul;                  // multiplicative identity for OOB threads
    }

    scratch[tid] = term;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Power-of-two tree reduction (TG_WIDTH = 256 = 2^8).
    for (uint stride = TG_WIDTH >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            scratch[tid] = mod_mul(scratch[tid], scratch[tid + stride], prime_kind);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        partial[tgid] = scratch[0];
    }
}
```

Incumbent result:
          gold_M4K: correct, 1.19 ms, 0.1 Gmodmul/s (int64) (0.1% of 53 Gops/s (int64 mul, est))
         gold_M64K: correct, 16.42 ms, 0.1 Gmodmul/s (int64) (0.1% of 53 Gops/s (int64 mul, est))
          gold_M1M: correct, 261.69 ms, 0.1 Gmodmul/s (int64) (0.1% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0011

## History

- iter  0: compile=OK | correct=True | score=0.0010692109512541036
- iter  1: compile=FAIL | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
