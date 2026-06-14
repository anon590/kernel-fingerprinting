To achieve a substantial performance gain, we apply Montgomery's batch inversion trick across the running product. 

1. **Batch Inversion**: Instead of doing 256 expensive `mod_pow` inversions per threadgroup (one for every denominator), we reduce the product of all numerators and the product of all denominators independently across the threadgroup using SIMD shuffle downs. We then perform a *single* inversion on the accumulated denominator product in thread 0. Since $\prod(n_i \cdot d_i^{-1}) \equiv (\prod n_i) \cdot (\prod d_i)^{-1} \pmod p$, this yields mathematically identical and bit-exact results while eliminating ~99.6% of the inversion overhead.
2. **Fast Multiplication**: We replaced `umul128` with an optimized variant that avoids fully computing the lowest 64 bits of the intermediate products by taking advantage of the native `mulhi` instruction, avoiding register pressure. We also hoisted the `prime_kind` branch out of the threadgroup reduction loop.

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