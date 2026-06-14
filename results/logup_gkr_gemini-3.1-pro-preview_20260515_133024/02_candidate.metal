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