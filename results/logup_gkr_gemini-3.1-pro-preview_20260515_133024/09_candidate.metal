#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

constant ulong P_GOLD       = 0xFFFFFFFF00000001ul;
constant ulong EPSILON      = 0x00000000FFFFFFFFul;
constant ulong P_BB         = 2013265921ul;
constant ulong GOLD_INV_EXP = 0xFFFFFFFEFFFFFFFFul;
constant ulong BB_INV_EXP   = 2013265919ul;

inline ulong simd_shuffle_down_ulong(ulong val, ushort delta) {
    uint2 v = as_type<uint2>(val);
    uint2 res;
    res.x = simd_shuffle_down(v.x, delta);
    res.y = simd_shuffle_down(v.y, delta);
    return as_type<ulong>(res);
}

// ---------------------- Goldilocks helpers ----------------------------

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    if (t > a) t -= EPSILON;
    return t;
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong lo = a * b;
    uint a0 = (uint)a, a1 = (uint)(a >> 32);
    uint b0 = (uint)b, b1 = (uint)(b >> 32);
    
    ulong p01 = (ulong)a0 * b1;
    ulong p10 = (ulong)a1 * b0;
    ulong sum_p = p01 + p10;
    ulong carry = (sum_p < p01) ? 1ul : 0ul;
    
    ulong mid = sum_p + mulhi(a0, b0);
    ulong hi = ((ulong)a1 * b1) + (mid >> 32) + (carry << 32);
    
    uint hi_lo = (uint)hi;
    uint hi_hi = (uint)(hi >> 32);
    
    ulong t0 = lo - hi_hi;
    if (t0 > lo) t0 -= EPSILON;
    
    ulong t1 = ((ulong)hi_lo << 32) - hi_lo;
    
    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;
    
    return (t2 >= P_GOLD) ? (t2 - P_GOLD) : t2;
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
    ulong t = a + P_BB - b;
    return (t >= P_BB) ? (t - P_BB) : t;
}

inline ulong bb_mul(ulong a, ulong b) {
    ulong x = a * b;
    uint x_lo = (uint)x;
    uint x_hi = (uint)(x >> 32);
    
    // Barrett reduction using M = floor(2^64 / P_BB) = 0x222222222
    ulong p01 = (ulong)x_lo << 1;
    ulong p10 = (ulong)x_hi * 0x22222222u;
    ulong sum_p = p01 + p10;
    
    // sum_p max is well under 2^64, carry logic safely eliminated
    ulong mid = sum_p + mulhi(x_lo, 0x22222222u);
    uint q = (x_hi << 1) + (uint)(mid >> 32);
    
    ulong r = x - (ulong)q * P_BB;
    return (r >= P_BB) ? (r - P_BB) : r;
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
// Kernel B: Partial Product
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
    
    // Use ulong2 to coalesce shared memory loads/stores
    threadgroup ulong2 sh_nd[256];

    if (prime_kind == 0u) {
        if (gid < total) {
            ulong x;
            if (gid < N) {
                x = table[witness_idx[gid]];
            } else {
                x = table[gid - N];
                num = (ulong)multiplicities[gid - N];
            }
            denom = gold_sub(alpha, x);
        }
        
        sh_nd[tid] = ulong2(num, denom);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tid < 128) {
            ulong2 nd = sh_nd[tid + 128];
            num = gold_mul(num, nd.x);
            denom = gold_mul(denom, nd.y);
            sh_nd[tid] = ulong2(num, denom);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tid < 64) {
            ulong2 nd = sh_nd[tid + 64];
            num = gold_mul(num, nd.x);
            denom = gold_mul(denom, nd.y);
            sh_nd[tid] = ulong2(num, denom);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tid < 32) {
            ulong2 nd = sh_nd[tid + 32];
            num = gold_mul(num, nd.x);
            denom = gold_mul(denom, nd.y);
            
            num = gold_mul(num, simd_shuffle_down_ulong(num, 16));
            denom = gold_mul(denom, simd_shuffle_down_ulong(denom, 16));
            num = gold_mul(num, simd_shuffle_down_ulong(num, 8));
            denom = gold_mul(denom, simd_shuffle_down_ulong(denom, 8));
            num = gold_mul(num, simd_shuffle_down_ulong(num, 4));
            denom = gold_mul(denom, simd_shuffle_down_ulong(denom, 4));
            num = gold_mul(num, simd_shuffle_down_ulong(num, 2));
            denom = gold_mul(denom, simd_shuffle_down_ulong(denom, 2));
            num = gold_mul(num, simd_shuffle_down_ulong(num, 1));
            denom = gold_mul(denom, simd_shuffle_down_ulong(denom, 1));
            
            if (tid == 0) {
                ulong inv = gold_pow(denom, GOLD_INV_EXP);
                partial[tgid] = gold_mul(num, inv);
            }
        }
    } else {
        if (gid < total) {
            ulong x;
            if (gid < N) {
                x = table[witness_idx[gid]];
            } else {
                x = table[gid - N];
                num = (ulong)multiplicities[gid - N];
            }
            denom = bb_sub(alpha, x);
        }
        
        sh_nd[tid] = ulong2(num, denom);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tid < 128) {
            ulong2 nd = sh_nd[tid + 128];
            num = bb_mul(num, nd.x);
            denom = bb_mul(denom, nd.y);
            sh_nd[tid] = ulong2(num, denom);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tid < 64) {
            ulong2 nd = sh_nd[tid + 64];
            num = bb_mul(num, nd.x);
            denom = bb_mul(denom, nd.y);
            sh_nd[tid] = ulong2(num, denom);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tid < 32) {
            ulong2 nd = sh_nd[tid + 32];
            num = bb_mul(num, nd.x);
            denom = bb_mul(denom, nd.y);
            
            num = bb_mul(num, simd_shuffle_down_ulong(num, 16));
            denom = bb_mul(denom, simd_shuffle_down_ulong(denom, 16));
            num = bb_mul(num, simd_shuffle_down_ulong(num, 8));
            denom = bb_mul(denom, simd_shuffle_down_ulong(denom, 8));
            num = bb_mul(num, simd_shuffle_down_ulong(num, 4));
            denom = bb_mul(denom, simd_shuffle_down_ulong(denom, 4));
            num = bb_mul(num, simd_shuffle_down_ulong(num, 2));
            denom = bb_mul(denom, simd_shuffle_down_ulong(denom, 2));
            num = bb_mul(num, simd_shuffle_down_ulong(num, 1));
            denom = bb_mul(denom, simd_shuffle_down_ulong(denom, 1));
            
            if (tid == 0) {
                ulong inv = bb_pow(denom, BB_INV_EXP);
                partial[tgid] = bb_mul(num, inv);
            }
        }
    }
}