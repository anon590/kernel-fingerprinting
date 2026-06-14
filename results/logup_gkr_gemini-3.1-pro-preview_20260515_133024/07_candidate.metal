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
    uint2 av = as_type<uint2>(a);
    uint2 bv = as_type<uint2>(b);
    
    ulong lo = a * b;
    
    ulong p01 = (ulong)av.x * bv.y;
    ulong p10 = (ulong)av.y * bv.x;
    ulong sum_p = p01 + p10;
    
    ulong mid = sum_p + mulhi(av.x, bv.x);
    ulong hi = ((ulong)av.y * bv.y) + as_type<uint2>(mid).y + ((sum_p < p01) ? 0x100000000ul : 0ul);
    
    uint2 hiv = as_type<uint2>(hi);
    
    ulong t0 = lo - hiv.y;
    t0 -= (t0 > lo) ? EPSILON : 0ul;
    
    ulong t1 = (ulong)hiv.x * EPSILON;
    
    ulong t2 = t0 + t1;
    t2 += (t2 < t0) ? EPSILON : 0ul;
    
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
    uint2 xv = as_type<uint2>(x);
    
    ulong sum_p = ((ulong)xv.x << 1) + (ulong)xv.y * 0x22222222u;
    ulong mid = sum_p + mulhi(xv.x, 0x22222222u);
    ulong q = ((ulong)xv.y << 1) + as_type<uint2>(mid).y;
    
    ulong r = x - q * P_BB;
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
// Kernel B: Partial Product (Deferred Inversion via Dual Reduction)
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

    threadgroup ulong2 sh[256];

    if (prime_kind == 0u) {
        sh[tid] = ulong2(num, denom);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tid < 128) {
            ulong2 other = sh[tid + 128];
            num = gold_mul(num, other.x);
            denom = gold_mul(denom, other.y);
            sh[tid] = ulong2(num, denom);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tid < 64) {
            ulong2 other = sh[tid + 64];
            num = gold_mul(num, other.x);
            denom = gold_mul(denom, other.y);
            sh[tid] = ulong2(num, denom);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tid < 32) {
            ulong2 other = sh[tid + 32];
            num = gold_mul(num, other.x);
            denom = gold_mul(denom, other.y);
            
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
        sh[tid] = ulong2(num, denom);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tid < 128) {
            ulong2 other = sh[tid + 128];
            num = bb_mul(num, other.x);
            denom = bb_mul(denom, other.y);
            sh[tid] = ulong2(num, denom);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tid < 64) {
            ulong2 other = sh[tid + 64];
            num = bb_mul(num, other.x);
            denom = bb_mul(denom, other.y);
            sh[tid] = ulong2(num, denom);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tid < 32) {
            ulong2 other = sh[tid + 32];
            num = bb_mul(num, other.x);
            denom = bb_mul(denom, other.y);
            
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