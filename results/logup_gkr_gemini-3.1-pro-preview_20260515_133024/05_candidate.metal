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

// Lazy reduction multiplier; output is strictly bounded to [0, 2^64 - 1].
inline ulong gold_mul_lazy(ulong a, ulong b) {
    uint2 av = as_type<uint2>(a);
    uint2 bv = as_type<uint2>(b);
    
    ulong lo = a * b;
    
    ulong p01 = (ulong)av.x * bv.y;
    ulong p10 = (ulong)av.y * bv.x;
    ulong sum_p = p01 + p10;
    ulong carry = (sum_p < p01) ? 0x100000000ul : 0ul;
    
    ulong mid = sum_p + mulhi(av.x, bv.x);
    ulong hi = ((ulong)av.y * bv.y) + as_type<uint2>(mid).y + carry;
    
    uint2 hiv = as_type<uint2>(hi);
    
    ulong t0 = lo - hiv.y;
    t0 -= (t0 > lo) ? EPSILON : 0ul;
    
    ulong t1 = as_type<ulong>(uint2(0u, hiv.x)) - hiv.x;
    
    ulong t2 = t0 + t1;
    t2 += (t2 < t0) ? EPSILON : 0ul;
    
    return t2;
}

inline ulong gold_canon(ulong a) {
    return (a >= P_GOLD) ? (a - P_GOLD) : a;
}

inline ulong gold_pow(ulong base, ulong exp) {
    ulong r = 1ul;
    while (exp != 0ul) {
        if ((exp & 1ul) != 0ul) {
            r = gold_mul_lazy(r, base);
        }
        base = gold_mul_lazy(base, base);
        exp >>= 1;
    }
    return r; // Expected to be canonicalized by caller
}

// ---------------------- BabyBear helpers ------------------------------

inline ulong bb_sub(ulong a, ulong b) {
    ulong t = a + P_BB - b;
    return (t >= P_BB) ? (t - P_BB) : t;
}

inline ulong bb_mul(ulong a, ulong b) {
    ulong x = a * b;
    uint2 xv = as_type<uint2>(x);
    
    // Barrett reduction using M = 0x222222222u
    ulong p01 = (ulong)xv.x << 1;
    ulong p10 = (ulong)xv.y * 0x22222222u;
    ulong sum_p = p01 + p10;
    ulong carry = (sum_p < p01) ? 0x100000000ul : 0ul;
    
    ulong mid = sum_p + mulhi(xv.x, 0x22222222u);
    ulong q = ((ulong)xv.y << 1) + as_type<uint2>(mid).y + carry;
    
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

    threadgroup ulong sh_num[256];
    threadgroup ulong sh_denom[256];

    if (prime_kind == 0u) {
        sh_num[tid] = num;
        sh_denom[tid] = denom;
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Single warp early-exit reduction
        if (tid < 32) {
            ulong n = sh_num[tid];
            ulong d = sh_denom[tid];
            
            #pragma unroll
            for (int i = 1; i < 8; ++i) {
                n = gold_mul_lazy(n, sh_num[tid + i * 32]);
                d = gold_mul_lazy(d, sh_denom[tid + i * 32]);
            }
            
            #pragma unroll
            for (ushort offset = 16; offset > 0; offset /= 2) {
                n = gold_mul_lazy(n, simd_shuffle_down_ulong(n, offset));
                d = gold_mul_lazy(d, simd_shuffle_down_ulong(d, offset));
            }
            
            if (tid == 0) {
                ulong inv = gold_pow(d, GOLD_INV_EXP);
                partial[tgid] = gold_canon(gold_mul_lazy(n, inv));
            }
        }
    } else {
        sh_num[tid] = num;
        sh_denom[tid] = denom;
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tid < 32) {
            ulong n = sh_num[tid];
            ulong d = sh_denom[tid];
            
            #pragma unroll
            for (int i = 1; i < 8; ++i) {
                n = bb_mul(n, sh_num[tid + i * 32]);
                d = bb_mul(d, sh_denom[tid + i * 32]);
            }
            
            #pragma unroll
            for (ushort offset = 16; offset > 0; offset /= 2) {
                n = bb_mul(n, simd_shuffle_down_ulong(n, offset));
                d = bb_mul(d, simd_shuffle_down_ulong(d, offset));
            }
            
            if (tid == 0) {
                ulong inv = bb_pow(d, BB_INV_EXP);
                partial[tgid] = bb_mul(n, inv);
            }
        }
    }
}