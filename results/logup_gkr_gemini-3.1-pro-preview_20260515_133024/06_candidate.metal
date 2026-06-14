#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

constant ulong P_GOLD       = 0xFFFFFFFF00000001ul;
constant ulong EPSILON      = 0x00000000FFFFFFFFul;
constant ulong P_BB         = 2013265921ul;
constant ulong GOLD_INV_EXP = 0xFFFFFFFEFFFFFFFFul;
constant ulong BB_INV_EXP   = 2013265919ul;

inline ulong simd_shuffle_down_ulong(ulong val, ushort delta) {
    return as_type<ulong>(simd_shuffle_down(as_type<uint2>(val), delta));
}

// ---------------------- Goldilocks helpers ----------------------------

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    if (t > a) t -= EPSILON;
    return t;
}

// Lazy multiplier: safely handles and returns values bounded by [0, 2^64 - 1].
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
    if (t0 > lo) t0 -= EPSILON;
    
    ulong t1 = as_type<ulong>(uint2(0u, hiv.x)) - hiv.x;
    
    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;
    
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
    
    // Bounds tracking guarantees cross-term sum_p never overflows 64-bit unsigned.
    ulong p01 = (ulong)xv.x << 1;
    ulong p10 = (ulong)xv.y * 0x22222222u;
    ulong sum_p = p01 + p10;
    
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

    threadgroup ulong sh_num[8];
    threadgroup ulong sh_denom[8];

    uint warp_id = tid >> 5;
    uint lane_id = tid & 31;

    if (prime_kind == 0u) {
        #pragma unroll
        for (ushort offset = 16; offset > 0; offset >>= 1) {
            num = gold_mul_lazy(num, simd_shuffle_down_ulong(num, offset));
            denom = gold_mul_lazy(denom, simd_shuffle_down_ulong(denom, offset));
        }
        
        if (lane_id == 0) {
            sh_num[warp_id] = num;
            sh_denom[warp_id] = denom;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (warp_id == 0) {
            num = (lane_id < 8) ? sh_num[lane_id] : 1ul;
            denom = (lane_id < 8) ? sh_denom[lane_id] : 1ul;
            
            #pragma unroll
            for (ushort offset = 4; offset > 0; offset >>= 1) {
                num = gold_mul_lazy(num, simd_shuffle_down_ulong(num, offset));
                denom = gold_mul_lazy(denom, simd_shuffle_down_ulong(denom, offset));
            }
            
            if (lane_id == 0) {
                ulong inv = gold_pow(denom, GOLD_INV_EXP);
                partial[tgid] = gold_canon(gold_mul_lazy(num, inv));
            }
        }
    } else {
        #pragma unroll
        for (ushort offset = 16; offset > 0; offset >>= 1) {
            num = bb_mul(num, simd_shuffle_down_ulong(num, offset));
            denom = bb_mul(denom, simd_shuffle_down_ulong(denom, offset));
        }
        
        if (lane_id == 0) {
            sh_num[warp_id] = num;
            sh_denom[warp_id] = denom;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (warp_id == 0) {
            num = (lane_id < 8) ? sh_num[lane_id] : 1ul;
            denom = (lane_id < 8) ? sh_denom[lane_id] : 1ul;
            
            #pragma unroll
            for (ushort offset = 4; offset > 0; offset >>= 1) {
                num = bb_mul(num, simd_shuffle_down_ulong(num, offset));
                denom = bb_mul(denom, simd_shuffle_down_ulong(denom, offset));
            }
            
            if (lane_id == 0) {
                ulong inv = bb_pow(denom, BB_INV_EXP);
                partial[tgid] = bb_mul(num, inv);
            }
        }
    }
}