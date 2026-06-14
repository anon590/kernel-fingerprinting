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

inline ulong gold_mul(ulong a, ulong b) {
    ulong lo = a * b;
    uint a0 = (uint)a, a1 = (uint)(a >> 32);
    uint b0 = (uint)b, b1 = (uint)(b >> 32);
    
    // 32x32 -> 64 multiplies avoiding MSL mulhi(ulong, ulong) limits
    ulong mid = (ulong)a0 * b1 + ((ulong)a0 * b0 >> 32);
    ulong hi = (ulong)a1 * b1 + (mid >> 32) + (((uint)mid + (ulong)a1 * b0) >> 32);
    
    uint hi_lo = (uint)hi;
    uint hi_hi = (uint)(hi >> 32);
    
    ulong t0 = lo - hi_hi;
    t0 -= (t0 > lo) ? EPSILON : 0ul;
    
    ulong t1 = ((ulong)hi_lo << 32) - hi_lo;
    
    ulong t2 = t0 + t1;
    t2 += (t2 < t0) ? EPSILON : 0ul;
    
    return t2 - ((t2 >= P_GOLD) ? P_GOLD : 0ul);
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    t -= (t > a) ? EPSILON : 0ul;
    return t;
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
    return t - ((t >= P_BB) ? P_BB : 0ul);
}

inline ulong bb_mul(ulong a, ulong b) {
    // Exact fitting up to 2^62, compiler maps constant modulo to mulhi magically
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

    uint lane_id = tid % 32;
    uint simd_id = tid / 32;

    if (prime_kind == 0u) {
        // SIMD-first fold avoids bank conflicts entirely
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

        if (lane_id == 0) {
            sh_num[simd_id] = num;
            sh_denom[simd_id] = denom;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (simd_id == 0) {
            num = (lane_id < 8) ? sh_num[lane_id] : 1ul;
            denom = (lane_id < 8) ? sh_denom[lane_id] : 1ul;
            
            num = gold_mul(num, simd_shuffle_down_ulong(num, 4));
            denom = gold_mul(denom, simd_shuffle_down_ulong(denom, 4));
            num = gold_mul(num, simd_shuffle_down_ulong(num, 2));
            denom = gold_mul(denom, simd_shuffle_down_ulong(denom, 2));
            num = gold_mul(num, simd_shuffle_down_ulong(num, 1));
            denom = gold_mul(denom, simd_shuffle_down_ulong(denom, 1));
            
            if (lane_id == 0) {
                ulong inv = gold_pow(denom, GOLD_INV_EXP);
                partial[tgid] = gold_mul(num, inv);
            }
        }
    } else {
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

        if (lane_id == 0) {
            sh_num[simd_id] = num;
            sh_denom[simd_id] = denom;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (simd_id == 0) {
            num = (lane_id < 8) ? sh_num[lane_id] : 1ul;
            denom = (lane_id < 8) ? sh_denom[lane_id] : 1ul;
            
            num = bb_mul(num, simd_shuffle_down_ulong(num, 4));
            denom = bb_mul(denom, simd_shuffle_down_ulong(denom, 4));
            num = bb_mul(num, simd_shuffle_down_ulong(num, 2));
            denom = bb_mul(denom, simd_shuffle_down_ulong(denom, 2));
            num = bb_mul(num, simd_shuffle_down_ulong(num, 1));
            denom = bb_mul(denom, simd_shuffle_down_ulong(denom, 1));
            
            if (lane_id == 0) {
                ulong inv = bb_pow(denom, BB_INV_EXP);
                partial[tgid] = bb_mul(num, inv);
            }
        }
    }
}