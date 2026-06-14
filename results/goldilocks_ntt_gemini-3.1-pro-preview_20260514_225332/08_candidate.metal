#include <metal_stdlib>
using namespace metal;

#define P_GOLD  0xFFFFFFFF00000001ul
#define EPSILON 0x00000000FFFFFFFFul

// 64x64 -> 128 unsigned multiplication with purely 32-bit splits
inline ulong gold_mul(ulong a, ulong b) {
    uint a0 = (uint)a, a1 = (uint)(a >> 32);
    uint b0 = (uint)b, b1 = (uint)(b >> 32);

    // Standard 32x32 -> 64 hardware multiplication
    ulong p00 = (ulong)a0 * b0;
    ulong p01 = (ulong)a0 * b1;
    ulong p10 = (ulong)a1 * b0;
    ulong p11 = (ulong)a1 * b1;

    // Fast tracking of carries via 32-bit truncation (free register sub-selects)
    ulong mid = (p00 >> 32) + (uint)p01 + (uint)p10;
    ulong x_lo = (uint)p00 | (mid << 32);
    ulong x_hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);

    // Branchless Goldilocks reduction
    ulong x_hi_lo = (uint)x_hi;
    ulong x_hi_hi = x_hi >> 32;

    ulong t0 = x_lo - x_hi_hi;
    t0 -= (t0 > x_lo) ? EPSILON : 0;

    // Equivalent to x_hi_lo * EPSILON but executed in fast shift/sub arithmetic
    ulong t1 = (x_hi_lo << 32) - x_hi_lo;

    ulong t2 = t0 + t1;
    t2 += (t2 < t0) ? EPSILON : 0;

    return (t2 >= P_GOLD) ? (t2 - P_GOLD) : t2;
}

kernel void goldilocks_ntt_stage(
    device const ulong *in_data    [[buffer(0)]],
    device       ulong *out_data   [[buffer(1)]],
    device const ulong *twiddles   [[buffer(2)]],
    constant uint      &stage_idx  [[buffer(3)]],
    constant uint      &log_N      [[buffer(4)]],
    uint k [[thread_position_in_grid]],
    uint k_local [[thread_position_in_threadgroup]])
{
    // Guarantee 16-byte alignment of threadgroup memory for safe ulong2 casting
    threadgroup ulong2 shared_out_v[256];
    threadgroup ulong* shared_out = (threadgroup ulong*)shared_out_v;

    uint half_N = 1u << (log_N - 1u);
    bool active = k < half_N;

    uint s = stage_idx;
    uint mask = (1u << s) - 1u;
    uint not_mask = ~mask;

    ulong out0 = 0;
    ulong out1 = 0;
    uint o0 = 0;
    uint o1 = 0;

    if (active) {
        // Read offsets are consistently contiguous across all stages
        ulong u = in_data[k];
        ulong v = in_data[k + half_N];

        if (s != 0u) {
            uint tw_idx = (k & mask) << (log_N - 1u - s);
            v = gold_mul(v, twiddles[tw_idx]);
        }

        o0 = ((k & not_mask) << 1u) | (k & mask);
        o1 = o0 | (1u << s);

        out0 = u + v;
        out0 += (out0 < u) ? EPSILON : 0;
        out0 = (out0 >= P_GOLD) ? (out0 - P_GOLD) : out0;

        out1 = u - v;
        out1 -= (out1 > u) ? EPSILON : 0;
    }

    if (s == 0u) {
        if (active) {
            // Stage 0 is perfectly interleaved, 100% naturally coalesced ulong2 writes
            ((device ulong2*)out_data)[k] = ulong2(out0, out1);
        }
    } else if (s <= 4u) {
        // Stages 1-4 are fragmented; cooperate to gather contiguous vector blocks
        if (active) {
            uint local_o0 = ((k_local & not_mask) << 1u) | (k_local & mask);
            uint local_o1 = local_o0 | (1u << s);
            shared_out[local_o0] = out0;
            shared_out[local_o1] = out1;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (active) {
            uint k_base = k - k_local;
            ((device ulong2*)(out_data + (k_base << 1u)))[k_local] = shared_out_v[k_local];
        }
    } else {
        if (active) {
            // For s >= 5, structural sizes >= 32 naturally grant perfectly contiguous chunks per warp
            out_data[o0] = out0;
            out_data[o1] = out1;
        }
    }
}