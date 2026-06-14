#include <metal_stdlib>
using namespace metal;

#define P_GOLD  0xFFFFFFFF00000001ul
#define EPSILON 0x00000000FFFFFFFFul

// 64x64 -> 128 unsigned multiplication with purely 32-bit splits
inline __attribute__((always_inline)) ulong gold_mul(ulong a, ulong b) {
    uint a0 = (uint)a, a1 = (uint)(a >> 32);
    uint b0 = (uint)b, b1 = (uint)(b >> 32);

    // Standard 32x32 -> 64 hardware multiplication
    ulong p00 = (ulong)a0 * b0;
    ulong p01 = (ulong)a0 * b1;
    ulong p10 = (ulong)a1 * b0;
    ulong p11 = (ulong)a1 * b1;

    // Fast tracking of carries via 32-bit truncation
    ulong mid = (p00 >> 32) + (uint)p01 + (uint)p10;
    ulong x_lo = (uint)p00 | (mid << 32);
    ulong x_hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);

    // Branchless Goldilocks reduction
    ulong x_hi_lo = (uint)x_hi;
    ulong x_hi_hi = x_hi >> 32;

    ulong t0 = x_lo - x_hi_hi;
    t0 -= (t0 > x_lo) ? EPSILON : 0;

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
    uint k [[thread_position_in_grid]])
{
    uint ln = log_N;
    uint half_N = 1u << (ln - 1u);
    if (k >= half_N) return;

    uint s = stage_idx;

    // Uniformly coalesced 100% linear reads spanning all stages
    ulong u = in_data[k];
    ulong v = in_data[k + half_N];

    // Stage 0: pure addition, naturally outputs adjacent index pairs
    if (s == 0u) {
        ulong out0 = u + v;
        out0 += (out0 < u) ? EPSILON : 0;
        ulong out0_red = (out0 >= P_GOLD) ? (out0 - P_GOLD) : out0;

        ulong out1 = u - v;
        out1 -= (out1 > u) ? EPSILON : 0;

        // Perfectly sequential 128-bit store without alignment fragmentation
        ((device ulong2*)out_data)[k] = ulong2(out0_red, out1);
        return;
    }

    // Stage 1: perfectly interleaved across adjacent threads (stride 2)
    if (s == 1u) {
        uint tw_idx = (k & 1u) << (ln - 2u);
        v = gold_mul(v, twiddles[tw_idx]);

        ulong out0 = u + v;
        out0 += (out0 < u) ? EPSILON : 0;
        ulong out0_red = (out0 >= P_GOLD) ? (out0 - P_GOLD) : out0;

        ulong out1 = u - v;
        out1 -= (out1 > u) ? EPSILON : 0;

        // Exchange butterfly components between lane i and lane i^1 using valid MSL uint2 bitcasts
        ulong neighbor_out0 = as_type<ulong>(simd_shuffle_xor(as_type<uint2>(out0_red), 1u));
        ulong neighbor_out1 = as_type<ulong>(simd_shuffle_xor(as_type<uint2>(out1), 1u));

        if ((k & 1u) == 0u) {
            // Even threads execute 128-bit store for the first clustered pair
            ((device ulong2*)out_data)[k] = ulong2(out0_red, neighbor_out0);
        } else {
            // Odd threads execute 128-bit store for the offset clustered pair
            ((device ulong2*)out_data)[k] = ulong2(neighbor_out1, out1);
        }
        return;
    }

    // General case (Stages >= 2): hardware cache progressively handles coalescing natively
    uint mask = (1u << s) - 1u;
    uint not_mask = ~mask;

    uint tw_idx = (k & mask) << (ln - 1u - s);
    v = gold_mul(v, twiddles[tw_idx]);

    ulong out0 = u + v;
    out0 += (out0 < u) ? EPSILON : 0;
    out0 = (out0 >= P_GOLD) ? (out0 - P_GOLD) : out0;

    ulong out1 = u - v;
    out1 -= (out1 > u) ? EPSILON : 0;

    uint o0 = ((k & not_mask) << 1u) | (k & mask);
    uint o1 = o0 | (1u << s);

    out_data[o0] = out0;
    out_data[o1] = out1;
}