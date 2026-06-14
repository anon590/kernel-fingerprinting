This version keeps the incumbent’s faster 32×32 partial-product multiply, but replaces the reduction’s extra multiply-by-(2³²−1) with shift/sub/add folding. It also specializes the first two stages and uses SIMD-lane shuffles to share repeated early-stage twiddle loads within each 32-wide execution group, reducing redundant memory instructions without changing the one-butterfly-per-thread contract.

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD    = 0xFFFFFFFF00000001ul;  // 2^64 - 2^32 + 1
constant ulong EPSILON   = 0x00000000FFFFFFFFul;  // 2^32 - 1
constant ulong ROOT4_NEG = 0xFFFEFFFF00000001ul;  // p - 2^48

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    t += (t < a) ? EPSILON : 0ul;
    t -= (t >= P_GOLD) ? P_GOLD : 0ul;
    return t;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    t -= (t > a) ? EPSILON : 0ul;
    return t;
}

inline ulong gold_neg(ulong x) {
    return (x == 0ul) ? 0ul : (P_GOLD - x);
}

// Shuffle a 64-bit value through a uint2 shuffle, which is well-supported.
inline ulong simd_shuffle_u64(ulong x, ushort src_lane) {
    uint2 v = uint2((uint)x, (uint)(x >> 32));
    uint2 s = simd_shuffle(v, src_lane);
    return ((ulong)s.y << 32) | (ulong)s.x;
}

// Reduce lo + hi * 2^64 modulo p.
// Since 2^64 == 2^32 - 1 (mod p), with hi = h0 + h1*2^32:
//   hi*(2^32-1) == h0*2^32 - h0 - h1.
// This avoids the reduction multiply by EPSILON.
inline ulong gold_reduce128(ulong lo, ulong hi) {
    ulong h0 = hi & EPSILON;
    ulong h1 = hi >> 32;

    ulong sub = h0 + h1;
    ulong t = lo - sub;
    t -= (lo < sub) ? EPSILON : 0ul;

    ulong add = h0 << 32;
    ulong old = t;
    t += add;
    t += (t < old) ? EPSILON : 0ul;

    return gold_canonical(t);
}

inline ulong gold_mul(ulong a, ulong b) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    ulong p00 = (ulong)a0 * (ulong)b0;
    ulong p01 = (ulong)a0 * (ulong)b1;
    ulong p10 = (ulong)a1 * (ulong)b0;
    ulong p11 = (ulong)a1 * (ulong)b1;

    ulong mid = (p00 >> 32) + (p01 & EPSILON) + (p10 & EPSILON);
    ulong lo  = (p00 & EPSILON) | (mid << 32);
    ulong hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);

    return gold_reduce128(lo, hi);
}

// Multiply by +2^48 modulo Goldilocks.
inline ulong gold_mul_root4_pos(ulong x) {
    return gold_reduce128(x << 48, x >> 16);
}

kernel void goldilocks_ntt_stage(
    device const ulong *in_data    [[buffer(0)]],
    device       ulong *out_data   [[buffer(1)]],
    device const ulong *twiddles   [[buffer(2)]],
    constant uint      &stage_idx  [[buffer(3)]],
    constant uint      &log_N      [[buffer(4)]],
    uint k [[thread_position_in_grid]])
{
    uint lm1 = log_N - 1u;
    uint half_N = 1u << lm1;
    if (k >= half_N) return;

    uint s = stage_idx;

    ulong u = in_data[k];
    ulong x = in_data[k + half_N];

    // Stage 0: twiddle is 1 for every butterfly.
    if (s == 0u) {
        uint o0 = k << 1;
        out_data[o0]      = gold_add(u, x);
        out_data[o0 + 1u] = gold_sub(u, x);
        return;
    }

    // Stage 1: twiddles are {1, +/-2^48}; avoid full Goldilocks multiply.
    if (s == 1u) {
        uint r  = k & 1u;
        uint o0 = (k << 1) - r;

        ulong y = gold_mul_root4_pos(x);

        // Broadcast the single stage-1 nontrivial twiddle per SIMD group.
        uint lane = k & 31u;
        ulong w4_lane = 0ul;
        if (lane == 0u) {
            w4_lane = twiddles[half_N >> 1u];
        }
        ulong w4 = simd_shuffle_u64(w4_lane, (ushort)0);

        if (w4 == ROOT4_NEG) {
            y = gold_neg(y);
        }

        ulong v = (r == 0u) ? x : y;
        out_data[o0]      = gold_add(u, v);
        out_data[o0 + 2u] = gold_sub(u, v);
        return;
    }

    // Last stage has r == k, stride == 1, and output offsets equal input halves.
    if (s == lm1) {
        ulong v = gold_mul(x, twiddles[k]);
        out_data[k]          = gold_add(u, v);
        out_data[k + half_N] = gold_sub(u, v);
        return;
    }

    uint span_half = 1u << s;
    uint r = k & (span_half - 1u);
    uint o0 = (k << 1) - r;
    uint o1 = o0 + span_half;

    uint tw_shift = lm1 - s;
    ulong w;

    // For early stages, r repeats within a 32-lane SIMD group. Load only the
    // unique twiddles in low lanes and shuffle them to the matching lanes.
    if (s < 5u) {
        uint lane = k & 31u;
        ulong w_lane = 0ul;
        if (lane < span_half) {
            w_lane = twiddles[lane << tw_shift];
        }
        w = simd_shuffle_u64(w_lane, (ushort)r);
    } else {
        w = twiddles[r << tw_shift];
    }

    ulong v = gold_mul(x, w);

    out_data[o0] = gold_add(u, v);
    out_data[o1] = gold_sub(u, v);
}
```