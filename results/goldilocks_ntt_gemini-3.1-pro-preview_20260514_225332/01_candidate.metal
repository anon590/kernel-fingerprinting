#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD   = 0xFFFFFFFF00000001ul;  // 2^64 - 2^32 + 1
constant ulong EPSILON  = 0x00000000FFFFFFFFul;  // 2^32 - 1

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    ulong c = (t < a) ? EPSILON : 0;
    t += c;
    return (t >= P_GOLD) ? (t - P_GOLD) : t;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    ulong c = (t > a) ? EPSILON : 0;
    return t - c;
}

inline ulong gold_mul(ulong a, ulong b) {
    uint a0 = (uint)(a);
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)(b);
    uint b1 = (uint)(b >> 32);

    ulong p00 = (ulong)a0 * b0;
    ulong p01 = (ulong)a0 * b1;
    ulong p10 = (ulong)a1 * b0;
    ulong p11 = (ulong)a1 * b1;

    // Optimized 128-bit product using non-overflowing mid accumulator
    ulong mid = p01 + (p10 & EPSILON) + (p00 >> 32);
    ulong lo  = (p00 & EPSILON) | (mid << 32);
    ulong hi  = p11 + (p10 >> 32) + (mid >> 32);

    // gold_reduce128 inlined and made branchless
    ulong x_hi_hi = hi >> 32;

    ulong t0 = lo - x_hi_hi;
    ulong b_borrow = (t0 > lo) ? EPSILON : 0;
    t0 -= b_borrow;

    ulong t1 = ((hi & EPSILON) << 32) - (hi & EPSILON);

    ulong t2 = t0 + t1;
    ulong c_carry = (t2 < t0) ? EPSILON : 0;
    t2 += c_carry;

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
    uint half_N = 1u << (log_N - 1u);
    if (k >= half_N) return;

    uint s = stage_idx;
    uint r = k & ((1u << s) - 1u);
    uint j = k >> s;

    // i0 simplifies algebraically to exactly k across all stages
    ulong u = in_data[k];
    ulong v = in_data[k + half_N];

    // Branch perfectly skips multiplication for all threads uniformly in stage 0
    if (r != 0) {
        ulong w = twiddles[r << (log_N - 1u - s)];
        v = gold_mul(v, w);
    }

    // Write offsets match bit-reversed Cooley-Tukey pattern
    uint o0 = (j << (s + 1u)) + r;
    uint o1 = o0 + (1u << s);

    out_data[o0] = gold_add(u, v);
    out_data[o1] = gold_sub(u, v);
}