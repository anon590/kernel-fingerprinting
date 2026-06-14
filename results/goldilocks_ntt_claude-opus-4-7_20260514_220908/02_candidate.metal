#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    ulong c1 = (t < a) ? EPSILON : 0ul;
    t += c1;
    ulong t2 = t - P_GOLD;
    return (t >= P_GOLD) ? t2 : t;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    ulong c = (b > a) ? EPSILON : 0ul;
    return t - c;
}

// Reduce 128-bit (hi:lo) modulo p = 2^64 - 2^32 + 1.
inline ulong gold_reduce128(ulong lo, ulong hi) {
    uint  hi_hi = (uint)(hi >> 32);
    uint  hi_lo = (uint)(hi);

    // t0 = lo - hi_hi (subtract; if borrow, add EPSILON)
    ulong t0 = lo - (ulong)hi_hi;
    if (lo < (ulong)hi_hi) t0 -= EPSILON;

    // t1 = hi_lo * (2^32 - 1) = (hi_lo << 32) - hi_lo
    ulong hi_lo64 = (ulong)hi_lo;
    ulong t1 = (hi_lo64 << 32) - hi_lo64;

    // t2 = t0 + t1 mod p
    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;
    if (t2 >= P_GOLD) t2 -= P_GOLD;
    return t2;
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong lo = a * b;
    ulong hi = mulhi(a, b);
    return gold_reduce128(lo, hi);
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

    uint s         = stage_idx;
    uint span_half = 1u << s;

    uint mask = span_half - 1u;
    uint j    = k >> s;
    uint r    = k & mask;

    uint i0 = (j << (s + 1u)) + r;        // j * span + r? No — read uses span_half
    // Per spec: read offsets are (j * 2^s + r) and (j * 2^s + r + N/2)
    //           write offsets are (j * 2^(s+1) + r) and same + 2^s
    uint read0  = (j << s) + r;
    uint read1  = read0 + half_N;
    uint write0 = (j << (s + 1u)) + r;
    uint write1 = write0 + span_half;

    // Issue both loads early to hide latency.
    ulong u = in_data[read0];
    ulong x = in_data[read1];

    ulong v;
    if (s == 0u) {
        // r is always 0 at stage 0, twiddle = 1
        v = x;
    } else {
        uint tw_stride = 1u << (log_N - 1u - s);
        ulong w = twiddles[r * tw_stride];
        v = gold_mul(x, w);
    }

    out_data[write0] = gold_add(u, v);
    out_data[write1] = gold_sub(u, v);

    (void)i0;
}