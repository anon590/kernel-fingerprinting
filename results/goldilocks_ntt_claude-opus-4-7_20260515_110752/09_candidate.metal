#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;  // 2^64 - 2^32 + 1
constant ulong EPSILON = 0x00000000FFFFFFFFul;  // 2^32 - 1

inline ulong gold_canonical(ulong x) {
    return select(x, x - P_GOLD, x >= P_GOLD);
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t  = a + b;
    ulong c1 = select(0ul, EPSILON, t < a);    // u64 overflow
    t += c1;
    return select(t, t - P_GOLD, t >= P_GOLD);
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t  = a - b;
    ulong c1 = select(0ul, EPSILON, t > a);    // u64 underflow
    return t - c1;
}

// Goldilocks reduction of 128-bit (lo, hi):
//   x ≡ lo + (hi_lo) * (2^32 - 1) - hi_hi   (mod p)
inline ulong gold_reduce128(ulong lo, ulong hi) {
    ulong hi_lo = hi & EPSILON;       // low 32
    ulong hi_hi = hi >> 32;            // high 32

    // t0 = lo - hi_hi (mod p)
    ulong t0 = lo - hi_hi;
    t0 -= select(0ul, EPSILON, t0 > lo);   // borrow fixup

    // t1 = hi_lo * (2^32 - 1) = (hi_lo << 32) - hi_lo, fits in u64.
    ulong t1 = (hi_lo << 32) - hi_lo;

    // t2 = t0 + t1
    ulong t2 = t0 + t1;
    t2 += select(0ul, EPSILON, t2 < t0);   // carry fixup

    return select(t2, t2 - P_GOLD, t2 >= P_GOLD);
}

// 64x64 -> 128 via Metal's native ulong mulhi (one hi-mul instruction).
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
    uint lN     = log_N;
    uint half_N = 1u << (lN - 1u);
    if (k >= half_N) return;

    uint s         = stage_idx;
    uint span_half = 1u << s;            // 2^s
    uint mask      = span_half - 1u;

    uint j = k >> s;
    uint r = k & mask;

    uint i0 = (j << s) + r;
    uint i1 = i0 + half_N;
    uint o0 = (j << (s + 1u)) + r;
    uint o1 = o0 + span_half;

    // Issue both loads up front to overlap memory latency.
    ulong u = in_data[i0];
    ulong x = in_data[i1];

    ulong v;
    if (s == 0u) {
        // r = 0 always at stage 0, twiddle = omega_2^0 = 1.
        v = gold_canonical(x);
    } else {
        uint tw_idx = r << (lN - 1u - s);
        ulong w     = twiddles[tw_idx];
        v = gold_mul(x, w);
    }

    out_data[o0] = gold_add(u, v);
    out_data[o1] = gold_sub(u, v);
}