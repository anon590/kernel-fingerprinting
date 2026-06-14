#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;  // 2^64 - 2^32 + 1
constant ulong EPSILON = 0x00000000FFFFFFFFul;  // 2^32 - 1

inline ulong gold_canonical(ulong x) {
    return x >= P_GOLD ? x - P_GOLD : x;
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    // overflow: add EPSILON; then canonicalize
    ulong c1 = (t < a) ? EPSILON : 0ul;
    t += c1;
    // The added EPSILON could itself overflow if t was near 2^64-1; that
    // can only happen when a+b overflowed, after which t < a originally,
    // so t = (a+b)-2^64 < 2^32-1, and t + EPSILON < 2^33 — no second overflow.
    if (t >= P_GOLD) t -= P_GOLD;
    return t;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    if (t > a) t -= EPSILON;
    return t;
}

// Goldilocks reduction of 128-bit value (lo, hi).
// x = lo + hi * 2^64, with 2^64 ≡ 2^32 - 1 (mod p), 2^96 ≡ -1 (mod p).
// So x ≡ lo + (hi_lo)*(2^32 - 1) - hi_hi   (mod p).
inline ulong gold_reduce128(ulong lo, ulong hi) {
    ulong hi_lo = hi & EPSILON;        // low 32
    ulong hi_hi = hi >> 32;             // high 32

    // t0 = lo - hi_hi  (mod p)
    ulong t0 = lo - hi_hi;
    if (t0 > lo) t0 -= EPSILON;

    // t1 = hi_lo * (2^32 - 1) = (hi_lo << 32) - hi_lo, no overflow since hi_lo < 2^32.
    ulong t1 = (hi_lo << 32) - hi_lo;

    // t2 = t0 + t1
    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;

    return gold_canonical(t2);
}

// Native 64x64->128 using Metal's mulhi for ulong (compiles to one hi-mul).
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
    uint span_half = 1u << s;          // 2^s
    uint mask      = span_half - 1u;

    uint j = k >> s;
    uint r = k & mask;

    uint i0 = (j << s) + r;
    uint i1 = i0 + half_N;
    uint o0 = (j << (s + 1u)) + r;
    uint o1 = o0 + span_half;

    // Issue both loads up front so the compiler can overlap latency.
    ulong u = in_data[i0];
    ulong x = in_data[i1];

    ulong v;
    if (s == 0u) {
        // r = 0, twiddle = omega_2^0 = 1; just canonicalize x.
        v = (x >= P_GOLD) ? (x - P_GOLD) : x;
    } else {
        uint tw_idx = r << (lN - 1u - s);
        ulong w     = twiddles[tw_idx];
        v = gold_mul(x, w);
    }

    out_data[o0] = gold_add(u, v);
    out_data[o1] = gold_sub(u, v);
}