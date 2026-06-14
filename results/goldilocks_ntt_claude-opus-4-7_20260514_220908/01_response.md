## Optimization

The seed uses a slow 4-way 32x32 split for the 64x64→128 multiply and computes too many shifts per butterfly. I'm making two key improvements:

1. **Faster mul-hi**: Use Metal's `mulhi(ulong, ulong)` builtin (which Apple GPUs lower more efficiently than manual 32x32 splits), with `a*b` for the low half. This shrinks the hot path significantly.
2. **Reduce branchy fixups**: Use `select`/arithmetic for canonical wrapping where cheap, and use `metal::select` to avoid serializing control flow.
3. **Specialize stage 0**: When `s == 0`, `r = 0` so the twiddle is always 1, eliminating a multiply and a load. Detect at runtime with a tiny branch (uniform per warp).

This should help especially at smaller N where multiply cost dominates over bandwidth.

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    if (t < a) t += EPSILON;
    if (t >= P_GOLD) t -= P_GOLD;
    return t;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    if (t > a) t -= EPSILON;
    return t;
}

// 128-bit reduction mod Goldilocks p = 2^64 - 2^32 + 1.
inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong x_hi_lo = x_hi & EPSILON;
    ulong x_hi_hi = x_hi >> 32;

    ulong t0 = x_lo - x_hi_hi;
    if (t0 > x_lo) t0 -= EPSILON;

    ulong t1 = x_hi_lo * EPSILON;        // both < 2^32, fits in u64

    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;

    return gold_canonical(t2);
}

inline ulong gold_mul(ulong a, ulong b) {
    // Use Metal builtins: a*b gives low 64 bits, mulhi gives high 64.
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
    uint span      = span_half << 1u;

    uint mask = span_half - 1u;
    uint j    = k >> s;
    uint r    = k & mask;

    uint i0 = j * span_half + r;
    uint i1 = i0 + half_N;
    uint o0 = j * span + r;
    uint o1 = o0 + span_half;

    ulong u = in_data[i0];
    ulong x = in_data[i1];

    ulong v;
    if (s == 0u) {
        // r == 0 -> twiddle is omega^0 == 1
        v = gold_canonical(x);
    } else {
        uint tw_stride = 1u << (log_N - 1u - s);
        ulong w = twiddles[r * tw_stride];
        v = gold_mul(x, w);
    }

    out_data[o0] = gold_add(u, v);
    out_data[o1] = gold_sub(u, v);
}
```