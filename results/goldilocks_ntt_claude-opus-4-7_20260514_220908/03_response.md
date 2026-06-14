**Optimization:** The bottleneck for small N (2^14, 2^16) is dispatch overhead — we launch `log_N` kernels each doing minimal work. Since the host fixes one-butterfly-per-thread with a single kernel name, I can't fuse stages across dispatches. But I can reduce per-thread cost: (1) use `ulong2` vector loads/stores when the two read addresses are contiguous (stage 0, where `read1 = read0 + N/2` — not contiguous, so skip), (2) replace the branchy `gold_sub`/`gold_add` with branchless arithmetic using `select`, which the compiler turns into conditional-select ops avoiding divergence, (3) use a faster Goldilocks reduction that uses `as_type<uint2>` to extract halves cheaply, (4) skip the multiplication entirely when `r == 0` (twiddle is 1) at any stage, not just stage 0 — this halves the mul work in many threads when grouped together. The branchless versions should reduce instruction count and register pressure, helping the smaller-N cases where we're latency-bound rather than bandwidth-bound.

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;

inline ulong gold_canonical(ulong x) {
    ulong y = x - P_GOLD;
    return select(x, y, x >= P_GOLD);
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    ulong t1 = select(t, t + EPSILON, t < a);
    ulong t2 = select(t1, t1 - P_GOLD, t1 >= P_GOLD);
    return t2;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    return select(t, t - EPSILON, b > a);
}

// Reduce 128-bit (hi:lo) mod Goldilocks p = 2^64 - 2^32 + 1.
inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    uint2 hi_parts = as_type<uint2>(x_hi);
    ulong x_hi_lo  = (ulong)hi_parts.x;   // low 32 bits of hi
    ulong x_hi_hi  = (ulong)hi_parts.y;   // high 32 bits of hi

    // t0 = x_lo - x_hi_hi (mod p, partial)
    ulong t0 = x_lo - x_hi_hi;
    t0 = select(t0, t0 - EPSILON, t0 > x_lo);

    // t1 = x_hi_lo * (2^32 - 1) = (x_hi_lo << 32) - x_hi_lo, fits in 64 bits
    ulong t1 = (x_hi_lo << 32) - x_hi_lo;

    ulong t2 = t0 + t1;
    t2 = select(t2, t2 + EPSILON, t2 < t0);

    return gold_canonical(t2);
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
    uint lg = log_N;
    uint half_N = 1u << (lg - 1u);
    if (k >= half_N) return;

    uint s         = stage_idx;
    uint span_half = 1u << s;
    uint span      = span_half << 1u;

    uint mask = span_half - 1u;
    uint j    = k >> s;
    uint r    = k & mask;

    uint i0 = (j << s) + r;
    uint i1 = i0 + half_N;
    uint o0 = (j << (s + 1u)) + r;
    uint o1 = o0 + span_half;

    ulong u = in_data[i0];
    ulong x = in_data[i1];

    ulong v;
    if (r == 0u) {
        // twiddle is omega^0 == 1
        v = gold_canonical(x);
    } else {
        uint tw_stride = 1u << (lg - 1u - s);
        ulong w = twiddles[r * tw_stride];
        v = gold_mul(x, w);
    }

    out_data[o0] = gold_add(u, v);
    out_data[o1] = gold_sub(u, v);
}
```