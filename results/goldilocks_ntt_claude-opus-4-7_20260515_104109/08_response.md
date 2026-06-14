I'll note the keybindings-help skill is available but not relevant to this MSL kernel optimization task.

**Optimization rationale:** The incumbent is bandwidth-bound at large N (86.6% at N=2^18). The main lever is reducing the cost of multiplication and reduction. I'll apply two micro-optimizations:

1. **Faster Goldilocks reduction** using the structure `x = lo + hi_lo*(2^32-1) - hi_hi`. I'll rewrite to avoid one conditional branch by combining the subtract and add fixups via signed arithmetic on the carry/borrow predicates using `select` returning predicates and merging adjustments.
2. **Replace conditional branches with branchless `select`** in `gold_add`/`gold_sub` to avoid divergence on Apple GPUs, where predicated execution is cheaper than branches in tight loops.
3. **Use `mulhi(uint,uint)` intrinsic** for the high 32 bits of the low partial product — this is a single hardware instruction on Apple GPUs vs. a full 64-bit multiply, reducing ALU pressure in `umul128`.
4. **Hoist constants and simplify indexing** so the compiler can schedule the memory loads earlier.

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;  // 2^64 - 2^32 + 1
constant ulong EPSILON = 0x00000000FFFFFFFFul;  // 2^32 - 1

inline ulong gold_canonical(ulong x) {
    return select(x, x - P_GOLD, x >= P_GOLD);
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    t = select(t, t + EPSILON, t < a);
    t = select(t, t - P_GOLD,  t >= P_GOLD);
    return t;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    t = select(t, t - EPSILON, t > a);
    return t;
}

// 64x64 -> 128 unsigned multiply. Uses mulhi(uint,uint) for the high
// half of the a0*b0 partial product, which is a single hardware op
// on Apple GPUs (vs. a full 64-bit multiply followed by a shift).
inline ulong2 umul128(ulong a, ulong b) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    uint  p00_lo = a0 * b0;
    uint  p00_hi = mulhi(a0, b0);
    ulong p01    = (ulong)a0 * (ulong)b1;
    ulong p10    = (ulong)a1 * (ulong)b0;
    ulong p11    = (ulong)a1 * (ulong)b1;

    ulong mid = (ulong)p00_hi + (p01 & EPSILON) + (p10 & EPSILON);
    ulong lo  = ((ulong)p00_lo) | (mid << 32);
    ulong hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return ulong2(lo, hi);
}

// Reduce (lo, hi) mod p using: 2^64 ≡ 2^32 - 1, 2^96 ≡ -1.
//   r = lo + hi_lo*(2^32 - 1) - hi_hi
// hi_lo < 2^32 so (hi_lo << 32) - hi_lo fits in u64.
inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong x_hi_lo = x_hi & EPSILON;
    ulong x_hi_hi = x_hi >> 32;

    ulong t0 = x_lo - x_hi_hi;
    t0 = select(t0, t0 - EPSILON, t0 > x_lo);

    ulong t1 = (x_hi_lo << 32) - x_hi_lo;   // hi_lo * (2^32 - 1)

    ulong t2 = t0 + t1;
    t2 = select(t2, t2 + EPSILON, t2 < t0);

    return gold_canonical(t2);
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong2 prod = umul128(a, b);
    return gold_reduce128(prod.x, prod.y);
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
    uint span_half = 1u << s;
    uint mask      = span_half - 1u;

    uint r  = k & mask;
    uint j  = k >> s;
    uint jh = j << s;

    uint i0 = jh + r;
    uint i1 = i0 + half_N;
    uint o0 = (jh << 1) + r;
    uint o1 = o0 + span_half;

    uint tw_shift = lN - 1u - s;
    ulong w = twiddles[(ulong)r << tw_shift];

    // Issue both global loads before the multiply to overlap latency.
    ulong x0 = in_data[i0];
    ulong x1 = in_data[i1];
    ulong v  = gold_mul(x1, w);

    out_data[o0] = gold_add(x0, v);
    out_data[o1] = gold_sub(x0, v);
}
```