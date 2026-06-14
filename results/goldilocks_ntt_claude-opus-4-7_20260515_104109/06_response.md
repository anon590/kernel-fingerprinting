I notice the system-reminder about the keybindings-help skill, but it's not relevant to this Metal kernel optimization task, so I'll proceed with the user's request.

**Optimization rationale:** The previous attempts suffered from too many branches in the modular arithmetic (`if` statements compile to predicated execution and serialize threads). I'll replace all branchy fixups with `select()` calls which Metal compiles to conditional-select instructions with no divergence cost. I'll also use Metal's native `mulhi(uint, uint)` for the high half of 32x32 products (avoiding a redundant 64-bit multiply for `p00`), use `bitselect`-style canonical reduction, and precompute the twiddle index with a single shift. The hot path becomes mostly straight-line ALU which lets the GPU hide memory latency better — particularly important for the small-N cases (N=2^14, 2^16) where kernel launch + ALU dominates over bandwidth.

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
    t = select(t, t + EPSILON, t < a);          // u64 overflow fixup
    t = select(t, t - P_GOLD,  t >= P_GOLD);    // canonicalize
    return t;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    t = select(t, t - EPSILON, t > a);          // u64 underflow fixup
    return t;
}

// 64x64 -> 128 multiply: returns (lo, hi).
// Uses Metal's native mulhi(uint,uint) for the high-half of p00 and
// combines via 32-bit partial products.
inline ulong2 umul128(ulong a, ulong b) {
    uint a0 = (uint)(a);
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)(b);
    uint b1 = (uint)(b >> 32);

    uint  p00_lo = a0 * b0;
    uint  p00_hi = mulhi(a0, b0);
    ulong p01    = (ulong)a0 * (ulong)b1;
    ulong p10    = (ulong)a1 * (ulong)b0;
    ulong p11    = (ulong)a1 * (ulong)b1;

    // mid sums fit in 34 bits
    ulong mid = (ulong)p00_hi + (p01 & EPSILON) + (p10 & EPSILON);
    ulong lo  = ((ulong)p00_lo) | (mid << 32);
    ulong hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return ulong2(lo, hi);
}

// Reduce 128-bit (x_lo, x_hi) modulo p = 2^64 - 2^32 + 1 using
// 2^64 ≡ 2^32 - 1, 2^96 ≡ -1 (mod p).
inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong x_hi_lo = x_hi & EPSILON;
    ulong x_hi_hi = x_hi >> 32;

    // t0 = x_lo - x_hi_hi  (subtract 2^96 component)
    ulong t0 = x_lo - x_hi_hi;
    t0 = select(t0, t0 - EPSILON, t0 > x_lo);

    // t1 = x_hi_lo * (2^32 - 1)  (factors < 2^32, fits in u64)
    ulong t1 = (x_hi_lo << 32) - x_hi_lo;

    // t2 = t0 + t1
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
    uint span_half = 1u << s;            // 2^s
    uint mask      = span_half - 1u;

    uint r  = k & mask;
    uint j  = k >> s;
    uint jh = j << s;                    // j * 2^s

    uint i0 = jh + r;
    uint i1 = i0 + half_N;
    uint o0 = (jh << 1) + r;             // j * 2^(s+1) + r
    uint o1 = o0 + span_half;

    uint tw_shift = lN - 1u - s;
    ulong w = twiddles[(ulong)r << tw_shift];

    ulong v_in = in_data[i1];
    ulong u    = in_data[i0];
    ulong v    = gold_mul(v_in, w);

    out_data[o0] = gold_add(u, v);
    out_data[o1] = gold_sub(u, v);
}
```