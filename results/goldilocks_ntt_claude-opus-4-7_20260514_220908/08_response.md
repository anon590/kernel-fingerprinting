I'll note the skill reminder isn't relevant here — this is a Metal kernel task.

**Optimization idea:** The incumbent is already near peak at N=2^18 (92%), so the bottleneck for smaller N is dispatch/launch overhead and per-thread work. The biggest win I can squeeze without restructuring dispatch is to (1) tighten the modular reduction with branchless `select`s (better than ternary on Metal, fewer divergence-induced stalls), (2) skip the full 128-bit reduction in stage 0 entirely (twiddle = 1, so just canonical-reduce), and (3) use a fused identity in `gold_reduce128` that avoids one carry-correction step by directly computing `x_hi_lo * EPSILON = (x_hi_lo << 32) - x_hi_lo` (cheaper than a 32×32 multiply on Apple GPU's integer unit, since shift+sub is two ALU ops vs a multiply latency). Also marking helpers with explicit branchless arithmetic so the compiler emits predicated ALU rather than control flow.

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul; // 2^32 - 1

inline ulong gold_canonical(ulong x) {
    return x - select((ulong)0, P_GOLD, x >= P_GOLD);
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    t += select((ulong)0, EPSILON, t < a);
    return t - select((ulong)0, P_GOLD, t >= P_GOLD);
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    return t - select((ulong)0, EPSILON, t > a);
}

// Reduce (lo + 2^64*hi) mod p. Identity: 2^64 ≡ 2^32-1, 2^96 ≡ -1.
// Split hi = H1*2^32 + H0; result ≡ lo + H0*(2^32-1) - H1.
inline ulong gold_reduce128(ulong lo, ulong hi) {
    ulong H0 = hi & EPSILON;           // low 32 bits
    ulong H1 = hi >> 32;               // high 32 bits

    // t0 = lo - H1  (borrow => -EPSILON)
    ulong t0 = lo - H1;
    t0 -= select((ulong)0, EPSILON, t0 > lo);

    // H0 * (2^32 - 1) = (H0 << 32) - H0; H0 < 2^32 so fits in u64.
    ulong t1 = (H0 << 32) - H0;

    ulong t2 = t0 + t1;
    t2 += select((ulong)0, EPSILON, t2 < t0);

    return t2 - select((ulong)0, P_GOLD, t2 >= P_GOLD);
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
    uint lg     = log_N;
    uint half_N = 1u << (lg - 1u);
    if (k >= half_N) return;

    uint s         = stage_idx;
    uint span_half = 1u << s;
    uint mask      = span_half - 1u;

    uint r  = k & mask;
    uint j  = k >> s;

    uint i0 = (j << s) + r;
    uint i1 = i0 + half_N;
    uint o0 = (j << (s + 1u)) + r;
    uint o1 = o0 + span_half;

    ulong u = in_data[i0];
    ulong x = in_data[i1];

    ulong v;
    if (s == 0u) {
        // r == 0 => twiddle is 1
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