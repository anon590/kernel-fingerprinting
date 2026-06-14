I'll acknowledge the system reminder about the keybindings-help skill, but it's not relevant to this MSL kernel optimization task.

**Optimization strategy:** The incumbent does well at N=2^18 (92% BW) but poorly at smaller sizes. The bottleneck at small N is dispatch/launch overhead and not fully saturating memory. I'll focus on micro-optimizations within the kernel: (1) use a tighter Goldilocks reduction via `mad`-style 128-bit reduction that exploits that hi < 2^64, reformulated to reduce dependency chain, (2) eliminate the canonical reduction on the multiplier output path since add/sub handle inputs up to 2p anyway (but final outputs must be canonical — so we keep the canonicalization on writes), (3) use `as_type<uint2>` decomposition which compiles to free register renames rather than mask/shift, (4) avoid branching on `s == 0` by always loading from twiddle buffer with stride such that index 0 is read (twiddles[0]=1 is guaranteed by host convention — but we can't assume that, so keep the branch but make it cheap). Actually, safer to just always do the multiply; the cost of one extra mul on stage 0 is small and removes divergence.

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
    uint2 hi_parts = as_type<uint2>(x_hi);
    ulong x_hi_lo = (ulong)hi_parts.x;
    ulong x_hi_hi = (ulong)hi_parts.y;

    ulong t0 = x_lo - x_hi_hi;
    if (t0 > x_lo) t0 -= EPSILON;

    ulong t1 = (x_hi_lo << 32) - x_hi_lo;

    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;

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
    if (s == 0u) {
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