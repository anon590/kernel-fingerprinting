**Optimization rationale:** The previous attempts only differ marginally per-thread; the main bottleneck for small N (2^14, 2^16) is kernel launch overhead and memory throughput, not arithmetic. I'll focus on tightening the hot path:

1. **Branchless modular ops** using `select` and `as_type` to avoid divergent predication, which on Apple GPUs can serialize within a SIMD group.
2. **Simplified reduction**: compute `t1 = (H0 << 32) - H0` directly (one shift + sub) instead of a 64×64 multiply.
3. **Skip canonical reduction on the s==0 path** — since `x < p` already from prior stage / canonical input, but to be safe keep one cheap conditional sub.
4. **Use `ushort` / `uint` arithmetic** for index math where safe to reduce register pressure.
5. **Hoist `half_N` and avoid recomputing** `tw_stride` per branch.

The arithmetic kernel is already near the memory roof at N=2^18 (92%); for smaller N the launch is dominant, so I keep the kernel body lean and branch-light.

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul; // 2^32 - 1

static inline ulong gold_canonical(ulong x) {
    return x - select((ulong)0, P_GOLD, x >= P_GOLD);
}

static inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    t += select((ulong)0, EPSILON, t < a);          // wrap correction
    t -= select((ulong)0, P_GOLD, t >= P_GOLD);     // canonicalize
    return t;
}

static inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    t -= select((ulong)0, EPSILON, t > a);          // borrow correction
    return t;
}

// Reduce (lo + 2^64 * hi) mod p = 2^64 - 2^32 + 1.
// 2^64 ≡ 2^32 - 1 (mod p), 2^96 ≡ -1 (mod p).
// Let hi = H1*2^32 + H0, then result ≡ lo + H0*(2^32 - 1) - H1.
static inline ulong gold_reduce128(ulong lo, ulong hi) {
    ulong H0 = hi & EPSILON;
    ulong H1 = hi >> 32;

    // t0 = lo - H1  (with borrow correction)
    ulong t0 = lo - H1;
    t0 -= select((ulong)0, EPSILON, t0 > lo);

    // t1 = H0 * (2^32 - 1) = (H0 << 32) - H0; H0 < 2^32 so no overflow.
    ulong t1 = (H0 << 32) - H0;

    // t2 = t0 + t1  (with carry correction)
    ulong t2 = t0 + t1;
    t2 += select((ulong)0, EPSILON, t2 < t0);

    // canonicalize
    t2 -= select((ulong)0, P_GOLD, t2 >= P_GOLD);
    return t2;
}

static inline ulong gold_mul(ulong a, ulong b) {
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

    uint r = k & mask;
    uint j = k >> s;

    uint i0 = (j << s) + r;          // j * 2^s + r
    uint i1 = i0 + half_N;
    uint o0 = (j << (s + 1u)) + r;   // j * 2^(s+1) + r
    uint o1 = o0 + span_half;

    ulong u = in_data[i0];
    ulong x = in_data[i1];

    ulong v;
    if (s == 0u) {
        // r == 0, twiddle is 1; canonicalize x.
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