I acknowledge the system reminder about the keybindings-help skill, but it's not relevant to this MSL kernel optimization task.

**Optimization rationale:** The seed uses a full 128-bit multiply + complex reduction. I'll apply several improvements:

1. **Faster Goldilocks reduction**: Use the structure `p = 2^64 - 2^32 + 1` more directly with a streamlined reduce that avoids redundant canonicalization branches.
2. **Lazy reduction in add/sub**: Since outputs feed back as inputs, I keep them canonical but minimize branches by using arithmetic on conditions.
3. **Skip multiply when twiddle = 1**: At stage 0, `r=0` always, so `w = tw[0] = 1`. Detecting this saves a full mul on stage 0 (which is half the butterflies on the smallest stage but, more importantly, when `r=0` in any stage the twiddle is 1).
4. **Strength-reduced index math**: Replace `j * span` with shift/OR using bit decomposition (`k = j*span_half + r`, so output index = `(j << (s+1)) | r`).
5. **Use `__builtin`-style carry via `select` / arithmetic** instead of branches where possible to reduce divergence.

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    ulong over = (t < a) ? EPSILON : 0ul;
    t += over;
    if (t >= P_GOLD) t -= P_GOLD;
    return t;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    ulong under = (t > a) ? EPSILON : 0ul;
    t -= under;
    return t;
}

// 64x64 -> 128 unsigned multiplication via 32x32 splits.
inline ulong2 umul128(ulong a, ulong b) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    ulong p00 = (ulong)a0 * (ulong)b0;
    ulong p01 = (ulong)a0 * (ulong)b1;
    ulong p10 = (ulong)a1 * (ulong)b0;
    ulong p11 = (ulong)a1 * (ulong)b1;

    ulong mid = (p00 >> 32) + (p01 & EPSILON) + (p10 & EPSILON);
    ulong lo  = (p00 & EPSILON) | (mid << 32);
    ulong hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return ulong2(lo, hi);
}

// Reduce 128-bit (lo, hi) mod p = 2^64 - 2^32 + 1.
// hi = hi_hi*2^32 + hi_lo
// 2^64 ≡ 2^32 - 1 (mod p), 2^96 ≡ -1 (mod p)
// so x = lo + hi_lo * (2^32 - 1) - hi_hi
inline ulong gold_reduce128(ulong lo, ulong hi) {
    ulong hi_lo = hi & EPSILON;
    ulong hi_hi = hi >> 32;

    // t0 = lo - hi_hi
    ulong t0 = lo - hi_hi;
    if (t0 > lo) t0 -= EPSILON;

    // t1 = hi_lo * (2^32 - 1) = (hi_lo << 32) - hi_lo  (fits in u64)
    ulong t1 = (hi_lo << 32) - hi_lo;

    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;

    if (t2 >= P_GOLD) t2 -= P_GOLD;
    return t2;
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong2 p = umul128(a, b);
    return gold_reduce128(p.x, p.y);
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
    uint mask      = span_half - 1u;

    uint r  = k & mask;
    uint j  = k >> s;

    uint i0 = (j << s) + r;
    uint i1 = i0 + half_N;

    uint o0 = (j << (s + 1u)) | r;
    uint o1 = o0 | span_half;

    uint tw_shift = log_N - 1u - s;
    ulong w = twiddles[(ulong)r << tw_shift];

    ulong u = in_data[i0];
    ulong x = in_data[i1];

    // Skip multiply when w == 1 (happens when r == 0)
    ulong v = (r == 0u) ? x : gold_mul(x, w);

    out_data[o0] = gold_add(u, v);
    out_data[o1] = gold_sub(u, v);
}
```