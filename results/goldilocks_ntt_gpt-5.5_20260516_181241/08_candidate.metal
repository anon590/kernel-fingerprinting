#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;  // 2^64 - 2^32 + 1
constant ulong EPSILON = 0x00000000FFFFFFFFul;  // 2^32 - 1

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    t += (t < a) ? EPSILON : 0ul;
    return (t >= P_GOLD) ? (t - P_GOLD) : t;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    t -= (t > a) ? EPSILON : 0ul;
    return t;
}

// 64x64 -> Goldilocks using explicit 32-bit partial products.
// This avoids depending on generic ulong multiply lowering for the high half.
inline ulong gold_mul(ulong a, ulong b) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    uint p00_lo = a0 * b0;
    uint p00_hi = mulhi(a0, b0);

    uint p01_lo = a0 * b1;
    uint p01_hi = mulhi(a0, b1);

    uint p10_lo = a1 * b0;
    uint p10_hi = mulhi(a1, b0);

    uint p11_lo = a1 * b1;
    uint p11_hi = mulhi(a1, b1);

    // Assemble 128-bit product limbs z0 + z1*2^32 + z2*2^64 + z3*2^96.
    uint s1 = p00_hi + p01_lo;
    uint c1 = (s1 < p00_hi) ? 1u : 0u;

    uint z1 = s1 + p10_lo;
    c1 += (z1 < s1) ? 1u : 0u;

    uint s2 = p01_hi + p10_hi;
    uint c2 = (s2 < p01_hi) ? 1u : 0u;

    uint s3 = s2 + p11_lo;
    c2 += (s3 < s2) ? 1u : 0u;

    uint z2 = s3 + c1;
    c2 += (z2 < s3) ? 1u : 0u;

    uint z3 = p11_hi + c2;

    // Goldilocks reduction:
    // 2^64 == 2^32 - 1, 2^96 == -1 mod p.
    // product == x_lo + z2*(2^32 - 1) - z3.
    ulong x_lo = ((ulong)z1 << 32) | (ulong)p00_lo;

    ulong sub = (ulong)z2 + (ulong)z3;
    ulong t = x_lo - sub;
    t -= (t > x_lo) ? EPSILON : 0ul;

    ulong add = ((ulong)z2) << 32;
    ulong r = t + add;
    r += (r < t) ? EPSILON : 0ul;

    return (r >= P_GOLD) ? (r - P_GOLD) : r;
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

    uint s = stage_idx;
    uint m = 1u << s;
    uint r = k & (m - 1u);

    // Contract gives i0 = (k >> s) * 2^s + r, which is exactly k.
    ulong u = in_data[k];
    ulong x = in_data[k + half_N];

    uint tw_stride = half_N >> s;
    ulong w = twiddles[r * tw_stride];
    ulong v = gold_mul(x, w);

    // o0 = (k >> s) * 2^(s+1) + r = 2*k - r.
    uint o0 = (k << 1u) - r;

    out_data[o0]     = gold_add(u, v);
    out_data[o0 + m] = gold_sub(u, v);
}