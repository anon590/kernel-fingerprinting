I fuse the two cross-SIMD stages (`len=64` and `len=32`) after the first butterfly. Each thread materializes the stage-0 low/high halves once in threadgroup memory, then recomputes the two needed stage-1 intermediates locally to produce its stage-2 pair. This removes one full threadgroup barrier and one scratch exchange; for these tiny 256-point NTTs, avoiding synchronization should outweigh the modest extra modular arithmetic.

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_MAX = 256u;

inline uint mod_reduce_generic(ulong t, uint q) {
    return (uint)(t % (ulong)q);
}

inline uint mod_mul_generic(uint a, uint b, uint q) {
    return mod_reduce_generic((ulong)a * (ulong)b, q);
}

inline uint mod_add_generic(uint a, uint b, uint q) {
    uint s = a + b;
    return ((s < a) || (s >= q)) ? (s - q) : s;
}

inline uint mod_sub_generic(uint a, uint b, uint q) {
    uint d = a - b;
    return (a >= b) ? d : (d + q);
}

// Exact for canonical inputs modulo 3329.
// For v < 3329^2, qhat=(v*315)>>20 is exact or one too large.
inline uint mod_mul_3329(uint a, uint b) {
    uint v    = a * b;
    uint qhat = (v * 315u) >> 20;
    uint r    = v - qhat * 3329u;
    return (r >= 3329u) ? (r + 3329u) : r;
}

inline uint mod_add_3329(uint a, uint b) {
    uint s = a + b;
    uint r = s - 3329u;
    return (s >= 3329u) ? r : s;
}

inline uint mod_sub_3329(uint a, uint b) {
    uint d = a - b;
    return (a >= b) ? d : (d + 3329u);
}

inline void bfly_tg_generic(threadgroup uint *s, uint j, uint len, uint zeta, uint q) {
    uint x = s[j];
    uint y = s[j + len];
    uint t = mod_mul_generic(zeta, y, q);
    s[j]       = mod_add_generic(x, t, q);
    s[j + len] = mod_sub_generic(x, t, q);
}

inline void finish_kyber_3329_from_stage2(
    device       uint *poly,
    device const uint *zetas,
    uint ltid,
    uint s2lo,
    uint s2hi)
{
    // Stages 3..6 are wholly inside one SIMD group.
    uint lane  = ltid & 31u;
    uint r     = lane & 15u;
    uint chunk = ltid >> 4u;          // 0..7
    uint base  = chunk << 5u;

    // Stage 3, len = 16.
    uint2 got = simd_shuffle_xor(uint2(s2lo, s2hi), (ushort)16);
    uint upper = lane >> 4u;
    uint x = (upper != 0u) ? got.y : s2lo;
    uint y = (upper != 0u) ? s2hi : got.x;
    uint t = mod_mul_3329(zetas[8u + chunk], y);
    uint v0 = mod_add_3329(x, t);
    uint v1 = mod_sub_3329(x, t);

    // Stage 4, len = 8.
    got = simd_shuffle_xor(uint2(v0, v1), (ushort)8);
    upper = r >> 3u;
    x = (upper != 0u) ? got.y : v0;
    y = (upper != 0u) ? v1 : got.x;
    t = mod_mul_3329(zetas[16u + (chunk << 1u) + upper], y);
    v0 = mod_add_3329(x, t);
    v1 = mod_sub_3329(x, t);

    // Stage 5, len = 4.
    got = simd_shuffle_xor(uint2(v0, v1), (ushort)4);
    upper = (r >> 2u) & 1u;
    x = (upper != 0u) ? got.y : v0;
    y = (upper != 0u) ? v1 : got.x;
    t = mod_mul_3329(zetas[32u + (chunk << 2u) + (r >> 2u)], y);
    v0 = mod_add_3329(x, t);
    v1 = mod_sub_3329(x, t);

    // Stage 6, len = 2.
    got = simd_shuffle_xor(uint2(v0, v1), (ushort)2);
    upper = (r >> 1u) & 1u;
    x = (upper != 0u) ? got.y : v0;
    y = (upper != 0u) ? v1 : got.x;
    t = mod_mul_3329(zetas[64u + (chunk << 3u) + (r >> 1u)], y);

    uint p = ((r >> 1u) << 2u) | (r & 1u);
    poly[base + p]      = mod_add_3329(x, t);
    poly[base + p + 2u] = mod_sub_3329(x, t);
}

kernel void kyber_ntt(
    device       uint *coeffs     [[buffer(0)]],
    device const uint *zetas      [[buffer(1)]],
    constant uint     &q          [[buffer(2)]],
    constant uint     &n          [[buffer(3)]],
    constant uint     &n_levels   [[buffer(4)]],
    constant uint     &batch      [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ltid [[thread_position_in_threadgroup]])
{
    if (tgid >= batch) return;

    threadgroup uint a[N_MAX];

    // Runtime-selected Kyber fast path: q=3329, n=256, stages len=128..2.
    // Stage 1 and stage 2 are fused to eliminate one threadgroup barrier.
    if (q == 3329u && n == 256u && n_levels == 7u) {
        device uint *poly = coeffs + ((size_t)tgid << 8);

        // Stage 0, len = 128.
        uint x0 = poly[ltid];
        uint y0 = poly[ltid + 128u];
        uint t0 = mod_mul_3329(zetas[1u], y0);

        // Store the two stage-0 halves: a[0..127] are positions 0..127,
        // a[128..255] are positions 128..255.
        a[ltid]          = mod_add_3329(x0, t0);
        a[ltid + 128u]   = mod_sub_3329(x0, t0);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Fused stages 1 and 2.
        // For a stage-2 butterfly group g32, the needed stage-1 butterflies
        // are r and r+32 inside either the low or high stage-0 half.
        uint group32  = ltid >> 5u;       // stage-2 group index, 0..3
        uint r32      = ltid & 31u;
        uint half_off = (group32 & 2u) << 6u;  // 0 for low half, 128 for high half

        uint z_s1 = zetas[2u + (group32 >> 1u)];

        uint a0 = a[half_off + r32];
        uint b0 = a[half_off + r32 + 64u];
        uint a1 = a[half_off + r32 + 32u];
        uint b1 = a[half_off + r32 + 96u];

        uint m0  = mod_mul_3329(z_s1, b0);
        uint lo0 = mod_add_3329(a0, m0);
        uint hi0 = mod_sub_3329(a0, m0);

        uint m1  = mod_mul_3329(z_s1, b1);
        uint lo1 = mod_add_3329(a1, m1);
        uint hi1 = mod_sub_3329(a1, m1);

        uint take_hi = group32 & 1u;
        uint sx = (take_hi != 0u) ? hi0 : lo0;
        uint sy = (take_hi != 0u) ? hi1 : lo1;

        uint m2   = mod_mul_3329(zetas[4u + group32], sy);
        uint s2lo = mod_add_3329(sx, m2);
        uint s2hi = mod_sub_3329(sx, m2);

        finish_kyber_3329_from_stage2(poly, zetas, ltid, s2lo, s2hi);
        return;
    }

    // Fully generic fallback for all runtime parameter sets.
    uint half_n = n >> 1u;
    device uint *poly = coeffs + (size_t)tgid * (size_t)n;

    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length  = half_n;
    uint len_log = 31u - clz(half_n);
    uint k_start = 1u;

    for (uint level = 0u; level < n_levels; ++level) {
        uint group_idx  = ltid >> len_log;
        uint j_in_group = ltid & (length - 1u);
        uint j          = (group_idx << (len_log + 1u)) | j_in_group;
        uint zeta       = zetas[k_start + group_idx];

        bfly_tg_generic(a, j, length, zeta, q);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length  >>= 1u;
        len_log -= 1u;
    }

    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}
```