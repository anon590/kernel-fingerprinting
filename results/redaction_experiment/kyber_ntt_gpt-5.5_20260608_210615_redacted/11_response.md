I use a single threadgroup barrier fast path: materialize only stage 0, then let each SIMD group recompute the stage-1 values needed for its 64-output region. This removes the second cross-SIMD scratch exchange/barrier. I also switch the in-SIMD tail from `uint2` shuffles to one scalar shuffle per stage and use branchless Kyber modular corrections, reducing shuffle and predicate overhead. 

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
    ulong s = (ulong)a + (ulong)b;
    if (s >= (ulong)q) s -= (ulong)q;
    return (uint)s;
}

inline uint mod_sub_generic(uint a, uint b, uint q) {
    return (a >= b) ? (a - b) : (uint)((ulong)a + (ulong)q - (ulong)b);
}

inline uint corr_3329(uint r) {
    return r + ((0u - (r >> 31u)) & 3329u);
}

inline uint mod_mul_3329(uint a, uint b) {
    uint v    = a * b;
    uint qhat = (v * 315u) >> 20;
    uint r    = v - qhat * 3329u;
    return corr_3329(r);
}

inline uint mod_add_3329(uint a, uint b) {
    return corr_3329(a + b - 3329u);
}

inline uint mod_sub_3329(uint a, uint b) {
    return corr_3329(a - b);
}

inline void bfly_tg_generic(threadgroup uint *s, uint j, uint len, uint zeta, uint q) {
    uint x = s[j];
    uint y = s[j + len];
    uint t = mod_mul_generic(zeta, y, q);
    s[j]       = mod_add_generic(x, t, q);
    s[j + len] = mod_sub_generic(x, t, q);
}

inline void finish_kyber_3329_scalar_shuffle(
    device       uint *poly,
    device const uint *zetas,
    uint ltid,
    uint s2lo,
    uint s2hi)
{
    uint lane  = ltid & 31u;
    uint r     = lane & 15u;
    uint chunk = ltid >> 4u;
    uint base  = chunk << 5u;

    uint upper = lane >> 4u;
    uint share = (upper != 0u) ? s2lo : s2hi;
    uint other = simd_shuffle_xor(share, (ushort)16);
    uint x     = (upper != 0u) ? other : s2lo;
    uint y     = (upper != 0u) ? s2hi : other;
    uint t     = mod_mul_3329(zetas[8u + chunk], y);
    uint v0    = mod_add_3329(x, t);
    uint v1    = mod_sub_3329(x, t);

    upper = r >> 3u;
    share = (upper != 0u) ? v0 : v1;
    other = simd_shuffle_xor(share, (ushort)8);
    x     = (upper != 0u) ? other : v0;
    y     = (upper != 0u) ? v1 : other;
    t     = mod_mul_3329(zetas[16u + (chunk << 1u) + upper], y);
    v0    = mod_add_3329(x, t);
    v1    = mod_sub_3329(x, t);

    upper = (r >> 2u) & 1u;
    share = (upper != 0u) ? v0 : v1;
    other = simd_shuffle_xor(share, (ushort)4);
    x     = (upper != 0u) ? other : v0;
    y     = (upper != 0u) ? v1 : other;
    t     = mod_mul_3329(zetas[32u + (chunk << 2u) + (r >> 2u)], y);
    v0    = mod_add_3329(x, t);
    v1    = mod_sub_3329(x, t);

    upper = (r >> 1u) & 1u;
    share = (upper != 0u) ? v0 : v1;
    other = simd_shuffle_xor(share, (ushort)2);
    x     = (upper != 0u) ? other : v0;
    y     = (upper != 0u) ? v1 : other;
    t     = mod_mul_3329(zetas[64u + (chunk << 3u) + (r >> 1u)], y);

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

    if (q == 3329u && n == 256u && n_levels == 7u) {
        device uint *poly = coeffs + ((size_t)tgid << 8);

        uint x0 = poly[ltid];
        uint y0 = poly[ltid + 128u];
        uint t0 = mod_mul_3329(zetas[1u], y0);

        a[ltid]         = mod_add_3329(x0, t0);
        a[ltid + 128u]  = mod_sub_3329(x0, t0);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint lane    = ltid & 31u;
        uint sg      = ltid >> 5u;
        uint half128 = sg >> 1u;
        uint upper64 = sg & 1u;
        uint base128 = half128 << 7u;
        uint z1      = zetas[2u + half128];

        uint u0 = a[base128 + lane];
        uint v0 = a[base128 + lane + 64u];
        uint m0 = mod_mul_3329(z1, v0);
        uint sx = (upper64 != 0u) ? mod_sub_3329(u0, m0)
                                   : mod_add_3329(u0, m0);

        uint u1 = a[base128 + lane + 32u];
        uint v1 = a[base128 + lane + 96u];
        uint m1 = mod_mul_3329(z1, v1);
        uint sy = (upper64 != 0u) ? mod_sub_3329(u1, m1)
                                   : mod_add_3329(u1, m1);

        uint m2   = mod_mul_3329(zetas[4u + sg], sy);
        uint s2lo = mod_add_3329(sx, m2);
        uint s2hi = mod_sub_3329(sx, m2);

        finish_kyber_3329_scalar_shuffle(poly, zetas, ltid, s2lo, s2hi);
        return;
    }

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