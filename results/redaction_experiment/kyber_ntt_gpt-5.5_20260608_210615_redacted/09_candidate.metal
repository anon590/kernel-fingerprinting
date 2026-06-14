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
    uint d = a - b;
    return (a >= b) ? d : (d + q);
}

// Exact for canonical inputs modulo 3329.
// For v < 3329^2, qhat=(v*315)>>20 is never too small and is at most
// one too large; an underflowed residue is corrected by adding q.
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
    uint lane  = ltid & 31u;
    uint r     = lane & 15u;
    uint chunk = ltid >> 4u;
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

    if (q == 3329u && n == 256u && n_levels == 7u) {
        device uint *poly = coeffs + ((size_t)tgid << 8);

        // Latency path for small batches: recompute the two stage-0 operands
        // needed by this thread's stage-1 butterfly, avoiding the first
        // cross-SIMD scratch exchange.  A single barrier before final stores
        // also guarantees all original in-place inputs have been consumed.
        if (batch <= 16u) {
            uint r64     = ltid & 63u;
            uint upper64 = ltid >> 6u;

            uint z0 = zetas[1u];

            uint u0 = poly[r64];
            uint v0 = poly[r64 + 128u];
            uint m0 = mod_mul_3329(z0, v0);
            uint s0a = (upper64 != 0u) ? mod_sub_3329(u0, m0)
                                        : mod_add_3329(u0, m0);

            uint u1 = poly[r64 + 64u];
            uint v1 = poly[r64 + 192u];
            uint m1 = mod_mul_3329(z0, v1);
            uint s0b = (upper64 != 0u) ? mod_sub_3329(u1, m1)
                                        : mod_add_3329(u1, m1);

            uint t = mod_mul_3329(zetas[2u + upper64], s0b);
            uint lo = mod_add_3329(s0a, t);
            uint hi = mod_sub_3329(s0a, t);

            // Stage 2, len = 32, is the only remaining cross-SIMD exchange.
            uint upper32 = (ltid >> 5u) & 1u;
            a[ltid] = (upper32 != 0u) ? lo : hi;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint ex = a[ltid ^ 32u];
            uint x  = (upper32 != 0u) ? ex : lo;
            uint y  = (upper32 != 0u) ? hi : ex;

            uint group32 = ltid >> 5u;
            t = mod_mul_3329(zetas[4u + group32], y);
            uint s2lo = mod_add_3329(x, t);
            uint s2hi = mod_sub_3329(x, t);

            finish_kyber_3329_from_stage2(poly, zetas, ltid, s2lo, s2hi);
            return;
        }

        // Throughput path: incumbent two-exchange schedule, best for many
        // resident threadgroups where redundant global loads are less helpful.
        uint x = poly[ltid];
        uint y = poly[ltid + 128u];
        uint t = mod_mul_3329(zetas[1u], y);
        uint lo = mod_add_3329(x, t);
        uint hi = mod_sub_3329(x, t);

        // Stage 1, len = 64.
        uint upper64 = ltid >> 6u;
        a[ltid] = (upper64 != 0u) ? lo : hi;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint ex = a[ltid ^ 64u];
        x = (upper64 != 0u) ? ex : lo;
        y = (upper64 != 0u) ? hi : ex;
        t = mod_mul_3329(zetas[2u + upper64], y);
        lo = mod_add_3329(x, t);
        hi = mod_sub_3329(x, t);

        // Stage 2, len = 32.
        uint upper32 = (ltid >> 5u) & 1u;
        a[128u + ltid] = (upper32 != 0u) ? lo : hi;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        ex = a[128u + (ltid ^ 32u)];
        x = (upper32 != 0u) ? ex : lo;
        y = (upper32 != 0u) ? hi : ex;

        uint group32 = ltid >> 5u;
        t = mod_mul_3329(zetas[4u + group32], y);
        uint s2lo = mod_add_3329(x, t);
        uint s2hi = mod_sub_3329(x, t);

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