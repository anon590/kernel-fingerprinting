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

inline void bfly_tg_3329(threadgroup uint *s, uint j, uint len, uint zeta) {
    uint x = s[j];
    uint y = s[j + len];
    uint t = mod_mul_3329(zeta, y);
    s[j]       = mod_add_3329(x, t);
    s[j + len] = mod_sub_3329(x, t);
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
    (void)batch;

    threadgroup uint a[N_MAX];

    // Runtime-selected Kyber fast path: q=3329, n=256, stages len=128..2.
    if (q == 3329u && n == 256u && n_levels == 7u) {
        device uint *poly = coeffs + ((size_t)tgid << 8);

        // Stage 0, len = 128.
        uint x = poly[ltid];
        uint y = poly[ltid + 128u];
        uint t = mod_mul_3329(zetas[1u], y);
        uint lo = mod_add_3329(x, t);
        uint hi = mod_sub_3329(x, t);

        // Stage 1, len = 64.  The upper64 predicate is uniform within each
        // SIMD group, so explicit branches avoid per-lane x/y selects.
        uint upper64 = ltid >> 6u;

        uint pub = hi;
        if (upper64 != 0u) {
            pub = lo;
        }
        a[ltid] = pub;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (upper64 == 0u) {
            x = lo;
            y = a[ltid + 64u];
        } else {
            x = a[ltid - 64u];
            y = hi;
        }

        t = mod_mul_3329(zetas[2u + upper64], y);
        lo = mod_add_3329(x, t);
        hi = mod_sub_3329(x, t);

        // Stage 2, len = 32.  Also SIMD-uniform per 32-thread SIMD group.
        uint group32 = ltid >> 5u;
        uint upper32 = group32 & 1u;

        pub = hi;
        if (upper32 != 0u) {
            pub = lo;
        }
        a[128u + ltid] = pub;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (upper32 == 0u) {
            x = lo;
            y = a[128u + ltid + 32u];
        } else {
            x = a[128u + ltid - 32u];
            y = hi;
        }

        t = mod_mul_3329(zetas[4u + group32], y);
        uint s2lo = mod_add_3329(x, t);
        uint s2hi = mod_sub_3329(x, t);

        // Stages 3..6 stay inside one SIMD group.
        uint lane  = ltid & 31u;
        uint r     = lane & 15u;
        uint chunk = ltid >> 4u;
        uint base  = chunk << 5u;

        // Stage 3, len = 16.
        uint2 got = simd_shuffle_xor(uint2(s2lo, s2hi), (ushort)16);
        uint upper = lane >> 4u;
        x = (upper != 0u) ? got.y : s2lo;
        y = (upper != 0u) ? s2hi  : got.x;
        t = mod_mul_3329(zetas[8u + chunk], y);
        uint v0 = mod_add_3329(x, t);
        uint v1 = mod_sub_3329(x, t);

        // Stage 4, len = 8.
        got = simd_shuffle_xor(uint2(v0, v1), (ushort)8);
        upper = r >> 3u;
        x = (upper != 0u) ? got.y : v0;
        y = (upper != 0u) ? v1    : got.x;
        t = mod_mul_3329(zetas[16u + (chunk << 1u) + upper], y);
        v0 = mod_add_3329(x, t);
        v1 = mod_sub_3329(x, t);

        // Stage 5, len = 4.
        got = simd_shuffle_xor(uint2(v0, v1), (ushort)4);
        upper = (r >> 2u) & 1u;
        x = (upper != 0u) ? got.y : v0;
        y = (upper != 0u) ? v1    : got.x;
        t = mod_mul_3329(zetas[32u + (chunk << 2u) + (r >> 2u)], y);
        v0 = mod_add_3329(x, t);
        v1 = mod_sub_3329(x, t);

        // Stage 6, len = 2.
        got = simd_shuffle_xor(uint2(v0, v1), (ushort)2);
        upper = (r >> 1u) & 1u;
        x = (upper != 0u) ? got.y : v0;
        y = (upper != 0u) ? v1    : got.x;
        t = mod_mul_3329(zetas[64u + (chunk << 3u) + (r >> 1u)], y);

        uint p = ((r >> 1u) << 2u) | (r & 1u);
        poly[base + p]      = mod_add_3329(x, t);
        poly[base + p + 2u] = mod_sub_3329(x, t);
        return;
    }

    uint half_n = n >> 1u;
    device uint *poly = coeffs + (size_t)tgid * (size_t)n;

    // Runtime q=3329 fallback for non-Kyber shapes: avoid generic u64 %.
    if (q == 3329u) {
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

            bfly_tg_3329(a, j, length, zeta);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            k_start <<= 1u;
            length  >>= 1u;
            len_log -= 1u;
        }

        poly[ltid]          = a[ltid];
        poly[ltid + half_n] = a[ltid + half_n];
        return;
    }

    // Fully generic fallback for all other runtime parameter sets.
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