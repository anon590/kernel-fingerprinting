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

// For q = 3329 and canonical inputs.  The intermediate "negative" cases
// are unsigned underflows with the high bit set; adding q wraps them back.
inline uint fix_under_3329(uint x) {
    return x + ((0u - (x >> 31u)) & 3329u);
}

inline uint mod_mul_3329(uint a, uint b) {
    uint v    = a * b;                 // < 3329^2, so v*315 fits uint32.
    uint qhat = (v * 315u) >> 20u;     // exact quotient or one too high.
    return fix_under_3329(v - qhat * 3329u);
}

inline uint mod_add_3329(uint a, uint b) {
    return fix_under_3329(a + b - 3329u);
}

inline uint mod_sub_3329(uint a, uint b) {
    return fix_under_3329(a - b);
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
    if (tgid >= batch) return;

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

        // Stage 1, len = 64.  Cross-SIMD exchange through threadgroup memory.
        uint upper64 = ltid >> 6u;
        uint z_s1    = zetas[2u + upper64];
        a[ltid] = (upper64 != 0u) ? lo : hi;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint ex = a[ltid ^ 64u];
        x = (upper64 != 0u) ? ex : lo;
        y = (upper64 != 0u) ? hi : ex;
        t = mod_mul_3329(z_s1, y);
        lo = mod_add_3329(x, t);
        hi = mod_sub_3329(x, t);

        // Stage 2, len = 32.  Cross-SIMD exchange through the upper scratch half.
        uint group32 = ltid >> 5u;
        uint upper32 = group32 & 1u;
        uint z_s2    = zetas[4u + group32];
        a[128u + ltid] = (upper32 != 0u) ? lo : hi;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        ex = a[128u + (ltid ^ 32u)];
        x = (upper32 != 0u) ? ex : lo;
        y = (upper32 != 0u) ? hi : ex;
        t = mod_mul_3329(z_s2, y);
        uint v0 = mod_add_3329(x, t);
        uint v1 = mod_sub_3329(x, t);

        // Stages 3..6 are inside one SIMD group.  Shuffle only the scalar
        // needed by the paired lane instead of shuffling a uint2.
        uint lane  = ltid & 31u;
        uint r     = lane & 15u;
        uint chunk = ltid >> 4u;       // 0..7
        uint base  = chunk << 5u;

        // Stage 3, len = 16.
        uint upper = lane >> 4u;
        uint z_s3  = zetas[8u + chunk];
        uint send  = (upper != 0u) ? v0 : v1;
        ex = simd_shuffle_xor(send, (ushort)16);
        x = (upper != 0u) ? ex : v0;
        y = (upper != 0u) ? v1 : ex;
        t = mod_mul_3329(z_s3, y);
        v0 = mod_add_3329(x, t);
        v1 = mod_sub_3329(x, t);

        // Stage 4, len = 8.
        upper = r >> 3u;
        uint z_s4 = zetas[16u + (chunk << 1u) + upper];
        send = (upper != 0u) ? v0 : v1;
        ex = simd_shuffle_xor(send, (ushort)8);
        x = (upper != 0u) ? ex : v0;
        y = (upper != 0u) ? v1 : ex;
        t = mod_mul_3329(z_s4, y);
        v0 = mod_add_3329(x, t);
        v1 = mod_sub_3329(x, t);

        // Stage 5, len = 4.
        uint r2 = r >> 2u;
        upper = r2 & 1u;
        uint z_s5 = zetas[32u + (chunk << 2u) + r2];
        send = (upper != 0u) ? v0 : v1;
        ex = simd_shuffle_xor(send, (ushort)4);
        x = (upper != 0u) ? ex : v0;
        y = (upper != 0u) ? v1 : ex;
        t = mod_mul_3329(z_s5, y);
        v0 = mod_add_3329(x, t);
        v1 = mod_sub_3329(x, t);

        // Stage 6, len = 2.
        uint r1 = r >> 1u;
        upper = r1 & 1u;
        uint z_s6 = zetas[64u + (chunk << 3u) + r1];
        send = (upper != 0u) ? v0 : v1;
        ex = simd_shuffle_xor(send, (ushort)2);
        x = (upper != 0u) ? ex : v0;
        y = (upper != 0u) ? v1 : ex;
        t = mod_mul_3329(z_s6, y);

        uint p = (r1 << 2u) | (r & 1u);
        poly[base + p]      = mod_add_3329(x, t);
        poly[base + p + 2u] = mod_sub_3329(x, t);
        return;
    }

    uint half_n = n >> 1u;
    device uint *poly = coeffs + (size_t)tgid * (size_t)n;

    // q=3329 generic fallback: same runtime n/n_levels path, but avoid u64 %.
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