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
//
// v < 3329^2 and ceil(2^20 / 3329) = 315, with
// 315*3329 = 2^20 + 59.
// qhat = floor(v*315 / 2^20) is therefore either floor(v/q)
// or one too large; it is never too small.  The one-too-large case
// produces an unsigned wrapped negative residue, corrected by adding q.
inline uint mod_mul_3329(uint a, uint b) {
    uint v    = a * b;                  // exact: v < 2^24
    uint qhat = (v * 315u) >> 20;       // exact: v*315 < 2^32
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

inline void bfly_tg_3329(threadgroup uint *s, uint j, uint len, uint zeta) {
    uint x = s[j];
    uint y = s[j + len];
    uint t = mod_mul_3329(zeta, y);
    s[j]       = mod_add_3329(x, t);
    s[j + len] = mod_sub_3329(x, t);
}

inline void bfly_tg_generic(threadgroup uint *s, uint j, uint len, uint zeta, uint q) {
    uint x = s[j];
    uint y = s[j + len];
    uint t = mod_mul_generic(zeta, y, q);
    s[j]       = mod_add_generic(x, t, q);
    s[j + len] = mod_sub_generic(x, t, q);
}

// Fetch a coefficient from the previous register-resident CT stage.
// vals.x/vals.y are the low/high outputs owned by each lane in a 16-lane
// half-simdgroup. pos is local to the 32-coefficient chunk.
inline uint fetch_reg_pair(uint2 vals, uint pos, uint prev_log, uint half_base) {
    uint prev_len = 1u << prev_log;
    uint span_log = prev_log + 1u;
    uint offset   = pos & ((1u << span_log) - 1u);
    uint owner    = ((pos >> span_log) << prev_log) | (offset & (prev_len - 1u));
    uint2 got     = simd_shuffle(vals, (ushort)(half_base + owner));
    return ((offset & prev_len) != 0u) ? got.y : got.x;
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

    // Fast path for the Kyber parameter set, selected from runtime buffers.
    if (q == 3329u && n == 256u && n_levels == 7u) {
        device uint *poly = coeffs + ((size_t)tgid << 8);

        // Stage 0, len = 128.
        uint x0 = poly[ltid];
        uint y0 = poly[ltid + 128u];
        uint t0 = mod_mul_3329(zetas[1u], y0);
        a[ltid]        = mod_add_3329(x0, t0);
        a[ltid + 128u] = mod_sub_3329(x0, t0);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Stage 1, len = 64.
        uint g = ltid >> 6u;
        uint j = (g << 7u) | (ltid & 63u);
        bfly_tg_3329(a, j, 64u, zetas[2u + g]);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Stage 2, len = 32.
        g = ltid >> 5u;
        j = (g << 6u) | (ltid & 31u);
        bfly_tg_3329(a, j, 32u, zetas[4u + g]);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Stages 3..6 are independent inside 32-coefficient chunks.
        // Each half-simdgroup owns one 32-point chunk and keeps the active
        // butterfly pair in registers.
        uint lane      = ltid & 31u;
        uint half_base = lane & 16u;
        uint r         = lane & 15u;
        uint chunk     = ltid >> 4u;      // 0..7
        uint base      = chunk << 5u;     // coefficient base of 32-point chunk

        // Stage 3, len = 16.
        uint rx = a[base + r];
        uint ry = a[base + r + 16u];
        uint rt = mod_mul_3329(zetas[8u + chunk], ry);
        uint v0 = mod_add_3329(rx, rt);
        uint v1 = mod_sub_3329(rx, rt);

        // Stage 4, len = 8.
        uint p = ((r >> 3u) << 4u) | (r & 7u);
        uint2 vals = uint2(v0, v1);
        rx = fetch_reg_pair(vals, p,      4u, half_base);
        ry = fetch_reg_pair(vals, p + 8u, 4u, half_base);
        rt = mod_mul_3329(zetas[16u + (chunk << 1u) + (r >> 3u)], ry);
        v0 = mod_add_3329(rx, rt);
        v1 = mod_sub_3329(rx, rt);

        // Stage 5, len = 4.
        p = ((r >> 2u) << 3u) | (r & 3u);
        vals = uint2(v0, v1);
        rx = fetch_reg_pair(vals, p,      3u, half_base);
        ry = fetch_reg_pair(vals, p + 4u, 3u, half_base);
        rt = mod_mul_3329(zetas[32u + (chunk << 2u) + (r >> 2u)], ry);
        v0 = mod_add_3329(rx, rt);
        v1 = mod_sub_3329(rx, rt);

        // Stage 6, len = 2.
        p = ((r >> 1u) << 2u) | (r & 1u);
        vals = uint2(v0, v1);
        rx = fetch_reg_pair(vals, p,      2u, half_base);
        ry = fetch_reg_pair(vals, p + 2u, 2u, half_base);
        rt = mod_mul_3329(zetas[64u + (chunk << 3u) + (r >> 1u)], ry);

        poly[base + p]      = mod_add_3329(rx, rt);
        poly[base + p + 2u] = mod_sub_3329(rx, rt);
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