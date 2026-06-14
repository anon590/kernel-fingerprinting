The fast path is still selected only from runtime parameters, but now stage 2 is kept in SIMD registers instead of threadgroup memory. I also convert shared twiddles to Montgomery form per SIMD subgroup and broadcast them, so most butterflies avoid repeated reciprocal high-half reductions. This should reduce both barrier pressure and integer multiply cost versus the previous version while preserving the generic fallback.

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
    return (a >= b) ? (a - b) : (a + q - b);
}

// Exact for canonical inputs modulo 3329, product < 3329^2.
// floor(2^32 / 3329) = 1290167; qhat is at most one low.
inline uint mod_mul_barrett_3329(uint a, uint b) {
    uint v = a * b;
    uint qhat = mulhi(v, 1290167u);
    uint r = v - qhat * 3329u;
    return (r >= 3329u) ? (r - 3329u) : r;
}

inline uint mod_add_3329(uint a, uint b) {
    uint s = a + b;
    return (s >= 3329u) ? (s - 3329u) : s;
}

inline uint mod_sub_3329(uint a, uint b) {
    return (a >= b) ? (a - b) : (a + 3329u - b);
}

// Convert normal-domain zeta to Montgomery domain: zeta * 2^16 mod 3329.
// 2^16 mod 3329 = 2285.
inline uint to_mont_3329(uint z) {
    return mod_mul_barrett_3329(z, 2285u);
}

// REDC(x) for x < 3329^2, with q' = -q^{-1} mod 2^16 = 3327.
// If bR = b * 2^16 mod q, returns a*b mod q.
inline uint mont_mul_3329(uint a, uint bR) {
    uint x = a * bR;
    uint u = (x * 3327u) & 0xFFFFu;
    uint t = (x + u * 3329u) >> 16;
    return (t >= 3329u) ? (t - 3329u) : t;
}

inline uint bcast_zeta_mont_3329(device const uint *zetas,
                                  uint zidx,
                                  uint lane,
                                  uint leader) {
    uint zr = 0u;
    if (lane == leader) {
        zr = to_mont_3329(zetas[zidx]);
    }
    return simd_shuffle(zr, (ushort)leader);
}

inline void bfly_tg_mont_3329(threadgroup uint *s, uint j, uint len, uint zR) {
    uint x = s[j];
    uint y = s[j + len];
    uint t = mont_mul_3329(y, zR);
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

// Fetch a coefficient from a previous register-resident CT stage.
// vals.x/vals.y are the low/high outputs owned by each lane in the
// current lane block. pos is local to that block.
inline uint fetch_reg_pair(uint2 vals, uint pos, uint prev_log, uint lane_base) {
    uint prev_len = 1u << prev_log;
    uint span_log = prev_log + 1u;
    uint offset   = pos & ((1u << span_log) - 1u);
    uint owner    = ((pos >> span_log) << prev_log) | (offset & (prev_len - 1u));
    uint2 got     = simd_shuffle(vals, (ushort)(lane_base + owner));
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

    if (q == 3329u && n == 256u && n_levels == 7u) {
        device uint *poly = coeffs + (size_t)tgid * (size_t)256u;

        uint lane = ltid & 31u;

        // Stage 0, len = 128.
        uint zR = bcast_zeta_mont_3329(zetas, 1u, lane, 0u);
        uint x = poly[ltid];
        uint y = poly[ltid + 128u];
        uint t = mont_mul_3329(y, zR);
        a[ltid]        = mod_add_3329(x, t);
        a[ltid + 128u] = mod_sub_3329(x, t);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Stage 1, len = 64.
        uint g = ltid >> 6u;
        uint j = (g << 7u) | (ltid & 63u);
        zR = bcast_zeta_mont_3329(zetas, 2u + g, lane, 0u);
        bfly_tg_mont_3329(a, j, 64u, zR);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Stage 2, len = 32, enters registers. One simdgroup owns 64 coeffs.
        uint sg     = ltid >> 5u;       // 0..3
        uint base64 = sg << 6u;
        zR = bcast_zeta_mont_3329(zetas, 4u + sg, lane, 0u);

        uint rx = a[base64 + lane];
        uint ry = a[base64 + lane + 32u];
        uint rt = mont_mul_3329(ry, zR);
        uint v0 = mod_add_3329(rx, rt);
        uint v1 = mod_sub_3329(rx, rt);

        // Stage 3, len = 16. Split the 64-coeff simdgroup into two 32-coeff chunks.
        uint half_base = lane & 16u;
        uint r         = lane & 15u;
        uint chunk     = (sg << 1u) + (lane >> 4u); // 0..7
        uint p64       = ((lane >> 4u) << 5u) | r;

        uint2 vals = uint2(v0, v1);
        rx = fetch_reg_pair(vals, p64,       5u, 0u);
        ry = fetch_reg_pair(vals, p64 + 16u, 5u, 0u);
        zR = bcast_zeta_mont_3329(zetas, 8u + chunk, lane, half_base);
        rt = mont_mul_3329(ry, zR);
        v0 = mod_add_3329(rx, rt);
        v1 = mod_sub_3329(rx, rt);

        // Stage 4, len = 8.
        uint p = ((r >> 3u) << 4u) | (r & 7u);
        vals = uint2(v0, v1);
        rx = fetch_reg_pair(vals, p,      4u, half_base);
        ry = fetch_reg_pair(vals, p + 8u, 4u, half_base);
        uint subgroup = r >> 3u;
        zR = bcast_zeta_mont_3329(zetas,
                                  16u + (chunk << 1u) + subgroup,
                                  lane,
                                  half_base + (subgroup << 3u));
        rt = mont_mul_3329(ry, zR);
        v0 = mod_add_3329(rx, rt);
        v1 = mod_sub_3329(rx, rt);

        // Stage 5, len = 4.
        p = ((r >> 2u) << 3u) | (r & 3u);
        vals = uint2(v0, v1);
        rx = fetch_reg_pair(vals, p,      3u, half_base);
        ry = fetch_reg_pair(vals, p + 4u, 3u, half_base);
        subgroup = r >> 2u;
        zR = bcast_zeta_mont_3329(zetas,
                                  32u + (chunk << 2u) + subgroup,
                                  lane,
                                  half_base + (subgroup << 2u));
        rt = mont_mul_3329(ry, zR);
        v0 = mod_add_3329(rx, rt);
        v1 = mod_sub_3329(rx, rt);

        // Stage 6, len = 2. Direct Barrett is cheaper than converting a zeta used by 2 lanes.
        p = ((r >> 1u) << 2u) | (r & 1u);
        vals = uint2(v0, v1);
        rx = fetch_reg_pair(vals, p,      2u, half_base);
        ry = fetch_reg_pair(vals, p + 2u, 2u, half_base);
        subgroup = r >> 1u;
        rt = mod_mul_barrett_3329(zetas[64u + (chunk << 3u) + subgroup], ry);

        uint base = chunk << 5u;
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
```