#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_MAX = 256u;

inline uint mod_add_q(uint a, uint b, uint q) {
    uint t = a + b;
    uint r = t - q;
    return ((t >= q) || (t < a)) ? r : t;
}

inline uint mod_sub_q(uint a, uint b, uint q) {
    uint t = a - b;
    return (a < b) ? (t + q) : t;
}

// Exact for q = 3329 and b < 9*q, covering the lazy 256-point paths.
inline uint mod_mul_3329(uint a, uint b) {
    uint x = a * b;
    uint qhat = mulhi(x, 1290167u); // floor(2^32 / 3329)
    uint r = x - qhat * 3329u;
    return (r >= 3329u) ? (r - 3329u) : r;
}

inline uint reduce_3329_small(uint x) {
    if (x >= 26632u) x -= 26632u; // 8*q
    if (x >= 13316u) x -= 13316u; // 4*q
    if (x >=  6658u) x -=  6658u; // 2*q
    if (x >=  3329u) x -=  3329u; // q
    return x;
}

inline uint mod_mul_generic(uint a, uint b, uint q) {
    return (uint)(((ulong)a * (ulong)b) % (ulong)q);
}

inline uint mod_mul_fast(uint a, uint b, uint q) {
    if (q == 3329u) return mod_mul_3329(a, b);
    return mod_mul_generic(a, b, q);
}

#define REG_STAGE_3329(LEN, LOGV, KSTART) do {                                      \
    uint _local = lane & 15u;                                                       \
    uint _grp   = _local >> (LOGV);                                                 \
    uint _j     = _local & ((LEN) - 1u);                                            \
    uint _p0    = (_grp << ((LOGV) + 1u)) | _j;                                     \
    uint _p1    = _p0 + (LEN);                                                      \
                                                                                     \
    uint _x0 = simd_shuffle(r0, (ushort)_p0);                                       \
    uint _y0 = simd_shuffle(r0, (ushort)_p1);                                       \
    uint _x1 = simd_shuffle(r1, (ushort)_p0);                                       \
    uint _y1 = simd_shuffle(r1, (ushort)_p1);                                       \
    uint _x  = (lane < 16u) ? _x0 : _x1;                                            \
    uint _y  = (lane < 16u) ? _y0 : _y1;                                            \
                                                                                     \
    uint _groups_per_half = 16u >> (LOGV);                                          \
    uint _global_group = (sg << (5u - (LOGV))) + ((lane >> 4u) * _groups_per_half) + _grp; \
    uint _z = zt[(KSTART) + _global_group];                                         \
    uint _t = mod_mul_3329(_z, _y);                                                 \
    uint _top = _x + _t;                                                            \
    uint _bot = _x + 3329u - _t;                                                    \
                                                                                     \
    uint _dst = ((lane >> ((LOGV) + 1u)) << (LOGV)) | (lane & ((LEN) - 1u));         \
    bool _upper = ((lane & (LEN)) != 0u);                                           \
                                                                                     \
    uint _r0t = simd_shuffle(_top, (ushort)_dst);                                   \
    uint _r0b = simd_shuffle(_bot, (ushort)_dst);                                   \
    uint _r1t = simd_shuffle(_top, (ushort)(16u + _dst));                           \
    uint _r1b = simd_shuffle(_bot, (ushort)(16u + _dst));                           \
                                                                                     \
    r0 = _upper ? _r0b : _r0t;                                                      \
    r1 = _upper ? _r1b : _r1t;                                                      \
} while (0)

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
    threadgroup uint zt[N_MAX];

    if (n == 256u && q == 3329u && (n_levels == 7u || n_levels == 8u)) {
        device uint *poly = coeffs + ((size_t)tgid << 8);

        zt[ltid] = zetas[ltid];
        if (n_levels == 8u) {
            zt[ltid + 128u] = zetas[ltid + 128u];
        }

        uint lane = ltid & 31u;
        uint zl = (lane == 0u) ? zetas[1u] : 0u;
        uint z = simd_shuffle(zl, (ushort)0);

        uint x = poly[ltid];
        uint y = poly[ltid + 128u];
        uint t = mod_mul_3329(z, y);

        a[ltid]        = x + t;
        a[ltid + 128u] = x + 3329u - t;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint g64 = ltid >> 6u;
        uint j64 = (g64 << 7u) | (ltid & 63u);
        z = zt[2u + g64];

        x = a[j64];
        y = a[j64 + 64u];
        t = mod_mul_3329(z, y);
        a[j64]       = x + t;
        a[j64 + 64u] = x + 3329u - t;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint sg = ltid >> 5u;
        uint base = sg << 6u;
        z = zt[4u + sg];

        x = a[base + lane];
        y = a[base + lane + 32u];
        t = mod_mul_3329(z, y);

        uint r0 = x + t;
        uint r1 = x + 3329u - t;

        REG_STAGE_3329(16u, 4u,   8u);
        REG_STAGE_3329( 8u, 3u,  16u);
        REG_STAGE_3329( 4u, 2u,  32u);
        REG_STAGE_3329( 2u, 1u,  64u);

        if (n_levels == 8u) {
            REG_STAGE_3329(1u, 0u, 128u);
        }

        poly[base + lane]       = reduce_3329_small(r0);
        poly[base + lane + 32u] = reduce_3329_small(r1);
        return;
    }

    uint half_n = n >> 1u;
    device uint *poly = coeffs + (size_t)tgid * (size_t)n;

    a[ltid] = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];

    uint zcount = 1u << n_levels;
    if (ltid < zcount) {
        zt[ltid] = zetas[ltid];
    }
    uint z2 = ltid + half_n;
    if (z2 < zcount) {
        zt[z2] = zetas[z2];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (n_levels == 0u) {
        poly[ltid] = a[ltid];
        poly[ltid + half_n] = a[ltid + half_n];
        return;
    }

    uint length = half_n;
    uint k_start = 1u;

    for (uint level = 0u; level + 1u < n_levels; ++level) {
        uint group_idx = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j = (group_idx << 1u) * length + j_in_group;
        uint zeta = zt[k_start + group_idx];

        uint vx = a[j];
        uint vy = a[j + length];
        uint vt = mod_mul_fast(zeta, vy, q);

        a[j] = mod_add_q(vx, vt, q);
        a[j + length] = mod_sub_q(vx, vt, q);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length >>= 1u;
    }

    uint group_idx = ltid / length;
    uint j_in_group = ltid - group_idx * length;
    uint j = (group_idx << 1u) * length + j_in_group;
    uint zeta = zt[k_start + group_idx];

    uint vx = a[j];
    uint vy = a[j + length];
    uint vt = mod_mul_fast(zeta, vy, q);

    poly[j] = mod_add_q(vx, vt, q);
    poly[j + length] = mod_sub_q(vx, vt, q);
}