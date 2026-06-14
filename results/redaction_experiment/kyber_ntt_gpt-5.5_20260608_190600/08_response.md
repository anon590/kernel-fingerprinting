This version adds a register/simd-shuffle fast path for the common `n=256, q=3329` case. After the first two cross-simd stages, each simdgroup keeps its 64-coefficient subtransform in registers and performs the remaining butterflies via `simd_shuffle`, eliminating several threadgroup-memory round trips and barriers while preserving the required per-thread butterfly mapping and canonical final output.

```metal
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

inline uint mod_add_8380417(uint a, uint b) {
    uint t = a + b;
    return (t >= 8380417u) ? (t - 8380417u) : t;
}

inline uint mod_sub_8380417(uint a, uint b) {
    uint t = a - b;
    return (a < b) ? (t + 8380417u) : t;
}

inline uint mod_mul_3329(uint a, uint b) {
    uint x = a * b;
    uint qhat = mulhi(x, 1290167u);
    uint r = x - qhat * 3329u;
    return (r >= 3329u) ? (r - 3329u) : r;
}

inline uint reduce_3329_small(uint x) {
    if (x >= 26632u) x -= 26632u;
    if (x >= 13316u) x -= 13316u;
    if (x >=  6658u) x -=  6658u;
    if (x >=  3329u) x -=  3329u;
    return x;
}

inline uint mod_mul_8380417(uint a, uint b) {
    ulong x = (ulong)a * (ulong)b;

    ulong r1 = (x & (ulong)0x7fffffu) + (x >> 23) * (ulong)8191u;
    uint r = (uint)(r1 & (ulong)0x7fffffu) + (uint)(r1 >> 23) * 8191u;

    if (r >= 67043336u) r -= 67043336u;
    if (r >= 33521668u) r -= 33521668u;
    if (r >= 16760834u) r -= 16760834u;
    if (r >=  8380417u) r -=  8380417u;
    return r;
}

inline uint mod_mul_generic(uint a, uint b, uint q) {
    return (uint)(((ulong)a * (ulong)b) % (ulong)q);
}

inline uint mod_mul_fast(uint a, uint b, uint q) {
    if (q == 3329u)    return mod_mul_3329(a, b);
    if (q == 8380417u) return mod_mul_8380417(a, b);
    return mod_mul_generic(a, b, q);
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

    if (n == 256u && q == 3329u && (n_levels == 7u || n_levels == 8u)) {
        device uint *poly = coeffs + ((size_t)tgid << 8);

        uint lane = ltid & 31u;
        uint sg   = ltid >> 5u;

        uint zload = (lane == 0u) ? zetas[1u] : 0u;
        uint z = simd_shuffle(zload, (ushort)0);

        uint x = poly[ltid];
        uint y = poly[ltid + 128u];
        uint t = mod_mul_3329(z, y);

        a[ltid]        = x + t;
        a[ltid + 128u] = x + 3329u - t;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint g64 = ltid >> 6u;
        uint j64 = (g64 << 7u) | (ltid & 63u);

        zload = (lane == 0u) ? zetas[2u + g64] : 0u;
        z = simd_shuffle(zload, (ushort)0);

        x = a[j64];
        y = a[j64 + 64u];
        t = mod_mul_3329(z, y);

        a[j64]       = x + t;
        a[j64 + 64u] = x + 3329u - t;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint base64 = sg << 6u;

        zload = (lane == 0u) ? zetas[4u + sg] : 0u;
        z = simd_shuffle(zload, (ushort)0);

        x = a[base64 + lane];
        y = a[base64 + 32u + lane];
        t = mod_mul_3329(z, y);

        uint r0 = x + t;
        uint r1 = x + 3329u - t;

        uint zl16 = 0u;
        if (lane == 0u)  zl16 = zetas[8u + (sg << 1u)];
        if (lane == 16u) zl16 = zetas[8u + (sg << 1u) + 1u];

        uint z16lo = simd_shuffle(zl16, (ushort)0);
        uint z16hi = simd_shuffle(zl16, (ushort)16);

        uint ylo16 = simd_shuffle(r0, (ushort)((lane + 16u) & 31u));
        uint xhi16 = simd_shuffle(r1, (ushort)((lane + 16u) & 31u));

        bool lo16 = (lane < 16u);
        x = lo16 ? r0    : xhi16;
        y = lo16 ? ylo16 : r1;
        z = lo16 ? z16lo : z16hi;
        t = mod_mul_3329(z, y);
        r0 = x + t;
        r1 = x + 3329u - t;

        uint zl8 = ((lane & 7u) == 0u) ? zetas[16u + (ltid >> 3u)] : 0u;
        uint z8 = simd_shuffle(zl8, (ushort)(lane & 24u));

        uint ylo8 = simd_shuffle(r0, (ushort)((lane + 8u) & 31u));
        uint xhi8 = simd_shuffle(r1, (ushort)((lane + 24u) & 31u));

        bool lo8 = ((lane & 15u) < 8u);
        x = lo8 ? r0   : xhi8;
        y = lo8 ? ylo8 : r1;
        t = mod_mul_3329(z8, y);
        r0 = x + t;
        r1 = x + 3329u - t;

        uint zl4 = ((lane & 3u) == 0u) ? zetas[32u + (ltid >> 2u)] : 0u;
        uint z4 = simd_shuffle(zl4, (ushort)(lane & 28u));

        uint ylo4 = simd_shuffle(r0, (ushort)((lane + 4u) & 31u));
        uint xhi4 = simd_shuffle(r1, (ushort)((lane + 28u) & 31u));

        bool lo4 = ((lane & 7u) < 4u);
        x = lo4 ? r0   : xhi4;
        y = lo4 ? ylo4 : r1;
        t = mod_mul_3329(z4, y);
        r0 = x + t;
        r1 = x + 3329u - t;

        uint zl2 = ((lane & 1u) == 0u) ? zetas[64u + (ltid >> 1u)] : 0u;
        uint z2 = simd_shuffle(zl2, (ushort)(lane & 30u));

        uint ylo2 = simd_shuffle(r0, (ushort)((lane + 2u) & 31u));
        uint xhi2 = simd_shuffle(r1, (ushort)((lane + 30u) & 31u));

        bool lo2 = ((lane & 3u) < 2u);
        x = lo2 ? r0   : xhi2;
        y = lo2 ? ylo2 : r1;
        t = mod_mul_3329(z2, y);
        r0 = x + t;
        r1 = x + 3329u - t;

        if (n_levels == 7u) {
            uint out = (sg << 6u) | ((lane >> 1u) << 2u) | (lane & 1u);
            poly[out]      = reduce_3329_small(r0);
            poly[out + 2u] = reduce_3329_small(r1);
            return;
        }

        uint ye1 = simd_shuffle(r0, (ushort)((lane + 1u) & 31u));
        uint xo1 = simd_shuffle(r1, (ushort)((lane + 31u) & 31u));

        bool even = ((lane & 1u) == 0u);
        x = even ? r0  : xo1;
        y = even ? ye1 : r1;
        z = zetas[128u + ltid];
        t = mod_mul_3329(z, y);

        uint out = ltid << 1u;
        poly[out]      = reduce_3329_small(x + t);
        poly[out + 1u] = reduce_3329_small(x + 3329u - t);
        return;
    }

    uint half_n = n >> 1u;
    device uint *poly = coeffs + (size_t)tgid * (size_t)n;

    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (n_levels == 0u) {
        poly[ltid]          = a[ltid];
        poly[ltid + half_n] = a[ltid + half_n];
        return;
    }

    uint length = half_n;
    uint k_start = 1u;

    for (uint level = 0u; level + 1u < n_levels; ++level) {
        uint group_idx = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j = (group_idx << 1u) * length + j_in_group;

        uint zeta = zetas[k_start + group_idx];

        uint x0 = a[j];
        uint y0 = a[j + length];
        uint tt = mod_mul_fast(zeta, y0, q);

        a[j]          = mod_add_q(x0, tt, q);
        a[j + length] = mod_sub_q(x0, tt, q);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length >>= 1u;
    }

    uint group_idx = ltid / length;
    uint j_in_group = ltid - group_idx * length;
    uint j = (group_idx << 1u) * length + j_in_group;

    uint zeta = zetas[k_start + group_idx];

    uint x0 = a[j];
    uint y0 = a[j + length];
    uint tt = mod_mul_fast(zeta, y0, q);

    poly[j]          = mod_add_q(x0, tt, q);
    poly[j + length] = mod_sub_q(x0, tt, q);
}
```