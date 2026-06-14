I fix the prior compile failure by placing the file-scope size constant in the `constant` address space. The main optimization is a specialized Kyber `q=3329,n=256,n_levels=7` path that performs the first three high-distance NTT stages in registers using a transposed per-simdgroup layout, then uses one threadgroup-memory transpose before finishing all remaining stages with simd shuffles. This removes one threadgroup barrier and a large amount of scratch traffic versus the incumbent two-barrier fast path while preserving runtime zeta loads and canonical modular outputs.  
```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_MAX = 256u;

inline uint add_3329(uint a, uint b) {
    uint s = a + b;
    return (s >= 3329u) ? (s - 3329u) : s;
}

inline uint sub_3329(uint a, uint b) {
    uint d = a - b;
    return (a < b) ? (d + 3329u) : d;
}

inline uint mod_mul_3329(uint a, uint b) {
    // Exact for a,b < 3329.
    // 3329 * 315 = 2^20 + 59, and x <= (3328)^2, so
    // qhat = (x * 315) >> 20 is floor(x/q) or floor(x/q)+1.
    uint x = a * b;
    uint qhat = (x * 315u) >> 20;
    uint prod = qhat * 3329u;
    uint r = x - prod;
    return (x < prod) ? (r + 3329u) : r;
}

inline uint add_8380417(uint a, uint b) {
    uint s = a + b;
    return (s >= 8380417u) ? (s - 8380417u) : s;
}

inline uint sub_8380417(uint a, uint b) {
    uint d = a - b;
    return (a < b) ? (d + 8380417u) : d;
}

inline uint mod_mul_8380417(uint a, uint b) {
    // 8380417 = 2^23 - 8191, so 2^23 == 8191 (mod q).
    ulong x = (ulong)a * (ulong)b;
    ulong r = (x & 0x7ffffful) + ((x >> 23) * 8191ul);
    r = (r & 0x7ffffful) + ((r >> 23) * 8191ul);
    uint s = (uint)((r & 0x7ffffful) + ((r >> 23) * 8191ul));
    if (s >= 8380417u) s -= 8380417u;
    return s;
}

inline uint mod_add_generic(uint a, uint b, uint q) {
    uint s = a + b;
    if ((s < a) || (s >= q)) s -= q;
    return s;
}

inline uint mod_sub_generic(uint a, uint b, uint q) {
    uint d = a - b;
    return (a < b) ? (d + q) : d;
}

inline uint mod_mul_generic(uint a, uint b, uint q) {
    return (uint)(((ulong)a * (ulong)b) % (ulong)q);
}

inline uint2 simd_bfly2_3329(uint v0,
                             uint v1,
                             ushort dist,
                             bool lower,
                             uint z0,
                             uint z1) {
    uint p0 = simd_shuffle_xor(v0, dist);
    uint p1 = simd_shuffle_xor(v1, dist);

    uint s0 = 0u;
    uint d0 = 0u;
    uint s1 = 0u;
    uint d1 = 0u;

    if (lower) {
        uint t0 = mod_mul_3329(z0, p0);
        uint t1 = mod_mul_3329(z1, p1);
        s0 = add_3329(v0, t0);
        d0 = sub_3329(v0, t0);
        s1 = add_3329(v1, t1);
        d1 = sub_3329(v1, t1);
    }

    uint from_lower0 = simd_shuffle_xor(d0, dist);
    uint from_lower1 = simd_shuffle_xor(d1, dist);
    return uint2(lower ? s0 : from_lower0,
                 lower ? s1 : from_lower1);
}

inline uint2 simd_step_3329(uint v0,
                            uint v1,
                            uint base0,
                            uint lane,
                            uint length,
                            uint k_start,
                            uint pair_shift,
                            device const uint *zetas) {
    uint pair_mask = (length << 1u) - 1u;
    uint source_lane = lane & ~pair_mask;

    uint zload0 = 0u;
    uint zload1 = 0u;
    if (lane == source_lane) {
        zload0 = zetas[k_start + ((base0 + source_lane) >> pair_shift)];
        zload1 = zetas[k_start + ((base0 + 32u + source_lane) >> pair_shift)];
    }

    uint z0 = simd_shuffle(zload0, (ushort)source_lane);
    uint z1 = simd_shuffle(zload1, (ushort)source_lane);

    bool lower = ((lane & length) == 0u);
    return simd_bfly2_3329(v0, v1, (ushort)length, lower, z0, z1);
}

// Fast path for q=3329, n=256, n_levels=7.
// First 3 high-bit stages are register-only in a transposed lane layout;
// one scratch-memory transpose then makes the remaining stages intra-simdgroup.
inline void ntt_256_3329_l7(threadgroup uint *a,
                            device uint *poly,
                            device const uint *zetas,
                            uint ltid) {
    uint lane = ltid & 31u;
    uint sg   = ltid >> 5u;

    // Layout per simdgroup:
    //   low = sg*8 + lane[2:0]
    //   lane group m = lane[4:3] stores h=m in v0 and h=m+4 in v1.
    uint low = (sg << 3u) | (lane & 7u);
    uint m   = lane >> 3u;

    uint idx0 = (m << 5u) | low;
    uint idx1 = ((m + 4u) << 5u) | low;

    uint v0 = poly[idx0];
    uint v1 = poly[idx1];

    // Load zetas[1..7] once per simdgroup using lanes 0..6.
    uint zload = (lane < 7u) ? zetas[lane + 1u] : 0u;

    // level 0, len = 128: pairs h with h+4, zeta[1].
    uint z1 = simd_broadcast(zload, (ushort)0);
    uint t = mod_mul_3329(z1, v1);
    uint x = v0;
    v0 = add_3329(x, t);
    v1 = sub_3329(x, t);

    // level 1, len = 64:
    //   v0: (h0,h2),(h1,h3) with zeta[2]
    //   v1: (h4,h6),(h5,h7) with zeta[3]
    uint z2 = simd_broadcast(zload, (ushort)1);
    uint z3 = simd_broadcast(zload, (ushort)2);
    uint2 r = simd_bfly2_3329(v0, v1, (ushort)16,
                              ((lane & 16u) == 0u), z2, z3);
    v0 = r.x;
    v1 = r.y;

    // level 2, len = 32:
    //   v0: (h0,h1) z4, (h2,h3) z5
    //   v1: (h4,h5) z6, (h6,h7) z7
    uint z4 = simd_broadcast(zload, (ushort)3);
    uint z5 = simd_broadcast(zload, (ushort)4);
    uint z6 = simd_broadcast(zload, (ushort)5);
    uint z7 = simd_broadcast(zload, (ushort)6);

    uint zlo = ((lane & 16u) == 0u) ? z4 : z5;
    uint zhi = ((lane & 16u) == 0u) ? z6 : z7;

    r = simd_bfly2_3329(v0, v1, (ushort)8,
                        ((lane & 8u) == 0u), zlo, zhi);
    v0 = r.x;
    v1 = r.y;

    // Transpose back to natural coefficient order for the intra-32 stages.
    a[idx0] = v0;
    a[idx1] = v1;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Remaining levels are independent within 32-coefficient blocks.
    // Each simdgroup owns two adjacent 32-coefficient blocks.
    uint base0 = sg << 6u;
    uint base1 = base0 + 32u;

    v0 = a[base0 + lane];
    v1 = a[base1 + lane];

    // level 3, len = 16, k = 8
    r = simd_step_3329(v0, v1, base0, lane, 16u, 8u, 5u, zetas);
    v0 = r.x; v1 = r.y;

    // level 4, len = 8, k = 16
    r = simd_step_3329(v0, v1, base0, lane, 8u, 16u, 4u, zetas);
    v0 = r.x; v1 = r.y;

    // level 5, len = 4, k = 32
    r = simd_step_3329(v0, v1, base0, lane, 4u, 32u, 3u, zetas);
    v0 = r.x; v1 = r.y;

    // level 6, len = 2, k = 64
    r = simd_step_3329(v0, v1, base0, lane, 2u, 64u, 2u, zetas);
    v0 = r.x; v1 = r.y;

    poly[base0 + lane] = v0;
    poly[base1 + lane] = v1;
}

#define MUL3329(z, y, q_)      mod_mul_3329((z), (y))
#define ADD3329(x, t, q_)      add_3329((x), (t))
#define SUB3329(x, t, q_)      sub_3329((x), (t))

#define MUL8380417(z, y, q_)   mod_mul_8380417((z), (y))
#define ADD8380417(x, t, q_)   add_8380417((x), (t))
#define SUB8380417(x, t, q_)   sub_8380417((x), (t))

#define MULGEN(z, y, q_)       mod_mul_generic((z), (y), (q_))
#define ADDGEN(x, t, q_)       mod_add_generic((x), (t), (q_))
#define SUBGEN(x, t, q_)       mod_sub_generic((x), (t), (q_))

#define DEFINE_NTT_BODY(FN, MUL, ADD, SUB)                                      \
inline void FN(threadgroup uint *a,                                              \
               threadgroup uint *zs,                                             \
               device uint *poly,                                                \
               uint q,                                                           \
               uint n,                                                           \
               uint n_levels,                                                    \
               uint ltid)                                                        \
{                                                                                \
    uint half_n = n >> 1u;                                                        \
    uint logn = 31u - clz(n);                                                     \
    uint length = half_n;                                                         \
    uint k_start = 1u;                                                            \
    uint level = 0u;                                                              \
                                                                                 \
    for (; (level < n_levels) && (length >= 32u); ++level) {                     \
        uint len_shift = logn - 1u - level;                                      \
        uint group_idx = ltid >> len_shift;                                      \
        uint j_in_group = ltid & (length - 1u);                                  \
        uint j = (group_idx << (len_shift + 1u)) | j_in_group;                   \
        uint z = zs[k_start + group_idx];                                        \
                                                                                 \
        uint x = a[j];                                                           \
        uint y = a[j + length];                                                  \
        uint t = MUL(z, y, q);                                                   \
                                                                                 \
        a[j] = ADD(x, t, q);                                                     \
        a[j + length] = SUB(x, t, q);                                            \
                                                                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                         \
        k_start <<= 1u;                                                          \
        length >>= 1u;                                                           \
    }                                                                            \
                                                                                 \
    if ((level < n_levels) && (half_n >= 32u)) {                                 \
        uint sg = ltid >> 5u;                                                     \
        uint lane = ltid & 31u;                                                   \
        uint num_sg = half_n >> 5u;                                               \
        uint base0 = sg << 5u;                                                    \
        uint base1 = (sg + num_sg) << 5u;                                         \
                                                                                 \
        uint v0 = a[base0 + lane];                                                \
        uint v1 = a[base1 + lane];                                                \
                                                                                 \
        for (; level < n_levels; ++level) {                                      \
            uint len_shift = logn - 1u - level;                                  \
            uint pair_shift = len_shift + 1u;                                    \
            bool lower = ((lane & length) == 0u);                                \
                                                                                 \
            uint p0 = simd_shuffle_xor(v0, (ushort)length);                      \
            uint p1 = simd_shuffle_xor(v1, (ushort)length);                      \
                                                                                 \
            uint sum0 = 0u;                                                      \
            uint dif0 = 0u;                                                      \
            uint sum1 = 0u;                                                      \
            uint dif1 = 0u;                                                      \
                                                                                 \
            if (lower) {                                                         \
                uint group0 = (base0 + lane) >> pair_shift;                      \
                uint group1 = (base1 + lane) >> pair_shift;                      \
                uint z0 = zs[k_start + group0];                                  \
                uint z1 = zs[k_start + group1];                                  \
                uint t0 = MUL(z0, p0, q);                                        \
                uint t1 = MUL(z1, p1, q);                                        \
                sum0 = ADD(v0, t0, q);                                           \
                dif0 = SUB(v0, t0, q);                                           \
                sum1 = ADD(v1, t1, q);                                           \
                dif1 = SUB(v1, t1, q);                                           \
            }                                                                    \
                                                                                 \
            uint from_lower0 = simd_shuffle_xor(dif0, (ushort)length);           \
            uint from_lower1 = simd_shuffle_xor(dif1, (ushort)length);           \
                                                                                 \
            v0 = lower ? sum0 : from_lower0;                                     \
            v1 = lower ? sum1 : from_lower1;                                     \
                                                                                 \
            k_start <<= 1u;                                                      \
            length >>= 1u;                                                       \
        }                                                                        \
                                                                                 \
        poly[base0 + lane] = v0;                                                  \
        poly[base1 + lane] = v1;                                                  \
        return;                                                                  \
    }                                                                            \
                                                                                 \
    for (; level < n_levels; ++level) {                                          \
        uint len_shift = logn - 1u - level;                                      \
        uint group_idx = ltid >> len_shift;                                      \
        uint j_in_group = ltid & (length - 1u);                                  \
        uint j = (group_idx << (len_shift + 1u)) | j_in_group;                   \
        uint z = zs[k_start + group_idx];                                        \
                                                                                 \
        uint x = a[j];                                                           \
        uint y = a[j + length];                                                  \
        uint t = MUL(z, y, q);                                                   \
                                                                                 \
        a[j] = ADD(x, t, q);                                                     \
        a[j + length] = SUB(x, t, q);                                            \
                                                                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                         \
        k_start <<= 1u;                                                          \
        length >>= 1u;                                                           \
    }                                                                            \
                                                                                 \
    poly[ltid] = a[ltid];                                                        \
    poly[ltid + half_n] = a[ltid + half_n];                                      \
}

DEFINE_NTT_BODY(ntt_body_3329,    MUL3329,    ADD3329,    SUB3329)
DEFINE_NTT_BODY(ntt_body_8380417, MUL8380417, ADD8380417, SUB8380417)
DEFINE_NTT_BODY(ntt_body_generic, MULGEN,     ADDGEN,     SUBGEN)

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

    device uint *poly = coeffs + (size_t)tgid * (size_t)n;

    if ((q == 3329u) && (n == 256u) && (n_levels == 7u)) {
        ntt_256_3329_l7(a, poly, zetas, ltid);
        return;
    }

    threadgroup uint zs[N_MAX];

    uint half_n = n >> 1u;

    a[ltid] = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];

    uint zeta_count = 1u << n_levels;
    if (ltid < zeta_count) {
        zs[ltid] = zetas[ltid];
    }
    uint zi = ltid + half_n;
    if (zi < zeta_count) {
        zs[zi] = zetas[zi];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (q == 3329u) {
        ntt_body_3329(a, zs, poly, q, n, n_levels, ltid);
    } else if (q == 8380417u) {
        ntt_body_8380417(a, zs, poly, q, n, n_levels, ltid);
    } else {
        ntt_body_generic(a, zs, poly, q, n, n_levels, ltid);
    }
}
```