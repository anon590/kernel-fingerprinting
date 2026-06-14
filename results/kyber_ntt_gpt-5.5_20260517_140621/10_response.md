I split the q=3329, n=256 fast path into separate level-7 and level-8 kernels. This removes the per-thread dynamic `n_levels == 8` branch from the hot Kyber level-7 path and shortens live ranges while preserving the incumbent’s barrier-free simdgroup recomputation strategy, which is fastest for large batches on Apple GPUs. The generic/runtime-parameter path is retained unchanged for other moduli or sizes. 

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_MAX = 256u;

inline uint add_3329(uint a, uint b) {
    uint r = a + b - 3329u;
    r += (0u - (r >> 31)) & 3329u;
    return r;
}

inline uint sub_3329(uint a, uint b) {
    uint r = a - b;
    r += (0u - (r >> 31)) & 3329u;
    return r;
}

inline uint mod_mul_3329(uint a, uint b) {
    uint x = a * b;
    uint qhat = (x * 315u) >> 20;
    uint r = x - qhat * 3329u;
    r += (0u - (r >> 31)) & 3329u;
    return r;
}

inline uint add_8380417(uint a, uint b) {
    uint r = a + b - 8380417u;
    r += (0u - (r >> 31)) & 8380417u;
    return r;
}

inline uint sub_8380417(uint a, uint b) {
    uint r = a - b;
    r += (0u - (r >> 31)) & 8380417u;
    return r;
}

inline uint reduce_once_8380417(uint s) {
    uint r = s - 8380417u;
    r += (0u - (r >> 31)) & 8380417u;
    return r;
}

inline uint mod_mul_8380417(uint a, uint b) {
    ulong x = (ulong)a * (ulong)b;
    ulong r = (x & 0x7ffffful) + ((x >> 23) * 8191ul);
    r = (r & 0x7ffffful) + ((r >> 23) * 8191ul);
    uint s = (uint)((r & 0x7ffffful) + ((r >> 23) * 8191ul));
    return reduce_once_8380417(s);
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

inline uint bcast_zeta_lane0(device const uint *zetas, uint idx, uint lane) {
    uint z = (lane == 0u) ? zetas[idx] : 0u;
    return simd_broadcast(z, (ushort)0);
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

inline uint2 simd_stage16_3329(uint v0,
                               uint v1,
                               uint sg,
                               uint lane,
                               device const uint *zetas) {
    uint zl = (lane < 2u) ? zetas[8u + (sg << 1u) + lane] : 0u;
    uint z0 = simd_shuffle(zl, (ushort)0);
    uint z1 = simd_shuffle(zl, (ushort)1);
    return simd_bfly2_3329(v0, v1, (ushort)16, (lane < 16u), z0, z1);
}

inline uint2 simd_stage8_3329(uint v0,
                              uint v1,
                              uint sg,
                              uint lane,
                              device const uint *zetas) {
    uint zl = (lane < 4u) ? zetas[16u + (sg << 2u) + lane] : 0u;
    uint seg = lane >> 4u;
    uint z0 = simd_shuffle(zl, (ushort)seg);
    uint z1 = simd_shuffle(zl, (ushort)(seg + 2u));
    return simd_bfly2_3329(v0, v1, (ushort)8, ((lane & 8u) == 0u), z0, z1);
}

inline uint2 simd_stage4_3329(uint v0,
                              uint v1,
                              uint sg,
                              uint lane,
                              device const uint *zetas) {
    uint zl = (lane < 8u) ? zetas[32u + (sg << 3u) + lane] : 0u;
    uint seg = (lane >> 3u) & 3u;
    uint z0 = simd_shuffle(zl, (ushort)seg);
    uint z1 = simd_shuffle(zl, (ushort)(seg + 4u));
    return simd_bfly2_3329(v0, v1, (ushort)4, ((lane & 4u) == 0u), z0, z1);
}

inline uint2 simd_stage2_3329(uint v0,
                              uint v1,
                              uint sg,
                              uint lane,
                              device const uint *zetas) {
    uint zl = (lane < 16u) ? zetas[64u + (sg << 4u) + lane] : 0u;
    uint seg = (lane >> 2u) & 7u;
    uint z0 = simd_shuffle(zl, (ushort)seg);
    uint z1 = simd_shuffle(zl, (ushort)(seg + 8u));
    return simd_bfly2_3329(v0, v1, (ushort)2, ((lane & 2u) == 0u), z0, z1);
}

inline uint2 simd_stage1_3329(uint v0,
                              uint v1,
                              uint sg,
                              uint lane,
                              device const uint *zetas) {
    uint zl0 = (lane < 16u) ? zetas[128u + (sg << 5u) + lane] : 0u;
    uint zl1 = (lane < 16u) ? zetas[128u + (sg << 5u) + 16u + lane] : 0u;
    uint seg = lane >> 1u;
    uint z0 = simd_shuffle(zl0, (ushort)seg);
    uint z1 = simd_shuffle(zl1, (ushort)seg);
    return simd_bfly2_3329(v0, v1, (ushort)1, ((lane & 1u) == 0u), z0, z1);
}

inline void ntt_256_3329_recompute_l7(device uint *poly,
                                       device const uint *zetas,
                                       uint ltid) {
    uint lane = ltid & 31u;
    uint sg   = ltid >> 5u;

    uint c0 = poly[lane +   0u];
    uint c1 = poly[lane +  32u];
    uint c2 = poly[lane +  64u];
    uint c3 = poly[lane +  96u];
    uint c4 = poly[lane + 128u];
    uint c5 = poly[lane + 160u];
    uint c6 = poly[lane + 192u];
    uint c7 = poly[lane + 224u];

    uint z1 = bcast_zeta_lane0(zetas, 1u, lane);

    uint x;
    uint y;

    if (sg < 2u) {
        uint a0 = add_3329(c0, mod_mul_3329(z1, c4));
        uint a1 = add_3329(c1, mod_mul_3329(z1, c5));
        uint a2 = add_3329(c2, mod_mul_3329(z1, c6));
        uint a3 = add_3329(c3, mod_mul_3329(z1, c7));

        uint z2 = bcast_zeta_lane0(zetas, 2u, lane);
        uint t0 = mod_mul_3329(z2, a2);
        uint t1 = mod_mul_3329(z2, a3);

        if (sg == 0u) {
            x = add_3329(a0, t0);
            y = add_3329(a1, t1);
        } else {
            x = sub_3329(a0, t0);
            y = sub_3329(a1, t1);
        }
    } else {
        uint a4 = sub_3329(c0, mod_mul_3329(z1, c4));
        uint a5 = sub_3329(c1, mod_mul_3329(z1, c5));
        uint a6 = sub_3329(c2, mod_mul_3329(z1, c6));
        uint a7 = sub_3329(c3, mod_mul_3329(z1, c7));

        uint z3 = bcast_zeta_lane0(zetas, 3u, lane);
        uint t0 = mod_mul_3329(z3, a6);
        uint t1 = mod_mul_3329(z3, a7);

        if (sg == 2u) {
            x = add_3329(a4, t0);
            y = add_3329(a5, t1);
        } else {
            x = sub_3329(a4, t0);
            y = sub_3329(a5, t1);
        }
    }

    uint z4 = bcast_zeta_lane0(zetas, 4u + sg, lane);
    uint t = mod_mul_3329(z4, y);

    uint v0 = add_3329(x, t);
    uint v1 = sub_3329(x, t);

    uint2 r = simd_stage16_3329(v0, v1, sg, lane, zetas);
    v0 = r.x; v1 = r.y;

    r = simd_stage8_3329(v0, v1, sg, lane, zetas);
    v0 = r.x; v1 = r.y;

    r = simd_stage4_3329(v0, v1, sg, lane, zetas);
    v0 = r.x; v1 = r.y;

    r = simd_stage2_3329(v0, v1, sg, lane, zetas);
    v0 = r.x; v1 = r.y;

    threadgroup_barrier(mem_flags::mem_none);

    uint base = sg << 6u;
    poly[base + lane]       = v0;
    poly[base + 32u + lane] = v1;
}

inline void ntt_256_3329_recompute_l8(device uint *poly,
                                       device const uint *zetas,
                                       uint ltid) {
    uint lane = ltid & 31u;
    uint sg   = ltid >> 5u;

    uint c0 = poly[lane +   0u];
    uint c1 = poly[lane +  32u];
    uint c2 = poly[lane +  64u];
    uint c3 = poly[lane +  96u];
    uint c4 = poly[lane + 128u];
    uint c5 = poly[lane + 160u];
    uint c6 = poly[lane + 192u];
    uint c7 = poly[lane + 224u];

    uint z1 = bcast_zeta_lane0(zetas, 1u, lane);

    uint x;
    uint y;

    if (sg < 2u) {
        uint a0 = add_3329(c0, mod_mul_3329(z1, c4));
        uint a1 = add_3329(c1, mod_mul_3329(z1, c5));
        uint a2 = add_3329(c2, mod_mul_3329(z1, c6));
        uint a3 = add_3329(c3, mod_mul_3329(z1, c7));

        uint z2 = bcast_zeta_lane0(zetas, 2u, lane);
        uint t0 = mod_mul_3329(z2, a2);
        uint t1 = mod_mul_3329(z2, a3);

        if (sg == 0u) {
            x = add_3329(a0, t0);
            y = add_3329(a1, t1);
        } else {
            x = sub_3329(a0, t0);
            y = sub_3329(a1, t1);
        }
    } else {
        uint a4 = sub_3329(c0, mod_mul_3329(z1, c4));
        uint a5 = sub_3329(c1, mod_mul_3329(z1, c5));
        uint a6 = sub_3329(c2, mod_mul_3329(z1, c6));
        uint a7 = sub_3329(c3, mod_mul_3329(z1, c7));

        uint z3 = bcast_zeta_lane0(zetas, 3u, lane);
        uint t0 = mod_mul_3329(z3, a6);
        uint t1 = mod_mul_3329(z3, a7);

        if (sg == 2u) {
            x = add_3329(a4, t0);
            y = add_3329(a5, t1);
        } else {
            x = sub_3329(a4, t0);
            y = sub_3329(a5, t1);
        }
    }

    uint z4 = bcast_zeta_lane0(zetas, 4u + sg, lane);
    uint t = mod_mul_3329(z4, y);

    uint v0 = add_3329(x, t);
    uint v1 = sub_3329(x, t);

    uint2 r = simd_stage16_3329(v0, v1, sg, lane, zetas);
    v0 = r.x; v1 = r.y;

    r = simd_stage8_3329(v0, v1, sg, lane, zetas);
    v0 = r.x; v1 = r.y;

    r = simd_stage4_3329(v0, v1, sg, lane, zetas);
    v0 = r.x; v1 = r.y;

    r = simd_stage2_3329(v0, v1, sg, lane, zetas);
    v0 = r.x; v1 = r.y;

    r = simd_stage1_3329(v0, v1, sg, lane, zetas);
    v0 = r.x; v1 = r.y;

    threadgroup_barrier(mem_flags::mem_none);

    uint base = sg << 6u;
    poly[base + lane]       = v0;
    poly[base + 32u + lane] = v1;
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
            uint t0 = 0u;                                                        \
            uint t1 = 0u;                                                        \
            if (lower) {                                                         \
                uint group0 = (base0 + lane) >> pair_shift;                      \
                uint group1 = (base1 + lane) >> pair_shift;                      \
                uint z0 = zs[k_start + group0];                                  \
                uint z1 = zs[k_start + group1];                                  \
                t0 = MUL(z0, p0, q);                                             \
                t1 = MUL(z1, p1, q);                                             \
            }                                                                    \
                                                                                 \
            uint sum0 = ADD(v0, t0, q);                                          \
            uint dif0 = SUB(v0, t0, q);                                          \
            uint sum1 = ADD(v1, t1, q);                                          \
            uint dif1 = SUB(v1, t1, q);                                          \
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

    device uint *poly = coeffs + (size_t)tgid * (size_t)n;

    if ((q == 3329u) && (n == 256u)) {
        if (n_levels == 7u) {
            ntt_256_3329_recompute_l7(poly, zetas, ltid);
            return;
        }
        if (n_levels == 8u) {
            ntt_256_3329_recompute_l8(poly, zetas, ltid);
            return;
        }
    }

    threadgroup uint a[N_MAX];
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