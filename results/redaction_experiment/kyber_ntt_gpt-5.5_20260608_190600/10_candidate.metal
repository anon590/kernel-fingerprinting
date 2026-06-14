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

// Exact for q = 3329 for b < 9*q, covering the lazy 256-point paths.
inline uint mod_mul_3329(uint a, uint b) {
    uint x = a * b;
    uint qhat = mulhi(x, 1290167u); // floor(2^32 / 3329)
    uint r = x - qhat * 3329u;
    return (r >= 3329u) ? (r - 3329u) : r;
}

// Reduce x < 9*3329 to canonical [0, 3329).
inline uint reduce_3329_small(uint x) {
    if (x >= 26632u) x -= 26632u;
    if (x >= 13316u) x -= 13316u;
    if (x >=  6658u) x -=  6658u;
    if (x >=  3329u) x -=  3329u;
    return x;
}

// q = 8380417 = 2^23 - 2^13 + 1, canonical inputs.
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

#define MOD_MUL_8380417(a,b)   mod_mul_8380417((a), (b))
#define MOD_MUL_GENERIC_Q(a,b) mod_mul_generic((a), (b), q)
#define MOD_ADD_Q(a,b)         mod_add_q((a), (b), q)
#define MOD_SUB_Q(a,b)         mod_sub_q((a), (b), q)
#define MOD_ADD_838(a,b)       mod_add_8380417((a), (b))
#define MOD_SUB_838(a,b)       mod_sub_8380417((a), (b))

#define INIT_128_DIRECT(MUL, ADD, SUB) do {                              \
    uint _lane = ltid & 31u;                                              \
    uint _zl = (_lane == 0u) ? zetas[1u] : 0u;                            \
    uint _z = simd_shuffle(_zl, (ushort)0);                               \
    uint _x = poly[ltid];                                                 \
    uint _y = poly[ltid + 128u];                                          \
    uint _t = MUL(_z, _y);                                                \
    a[ltid] = ADD(_x, _t);                                                \
    a[ltid + 128u] = SUB(_x, _t);                                         \
} while (0)

#define INIT_128_3329_LAZY() do {                                         \
    uint _lane = ltid & 31u;                                              \
    uint _zl = (_lane == 0u) ? zetas[1u] : 0u;                            \
    uint _z = simd_shuffle(_zl, (ushort)0);                               \
    uint _x = poly[ltid];                                                 \
    uint _y = poly[ltid + 128u];                                          \
    uint _t = mod_mul_3329(_z, _y);                                       \
    a[ltid] = _x + _t;                                                    \
    a[ltid + 128u] = _x + 3329u - _t;                                     \
} while (0)

#define STAGE64_3329_LAZY() do {                                          \
    uint _g = ltid >> 6u;                                                 \
    uint _j = (_g << 7u) | (ltid & 63u);                                  \
    uint _z = zt[2u + _g];                                                \
    uint _x = a[_j];                                                      \
    uint _y = a[_j + 64u];                                                \
    uint _t = mod_mul_3329(_z, _y);                                       \
    a[_j] = _x + _t;                                                      \
    a[_j + 64u] = _x + 3329u - _t;                                        \
} while (0)

#define REG_STAGE32_3329() do {                                           \
    uint _z = zt[4u + (ltid >> 5u)];                                      \
    uint _x = r0;                                                         \
    uint _t = mod_mul_3329(_z, r1);                                       \
    r0 = _x + _t;                                                         \
    r1 = _x + 3329u - _t;                                                 \
} while (0)

#define REG_STAGE16_3329() do {                                           \
    uint _lane = ltid & 31u;                                              \
    uint _z = zt[8u + (ltid >> 4u)];                                      \
    uint _s = 0u;                                                         \
    uint _d = 0u;                                                         \
    if (_lane < 16u) {                                                    \
        uint _x = r0;                                                     \
        uint _y = simd_shuffle(r0, (ushort)(_lane + 16u));                \
        uint _t = mod_mul_3329(_z, _y);                                   \
        _s = _x + _t;                                                     \
        _d = _x + 3329u - _t;                                             \
    } else {                                                             \
        uint _x = simd_shuffle(r1, (ushort)(_lane - 16u));                \
        uint _y = r1;                                                     \
        uint _t = mod_mul_3329(_z, _y);                                   \
        _s = _x + _t;                                                     \
        _d = _x + 3329u - _t;                                             \
    }                                                                     \
    uint _r0u = simd_shuffle(_d, (ushort)(_lane & 15u));                  \
    uint _r1l = simd_shuffle(_s, (ushort)((_lane & 15u) + 16u));          \
    r0 = (_lane < 16u) ? _s : _r0u;                                       \
    r1 = (_lane < 16u) ? _r1l : _d;                                       \
} while (0)

#define REG_STAGE_GENERIC_3329(L, LOGV, KSTART) do {                      \
    uint _lane = ltid & 31u;                                              \
    uint _px = ((_lane >> (LOGV)) << ((LOGV) + 1u)) |                    \
               (_lane & ((L) - 1u));                                      \
    uint _py = _px + (L);                                                 \
    uint _xv = (_px < 32u) ?                                             \
        simd_shuffle(r0, (ushort)_px) :                                  \
        simd_shuffle(r1, (ushort)(_px - 32u));                            \
    uint _yv = (_py < 32u) ?                                             \
        simd_shuffle(r0, (ushort)_py) :                                  \
        simd_shuffle(r1, (ushort)(_py - 32u));                            \
    uint _z = zt[(KSTART) + (ltid >> (LOGV))];                            \
    uint _t = mod_mul_3329(_z, _yv);                                      \
    uint _s = _xv + _t;                                                   \
    uint _d = _xv + 3329u - _t;                                           \
    uint _p0 = _lane;                                                     \
    uint _p1 = _lane + 32u;                                               \
    uint _owner0 = ((_p0 >> ((LOGV) + 1u)) << (LOGV)) |                  \
                   (_p0 & ((L) - 1u));                                    \
    uint _owner1 = ((_p1 >> ((LOGV) + 1u)) << (LOGV)) |                  \
                   (_p1 & ((L) - 1u));                                    \
    bool _lo0 = ((_p0 & (((L) << 1u) - 1u)) < (L));                      \
    bool _lo1 = ((_p1 & (((L) << 1u) - 1u)) < (L));                      \
    uint _s0 = simd_shuffle(_s, (ushort)_owner0);                         \
    uint _d0 = simd_shuffle(_d, (ushort)_owner0);                         \
    uint _s1 = simd_shuffle(_s, (ushort)_owner1);                         \
    uint _d1 = simd_shuffle(_d, (ushort)_owner1);                         \
    r0 = _lo0 ? _s0 : _d0;                                                \
    r1 = _lo1 ? _s1 : _d1;                                                \
} while (0)

#define RUN_256_7_3329_REG() do {                                         \
    INIT_128_3329_LAZY();                                                 \
    threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    STAGE64_3329_LAZY();                                                  \
    threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    uint _lane_r = ltid & 31u;                                            \
    uint _base_r = (ltid >> 5u) << 6u;                                    \
    uint r0 = a[_base_r + _lane_r];                                       \
    uint r1 = a[_base_r + _lane_r + 32u];                                 \
    REG_STAGE32_3329();                                                   \
    REG_STAGE16_3329();                                                   \
    REG_STAGE_GENERIC_3329( 8u, 3u, 16u);                                 \
    REG_STAGE_GENERIC_3329( 4u, 2u, 32u);                                 \
    REG_STAGE_GENERIC_3329( 2u, 1u, 64u);                                 \
    poly[_base_r + _lane_r] = reduce_3329_small(r0);                      \
    poly[_base_r + _lane_r + 32u] = reduce_3329_small(r1);                \
} while (0)

#define RUN_256_8_3329_REG() do {                                         \
    INIT_128_3329_LAZY();                                                 \
    threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    STAGE64_3329_LAZY();                                                  \
    threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    uint _lane_r = ltid & 31u;                                            \
    uint _base_r = (ltid >> 5u) << 6u;                                    \
    uint r0 = a[_base_r + _lane_r];                                       \
    uint r1 = a[_base_r + _lane_r + 32u];                                 \
    REG_STAGE32_3329();                                                   \
    REG_STAGE16_3329();                                                   \
    REG_STAGE_GENERIC_3329( 8u, 3u,  16u);                                \
    REG_STAGE_GENERIC_3329( 4u, 2u,  32u);                                \
    REG_STAGE_GENERIC_3329( 2u, 1u,  64u);                                \
    REG_STAGE_GENERIC_3329( 1u, 0u, 128u);                                \
    poly[_base_r + _lane_r] = reduce_3329_small(r0);                      \
    poly[_base_r + _lane_r + 32u] = reduce_3329_small(r1);                \
} while (0)

#define NTT_STAGE_CT_M(LEN, LOGV, KSTART, MUL) do {                       \
    uint _g = ltid >> (LOGV);                                             \
    uint _j = (_g << ((LOGV) + 1u)) | (ltid & ((LEN) - 1u));              \
    uint _z = zt[(KSTART) + _g];                                          \
    uint _x = a[_j];                                                      \
    uint _y = a[_j + (LEN)];                                              \
    uint _t = MUL(_z, _y);                                                \
    a[_j] = mod_add_q(_x, _t, q);                                         \
    a[_j + (LEN)] = mod_sub_q(_x, _t, q);                                 \
} while (0)

#define NTT_FINAL_CT_M(LEN, LOGV, KSTART, MUL) do {                       \
    uint _g = ltid >> (LOGV);                                             \
    uint _j = (_g << ((LOGV) + 1u)) | (ltid & ((LEN) - 1u));              \
    uint _z = zt[(KSTART) + _g];                                          \
    uint _x = a[_j];                                                      \
    uint _y = a[_j + (LEN)];                                              \
    uint _t = MUL(_z, _y);                                                \
    poly[_j] = mod_add_q(_x, _t, q);                                      \
    poly[_j + (LEN)] = mod_sub_q(_x, _t, q);                              \
} while (0)

#define NTT_STAGE_838(LEN, LOGV, KSTART) do {                             \
    uint _g = ltid >> (LOGV);                                             \
    uint _j = (_g << ((LOGV) + 1u)) | (ltid & ((LEN) - 1u));              \
    uint _z = zt[(KSTART) + _g];                                          \
    uint _x = a[_j];                                                      \
    uint _y = a[_j + (LEN)];                                              \
    uint _t = mod_mul_8380417(_z, _y);                                    \
    a[_j] = mod_add_8380417(_x, _t);                                      \
    a[_j + (LEN)] = mod_sub_8380417(_x, _t);                              \
} while (0)

#define NTT_FINAL_838(LEN, LOGV, KSTART) do {                             \
    uint _g = ltid >> (LOGV);                                             \
    uint _j = (_g << ((LOGV) + 1u)) | (ltid & ((LEN) - 1u));              \
    uint _z = zt[(KSTART) + _g];                                          \
    uint _x = a[_j];                                                      \
    uint _y = a[_j + (LEN)];                                              \
    uint _t = mod_mul_8380417(_z, _y);                                    \
    poly[_j] = mod_add_8380417(_x, _t);                                   \
    poly[_j + (LEN)] = mod_sub_8380417(_x, _t);                           \
} while (0)

#define RUN_REST_256_7_GENERIC(MUL) do {                                  \
    threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    NTT_STAGE_CT_M( 64u, 6u,  2u, MUL);                                   \
    threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    NTT_STAGE_CT_M( 32u, 5u,  4u, MUL);                                   \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_STAGE_CT_M( 16u, 4u,  8u, MUL);                                   \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_STAGE_CT_M(  8u, 3u, 16u, MUL);                                   \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_STAGE_CT_M(  4u, 2u, 32u, MUL);                                   \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_FINAL_CT_M(  2u, 1u, 64u, MUL);                                   \
} while (0)

#define RUN_REST_256_8_GENERIC(MUL) do {                                  \
    threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    NTT_STAGE_CT_M( 64u, 6u,   2u, MUL);                                  \
    threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    NTT_STAGE_CT_M( 32u, 5u,   4u, MUL);                                  \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_STAGE_CT_M( 16u, 4u,   8u, MUL);                                  \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_STAGE_CT_M(  8u, 3u,  16u, MUL);                                  \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_STAGE_CT_M(  4u, 2u,  32u, MUL);                                  \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_STAGE_CT_M(  2u, 1u,  64u, MUL);                                  \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_FINAL_CT_M(  1u, 0u, 128u, MUL);                                  \
} while (0)

#define RUN_REST_256_7_838() do {                                         \
    threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    NTT_STAGE_838( 64u, 6u,  2u);                                         \
    threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    NTT_STAGE_838( 32u, 5u,  4u);                                         \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_STAGE_838( 16u, 4u,  8u);                                         \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_STAGE_838(  8u, 3u, 16u);                                         \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_STAGE_838(  4u, 2u, 32u);                                         \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_FINAL_838(  2u, 1u, 64u);                                         \
} while (0)

#define RUN_REST_256_8_838() do {                                         \
    threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    NTT_STAGE_838( 64u, 6u,   2u);                                        \
    threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    NTT_STAGE_838( 32u, 5u,   4u);                                        \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_STAGE_838( 16u, 4u,   8u);                                        \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_STAGE_838(  8u, 3u,  16u);                                        \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_STAGE_838(  4u, 2u,  32u);                                        \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_STAGE_838(  2u, 1u,  64u);                                        \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_FINAL_838(  1u, 0u, 128u);                                        \
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

    if (n == 256u && n_levels == 7u) {
        device uint *poly = coeffs + ((size_t)tgid << 8);

        zt[ltid] = zetas[ltid];

        if (q == 3329u) {
            RUN_256_7_3329_REG();
        } else if (q == 8380417u) {
            INIT_128_DIRECT(MOD_MUL_8380417, MOD_ADD_838, MOD_SUB_838);
            RUN_REST_256_7_838();
        } else {
            INIT_128_DIRECT(MOD_MUL_GENERIC_Q, MOD_ADD_Q, MOD_SUB_Q);
            RUN_REST_256_7_GENERIC(MOD_MUL_GENERIC_Q);
        }
        return;
    }

    if (n == 256u && n_levels == 8u) {
        device uint *poly = coeffs + ((size_t)tgid << 8);

        zt[ltid] = zetas[ltid];
        zt[ltid + 128u] = zetas[ltid + 128u];

        if (q == 3329u) {
            RUN_256_8_3329_REG();
        } else if (q == 8380417u) {
            INIT_128_DIRECT(MOD_MUL_8380417, MOD_ADD_838, MOD_SUB_838);
            RUN_REST_256_8_838();
        } else {
            INIT_128_DIRECT(MOD_MUL_GENERIC_Q, MOD_ADD_Q, MOD_SUB_Q);
            RUN_REST_256_8_GENERIC(MOD_MUL_GENERIC_Q);
        }
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

        uint x = a[j];
        uint y = a[j + length];
        uint t = mod_mul_fast(zeta, y, q);

        a[j] = mod_add_q(x, t, q);
        a[j + length] = mod_sub_q(x, t, q);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length >>= 1u;
    }

    uint group_idx = ltid / length;
    uint j_in_group = ltid - group_idx * length;
    uint j = (group_idx << 1u) * length + j_in_group;
    uint zeta = zt[k_start + group_idx];

    uint x = a[j];
    uint y = a[j + length];
    uint t = mod_mul_fast(zeta, y, q);

    poly[j] = mod_add_q(x, t, q);
    poly[j + length] = mod_sub_q(x, t, q);
}