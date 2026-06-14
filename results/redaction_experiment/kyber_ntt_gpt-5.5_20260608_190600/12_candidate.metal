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

// Exact for q = 3329 for the lazy ranges used here: b < 9*q.
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

#define MOD_MUL_3329(a,b)      mod_mul_3329((a), (b))
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

#define INIT_128_3329_LAZY_DIRECT() do {                                  \
    uint _lane = ltid & 31u;                                              \
    uint _zl = (_lane == 0u) ? zetas[1u] : 0u;                            \
    uint _z = simd_shuffle(_zl, (ushort)0);                               \
    uint _x = poly[ltid];                                                 \
    uint _y = poly[ltid + 128u];                                          \
    uint _t = mod_mul_3329(_z, _y);                                       \
    a[ltid] = _x + _t;                                                    \
    a[ltid + 128u] = _x + 3329u - _t;                                     \
} while (0)

#define STAGE64_DIRECT(MUL, ADD, SUB) do {                                \
    uint _lane = ltid & 31u;                                              \
    uint _g = ltid >> 6u;                                                 \
    uint _j = (_g << 7u) | (ltid & 63u);                                  \
    uint _zl = (_lane == 0u) ? zetas[2u + _g] : 0u;                       \
    uint _z = simd_shuffle(_zl, (ushort)0);                               \
    uint _x = a[_j];                                                      \
    uint _y = a[_j + 64u];                                                \
    uint _t = MUL(_z, _y);                                                \
    a[_j] = ADD(_x, _t);                                                  \
    a[_j + 64u] = SUB(_x, _t);                                            \
} while (0)

#define STAGE64_3329_LAZY_DIRECT() do {                                   \
    uint _lane = ltid & 31u;                                              \
    uint _g = ltid >> 6u;                                                 \
    uint _j = (_g << 7u) | (ltid & 63u);                                  \
    uint _zl = (_lane == 0u) ? zetas[2u + _g] : 0u;                       \
    uint _z = simd_shuffle(_zl, (ushort)0);                               \
    uint _x = a[_j];                                                      \
    uint _y = a[_j + 64u];                                                \
    uint _t = mod_mul_3329(_z, _y);                                       \
    a[_j] = _x + _t;                                                      \
    a[_j + 64u] = _x + 3329u - _t;                                        \
} while (0)

#define REG_STEP_3329(HALF, LOGV, KSTART) do {                            \
    bool _lower = ((lane & (HALF)) == 0u);                                \
    uint _idx = _lower ? (lane + (HALF)) : (lane - (HALF));               \
    uint2 _lh = uint2(lo, hi);                                            \
    uint2 _peer = simd_shuffle(_lh, (ushort)_idx);                        \
    uint _x = _lower ? lo : _peer.y;                                      \
    uint _y = _lower ? _peer.x : hi;                                      \
    uint _z = zetas[(KSTART) + (ltid >> (LOGV))];                         \
    uint _t = mod_mul_3329(_z, _y);                                       \
    lo = _x + _t;                                                         \
    hi = _x + 3329u - _t;                                                 \
} while (0)

#define REG_STEP_CANON(HALF, LOGV, KSTART, MUL, ADD, SUB) do {            \
    bool _lower = ((lane & (HALF)) == 0u);                                \
    uint _idx = _lower ? (lane + (HALF)) : (lane - (HALF));               \
    uint2 _lh = uint2(lo, hi);                                            \
    uint2 _peer = simd_shuffle(_lh, (ushort)_idx);                        \
    uint _x = _lower ? lo : _peer.y;                                      \
    uint _y = _lower ? _peer.x : hi;                                      \
    uint _z = zetas[(KSTART) + (ltid >> (LOGV))];                         \
    uint _t = MUL(_z, _y);                                                \
    lo = ADD(_x, _t);                                                     \
    hi = SUB(_x, _t);                                                     \
} while (0)

#define TAIL32_TO_2_3329_LAZY() do {                                      \
    uint lane = ltid & 31u;                                               \
    uint sg = ltid >> 5u;                                                 \
    uint base = sg << 6u;                                                 \
    uint _zl = (lane == 0u) ? zetas[4u + sg] : 0u;                        \
    uint _z = simd_shuffle(_zl, (ushort)0);                               \
    uint _x = a[base + lane];                                             \
    uint _y = a[base + lane + 32u];                                       \
    uint _t = mod_mul_3329(_z, _y);                                       \
    uint lo = _x + _t;                                                    \
    uint hi = _x + 3329u - _t;                                            \
    REG_STEP_3329(16u, 4u,  8u);                                          \
    REG_STEP_3329( 8u, 3u, 16u);                                          \
    REG_STEP_3329( 4u, 2u, 32u);                                          \
    REG_STEP_3329( 2u, 1u, 64u);                                          \
    uint _g = ltid >> 1u;                                                 \
    uint _j = (_g << 2u) | (ltid & 1u);                                   \
    poly[_j] = reduce_3329_small(lo);                                     \
    poly[_j + 2u] = reduce_3329_small(hi);                                \
} while (0)

#define TAIL32_TO_1_3329_LAZY() do {                                      \
    uint lane = ltid & 31u;                                               \
    uint sg = ltid >> 5u;                                                 \
    uint base = sg << 6u;                                                 \
    uint _zl = (lane == 0u) ? zetas[4u + sg] : 0u;                        \
    uint _z = simd_shuffle(_zl, (ushort)0);                               \
    uint _x = a[base + lane];                                             \
    uint _y = a[base + lane + 32u];                                       \
    uint _t = mod_mul_3329(_z, _y);                                       \
    uint lo = _x + _t;                                                    \
    uint hi = _x + 3329u - _t;                                            \
    REG_STEP_3329(16u, 4u,   8u);                                         \
    REG_STEP_3329( 8u, 3u,  16u);                                         \
    REG_STEP_3329( 4u, 2u,  32u);                                         \
    REG_STEP_3329( 2u, 1u,  64u);                                         \
    REG_STEP_3329( 1u, 0u, 128u);                                         \
    uint _j = ltid << 1u;                                                 \
    poly[_j] = reduce_3329_small(lo);                                     \
    poly[_j + 1u] = reduce_3329_small(hi);                                \
} while (0)

#define TAIL32_TO_2_CANON(MUL, ADD, SUB) do {                             \
    uint lane = ltid & 31u;                                               \
    uint sg = ltid >> 5u;                                                 \
    uint base = sg << 6u;                                                 \
    uint _zl = (lane == 0u) ? zetas[4u + sg] : 0u;                        \
    uint _z = simd_shuffle(_zl, (ushort)0);                               \
    uint _x = a[base + lane];                                             \
    uint _y = a[base + lane + 32u];                                       \
    uint _t = MUL(_z, _y);                                                \
    uint lo = ADD(_x, _t);                                                \
    uint hi = SUB(_x, _t);                                                \
    REG_STEP_CANON(16u, 4u,  8u, MUL, ADD, SUB);                          \
    REG_STEP_CANON( 8u, 3u, 16u, MUL, ADD, SUB);                          \
    REG_STEP_CANON( 4u, 2u, 32u, MUL, ADD, SUB);                          \
    REG_STEP_CANON( 2u, 1u, 64u, MUL, ADD, SUB);                          \
    uint _g = ltid >> 1u;                                                 \
    uint _j = (_g << 2u) | (ltid & 1u);                                   \
    poly[_j] = lo;                                                        \
    poly[_j + 2u] = hi;                                                   \
} while (0)

#define TAIL32_TO_1_CANON(MUL, ADD, SUB) do {                             \
    uint lane = ltid & 31u;                                               \
    uint sg = ltid >> 5u;                                                 \
    uint base = sg << 6u;                                                 \
    uint _zl = (lane == 0u) ? zetas[4u + sg] : 0u;                        \
    uint _z = simd_shuffle(_zl, (ushort)0);                               \
    uint _x = a[base + lane];                                             \
    uint _y = a[base + lane + 32u];                                       \
    uint _t = MUL(_z, _y);                                                \
    uint lo = ADD(_x, _t);                                                \
    uint hi = SUB(_x, _t);                                                \
    REG_STEP_CANON(16u, 4u,   8u, MUL, ADD, SUB);                         \
    REG_STEP_CANON( 8u, 3u,  16u, MUL, ADD, SUB);                         \
    REG_STEP_CANON( 4u, 2u,  32u, MUL, ADD, SUB);                         \
    REG_STEP_CANON( 2u, 1u,  64u, MUL, ADD, SUB);                         \
    REG_STEP_CANON( 1u, 0u, 128u, MUL, ADD, SUB);                         \
    uint _j = ltid << 1u;                                                 \
    poly[_j] = lo;                                                        \
    poly[_j + 1u] = hi;                                                   \
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

    if (n == 256u && (n_levels == 7u || n_levels == 8u)) {
        device uint *poly = coeffs + ((size_t)tgid << 8);
        bool full8 = (n_levels == 8u);

        if (q == 3329u) {
            INIT_128_3329_LAZY_DIRECT();
            threadgroup_barrier(mem_flags::mem_threadgroup);
            STAGE64_3329_LAZY_DIRECT();
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (full8) {
                TAIL32_TO_1_3329_LAZY();
            } else {
                TAIL32_TO_2_3329_LAZY();
            }
            return;
        }

        if (q == 8380417u) {
            INIT_128_DIRECT(MOD_MUL_8380417, MOD_ADD_838, MOD_SUB_838);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            STAGE64_DIRECT(MOD_MUL_8380417, MOD_ADD_838, MOD_SUB_838);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (full8) {
                TAIL32_TO_1_CANON(MOD_MUL_8380417, MOD_ADD_838, MOD_SUB_838);
            } else {
                TAIL32_TO_2_CANON(MOD_MUL_8380417, MOD_ADD_838, MOD_SUB_838);
            }
            return;
        }

        INIT_128_DIRECT(MOD_MUL_GENERIC_Q, MOD_ADD_Q, MOD_SUB_Q);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        STAGE64_DIRECT(MOD_MUL_GENERIC_Q, MOD_ADD_Q, MOD_SUB_Q);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (full8) {
            TAIL32_TO_1_CANON(MOD_MUL_GENERIC_Q, MOD_ADD_Q, MOD_SUB_Q);
        } else {
            TAIL32_TO_2_CANON(MOD_MUL_GENERIC_Q, MOD_ADD_Q, MOD_SUB_Q);
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