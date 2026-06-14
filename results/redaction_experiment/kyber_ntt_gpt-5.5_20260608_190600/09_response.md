This version specializes the 256-point runtime-checked paths without staging the twiddle table into threadgroup memory. For Kyber’s common `q=3329` case it keeps lazy residues across all stages, unrolls the stage schedule, and reads twiddles directly from the runtime `zetas` buffer. This removes the incumbent’s extra threadgroup twiddle array traffic and should improve occupancy/cache behavior while preserving the same synchronization pattern and canonical final reduction.  

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

#define MOD_ADD_LAZY_3329(a,b) ((a) + (b))
#define MOD_SUB_LAZY_3329(a,b) ((a) + 3329u - (b))

#define INIT_128_DIRECT_D(MUL, ADD, SUB) do {                            \
    uint _z = zetas[1u];                                                  \
    uint _x = poly[ltid];                                                 \
    uint _y = poly[ltid + 128u];                                          \
    uint _t = MUL(_z, _y);                                                \
    a[ltid] = ADD(_x, _t);                                                \
    a[ltid + 128u] = SUB(_x, _t);                                         \
} while (0)

#define NTT_STAGE_CT_D(LEN, LOGV, KSTART, MUL) do {                       \
    uint _g = ltid >> (LOGV);                                             \
    uint _j = (_g << ((LOGV) + 1u)) | (ltid & ((LEN) - 1u));              \
    uint _z = zetas[(KSTART) + _g];                                       \
    uint _x = a[_j];                                                      \
    uint _y = a[_j + (LEN)];                                              \
    uint _t = MUL(_z, _y);                                                \
    a[_j] = mod_add_q(_x, _t, q);                                         \
    a[_j + (LEN)] = mod_sub_q(_x, _t, q);                                 \
} while (0)

#define NTT_FINAL_CT_D(LEN, LOGV, KSTART, MUL) do {                       \
    uint _g = ltid >> (LOGV);                                             \
    uint _j = (_g << ((LOGV) + 1u)) | (ltid & ((LEN) - 1u));              \
    uint _z = zetas[(KSTART) + _g];                                       \
    uint _x = a[_j];                                                      \
    uint _y = a[_j + (LEN)];                                              \
    uint _t = MUL(_z, _y);                                                \
    poly[_j] = mod_add_q(_x, _t, q);                                      \
    poly[_j + (LEN)] = mod_sub_q(_x, _t, q);                              \
} while (0)

#define NTT_STAGE_838_D(LEN, LOGV, KSTART) do {                           \
    uint _g = ltid >> (LOGV);                                             \
    uint _j = (_g << ((LOGV) + 1u)) | (ltid & ((LEN) - 1u));              \
    uint _z = zetas[(KSTART) + _g];                                       \
    uint _x = a[_j];                                                      \
    uint _y = a[_j + (LEN)];                                              \
    uint _t = mod_mul_8380417(_z, _y);                                    \
    a[_j] = mod_add_8380417(_x, _t);                                      \
    a[_j + (LEN)] = mod_sub_8380417(_x, _t);                              \
} while (0)

#define NTT_FINAL_838_D(LEN, LOGV, KSTART) do {                           \
    uint _g = ltid >> (LOGV);                                             \
    uint _j = (_g << ((LOGV) + 1u)) | (ltid & ((LEN) - 1u));              \
    uint _z = zetas[(KSTART) + _g];                                       \
    uint _x = a[_j];                                                      \
    uint _y = a[_j + (LEN)];                                              \
    uint _t = mod_mul_8380417(_z, _y);                                    \
    poly[_j] = mod_add_8380417(_x, _t);                                   \
    poly[_j + (LEN)] = mod_sub_8380417(_x, _t);                           \
} while (0)

#define NTT_STAGE_3329_LAZY_D(LEN, LOGV, KSTART) do {                     \
    uint _g = ltid >> (LOGV);                                             \
    uint _j = (_g << ((LOGV) + 1u)) | (ltid & ((LEN) - 1u));              \
    uint _z = zetas[(KSTART) + _g];                                       \
    uint _x = a[_j];                                                      \
    uint _y = a[_j + (LEN)];                                              \
    uint _t = mod_mul_3329(_z, _y);                                       \
    a[_j] = _x + _t;                                                      \
    a[_j + (LEN)] = _x + 3329u - _t;                                      \
} while (0)

#define NTT_FINAL_3329_LAZY_D(LEN, LOGV, KSTART) do {                     \
    uint _g = ltid >> (LOGV);                                             \
    uint _j = (_g << ((LOGV) + 1u)) | (ltid & ((LEN) - 1u));              \
    uint _z = zetas[(KSTART) + _g];                                       \
    uint _x = a[_j];                                                      \
    uint _y = a[_j + (LEN)];                                              \
    uint _t = mod_mul_3329(_z, _y);                                       \
    poly[_j] = reduce_3329_small(_x + _t);                                \
    poly[_j + (LEN)] = reduce_3329_small(_x + 3329u - _t);                \
} while (0)

#define RUN_REST_256_7_GENERIC_D(MUL) do {                                \
    threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    NTT_STAGE_CT_D( 64u, 6u,  2u, MUL);                                   \
    threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    NTT_STAGE_CT_D( 32u, 5u,  4u, MUL);                                   \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_STAGE_CT_D( 16u, 4u,  8u, MUL);                                   \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_STAGE_CT_D(  8u, 3u, 16u, MUL);                                   \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_STAGE_CT_D(  4u, 2u, 32u, MUL);                                   \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_FINAL_CT_D(  2u, 1u, 64u, MUL);                                   \
} while (0)

#define RUN_REST_256_8_GENERIC_D(MUL) do {                                \
    threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    NTT_STAGE_CT_D( 64u, 6u,   2u, MUL);                                  \
    threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    NTT_STAGE_CT_D( 32u, 5u,   4u, MUL);                                  \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_STAGE_CT_D( 16u, 4u,   8u, MUL);                                  \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_STAGE_CT_D(  8u, 3u,  16u, MUL);                                  \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_STAGE_CT_D(  4u, 2u,  32u, MUL);                                  \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_STAGE_CT_D(  2u, 1u,  64u, MUL);                                  \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_FINAL_CT_D(  1u, 0u, 128u, MUL);                                  \
} while (0)

#define RUN_REST_256_7_838_D() do {                                       \
    threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    NTT_STAGE_838_D( 64u, 6u,  2u);                                       \
    threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    NTT_STAGE_838_D( 32u, 5u,  4u);                                       \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_STAGE_838_D( 16u, 4u,  8u);                                       \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_STAGE_838_D(  8u, 3u, 16u);                                       \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_STAGE_838_D(  4u, 2u, 32u);                                       \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_FINAL_838_D(  2u, 1u, 64u);                                       \
} while (0)

#define RUN_REST_256_8_838_D() do {                                       \
    threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    NTT_STAGE_838_D( 64u, 6u,   2u);                                      \
    threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    NTT_STAGE_838_D( 32u, 5u,   4u);                                      \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_STAGE_838_D( 16u, 4u,   8u);                                      \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_STAGE_838_D(  8u, 3u,  16u);                                      \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_STAGE_838_D(  4u, 2u,  32u);                                      \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_STAGE_838_D(  2u, 1u,  64u);                                      \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_FINAL_838_D(  1u, 0u, 128u);                                      \
} while (0)

#define RUN_REST_256_7_3329_LAZY_D() do {                                 \
    threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    NTT_STAGE_3329_LAZY_D( 64u, 6u,  2u);                                 \
    threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    NTT_STAGE_3329_LAZY_D( 32u, 5u,  4u);                                 \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_STAGE_3329_LAZY_D( 16u, 4u,  8u);                                 \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_STAGE_3329_LAZY_D(  8u, 3u, 16u);                                 \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_STAGE_3329_LAZY_D(  4u, 2u, 32u);                                 \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_FINAL_3329_LAZY_D(  2u, 1u, 64u);                                 \
} while (0)

#define RUN_REST_256_8_3329_LAZY_D() do {                                 \
    threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    NTT_STAGE_3329_LAZY_D( 64u, 6u,   2u);                                \
    threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    NTT_STAGE_3329_LAZY_D( 32u, 5u,   4u);                                \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_STAGE_3329_LAZY_D( 16u, 4u,   8u);                                \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_STAGE_3329_LAZY_D(  8u, 3u,  16u);                                \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_STAGE_3329_LAZY_D(  4u, 2u,  32u);                                \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_STAGE_3329_LAZY_D(  2u, 1u,  64u);                                \
    simdgroup_barrier(mem_flags::mem_threadgroup);                        \
    NTT_FINAL_3329_LAZY_D(  1u, 0u, 128u);                                \
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

    if (n == 256u && n_levels == 7u) {
        device uint *poly = coeffs + ((size_t)tgid << 8);

        if (q == 3329u) {
            INIT_128_DIRECT_D(MOD_MUL_3329, MOD_ADD_LAZY_3329, MOD_SUB_LAZY_3329);
            RUN_REST_256_7_3329_LAZY_D();
        } else if (q == 8380417u) {
            INIT_128_DIRECT_D(MOD_MUL_8380417, MOD_ADD_838, MOD_SUB_838);
            RUN_REST_256_7_838_D();
        } else {
            INIT_128_DIRECT_D(MOD_MUL_GENERIC_Q, MOD_ADD_Q, MOD_SUB_Q);
            RUN_REST_256_7_GENERIC_D(MOD_MUL_GENERIC_Q);
        }
        return;
    }

    if (n == 256u && n_levels == 8u) {
        device uint *poly = coeffs + ((size_t)tgid << 8);

        if (q == 3329u) {
            INIT_128_DIRECT_D(MOD_MUL_3329, MOD_ADD_LAZY_3329, MOD_SUB_LAZY_3329);
            RUN_REST_256_8_3329_LAZY_D();
        } else if (q == 8380417u) {
            INIT_128_DIRECT_D(MOD_MUL_8380417, MOD_ADD_838, MOD_SUB_838);
            RUN_REST_256_8_838_D();
        } else {
            INIT_128_DIRECT_D(MOD_MUL_GENERIC_Q, MOD_ADD_Q, MOD_SUB_Q);
            RUN_REST_256_8_GENERIC_D(MOD_MUL_GENERIC_Q);
        }
        return;
    }

    uint half_n = n >> 1u;
    device uint *poly = coeffs + (size_t)tgid * (size_t)n;

    a[ltid] = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];

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

        uint zeta = zetas[k_start + group_idx];

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

    uint zeta = zetas[k_start + group_idx];

    uint x = a[j];
    uint y = a[j + length];
    uint t = mod_mul_fast(zeta, y, q);

    poly[j] = mod_add_q(x, t, q);
    poly[j + length] = mod_sub_q(x, t, q);
}
```