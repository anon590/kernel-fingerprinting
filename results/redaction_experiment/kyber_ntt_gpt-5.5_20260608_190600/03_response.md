I fuse the polynomial load with the first length-128 butterfly for the common n=256 paths, while still cooperatively staging runtime twiddles into threadgroup memory. This removes the incumbent’s initial full threadgroup round-trip and one threadgroup barrier. I also specialize add/sub reductions for q=3329 and q=8380417 so the hot paths avoid the generic overflow-aware modular add logic while preserving canonical outputs and runtime-loaded parameters.

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

inline uint mod_add_3329(uint a, uint b) {
    uint t = a + b;
    return (t >= 3329u) ? (t - 3329u) : t;
}

inline uint mod_sub_3329(uint a, uint b) {
    uint t = a - b;
    return (a < b) ? (t + 3329u) : t;
}

inline uint mod_add_8380417(uint a, uint b) {
    uint t = a + b;
    return (t >= 8380417u) ? (t - 8380417u) : t;
}

inline uint mod_sub_8380417(uint a, uint b) {
    uint t = a - b;
    return (a < b) ? (t + 8380417u) : t;
}

// Exact for q = 3329, a,b in [0,q), product < q^2.
// floor(2^32 / 3329) = 1290167.
inline uint mod_mul_3329(uint a, uint b) {
    uint x = a * b;
    uint qhat = mulhi(x, 1290167u);
    uint r = x - qhat * 3329u;
    return (r >= 3329u) ? (r - 3329u) : r;
}

// Exact for q = 8380417 = 2^23 - 2^13 + 1, so 2^23 == 8191 mod q.
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

#define MOD_MUL_3329(a,b)       mod_mul_3329((a), (b))
#define MOD_MUL_8380417(a,b)    mod_mul_8380417((a), (b))
#define MOD_MUL_GENERIC_Q(a,b)  mod_mul_generic((a), (b), q)
#define MOD_MUL_FAST_Q(a,b)     mod_mul_fast((a), (b), q)

#define MOD_ADD_3329(a,b)       mod_add_3329((a), (b))
#define MOD_SUB_3329(a,b)       mod_sub_3329((a), (b))
#define MOD_ADD_8380417(a,b)    mod_add_8380417((a), (b))
#define MOD_SUB_8380417(a,b)    mod_sub_8380417((a), (b))
#define MOD_ADD_Q(a,b)          mod_add_q((a), (b), q)
#define MOD_SUB_Q(a,b)          mod_sub_q((a), (b), q)

#define NTT_INIT_256_M(MUL, ADD, SUB) do {                                  \
    uint _lane = ltid & 31u;                                                 \
    uint _zload = (_lane == 0u) ? zetas[1u] : 0u;                            \
    uint _z = simd_shuffle(_zload, (ushort)0);                               \
    uint _x = poly[ltid];                                                    \
    uint _y = poly[ltid + 128u];                                             \
    uint _t = MUL(_z, _y);                                                   \
    a[ltid] = ADD(_x, _t);                                                   \
    a[ltid + 128u] = SUB(_x, _t);                                            \
} while (0)

#define NTT_STAGE_CT_M(LEN, LOGV, KSTART, MUL, ADD, SUB) do {                \
    uint _g = ltid >> (LOGV);                                                \
    uint _j = (_g << ((LOGV) + 1u)) | (ltid & ((LEN) - 1u));                 \
    uint _z = zt[(KSTART) + _g];                                             \
    uint _x = a[_j];                                                         \
    uint _y = a[_j + (LEN)];                                                 \
    uint _t = MUL(_z, _y);                                                   \
    a[_j] = ADD(_x, _t);                                                     \
    a[_j + (LEN)] = SUB(_x, _t);                                             \
} while (0)

#define NTT_FINAL_CT_M(LEN, LOGV, KSTART, MUL, ADD, SUB) do {                \
    uint _g = ltid >> (LOGV);                                                \
    uint _j = (_g << ((LOGV) + 1u)) | (ltid & ((LEN) - 1u));                 \
    uint _z = zt[(KSTART) + _g];                                             \
    uint _x = a[_j];                                                         \
    uint _y = a[_j + (LEN)];                                                 \
    uint _t = MUL(_z, _y);                                                   \
    poly[_j] = ADD(_x, _t);                                                  \
    poly[_j + (LEN)] = SUB(_x, _t);                                          \
} while (0)

#define RUN_NTT_256_7_FUSED(MUL, ADD, SUB) do {                              \
    NTT_INIT_256_M(MUL, ADD, SUB);                                           \
    threadgroup_barrier(mem_flags::mem_threadgroup);                         \
    NTT_STAGE_CT_M( 64u, 6u,  2u, MUL, ADD, SUB);                            \
    threadgroup_barrier(mem_flags::mem_threadgroup);                         \
    NTT_STAGE_CT_M( 32u, 5u,  4u, MUL, ADD, SUB);                            \
    simdgroup_barrier(mem_flags::mem_threadgroup);                           \
    NTT_STAGE_CT_M( 16u, 4u,  8u, MUL, ADD, SUB);                            \
    simdgroup_barrier(mem_flags::mem_threadgroup);                           \
    NTT_STAGE_CT_M(  8u, 3u, 16u, MUL, ADD, SUB);                            \
    simdgroup_barrier(mem_flags::mem_threadgroup);                           \
    NTT_STAGE_CT_M(  4u, 2u, 32u, MUL, ADD, SUB);                            \
    simdgroup_barrier(mem_flags::mem_threadgroup);                           \
    NTT_FINAL_CT_M(  2u, 1u, 64u, MUL, ADD, SUB);                            \
} while (0)

#define RUN_NTT_256_8_FUSED(MUL, ADD, SUB) do {                              \
    NTT_INIT_256_M(MUL, ADD, SUB);                                           \
    threadgroup_barrier(mem_flags::mem_threadgroup);                         \
    NTT_STAGE_CT_M( 64u, 6u,   2u, MUL, ADD, SUB);                           \
    threadgroup_barrier(mem_flags::mem_threadgroup);                         \
    NTT_STAGE_CT_M( 32u, 5u,   4u, MUL, ADD, SUB);                           \
    simdgroup_barrier(mem_flags::mem_threadgroup);                           \
    NTT_STAGE_CT_M( 16u, 4u,   8u, MUL, ADD, SUB);                           \
    simdgroup_barrier(mem_flags::mem_threadgroup);                           \
    NTT_STAGE_CT_M(  8u, 3u,  16u, MUL, ADD, SUB);                           \
    simdgroup_barrier(mem_flags::mem_threadgroup);                           \
    NTT_STAGE_CT_M(  4u, 2u,  32u, MUL, ADD, SUB);                           \
    simdgroup_barrier(mem_flags::mem_threadgroup);                           \
    NTT_STAGE_CT_M(  2u, 1u,  64u, MUL, ADD, SUB);                           \
    simdgroup_barrier(mem_flags::mem_threadgroup);                           \
    NTT_FINAL_CT_M(  1u, 0u, 128u, MUL, ADD, SUB);                           \
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
            RUN_NTT_256_7_FUSED(MOD_MUL_3329, MOD_ADD_3329, MOD_SUB_3329);
        } else if (q == 8380417u) {
            RUN_NTT_256_7_FUSED(MOD_MUL_8380417, MOD_ADD_8380417, MOD_SUB_8380417);
        } else {
            RUN_NTT_256_7_FUSED(MOD_MUL_GENERIC_Q, MOD_ADD_Q, MOD_SUB_Q);
        }
        return;
    }

    if (n == 256u && n_levels == 8u) {
        device uint *poly = coeffs + ((size_t)tgid << 8);

        zt[ltid] = zetas[ltid];
        zt[ltid + 128u] = zetas[ltid + 128u];

        if (q == 3329u) {
            RUN_NTT_256_8_FUSED(MOD_MUL_3329, MOD_ADD_3329, MOD_SUB_3329);
        } else if (q == 8380417u) {
            RUN_NTT_256_8_FUSED(MOD_MUL_8380417, MOD_ADD_8380417, MOD_SUB_8380417);
        } else {
            RUN_NTT_256_8_FUSED(MOD_MUL_GENERIC_Q, MOD_ADD_Q, MOD_SUB_Q);
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
```