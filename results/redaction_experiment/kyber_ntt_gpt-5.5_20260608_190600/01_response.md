Applied a fast path for the tested 256-point cases: zetas are cached once in threadgroup memory, stages are unrolled, known moduli use exact division-free reductions, and after the 32-length stage only simdgroup barriers are used because remaining butterflies are SIMD-local. This removes expensive `%` operations, loop/division overhead, repeated zeta loads, and several full threadgroup barriers while preserving the exact Cooley-Tukey mapping. 

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_MAX = 256u;

inline uint mod_add(uint a, uint b, uint q) {
    uint t = a + b;
    return (t >= q || t < a) ? (t - q) : t;
}

inline uint mod_sub(uint a, uint b, uint q) {
    return (a >= b) ? (a - b) : (a + q - b);
}

// Exact for q = 3329, with a,b in [0,q).  Product fits in uint.
// floor(2^32 / 3329) = 1290167.  For x < q^2, quotient error is at most 1.
inline uint mod_mul_3329(uint a, uint b) {
    uint x = a * b;
    uint qhat = mulhi(x, 1290167u);
    uint r = x - qhat * 3329u;
    return (r >= 3329u) ? (r - 3329u) : r;
}

// Exact for q = 8380417 = 2^23 - 2^13 + 1.
// Therefore 2^23 == 8191 (mod q). Product is < q^2 < 2^46.
inline uint mod_mul_8380417(uint a, uint b) {
    ulong x = (ulong)a * (ulong)b;

    ulong r1 = (x & (ulong)0x7fffffu) + (x >> 23) * (ulong)8191u;
    uint r = (uint)(r1 & (ulong)0x7fffffu) + (uint)(r1 >> 23) * 8191u;

    if (r >= 67043336u) r -= 67043336u; // 8*q
    if (r >= 33521668u) r -= 33521668u; // 4*q
    if (r >= 16760834u) r -= 16760834u; // 2*q
    if (r >=  8380417u) r -=  8380417u; // q
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
#define MOD_MUL_FAST(a,b)      mod_mul_fast((a), (b), q)

#define NTT_STAGE_CT_M(LEN, LOGV, KSTART, MUL) do {                     \
    uint _g = ltid >> (LOGV);                                            \
    uint _j = (_g << ((LOGV) + 1u)) | (ltid & ((LEN) - 1u));             \
    uint _z = zt[(KSTART) + _g];                                         \
    uint _x = a[_j];                                                     \
    uint _y = a[_j + (LEN)];                                             \
    uint _t = MUL(_z, _y);                                               \
    a[_j] = mod_add(_x, _t, q);                                          \
    a[_j + (LEN)] = mod_sub(_x, _t, q);                                  \
} while (0)

#define NTT_FINAL_CT_M(LEN, LOGV, KSTART, MUL) do {                     \
    uint _g = ltid >> (LOGV);                                            \
    uint _j = (_g << ((LOGV) + 1u)) | (ltid & ((LEN) - 1u));             \
    uint _z = zt[(KSTART) + _g];                                         \
    uint _x = a[_j];                                                     \
    uint _y = a[_j + (LEN)];                                             \
    uint _t = MUL(_z, _y);                                               \
    poly[_j] = mod_add(_x, _t, q);                                       \
    poly[_j + (LEN)] = mod_sub(_x, _t, q);                               \
} while (0)

#define RUN_NTT_256_7(MUL) do {                                         \
    NTT_STAGE_CT_M(128u, 7u,  1u, MUL);                                  \
    threadgroup_barrier(mem_flags::mem_threadgroup);                    \
    NTT_STAGE_CT_M( 64u, 6u,  2u, MUL);                                  \
    threadgroup_barrier(mem_flags::mem_threadgroup);                    \
    NTT_STAGE_CT_M( 32u, 5u,  4u, MUL);                                  \
    simdgroup_barrier(mem_flags::mem_threadgroup);                      \
    NTT_STAGE_CT_M( 16u, 4u,  8u, MUL);                                  \
    simdgroup_barrier(mem_flags::mem_threadgroup);                      \
    NTT_STAGE_CT_M(  8u, 3u, 16u, MUL);                                  \
    simdgroup_barrier(mem_flags::mem_threadgroup);                      \
    NTT_STAGE_CT_M(  4u, 2u, 32u, MUL);                                  \
    simdgroup_barrier(mem_flags::mem_threadgroup);                      \
    NTT_FINAL_CT_M(  2u, 1u, 64u, MUL);                                  \
} while (0)

#define RUN_NTT_256_8(MUL) do {                                         \
    NTT_STAGE_CT_M(128u, 7u,   1u, MUL);                                 \
    threadgroup_barrier(mem_flags::mem_threadgroup);                    \
    NTT_STAGE_CT_M( 64u, 6u,   2u, MUL);                                 \
    threadgroup_barrier(mem_flags::mem_threadgroup);                    \
    NTT_STAGE_CT_M( 32u, 5u,   4u, MUL);                                 \
    simdgroup_barrier(mem_flags::mem_threadgroup);                      \
    NTT_STAGE_CT_M( 16u, 4u,   8u, MUL);                                 \
    simdgroup_barrier(mem_flags::mem_threadgroup);                      \
    NTT_STAGE_CT_M(  8u, 3u,  16u, MUL);                                 \
    simdgroup_barrier(mem_flags::mem_threadgroup);                      \
    NTT_STAGE_CT_M(  4u, 2u,  32u, MUL);                                 \
    simdgroup_barrier(mem_flags::mem_threadgroup);                      \
    NTT_STAGE_CT_M(  2u, 1u,  64u, MUL);                                 \
    simdgroup_barrier(mem_flags::mem_threadgroup);                      \
    NTT_FINAL_CT_M(  1u, 0u, 128u, MUL);                                 \
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

    uint half_n = n >> 1u;
    device uint *poly = coeffs + (size_t)tgid * (size_t)n;

    a[ltid]          = poly[ltid];
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

    if (n == 256u && n_levels == 7u) {
        if (q == 3329u) {
            RUN_NTT_256_7(MOD_MUL_3329);
        } else if (q == 8380417u) {
            RUN_NTT_256_7(MOD_MUL_8380417);
        } else {
            RUN_NTT_256_7(MOD_MUL_FAST);
        }
        return;
    }

    if (n == 256u && n_levels == 8u) {
        if (q == 3329u) {
            RUN_NTT_256_8(MOD_MUL_3329);
        } else if (q == 8380417u) {
            RUN_NTT_256_8(MOD_MUL_8380417);
        } else {
            RUN_NTT_256_8(MOD_MUL_FAST);
        }
        return;
    }

    if (n_levels == 0u) {
        poly[ltid]          = a[ltid];
        poly[ltid + half_n] = a[ltid + half_n];
        return;
    }

    uint length  = half_n;
    uint k_start = 1u;

    for (uint level = 0u; level + 1u < n_levels; ++level) {
        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zeta       = zt[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];
        uint t = mod_mul_fast(zeta, y, q);

        a[j]          = mod_add(x, t, q);
        a[j + length] = mod_sub(x, t, q);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length  >>= 1u;
    }

    uint group_idx  = ltid / length;
    uint j_in_group = ltid - group_idx * length;
    uint j          = (group_idx << 1u) * length + j_in_group;
    uint zeta       = zt[k_start + group_idx];

    uint x = a[j];
    uint y = a[j + length];
    uint t = mod_mul_fast(zeta, y, q);

    poly[j]          = mod_add(x, t, q);
    poly[j + length] = mod_sub(x, t, q);
}
```