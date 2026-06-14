I replace the late threadgroup-memory stages with a SIMD-register NTT network. After the two cross-simdgroup stages, each simdgroup owns a 64-coefficient block; subsequent butterflies are computed with `simd_shuffle_xor`, eliminating several threadgroup reads/writes and simd barriers. I also specialize add/sub for common Kyber/Dilithium moduli to avoid runtime modulus checks in the hot path. 
```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_MAX = 256u;

inline uint mod_add_q(uint a, uint b, uint q) {
    uint t = a + b;
    return (t >= q || t < a) ? (t - q) : t;
}

inline uint mod_sub_q(uint a, uint b, uint q) {
    return (a >= b) ? (a - b) : (a + q - b);
}

inline uint mod_add_3329(uint a, uint b) {
    uint t = a + b;
    return (t >= 3329u) ? (t - 3329u) : t;
}

inline uint mod_sub_3329(uint a, uint b) {
    return (a >= b) ? (a - b) : (a + 3329u - b);
}

inline uint mod_add_8380417(uint a, uint b) {
    uint t = a + b;
    return (t >= 8380417u) ? (t - 8380417u) : t;
}

inline uint mod_sub_8380417(uint a, uint b) {
    return (a >= b) ? (a - b) : (a + 8380417u - b);
}

// Exact for q = 3329, a,b in [0,q), x < q^2.
// floor(2^32 / 3329) = 1290167; quotient error is at most 1.
inline uint mod_mul_3329(uint a, uint b) {
    uint x = a * b;
    uint qhat = mulhi(x, 1290167u);
    uint r = x - qhat * 3329u;
    return (r >= 3329u) ? (r - 3329u) : r;
}

// Exact for q = 8380417 = 2^23 - 2^13 + 1.
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

// Register stage for len = L <= 16.  v0/v1 hold the two outputs of the
// previous stage's butterfly for this logical ltid within the 64-coeff block.
#define REG_STAGE_XOR_M(L, LOGV, KSTART, MUL, ADD, SUB) do {                    \
    uint _px0 = simd_shuffle_xor(v0, (ushort)(L));                              \
    uint _px1 = simd_shuffle_xor(v1, (ushort)(L));                              \
    bool _upper = ((lane & (L)) != 0u);                                         \
    uint _x = _upper ? _px1 : v0;                                               \
    uint _y = _upper ? v1   : _px0;                                             \
    uint _leader = lane & ~((L) - 1u);                                          \
    uint _zload = ((lane & ((L) - 1u)) == 0u) ?                                 \
                  zetas[(KSTART) + (ltid >> (LOGV))] : 0u;                     \
    uint _z = simd_shuffle(_zload, (ushort)_leader);                            \
    uint _t = MUL(_z, _y);                                                      \
    v0 = ADD(_x, _t);                                                           \
    v1 = SUB(_x, _t);                                                           \
} while (0)

// Final register stage for len = 1; each lane has a unique twiddle.
#define REG_STAGE_XOR1_M(KSTART, MUL, ADD, SUB) do {                            \
    uint _px0 = simd_shuffle_xor(v0, (ushort)1);                                \
    uint _px1 = simd_shuffle_xor(v1, (ushort)1);                                \
    bool _upper = ((lane & 1u) != 0u);                                          \
    uint _x = _upper ? _px1 : v0;                                               \
    uint _y = _upper ? v1   : _px0;                                             \
    uint _z = zetas[(KSTART) + ltid];                                           \
    uint _t = MUL(_z, _y);                                                      \
    v0 = ADD(_x, _t);                                                           \
    v1 = SUB(_x, _t);                                                           \
} while (0)

#define RUN_OPT_NTT_256(MUL, ADD, SUB) do {                                     \
    uint lane = ltid & 31u;                                                     \
    uint sg   = ltid >> 5u;                                                     \
                                                                                \
    uint _z1load = (lane == 0u) ? zetas[1u] : 0u;                               \
    uint _z1 = simd_shuffle(_z1load, (ushort)0);                                \
                                                                                \
    uint _x0 = poly[ltid];                                                      \
    uint _y0 = poly[ltid + 128u];                                               \
    uint _t0 = MUL(_z1, _y0);                                                   \
    uint v0 = ADD(_x0, _t0);                                                    \
    uint v1 = SUB(_x0, _t0);                                                    \
                                                                                \
    a[ltid]         = v0;                                                       \
    a[ltid + 128u]  = v1;                                                       \
    threadgroup_barrier(mem_flags::mem_threadgroup);                            \
                                                                                \
    uint _g64 = ltid >> 6u;                                                     \
    uint _j64 = (_g64 << 7u) | (ltid & 63u);                                    \
    uint _z2load = (lane == 0u) ? zetas[2u + _g64] : 0u;                        \
    uint _z2 = simd_shuffle(_z2load, (ushort)0);                                \
    uint _x1 = a[_j64];                                                         \
    uint _y1 = a[_j64 + 64u];                                                   \
    uint _t1 = MUL(_z2, _y1);                                                   \
    a[_j64]       = ADD(_x1, _t1);                                              \
    a[_j64 + 64u] = SUB(_x1, _t1);                                              \
    threadgroup_barrier(mem_flags::mem_threadgroup);                            \
                                                                                \
    uint _base64 = sg << 6u;                                                    \
    uint _x32 = a[_base64 + lane];                                              \
    uint _y32 = a[_base64 + lane + 32u];                                        \
    uint _z32load = (lane == 0u) ? zetas[4u + sg] : 0u;                         \
    uint _z32 = simd_shuffle(_z32load, (ushort)0);                              \
    uint _t32 = MUL(_z32, _y32);                                                \
    v0 = ADD(_x32, _t32);                                                       \
    v1 = SUB(_x32, _t32);                                                       \
                                                                                \
    REG_STAGE_XOR_M(16u, 4u,   8u, MUL, ADD, SUB);                              \
    REG_STAGE_XOR_M( 8u, 3u,  16u, MUL, ADD, SUB);                              \
    REG_STAGE_XOR_M( 4u, 2u,  32u, MUL, ADD, SUB);                              \
    REG_STAGE_XOR_M( 2u, 1u,  64u, MUL, ADD, SUB);                              \
                                                                                \
    if (n_levels == 8u) {                                                       \
        REG_STAGE_XOR1_M(128u, MUL, ADD, SUB);                                  \
        uint _jout = lane << 1u;                                                \
        poly[_base64 + _jout]      = v0;                                        \
        poly[_base64 + _jout + 1u] = v1;                                        \
    } else {                                                                    \
        uint _jout = ((lane >> 1u) << 2u) | (lane & 1u);                        \
        poly[_base64 + _jout]      = v0;                                        \
        poly[_base64 + _jout + 2u] = v1;                                        \
    }                                                                           \
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

    device uint *poly = coeffs + (size_t)tgid * (size_t)n;

    if (n == 256u && (n_levels == 7u || n_levels == 8u)) {
        if (q == 3329u) {
            RUN_OPT_NTT_256(MOD_MUL_3329, MOD_ADD_3329, MOD_SUB_3329);
        } else if (q == 8380417u) {
            RUN_OPT_NTT_256(MOD_MUL_8380417, MOD_ADD_8380417, MOD_SUB_8380417);
        } else {
            RUN_OPT_NTT_256(MOD_MUL_GENERIC_Q, MOD_ADD_Q, MOD_SUB_Q);
        }
        return;
    }

    uint half_n = n >> 1u;

    uint lo = poly[ltid];
    uint hi = poly[ltid + half_n];
    a[ltid]          = lo;
    a[ltid + half_n] = hi;

    if (n_levels == 0u) {
        poly[ltid]          = lo;
        poly[ltid + half_n] = hi;
        return;
    }

    uint length  = half_n;
    uint k_start = 1u;

    for (uint level = 0u; level + 1u < n_levels; ++level) {
        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zeta       = zetas[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];
        uint t = mod_mul_fast(zeta, y, q);

        a[j]          = mod_add_q(x, t, q);
        a[j + length] = mod_sub_q(x, t, q);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length  >>= 1u;
    }

    uint group_idx  = ltid / length;
    uint j_in_group = ltid - group_idx * length;
    uint j          = (group_idx << 1u) * length + j_in_group;
    uint zeta       = zetas[k_start + group_idx];

    uint x = a[j];
    uint y = a[j + length];
    uint t = mod_mul_fast(zeta, y, q);

    poly[j]          = mod_add_q(x, t, q);
    poly[j + length] = mod_sub_q(x, t, q);
}
```