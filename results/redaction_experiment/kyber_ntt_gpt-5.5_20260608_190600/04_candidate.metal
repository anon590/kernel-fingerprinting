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

// Exact for q = 3329. Also exact here for lazy inputs b < 9*q.
inline uint mod_mul_3329(uint a, uint b) {
    uint x = a * b;
    uint qhat = mulhi(x, 1290167u); // floor(2^32 / 3329)
    uint r = x - qhat * 3329u;
    return (r >= 3329u) ? (r - 3329u) : r;
}

// q = 8380417 = 2^23 - 2^13 + 1, canonical inputs only.
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

// Reduce x < 9*3329 to canonical [0,3329).
inline uint reduce_3329_small(uint x) {
    if (x >= 26632u) x -= 26632u; // 8*q
    if (x >= 13316u) x -= 13316u; // 4*q
    if (x >=  6658u) x -=  6658u; // 2*q
    if (x >=  3329u) x -=  3329u; // q
    return x;
}

#define MOD_MUL_3329(a,b)        mod_mul_3329((a), (b))
#define MOD_MUL_8380417(a,b)     mod_mul_8380417((a), (b))
#define MOD_MUL_GENERIC_Q(a,b)   mod_mul_generic((a), (b), q)
#define MOD_MUL_FAST_Q(a,b)      mod_mul_fast((a), (b), q)

#define MOD_ADD_3329(a,b)        mod_add_3329((a), (b))
#define MOD_SUB_3329(a,b)        mod_sub_3329((a), (b))
#define MOD_ADD_8380417(a,b)     mod_add_8380417((a), (b))
#define MOD_SUB_8380417(a,b)     mod_sub_8380417((a), (b))
#define MOD_ADD_Q(a,b)           mod_add_q((a), (b), q)
#define MOD_SUB_Q(a,b)           mod_sub_q((a), (b), q)

#define MOD_ADD_LAZY_3329(a,b)   ((a) + (b))
#define MOD_SUB_LAZY_3329(a,b)   ((a) + 3329u - (b))
#define MOD_ADD_FINAL_3329(a,b)  reduce_3329_small((a) + (b))
#define MOD_SUB_FINAL_3329(a,b)  reduce_3329_small((a) + 3329u - (b))

#define NTT_INIT_128_DIRECT(MUL, ADD, SUB) do {                          \
    uint _lane = ltid & 31u;                                              \
    uint _zl = (_lane == 0u) ? zetas[1u] : 0u;                            \
    uint _z = simd_shuffle(_zl, (ushort)0);                               \
    uint _x = poly[ltid];                                                 \
    uint _y = poly[ltid + 128u];                                          \
    uint _t = MUL(_z, _y);                                                \
    a[ltid] = ADD(_x, _t);                                                \
    a[ltid + 128u] = SUB(_x, _t);                                         \
} while (0)

#define NTT_STAGE_SCRATCH_ZB(LEN, LOGV, KSTART, LANEMASK, MUL, ADD, SUB) do { \
    uint _g = ltid >> (LOGV);                                            \
    uint _j = (_g << ((LOGV) + 1u)) | (ltid & ((LEN) - 1u));             \
    uint _lane = ltid & 31u;                                             \
    uint _leader = _lane & ~(LANEMASK);                                  \
    uint _zl = (_lane == _leader) ? zetas[(KSTART) + _g] : 0u;           \
    uint _z = simd_shuffle(_zl, (ushort)_leader);                        \
    uint _x = a[_j];                                                     \
    uint _y = a[_j + (LEN)];                                             \
    uint _t = MUL(_z, _y);                                               \
    a[_j] = ADD(_x, _t);                                                 \
    a[_j + (LEN)] = SUB(_x, _t);                                         \
} while (0)

#define REG_STAGE_ZB(H, LOGV, KSTART, LANEMASK, MUL, ADD, SUB) do {       \
    uint _lane = ltid & 31u;                                              \
    uint _g = ltid >> (LOGV);                                             \
    uint _leader = _lane & ~(LANEMASK);                                   \
    uint _zl = (_lane == _leader) ? zetas[(KSTART) + _g] : 0u;            \
    uint _z = simd_shuffle(_zl, (ushort)_leader);                         \
    uint2 _rp = simd_shuffle_xor(uint2(r0, r1), (ushort)(H));             \
    bool _lo = ((_lane & (H)) == 0u);                                     \
    uint _x = _lo ? r0 : _rp.y;                                           \
    uint _y = _lo ? _rp.x : r1;                                           \
    uint _t = MUL(_z, _y);                                                \
    r0 = ADD(_x, _t);                                                     \
    r1 = SUB(_x, _t);                                                     \
} while (0)

#define REG_STORE_FINAL(H, LOGV) do {                                     \
    uint _g = ltid >> (LOGV);                                             \
    uint _j = (_g << ((LOGV) + 1u)) | (ltid & ((H) - 1u));                \
    poly[_j] = r0;                                                        \
    poly[_j + (H)] = r1;                                                  \
} while (0)

#define NTT_TAIL_256_7_REG(MUL, ADD, SUB, FADD, FSUB) do {                \
    uint _lane0 = ltid & 31u;                                             \
    uint _g0 = ltid >> 5u;                                                \
    uint _j0 = (_g0 << 6u) | _lane0;                                      \
    uint _zl0 = (_lane0 == 0u) ? zetas[4u + _g0] : 0u;                   \
    uint _z0 = simd_shuffle(_zl0, (ushort)0);                             \
    uint _x0 = a[_j0];                                                    \
    uint _y0 = a[_j0 + 32u];                                              \
    uint _t0 = MUL(_z0, _y0);                                             \
    uint r0 = ADD(_x0, _t0);                                              \
    uint r1 = SUB(_x0, _t0);                                              \
    REG_STAGE_ZB(16u, 4u,  8u, 15u, MUL, ADD,  SUB);                     \
    REG_STAGE_ZB( 8u, 3u, 16u,  7u, MUL, ADD,  SUB);                     \
    REG_STAGE_ZB( 4u, 2u, 32u,  3u, MUL, ADD,  SUB);                     \
    REG_STAGE_ZB( 2u, 1u, 64u,  1u, MUL, FADD, FSUB);                    \
    REG_STORE_FINAL(2u, 1u);                                              \
} while (0)

#define NTT_TAIL_256_8_REG(MUL, ADD, SUB, FADD, FSUB) do {                \
    uint _lane0 = ltid & 31u;                                             \
    uint _g0 = ltid >> 5u;                                                \
    uint _j0 = (_g0 << 6u) | _lane0;                                      \
    uint _zl0 = (_lane0 == 0u) ? zetas[4u + _g0] : 0u;                   \
    uint _z0 = simd_shuffle(_zl0, (ushort)0);                             \
    uint _x0 = a[_j0];                                                    \
    uint _y0 = a[_j0 + 32u];                                              \
    uint _t0 = MUL(_z0, _y0);                                             \
    uint r0 = ADD(_x0, _t0);                                              \
    uint r1 = SUB(_x0, _t0);                                              \
    REG_STAGE_ZB(16u, 4u,   8u, 15u, MUL, ADD,  SUB);                    \
    REG_STAGE_ZB( 8u, 3u,  16u,  7u, MUL, ADD,  SUB);                    \
    REG_STAGE_ZB( 4u, 2u,  32u,  3u, MUL, ADD,  SUB);                    \
    REG_STAGE_ZB( 2u, 1u,  64u,  1u, MUL, ADD,  SUB);                    \
    REG_STAGE_ZB( 1u, 0u, 128u,  0u, MUL, FADD, FSUB);                   \
    REG_STORE_FINAL(1u, 0u);                                              \
} while (0)

#define RUN_NTT_256_7_REG(MUL, ADD, SUB, FADD, FSUB) do {                 \
    NTT_INIT_128_DIRECT(MUL, ADD, SUB);                                   \
    threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    NTT_STAGE_SCRATCH_ZB(64u, 6u, 2u, 31u, MUL, ADD, SUB);                \
    threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    NTT_TAIL_256_7_REG(MUL, ADD, SUB, FADD, FSUB);                        \
} while (0)

#define RUN_NTT_256_8_REG(MUL, ADD, SUB, FADD, FSUB) do {                 \
    NTT_INIT_128_DIRECT(MUL, ADD, SUB);                                   \
    threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    NTT_STAGE_SCRATCH_ZB(64u, 6u, 2u, 31u, MUL, ADD, SUB);                \
    threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    NTT_TAIL_256_8_REG(MUL, ADD, SUB, FADD, FSUB);                        \
} while (0)

#define NTT_STAGE_CT_FALLBACK(LEN, LOGV, KSTART, MUL) do {                \
    uint _g = ltid >> (LOGV);                                             \
    uint _j = (_g << ((LOGV) + 1u)) | (ltid & ((LEN) - 1u));              \
    uint _z = zt[(KSTART) + _g];                                          \
    uint _x = a[_j];                                                      \
    uint _y = a[_j + (LEN)];                                              \
    uint _t = MUL(_z, _y);                                                \
    a[_j] = mod_add_q(_x, _t, q);                                         \
    a[_j + (LEN)] = mod_sub_q(_x, _t, q);                                 \
} while (0)

#define NTT_FINAL_CT_FALLBACK(LEN, LOGV, KSTART, MUL) do {                \
    uint _g = ltid >> (LOGV);                                             \
    uint _j = (_g << ((LOGV) + 1u)) | (ltid & ((LEN) - 1u));              \
    uint _z = zt[(KSTART) + _g];                                          \
    uint _x = a[_j];                                                      \
    uint _y = a[_j + (LEN)];                                              \
    uint _t = MUL(_z, _y);                                                \
    poly[_j] = mod_add_q(_x, _t, q);                                      \
    poly[_j + (LEN)] = mod_sub_q(_x, _t, q);                              \
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

        if (q == 3329u) {
            RUN_NTT_256_7_REG(MOD_MUL_3329,
                              MOD_ADD_LAZY_3329, MOD_SUB_LAZY_3329,
                              MOD_ADD_FINAL_3329, MOD_SUB_FINAL_3329);
        } else if (q == 8380417u) {
            RUN_NTT_256_7_REG(MOD_MUL_8380417,
                              MOD_ADD_8380417, MOD_SUB_8380417,
                              MOD_ADD_8380417, MOD_SUB_8380417);
        } else {
            RUN_NTT_256_7_REG(MOD_MUL_GENERIC_Q,
                              MOD_ADD_Q, MOD_SUB_Q,
                              MOD_ADD_Q, MOD_SUB_Q);
        }
        return;
    }

    if (n == 256u && n_levels == 8u) {
        device uint *poly = coeffs + ((size_t)tgid << 8);

        if (q == 3329u) {
            RUN_NTT_256_8_REG(MOD_MUL_3329,
                              MOD_ADD_LAZY_3329, MOD_SUB_LAZY_3329,
                              MOD_ADD_FINAL_3329, MOD_SUB_FINAL_3329);
        } else if (q == 8380417u) {
            RUN_NTT_256_8_REG(MOD_MUL_8380417,
                              MOD_ADD_8380417, MOD_SUB_8380417,
                              MOD_ADD_8380417, MOD_SUB_8380417);
        } else {
            RUN_NTT_256_8_REG(MOD_MUL_GENERIC_Q,
                              MOD_ADD_Q, MOD_SUB_Q,
                              MOD_ADD_Q, MOD_SUB_Q);
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