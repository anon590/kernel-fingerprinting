## Task: kyber_ntt

Batched forward Cooley-Tukey NTT over a small prime field for a negacyclic polynomial ring Z_q[X] / (X^n + 1). Each polynomial has length n; the modulus q, the polynomial length n, the number of NTT stages n_levels, and the precomputed twiddle table are all bound at runtime so a single kernel runs every parameter set the host supplies.

Convention (matches the FIPS 203 / FIPS 204 / pqclean reference C implementations):
  k = 1
  for level = 0..n_levels:
      len = n >> (level + 1)
      for start = 0, 2*len, ..., n - 2*len:
          z = zetas[k++]
          for j = start..start + len - 1:
              t          = (z * a[j+len]) mod q
              a[j+len]   = (a[j] - t)       mod q
              a[j]       = (a[j] + t)       mod q
Equivalent per-thread mapping (ltid in [0, n/2) owns one butterfly per level):
  group_idx   = ltid / len
  j_in_group  = ltid - group_idx * len   (= ltid mod len)
  j           = (group_idx << 1) * len + j_in_group
  zeta_index  = (1 << level) + group_idx

Zetas table (host-precomputed, length 1 << n_levels):
  zetas[k] = zeta^bit_reverse(k, n_levels)  mod q
where zeta is a primitive 2^(n_levels+1)-th root of unity in F_q. The forward NTT consumes zetas[1..(1 << n_levels) - 1] in increasing index order; zetas[0] = 1 is the unread identity element.

Bounds for kernel design: q fits in a 32-bit unsigned integer; n is a power of two with n <= 256; n_levels <= 8 (so the zetas table has at most 256 entries). The kernel MUST read q, n, and n_levels from their bound buffers and load every twiddle from the zetas buffer at runtime; compile-time constants for any of these values are incorrect.

Storage: ``uint32`` per coefficient, in-place. The host writes the input coefficients into ``coeffs`` and reads the result back from the same buffer; ``coeffs`` is (batch * n) uint values in row-major order (polynomial p's coefficients live at offsets p*n .. p*n + n - 1).

All inputs are canonical: a[i] in [0, q). Outputs MUST also be canonical -- a value in [q, 2^32) with the same residue class still counts as a mismatch on the bit-exact reference comparison.

## Required kernel signature(s)

```
kernel void kyber_ntt(
    device       uint *coeffs     [[buffer(0)]],
    device const uint *zetas      [[buffer(1)]],
    constant uint     &q           [[buffer(2)]],
    constant uint     &n           [[buffer(3)]],
    constant uint     &n_levels    [[buffer(4)]],
    constant uint     &batch       [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ltid [[thread_position_in_threadgroup]]);

Dispatch (host-provided):
  threadsPerGrid        = (batch * (n/2), 1, 1)
  threadsPerThreadgroup = (n/2, 1, 1)
Each threadgroup owns ONE polynomial; tgid in [0, batch) selects the polynomial, ltid in [0, n/2) owns one butterfly per level. Every test size uses n = 256, so n/2 = 128 threads per threadgroup is sufficient; a static threadgroup scratch of size 256 covers every case. Threadgroup-cooperative and simdgroup schemes are valid as long as the buffer layout and the canonical-output contract are preserved.
```

## Your previous attempt

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

// Exact for q = 3329 and b < 9*q, covering the lazy 256-point paths.
inline uint mod_mul_3329(uint a, uint b) {
    uint x = a * b;
    uint qhat = mulhi(x, 1290167u); // floor(2^32 / 3329)
    uint r = x - qhat * 3329u;
    return (r >= 3329u) ? (r - 3329u) : r;
}

inline uint reduce_3329_small(uint x) {
    if (x >= 26632u) x -= 26632u; // 8*q
    if (x >= 13316u) x -= 13316u; // 4*q
    if (x >=  6658u) x -=  6658u; // 2*q
    if (x >=  3329u) x -=  3329u; // q
    return x;
}

inline uint mod_mul_generic(uint a, uint b, uint q) {
    return (uint)(((ulong)a * (ulong)b) % (ulong)q);
}

inline uint mod_mul_fast(uint a, uint b, uint q) {
    if (q == 3329u) return mod_mul_3329(a, b);
    return mod_mul_generic(a, b, q);
}

#define REG_STAGE_3329(LEN, LOGV, KSTART) do {                                      \
    uint _local = lane & 15u;                                                       \
    uint _grp   = _local >> (LOGV);                                                 \
    uint _j     = _local & ((LEN) - 1u);                                            \
    uint _p0    = (_grp << ((LOGV) + 1u)) | _j;                                     \
    uint _p1    = _p0 + (LEN);                                                      \
                                                                                     \
    uint _x0 = simd_shuffle(r0, (ushort)_p0);                                       \
    uint _y0 = simd_shuffle(r0, (ushort)_p1);                                       \
    uint _x1 = simd_shuffle(r1, (ushort)_p0);                                       \
    uint _y1 = simd_shuffle(r1, (ushort)_p1);                                       \
    uint _x  = (lane < 16u) ? _x0 : _x1;                                            \
    uint _y  = (lane < 16u) ? _y0 : _y1;                                            \
                                                                                     \
    uint _groups_per_half = 16u >> (LOGV);                                          \
    uint _global_group = (sg << (5u - (LOGV))) + ((lane >> 4u) * _groups_per_half) + _grp; \
    uint _z = zt[(KSTART) + _global_group];                                         \
    uint _t = mod_mul_3329(_z, _y);                                                 \
    uint _top = _x + _t;                                                            \
    uint _bot = _x + 3329u - _t;                                                    \
                                                                                     \
    uint _dst = ((lane >> ((LOGV) + 1u)) << (LOGV)) | (lane & ((LEN) - 1u));         \
    bool _upper = ((lane & (LEN)) != 0u);                                           \
                                                                                     \
    uint _r0t = simd_shuffle(_top, (ushort)_dst);                                   \
    uint _r0b = simd_shuffle(_bot, (ushort)_dst);                                   \
    uint _r1t = simd_shuffle(_top, (ushort)(16u + _dst));                           \
    uint _r1b = simd_shuffle(_bot, (ushort)(16u + _dst));                           \
                                                                                     \
    r0 = _upper ? _r0b : _r0t;                                                      \
    r1 = _upper ? _r1b : _r1t;                                                      \
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

    if (n == 256u && q == 3329u && (n_levels == 7u || n_levels == 8u)) {
        device uint *poly = coeffs + ((size_t)tgid << 8);

        zt[ltid] = zetas[ltid];
        if (n_levels == 8u) {
            zt[ltid + 128u] = zetas[ltid + 128u];
        }

        uint lane = ltid & 31u;
        uint zl = (lane == 0u) ? zetas[1u] : 0u;
        uint z = simd_shuffle(zl, (ushort)0);

        uint x = poly[ltid];
        uint y = poly[ltid + 128u];
        uint t = mod_mul_3329(z, y);

        a[ltid]        = x + t;
        a[ltid + 128u] = x + 3329u - t;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint g64 = ltid >> 6u;
        uint j64 = (g64 << 7u) | (ltid & 63u);
        z = zt[2u + g64];

        x = a[j64];
        y = a[j64 + 64u];
        t = mod_mul_3329(z, y);
        a[j64]       = x + t;
        a[j64 + 64u] = x + 3329u - t;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint sg = ltid >> 5u;
        uint base = sg << 6u;
        z = zt[4u + sg];

        x = a[base + lane];
        y = a[base + lane + 32u];
        t = mod_mul_3329(z, y);

        uint r0 = x + t;
        uint r1 = x + 3329u - t;

        REG_STAGE_3329(16u, 4u,   8u);
        REG_STAGE_3329( 8u, 3u,  16u);
        REG_STAGE_3329( 4u, 2u,  32u);
        REG_STAGE_3329( 2u, 1u,  64u);

        if (n_levels == 8u) {
            REG_STAGE_3329(1u, 0u, 128u);
        }

        poly[base + lane]       = reduce_3329_small(r0);
        poly[base + lane + 32u] = reduce_3329_small(r1);
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

        uint vx = a[j];
        uint vy = a[j + length];
        uint vt = mod_mul_fast(zeta, vy, q);

        a[j] = mod_add_q(vx, vt, q);
        a[j + length] = mod_sub_q(vx, vt, q);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length >>= 1u;
    }

    uint group_idx = ltid / length;
    uint j_in_group = ltid - group_idx * length;
    uint j = (group_idx << 1u) * length + j_in_group;
    uint zeta = zt[k_start + group_idx];

    uint vx = a[j];
    uint vy = a[j + length];
    uint vt = mod_mul_fast(zeta, vy, q);

    poly[j] = mod_add_q(vx, vt, q);
    poly[j + length] = mod_sub_q(vx, vt, q);
}
```

Result of previous attempt:
            kyb_B1: correct, 0.02 ms, 0.2 GB/s (0.1% of 200 GB/s)
           kyb_B16: correct, 0.02 ms, 2.2 GB/s (1.1% of 200 GB/s)
          kyb_B256: correct, 0.06 ms, 10.8 GB/s (5.4% of 200 GB/s)
  score (gmean of fraction): 0.0079

## Current best (incumbent)

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

// Exact for q = 3329 for b < 9*q, which covers the lazy 256-point paths.
inline uint mod_mul_3329(uint a, uint b) {
    uint x = a * b;
    uint qhat = mulhi(x, 1290167u); // floor(2^32 / 3329)
    uint r = x - qhat * 3329u;
    return (r >= 3329u) ? (r - 3329u) : r;
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

// Reduce x < 9*3329 to canonical [0, 3329).
inline uint reduce_3329_small(uint x) {
    if (x >= 26632u) x -= 26632u; // 8*q
    if (x >= 13316u) x -= 13316u; // 4*q
    if (x >=  6658u) x -=  6658u; // 2*q
    if (x >=  3329u) x -=  3329u; // q
    return x;
}

#define MOD_MUL_3329(a,b)      mod_mul_3329((a), (b))
#define MOD_MUL_8380417(a,b)   mod_mul_8380417((a), (b))
#define MOD_MUL_GENERIC_Q(a,b) mod_mul_generic((a), (b), q)
#define MOD_MUL_FAST_Q(a,b)    mod_mul_fast((a), (b), q)

#define MOD_ADD_Q(a,b)         mod_add_q((a), (b), q)
#define MOD_SUB_Q(a,b)         mod_sub_q((a), (b), q)
#define MOD_ADD_838(a,b)       mod_add_8380417((a), (b))
#define MOD_SUB_838(a,b)       mod_sub_8380417((a), (b))

#define MOD_ADD_LAZY_3329(a,b) ((a) + (b))
#define MOD_SUB_LAZY_3329(a,b) ((a) + 3329u - (b))

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

#define NTT_STAGE_3329_LAZY(LEN, LOGV, KSTART) do {                       \
    uint _g = ltid >> (LOGV);                                             \
    uint _j = (_g << ((LOGV) + 1u)) | (ltid & ((LEN) - 1u));              \
    uint _z = zt[(KSTART) + _g];                                          \
    uint _x = a[_j];                                                      \
    uint _y = a[_j + (LEN)];                                              \
    uint _t = mod_mul_3329(_z, _y);                                       \
    a[_j] = _x + _t;                                                      \
    a[_j + (LEN)] = _x + 3329u - _t;                                      \
} while (0)

#define NTT_FINAL_3329_LAZY(LEN, LOGV, KSTART) do {                       \
    uint _g = ltid >> (LOGV);                                             \
    uint _j = (_g << ((LOGV) + 1u)) | (ltid & ((LEN) - 1u));              \
    uint _z = zt[(KSTART) + _g];                                          \
    uint _x = a[_j];                                                      \
    uint _y = a[_j + (LEN)];                                              \
    uint _t = mod_mul_3329(_z, _y);                                       \
    poly[_j] = reduce_3329_small(_x + _t);                                \
    poly[_j + (LEN)] = reduce_3329_small(_x + 3329u - _t);                \
} while (0)

#define RUN_REST_256_7_GENERIC(MUL) do {                                  \
    threadgroup_barrier(mem_flags::mem_threadgroup);                     \
    NTT_STAGE_CT_M( 64u, 6u,  2u, MUL);                                   \
    threadgroup_barrier(mem_flags::mem_threadgroup);                     \
    NTT_STAGE_CT_M( 32u, 5u,  4u, MUL);                                   \
    simdgroup_barrier(mem_flags::mem_threadgroup);                       \
    NTT_STAGE_CT_M( 16u, 4u,  8u, MUL);                                   \
    simdgroup_barrier(mem_flags::mem_threadgroup);                       \
    NTT_STAGE_CT_M(  8u, 3u, 16u, MUL);                                   \
    simdgroup_barrier(mem_flags::mem_threadgroup);                       \
    NTT_STAGE_CT_M(  4u, 2u, 32u, MUL);                                   \
    simdgroup_barrier(mem_flags::mem_threadgroup);                       \
    NTT_FINAL_CT_M(  2u, 1u, 64u, MUL);                                   \
} while (0)

#define RUN_REST_256_8_GENERIC(MUL) do {                                  \
    threadgroup_barrier(mem_flags::mem_threadgroup);                     \
    NTT_STAGE_CT_M( 64u, 6u,   2u, MUL);                                  \
    threadgroup_barrier(mem_flags::mem_threadgroup);                     \
    NTT_STAGE_CT_M( 32u, 5u,   4u, MUL);                                  \
    simdgroup_barrier(mem_flags::mem_threadgroup);                       \
    NTT_STAGE_CT_M( 16u, 4u,   8u, MUL);                                  \
    simdgroup_barrier(mem_flags::mem_threadgroup);                       \
    NTT_STAGE_CT_M(  8u, 3u,  16u, MUL);                                  \
    simdgroup_barrier(mem_flags::mem_threadgroup);                       \
    NTT_STAGE_CT_M(  4u, 2u,  32u, MUL);                                  \
    simdgroup_barrier(mem_flags::mem_threadgroup);                       \
    NTT_STAGE_CT_M(  2u, 1u,  64u, MUL);                                  \
    simdgroup_barrier(mem_flags::mem_threadgroup);                       \
    NTT_FINAL_CT_M(  1u, 0u, 128u, MUL);                                  \
} while (0)

#define RUN_REST_256_7_838() do {                                         \
    threadgroup_barrier(mem_flags::mem_threadgroup);                     \
    NTT_STAGE_838( 64u, 6u,  2u);                                         \
    threadgroup_barrier(mem_flags::mem_threadgroup);                     \
    NTT_STAGE_838( 32u, 5u,  4u);                                         \
    simdgroup_barrier(mem_flags::mem_threadgroup);                       \
    NTT_STAGE_838( 16u, 4u,  8u);                                         \
    simdgroup_barrier(mem_flags::mem_threadgroup);                       \
    NTT_STAGE_838(  8u, 3u, 16u);                                         \
    simdgroup_barrier(mem_flags::mem_threadgroup);                       \
    NTT_STAGE_838(  4u, 2u, 32u);                                         \
    simdgroup_barrier(mem_flags::mem_threadgroup);                       \
    NTT_FINAL_838(  2u, 1u, 64u);                                         \
} while (0)

#define RUN_REST_256_8_838() do {                                         \
    threadgroup_barrier(mem_flags::mem_threadgroup);                     \
    NTT_STAGE_838( 64u, 6u,   2u);                                        \
    threadgroup_barrier(mem_flags::mem_threadgroup);                     \
    NTT_STAGE_838( 32u, 5u,   4u);                                        \
    simdgroup_barrier(mem_flags::mem_threadgroup);                       \
    NTT_STAGE_838( 16u, 4u,   8u);                                        \
    simdgroup_barrier(mem_flags::mem_threadgroup);                       \
    NTT_STAGE_838(  8u, 3u,  16u);                                        \
    simdgroup_barrier(mem_flags::mem_threadgroup);                       \
    NTT_STAGE_838(  4u, 2u,  32u);                                        \
    simdgroup_barrier(mem_flags::mem_threadgroup);                       \
    NTT_STAGE_838(  2u, 1u,  64u);                                        \
    simdgroup_barrier(mem_flags::mem_threadgroup);                       \
    NTT_FINAL_838(  1u, 0u, 128u);                                        \
} while (0)

#define RUN_REST_256_7_3329_LAZY() do {                                   \
    threadgroup_barrier(mem_flags::mem_threadgroup);                     \
    NTT_STAGE_3329_LAZY( 64u, 6u,  2u);                                   \
    threadgroup_barrier(mem_flags::mem_threadgroup);                     \
    NTT_STAGE_3329_LAZY( 32u, 5u,  4u);                                   \
    simdgroup_barrier(mem_flags::mem_threadgroup);                       \
    NTT_STAGE_3329_LAZY( 16u, 4u,  8u);                                   \
    simdgroup_barrier(mem_flags::mem_threadgroup);                       \
    NTT_STAGE_3329_LAZY(  8u, 3u, 16u);                                   \
    simdgroup_barrier(mem_flags::mem_threadgroup);                       \
    NTT_STAGE_3329_LAZY(  4u, 2u, 32u);                                   \
    simdgroup_barrier(mem_flags::mem_threadgroup);                       \
    NTT_FINAL_3329_LAZY(  2u, 1u, 64u);                                   \
} while (0)

#define RUN_REST_256_8_3329_LAZY() do {                                   \
    threadgroup_barrier(mem_flags::mem_threadgroup);                     \
    NTT_STAGE_3329_LAZY( 64u, 6u,   2u);                                  \
    threadgroup_barrier(mem_flags::mem_threadgroup);                     \
    NTT_STAGE_3329_LAZY( 32u, 5u,   4u);                                  \
    simdgroup_barrier(mem_flags::mem_threadgroup);                       \
    NTT_STAGE_3329_LAZY( 16u, 4u,   8u);                                  \
    simdgroup_barrier(mem_flags::mem_threadgroup);                       \
    NTT_STAGE_3329_LAZY(  8u, 3u,  16u);                                  \
    simdgroup_barrier(mem_flags::mem_threadgroup);                       \
    NTT_STAGE_3329_LAZY(  4u, 2u,  32u);                                  \
    simdgroup_barrier(mem_flags::mem_threadgroup);                       \
    NTT_STAGE_3329_LAZY(  2u, 1u,  64u);                                  \
    simdgroup_barrier(mem_flags::mem_threadgroup);                       \
    NTT_FINAL_3329_LAZY(  1u, 0u, 128u);                                  \
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
            INIT_128_DIRECT(MOD_MUL_3329, MOD_ADD_LAZY_3329, MOD_SUB_LAZY_3329);
            RUN_REST_256_7_3329_LAZY();
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
            INIT_128_DIRECT(MOD_MUL_3329, MOD_ADD_LAZY_3329, MOD_SUB_LAZY_3329);
            RUN_REST_256_8_3329_LAZY();
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
```

Incumbent result:
            kyb_B1: correct, 0.02 ms, 0.2 GB/s (0.1% of 200 GB/s)
           kyb_B16: correct, 0.02 ms, 2.1 GB/s (1.0% of 200 GB/s)
          kyb_B256: correct, 0.03 ms, 19.4 GB/s (9.7% of 200 GB/s)
  score (gmean of fraction): 0.0092

## History

- iter  0: compile=OK | correct=True | score=0.0048458368752571255
- iter  1: compile=OK | correct=True | score=0.007885419300440817
- iter  2: compile=OK | correct=True | score=0.0060559767756625076
- iter  3: compile=OK | correct=True | score=0.007851809856195888
- iter  4: compile=OK | correct=True | score=0.0061822304437018655
- iter  5: compile=OK | correct=True | score=0.00921003523254375
- iter  6: compile=OK | correct=True | score=0.007864404023526565

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
