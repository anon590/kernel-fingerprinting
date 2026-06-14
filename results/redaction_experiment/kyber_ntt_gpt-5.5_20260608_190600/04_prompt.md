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

Result of previous attempt:
            kyb_B1: correct, 0.02 ms, 0.1 GB/s (0.1% of 200 GB/s)
           kyb_B16: correct, 0.02 ms, 1.8 GB/s (0.9% of 200 GB/s)
          kyb_B256: correct, 0.03 ms, 19.5 GB/s (9.7% of 200 GB/s)
  score (gmean of fraction): 0.0079

## Current best (incumbent)

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

Incumbent result:
            kyb_B1: correct, 0.02 ms, 0.1 GB/s (0.1% of 200 GB/s)
           kyb_B16: correct, 0.02 ms, 1.8 GB/s (0.9% of 200 GB/s)
          kyb_B256: correct, 0.03 ms, 19.5 GB/s (9.8% of 200 GB/s)
  score (gmean of fraction): 0.0079

## History

- iter  0: compile=OK | correct=True | score=0.0048458368752571255
- iter  1: compile=OK | correct=True | score=0.007885419300440817
- iter  2: compile=OK | correct=True | score=0.0060559767756625076
- iter  3: compile=OK | correct=True | score=0.007851809856195888

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
