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

inline uint add_3329(uint a, uint b) {
    uint s = a + b;
    return (s >= 3329u) ? (s - 3329u) : s;
}

inline uint sub_3329(uint a, uint b) {
    uint d = a - b;
    return (a < b) ? (d + 3329u) : d;
}

inline uint mod_mul_3329(uint a, uint b) {
    uint x = a * b;
    uint qhat = mulhi(x, 1290167u);
    uint r = x - qhat * 3329u;
    if (r >= 3329u) r -= 3329u;
    return r;
}

inline uint add_8380417(uint a, uint b) {
    uint s = a + b;
    return (s >= 8380417u) ? (s - 8380417u) : s;
}

inline uint sub_8380417(uint a, uint b) {
    uint d = a - b;
    return (a < b) ? (d + 8380417u) : d;
}

inline uint mod_mul_8380417(uint a, uint b) {
    ulong x = (ulong)a * (ulong)b;
    ulong r = (x & 0x7ffffful) + ((x >> 23) * 8191ul);
    r = (r & 0x7ffffful) + ((r >> 23) * 8191ul);
    uint s = (uint)((r & 0x7ffffful) + ((r >> 23) * 8191ul));
    if (s >= 8380417u) s -= 8380417u;
    return s;
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

inline uint preload_late_zeta_3329(device const uint *zetas, uint sg, uint lane) {
    if (lane < 2u) {
        return zetas[8u + (sg << 1u) + lane];
    } else if (lane < 6u) {
        return zetas[16u + (sg << 2u) + (lane - 2u)];
    } else if (lane < 14u) {
        return zetas[32u + (sg << 3u) + (lane - 6u)];
    } else if (lane < 30u) {
        return zetas[64u + (sg << 4u) + (lane - 14u)];
    }
    return 0u;
}

inline uint2 simd_step_3329_pre16(uint v0, uint v1, uint lane, uint zpre) {
    uint z0 = simd_broadcast(zpre, (ushort)0);
    uint z1 = simd_broadcast(zpre, (ushort)1);
    return simd_bfly2_3329(v0, v1, (ushort)16, ((lane & 16u) == 0u), z0, z1);
}

inline uint2 simd_step_3329_pre8(uint v0, uint v1, uint lane, uint zpre) {
    uint seg = lane >> 4u;
    uint z0 = simd_shuffle(zpre, (ushort)(2u + seg));
    uint z1 = simd_shuffle(zpre, (ushort)(4u + seg));
    return simd_bfly2_3329(v0, v1, (ushort)8, ((lane & 8u) == 0u), z0, z1);
}

inline uint2 simd_step_3329_pre4(uint v0, uint v1, uint lane, uint zpre) {
    uint seg = (lane >> 3u) & 3u;
    uint z0 = simd_shuffle(zpre, (ushort)(6u + seg));
    uint z1 = simd_shuffle(zpre, (ushort)(10u + seg));
    return simd_bfly2_3329(v0, v1, (ushort)4, ((lane & 4u) == 0u), z0, z1);
}

inline uint2 simd_step_3329_pre2(uint v0, uint v1, uint lane, uint zpre) {
    uint seg = (lane >> 2u) & 7u;
    uint z0 = simd_shuffle(zpre, (ushort)(14u + seg));
    uint z1 = simd_shuffle(zpre, (ushort)(22u + seg));
    return simd_bfly2_3329(v0, v1, (ushort)2, ((lane & 2u) == 0u), z0, z1);
}

inline uint2 simd_step_3329(uint v0,
                            uint v1,
                            uint base0,
                            uint lane,
                            uint length,
                            uint k_start,
                            uint pair_shift,
                            device const uint *zetas) {
    uint pair_mask = (length << 1u) - 1u;
    uint source_lane = lane & ~pair_mask;

    uint zload0 = 0u;
    uint zload1 = 0u;
    if (lane == source_lane) {
        zload0 = zetas[k_start + ((base0 + source_lane) >> pair_shift)];
        zload1 = zetas[k_start + ((base0 + 32u + source_lane) >> pair_shift)];
    }

    uint z0 = simd_shuffle(zload0, (ushort)source_lane);
    uint z1 = simd_shuffle(zload1, (ushort)source_lane);

    return simd_bfly2_3329(v0, v1, (ushort)length, ((lane & length) == 0u), z0, z1);
}

// Fast path for q = 3329, n = 256, n_levels = 7.
// Stages len=128 and len=64 are fused in a swizzled SIMD layout;
// one threadgroup barrier/transposition feeds the remaining register stages.
inline void ntt_256_3329_l7(threadgroup uint *a,
                            device uint *poly,
                            device const uint *zetas,
                            uint ltid) {
    uint lane = ltid & 31u;
    uint sg   = ltid >> 5u;

    uint zpre = preload_late_zeta_3329(zetas, sg, lane);

    // Swizzle each simdgroup as 16 butterflies from the lower 64 half
    // and their +64 partners, so level 1 is an intra-simdgroup xor-16.
    uint base16 = sg << 4u;
    uint j = base16 + (lane & 15u) + ((lane & 16u) << 2u);

    uint z1load = (lane == 0u) ? zetas[1u] : 0u;
    uint z1 = simd_broadcast(z1load, (ushort)0);

    uint x = poly[j];
    uint y = poly[j + 128u];

    uint t = mod_mul_3329(z1, y);
    uint pos = add_3329(x, t);
    uint neg = sub_3329(x, t);

    uint z23load = (lane < 2u) ? zetas[2u + lane] : 0u;
    uint z2 = simd_broadcast(z23load, (ushort)0);
    uint z3 = simd_broadcast(z23load, (ushort)1);

    uint2 r = simd_bfly2_3329(pos, neg, (ushort)16, (lane < 16u), z2, z3);

    a[j] = r.x;
    a[j + 128u] = r.y;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage len = 32, then all remaining stages are intra-32-lane chunks.
    uint base0 = sg << 6u;
    uint base1 = base0 + 32u;

    uint z4 = bcast_zeta_lane0(zetas, 4u + sg, lane);

    x = a[base0 + lane];
    y = a[base1 + lane];

    t = mod_mul_3329(z4, y);
    uint v0 = add_3329(x, t);
    uint v1 = sub_3329(x, t);

    r = simd_step_3329_pre16(v0, v1, lane, zpre);
    v0 = r.x; v1 = r.y;

    r = simd_step_3329_pre8(v0, v1, lane, zpre);
    v0 = r.x; v1 = r.y;

    r = simd_step_3329_pre4(v0, v1, lane, zpre);
    v0 = r.x; v1 = r.y;

    r = simd_step_3329_pre2(v0, v1, lane, zpre);
    v0 = r.x; v1 = r.y;

    poly[base0 + lane] = v0;
    poly[base1 + lane] = v1;
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

    threadgroup uint a[N_MAX];

    device uint *poly = coeffs + (size_t)tgid * (size_t)n;

    if ((q == 3329u) && (n == 256u) && (n_levels == 7u)) {
        ntt_256_3329_l7(a, poly, zetas, ltid);
        return;
    }

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

Result of previous attempt:
            kyb_B1: correct, 0.02 ms, 0.0 Gmodmul/s (u32) (0.1% of 53 Gops/s (int64 mul, est))
           kyb_B16: correct, 0.02 ms, 0.7 Gmodmul/s (u32) (1.4% of 53 Gops/s (int64 mul, est))
          kyb_B256: correct, 0.04 ms, 5.4 Gmodmul/s (u32) (10.1% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0106

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_MAX = 256u;

inline uint add_3329(uint a, uint b) {
    uint s = a + b;
    return (s >= 3329u) ? (s - 3329u) : s;
}

inline uint sub_3329(uint a, uint b) {
    uint d = a - b;
    return (a < b) ? (d + 3329u) : d;
}

inline uint mod_mul_3329(uint a, uint b) {
    // Exact for a,b < 3329. floor(2^32 / 3329) = 1290167.
    uint x = a * b;
    uint qhat = mulhi(x, 1290167u);
    uint r = x - qhat * 3329u;
    if (r >= 3329u) r -= 3329u;
    return r;
}

inline uint add_8380417(uint a, uint b) {
    uint s = a + b;
    return (s >= 8380417u) ? (s - 8380417u) : s;
}

inline uint sub_8380417(uint a, uint b) {
    uint d = a - b;
    return (a < b) ? (d + 8380417u) : d;
}

inline uint mod_mul_8380417(uint a, uint b) {
    // 8380417 = 2^23 - 8191, so 2^23 == 8191 (mod q).
    ulong x = (ulong)a * (ulong)b;
    ulong r = (x & 0x7ffffful) + ((x >> 23) * 8191ul);
    r = (r & 0x7ffffful) + ((r >> 23) * 8191ul);
    uint s = (uint)((r & 0x7ffffful) + ((r >> 23) * 8191ul));
    if (s >= 8380417u) s -= 8380417u;
    return s;
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

inline uint2 simd_step_3329(uint v0,
                            uint v1,
                            uint base0,
                            uint lane,
                            uint length,
                            uint k_start,
                            uint pair_shift,
                            device const uint *zetas) {
    uint pair_mask = (length << 1u) - 1u;
    uint source_lane = lane & ~pair_mask;

    uint zload0 = 0u;
    uint zload1 = 0u;
    if (lane == source_lane) {
        zload0 = zetas[k_start + ((base0 + source_lane) >> pair_shift)];
        zload1 = zetas[k_start + ((base0 + 32u + source_lane) >> pair_shift)];
    }

    uint z0 = simd_shuffle(zload0, (ushort)source_lane);
    uint z1 = simd_shuffle(zload1, (ushort)source_lane);

    uint p0 = simd_shuffle_xor(v0, (ushort)length);
    uint p1 = simd_shuffle_xor(v1, (ushort)length);

    bool lower = ((lane & length) == 0u);

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

    uint from_lower0 = simd_shuffle_xor(d0, (ushort)length);
    uint from_lower1 = simd_shuffle_xor(d1, (ushort)length);

    uint nv0 = lower ? s0 : from_lower0;
    uint nv1 = lower ? s1 : from_lower1;
    return uint2(nv0, nv1);
}

// Fast path for the FIPS/PQClean Kyber forward NTT shape:
// q = 3329, n = 256, n_levels = 7, stages len = 128..2.
inline void ntt_256_3329_l7(threadgroup uint *a,
                            device uint *poly,
                            device const uint *zetas,
                            uint ltid) {
    uint lane = ltid & 31u;
    uint sg   = ltid >> 5u;

    // level 0, len = 128
    uint z1 = bcast_zeta_lane0(zetas, 1u, lane);
    uint x = poly[ltid];
    uint y = poly[ltid + 128u];
    uint t = mod_mul_3329(z1, y);
    a[ltid] = add_3329(x, t);
    a[ltid + 128u] = sub_3329(x, t);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // level 1, len = 64
    uint g1 = ltid >> 6u;
    uint j1 = (g1 << 7u) | (ltid & 63u);
    uint z2 = bcast_zeta_lane0(zetas, 2u + g1, lane);

    x = a[j1];
    y = a[j1 + 64u];
    t = mod_mul_3329(z2, y);
    a[j1] = add_3329(x, t);
    a[j1 + 64u] = sub_3329(x, t);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // level 2, len = 32.  Move into registers as two 32-coefficient chunks
    // per simdgroup; all later levels are intra-simdgroup.
    uint base0 = sg << 6u;
    uint base1 = base0 + 32u;
    uint z4 = bcast_zeta_lane0(zetas, 4u + sg, lane);

    x = a[base0 + lane];
    y = a[base1 + lane];
    t = mod_mul_3329(z4, y);

    uint v0 = add_3329(x, t);
    uint v1 = sub_3329(x, t);

    uint2 r;

    // level 3, len = 16, k = 8
    r = simd_step_3329(v0, v1, base0, lane, 16u, 8u, 5u, zetas);
    v0 = r.x; v1 = r.y;

    // level 4, len = 8, k = 16
    r = simd_step_3329(v0, v1, base0, lane, 8u, 16u, 4u, zetas);
    v0 = r.x; v1 = r.y;

    // level 5, len = 4, k = 32
    r = simd_step_3329(v0, v1, base0, lane, 4u, 32u, 3u, zetas);
    v0 = r.x; v1 = r.y;

    // level 6, len = 2, k = 64
    r = simd_step_3329(v0, v1, base0, lane, 2u, 64u, 2u, zetas);
    v0 = r.x; v1 = r.y;

    poly[base0 + lane] = v0;
    poly[base1 + lane] = v1;
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

    threadgroup uint a[N_MAX];
    threadgroup uint zs[N_MAX];

    device uint *poly = coeffs + (size_t)tgid * (size_t)n;

    if ((q == 3329u) && (n == 256u) && (n_levels == 7u)) {
        ntt_256_3329_l7(a, poly, zetas, ltid);
        return;
    }

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

Incumbent result:
            kyb_B1: correct, 0.02 ms, 0.0 Gmodmul/s (u32) (0.1% of 53 Gops/s (int64 mul, est))
           kyb_B16: correct, 0.02 ms, 0.7 Gmodmul/s (u32) (1.4% of 53 Gops/s (int64 mul, est))
          kyb_B256: correct, 0.04 ms, 5.6 Gmodmul/s (u32) (10.5% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0108

## History

- iter  0: compile=OK | correct=True | score=0.003669715246498924
- iter  1: compile=OK | correct=True | score=0.010244118344773873
- iter  2: compile=OK | correct=True | score=0.010786251824175287
- iter  3: compile=FAIL | correct=False | score=N/A
- iter  4: compile=OK | correct=True | score=0.009906472983308016
- iter  5: compile=OK | correct=True | score=0.010598012872478885

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
