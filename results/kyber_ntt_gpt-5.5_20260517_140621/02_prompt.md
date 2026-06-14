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
    // Exact for a,b < 3329.  floor(2^32 / 3329) = 1290167.
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

    uint half_n = n >> 1u;
    device uint *poly = coeffs + (size_t)tgid * (size_t)n;

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
           kyb_B16: correct, 0.02 ms, 0.7 Gmodmul/s (u32) (1.3% of 53 Gops/s (int64 mul, est))
          kyb_B256: correct, 0.04 ms, 5.9 Gmodmul/s (u32) (11.0% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0102

## History

- iter  0: compile=OK | correct=True | score=0.003669715246498924
- iter  1: compile=OK | correct=True | score=0.010244118344773873

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
