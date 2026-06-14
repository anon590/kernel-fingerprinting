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

template<uint Q, uint N_LEVELS>
inline void ntt_core_unrolled(threadgroup uint* a, device const uint* zetas, uint half_n, uint ltid) {
    uint length_shift = ctz(half_n);
    
    #pragma unroll
    for (uint level = 0u; level < N_LEVELS; ++level) {
        uint length     = 1u << length_shift;
        uint group_idx  = ltid >> length_shift;
        uint j_in_group = ltid & (length - 1u);
        uint j          = (group_idx << (length_shift + 1u)) | j_in_group;
        uint zeta       = zetas[(1u << level) + group_idx];

        uint x = a[j];
        uint y = a[j + length];
        
        uint t;
        if (Q <= 65535u) {
            t = (zeta * y) % Q;
        } else {
            t = (uint)(((ulong)zeta * y) % Q);
        }

        uint t_add = x + t;
        a[j]          = (t_add >= Q) ? (t_add - Q) : t_add;
        a[j + length] = (x >= t) ? (x - t) : (x + Q - t);

        if (length > 32u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        length_shift--;
    }
}

inline void ntt_core_dynamic(threadgroup uint* a, device const uint* zetas, uint n_levels_val, uint half_n, uint ltid, uint q_val) {
    uint length_shift = ctz(half_n);
    for (uint level = 0u; level < n_levels_val; ++level) {
        uint length     = 1u << length_shift;
        uint group_idx  = ltid >> length_shift;
        uint j_in_group = ltid & (length - 1u);
        uint j          = (group_idx << (length_shift + 1u)) | j_in_group;
        uint zeta       = zetas[(1u << level) + group_idx];

        uint x = a[j];
        uint y = a[j + length];
        
        uint t = (uint)(((ulong)zeta * y) % q_val);

        uint t_add = x + t;
        a[j]          = (t_add >= q_val) ? (t_add - q_val) : t_add;
        a[j + length] = (x >= t) ? (x - t) : (x + q_val - t);

        if (length > 32u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        length_shift--;
    }
}

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

    uint n_val = n;
    uint half_n = n_val >> 1u;
    
    // Use uint2 for 100% coalesced and halved memory instructions.
    // coeffs + offset is safely 8-byte aligned because n is a power of 2 (>= 2).
    device uint2 *poly2 = (device uint2 *)(coeffs + (size_t)tgid * n_val);
    threadgroup uint2 *a2 = (threadgroup uint2 *)a;

    // Load coefficients into fast threadgroup memory
    a2[ltid] = poly2[ltid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint q_val = q;
    uint n_levels_val = n_levels;

    // Dispatch to cleanly unrolled templates to heavily bias the compiler arithmetic
    // and completely eliminate loop overhead and runtime barrier branching.
    if (n_val == 256u) {
        if (q_val == 3329u && n_levels_val == 7u) {
            ntt_core_unrolled<3329u, 7u>(a, zetas, 128u, ltid);
        } else if (q_val == 8380417u && n_levels_val == 8u) {
            ntt_core_unrolled<8380417u, 8u>(a, zetas, 128u, ltid);
        } else if (q_val == 3329u && n_levels_val == 8u) {
            ntt_core_unrolled<3329u, 8u>(a, zetas, 128u, ltid);
        } else if (q_val == 8380417u && n_levels_val == 7u) {
            ntt_core_unrolled<8380417u, 7u>(a, zetas, 128u, ltid);
        } else {
            ntt_core_dynamic(a, zetas, n_levels_val, 128u, ltid, q_val);
        }
    } else {
        ntt_core_dynamic(a, zetas, n_levels_val, half_n, ltid, q_val);
    }

    // Write outputs back to global memory
    // (A threadgroup_barrier is omitted here because for lengths <= 32
    //  the SIMD group perfectly isolates its memory reads/writes, meaning
    //  a2[ltid] exactly matches the elements the SIMD wrote in the final step).
    poly2[ltid] = a2[ltid];
}
```

Result of previous attempt:
            kyb_B1: correct, 0.02 ms, 0.1 GB/s (0.1% of 200 GB/s)
           kyb_B16: correct, 0.02 ms, 2.2 GB/s (1.1% of 200 GB/s)
          kyb_B256: correct, 0.03 ms, 22.8 GB/s (11.4% of 200 GB/s)
  score (gmean of fraction): 0.0096

## History

- iter  0: compile=OK | correct=True | score=0.003301613412984647
- iter  1: compile=OK | correct=True | score=0.00900193407898856
- iter  2: compile=FAIL | correct=False | score=N/A
- iter  3: compile=OK | correct=True | score=0.009567304668001356

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
