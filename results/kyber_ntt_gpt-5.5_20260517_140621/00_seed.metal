// Naive seed for a batched negacyclic NTT (Z6, forward).
//
// One threadgroup per polynomial: each threadgroup runs all n_levels
// Cooley-Tukey butterfly stages in threadgroup memory and writes back.
// Per-stage barrier flushes the in-place updates so the next stage
// reads consistent values.
//
// Convention (matches the FIPS 203 / FIPS 204 / pqclean references):
//
//   k = 1
//   for level = 0..n_levels:
//       len = n >> (level + 1)
//       for start = 0, 2*len, ..., n - 2*len:
//           zeta = zetas[k++]
//           for j = start..start + len - 1:
//               t          = (zeta * a[j+len]) mod q
//               a[j+len]   = (a[j] - t)        mod q
//               a[j]       = (a[j] + t)        mod q
//
// Equivalent per-thread mapping (ltid in [0, n/2) owns one butterfly
// at every level):
//
//   group_idx   = ltid / len
//   j_in_group  = ltid - group_idx * len      // ltid % len
//   j           = (group_idx << 1) * len + j_in_group
//   zeta_index  = k_start + group_idx          // k_start = 1 << level
//
// Zetas table (host-precomputed, length 2^n_levels):
//   zetas[k] = zeta^bit_reverse(k, n_levels)   mod q
// where zeta is a primitive 2^(n_levels+1)-th root of unity in F_q.
// The concrete (q, n_levels, zeta) values are bound at runtime through
// the q and n_levels constant buffers and the zetas device buffer;
// the kernel does not need to know which parameter set is in play.
// Entry zetas[0] = 1 is the unread identity element (k starts at 1).
//
// Buffer layout (host-fixed; must be preserved by candidate kernels):
//   buffer 0: device       uint *coeffs       (length batch * n;
//             read+written in place)
//   buffer 1: device const uint *zetas        (length 1 << n_levels)
//   buffer 2: constant uint     &q            (modulus; 3329 or 8380417)
//   buffer 3: constant uint     &n            (polynomial length; 256)
//   buffer 4: constant uint     &n_levels     (number of NTT stages)
//   buffer 5: constant uint     &batch
//
// Dispatch (host-provided):
//   threadsPerGrid        = (batch * (n/2), 1, 1)
//   threadsPerThreadgroup = (n/2, 1, 1)
// Each threadgroup owns ONE polynomial; ltid in [0, n/2) owns one
// butterfly per stage.
//
// Outputs MUST be canonical ([0, q)); a non-canonical value with the
// same residue class still counts as a mismatch on the host-side
// reference comparison. n is a power of two with n <= 256 and
// n_levels <= 8 across every test size, so n/2 <= 128 threads per
// threadgroup and a static scratch of size 256 cover every case.

#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_MAX = 256u;

inline uint mod_reduce(ulong t, uint q) {
    return (uint)(t % (ulong)q);
}

inline uint mod_mul(uint a, uint b, uint q) {
    return mod_reduce((ulong)a * (ulong)b, q);
}

inline uint mod_add(uint a, uint b, uint q) {
    uint t = a + b;
    return (t >= q) ? (t - q) : t;
}

inline uint mod_sub(uint a, uint b, uint q) {
    return (a >= b) ? (a - b) : (a + q - b);
}

kernel void kyber_ntt(
    device       uint *coeffs     [[buffer(0)]],
    device const uint *zetas      [[buffer(1)]],
    constant uint     &q           [[buffer(2)]],
    constant uint     &n           [[buffer(3)]],
    constant uint     &n_levels    [[buffer(4)]],
    constant uint     &batch       [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ltid [[thread_position_in_threadgroup]])
{
    if (tgid >= batch) return;

    threadgroup uint a[N_MAX];

    uint half_n = n >> 1u;
    device uint *poly = coeffs + (size_t)tgid * n;

    // Each thread loads two coefficients into threadgroup memory.
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length  = half_n;          // halves each stage
    uint k_start = 1u;              // index into zetas for the first group
    for (uint level = 0u; level < n_levels; ++level) {
        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zeta       = zetas[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];
        uint t = mod_mul(zeta, y, q);

        a[j]          = mod_add(x, t, q);
        a[j + length] = mod_sub(x, t, q);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length  >>= 1u;
    }

    // Writeback.
    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}
