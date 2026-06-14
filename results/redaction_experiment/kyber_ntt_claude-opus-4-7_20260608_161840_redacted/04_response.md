I'll acknowledge the system reminder about keybindings-help skill (not relevant here) and focus on the NTT task.

**Optimization strategy:** The main bottleneck is the per-stage `threadgroup_barrier` — for stages where `length < 32`, all cooperating threads fall within one simdgroup, so we can use `simdgroup_barrier` instead. Additionally, the expensive `% q` operation dominates the modmul. Since `q < 2^32` and `a, zeta < q < 2^32`, the product fits in 64 bits, but we can replace `%` with a Barrett-style reduction using a precomputed reciprocal `q_inv = (2^64) / q`, computed once per thread. This avoids the hardware divide. I'll also keep registers hot by caching `q` locally and use unsigned compare tricks. For the last few stages (length ≤ 16), drop to `simdgroup_barrier`.

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_MAX = 256u;

inline uint mod_add(uint a, uint b, uint q) {
    uint t = a + b;
    return (t >= q) ? (t - q) : t;
}

inline uint mod_sub(uint a, uint b, uint q) {
    return (a >= b) ? (a - b) : (a + q - b);
}

// Barrett-like reduction using a precomputed 64-bit reciprocal m = floor(2^64 / q).
// Given x = a*b with a,b < q < 2^32, x < 2^64.
// Estimate quotient: qh = mulhi64(x, m); then r = x - qh*q; correct r if r >= q.
inline uint mod_mul_barrett(uint a, uint b, uint q, ulong m) {
    ulong x = (ulong)a * (ulong)b;
    // mulhi for 64x64: use Metal's mulhi only exists for 32-bit. Compute via split.
    // x_hi*m + x_lo*m high half.
    ulong x_hi = x >> 32;
    ulong x_lo = x & 0xFFFFFFFFul;
    ulong m_hi = m >> 32;
    ulong m_lo = m & 0xFFFFFFFFul;
    ulong ll = x_lo * m_lo;
    ulong lh = x_lo * m_hi;
    ulong hl = x_hi * m_lo;
    ulong hh = x_hi * m_hi;
    ulong mid = (ll >> 32) + (lh & 0xFFFFFFFFul) + (hl & 0xFFFFFFFFul);
    ulong qh = hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
    ulong r = x - qh * (ulong)q;
    // r may be in [0, 2q); correct twice to be safe (Barrett can be off by up to ~2).
    if (r >= (ulong)q) r -= (ulong)q;
    if (r >= (ulong)q) r -= (ulong)q;
    return (uint)r;
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

    uint qreg = q;
    uint nreg = n;
    uint nlv  = n_levels;
    uint half_n = nreg >> 1u;

    // Precomputed Barrett reciprocal m = floor(2^64 / q).
    // For q != 1, compute as: m = (~0ull) / q  (gives floor((2^64 - 1)/q));
    // adjust: if (~0ull) - m*q == q-1 then floor(2^64/q) = m+1, else m.
    ulong neg1 = (ulong)0xFFFFFFFFFFFFFFFFul;
    ulong m = neg1 / (ulong)qreg;
    ulong rem = neg1 - m * (ulong)qreg;
    if (rem == (ulong)(qreg - 1u)) m = m + 1ul;

    device uint *poly = coeffs + (size_t)tgid * nreg;

    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length  = half_n;
    uint k_start = 1u;

    for (uint level = 0u; level < nlv; ++level) {
        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zeta       = zetas[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];
        uint t = mod_mul_barrett(zeta, y, qreg, m);

        a[j]          = mod_add(x, t, qreg);
        a[j + length] = mod_sub(x, t, qreg);

        // Once length <= 16, the pairs (j, j+length) for every active
        // thread all lie within a single 32-lane simdgroup span of
        // threadgroup memory, so a simd-level barrier suffices.
        if (length > 16u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        k_start <<= 1u;
        length  >>= 1u;
    }

    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}
```