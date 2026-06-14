**Optimization:** The previous attempt failed to compile due to a stray placeholder. The incumbent uses runtime `t % q` which is extremely slow on GPU (uses software division). I'll replace it with **Barrett reduction** using a host-independent precomputed reciprocal computed once per threadgroup: `m = floor(2^64 / q)` via `ulong` division done by one thread and broadcast through threadgroup memory. Then `mod_mul` uses `mulhi`-style 128-bit emulation: compute `t = a*b` (≤ 2^46 since q < 2^23), and `q_est = ((t_hi * m) + ((t_lo * m) >> 32)) >> some_shift`. Actually simpler: since `t < 2^46` and `m < 2^53`, we can compute `q_est = mulhi64(t, m)` then correct. Even simpler and correct: use **Barrett with 32-bit** — since `q < 2^23`, precompute `m = floor(2^45 / q)` (fits in 32 bits when q ≥ 2^22, else larger). I'll use a robust formulation: precompute `m = floor(2^64 / q)` (ulong), and compute `q_est` via 64×64→high64 emulation from two 32-bit halves. This avoids the slow modulo entirely in the hot loop.

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_MAX = 256u;

// Compute high 64 bits of (a * b) where a, b are 64-bit unsigned.
inline ulong mulhi64(ulong a, ulong b) {
    ulong a_lo = a & 0xFFFFFFFFul;
    ulong a_hi = a >> 32;
    ulong b_lo = b & 0xFFFFFFFFul;
    ulong b_hi = b >> 32;

    ulong ll = a_lo * b_lo;
    ulong lh = a_lo * b_hi;
    ulong hl = a_hi * b_lo;
    ulong hh = a_hi * b_hi;

    ulong mid = (ll >> 32) + (lh & 0xFFFFFFFFul) + (hl & 0xFFFFFFFFul);
    ulong hi  = hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
    return hi;
}

// Barrett reduction: t mod q, given m = floor(2^64 / q), t < 2^46, q < 2^23.
// q_est = mulhi64(t, m); r = t - q_est * q; if (r >= q) r -= q;
// (May need one or two subtractions in the worst case.)
inline uint barrett_reduce(ulong t, uint q, ulong m) {
    ulong q_est = mulhi64(t, m);
    ulong r = t - q_est * (ulong)q;
    // r can be up to ~2*q in the worst case; subtract q if needed.
    if (r >= (ulong)q) r -= (ulong)q;
    if (r >= (ulong)q) r -= (ulong)q;
    return (uint)r;
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
    threadgroup uint zs[N_MAX];      // up to 256 zetas
    threadgroup ulong m_shared[1];

    uint half_n = n >> 1u;
    uint qv = q;

    // Thread 0 precomputes Barrett constant m = floor(2^64 / q).
    if (ltid == 0u) {
        // Compute 2^64 / q by computing (0 - q) / q + 1 trick:
        // floor(2^64 / q) = floor((2^64 - 1) / q)  when q does not divide 2^64.
        // For our primes (3329, 8380417) this holds.
        ulong num = ~(ulong)0;          // 2^64 - 1
        m_shared[0] = num / (ulong)qv;
    }

    device uint *poly = coeffs + (size_t)tgid * n;

    // Cooperative load of coefficients
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];

    // Cooperative load of zetas
    uint z_count = 1u << n_levels;
    if (ltid < z_count) {
        zs[ltid] = zetas[ltid];
    }
    uint idx2 = ltid + half_n;
    if (idx2 < z_count) {
        zs[idx2] = zetas[idx2];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ulong m = m_shared[0];

    uint length  = half_n;
    uint k_start = 1u;
    for (uint level = 0u; level < n_levels; ++level) {
        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zeta       = zs[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];
        ulong prod = (ulong)zeta * (ulong)y;
        uint t = barrett_reduce(prod, qv, m);

        uint sum  = mod_add(x, t, qv);
        uint diff = mod_sub(x, t, qv);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        a[j]          = sum;
        a[j + length] = diff;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length  >>= 1u;
    }

    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}
```