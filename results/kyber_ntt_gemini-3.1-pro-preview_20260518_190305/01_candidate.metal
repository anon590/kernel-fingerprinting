#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_MAX = 256u;

// Fast 32-bit Barrett reduction for Y < 2^32.
// Requires precomputed mu = floor(2^32 / q).
inline uint mod32(uint Y, uint q, uint mu) {
    uint q_est = mulhi(Y, mu);
    uint rem = Y - q_est * q;
    // Mathematical guarantee: error in q_est is at most 1, so rem < 2q.
    return (rem >= q) ? (rem - q) : rem;
}

// Computes (a * b) % q efficiently without 64-bit division.
inline uint mod_mul(uint a, uint b, uint q, uint mu, uint R) {
    ulong P = (ulong)a * b;
    uint P_hi = (uint)(P >> 32);
    uint P_lo = (uint)P;

    // Y = P_hi * R. Decompose Y into x_hi * 2^32 + x_lo
    uint x_hi = mulhi(P_hi, R);
    uint x_lo = P_hi * R;

    // Sum the lower parts and catch the carry (each carry represents 2^32 = R mod q)
    uint sum_lo = x_lo + P_lo;
    uint carry = (sum_lo < x_lo) ? 1u : 0u;

    uint sum_hi_R = (x_hi + carry) * R;

    uint sum_lo_mod = mod32(sum_lo, q, mu);
    uint total_sum = sum_hi_R + sum_lo_mod;

    return mod32(total_sum, q, mu);
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
    constant uint     &q          [[buffer(2)]],
    constant uint     &n          [[buffer(3)]],
    constant uint     &n_levels   [[buffer(4)]],
    constant uint     &batch      [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ltid [[thread_position_in_threadgroup]])
{
    if (tgid >= batch) return;

    // Load kernel-wide scalars into fast registers
    uint local_q        = q;
    uint local_n        = n;
    uint local_n_levels = n_levels;

    // Precompute constants for the fast 64-bit to 32-bit reduction
    uint mu = (uint)(0x100000000ull / local_q);
    uint R  = (uint)(0x100000000ull % local_q);

    threadgroup uint a[N_MAX];
    threadgroup uint shared_zetas[256];

    uint half_n = local_n >> 1u;
    device uint *poly = coeffs + (size_t)tgid * local_n;

    // Each thread loads two coefficients into threadgroup memory
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];

    // Cooperatively load zetas twiddles into threadgroup memory
    uint zetas_count = 1u << local_n_levels;
    for (uint i = ltid; i < zetas_count; i += half_n) {
        shared_zetas[i] = zetas[i];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint log_len = ctz(half_n);
    uint k_start = 1u;

    // Batched Cooley-Tukey NTT
    for (uint level = 0u; level < local_n_levels; ++level) {
        uint length     = 1u << log_len;
        
        // Fast power-of-two arithmetic (replaces hardware division and modulo)
        uint group_idx  = ltid >> log_len;
        uint j          = (group_idx << (log_len + 1u)) | (ltid & (length - 1u));
        uint zeta       = shared_zetas[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];
        uint t = mod_mul(zeta, y, local_q, mu, R);

        a[j]          = mod_add(x, t, local_q);
        a[j + length] = mod_sub(x, t, local_q);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        log_len--;
    }

    // Write-back out of place to VRAM
    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}