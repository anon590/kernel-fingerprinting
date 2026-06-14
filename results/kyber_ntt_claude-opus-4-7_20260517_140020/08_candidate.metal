#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_MAX = 256u;

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
    threadgroup uint a[N_MAX];
    threadgroup uint zcache[N_MAX];
    threadgroup uint zshoup[N_MAX];

    const uint qv     = q;
    const uint nv     = n;
    const uint nlv    = n_levels;
    const uint half_n = nv >> 1u;
    const uint zlen   = 1u << nlv;

    device uint *poly = coeffs + (size_t)tgid * (size_t)nv;

    // Load coefficients.
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];

    // Load zetas and precompute Shoup constants: w' = floor(w * 2^32 / q).
    // Each thread does up to two entries (since zlen <= 256, half_n <= 128).
    {
        uint z0 = zetas[ltid];
        zcache[ltid] = z0;
        zshoup[ltid] = (uint)(((ulong)z0 << 32) / (ulong)qv);
        uint idx1 = ltid + half_n;
        if (idx1 < zlen) {
            uint z1 = zetas[idx1];
            zcache[idx1] = z1;
            zshoup[idx1] = (uint)(((ulong)z1 << 32) / (ulong)qv);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length   = half_n;
    uint log2_len = (half_n == 0u) ? 0u : (31u - clz(half_n));
    uint k_start  = 1u;

    // Phase 1: threadgroup-cooperative levels (length >= 32).
    // Apple simd width is 32; for length < 32 each butterfly pair lies within
    // a single simdgroup and we can use shuffles.
    uint level = 0u;
    for (; level < nlv && length >= 32u; ++level) {
        uint mask       = length - 1u;
        uint group_idx  = ltid >> log2_len;
        uint j_in_group = ltid & mask;
        uint j          = (group_idx << (log2_len + 1u)) + j_in_group;

        uint zeta  = zcache[k_start + group_idx];
        uint zsh   = zshoup[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];

        // Shoup: q_hat = mulhi(zsh, y); r = zeta*y - q_hat*q; r in [0, 2q).
        uint q_hat = mulhi(zsh, y);
        uint r     = zeta * y - q_hat * qv;
        if (r >= qv) r -= qv;

        uint sum = x + r;       sum = (sum >= qv) ? (sum - qv) : sum;
        uint dif = x + qv - r;  dif = (dif >= qv) ? (dif - qv) : dif;

        a[j]          = sum;
        a[j + length] = dif;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length  >>= 1u;
        log2_len--;
    }

    // Phase 2: simdgroup-resident levels (length < 32).
    // Load register state: each thread keeps a register-pair derived from a[].
    // We process the remaining levels using simd_shuffle_xor.
    if (level < nlv) {
        // Reload values for this thread's two positions from threadgroup mem.
        // We operate per-pair, where each thread owns butterfly (j, j+length).
        // Now length < 32, so j and j+length lie in the same simdgroup of 32.
        // Strategy: each thread holds a single coefficient a[ltid'], and we
        // pair via shuffle. Reorganize: thread t owns a[2t] and a[2t+1] initially
        // ... but pairing changes per level. Simpler: each thread owns one coefficient
        // at position p = ltid (for ltid < n). Since we have only half_n threads,
        // each thread owns TWO coeffs: a[ltid] (low) and a[ltid + half_n] (high).
        // The simd pairing is across the two registers locally and across lanes via shuffle.
        //
        // Simpler reliable fallback: keep threadgroup memory + barriers but skip
        // barriers via simd. We still issue barriers; the inner work is tiny.
        for (; level < nlv; ++level) {
            uint mask       = length - 1u;
            uint group_idx  = ltid >> log2_len;
            uint j_in_group = ltid & mask;
            uint j          = (group_idx << (log2_len + 1u)) + j_in_group;

            uint zeta  = zcache[k_start + group_idx];
            uint zsh   = zshoup[k_start + group_idx];

            uint x = a[j];
            uint y = a[j + length];

            uint q_hat = mulhi(zsh, y);
            uint r     = zeta * y - q_hat * qv;
            if (r >= qv) r -= qv;

            uint sum = x + r;       sum = (sum >= qv) ? (sum - qv) : sum;
            uint dif = x + qv - r;  dif = (dif >= qv) ? (dif - qv) : dif;

            a[j]          = sum;
            a[j + length] = dif;

            simdgroup_barrier(mem_flags::mem_threadgroup);

            k_start <<= 1u;
            length  >>= 1u;
            log2_len = (log2_len == 0u) ? 0u : (log2_len - 1u);
        }
    }

    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}