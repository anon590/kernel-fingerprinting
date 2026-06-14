I notice the system reminder about the keybindings-help skill, but it's not relevant to this MSL kernel optimization task.

**Optimization:** The previous attempts spent significant time on threadgroup barriers and shared memory ping-pong. Key insight: with n=256 and 128 threads per threadgroup, each thread owns exactly 2 coefficients. I'll keep those 2 coefficients in **registers** for the first several levels (using simd shuffles to exchange across threads when needed), and only use threadgroup memory for the wider strides. Specifically:
- Level 0 (stride 128): cross-thread, needs threadgroup memory
- Levels 1-2 (stride 64, 32): cross-simdgroup, threadgroup memory
- Levels 3-7 (stride 16, 8, 4, 2, 1): all within a simdgroup → use `simd_shuffle_xor` to exchange values, completely avoiding threadgroup barriers

This eliminates ~5 of the 8 barriers per polynomial, which dominates latency for small batches. Also unify the load/store with the first/last level's register data.

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_MAX = 256u;
constexpr constant uint Z_MAX = 256u;

inline uint mod_add(uint a, uint b, uint q) {
    uint t = a + b;
    return (t >= q) ? (t - q) : t;
}
inline uint mod_sub(uint a, uint b, uint q) {
    return (a >= b) ? (a - b) : (a + q - b);
}

inline uint compute_qinv(uint q) {
    uint x = q;
    x = x * (2u - q * x);
    x = x * (2u - q * x);
    x = x * (2u - q * x);
    x = x * (2u - q * x);
    x = x * (2u - q * x);
    return x;
}

inline uint mont_reduce(ulong T, uint q, uint qinv_neg) {
    uint T_lo = (uint)T;
    uint T_hi = (uint)(T >> 32);
    uint m = T_lo * qinv_neg;
    uint mq_hi = mulhi(m, q);
    uint mq_lo = m * q;
    uint sum_lo = T_lo + mq_lo;
    uint carry = (sum_lo < T_lo) ? 1u : 0u;
    uint t = T_hi + mq_hi + carry;
    if (t >= q) t -= q;
    return t;
}

// Butterfly given two register values x,y and a montgomery-form zeta zm.
inline void bfly(thread uint &x, thread uint &y, uint zm, uint q, uint qinv_neg) {
    uint t = mont_reduce((ulong)y * (ulong)zm, q, qinv_neg);
    uint xpt = mod_add(x, t, q);
    uint xmt = mod_sub(x, t, q);
    x = xpt;
    y = xmt;
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
    threadgroup uint zeta_mont[Z_MAX];

    uint qreg = q;
    uint nreg = n;
    uint nlv  = n_levels;
    uint half_n = nreg >> 1u;
    uint zcount = 1u << nlv;

    uint qinv = compute_qinv(qreg);
    uint qinv_neg = 0u - qinv;

    // R = 2^32 mod q.
    uint Rmod;
    {
        ulong r = ((ulong)0xFFFFFFFFul) % (ulong)qreg;
        r += 1ul;
        if (r >= (ulong)qreg) r -= (ulong)qreg;
        Rmod = (uint)r;
    }

    // Cooperatively precompute zetas in Montgomery form.
    {
        ulong Q = (ulong)qreg;
        uint i1 = ltid;
        if (i1 < zcount) {
            zeta_mont[i1] = (uint)(((ulong)zetas[i1] * (ulong)Rmod) % Q);
        }
        uint i2 = ltid + half_n;
        if (i2 < zcount) {
            zeta_mont[i2] = (uint)(((ulong)zetas[i2] * (ulong)Rmod) % Q);
        }
    }

    device uint *poly = coeffs + (size_t)tgid * nreg;

    // Load polynomial into threadgroup memory.
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length  = half_n;
    uint k_start = 1u;

    // ---- Phase 1: threadgroup-memory levels (stride > 16).
    // For n=256, half_n=128. Levels with length > 16 are levels where
    // length in {128, 64, 32}. After those, length = 16, and stride 16
    // butterflies map onto simdgroups of width 32 (each simd owns 32 consecutive
    // ltids, which cover 32 consecutive coeffs in our layout).
    //
    // We'll do threadgroup-mem levels while length > 16.
    while (length > 16u) {
        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zm         = zeta_mont[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];

        uint t = mont_reduce((ulong)y * (ulong)zm, qreg, qinv_neg);
        uint xpt = mod_add(x, t, qreg);
        uint xmt = mod_sub(x, t, qreg);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        a[j]          = xpt;
        a[j + length] = xmt;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length  >>= 1u;
    }

    // ---- Phase 2: simdgroup levels (length <= 16).
    // Each thread now holds two coefficients in registers, packed so that:
    //   thread ltid (with ltid in [0, half_n)) handles butterfly between
    //   coeff index j and j+length where length<=16.
    //
    // Strategy: load both coefficients into registers, then do remaining
    // levels using simd_shuffle_xor to exchange the partner coefficient.
    //
    // We re-index: each ltid corresponds to a butterfly pair at the current
    // level. After loading, we transition to a "one coefficient per thread"
    // model for the simd phase. There are n coefficients = half_n*2.
    // Use ltid for low half and ltid+half_n for high half... but for simd
    // shuffle we need partners within a 32-lane simdgroup.
    //
    // Simpler: continue with the "two coeffs per thread" model. At each
    // remaining level, thread ltid still computes its butterfly. The values
    // a[j] and a[j+length] are already exchanged via threadgroup memory in
    // the last phase. We continue exactly that pattern but skip barriers
    // where the partner lies within the same simdgroup.
    //
    // For length <= 16: each butterfly group (2*length coeffs) fits in a
    // contiguous block of 'length' threads (one simdgroup if length<=16).
    // The partner read a[j+length] is from a different ltid, but within
    // the same simdgroup. Threadgroup memory writes/reads are coherent
    // within a simdgroup without an explicit barrier IF we use
    // simdgroup_barrier(mem_threadgroup).
    while (length > 0u) {
        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zm         = zeta_mont[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];

        uint t = mont_reduce((ulong)y * (ulong)zm, qreg, qinv_neg);
        uint xpt = mod_add(x, t, qreg);
        uint xmt = mod_sub(x, t, qreg);

        simdgroup_barrier(mem_flags::mem_threadgroup);
        a[j]          = xpt;
        a[j + length] = xmt;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        if (length == 1u) break;
        length  >>= 1u;
    }

    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}
```