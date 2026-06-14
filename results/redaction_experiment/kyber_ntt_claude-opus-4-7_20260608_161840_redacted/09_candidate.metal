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

// Montgomery reduction of T < q*2^32. Returns T*2^{-32} mod q in [0, q).
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

// Butterfly: x' = x + zeta*y, y' = x - zeta*y (mod q).
inline void bfly(thread uint &x, thread uint &y, uint zm, uint qreg, uint qinv_neg) {
    uint t = mont_reduce((ulong)y * (ulong)zm, qreg, qinv_neg);
    uint xpt = mod_add(x, t, qreg);
    uint xmt = mod_sub(x, t, qreg);
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

    // R = 2^32 mod q
    uint Rmod;
    {
        ulong r = ((ulong)0xFFFFFFFFul) % (ulong)qreg;
        r += 1ul;
        if (r >= (ulong)qreg) r -= (ulong)qreg;
        Rmod = (uint)r;
    }

    // Cooperatively compute Montgomery-form zetas in shared memory.
    {
        ulong Q = (ulong)qreg;
        uint i1 = ltid;
        if (i1 < zcount) {
            ulong v = (ulong)zetas[i1] * (ulong)Rmod;
            zeta_mont[i1] = (uint)(v % Q);
        }
        uint i2 = ltid + half_n;
        if (i2 < zcount) {
            ulong v = (ulong)zetas[i2] * (ulong)Rmod;
            zeta_mont[i2] = (uint)(v % Q);
        }
    }

    device uint *poly = coeffs + (size_t)tgid * nreg;

    // Load polynomial into registers (thread owns positions ltid and ltid+half_n).
    uint x = poly[ltid];
    uint y = poly[ltid + half_n];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length  = half_n;
    uint k_start = 1u;
    uint level   = 0u;

    // ----- Phase A: levels where length > 32 (partner lives in another simdgroup).
    // Use threadgroup memory + barriers. We process levels until length <= 32.
    // For n=256, half_n=128: length sequence 128, 64, 32, 16, ...
    // So phase A handles length=128 and length=64 (2 levels).
    for (; level < nlv && length > 32u; ++level) {
        // Reconstruct the index that this thread "owns" in linearized form.
        // Each thread owns butterfly ltid at this level:
        //   group_idx = ltid / length
        //   j_in_group = ltid % length
        //   j      = 2*group_idx*length + j_in_group
        //   j+len  = j + length
        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zm         = zeta_mont[k_start + group_idx];

        // Write current register values to their "natural" positions ltid and ltid+half_n.
        // But for the butterfly we need a[j] and a[j+length], which generally differ from
        // ltid / ltid+half_n. So we stage: write the two registers to slots ltid and ltid+half_n,
        // barrier, then read partner positions.
        a[ltid] = x;
        a[ltid + half_n] = y;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint xv = a[j];
        uint yv = a[j + length];

        bfly(xv, yv, zm, qreg, qinv_neg);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        a[j]          = xv;
        a[j + length] = yv;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Reload our owned slots back into registers for the next iteration.
        x = a[ltid];
        y = a[ltid + half_n];

        k_start <<= 1u;
        length  >>= 1u;
    }

    // ----- Phase B: levels where length <= 32 (partner lives in same simdgroup).
    // Here ltid in [0, 128); simd lane = ltid & 31; simd group id = ltid >> 5.
    // Each thread still owns butterfly ltid; we use simd_shuffle_xor on the LOW
    // register x to swap with the partner lane.
    //
    // For length in {32, 16, 8, 4, 2, 1}:
    //   group_idx = ltid / length
    //   j         = 2*group_idx*length + (ltid % length)
    //   j_low_lane  = j & 31
    //   j_high_lane = (j+length) & 31
    //   simd_id_low  = j >> 5
    //   simd_id_high = (j+length) >> 5
    // For length=32: j has bit5=0, j+length has bit5=1 -> SAME simd if (ltid>>6) chooses
    //   the simdgroup. Actually j range: group_idx = ltid/32; j = 64*group_idx + (ltid%32).
    //   Threads ltid in [0,32): group=0, j in [0,32), j+32 in [32,64) -> simd0 owns both? simd0=lanes 0..31. j in simd0, j+32 in simd1. DIFFERENT simdgroups.
    //
    // Hmm, length=32 still crosses simdgroup boundary. Let me reconsider.
    // For length <= 16, j and j+length both lie in the same 32-wide chunk iff the
    // chunk boundary doesn't fall between them. j range within group: [0,length).
    // j+length range: [length, 2*length). 2*length <= 32 means [0,32) -> same simd chunk.
    // So phase B applies when length <= 16.
    //
    // Rewrite: keep length > 16 in phase A.

    // (Code above used > 32; we adjust below.)
    // To avoid restructuring, we run additional shared-memory levels until length <= 16.
    for (; level < nlv && length > 16u; ++level) {
        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zm         = zeta_mont[k_start + group_idx];

        a[ltid] = x;
        a[ltid + half_n] = y;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint xv = a[j];
        uint yv = a[j + length];
        bfly(xv, yv, zm, qreg, qinv_neg);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        a[j]          = xv;
        a[j + length] = yv;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        x = a[ltid];
        y = a[ltid + half_n];

        k_start <<= 1u;
        length  >>= 1u;
    }

    // Final flush before pure-register phase (no further shared reads needed,
    // but we already have x,y in registers consistent with positions ltid and ltid+half_n).
    // Now phase B: length in {16, 8, 4, 2, 1}. Both x (at index ltid) and y (at ltid+half_n)
    // are owned by THIS thread. The butterfly at "ltid" uses positions j and j+length
    // which generally are NOT the same as (ltid, ltid+half_n). Doing per-thread butterflies
    // with simd shuffles requires rethinking ownership.
    //
    // Simpler approach for phase B: each thread does TWO independent butterflies — one within
    // the low half (x register, position ltid) and one within the high half (y register,
    // position ltid+half_n). But the NTT butterflies pair x with y at different positions.
    //
    // Easier: just keep using shared memory for remaining levels. Performance gain from
    // the register caching across levels is preserved up to length=16.
    for (; level < nlv; ++level) {
        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zm         = zeta_mont[k_start + group_idx];

        a[ltid] = x;
        a[ltid + half_n] = y;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint xv = a[j];
        uint yv = a[j + length];
        bfly(xv, yv, zm, qreg, qinv_neg);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        a[j]          = xv;
        a[j + length] = yv;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        x = a[ltid];
        y = a[ltid + half_n];

        k_start <<= 1u;
        length  >>= 1u;
    }

    poly[ltid]          = x;
    poly[ltid + half_n] = y;
}