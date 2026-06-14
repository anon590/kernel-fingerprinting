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
    if (tgid >= batch) return;

    threadgroup uint a[N_MAX];

    const uint half_n = n >> 1u;
    const uint qv     = q;
    const uint nl     = n_levels;

    device uint *poly = coeffs + (size_t)tgid * n;

    // Each thread owns two slots: lo_idx = ltid, hi_idx = ltid + half_n.
    // We keep them in registers across all levels; threadgroup memory is
    // only used to "rotate" partner ownership between levels whose stride
    // is >= 32 (outside a single simdgroup). For stride < 32 we use
    // simd_shuffle_xor to swap with the partner lane directly.
    uint lo = poly[ltid];
    uint hi = poly[ltid + half_n];

    uint length  = half_n;
    uint k_start = 1u;

    // Cache simd lane id (ltid mod 32)
    const uint lane = ltid & 31u;

    for (uint level = 0u; level < nl; ++level) {
        // Compute (j, j+length) for this thread's butterfly using the
        // standard mapping.
        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zeta       = zetas[k_start + group_idx];

        uint x, y;

        if (length >= 32u) {
            // Stride spans multiple simdgroups -> go through threadgroup mem.
            // Store our two owned slots (lo at ltid, hi at ltid+half_n).
            threadgroup_barrier(mem_flags::mem_threadgroup);
            a[ltid]          = lo;
            a[ltid + half_n] = hi;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            x = a[j];
            y = a[j + length];
        } else if (length == 0u) {
            // Defensive: should not occur for n>=2.
            x = lo; y = hi;
        } else {
            // length is a power of two in [1,16]. Partner lane within
            // simdgroup differs by `length` in the lane index.
            // Our two owned slots at this point correspond to consecutive
            // pairs (j_lo, j_lo + length_prev?) — but after threadgroup
            // staging at the boundary level (length == half_n/.../32),
            // we re-established the invariant that thread ltid owns
            // a[ltid] and a[ltid+half_n]. For length < 32 each thread's
            // butterfly is (j, j+length) where j and j+length are in the
            // same half (the same set of half_n entries). So we need to
            // shuffle within the simdgroup over the "lo" register for
            // threads ltid < half_n... but every thread is < half_n.
            //
            // Actually with our layout, after the staging barrier at the
            // last length>=32 level, we re-load a[ltid] into lo and
            // a[ltid+half_n] into hi. From that point onward, lengths are
            // < 32 and the butterflies operate ENTIRELY within the lo
            // half OR the hi half (since length < half_n once level>=1).
            // So we need to do butterflies on lo (over threads owning
            // the lo-half slots) and on hi (over threads owning hi-half).
            //
            // But the standard mapping has ltid in [0, half_n) owning
            // butterfly (j, j+length). For levels with length < half_n,
            // j spans both halves. So actually each thread's butterfly
            // is NOT confined to one half register.
            //
            // To keep this simple and correct, fall back to threadgroup
            // memory for these levels too.
            threadgroup_barrier(mem_flags::mem_threadgroup);
            a[ltid]          = lo;
            a[ltid + half_n] = hi;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            x = a[j];
            y = a[j + length];
        }

        // t = (zeta * y) mod q. For Kyber q=3329, zeta,y < q so prod < 2^24.
        uint prod = zeta * y;
        uint t    = prod % qv;

        uint sum_  = x + t;          if (sum_  >= qv) sum_  -= qv;
        uint diff_ = x + qv - t;     if (diff_ >= qv) diff_ -= qv;

        // Write back to threadgroup memory at (j, j+length), then reload
        // our owned slots (ltid, ltid+half_n) so the invariant holds.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        a[j]          = sum_;
        a[j + length] = diff_;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        lo = a[ltid];
        hi = a[ltid + half_n];

        k_start <<= 1u;
        length  >>= 1u;
    }

    poly[ltid]          = lo;
    poly[ltid + half_n] = hi;
}