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

    uint qv     = q;
    uint nv     = n;
    uint nlv    = n_levels;
    uint half_n = nv >> 1u;

    // Barrett constants.
    // s = bit_width(q): smallest integer with 2^s >= q+1 (i.e., q < 2^s).
    // For q in [1, 2^31], s in [1, 31].
    // mbar = floor(2^(32 + s) / q).
    //   2^(32+s)/q < 2^(32+s)/2^(s-1) = 2^33, so mbar can be up to 33 bits.
    //   But if q > 2^(s-1), tighter: mbar < 2^(32+s)/q.
    // To keep mbar in 32 bits we'd need q >= 2^s, contradiction. So mbar can be 33 bits.
    // We store mbar as ulong but split into (mbar_lo, mbar_hi in {0,1}).
    //
    // For tfull < q^2 < 2^(2s) <= 2^(32+s) (since s <= 32),
    //   thi = tfull >> s   < 2^s <= 2^32, fits in u32 (when s<32; for s=32 we'd need care).
    //   q_est = (thi * mbar) >> 32 = mulhi(thi, mbar_lo) + thi * mbar_hi
    //         in {floor(tfull/q) - e, floor(tfull/q)} for small e (<=1).
    //   r = (u32)tfull - q_est * q, in [0, 2q); one correction.
    uint s = 32u - clz(qv);                 // ceil(log2(q+1)); 0 only if q==0
    if (qv == 0u) s = 0u;
    // (qv is a real prime, never 0; guard kept for safety)
    ulong mbar = ((ulong)1 << (32u + s)) / (ulong)qv;
    uint  mbar_lo = (uint)mbar;
    uint  mbar_hi = (uint)(mbar >> 32);     // 0 or 1

    device uint *poly = coeffs + (size_t)tgid * nv;

    // Load coefficients into threadgroup memory.
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length  = half_n;
    uint k_start = 1u;
    uint log2_len = (half_n == 0u) ? 0u : (31u - clz(half_n));

    for (uint level = 0u; level < nlv; ++level) {
        uint mask       = length - 1u;
        uint group_idx  = ltid >> log2_len;
        uint j_in_group = ltid & mask;
        uint j          = (group_idx << (log2_len + 1u)) + j_in_group;
        uint zeta       = zetas[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];

        // tfull = zeta * y, < q^2 < 2^(2s).
        ulong tfull = (ulong)zeta * (ulong)y;

        // Barrett reduction.
        uint  thi   = (uint)(tfull >> s);
        // q_est = high-32 of (thi * mbar). Since mbar = mbar_hi*2^32 + mbar_lo:
        //   thi * mbar = thi*mbar_lo + thi*mbar_hi * 2^32
        //   high 32 of thi*mbar_lo = mulhi(thi, mbar_lo)
        //   thi*mbar_hi contributes thi*mbar_hi (mbar_hi in {0,1})
        uint  qest  = mulhi(thi, mbar_lo) + thi * mbar_hi;

        uint  r     = (uint)tfull - qest * qv;
        // r in [0, 2q); one correction (Barrett may under-estimate by 1).
        if (r >= qv) r -= qv;
        // Extra safety: in rare edge cases r could be 2q (if estimate under by 2).
        // Bound: |q_est - floor(t/q)| <= 1 with this Barrett, so r in [0, 2q). One sub suffices.

        // Butterfly.
        uint sum = x + r;
        sum = (sum >= qv) ? (sum - qv) : sum;
        uint dif = x + qv - r;
        dif = (dif >= qv) ? (dif - qv) : dif;

        a[j]          = sum;
        a[j + length] = dif;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length  >>= 1u;
        log2_len = (log2_len == 0u) ? 0u : (log2_len - 1u);
    }

    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}