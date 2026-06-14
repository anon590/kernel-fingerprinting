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

    const uint qv     = q;
    const uint nv     = n;
    const uint nlv    = n_levels;
    const uint half_n = nv >> 1u;

    // Barrett with 32-bit reciprocal.
    // L = bit_width(q): smallest s with 2^s >= q  ->  L = 32 - clz(q-1) (for q>=2)
    // Equivalently L = 32 - clz(q) when q is not a power of two;
    // we use L = 32 - clz(q) which gives L such that q <= 2^L, and 2^(L-1) <= q (for q>=2).
    // For zeta, y in [0, q), t = zeta*y < q^2 <= 2^(2L), but more tightly t < q^2.
    // thi = t >> (L-1):  thi < q^2 / 2^(L-1) <= q * 2^(L+1) / 2^(L-1) ... we need a cleaner bound.
    //
    // Use 64-bit Barrett: shift k = 2L; mbar = floor(2^k / q).
    // q_est = (t * mbar) >> k. We have t < q^2 <= 2^(2L), mbar < 2^(2L)/q + 1 <= 2^(L+1).
    // So t*mbar < 2^(2L) * 2^(L+1) = 2^(3L+1) -- may overflow 64 bits if L>21.
    //
    // Safer scheme that handles q up to 2^31:
    //   Compute thi = t >> 32 (high 32 bits of t).
    //   mbar = floor(2^64 / q); mbar fits in 64 bits.
    //   q_est = mulhi32(thi, mbar_hi) approx... too coarse.
    //
    // We fall back to: 32-bit Barrett where mbar = floor(2^(2L) / q), shift = 2L,
    // but split product. Since L <= 32, mbar <= 2^(L+1) fits in 33 bits.
    //
    // Cleanest correct fast path: split t into (t_hi, t_lo) 32-bit halves.
    // floor(t / q) = floor((t_hi * 2^32 + t_lo) / q).
    // Precompute: M = floor(2^64 / q) as ulong (one-time, cheap).
    // q_est_hi = mulhi(t_hi, M_hi) + ... -- this is the mulhi64 path (4 muls).
    //
    // FASTER: For our regime (q < 2^L, L = bit_width(q)),
    //   t = zeta * y < q^2.  Let t32 = t >> (L-1).  Then t32 < q^2 / 2^(L-1).
    //   Since q < 2^L: t32 < 2^(2L) / 2^(L-1) = 2^(L+1) <= 2^32 (when L<=31).
    //   Set mbar = floor(2^(L+1) * 2^31 / q) = floor(2^(L+32) / q), this fits in <= 33 bits.
    //   But simpler: use mbar32 = floor(2^32 * 2^(L-1) / q) so q_est = mulhi(t32, mbar32).
    //   Need mbar32 in 32 bits: mbar32 = floor(2^(L+31) / q). Since q >= 2^(L-1),
    //   mbar32 <= 2^(L+31)/2^(L-1) = 2^32. So mbar32 fits in 32 bits (saturates at 2^32-1 ok).
    //
    // q_est = mulhi(t32, mbar32) gives approximately floor(t / q) with error <= 2.
    // r = (uint)t - q_est * q, then correct down by q up to twice.

    uint L;
    if (qv <= 1u) { L = 1u; }
    else { L = 32u - clz(qv - 1u); }   // smallest L s.t. q <= 2^L; q >= 2 => L>=1
    // Ensure 2^(L-1) <= q (true when L = ceil(log2 q) and q >= 2)
    uint sh = (L == 0u) ? 0u : (L - 1u);

    // mbar32 = floor(2^(32+sh) / q), saturated to 2^32-1 if overflow.
    ulong num  = (sh >= 32u) ? (~(ulong)0) : ((ulong)1 << (32u + sh));
    ulong mbig = num / (ulong)qv;
    uint  mbar32 = (mbig > 0xFFFFFFFFULL) ? 0xFFFFFFFFu : (uint)mbig;

    device uint *poly = coeffs + (size_t)tgid * nv;

    // Load coefficients & zetas into threadgroup memory.
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];
    zcache[ltid]          = zetas[ltid];
    zcache[ltid + half_n] = zetas[ltid + half_n];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length   = half_n;
    uint log2_len = (half_n == 0u) ? 0u : (31u - clz(half_n));
    uint k_start  = 1u;

    for (uint level = 0u; level < nlv; ++level) {
        uint mask       = length - 1u;
        uint group_idx  = ltid >> log2_len;
        uint j_in_group = ltid & mask;
        uint j          = (group_idx << (log2_len + 1u)) + j_in_group;
        uint zeta       = zcache[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];

        // t = zeta * y < q^2; full 64-bit product.
        ulong tfull = (ulong)zeta * (ulong)y;

        // Barrett: t32 = tfull >> sh, fits in 32 bits.
        uint t32  = (uint)(tfull >> sh);
        uint qest = mulhi(t32, mbar32);
        uint r    = (uint)tfull - qest * qv;
        // r in [0, ~3q); correct down up to 2 times.
        if (r >= qv) r -= qv;
        if (r >= qv) r -= qv;

        // Butterfly (canonical).
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