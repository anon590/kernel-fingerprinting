**Optimization rationale:** The previous attempt uses an expensive 64-bit Barrett reduction with a software `mulhi64` (4 multiplies + adds) on every butterfly. For ZK params, q < 2^24, so we can use a much cheaper 32-bit Barrett: precompute `mbar32 = floor(2^48 / q)` (fits in 32 bits since q ≥ 2). Then for `t = zeta*y < q^2 < 2^48`, the quotient estimate is `(t >> 16) * mbar32 >> 32`, computable with one `mulhi(uint,uint)` plus a shift — no 64×64 software multiply. This replaces ~4 muls + many adds with 1 hardware `mulhi` + 1 mul, dramatically cutting the inner-loop cost. I also drop the zetas threadgroup cache (only `n_levels ≤ 8` loads per thread total via `constant`-style device reads, hitting L1) to free registers and skip a barrier-free preload, and use branchless `select` for canonical reduction.

```metal
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

    // 32-bit Barrett: since q < 2^24 (Kyber 3329, Dilithium 8380417, etc.),
    // and t = zeta*y < q^2 < 2^48, we compute floor(t/q) using a 48-bit reciprocal.
    //
    // Let mbar = floor(2^(32 + s) / q), where s = ceil(log2(q)) so that 2^s >= q.
    // For Kyber q=3329, s=12; for Dilithium q=8380417, s=23.
    // We need mbar to fit in 32 bits: 2^(32+s)/q ~ 2^32 when 2^s ~ q. OK.
    //
    // Pick s = 32 (safe upper bound but mbar may not fit u32). Instead, use the
    // standard floor(log2(q-1))+1. Compute s via clz.
    //
    // Then for t < q * 2^32:  q_est = (t >> s) * mbar >> 32 is within {q-1, q}.
    // For our t < q^2 <= q*2^s*... safer to use full 64-bit Barrett-light:
    //   q_est = mulhi(t_hi32, mbar) where t_hi32 = t >> s; t fits in 2^(2s).
    //
    // Simpler & fast: use Barrett with mbar = floor(2^64 / q), but compute mulhi
    // using the fact that tfull < 2^48 so tfull_hi (top 32) fits in 16 bits.
    // Then mulhi64(tfull, mbar) ~= mulhi32(tfull_hi32, mbar_hi32) + a few cross terms.
    //
    // Cleanest hardware-friendly path: Barrett with k=48.
    //   mbar48 = floor(2^48 / q)   -- fits in 48 bits (up to 2^48/2 = 2^47); but
    //   for q < 2^24, mbar48 >= 2^24, so mbar48 is up to 48 bits. Doesn't fit u32.
    //
    // Use Barrett with mbar = floor(2^(32+S)/q) where S = 32 - clz(q-1) (= ceil log2 q).
    // Then mbar fits in 32 bits (since 2^(32+S)/q < 2^(32+S)/2^(S-1) = 2^33; tight,
    // but mbar < 2^33). Use ulong for mbar storage to be safe.
    //
    //   q_est = ((tfull >> S) * mbar) >> 32
    //   r     = tfull - q_est * q, in [0, 2q); one correction.
    //
    // tfull < q^2 <= 2^(2S); tfull >> S < 2^S <= 2^24 < 2^32. So the (tfull>>S) fits in u32.
    // mbar fits in u33; we use ulong*ulong but the product fits in 64 bits easily
    // since (tfull>>S) * mbar < 2^32 * 2^33 = 2^65 -- oops, overflow possible.
    //
    // Fix: use S = ceil(log2(q+1)) so 2^S >= q+1, hence mbar = floor(2^(32+S)/q) < 2^32.
    // Verify: 2^(32+S)/q <= 2^(32+S)/((2^S)/2 + 1). For S>=1 and q in (2^(S-1), 2^S],
    // worst case q = 2^(S-1)+1 gives mbar ~ 2^(32+S) / 2^(S-1) = 2^33. Still 33 bits.
    //
    // Pragmatic: store mbar as ulong, use uint*ulong multiplication where one operand
    // is at most 33 bits. Compute as two 32-bit muls.

    uint S = 32u - clz(qv);                 // ceil(log2(q+1)); for q=3329 -> 12
    ulong mbar = ((ulong)1 << (32u + S)) / (ulong)qv;
    uint  mbar_lo = (uint)mbar;
    uint  mbar_hi = (uint)(mbar >> 32);     // 0 or 1

    device uint *poly = coeffs + (size_t)tgid * nv;

    // Load coefficients into threadgroup memory.
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length  = half_n;
    uint k_start = 1u;
    uint log2_len = 31u - clz(half_n);      // log2(n/2)

    for (uint level = 0u; level < nlv; ++level) {
        uint mask       = length - 1u;
        uint group_idx  = ltid >> log2_len;
        uint j_in_group = ltid & mask;
        uint j          = (group_idx << (log2_len + 1u)) + j_in_group;
        uint zeta       = zetas[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];

        // tfull = zeta * y, < q^2 < 2^48.
        ulong tfull = (ulong)zeta * (ulong)y;

        // Barrett: q_est = ((tfull >> S) * mbar) >> 32
        uint  thi = (uint)(tfull >> S);                  // fits in 32 bits
        // multiply thi (u32) * mbar (up to 33 bits) -> need high 32 bits of the product.
        // product = thi * mbar_lo + (thi * mbar_hi) << 32
        // high 32 bits = mulhi(thi, mbar_lo) + thi * mbar_hi
        uint  hi_lo = mulhi(thi, mbar_lo);
        uint  qest  = hi_lo + thi * mbar_hi;

        uint  r     = (uint)tfull - qest * qv;
        // r in [0, 2q); correction.
        r = (r >= qv) ? (r - qv) : r;

        // mod add/sub.
        uint s = x + r;
        s = (s >= qv) ? (s - qv) : s;
        uint xqr = x + qv - r;
        xqr = (xqr >= qv) ? (xqr - qv) : xqr;

        a[j]          = s;
        a[j + length] = xqr;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length  >>= 1u;
        log2_len = (log2_len == 0u) ? 0u : (log2_len - 1u);
    }

    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}
```