Acknowledging the skill reminder: no keybinding customization is requested, so I will not invoke the keybindings-help skill.

**Optimization rationale:** The incumbent uses a costly 64-bit Barrett reduction (full 128-bit multiplication via 4 muls + adds) per butterfly. I'll switch to **Montgomery multiplication using 32-bit `mulhi`**, which is much cheaper on Apple GPUs: one `mulhi(uint,uint)` plus a few 32-bit ops per reduction. To avoid expensive per-butterfly Montgomery conversion of zetas, I pre-convert all zetas to Montgomery form once into threadgroup memory (cooperatively), then every butterfly is just `mont_reduce((ulong)y * zeta_mont)`. The final write converts back to standard form by Montgomery-reducing with multiplier 1 — but a simpler approach: keep `a[]` in standard form and use `mont_reduce(y * (zeta * R mod q))` which yields `y * zeta mod q` directly. This keeps coefficients canonical throughout. Reduced barriers at the end levels via simdgroup barriers (already in incumbent).

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

// Compute -q^{-1} mod 2^32 via Newton iteration on odd q.
inline uint compute_qinv_neg(uint q) {
    // Start from a 3-bit inverse: for odd q, q*q ≡ 1 (mod 8), so x0 = q.
    uint x = q;
    x = x * (2u - q * x);   // 6 bits
    x = x * (2u - q * x);   // 12 bits
    x = x * (2u - q * x);   // 24 bits
    x = x * (2u - q * x);   // 48 bits -> 32 valid
    x = x * (2u - q * x);   // safety
    // x = q^{-1} mod 2^32; we want -q^{-1} mod 2^32.
    return 0u - x;
}

// 2^32 mod q.
inline uint compute_R(uint q) {
    ulong r = ((ulong)0xFFFFFFFFul) % (ulong)q;
    r += 1ul;
    if (r >= (ulong)q) r -= (ulong)q;
    return (uint)r;
}

// Montgomery reduction of T (T < q*2^32 ensures result < q after one conditional sub).
// Returns T * 2^{-32} mod q in [0, q), assuming q < 2^31 (Kyber q = 3329, well within).
inline uint mont_reduce(ulong T, uint q, uint qinv_neg) {
    uint T_lo = (uint)T;
    uint T_hi = (uint)(T >> 32);
    uint m    = T_lo * qinv_neg;          // mod 2^32
    uint mq_hi = mulhi(m, q);
    uint mq_lo = m * q;
    uint sum_lo = T_lo + mq_lo;           // low 32 bits sum to 0
    uint carry  = (sum_lo < T_lo) ? 1u : 0u;
    uint t = T_hi + mq_hi + carry;        // t in [0, 2q)
    if (t >= q) t -= q;
    return t;
}

// a * b_mont * 2^{-32} mod q. If b_mont = (b * 2^32) mod q (Montgomery form of b),
// the result equals (a * b) mod q in standard form.
inline uint mod_mul_mont(uint a, uint b_mont, uint q, uint qinv_neg) {
    return mont_reduce((ulong)a * (ulong)b_mont, q, qinv_neg);
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
    threadgroup uint qinv_share;
    threadgroup uint R_share;

    uint qreg   = q;
    uint nreg   = n;
    uint nlv    = n_levels;
    uint half_n = nreg >> 1u;
    uint zcount = 1u << nlv;

    // One thread computes Montgomery constants for the threadgroup.
    if (ltid == 0u) {
        qinv_share = compute_qinv_neg(qreg);
        R_share    = compute_R(qreg);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint qinv_neg = qinv_share;
    uint Rmod     = R_share;

    // Cooperatively precompute zeta_mont[k] = (zetas[k] * 2^32) mod q.
    // zcount <= 256, threads = half_n (up to 128); cover with 2 slots each.
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

    // Load polynomial into threadgroup memory.
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length  = half_n;
    uint k_start = 1u;

    for (uint level = 0u; level < nlv; ++level) {
        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zm         = zeta_mont[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];

        // t = (zeta * y) mod q via Montgomery.
        uint t = mod_mul_mont(y, zm, qreg, qinv_neg);

        uint xpt = mod_add(x, t, qreg);
        uint xmt = mod_sub(x, t, qreg);

        if (length > 16u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            a[j]          = xpt;
            a[j + length] = xmt;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            a[j]          = xpt;
            a[j + length] = xmt;
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        k_start <<= 1u;
        length  >>= 1u;
    }

    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}
```