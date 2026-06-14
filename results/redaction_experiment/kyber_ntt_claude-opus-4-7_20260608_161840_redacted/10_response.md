**Optimization plan:** The incumbent uses threadgroup memory + 2 barriers per level (8 levels = 16 barriers). I'll switch to a **register-resident NTT** where each thread keeps both its `(x, y)` coefficients in registers throughout, and exchanges them with the partner thread via threadgroup memory **only once per level** (single barrier per level instead of two). 

Key trick: at every level, thread `ltid` owns butterfly index `ltid`, which reads `a[j]` and `a[j+len]`. I keep `(x, y)` always representing `(a[ltid], a[ltid + half_n])`. Each level I write the y-half to scratch, barrier once, read partner's value, do butterfly, then write back into x/y registers — this halves barriers and avoids writing x to memory entirely most levels by computing the correct partner mapping. Actually simpler: write both halves once, single barrier, read both partners, compute, single barrier, write — same as incumbent. Better approach: **fuse multiple low-len levels** in registers at the end (levels with len ≤ what each thread can hold). Since each thread owns 2 values, when `len == 1` and partner is the same thread, we can do it purely in registers with no barrier.

Actually the cleanest win: skip the second barrier per level by writing results to scratch buffer A in even levels, B in odd levels (double-buffer), eliminating write-after-read hazards.

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

    // Double-buffered scratch: ping-pong between bufA and bufB to avoid the
    // second per-level barrier.
    threadgroup uint bufA[N_MAX];
    threadgroup uint bufB[N_MAX];
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

    // Each thread owns two coefficient positions: ltid (low) and ltid+half_n (high).
    // Load directly into registers.
    uint x = poly[ltid];
    uint y = poly[ltid + half_n];

    // Seed bufA with full polynomial.
    bufA[ltid]          = x;
    bufA[ltid + half_n] = y;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length  = half_n;
    uint k_start = 1u;

    // Ping-pong: read from "src", write to "dst". After each level swap.
    // We unroll the read/write pattern: src/dst alternate between bufA and bufB.
    bool a_is_src = true;

    for (uint level = 0u; level < nlv; ++level) {
        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zm         = zeta_mont[k_start + group_idx];

        threadgroup uint *src = a_is_src ? bufA : bufB;
        threadgroup uint *dst = a_is_src ? bufB : bufA;

        uint xv = src[j];
        uint yv = src[j + length];

        uint t = mont_reduce((ulong)yv * (ulong)zm, qreg, qinv_neg);
        uint xpt = mod_add(xv, t, qreg);
        uint xmt = mod_sub(xv, t, qreg);

        dst[j]          = xpt;
        dst[j + length] = xmt;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length  >>= 1u;
        a_is_src = !a_is_src;
    }

    // Final result lives in whichever buffer is "src" after the last swap.
    threadgroup uint *final_buf = a_is_src ? bufA : bufB;
    poly[ltid]          = final_buf[ltid];
    poly[ltid + half_n] = final_buf[ltid + half_n];
}
```