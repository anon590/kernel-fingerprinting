Acknowledging the system reminder about the keybindings-help skill — it's not relevant to this Metal kernel optimization task, so I'll proceed with the kernel.

**Optimization:** The dominant cost at small batch sizes is launch/dispatch overhead and the per-butterfly modular multiply. I'll (1) replace the expensive 64-bit Barrett with **Montgomery reduction using 32-bit `mulhi`** (no 128-bit emulation needed — just two `mulhi(uint,uint)` calls plus a `mul`), since q fits in 31 bits and we can precompute `q_inv_neg = -q^{-1} mod 2^32` once per threadgroup; (2) keep zeta in **Montgomery form** by premultiplying with `R = 2^32 mod q` on the fly — wait, even simpler: use the identity `mont_reduce(a * zeta_mont)` where we convert zeta to Montgomery form once at load. (3) Cache zetas for each level into registers/threadgroup memory to avoid redundant device loads. (4) Use simdgroup barriers for the last 5 levels (len ≤ 16) where butterflies stay within a 32-lane span.

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_MAX = 256u;

inline uint mod_add(uint a, uint b, uint q) {
    uint t = a + b;
    return (t >= q) ? (t - q) : t;
}

inline uint mod_sub(uint a, uint b, uint q) {
    return (a >= b) ? (a - b) : (a + q - b);
}

// Montgomery reduction: given T < q * 2^32, returns T * 2^{-32} mod q in [0, q).
// Requires qinv_neg = (-q^{-1}) mod 2^32, i.e. q * qinv_neg = -1 mod 2^32.
inline uint mont_reduce(ulong T, uint q, uint qinv_neg) {
    uint T_lo = (uint)T;
    uint m = T_lo * qinv_neg;             // m = (T mod 2^32) * (-q^{-1}) mod 2^32
    // t = (T + m*q) / 2^32; since T + m*q is divisible by 2^32.
    // High 32 bits of (T + m*q): high(T) + high(m*q) + carry.
    uint T_hi = (uint)(T >> 32);
    uint mq_hi = mulhi(m, q);
    uint mq_lo = m * q;
    // carry from T_lo + mq_lo
    uint sum_lo = T_lo + mq_lo;
    uint carry = (sum_lo < T_lo) ? 1u : 0u;
    uint t = T_hi + mq_hi + carry;
    // t in [0, 2q); reduce.
    if (t >= q) t -= q;
    return t;
}

// Modular multiply: a * b mod q, using Montgomery on b that is ALREADY in Mont form.
// result = a * b * 2^{-32} mod q. If b_mont = b * 2^32 mod q, this yields a*b mod q.
inline uint mod_mul_mont(uint a, uint b_mont, uint q, uint qinv_neg) {
    return mont_reduce((ulong)a * (ulong)b_mont, q, qinv_neg);
}

// Compute (-q^{-1}) mod 2^32 via Newton iteration (5 iters give 32 valid bits for odd q).
inline uint compute_qinv_neg(uint q) {
    // Start: x = q; x*q == 1 mod 2^3 since q odd? Actually q*q == 1 mod 8 for odd q.
    uint x = q;                  // q*q == 1 mod 8
    x = x * (2u - q * x);        // mod 16
    x = x * (2u - q * x);        // mod 256
    x = x * (2u - q * x);        // mod 2^16
    x = x * (2u - q * x);        // mod 2^32
    // Now x = q^{-1} mod 2^32; we want -q^{-1}.
    return 0u - x;
}

// Compute R = 2^32 mod q.
inline uint compute_R(uint q) {
    // 2^32 mod q = ((2^32 - 1) mod q) + 1, then reduce.
    ulong r = ((ulong)0xFFFFFFFFul) % (ulong)q;
    r += 1ul;
    if (r >= (ulong)q) r -= (ulong)q;
    return (uint)r;
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
    threadgroup uint zeta_mont_tg[N_MAX];   // precomputed Montgomery-form zetas
    threadgroup uint q_share;
    threadgroup uint qinv_share;
    threadgroup uint R_share;

    uint qreg = q;
    uint nreg = n;
    uint nlv  = n_levels;
    uint half_n = nreg >> 1u;
    uint zcount = 1u << nlv;   // number of zetas

    // Thread 0 computes Montgomery constants once per threadgroup.
    if (ltid == 0u) {
        q_share = qreg;
        qinv_share = compute_qinv_neg(qreg);
        R_share = compute_R(qreg);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint qinv_neg = qinv_share;
    uint Rmod = R_share;

    // Cooperatively convert zetas to Montgomery form into threadgroup memory.
    // zcount <= 256, threads = n/2 = up to 128. Two slots each suffices.
    {
        uint i = ltid;
        if (i < zcount) {
            uint z = zetas[i];
            zeta_mont_tg[i] = mod_mul_mont(z, Rmod, qreg, qinv_neg) == 0u && z != 0u
                ? mod_mul_mont(z, Rmod, qreg, qinv_neg)
                : mod_mul_mont(z, Rmod, qreg, qinv_neg);
            // Simpler: zeta_mont = z * R mod q = mont_reduce(z * R * 2^32)? No.
            // mod_mul_mont(z, R, q, qinv_neg) = z * R * 2^{-32} mod q = z mod q. Wrong!
        }
        uint i2 = ltid + half_n;
        if (i2 < zcount) {
            uint z = zetas[i2];
            (void)z;
        }
    }
    // The above is wrong; recompute properly below.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Correct conversion: zeta_mont = (z * R) mod q. Use 64-bit multiply.
    {
        ulong Q = (ulong)qreg;
        uint i = ltid;
        if (i < zcount) {
            ulong v = (ulong)zetas[i] * (ulong)Rmod;
            zeta_mont_tg[i] = (uint)(v % Q);
        }
        uint i2 = ltid + half_n;
        if (i2 < zcount) {
            ulong v = (ulong)zetas[i2] * (ulong)Rmod;
            zeta_mont_tg[i2] = (uint)(v % Q);
        }
    }

    device uint *poly = coeffs + (size_t)tgid * nreg;

    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length  = half_n;
    uint k_start = 1u;

    for (uint level = 0u; level < nlv; ++level) {
        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zeta_m     = zeta_mont_tg[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];
        // t = (zeta * y) mod q via Montgomery: mont_reduce(y * zeta_mont).
        uint t = mod_mul_mont(y, zeta_m, qreg, qinv_neg);

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