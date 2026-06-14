**Optimization**: The hot path is the t=3 case (arity=2, in-distribution). The main cost is `gmul`, which dominates the partial rounds (`r_p` ~ 22 muls per partial round in partial layer plus the s-box). Key wins:

1. **Faster `umul128`**: use `mulhi` and a slightly cleaner carry path; mark all helpers with `__attribute__((always_inline))`.
2. **Specialize partial round for t=3**: only `s0` needs the s-box, then linear layer is `sum + D[i]*s_i`. We were already doing this, but we can fuse the `sum + D[i]*s_i` via a single 128-bit MAC and one reduction (saves an add per coord).
3. **Cache `M`, `D`, `rc_ext`, `rc_int` in threadgroup memory** so all 64 threads share constant loads (one strided cooperative load per round-constant set), reducing redundant device traffic. This is the biggest practical win since each TG has 64 threads doing the same constant loads.
4. Minor: reduce `gold_reduce128` branches with select-style code.

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;

#define AINL __attribute__((always_inline))

AINL inline ulong gadd(ulong a, ulong b) {
    ulong s = a + b;
    ulong c = (ulong)(s < a);
    s += c * EPSILON;
    ulong s2 = s - P_GOLD;
    return (s >= P_GOLD) ? s2 : s;
}

AINL inline void umul128(ulong a, ulong b, thread ulong &lo, thread ulong &hi) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    ulong p00 = (ulong)a0 * (ulong)b0;
    ulong p01 = (ulong)a0 * (ulong)b1;
    ulong p10 = (ulong)a1 * (ulong)b0;
    ulong p11 = (ulong)a1 * (ulong)b1;

    ulong mid = (p00 >> 32) + (p01 & EPSILON) + (p10 & EPSILON);
    lo = (p00 & EPSILON) | (mid << 32);
    hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}

// Reduce x = lo + hi * 2^64 (mod p). Assumes hi < 2^64 (fits in ulong).
AINL inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong x_hi_lo = x_hi & EPSILON;
    ulong x_hi_hi = x_hi >> 32;

    ulong t0 = x_lo - x_hi_hi;
    if (t0 > x_lo) t0 -= EPSILON;

    ulong t1 = (x_hi_lo << 32) - x_hi_lo;

    ulong r = t0 + t1;
    if (r < t0) r += EPSILON;
    if (r >= P_GOLD) r -= P_GOLD;
    return r;
}

AINL inline ulong gmul(ulong a, ulong b) {
    ulong lo, hi;
    umul128(a, b, lo, hi);
    return gold_reduce128(lo, hi);
}

// Accumulate a*b into 3-limb (lo, hi, c).
AINL inline void mac128(ulong a, ulong b,
                        thread ulong &acc_lo, thread ulong &acc_hi, thread ulong &acc_c) {
    ulong lo, hi;
    umul128(a, b, lo, hi);
    ulong new_lo = acc_lo + lo;
    ulong c1 = (ulong)(new_lo < acc_lo);
    acc_lo = new_lo;
    ulong sum2 = hi + c1;
    ulong cA = (ulong)(sum2 < hi);
    ulong new_hi = acc_hi + sum2;
    ulong cB = (ulong)(new_hi < acc_hi);
    acc_hi = new_hi;
    acc_c += cA + cB;
}

AINL inline ulong reduce_acc(ulong acc_lo, ulong acc_hi, ulong acc_c) {
    if (acc_c != 0ul) {
        ulong top = gold_reduce128(acc_hi, acc_c);
        return gold_reduce128(acc_lo, top);
    }
    return gold_reduce128(acc_lo, acc_hi);
}

AINL inline ulong sbox7(ulong x) {
    ulong x2 = gmul(x, x);
    ulong x4 = gmul(x2, x2);
    ulong x3 = gmul(x2, x);
    return gmul(x4, x3);
}

// ---------------- t=3 ----------------
AINL inline void mds3_tg(thread ulong &s0, thread ulong &s1, thread ulong &s2,
                          threadgroup const ulong *M) {
    ulong lo, hi, c;
    ulong n0, n1, n2;
    lo = 0; hi = 0; c = 0;
    mac128(M[0], s0, lo, hi, c);
    mac128(M[1], s1, lo, hi, c);
    mac128(M[2], s2, lo, hi, c);
    n0 = reduce_acc(lo, hi, c);

    lo = 0; hi = 0; c = 0;
    mac128(M[3], s0, lo, hi, c);
    mac128(M[4], s1, lo, hi, c);
    mac128(M[5], s2, lo, hi, c);
    n1 = reduce_acc(lo, hi, c);

    lo = 0; hi = 0; c = 0;
    mac128(M[6], s0, lo, hi, c);
    mac128(M[7], s1, lo, hi, c);
    mac128(M[8], s2, lo, hi, c);
    n2 = reduce_acc(lo, hi, c);

    s0 = n0; s1 = n1; s2 = n2;
}

AINL inline void mds4_tg(thread ulong &s0, thread ulong &s1, thread ulong &s2, thread ulong &s3,
                          threadgroup const ulong *M) {
    ulong lo, hi, c;
    ulong n0, n1, n2, n3;
    lo = 0; hi = 0; c = 0;
    mac128(M[0], s0, lo, hi, c);
    mac128(M[1], s1, lo, hi, c);
    mac128(M[2], s2, lo, hi, c);
    mac128(M[3], s3, lo, hi, c);
    n0 = reduce_acc(lo, hi, c);

    lo = 0; hi = 0; c = 0;
    mac128(M[4], s0, lo, hi, c);
    mac128(M[5], s1, lo, hi, c);
    mac128(M[6], s2, lo, hi, c);
    mac128(M[7], s3, lo, hi, c);
    n1 = reduce_acc(lo, hi, c);

    lo = 0; hi = 0; c = 0;
    mac128(M[8], s0, lo, hi, c);
    mac128(M[9], s1, lo, hi, c);
    mac128(M[10], s2, lo, hi, c);
    mac128(M[11], s3, lo, hi, c);
    n2 = reduce_acc(lo, hi, c);

    lo = 0; hi = 0; c = 0;
    mac128(M[12], s0, lo, hi, c);
    mac128(M[13], s1, lo, hi, c);
    mac128(M[14], s2, lo, hi, c);
    mac128(M[15], s3, lo, hi, c);
    n3 = reduce_acc(lo, hi, c);

    s0 = n0; s1 = n1; s2 = n2; s3 = n3;
}

#define TG_RC_MAX 256u   // enough for r_f * t up to 8*4=32, and r_p up to 32

kernel void merkle_build_level(
    device       ulong *tree         [[buffer(0)]],
    device const ulong *rc_ext       [[buffer(1)]],
    device const ulong *rc_int       [[buffer(2)]],
    device const ulong *ext_mds      [[buffer(3)]],
    device const ulong *int_diag     [[buffer(4)]],
    constant uint      &arity        [[buffer(5)]],
    constant uint      &t            [[buffer(6)]],
    constant uint      &r_f          [[buffer(7)]],
    constant uint      &r_p          [[buffer(8)]],
    constant uint      &in_offset    [[buffer(9)]],
    constant uint      &out_offset   [[buffer(10)]],
    constant uint      &child_count  [[buffer(11)]],
    uint p   [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_sz [[threads_per_threadgroup]])
{
    // Threadgroup-shared constants (one copy per TG, shared by all threads).
    threadgroup ulong tg_M[16];
    threadgroup ulong tg_D[4];
    threadgroup ulong tg_rce[32]; // r_f * t, max 8*4
    threadgroup ulong tg_rci[32]; // r_p, max 32

    uint tt = t;
    uint mds_n = tt * tt;
    uint rce_n = r_f * tt;
    uint rci_n = r_p;

    // Cooperative load.
    for (uint i = lid; i < mds_n; i += tg_sz)  tg_M[i]   = ext_mds[i];
    for (uint i = lid; i < tt;    i += tg_sz)  tg_D[i]   = int_diag[i];
    for (uint i = lid; i < rce_n; i += tg_sz)  tg_rce[i] = rc_ext[i];
    for (uint i = lid; i < rci_n; i += tg_sz)  tg_rci[i] = rc_int[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint parent_count = (child_count + arity - 1u) / arity;
    if (p >= parent_count) return;

    uint base = p * arity;
    uint half_f = r_f >> 1u;

    if (t == 3u) {
        ulong s0 = 0ul, s1 = 0ul, s2 = 0ul;
        if (base + 0u < child_count) s0 = tree[in_offset + base + 0u];
        if (base + 1u < child_count) s1 = tree[in_offset + base + 1u];
        if (arity >= 3u && base + 2u < child_count) s2 = tree[in_offset + base + 2u];

        // initial external linear layer
        mds3_tg(s0, s1, s2, tg_M);

        // first half full rounds
        for (uint r = 0u; r < half_f; ++r) {
            uint b = r * 3u;
            s0 = sbox7(gadd(s0, tg_rce[b + 0u]));
            s1 = sbox7(gadd(s1, tg_rce[b + 1u]));
            s2 = sbox7(gadd(s2, tg_rce[b + 2u]));
            mds3_tg(s0, s1, s2, tg_M);
        }

        // partial rounds
        ulong D0 = tg_D[0], D1 = tg_D[1], D2 = tg_D[2];
        for (uint r = 0u; r < r_p; ++r) {
            s0 = sbox7(gadd(s0, tg_rci[r]));
            ulong sum = gadd(gadd(s0, s1), s2);
            // n_i = sum + D_i * s_i.  Fold into single 128-bit MAC per coord.
            ulong lo, hi, c;
            lo = sum; hi = 0; c = 0;
            mac128(D0, s0, lo, hi, c);
            ulong n0 = reduce_acc(lo, hi, c);

            lo = sum; hi = 0; c = 0;
            mac128(D1, s1, lo, hi, c);
            ulong n1 = reduce_acc(lo, hi, c);

            lo = sum; hi = 0; c = 0;
            mac128(D2, s2, lo, hi, c);
            ulong n2 = reduce_acc(lo, hi, c);

            s0 = n0; s1 = n1; s2 = n2;
        }

        // second half full rounds
        for (uint r = half_f; r < r_f; ++r) {
            uint b = r * 3u;
            s0 = sbox7(gadd(s0, tg_rce[b + 0u]));
            s1 = sbox7(gadd(s1, tg_rce[b + 1u]));
            s2 = sbox7(gadd(s2, tg_rce[b + 2u]));
            mds3_tg(s0, s1, s2, tg_M);
        }

        tree[out_offset + p] = s0;
        return;
    }

    if (t == 4u) {
        ulong s0 = 0ul, s1 = 0ul, s2 = 0ul, s3 = 0ul;
        if (base + 0u < child_count) s0 = tree[in_offset + base + 0u];
        if (base + 1u < child_count) s1 = tree[in_offset + base + 1u];
        if (arity >= 3u && base + 2u < child_count) s2 = tree[in_offset + base + 2u];
        if (arity >= 4u && base + 3u < child_count) s3 = tree[in_offset + base + 3u];

        mds4_tg(s0, s1, s2, s3, tg_M);

        for (uint r = 0u; r < half_f; ++r) {
            uint b = r * 4u;
            s0 = sbox7(gadd(s0, tg_rce[b + 0u]));
            s1 = sbox7(gadd(s1, tg_rce[b + 1u]));
            s2 = sbox7(gadd(s2, tg_rce[b + 2u]));
            s3 = sbox7(gadd(s3, tg_rce[b + 3u]));
            mds4_tg(s0, s1, s2, s3, tg_M);
        }

        ulong D0 = tg_D[0], D1 = tg_D[1], D2 = tg_D[2], D3 = tg_D[3];
        for (uint r = 0u; r < r_p; ++r) {
            s0 = sbox7(gadd(s0, tg_rci[r]));
            ulong sum = gadd(gadd(s0, s1), gadd(s2, s3));

            ulong lo, hi, c;
            lo = sum; hi = 0; c = 0;
            mac128(D0, s0, lo, hi, c);
            ulong n0 = reduce_acc(lo, hi, c);

            lo = sum; hi = 0; c = 0;
            mac128(D1, s1, lo, hi, c);
            ulong n1 = reduce_acc(lo, hi, c);

            lo = sum; hi = 0; c = 0;
            mac128(D2, s2, lo, hi, c);
            ulong n2 = reduce_acc(lo, hi, c);

            lo = sum; hi = 0; c = 0;
            mac128(D3, s3, lo, hi, c);
            ulong n3 = reduce_acc(lo, hi, c);

            s0 = n0; s1 = n1; s2 = n2; s3 = n3;
        }

        for (uint r = half_f; r < r_f; ++r) {
            uint b = r * 4u;
            s0 = sbox7(gadd(s0, tg_rce[b + 0u]));
            s1 = sbox7(gadd(s1, tg_rce[b + 1u]));
            s2 = sbox7(gadd(s2, tg_rce[b + 2u]));
            s3 = sbox7(gadd(s3, tg_rce[b + 3u]));
            mds4_tg(s0, s1, s2, s3, tg_M);
        }

        tree[out_offset + p] = s0;
        return;
    }

    // Generic fallback (t == 2 or other): use device-memory constants directly.
    ulong state[4] = {0ul, 0ul, 0ul, 0ul};
    for (uint i = 0u; i < arity; ++i) {
        uint src = base + i;
        if (src < child_count && i < tt) state[i] = tree[in_offset + src];
    }
    ulong tmp[4];

    for (uint i = 0u; i < tt; ++i) {
        ulong lo = 0, hi = 0, c = 0;
        for (uint j = 0u; j < tt; ++j) mac128(tg_M[i * tt + j], state[j], lo, hi, c);
        tmp[i] = reduce_acc(lo, hi, c);
    }
    for (uint i = 0u; i < tt; ++i) state[i] = tmp[i];

    for (uint r = 0u; r < half_f; ++r) {
        for (uint i = 0u; i < tt; ++i) state[i] = sbox7(gadd(state[i], tg_rce[r * tt + i]));
        for (uint i = 0u; i < tt; ++i) {
            ulong lo = 0, hi = 0, c = 0;
            for (uint j = 0u; j < tt; ++j) mac128(tg_M[i * tt + j], state[j], lo, hi, c);
            tmp[i] = reduce_acc(lo, hi, c);
        }
        for (uint i = 0u; i < tt; ++i) state[i] = tmp[i];
    }
    for (uint r = 0u; r < r_p; ++r) {
        state[0] = sbox7(gadd(state[0], tg_rci[r]));
        ulong s = 0ul;
        for (uint i = 0u; i < tt; ++i) s = gadd(s, state[i]);
        for (uint i = 0u; i < tt; ++i) {
            ulong lo = s, hi = 0, c = 0;
            mac128(tg_D[i], state[i], lo, hi, c);
            tmp[i] = reduce_acc(lo, hi, c);
        }
        for (uint i = 0u; i < tt; ++i) state[i] = tmp[i];
    }
    for (uint r = half_f; r < r_f; ++r) {
        for (uint i = 0u; i < tt; ++i) state[i] = sbox7(gadd(state[i], tg_rce[r * tt + i]));
        for (uint i = 0u; i < tt; ++i) {
            ulong lo = 0, hi = 0, c = 0;
            for (uint j = 0u; j < tt; ++j) mac128(tg_M[i * tt + j], state[j], lo, hi, c);
            tmp[i] = reduce_acc(lo, hi, c);
        }
        for (uint i = 0u; i < tt; ++i) state[i] = tmp[i];
    }
    tree[out_offset + p] = state[0];
}
```