## Task: merkle_build

Level-by-level Merkle tree build over the Goldilocks field (p = 2^64 - 2^32 + 1) using Poseidon2 as the inner compression function. The tree has ``n_leaves`` input leaves at level 0; level k+1 has ``ceil(level_k / arity)`` nodes, computed by hashing groups of ``arity`` consecutive children. The build terminates at the 1-element root.

Compression convention (1 Goldilocks element per digest):
  state = [c0, c1, ..., c_{arity-1}, 0, ..., 0]   (zero-pad to width t)
  state = Poseidon2_t(state)
  parent_digest = state[0]

The Poseidon2 permutation parameters (alpha=7 S-box, ``r_f`` full rounds split half+half, ``r_p`` partial rounds, external MDS, internal-MDS diagonal ``int_diag`` with M_I = J + diag(int_diag) where J is the all-ones matrix) are all read at runtime from the bound device buffers, mirroring the layout of the Z3 ``poseidon2_hash`` task. The same kernel must therefore work at t=3 / arity=2 (in-distribution) and t=4 / arity=4 (held-out) without changes -- in particular, the kernel must use the runtime arity, the runtime t, and the runtime round-count parameters, not compile-time constants.

Tree layout: a single contiguous ``ulong`` buffer holds **all levels** concatenated -- leaves first, then each parent level, finally the 1-element root. The total length is the sum of all level node counts. The host issues one kernel dispatch per parent level, binding the per-level scalars (``in_offset``, ``out_offset``, ``child_count``); each dispatch reads from ``tree[in_offset .. in_offset + child_count)`` and writes to ``tree[out_offset .. out_offset + parent_count)`` with ``parent_count = ceil(child_count / arity)`` computed in-kernel. The serial compute encoder gives read-after-write ordering between consecutive level dispatches; the candidate need not insert any explicit barriers between levels.

Boundary policy: at each level, if ``child_count`` is not a multiple of ``arity`` the last group is padded with **zero** field elements (i.e. the missing children read as zero into the Poseidon2 state). The CPU reference uses the same policy; any other padding scheme is a correctness failure. At arity=4 with N=2^19 leaves the padding kicks in only at the topmost level (2 children -> [c0, c1, 0, 0]).

Correctness is bit-exact against the Python bigint reference applied to the entire tree (every intermediate digest is checked, not only the root). Outputs MUST be canonical (< p); a non-canonical value with the same residue class still counts as a mismatch. All test sizes satisfy t <= 4 and R_F <= 8, R_P <= 32; thread-private state arrays of size 4 are sufficient.

## Required kernel signature(s)

```
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
    uint p [[thread_position_in_grid]]);

Dispatch (host-fixed, one call per parent level):
  threadsPerGrid        = (parent_count, 1, 1)   rounded up to the TG width
  threadsPerThreadgroup = (min(parent_count, 64), 1, 1)
Each thread owns ONE parent; guard against p >= parent_count (the grid is rounded up to a multiple of the TG width). Threadgroup-cooperative and simdgroup-cooperative implementations are valid so long as the external buffer layout above is preserved and the host's one-dispatch-per-level invocation pattern is honored (the kernel sees exactly one level's worth of parents per dispatch via ``child_count``, ``in_offset``, ``out_offset``).
```

## Your previous attempt

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

Result of previous attempt:
           a2_N64K: correct, 1.59 ms, 13.7 Gmodmul/s (int64) (25.6% of 53 Gops/s (int64 mul, est))
          a2_N256K: correct, 3.98 ms, 21.8 Gmodmul/s (int64) (40.9% of 53 Gops/s (int64 mul, est))
            a2_N1M: correct, 13.42 ms, 25.9 Gmodmul/s (int64) (48.5% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.3702

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;

inline ulong gadd(ulong a, ulong b) {
    ulong s = a + b;
    if (s < a) s += EPSILON;
    if (s >= P_GOLD) s -= P_GOLD;
    return s;
}

inline void umul128(ulong a, ulong b, thread ulong &lo, thread ulong &hi) {
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

// Reduce 128-bit value with x_hi possibly up to ~2^62 (still safe as long as
// computations don't overflow intermediate u64). We use the standard
// Goldilocks reduction: x = x_lo + x_hi_lo * 2^32 - x_hi_lo - x_hi_hi (mod p).
inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
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

inline ulong gmul(ulong a, ulong b) {
    ulong lo, hi;
    umul128(a, b, lo, hi);
    return gold_reduce128(lo, hi);
}

// Accumulate a*b into (acc_lo, acc_hi, acc_carry). Sum of up to 4 products of
// canonical values fits in 130 bits, so a 3-limb accumulator (64+64+ small) is safe.
inline void mac128(ulong a, ulong b,
                   thread ulong &acc_lo, thread ulong &acc_hi, thread ulong &acc_c) {
    ulong lo, hi;
    umul128(a, b, lo, hi);
    ulong new_lo = acc_lo + lo;
    ulong c1 = (ulong)(new_lo < acc_lo);
    acc_lo = new_lo;
    ulong new_hi = acc_hi + hi + c1;
    ulong c2 = (ulong)(new_hi < acc_hi) | (ulong)(new_hi == acc_hi && (hi + c1) != 0ul && new_hi < hi + c1);
    // Simpler: detect carry by comparing.
    // Actually: new_hi = acc_hi + (hi + c1); carry if new_hi < acc_hi OR (hi+c1 overflowed).
    // hi+c1 overflows only if hi == ~0 and c1==1; treat as edge.
    ulong sum2 = hi + c1;
    ulong cA = (ulong)(sum2 < hi);
    ulong new_hi2 = acc_hi + sum2;
    ulong cB = (ulong)(new_hi2 < acc_hi);
    acc_hi = new_hi2;
    acc_c += cA + cB;
    (void)c2; (void)new_hi;
}

// Reduce a 3-limb accumulator (lo + hi*2^64 + c*2^128) mod p.
// p = 2^64 - 2^32 + 1.  2^64 ≡ 2^32 - 1 (mod p).  2^128 ≡ (2^32-1)^2 mod p.
// We fold: total = lo + hi*(2^32-1) + c*(2^32-1)^2 effectively, then reduce.
// Simpler: reduce (hi, c) first to a single u64-ish, then call gold_reduce128.
inline ulong reduce_acc(ulong acc_lo, ulong acc_hi, ulong acc_c) {
    // First reduce the top: combine acc_c (small) into acc_hi-style.
    // We have value V = acc_lo + acc_hi * 2^64 + acc_c * 2^128.
    // 2^128 mod p: compute once. (2^32-1)^2 = 2^64 - 2^33 + 1 ≡ (2^32-1) - 2^33 + 1 (mod p)
    //   = 2^32 - 1 - 2^33 + 1 = -2^32 (mod p) = p - 2^32 = 2^64 - 2*2^32 + 1.
    // So acc_c * (p - 2^32) ≡ -acc_c * 2^32 (mod p).
    // Just call gold_reduce128 twice via a loop: first reduce (acc_hi, acc_c) to a 64-bit-ish residue,
    // then combine with acc_lo.
    if (acc_c != 0ul) {
        // Reduce (acc_hi as low, acc_c as high) to a u64 in [0, p) then treat it as extra high contribution.
        ulong top = gold_reduce128(acc_hi, acc_c); // this is (acc_hi + acc_c*2^64) mod p, in [0,p)
        // Now value = acc_lo + top * 2^64.
        return gold_reduce128(acc_lo, top);
    }
    return gold_reduce128(acc_lo, acc_hi);
}

inline ulong sbox7(ulong x) {
    ulong x2 = gmul(x, x);
    ulong x4 = gmul(x2, x2);
    ulong x3 = gmul(x2, x);
    return gmul(x4, x3);
}

// Lazy 3x3 MDS multiply with single reduction per row.
inline void mds3(thread ulong &s0, thread ulong &s1, thread ulong &s2,
                 const thread ulong M[9]) {
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

inline void mds4(thread ulong &s0, thread ulong &s1, thread ulong &s2, thread ulong &s3,
                 const thread ulong M[16]) {
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

inline void poseidon2_t3(thread ulong &s0, thread ulong &s1, thread ulong &s2,
                         device const ulong *rc_ext,
                         device const ulong *rc_int,
                         const thread ulong M[9],
                         const thread ulong D[3],
                         uint r_f, uint r_p)
{
    mds3(s0, s1, s2, M);

    uint half_f = r_f >> 1u;

    for (uint r = 0u; r < half_f; ++r) {
        uint b = r * 3u;
        s0 = sbox7(gadd(s0, rc_ext[b + 0u]));
        s1 = sbox7(gadd(s1, rc_ext[b + 1u]));
        s2 = sbox7(gadd(s2, rc_ext[b + 2u]));
        mds3(s0, s1, s2, M);
    }

    for (uint r = 0u; r < r_p; ++r) {
        s0 = sbox7(gadd(s0, rc_int[r]));
        ulong sum = gadd(gadd(s0, s1), s2);
        ulong n0 = gadd(sum, gmul(D[0], s0));
        ulong n1 = gadd(sum, gmul(D[1], s1));
        ulong n2 = gadd(sum, gmul(D[2], s2));
        s0 = n0; s1 = n1; s2 = n2;
    }

    for (uint r = half_f; r < r_f; ++r) {
        uint b = r * 3u;
        s0 = sbox7(gadd(s0, rc_ext[b + 0u]));
        s1 = sbox7(gadd(s1, rc_ext[b + 1u]));
        s2 = sbox7(gadd(s2, rc_ext[b + 2u]));
        mds3(s0, s1, s2, M);
    }
}

inline void poseidon2_t4(thread ulong &s0, thread ulong &s1, thread ulong &s2, thread ulong &s3,
                         device const ulong *rc_ext,
                         device const ulong *rc_int,
                         const thread ulong M[16],
                         const thread ulong D[4],
                         uint r_f, uint r_p)
{
    mds4(s0, s1, s2, s3, M);

    uint half_f = r_f >> 1u;

    for (uint r = 0u; r < half_f; ++r) {
        uint b = r * 4u;
        s0 = sbox7(gadd(s0, rc_ext[b + 0u]));
        s1 = sbox7(gadd(s1, rc_ext[b + 1u]));
        s2 = sbox7(gadd(s2, rc_ext[b + 2u]));
        s3 = sbox7(gadd(s3, rc_ext[b + 3u]));
        mds4(s0, s1, s2, s3, M);
    }

    for (uint r = 0u; r < r_p; ++r) {
        s0 = sbox7(gadd(s0, rc_int[r]));
        ulong sum = gadd(gadd(s0, s1), gadd(s2, s3));
        ulong n0 = gadd(sum, gmul(D[0], s0));
        ulong n1 = gadd(sum, gmul(D[1], s1));
        ulong n2 = gadd(sum, gmul(D[2], s2));
        ulong n3 = gadd(sum, gmul(D[3], s3));
        s0 = n0; s1 = n1; s2 = n2; s3 = n3;
    }

    for (uint r = half_f; r < r_f; ++r) {
        uint b = r * 4u;
        s0 = sbox7(gadd(s0, rc_ext[b + 0u]));
        s1 = sbox7(gadd(s1, rc_ext[b + 1u]));
        s2 = sbox7(gadd(s2, rc_ext[b + 2u]));
        s3 = sbox7(gadd(s3, rc_ext[b + 3u]));
        mds4(s0, s1, s2, s3, M);
    }
}

inline void poseidon2_generic(thread ulong *state,
                              device const ulong *rc_ext,
                              device const ulong *rc_int,
                              device const ulong *ext_mds,
                              device const ulong *int_diag,
                              uint t, uint r_f, uint r_p)
{
    ulong tmp[4];

    for (uint i = 0u; i < t; ++i) {
        ulong lo = 0, hi = 0, c = 0;
        for (uint j = 0u; j < t; ++j) mac128(ext_mds[i * t + j], state[j], lo, hi, c);
        tmp[i] = reduce_acc(lo, hi, c);
    }
    for (uint i = 0u; i < t; ++i) state[i] = tmp[i];

    uint half_f = r_f >> 1u;
    for (uint r = 0u; r < half_f; ++r) {
        for (uint i = 0u; i < t; ++i) state[i] = sbox7(gadd(state[i], rc_ext[r * t + i]));
        for (uint i = 0u; i < t; ++i) {
            ulong lo = 0, hi = 0, c = 0;
            for (uint j = 0u; j < t; ++j) mac128(ext_mds[i * t + j], state[j], lo, hi, c);
            tmp[i] = reduce_acc(lo, hi, c);
        }
        for (uint i = 0u; i < t; ++i) state[i] = tmp[i];
    }
    for (uint r = 0u; r < r_p; ++r) {
        state[0] = sbox7(gadd(state[0], rc_int[r]));
        ulong s = 0ul;
        for (uint i = 0u; i < t; ++i) s = gadd(s, state[i]);
        for (uint i = 0u; i < t; ++i) tmp[i] = gadd(s, gmul(int_diag[i], state[i]));
        for (uint i = 0u; i < t; ++i) state[i] = tmp[i];
    }
    for (uint r = half_f; r < r_f; ++r) {
        for (uint i = 0u; i < t; ++i) state[i] = sbox7(gadd(state[i], rc_ext[r * t + i]));
        for (uint i = 0u; i < t; ++i) {
            ulong lo = 0, hi = 0, c = 0;
            for (uint j = 0u; j < t; ++j) mac128(ext_mds[i * t + j], state[j], lo, hi, c);
            tmp[i] = reduce_acc(lo, hi, c);
        }
        for (uint i = 0u; i < t; ++i) state[i] = tmp[i];
    }
}

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
    uint p [[thread_position_in_grid]])
{
    uint parent_count = (child_count + arity - 1u) / arity;
    if (p >= parent_count) return;

    uint base = p * arity;

    if (t == 3u) {
        ulong s0 = 0ul, s1 = 0ul, s2 = 0ul;
        if (base + 0u < child_count) s0 = tree[in_offset + base + 0u];
        if (base + 1u < child_count) s1 = tree[in_offset + base + 1u];
        if (arity >= 3u && base + 2u < child_count) s2 = tree[in_offset + base + 2u];

        ulong M[9];
        for (uint k = 0u; k < 9u; ++k) M[k] = ext_mds[k];
        ulong D[3];
        D[0] = int_diag[0]; D[1] = int_diag[1]; D[2] = int_diag[2];

        poseidon2_t3(s0, s1, s2, rc_ext, rc_int, M, D, r_f, r_p);

        tree[out_offset + p] = s0;
        return;
    }

    if (t == 4u) {
        ulong s0 = 0ul, s1 = 0ul, s2 = 0ul, s3 = 0ul;
        if (base + 0u < child_count) s0 = tree[in_offset + base + 0u];
        if (base + 1u < child_count) s1 = tree[in_offset + base + 1u];
        if (arity >= 3u && base + 2u < child_count) s2 = tree[in_offset + base + 2u];
        if (arity >= 4u && base + 3u < child_count) s3 = tree[in_offset + base + 3u];

        ulong M[16];
        for (uint k = 0u; k < 16u; ++k) M[k] = ext_mds[k];
        ulong D[4];
        D[0] = int_diag[0]; D[1] = int_diag[1]; D[2] = int_diag[2]; D[3] = int_diag[3];

        poseidon2_t4(s0, s1, s2, s3, rc_ext, rc_int, M, D, r_f, r_p);

        tree[out_offset + p] = s0;
        return;
    }

    ulong state[4] = {0ul, 0ul, 0ul, 0ul};
    for (uint i = 0u; i < arity; ++i) {
        uint src = base + i;
        if (src < child_count && i < t) state[i] = tree[in_offset + src];
    }
    poseidon2_generic(state, rc_ext, rc_int, ext_mds, int_diag, t, r_f, r_p);
    tree[out_offset + p] = state[0];
}
```

Incumbent result:
           a2_N64K: correct, 1.54 ms, 14.1 Gmodmul/s (int64) (26.4% of 53 Gops/s (int64 mul, est))
          a2_N256K: correct, 3.97 ms, 21.8 Gmodmul/s (int64) (40.9% of 53 Gops/s (int64 mul, est))
            a2_N1M: correct, 13.54 ms, 25.6 Gmodmul/s (int64) (48.1% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.3730

## History

- iter  1: compile=OK | correct=True | score=0.2823247251074376
- iter  2: compile=OK | correct=True | score=0.3306489769789361
- iter  3: compile=OK | correct=True | score=0.2998542962473718
- iter  4: compile=OK | correct=True | score=0.3242350932019218
- iter  5: compile=OK | correct=True | score=0.37302671448914543
- iter  6: compile=OK | correct=True | score=0.3294619463708326
- iter  7: compile=OK | correct=True | score=0.3595747297599333
- iter  8: compile=OK | correct=True | score=0.3702054103487961

## Stagnation notice

Your last 3 correct attempts all scored within 15% of
the incumbent without surpassing it. You are circling a local
optimum. STOP making incremental edits to the previous kernel and
propose a STRUCTURALLY different approach.

A reworded version of the previous kernel will not break out of
this plateau.

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
