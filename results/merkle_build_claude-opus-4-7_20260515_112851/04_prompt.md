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

// Canonical add: inputs < p, output < p.
inline ulong gadd(ulong a, ulong b) {
    ulong s = a + b;
    if (s < a) s += EPSILON;          // wrap: add 2^64 - p = EPSILON
    if (s >= P_GOLD) s -= P_GOLD;
    return s;
}

// 64x64 -> 128 multiply.
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

// Goldilocks reduce: given x = lo + (hi<<64), produce x mod p in [0,p).
// Using x = a + b*2^64 = a + b*(2^32 - 1) mod p
//   Let hi = h_hi*2^32 + h_lo. Then b*2^64 ≡ b*2^32 - b
//   = (h_hi*2^64 + h_lo*2^32) - (h_hi*2^32 + h_lo)
//   ≡ (h_hi*(2^32-1) + h_lo*2^32) - h_hi*2^32 - h_lo
//   = h_hi*2^32 - h_hi + h_lo*2^32 - h_hi*2^32 - h_lo  ... use known formula
inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong x_hi_lo = x_hi & EPSILON;
    ulong x_hi_hi = x_hi >> 32;

    // t0 = x_lo - x_hi_hi (mod p): subtract with borrow correction
    ulong t0 = x_lo - x_hi_hi;
    if (t0 > x_lo) t0 -= EPSILON;

    // t1 = x_hi_lo * (2^32 - 1) = (x_hi_lo << 32) - x_hi_lo
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

inline ulong sbox7(ulong x) {
    ulong x2 = gmul(x, x);
    ulong x4 = gmul(x2, x2);
    ulong x3 = gmul(x2, x);
    return gmul(x4, x3);
}

// ----- t=3 specialized permutation, with register-cached constants -----
inline void poseidon2_t3(thread ulong &s0, thread ulong &s1, thread ulong &s2,
                         device const ulong *rc_ext,
                         device const ulong *rc_int,
                         const thread ulong M[9],
                         const thread ulong D[3],
                         uint r_f, uint r_p)
{
    // Pre external MDS
    {
        ulong n0 = gadd(gadd(gmul(M[0], s0), gmul(M[1], s1)), gmul(M[2], s2));
        ulong n1 = gadd(gadd(gmul(M[3], s0), gmul(M[4], s1)), gmul(M[5], s2));
        ulong n2 = gadd(gadd(gmul(M[6], s0), gmul(M[7], s1)), gmul(M[8], s2));
        s0 = n0; s1 = n1; s2 = n2;
    }

    uint half_f = r_f >> 1u;

    // Cache full-round constants in registers (up to r_f * t = 8 * 3 = 24)
    ulong RCE[24];
    uint rce_count = r_f * 3u;
    for (uint k = 0u; k < rce_count; ++k) RCE[k] = rc_ext[k];

    // Cache partial-round constants in registers (up to 32)
    ulong RCI[32];
    for (uint k = 0u; k < r_p; ++k) RCI[k] = rc_int[k];

    for (uint r = 0u; r < half_f; ++r) {
        uint b = r * 3u;
        s0 = sbox7(gadd(s0, RCE[b + 0u]));
        s1 = sbox7(gadd(s1, RCE[b + 1u]));
        s2 = sbox7(gadd(s2, RCE[b + 2u]));
        ulong n0 = gadd(gadd(gmul(M[0], s0), gmul(M[1], s1)), gmul(M[2], s2));
        ulong n1 = gadd(gadd(gmul(M[3], s0), gmul(M[4], s1)), gmul(M[5], s2));
        ulong n2 = gadd(gadd(gmul(M[6], s0), gmul(M[7], s1)), gmul(M[8], s2));
        s0 = n0; s1 = n1; s2 = n2;
    }

    // Partial rounds: only s0 gets s-box; MDS is M_I = J + diag(D)
    // n_i = sum + D[i] * s_i, where sum = s0+s1+s2
    for (uint r = 0u; r < r_p; ++r) {
        s0 = sbox7(gadd(s0, RCI[r]));
        ulong sum = gadd(gadd(s0, s1), s2);
        ulong n0 = gadd(sum, gmul(D[0], s0));
        ulong n1 = gadd(sum, gmul(D[1], s1));
        ulong n2 = gadd(sum, gmul(D[2], s2));
        s0 = n0; s1 = n1; s2 = n2;
    }

    for (uint r = half_f; r < r_f; ++r) {
        uint b = r * 3u;
        s0 = sbox7(gadd(s0, RCE[b + 0u]));
        s1 = sbox7(gadd(s1, RCE[b + 1u]));
        s2 = sbox7(gadd(s2, RCE[b + 2u]));
        ulong n0 = gadd(gadd(gmul(M[0], s0), gmul(M[1], s1)), gmul(M[2], s2));
        ulong n1 = gadd(gadd(gmul(M[3], s0), gmul(M[4], s1)), gmul(M[5], s2));
        ulong n2 = gadd(gadd(gmul(M[6], s0), gmul(M[7], s1)), gmul(M[8], s2));
        s0 = n0; s1 = n1; s2 = n2;
    }
}

// ----- t=4 specialized permutation -----
inline void poseidon2_t4(thread ulong &s0, thread ulong &s1, thread ulong &s2, thread ulong &s3,
                         device const ulong *rc_ext,
                         device const ulong *rc_int,
                         const thread ulong M[16],
                         const thread ulong D[4],
                         uint r_f, uint r_p)
{
    {
        ulong n0 = gadd(gadd(gmul(M[ 0], s0), gmul(M[ 1], s1)), gadd(gmul(M[ 2], s2), gmul(M[ 3], s3)));
        ulong n1 = gadd(gadd(gmul(M[ 4], s0), gmul(M[ 5], s1)), gadd(gmul(M[ 6], s2), gmul(M[ 7], s3)));
        ulong n2 = gadd(gadd(gmul(M[ 8], s0), gmul(M[ 9], s1)), gadd(gmul(M[10], s2), gmul(M[11], s3)));
        ulong n3 = gadd(gadd(gmul(M[12], s0), gmul(M[13], s1)), gadd(gmul(M[14], s2), gmul(M[15], s3)));
        s0 = n0; s1 = n1; s2 = n2; s3 = n3;
    }

    uint half_f = r_f >> 1u;

    ulong RCE[32];
    uint rce_count = r_f * 4u;
    for (uint k = 0u; k < rce_count; ++k) RCE[k] = rc_ext[k];

    ulong RCI[32];
    for (uint k = 0u; k < r_p; ++k) RCI[k] = rc_int[k];

    for (uint r = 0u; r < half_f; ++r) {
        uint b = r * 4u;
        s0 = sbox7(gadd(s0, RCE[b + 0u]));
        s1 = sbox7(gadd(s1, RCE[b + 1u]));
        s2 = sbox7(gadd(s2, RCE[b + 2u]));
        s3 = sbox7(gadd(s3, RCE[b + 3u]));
        ulong n0 = gadd(gadd(gmul(M[ 0], s0), gmul(M[ 1], s1)), gadd(gmul(M[ 2], s2), gmul(M[ 3], s3)));
        ulong n1 = gadd(gadd(gmul(M[ 4], s0), gmul(M[ 5], s1)), gadd(gmul(M[ 6], s2), gmul(M[ 7], s3)));
        ulong n2 = gadd(gadd(gmul(M[ 8], s0), gmul(M[ 9], s1)), gadd(gmul(M[10], s2), gmul(M[11], s3)));
        ulong n3 = gadd(gadd(gmul(M[12], s0), gmul(M[13], s1)), gadd(gmul(M[14], s2), gmul(M[15], s3)));
        s0 = n0; s1 = n1; s2 = n2; s3 = n3;
    }

    for (uint r = 0u; r < r_p; ++r) {
        s0 = sbox7(gadd(s0, RCI[r]));
        ulong sum = gadd(gadd(s0, s1), gadd(s2, s3));
        ulong n0 = gadd(sum, gmul(D[0], s0));
        ulong n1 = gadd(sum, gmul(D[1], s1));
        ulong n2 = gadd(sum, gmul(D[2], s2));
        ulong n3 = gadd(sum, gmul(D[3], s3));
        s0 = n0; s1 = n1; s2 = n2; s3 = n3;
    }

    for (uint r = half_f; r < r_f; ++r) {
        uint b = r * 4u;
        s0 = sbox7(gadd(s0, RCE[b + 0u]));
        s1 = sbox7(gadd(s1, RCE[b + 1u]));
        s2 = sbox7(gadd(s2, RCE[b + 2u]));
        s3 = sbox7(gadd(s3, RCE[b + 3u]));
        ulong n0 = gadd(gadd(gmul(M[ 0], s0), gmul(M[ 1], s1)), gadd(gmul(M[ 2], s2), gmul(M[ 3], s3)));
        ulong n1 = gadd(gadd(gmul(M[ 4], s0), gmul(M[ 5], s1)), gadd(gmul(M[ 6], s2), gmul(M[ 7], s3)));
        ulong n2 = gadd(gadd(gmul(M[ 8], s0), gmul(M[ 9], s1)), gadd(gmul(M[10], s2), gmul(M[11], s3)));
        ulong n3 = gadd(gadd(gmul(M[12], s0), gmul(M[13], s1)), gadd(gmul(M[14], s2), gmul(M[15], s3)));
        s0 = n0; s1 = n1; s2 = n2; s3 = n3;
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

    // Generic fallback (t=2 or other)
    ulong state[4] = {0ul, 0ul, 0ul, 0ul};
    for (uint i = 0u; i < arity; ++i) {
        uint src = base + i;
        if (src < child_count && i < t) state[i] = tree[in_offset + src];
    }
    ulong tmp[4];
    // Pre MDS
    for (uint i = 0u; i < t; ++i) {
        ulong acc = 0ul;
        for (uint j = 0u; j < t; ++j) acc = gadd(acc, gmul(ext_mds[i * t + j], state[j]));
        tmp[i] = acc;
    }
    for (uint i = 0u; i < t; ++i) state[i] = tmp[i];
    uint half_f = r_f >> 1u;
    for (uint r = 0u; r < half_f; ++r) {
        for (uint i = 0u; i < t; ++i) state[i] = sbox7(gadd(state[i], rc_ext[r * t + i]));
        for (uint i = 0u; i < t; ++i) {
            ulong acc = 0ul;
            for (uint j = 0u; j < t; ++j) acc = gadd(acc, gmul(ext_mds[i * t + j], state[j]));
            tmp[i] = acc;
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
            ulong acc = 0ul;
            for (uint j = 0u; j < t; ++j) acc = gadd(acc, gmul(ext_mds[i * t + j], state[j]));
            tmp[i] = acc;
        }
        for (uint i = 0u; i < t; ++i) state[i] = tmp[i];
    }
    tree[out_offset + p] = state[0];
}
```

Result of previous attempt:
           a2_N64K: correct, 1.96 ms, 11.1 Gmodmul/s (int64) (20.7% of 53 Gops/s (int64 mul, est))
          a2_N256K: correct, 4.92 ms, 17.6 Gmodmul/s (int64) (33.1% of 53 Gops/s (int64 mul, est))
            a2_N1M: correct, 16.53 ms, 21.0 Gmodmul/s (int64) (39.4% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.2999

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;
constant ulong TWO_P   = 0xFFFFFFFE00000002ul;

// Canonical add: inputs < p, output < p.
inline ulong gadd(ulong a, ulong b) {
    ulong s = a + b;
    if (s < a) s += EPSILON;
    if (s >= P_GOLD) s -= P_GOLD;
    return s;
}

// 64x64 -> 128 multiply.
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

// Reduce 128 -> canonical [0, p).
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

inline ulong sbox7(ulong x) {
    ulong x2 = gmul(x, x);
    ulong x4 = gmul(x2, x2);
    ulong x3 = gmul(x2, x);
    return gmul(x4, x3);
}

// ----- t=3 specialized permutation -----
inline void poseidon2_t3(thread ulong &s0, thread ulong &s1, thread ulong &s2,
                         device const ulong *rc_ext,
                         device const ulong *rc_int,
                         const thread ulong M[9],
                         const thread ulong D[3],
                         uint r_f, uint r_p)
{
    // Pre external MDS
    {
        ulong n0 = gadd(gadd(gmul(M[0], s0), gmul(M[1], s1)), gmul(M[2], s2));
        ulong n1 = gadd(gadd(gmul(M[3], s0), gmul(M[4], s1)), gmul(M[5], s2));
        ulong n2 = gadd(gadd(gmul(M[6], s0), gmul(M[7], s1)), gmul(M[8], s2));
        s0 = n0; s1 = n1; s2 = n2;
    }

    uint half_f = r_f >> 1u;

    for (uint r = 0u; r < half_f; ++r) {
        uint b = r * 3u;
        s0 = sbox7(gadd(s0, rc_ext[b + 0u]));
        s1 = sbox7(gadd(s1, rc_ext[b + 1u]));
        s2 = sbox7(gadd(s2, rc_ext[b + 2u]));
        ulong n0 = gadd(gadd(gmul(M[0], s0), gmul(M[1], s1)), gmul(M[2], s2));
        ulong n1 = gadd(gadd(gmul(M[3], s0), gmul(M[4], s1)), gmul(M[5], s2));
        ulong n2 = gadd(gadd(gmul(M[6], s0), gmul(M[7], s1)), gmul(M[8], s2));
        s0 = n0; s1 = n1; s2 = n2;
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
        ulong n0 = gadd(gadd(gmul(M[0], s0), gmul(M[1], s1)), gmul(M[2], s2));
        ulong n1 = gadd(gadd(gmul(M[3], s0), gmul(M[4], s1)), gmul(M[5], s2));
        ulong n2 = gadd(gadd(gmul(M[6], s0), gmul(M[7], s1)), gmul(M[8], s2));
        s0 = n0; s1 = n1; s2 = n2;
    }
}

// ----- t=4 specialized permutation -----
inline void poseidon2_t4(thread ulong &s0, thread ulong &s1, thread ulong &s2, thread ulong &s3,
                         device const ulong *rc_ext,
                         device const ulong *rc_int,
                         const thread ulong M[16],
                         const thread ulong D[4],
                         uint r_f, uint r_p)
{
    {
        ulong n0 = gadd(gadd(gmul(M[ 0], s0), gmul(M[ 1], s1)), gadd(gmul(M[ 2], s2), gmul(M[ 3], s3)));
        ulong n1 = gadd(gadd(gmul(M[ 4], s0), gmul(M[ 5], s1)), gadd(gmul(M[ 6], s2), gmul(M[ 7], s3)));
        ulong n2 = gadd(gadd(gmul(M[ 8], s0), gmul(M[ 9], s1)), gadd(gmul(M[10], s2), gmul(M[11], s3)));
        ulong n3 = gadd(gadd(gmul(M[12], s0), gmul(M[13], s1)), gadd(gmul(M[14], s2), gmul(M[15], s3)));
        s0 = n0; s1 = n1; s2 = n2; s3 = n3;
    }

    uint half_f = r_f >> 1u;

    for (uint r = 0u; r < half_f; ++r) {
        uint b = r * 4u;
        s0 = sbox7(gadd(s0, rc_ext[b + 0u]));
        s1 = sbox7(gadd(s1, rc_ext[b + 1u]));
        s2 = sbox7(gadd(s2, rc_ext[b + 2u]));
        s3 = sbox7(gadd(s3, rc_ext[b + 3u]));
        ulong n0 = gadd(gadd(gmul(M[ 0], s0), gmul(M[ 1], s1)), gadd(gmul(M[ 2], s2), gmul(M[ 3], s3)));
        ulong n1 = gadd(gadd(gmul(M[ 4], s0), gmul(M[ 5], s1)), gadd(gmul(M[ 6], s2), gmul(M[ 7], s3)));
        ulong n2 = gadd(gadd(gmul(M[ 8], s0), gmul(M[ 9], s1)), gadd(gmul(M[10], s2), gmul(M[11], s3)));
        ulong n3 = gadd(gadd(gmul(M[12], s0), gmul(M[13], s1)), gadd(gmul(M[14], s2), gmul(M[15], s3)));
        s0 = n0; s1 = n1; s2 = n2; s3 = n3;
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
        ulong n0 = gadd(gadd(gmul(M[ 0], s0), gmul(M[ 1], s1)), gadd(gmul(M[ 2], s2), gmul(M[ 3], s3)));
        ulong n1 = gadd(gadd(gmul(M[ 4], s0), gmul(M[ 5], s1)), gadd(gmul(M[ 6], s2), gmul(M[ 7], s3)));
        ulong n2 = gadd(gadd(gmul(M[ 8], s0), gmul(M[ 9], s1)), gadd(gmul(M[10], s2), gmul(M[11], s3)));
        ulong n3 = gadd(gadd(gmul(M[12], s0), gmul(M[13], s1)), gadd(gmul(M[14], s2), gmul(M[15], s3)));
        s0 = n0; s1 = n1; s2 = n2; s3 = n3;
    }
}

// ----- generic fallback (t arbitrary <= 4) -----
inline void poseidon2_generic(thread ulong *state,
                              device const ulong *rc_ext,
                              device const ulong *rc_int,
                              device const ulong *ext_mds,
                              device const ulong *int_diag,
                              uint t, uint r_f, uint r_p)
{
    ulong tmp[4];

    // pre MDS
    for (uint i = 0u; i < t; ++i) {
        ulong acc = 0ul;
        for (uint j = 0u; j < t; ++j) acc = gadd(acc, gmul(ext_mds[i * t + j], state[j]));
        tmp[i] = acc;
    }
    for (uint i = 0u; i < t; ++i) state[i] = tmp[i];

    uint half_f = r_f >> 1u;
    for (uint r = 0u; r < half_f; ++r) {
        for (uint i = 0u; i < t; ++i) state[i] = sbox7(gadd(state[i], rc_ext[r * t + i]));
        for (uint i = 0u; i < t; ++i) {
            ulong acc = 0ul;
            for (uint j = 0u; j < t; ++j) acc = gadd(acc, gmul(ext_mds[i * t + j], state[j]));
            tmp[i] = acc;
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
            ulong acc = 0ul;
            for (uint j = 0u; j < t; ++j) acc = gadd(acc, gmul(ext_mds[i * t + j], state[j]));
            tmp[i] = acc;
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
        // load with zero-padding
        ulong s0 = 0ul, s1 = 0ul, s2 = 0ul;
        if (base + 0u < child_count) s0 = tree[in_offset + base + 0u];
        if (base + 1u < child_count) s1 = tree[in_offset + base + 1u];
        // arity could be 2 or 3; only load s2 if a child exists
        if (arity >= 3u && base + 2u < child_count) s2 = tree[in_offset + base + 2u];

        // Cache MDS + diag in registers (9 + 3 ulongs).
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

    // Generic fallback.
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
           a2_N64K: correct, 1.75 ms, 12.4 Gmodmul/s (int64) (23.2% of 53 Gops/s (int64 mul, est))
          a2_N256K: correct, 4.47 ms, 19.4 Gmodmul/s (int64) (36.4% of 53 Gops/s (int64 mul, est))
            a2_N1M: correct, 15.21 ms, 22.8 Gmodmul/s (int64) (42.8% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.3306

## History

- iter  0: compile=OK | correct=True | score=0.30787883955166867
- iter  1: compile=OK | correct=True | score=0.2823247251074376
- iter  2: compile=OK | correct=True | score=0.3306489769789361
- iter  3: compile=OK | correct=True | score=0.2998542962473718

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
