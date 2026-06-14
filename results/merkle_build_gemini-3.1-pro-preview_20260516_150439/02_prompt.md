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

constant ulong P_GOLD = 0xFFFFFFFF00000001ul;

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    ulong c = (t < a) ? 0xFFFFFFFFul : 0ul;
    t += c;
    return (t >= P_GOLD) ? t - P_GOLD : t;
}

inline ulong gold_mul(ulong a, ulong b) {
    uint a0 = (uint)a, a1 = (uint)(a >> 32);
    uint b0 = (uint)b, b1 = (uint)(b >> 32);

    ulong p00 = (ulong)a0 * b0;
    ulong p01 = (ulong)a0 * b1;
    ulong p10 = (ulong)a1 * b0;
    ulong p11 = (ulong)a1 * b1;

    ulong mid = (p00 >> 32) + (uint)p01 + (uint)p10;
    ulong lo = (uint)p00 | (mid << 32);
    ulong hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);

    ulong hi_hi = hi >> 32;
    ulong hi_lo = (uint)hi;

    ulong t0 = lo - hi_hi;
    if (t0 > lo) t0 -= 0xFFFFFFFFul;

    ulong t1 = ((ulong)hi_lo << 32) - hi_lo;
    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += 0xFFFFFFFFul;

    return t2 >= P_GOLD ? t2 - P_GOLD : t2;
}

inline ulong sbox7(ulong x) {
    ulong x2 = gold_mul(x, x);
    ulong x4 = gold_mul(x2, x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
}

inline void poseidon2_permute_3(thread ulong &s0, thread ulong &s1, thread ulong &s2,
                                device const ulong *rc_ext,
                                device const ulong *rc_int,
                                device const ulong *ext_mds,
                                device const ulong *int_diag,
                                uint r_f, uint r_p)
{
    ulong m00 = ext_mds[0], m01 = ext_mds[1], m02 = ext_mds[2];
    ulong m10 = ext_mds[3], m11 = ext_mds[4], m12 = ext_mds[5];
    ulong m20 = ext_mds[6], m21 = ext_mds[7], m22 = ext_mds[8];

    ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2];

    ulong n0 = gold_add(gold_add(gold_mul(m00, s0), gold_mul(m01, s1)), gold_mul(m02, s2));
    ulong n1 = gold_add(gold_add(gold_mul(m10, s0), gold_mul(m11, s1)), gold_mul(m12, s2));
    ulong n2 = gold_add(gold_add(gold_mul(m20, s0), gold_mul(m21, s1)), gold_mul(m22, s2));
    s0 = n0; s1 = n1; s2 = n2;

    uint half_f = r_f >> 1;
    uint rc_idx = 0;

    for (uint r = 0; r < half_f; ++r) {
        s0 = sbox7(gold_add(s0, rc_ext[rc_idx]));
        s1 = sbox7(gold_add(s1, rc_ext[rc_idx + 1]));
        s2 = sbox7(gold_add(s2, rc_ext[rc_idx + 2]));
        rc_idx += 3;
        
        n0 = gold_add(gold_add(gold_mul(m00, s0), gold_mul(m01, s1)), gold_mul(m02, s2));
        n1 = gold_add(gold_add(gold_mul(m10, s0), gold_mul(m11, s1)), gold_mul(m12, s2));
        n2 = gold_add(gold_add(gold_mul(m20, s0), gold_mul(m21, s1)), gold_mul(m22, s2));
        s0 = n0; s1 = n1; s2 = n2;
    }

    for (uint r = 0; r < r_p; ++r) {
        s0 = sbox7(gold_add(s0, rc_int[r]));
        
        ulong sum = gold_add(gold_add(s0, s1), s2);
        
        n0 = gold_add(sum, gold_mul(d0, s0));
        n1 = gold_add(sum, gold_mul(d1, s1));
        n2 = gold_add(sum, gold_mul(d2, s2));
        s0 = n0; s1 = n1; s2 = n2;
    }

    for (uint r = half_f; r < r_f; ++r) {
        s0 = sbox7(gold_add(s0, rc_ext[rc_idx]));
        s1 = sbox7(gold_add(s1, rc_ext[rc_idx + 1]));
        s2 = sbox7(gold_add(s2, rc_ext[rc_idx + 2]));
        rc_idx += 3;
        
        n0 = gold_add(gold_add(gold_mul(m00, s0), gold_mul(m01, s1)), gold_mul(m02, s2));
        n1 = gold_add(gold_add(gold_mul(m10, s0), gold_mul(m11, s1)), gold_mul(m12, s2));
        n2 = gold_add(gold_add(gold_mul(m20, s0), gold_mul(m21, s1)), gold_mul(m22, s2));
        s0 = n0; s1 = n1; s2 = n2;
    }
}

inline void poseidon2_permute_4(thread ulong &s0, thread ulong &s1, thread ulong &s2, thread ulong &s3,
                                device const ulong *rc_ext,
                                device const ulong *rc_int,
                                device const ulong *ext_mds,
                                device const ulong *int_diag,
                                uint r_f, uint r_p)
{
    ulong m00 = ext_mds[0], m01 = ext_mds[1], m02 = ext_mds[2], m03 = ext_mds[3];
    ulong m10 = ext_mds[4], m11 = ext_mds[5], m12 = ext_mds[6], m13 = ext_mds[7];
    ulong m20 = ext_mds[8], m21 = ext_mds[9], m22 = ext_mds[10], m23 = ext_mds[11];
    ulong m30 = ext_mds[12], m31 = ext_mds[13], m32 = ext_mds[14], m33 = ext_mds[15];

    ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2], d3 = int_diag[3];

    ulong n0 = gold_add(gold_add(gold_mul(m00, s0), gold_mul(m01, s1)), gold_add(gold_mul(m02, s2), gold_mul(m03, s3)));
    ulong n1 = gold_add(gold_add(gold_mul(m10, s0), gold_mul(m11, s1)), gold_add(gold_mul(m12, s2), gold_mul(m13, s3)));
    ulong n2 = gold_add(gold_add(gold_mul(m20, s0), gold_mul(m21, s1)), gold_add(gold_mul(m22, s2), gold_mul(m23, s3)));
    ulong n3 = gold_add(gold_add(gold_mul(m30, s0), gold_mul(m31, s1)), gold_add(gold_mul(m32, s2), gold_mul(m33, s3)));
    s0 = n0; s1 = n1; s2 = n2; s3 = n3;

    uint half_f = r_f >> 1;
    uint rc_idx = 0;

    for (uint r = 0; r < half_f; ++r) {
        s0 = sbox7(gold_add(s0, rc_ext[rc_idx]));
        s1 = sbox7(gold_add(s1, rc_ext[rc_idx + 1]));
        s2 = sbox7(gold_add(s2, rc_ext[rc_idx + 2]));
        s3 = sbox7(gold_add(s3, rc_ext[rc_idx + 3]));
        rc_idx += 4;
        
        n0 = gold_add(gold_add(gold_mul(m00, s0), gold_mul(m01, s1)), gold_add(gold_mul(m02, s2), gold_mul(m03, s3)));
        n1 = gold_add(gold_add(gold_mul(m10, s0), gold_mul(m11, s1)), gold_add(gold_mul(m12, s2), gold_mul(m13, s3)));
        n2 = gold_add(gold_add(gold_mul(m20, s0), gold_mul(m21, s1)), gold_add(gold_mul(m22, s2), gold_mul(m23, s3)));
        n3 = gold_add(gold_add(gold_mul(m30, s0), gold_mul(m31, s1)), gold_add(gold_mul(m32, s2), gold_mul(m33, s3)));
        s0 = n0; s1 = n1; s2 = n2; s3 = n3;
    }

    for (uint r = 0; r < r_p; ++r) {
        s0 = sbox7(gold_add(s0, rc_int[r]));
        
        ulong sum = gold_add(gold_add(s0, s1), gold_add(s2, s3));
        
        n0 = gold_add(sum, gold_mul(d0, s0));
        n1 = gold_add(sum, gold_mul(d1, s1));
        n2 = gold_add(sum, gold_mul(d2, s2));
        n3 = gold_add(sum, gold_mul(d3, s3));
        s0 = n0; s1 = n1; s2 = n2; s3 = n3;
    }

    for (uint r = half_f; r < r_f; ++r) {
        s0 = sbox7(gold_add(s0, rc_ext[rc_idx]));
        s1 = sbox7(gold_add(s1, rc_ext[rc_idx + 1]));
        s2 = sbox7(gold_add(s2, rc_ext[rc_idx + 2]));
        s3 = sbox7(gold_add(s3, rc_ext[rc_idx + 3]));
        rc_idx += 4;
        
        n0 = gold_add(gold_add(gold_mul(m00, s0), gold_mul(m01, s1)), gold_add(gold_mul(m02, s2), gold_mul(m03, s3)));
        n1 = gold_add(gold_add(gold_mul(m10, s0), gold_mul(m11, s1)), gold_add(gold_mul(m12, s2), gold_mul(m13, s3)));
        n2 = gold_add(gold_add(gold_mul(m20, s0), gold_mul(m21, s1)), gold_add(gold_mul(m22, s2), gold_mul(m23, s3)));
        n3 = gold_add(gold_add(gold_mul(m30, s0), gold_mul(m31, s1)), gold_add(gold_mul(m32, s2), gold_mul(m33, s3)));
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

    if (t <= 3) {
        ulong s0 = 0, s1 = 0, s2 = 0;
        uint base = p * arity;
        
        if (arity > 0 && base + 0 < child_count) s0 = tree[in_offset + base + 0];
        if (arity > 1 && base + 1 < child_count) s1 = tree[in_offset + base + 1];
        if (arity > 2 && base + 2 < child_count) s2 = tree[in_offset + base + 2];
        
        poseidon2_permute_3(s0, s1, s2, rc_ext, rc_int, ext_mds, int_diag, r_f, r_p);
        tree[out_offset + p] = s0;
        
    } else {
        ulong s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        uint base = p * arity;
        
        if (arity > 0 && base + 0 < child_count) s0 = tree[in_offset + base + 0];
        if (arity > 1 && base + 1 < child_count) s1 = tree[in_offset + base + 1];
        if (arity > 2 && base + 2 < child_count) s2 = tree[in_offset + base + 2];
        if (arity > 3 && base + 3 < child_count) s3 = tree[in_offset + base + 3];
        
        poseidon2_permute_4(s0, s1, s2, s3, rc_ext, rc_int, ext_mds, int_diag, r_f, r_p);
        tree[out_offset + p] = s0;
    }
}
```

Result of previous attempt:
           a2_N64K: correct, 1.68 ms, 12.9 Gmodmul/s (int64) (24.2% of 53 Gops/s (int64 mul, est))
          a2_N256K: correct, 4.56 ms, 19.0 Gmodmul/s (int64) (35.7% of 53 Gops/s (int64 mul, est))
            a2_N1M: correct, 16.03 ms, 21.6 Gmodmul/s (int64) (40.6% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.3271

## History

- iter  0: compile=OK | correct=True | score=0.30415083000511656
- iter  1: compile=OK | correct=True | score=0.3270893139365812

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
