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

constexpr constant uint T_MAX = 4u;

inline ulong gold_add(ulong a, ulong b) {
    ulong s = a + b;
    if (s < a) s += EPSILON;          // wrap: add 2^64 mod p = EPSILON
    if (s >= P_GOLD) s -= P_GOLD;
    return s;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong d = a - b;
    if (d > a) d -= EPSILON;          // borrow: subtract EPSILON
    return d;
}

// Full 128-bit product of two 64-bit ulongs using 32-bit limbs.
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

// Reduce a 128-bit value (x_hi:x_lo) modulo Goldilocks p = 2^64 - 2^32 + 1.
// Uses: 2^64 = 2^32 - 1 (mod p), 2^96 = -1 (mod p).
inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong x_hi_lo = x_hi & EPSILON;     // low 32 bits of hi
    ulong x_hi_hi = x_hi >> 32;         // high 32 bits of hi  (* -1 mod p)

    // t0 = x_lo - x_hi_hi  (mod 2^64), correct with -EPSILON on borrow
    ulong t0 = x_lo - x_hi_hi;
    if (t0 > x_lo) t0 -= EPSILON;

    // t1 = x_hi_lo * (2^32 - 1)  fits in 64 bits since x_hi_lo < 2^32
    ulong t1 = (x_hi_lo << 32) - x_hi_lo;

    ulong r = t0 + t1;
    if (r < t0) r += EPSILON;
    if (r >= P_GOLD) r -= P_GOLD;
    return r;
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong lo, hi;
    umul128(a, b, lo, hi);
    return gold_reduce128(lo, hi);
}

inline ulong sbox7(ulong x) {
    ulong x2 = gold_mul(x, x);
    ulong x4 = gold_mul(x2, x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
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

    uint tt = t;
    uint rf = r_f;
    uint rp = r_p;
    uint half_f = rf >> 1u;

    // Cache MDS matrix and internal diagonal in registers (t <= 4).
    ulong mds[T_MAX * T_MAX];
    ulong diag[T_MAX];
    #pragma unroll
    for (uint i = 0u; i < T_MAX; ++i) {
        diag[i] = (i < tt) ? int_diag[i] : 0ul;
        #pragma unroll
        for (uint j = 0u; j < T_MAX; ++j) {
            mds[i * T_MAX + j] = (i < tt && j < tt) ? ext_mds[i * tt + j] : 0ul;
        }
    }

    // Load children (zero-pad missing).
    ulong state[T_MAX];
    #pragma unroll
    for (uint i = 0u; i < T_MAX; ++i) state[i] = 0ul;

    uint base = p * arity;
    uint avail = (child_count > base) ? (child_count - base) : 0u;
    uint take = min(avail, arity);
    for (uint i = 0u; i < take; ++i) {
        state[i] = tree[in_offset + base + i];
    }

    // ---- Pre-multiply by external MDS ----
    {
        ulong tmp[T_MAX];
        #pragma unroll
        for (uint i = 0u; i < T_MAX; ++i) {
            ulong acc = 0ul;
            #pragma unroll
            for (uint j = 0u; j < T_MAX; ++j) {
                if (j < tt) acc = gold_add(acc, gold_mul(mds[i * T_MAX + j], state[j]));
            }
            tmp[i] = acc;
        }
        #pragma unroll
        for (uint i = 0u; i < T_MAX; ++i) state[i] = tmp[i];
    }

    // ---- First half full rounds ----
    for (uint r = 0u; r < half_f; ++r) {
        uint rc_base = r * tt;
        #pragma unroll
        for (uint i = 0u; i < T_MAX; ++i) {
            if (i < tt) {
                ulong v = gold_add(state[i], rc_ext[rc_base + i]);
                state[i] = sbox7(v);
            }
        }
        ulong tmp[T_MAX];
        #pragma unroll
        for (uint i = 0u; i < T_MAX; ++i) {
            ulong acc = 0ul;
            #pragma unroll
            for (uint j = 0u; j < T_MAX; ++j) {
                if (j < tt) acc = gold_add(acc, gold_mul(mds[i * T_MAX + j], state[j]));
            }
            tmp[i] = acc;
        }
        #pragma unroll
        for (uint i = 0u; i < T_MAX; ++i) state[i] = tmp[i];
    }

    // ---- Partial rounds ----
    for (uint r = 0u; r < rp; ++r) {
        ulong v = gold_add(state[0], rc_int[r]);
        state[0] = sbox7(v);

        // Internal MDS: y[i] = sum(state) + diag[i] * state[i]
        ulong s = 0ul;
        #pragma unroll
        for (uint i = 0u; i < T_MAX; ++i) {
            if (i < tt) s = gold_add(s, state[i]);
        }
        ulong tmp[T_MAX];
        #pragma unroll
        for (uint i = 0u; i < T_MAX; ++i) {
            tmp[i] = gold_add(s, gold_mul(diag[i], state[i]));
        }
        #pragma unroll
        for (uint i = 0u; i < T_MAX; ++i) state[i] = tmp[i];
    }

    // ---- Second half full rounds ----
    for (uint r = half_f; r < rf; ++r) {
        uint rc_base = r * tt;
        #pragma unroll
        for (uint i = 0u; i < T_MAX; ++i) {
            if (i < tt) {
                ulong v = gold_add(state[i], rc_ext[rc_base + i]);
                state[i] = sbox7(v);
            }
        }
        ulong tmp[T_MAX];
        #pragma unroll
        for (uint i = 0u; i < T_MAX; ++i) {
            ulong acc = 0ul;
            #pragma unroll
            for (uint j = 0u; j < T_MAX; ++j) {
                if (j < tt) acc = gold_add(acc, gold_mul(mds[i * T_MAX + j], state[j]));
            }
            tmp[i] = acc;
        }
        #pragma unroll
        for (uint i = 0u; i < T_MAX; ++i) state[i] = tmp[i];
    }

    // Canonicalize and write.
    ulong out = state[0];
    if (out >= P_GOLD) out -= P_GOLD;
    tree[out_offset + p] = out;
}
```

Result of previous attempt:
           a2_N64K: correct, 2.11 ms, 10.3 Gmodmul/s (int64) (19.3% of 53 Gops/s (int64 mul, est))
          a2_N256K: correct, 5.22 ms, 16.6 Gmodmul/s (int64) (31.2% of 53 Gops/s (int64 mul, est))
            a2_N1M: correct, 17.41 ms, 19.9 Gmodmul/s (int64) (37.4% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.2823

## Current best (incumbent)

```metal
// Naive seed for the Merkle tree build (one thread per parent, one
// kernel dispatch per level).
//
// Inner hash: Poseidon2 over the Goldilocks field (same permutation
// used by Z3). The state width ``t``, full/partial round counts, and
// the round-constant / MDS-coefficient buffers are all read at runtime
// from the bound device buffers, so the same kernel handles both the
// in-distribution arity=2 (t=3) and held-out arity=4 (t=4) cases.
//
// Compression convention (1 Goldilocks element per digest):
//     state = [c0, c1, ..., c_{arity-1}, 0, ..., 0]   // zero-pad to width t
//     state = Poseidon2_t(state)
//     parent_digest = state[0]
//
// Tree layout: a single contiguous ``ulong`` buffer holding all levels
// concatenated -- leaves first, then each parent level, finally the
// 1-element root. The host issues one kernel dispatch per level with
// the per-level scalars (in_offset, out_offset, child_count); each
// dispatch reads from ``tree[in_offset .. in_offset + child_count)``
// and writes to ``tree[out_offset .. out_offset + parent_count)``,
// where ``parent_count = ceil(child_count / arity)`` is computed in
// the kernel. The serial compute encoder gives read-after-write
// ordering between consecutive level dispatches.
//
// Boundary policy: at each level, if child_count is not a multiple of
// arity, the last group is padded with **zero** field elements (so
// the missing children read as zero in the Poseidon2 state). The CPU
// reference uses the same policy bit-for-bit.
//
// Buffer layout (host-fixed; preserved by candidate):
//   buffer  0: device       ulong *tree           (sum of level_counts)
//   buffer  1: device const ulong *rc_ext         (r_f * t, row-major)
//   buffer  2: device const ulong *rc_int         (r_p)
//   buffer  3: device const ulong *ext_mds        (t * t, row-major)
//   buffer  4: device const ulong *int_diag       (t)
//   buffer  5: constant uint &arity               (children per parent, in {2, 4})
//   buffer  6: constant uint &t                   (Poseidon2 state width, <= 4)
//   buffer  7: constant uint &r_f                 (full rounds, even)
//   buffer  8: constant uint &r_p                 (partial rounds)
//   buffer  9: constant uint &in_offset           (per-level: start of input slice)
//   buffer 10: constant uint &out_offset          (per-level: start of output slice)
//   buffer 11: constant uint &child_count         (per-level: number of input nodes)
//
// Dispatch (host-provided, once per level):
//   threadsPerGrid        = (parent_count, 1, 1)        rounded up to TG width
//   threadsPerThreadgroup = (min(parent_count, 64), 1, 1)

#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;

constexpr constant uint T_MAX = 4u;

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    if (t < a) t += EPSILON;
    if (t >= P_GOLD) t -= P_GOLD;
    return t;
}

inline ulong2 umul128(ulong a, ulong b) {
    uint a0 = (uint)(a);
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)(b);
    uint b1 = (uint)(b >> 32);

    ulong p00 = (ulong)a0 * (ulong)b0;
    ulong p01 = (ulong)a0 * (ulong)b1;
    ulong p10 = (ulong)a1 * (ulong)b0;
    ulong p11 = (ulong)a1 * (ulong)b1;

    ulong mid = (p00 >> 32) + (p01 & EPSILON) + (p10 & EPSILON);
    ulong lo  = (p00 & EPSILON) | (mid << 32);
    ulong hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return ulong2(lo, hi);
}

inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong x_hi_lo = x_hi & EPSILON;
    ulong x_hi_hi = x_hi >> 32;

    ulong t0 = x_lo - x_hi_hi;
    if (t0 > x_lo) t0 -= EPSILON;

    ulong t1 = x_hi_lo * EPSILON;

    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;

    return gold_canonical(t2);
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong2 prod = umul128(a, b);
    return gold_reduce128(prod.x, prod.y);
}

inline ulong sbox7(ulong x) {
    // x^7 = x^4 * x^2 * x
    ulong x2 = gold_mul(x, x);
    ulong x4 = gold_mul(x2, x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
}

inline void matvec_ext(thread ulong *state,
                       device const ulong *ext_mds,
                       uint t)
{
    ulong tmp[T_MAX];
    for (uint i = 0u; i < t; ++i) {
        ulong acc = 0ul;
        for (uint j = 0u; j < t; ++j) {
            acc = gold_add(acc, gold_mul(ext_mds[i * t + j], state[j]));
        }
        tmp[i] = acc;
    }
    for (uint i = 0u; i < t; ++i) state[i] = tmp[i];
}

inline void matvec_int(thread ulong *state,
                       device const ulong *int_diag,
                       uint t)
{
    // y[i] = sum(state) + d[i] * state[i]
    ulong s = 0ul;
    for (uint i = 0u; i < t; ++i) s = gold_add(s, state[i]);
    ulong tmp[T_MAX];
    for (uint i = 0u; i < t; ++i) {
        tmp[i] = gold_add(s, gold_mul(int_diag[i], state[i]));
    }
    for (uint i = 0u; i < t; ++i) state[i] = tmp[i];
}

inline void poseidon2_permute(thread ulong *state,
                              device const ulong *rc_ext,
                              device const ulong *rc_int,
                              device const ulong *ext_mds,
                              device const ulong *int_diag,
                              uint t, uint r_f, uint r_p)
{
    // Pre-multiply by external MDS.
    matvec_ext(state, ext_mds, t);

    uint half_f = r_f >> 1u;

    // First half full rounds.
    for (uint r = 0u; r < half_f; ++r) {
        for (uint i = 0u; i < t; ++i) {
            state[i] = gold_add(state[i], rc_ext[r * t + i]);
            state[i] = sbox7(state[i]);
        }
        matvec_ext(state, ext_mds, t);
    }

    // Partial rounds.
    for (uint r = 0u; r < r_p; ++r) {
        state[0] = gold_add(state[0], rc_int[r]);
        state[0] = sbox7(state[0]);
        matvec_int(state, int_diag, t);
    }

    // Second half full rounds.
    for (uint r = half_f; r < r_f; ++r) {
        for (uint i = 0u; i < t; ++i) {
            state[i] = gold_add(state[i], rc_ext[r * t + i]);
            state[i] = sbox7(state[i]);
        }
        matvec_ext(state, ext_mds, t);
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

    thread ulong state[T_MAX];
    for (uint i = 0u; i < t; ++i) state[i] = 0ul;

    uint base = p * arity;
    for (uint i = 0u; i < arity; ++i) {
        uint src = base + i;
        if (src < child_count) {
            state[i] = tree[in_offset + src];
        }
        // else: leave state[i] = 0 (zero-padding for the boundary group)
    }

    poseidon2_permute(state, rc_ext, rc_int, ext_mds, int_diag, t, r_f, r_p);

    tree[out_offset + p] = state[0];
}
```

Incumbent result:
           a2_N64K: correct, 1.88 ms, 11.5 Gmodmul/s (int64) (21.6% of 53 Gops/s (int64 mul, est))
          a2_N256K: correct, 4.81 ms, 18.1 Gmodmul/s (int64) (33.8% of 53 Gops/s (int64 mul, est))
            a2_N1M: correct, 16.32 ms, 21.3 Gmodmul/s (int64) (39.9% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.3079

## History

- iter  0: compile=OK | correct=True | score=0.30787883955166867
- iter  1: compile=OK | correct=True | score=0.2823247251074376

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
