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
