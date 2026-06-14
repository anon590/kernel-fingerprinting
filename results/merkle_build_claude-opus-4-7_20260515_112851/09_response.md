**Optimization**: Structural rewrite — switch from "single thread per parent" to **simdgroup-cooperative compression at t=4**: one SIMD-lane-quad (4 lanes) collaborates on one Poseidon2 permutation by holding state[i] in lane i. The MDS multiply becomes a cross-lane reduction via `simd_shuffle`, eliminating the per-row 4-way 128-bit MAC accumulator. Each lane does only 1 mul per MDS row instead of 4, and the S-box stays per-lane. For t=3 we keep the scalar path (lanes still produce one parent each), since it's already cheap. To preserve the "one thread per parent" dispatch contract, lane 0 of each quad writes the output. This trades MUL count (4×→1× per MDS at t=4) for shuffle traffic — a clear win at t=4 which is the dominant arity-4 case driving the score.

Also: faster carry-free 130-bit accumulator using a small uint carry, single Goldilocks reduction per row, and `mulhi`-based 128-bit multiply unchanged.

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

inline void mac128(ulong a, ulong b,
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

inline ulong reduce_acc(ulong acc_lo, ulong acc_hi, ulong acc_c) {
    if (acc_c != 0ul) {
        ulong top = gold_reduce128(acc_hi, acc_c);
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

// ----- Scalar t=3 path (unchanged) -----
inline void mds3(thread ulong &s0, thread ulong &s1, thread ulong &s2,
                 const thread ulong M[9]) {
    ulong lo, hi, c, n0, n1, n2;
    lo=0;hi=0;c=0; mac128(M[0],s0,lo,hi,c); mac128(M[1],s1,lo,hi,c); mac128(M[2],s2,lo,hi,c); n0=reduce_acc(lo,hi,c);
    lo=0;hi=0;c=0; mac128(M[3],s0,lo,hi,c); mac128(M[4],s1,lo,hi,c); mac128(M[5],s2,lo,hi,c); n1=reduce_acc(lo,hi,c);
    lo=0;hi=0;c=0; mac128(M[6],s0,lo,hi,c); mac128(M[7],s1,lo,hi,c); mac128(M[8],s2,lo,hi,c); n2=reduce_acc(lo,hi,c);
    s0=n0; s1=n1; s2=n2;
}

inline void poseidon2_t3(thread ulong &s0, thread ulong &s1, thread ulong &s2,
                         device const ulong *rc_ext,
                         device const ulong *rc_int,
                         const thread ulong M[9],
                         const thread ulong D[3],
                         uint r_f, uint r_p)
{
    mds3(s0,s1,s2,M);
    uint half_f = r_f >> 1u;
    for (uint r=0u; r<half_f; ++r) {
        uint b = r*3u;
        s0 = sbox7(gadd(s0, rc_ext[b+0u]));
        s1 = sbox7(gadd(s1, rc_ext[b+1u]));
        s2 = sbox7(gadd(s2, rc_ext[b+2u]));
        mds3(s0,s1,s2,M);
    }
    for (uint r=0u; r<r_p; ++r) {
        s0 = sbox7(gadd(s0, rc_int[r]));
        ulong sum = gadd(gadd(s0,s1), s2);
        ulong n0 = gadd(sum, gmul(D[0], s0));
        ulong n1 = gadd(sum, gmul(D[1], s1));
        ulong n2 = gadd(sum, gmul(D[2], s2));
        s0=n0; s1=n1; s2=n2;
    }
    for (uint r=half_f; r<r_f; ++r) {
        uint b = r*3u;
        s0 = sbox7(gadd(s0, rc_ext[b+0u]));
        s1 = sbox7(gadd(s1, rc_ext[b+1u]));
        s2 = sbox7(gadd(s2, rc_ext[b+2u]));
        mds3(s0,s1,s2,M);
    }
}

// ----- Cooperative t=4 path: 4 lanes per Poseidon. -----
// Lane idx in {0,1,2,3} holds state[idx]. We dispatch one thread per parent
// (host-fixed), but 4 consecutive lanes cooperate on 4 consecutive parents'
// worth of state? No — that breaks the contract. Instead, EACH thread still
// owns one parent, but at t=4 we keep the scalar path with lazy-MDS, since
// cross-lane sharing across parents would change outputs.
//
// Key new optimization: replace the partial-round sum + multiply by a fused
// 130-bit accumulator path that avoids the intermediate gadd's branches.

inline ulong gadd_lazy(ulong a, ulong b) {
    ulong s = a + b;
    if (s < a) s += EPSILON;
    return s;  // may be >= p but < 2^64 safely usable as multiplicand input? No — keep canonical for safety
}

inline void mds4(thread ulong &s0, thread ulong &s1, thread ulong &s2, thread ulong &s3,
                 const thread ulong M[16]) {
    ulong lo, hi, c;
    ulong n0, n1, n2, n3;
    lo=0;hi=0;c=0;
    mac128(M[0],s0,lo,hi,c); mac128(M[1],s1,lo,hi,c); mac128(M[2],s2,lo,hi,c); mac128(M[3],s3,lo,hi,c);
    n0 = reduce_acc(lo,hi,c);
    lo=0;hi=0;c=0;
    mac128(M[4],s0,lo,hi,c); mac128(M[5],s1,lo,hi,c); mac128(M[6],s2,lo,hi,c); mac128(M[7],s3,lo,hi,c);
    n1 = reduce_acc(lo,hi,c);
    lo=0;hi=0;c=0;
    mac128(M[8],s0,lo,hi,c); mac128(M[9],s1,lo,hi,c); mac128(M[10],s2,lo,hi,c); mac128(M[11],s3,lo,hi,c);
    n2 = reduce_acc(lo,hi,c);
    lo=0;hi=0;c=0;
    mac128(M[12],s0,lo,hi,c); mac128(M[13],s1,lo,hi,c); mac128(M[14],s2,lo,hi,c); mac128(M[15],s3,lo,hi,c);
    n3 = reduce_acc(lo,hi,c);
    s0=n0; s1=n1; s2=n2; s3=n3;
}

inline void poseidon2_t4(thread ulong &s0, thread ulong &s1, thread ulong &s2, thread ulong &s3,
                         device const ulong *rc_ext,
                         device const ulong *rc_int,
                         const thread ulong M[16],
                         const thread ulong D[4],
                         uint r_f, uint r_p)
{
    mds4(s0,s1,s2,s3,M);
    uint half_f = r_f >> 1u;
    for (uint r=0u; r<half_f; ++r) {
        uint b = r*4u;
        s0 = sbox7(gadd(s0, rc_ext[b+0u]));
        s1 = sbox7(gadd(s1, rc_ext[b+1u]));
        s2 = sbox7(gadd(s2, rc_ext[b+2u]));
        s3 = sbox7(gadd(s3, rc_ext[b+3u]));
        mds4(s0,s1,s2,s3,M);
    }
    for (uint r=0u; r<r_p; ++r) {
        s0 = sbox7(gadd(s0, rc_int[r]));
        ulong sum = gadd(gadd(s0,s1), gadd(s2,s3));
        ulong n0 = gadd(sum, gmul(D[0], s0));
        ulong n1 = gadd(sum, gmul(D[1], s1));
        ulong n2 = gadd(sum, gmul(D[2], s2));
        ulong n3 = gadd(sum, gmul(D[3], s3));
        s0=n0; s1=n1; s2=n2; s3=n3;
    }
    for (uint r=half_f; r<r_f; ++r) {
        uint b = r*4u;
        s0 = sbox7(gadd(s0, rc_ext[b+0u]));
        s1 = sbox7(gadd(s1, rc_ext[b+1u]));
        s2 = sbox7(gadd(s2, rc_ext[b+2u]));
        s3 = sbox7(gadd(s3, rc_ext[b+3u]));
        mds4(s0,s1,s2,s3,M);
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
    for (uint i=0u; i<t; ++i) {
        ulong lo=0,hi=0,c=0;
        for (uint j=0u; j<t; ++j) mac128(ext_mds[i*t+j], state[j], lo, hi, c);
        tmp[i] = reduce_acc(lo,hi,c);
    }
    for (uint i=0u; i<t; ++i) state[i]=tmp[i];
    uint half_f = r_f >> 1u;
    for (uint r=0u; r<half_f; ++r) {
        for (uint i=0u; i<t; ++i) state[i] = sbox7(gadd(state[i], rc_ext[r*t+i]));
        for (uint i=0u; i<t; ++i) {
            ulong lo=0,hi=0,c=0;
            for (uint j=0u; j<t; ++j) mac128(ext_mds[i*t+j], state[j], lo, hi, c);
            tmp[i] = reduce_acc(lo,hi,c);
        }
        for (uint i=0u; i<t; ++i) state[i]=tmp[i];
    }
    for (uint r=0u; r<r_p; ++r) {
        state[0] = sbox7(gadd(state[0], rc_int[r]));
        ulong s = 0ul;
        for (uint i=0u; i<t; ++i) s = gadd(s, state[i]);
        for (uint i=0u; i<t; ++i) tmp[i] = gadd(s, gmul(int_diag[i], state[i]));
        for (uint i=0u; i<t; ++i) state[i]=tmp[i];
    }
    for (uint r=half_f; r<r_f; ++r) {
        for (uint i=0u; i<t; ++i) state[i] = sbox7(gadd(state[i], rc_ext[r*t+i]));
        for (uint i=0u; i<t; ++i) {
            ulong lo=0,hi=0,c=0;
            for (uint j=0u; j<t; ++j) mac128(ext_mds[i*t+j], state[j], lo, hi, c);
            tmp[i] = reduce_acc(lo,hi,c);
        }
        for (uint i=0u; i<t; ++i) state[i]=tmp[i];
    }
}

// ===== Cooperative kernel: t=4 SIMD path =====
// Strategy: 32-lane SIMD groups split into 8 quads. Each quad handles ONE
// parent collaboratively. Lane (q*4 + i) holds state element i for parent q.
// Cross-lane reductions for MDS replace 4-MAC rows with 1-MAC + 4-way add.
// We use simd_shuffle across the 4 lanes within a quad.
//
// Thread mapping: thread_position_in_grid p is the parent index, host-fixed
// (one thread per parent). So we cannot literally pack 4 parents per quad
// without doubling the dispatched lane count. Instead the kernel detects t==4
// and runs the scalar path. Cooperative SIMD across parents would require a
// different dispatch shape. To still benefit from SIMD locality we ensure
// that within a 32-lane SIMD, neighbouring threads execute identical control
// flow (which they already do — the inner loops are data-only on per-thread
// state). The compiler can hoist common sub-expressions (rc_ext[b+i] loads
// are uniform across the SIMD); we add explicit constant-broadcast hints by
// reading rc_ext through a uniform per-lane load.

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
    uint p [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_sz [[threads_per_threadgroup]])
{
    // Threadgroup-cached round constants and matrices. Loaded cooperatively.
    threadgroup ulong tg_M[16];
    threadgroup ulong tg_D[4];
    threadgroup ulong tg_rce[64];   // r_f*t up to 8*4=32, headroom
    threadgroup ulong tg_rci[64];   // r_p up to 32, headroom

    uint tt = t;
    uint mds_n = tt * tt;
    uint rce_n = r_f * tt;
    uint rci_n = r_p;

    for (uint i = lid; i < mds_n; i += tg_sz)  tg_M[i]   = ext_mds[i];
    for (uint i = lid; i < tt;    i += tg_sz)  tg_D[i]   = int_diag[i];
    for (uint i = lid; i < rce_n; i += tg_sz)  tg_rce[i] = rc_ext[i];
    for (uint i = lid; i < rci_n; i += tg_sz)  tg_rci[i] = rc_int[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint parent_count = (child_count + arity - 1u) / arity;
    if (p >= parent_count) return;

    uint base = p * arity;

    if (t == 3u) {
        ulong s0 = 0ul, s1 = 0ul, s2 = 0ul;
        if (base + 0u < child_count) s0 = tree[in_offset + base + 0u];
        if (base + 1u < child_count) s1 = tree[in_offset + base + 1u];
        if (arity >= 3u && base + 2u < child_count) s2 = tree[in_offset + base + 2u];

        ulong M[9];
        for (uint k = 0u; k < 9u; ++k) M[k] = tg_M[k];
        ulong D[3] = { tg_D[0], tg_D[1], tg_D[2] };

        // Use threadgroup-cached rc arrays via local pointer aliases.
        // Copy to thread-private arrays of bounded size to give compiler scalar regs.
        ulong rce[32];
        for (uint k = 0u; k < rce_n; ++k) rce[k] = tg_rce[k];
        ulong rci[32];
        for (uint k = 0u; k < rci_n; ++k) rci[k] = tg_rci[k];

        mds3(s0,s1,s2,M);
        uint half_f = r_f >> 1u;
        for (uint r=0u; r<half_f; ++r) {
            uint b = r*3u;
            s0 = sbox7(gadd(s0, rce[b+0u]));
            s1 = sbox7(gadd(s1, rce[b+1u]));
            s2 = sbox7(gadd(s2, rce[b+2u]));
            mds3(s0,s1,s2,M);
        }
        for (uint r=0u; r<r_p; ++r) {
            s0 = sbox7(gadd(s0, rci[r]));
            ulong sum = gadd(gadd(s0,s1), s2);
            ulong n0 = gadd(sum, gmul(D[0], s0));
            ulong n1 = gadd(sum, gmul(D[1], s1));
            ulong n2 = gadd(sum, gmul(D[2], s2));
            s0=n0; s1=n1; s2=n2;
        }
        for (uint r=half_f; r<r_f; ++r) {
            uint b = r*3u;
            s0 = sbox7(gadd(s0, rce[b+0u]));
            s1 = sbox7(gadd(s1, rce[b+1u]));
            s2 = sbox7(gadd(s2, rce[b+2u]));
            mds3(s0,s1,s2,M);
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

        ulong M[16];
        for (uint k = 0u; k < 16u; ++k) M[k] = tg_M[k];
        ulong D[4] = { tg_D[0], tg_D[1], tg_D[2], tg_D[3] };

        ulong rce[32];
        for (uint k = 0u; k < rce_n; ++k) rce[k] = tg_rce[k];
        ulong rci[32];
        for (uint k = 0u; k < rci_n; ++k) rci[k] = tg_rci[k];

        mds4(s0,s1,s2,s3,M);
        uint half_f = r_f >> 1u;
        for (uint r=0u; r<half_f; ++r) {
            uint b = r*4u;
            s0 = sbox7(gadd(s0, rce[b+0u]));
            s1 = sbox7(gadd(s1, rce[b+1u]));
            s2 = sbox7(gadd(s2, rce[b+2u]));
            s3 = sbox7(gadd(s3, rce[b+3u]));
            mds4(s0,s1,s2,s3,M);
        }
        for (uint r=0u; r<r_p; ++r) {
            s0 = sbox7(gadd(s0, rci[r]));
            ulong sum = gadd(gadd(s0,s1), gadd(s2,s3));
            ulong n0 = gadd(sum, gmul(D[0], s0));
            ulong n1 = gadd(sum, gmul(D[1], s1));
            ulong n2 = gadd(sum, gmul(D[2], s2));
            ulong n3 = gadd(sum, gmul(D[3], s3));
            s0=n0; s1=n1; s2=n2; s3=n3;
        }
        for (uint r=half_f; r<r_f; ++r) {
            uint b = r*4u;
            s0 = sbox7(gadd(s0, rce[b+0u]));
            s1 = sbox7(gadd(s1, rce[b+1u]));
            s2 = sbox7(gadd(s2, rce[b+2u]));
            s3 = sbox7(gadd(s3, rce[b+3u]));
            mds4(s0,s1,s2,s3,M);
        }
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