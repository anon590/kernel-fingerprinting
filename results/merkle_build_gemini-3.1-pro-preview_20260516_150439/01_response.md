To optimize the Merkle tree build kernel, we apply two major transformations:
1. **Loop Unrolling and State Scalarization**: Since Poseidon2 is invoked with either $t=3$ or $t=4$ in this context, we branch on $t$ to completely specialize the permutations. This fully unrolls all state arrays into scalar variables (`s0` through `s3`), allowing the compiler to keep the entire Goldilocks state in high-speed registers, avoiding any thread-local arrays and significantly boosting throughput. 
2. **Optimized Arithmetic**: The baseline `gold_add` is streamlined using branchless carries. For `gold_mul`, the 128-bit Montgomery reduction is manually tuned to replace a costly multiplication with a simple shift and subtract (`hi_lo * 0xFFFFFFFFul` becomes `(hi_lo << 32) - hi_lo`), and component extractions are simplified through direct casts (`uint`). 

These combined structural optimizations drastically increase multiplication throughput, keeping all arithmetic strictly within execution units.

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