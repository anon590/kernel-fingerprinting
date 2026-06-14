To break out of the local optimum, this approach introduces two STRUCTURALLY different optimizations targeted at Apple Silicon's architecture:

1. **Instruction-Level Parallelism (ILP) Interleaving**: Apple's M-series GPUs feature extremely wide execution engines that thrive on independent operations. By computing the four `sbox7` calls concurrently and transposing the accumulation loops for the external MDS matrix (calculating `hi0..hi3` together), we completely decouple the MAC dependencies. The hardware scheduler can now dispatch independent 64-bit operations simultaneously instead of stalling on sequential register dependencies.
2. **Threadgroup Constant Caching**: Instead of streaming runtime parameters (`rc_ext`, `rc_int`, `ext_mds`) from the uniform cache in the tight permutation loops, all threads cooperatively preload these into `threadgroup` memory at startup, saving register spills and L1 data cache pressure.

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD = 0xFFFFFFFF00000001ul;

inline void mul128(ulong a, ulong b, thread ulong &hi, thread ulong &lo) {
    uint a0 = (uint)a, a1 = (uint)(a >> 32);
    uint b0 = (uint)b, b1 = (uint)(b >> 32);

    ulong p01 = (ulong)a0 * b1;
    ulong p10 = (ulong)a1 * b0;
    ulong p00 = (ulong)a0 * b0;
    ulong p11 = (ulong)a1 * b1;

    ulong mid = (p00 >> 32) + (ulong)(uint)p01 + (ulong)(uint)p10;
    lo = (uint)p00 | (mid << 32);
    hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}

inline void sqr128(ulong a, thread ulong &hi, thread ulong &lo) {
    uint a0 = (uint)a, a1 = (uint)(a >> 32);

    ulong p01 = (ulong)a0 * a1;
    ulong p00 = (ulong)a0 * a0;
    ulong p11 = (ulong)a1 * a1;

    ulong mid = (p00 >> 32) + ((ulong)(uint)p01 << 1);
    lo = (uint)p00 | (mid << 32);
    hi = p11 + ((p01 >> 32) << 1) + (mid >> 32);
}

inline void mac128(ulong a, ulong b, thread ulong &hi, thread ulong &lo) {
    uint a0 = (uint)a, a1 = (uint)(a >> 32);
    uint b0 = (uint)b, b1 = (uint)(b >> 32);

    ulong p01 = (ulong)a0 * b1;
    ulong p10 = (ulong)a1 * b0;
    ulong p00 = (ulong)a0 * b0;
    ulong p11 = (ulong)a1 * b1;

    ulong mid = (p00 >> 32) + (ulong)(uint)p01 + (ulong)(uint)p10;
    ulong l = (uint)p00 | (mid << 32);
    ulong h = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    
    lo += l;
    hi += h + (lo < l ? 1ul : 0ul);
}

inline ulong reduce128(ulong hi, ulong lo) {
    ulong hi_hi = hi >> 32;
    ulong hi_lo = (uint)hi;
    
    ulong sub = hi_hi + hi_lo;
    ulong t0 = lo - sub;
    t0 -= (lo < sub) ? 0xFFFFFFFFul : 0ul;
    
    ulong t1 = hi_lo << 32;
    ulong t2 = t0 + t1;
    t2 += (t2 < t0) ? 0xFFFFFFFFul : 0ul;
    
    return (t2 >= P_GOLD) ? t2 - P_GOLD : t2;
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    t += (t < a) ? 0xFFFFFFFFul : 0ul;
    return (t >= P_GOLD) ? t - P_GOLD : t;
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong hi, lo;
    mul128(a, b, hi, lo);
    return reduce128(hi, lo);
}

inline ulong gold_sqr(ulong a) {
    ulong hi, lo;
    sqr128(a, hi, lo);
    return reduce128(hi, lo);
}

inline ulong sbox7(ulong x) {
    ulong x2 = gold_sqr(x);
    ulong x4 = gold_sqr(x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
}

inline void poseidon2_permute_3(thread ulong &s0, thread ulong &s1, thread ulong &s2,
                                threadgroup const ulong *shared_rc_ext,
                                threadgroup const ulong *shared_rc_int,
                                threadgroup const ulong *shared_ext_mds,
                                threadgroup const ulong *shared_int_diag,
                                uint r_f, uint r_p)
{
    ulong m00 = shared_ext_mds[0], m01 = shared_ext_mds[1], m02 = shared_ext_mds[2];
    ulong m10 = shared_ext_mds[3], m11 = shared_ext_mds[4], m12 = shared_ext_mds[5];
    ulong m20 = shared_ext_mds[6], m21 = shared_ext_mds[7], m22 = shared_ext_mds[8];

    ulong d0 = shared_int_diag[0], d1 = shared_int_diag[1], d2 = shared_int_diag[2];

    {
        ulong hi0 = 0, lo0 = 0;
        ulong hi1 = 0, lo1 = 0;
        ulong hi2 = 0, lo2 = 0;
        
        mac128(m00, s0, hi0, lo0);
        mac128(m10, s0, hi1, lo1);
        mac128(m20, s0, hi2, lo2);
        
        mac128(m01, s1, hi0, lo0);
        mac128(m11, s1, hi1, lo1);
        mac128(m21, s1, hi2, lo2);
        
        mac128(m02, s2, hi0, lo0);
        mac128(m12, s2, hi1, lo1);
        mac128(m22, s2, hi2, lo2);
        
        s0 = reduce128(hi0, lo0);
        s1 = reduce128(hi1, lo1);
        s2 = reduce128(hi2, lo2);
    }

    uint half_f = r_f >> 1;
    uint rc_idx = 0;

    for (uint r = 0; r < half_f; ++r) {
        ulong x0 = gold_add(s0, shared_rc_ext[rc_idx]);
        ulong x1 = gold_add(s1, shared_rc_ext[rc_idx + 1]);
        ulong x2 = gold_add(s2, shared_rc_ext[rc_idx + 2]);
        rc_idx += 3;
        
        ulong x0_2 = gold_sqr(x0);
        ulong x1_2 = gold_sqr(x1);
        ulong x2_2 = gold_sqr(x2);
        
        ulong x0_4 = gold_sqr(x0_2);
        ulong x1_4 = gold_sqr(x1_2);
        ulong x2_4 = gold_sqr(x2_2);
        
        ulong x0_6 = gold_mul(x0_4, x0_2);
        ulong x1_6 = gold_mul(x1_4, x1_2);
        ulong x2_6 = gold_mul(x2_4, x2_2);
        
        s0 = gold_mul(x0_6, x0);
        s1 = gold_mul(x1_6, x1);
        s2 = gold_mul(x2_6, x2);
        
        ulong hi0 = 0, lo0 = 0;
        ulong hi1 = 0, lo1 = 0;
        ulong hi2 = 0, lo2 = 0;
        
        mac128(m00, s0, hi0, lo0);
        mac128(m10, s0, hi1, lo1);
        mac128(m20, s0, hi2, lo2);
        
        mac128(m01, s1, hi0, lo0);
        mac128(m11, s1, hi1, lo1);
        mac128(m21, s1, hi2, lo2);
        
        mac128(m02, s2, hi0, lo0);
        mac128(m12, s2, hi1, lo1);
        mac128(m22, s2, hi2, lo2);
        
        s0 = reduce128(hi0, lo0);
        s1 = reduce128(hi1, lo1);
        s2 = reduce128(hi2, lo2);
    }

    for (uint r = 0; r < r_p; ++r) {
        s0 = sbox7(gold_add(s0, shared_rc_int[r]));
        
        ulong sum = gold_add(gold_add(s0, s1), s2);
        
        ulong m0 = gold_mul(d0, s0);
        ulong m1 = gold_mul(d1, s1);
        ulong m2 = gold_mul(d2, s2);
        
        s0 = gold_add(sum, m0);
        s1 = gold_add(sum, m1);
        s2 = gold_add(sum, m2);
    }

    for (uint r = half_f; r < r_f; ++r) {
        ulong x0 = gold_add(s0, shared_rc_ext[rc_idx]);
        ulong x1 = gold_add(s1, shared_rc_ext[rc_idx + 1]);
        ulong x2 = gold_add(s2, shared_rc_ext[rc_idx + 2]);
        rc_idx += 3;
        
        ulong x0_2 = gold_sqr(x0);
        ulong x1_2 = gold_sqr(x1);
        ulong x2_2 = gold_sqr(x2);
        
        ulong x0_4 = gold_sqr(x0_2);
        ulong x1_4 = gold_sqr(x1_2);
        ulong x2_4 = gold_sqr(x2_2);
        
        ulong x0_6 = gold_mul(x0_4, x0_2);
        ulong x1_6 = gold_mul(x1_4, x1_2);
        ulong x2_6 = gold_mul(x2_4, x2_2);
        
        s0 = gold_mul(x0_6, x0);
        s1 = gold_mul(x1_6, x1);
        s2 = gold_mul(x2_6, x2);
        
        ulong hi0 = 0, lo0 = 0;
        ulong hi1 = 0, lo1 = 0;
        ulong hi2 = 0, lo2 = 0;
        
        mac128(m00, s0, hi0, lo0);
        mac128(m10, s0, hi1, lo1);
        mac128(m20, s0, hi2, lo2);
        
        mac128(m01, s1, hi0, lo0);
        mac128(m11, s1, hi1, lo1);
        mac128(m21, s1, hi2, lo2);
        
        mac128(m02, s2, hi0, lo0);
        mac128(m12, s2, hi1, lo1);
        mac128(m22, s2, hi2, lo2);
        
        s0 = reduce128(hi0, lo0);
        s1 = reduce128(hi1, lo1);
        s2 = reduce128(hi2, lo2);
    }
}

inline void poseidon2_permute_4(thread ulong &s0, thread ulong &s1, thread ulong &s2, thread ulong &s3,
                                threadgroup const ulong *shared_rc_ext,
                                threadgroup const ulong *shared_rc_int,
                                threadgroup const ulong *shared_ext_mds,
                                threadgroup const ulong *shared_int_diag,
                                uint r_f, uint r_p)
{
    ulong m00 = shared_ext_mds[0], m01 = shared_ext_mds[1], m02 = shared_ext_mds[2], m03 = shared_ext_mds[3];
    ulong m10 = shared_ext_mds[4], m11 = shared_ext_mds[5], m12 = shared_ext_mds[6], m13 = shared_ext_mds[7];
    ulong m20 = shared_ext_mds[8], m21 = shared_ext_mds[9], m22 = shared_ext_mds[10], m23 = shared_ext_mds[11];
    ulong m30 = shared_ext_mds[12], m31 = shared_ext_mds[13], m32 = shared_ext_mds[14], m33 = shared_ext_mds[15];

    ulong d0 = shared_int_diag[0], d1 = shared_int_diag[1], d2 = shared_int_diag[2], d3 = shared_int_diag[3];

    {
        ulong hi0 = 0, lo0 = 0;
        ulong hi1 = 0, lo1 = 0;
        ulong hi2 = 0, lo2 = 0;
        ulong hi3 = 0, lo3 = 0;
        
        mac128(m00, s0, hi0, lo0);
        mac128(m10, s0, hi1, lo1);
        mac128(m20, s0, hi2, lo2);
        mac128(m30, s0, hi3, lo3);
        
        mac128(m01, s1, hi0, lo0);
        mac128(m11, s1, hi1, lo1);
        mac128(m21, s1, hi2, lo2);
        mac128(m31, s1, hi3, lo3);
        
        mac128(m02, s2, hi0, lo0);
        mac128(m12, s2, hi1, lo1);
        mac128(m22, s2, hi2, lo2);
        mac128(m32, s2, hi3, lo3);
        
        mac128(m03, s3, hi0, lo0);
        mac128(m13, s3, hi1, lo1);
        mac128(m23, s3, hi2, lo2);
        mac128(m33, s3, hi3, lo3);
        
        s0 = reduce128(hi0, lo0);
        s1 = reduce128(hi1, lo1);
        s2 = reduce128(hi2, lo2);
        s3 = reduce128(hi3, lo3);
    }

    uint half_f = r_f >> 1;
    uint rc_idx = 0;

    for (uint r = 0; r < half_f; ++r) {
        ulong x0 = gold_add(s0, shared_rc_ext[rc_idx]);
        ulong x1 = gold_add(s1, shared_rc_ext[rc_idx + 1]);
        ulong x2 = gold_add(s2, shared_rc_ext[rc_idx + 2]);
        ulong x3 = gold_add(s3, shared_rc_ext[rc_idx + 3]);
        rc_idx += 4;
        
        ulong x0_2 = gold_sqr(x0);
        ulong x1_2 = gold_sqr(x1);
        ulong x2_2 = gold_sqr(x2);
        ulong x3_2 = gold_sqr(x3);
        
        ulong x0_4 = gold_sqr(x0_2);
        ulong x1_4 = gold_sqr(x1_2);
        ulong x2_4 = gold_sqr(x2_2);
        ulong x3_4 = gold_sqr(x3_2);
        
        ulong x0_6 = gold_mul(x0_4, x0_2);
        ulong x1_6 = gold_mul(x1_4, x1_2);
        ulong x2_6 = gold_mul(x2_4, x2_2);
        ulong x3_6 = gold_mul(x3_4, x3_2);
        
        s0 = gold_mul(x0_6, x0);
        s1 = gold_mul(x1_6, x1);
        s2 = gold_mul(x2_6, x2);
        s3 = gold_mul(x3_6, x3);
        
        ulong hi0 = 0, lo0 = 0;
        ulong hi1 = 0, lo1 = 0;
        ulong hi2 = 0, lo2 = 0;
        ulong hi3 = 0, lo3 = 0;
        
        mac128(m00, s0, hi0, lo0);
        mac128(m10, s0, hi1, lo1);
        mac128(m20, s0, hi2, lo2);
        mac128(m30, s0, hi3, lo3);
        
        mac128(m01, s1, hi0, lo0);
        mac128(m11, s1, hi1, lo1);
        mac128(m21, s1, hi2, lo2);
        mac128(m31, s1, hi3, lo3);
        
        mac128(m02, s2, hi0, lo0);
        mac128(m12, s2, hi1, lo1);
        mac128(m22, s2, hi2, lo2);
        mac128(m32, s2, hi3, lo3);
        
        mac128(m03, s3, hi0, lo0);
        mac128(m13, s3, hi1, lo1);
        mac128(m23, s3, hi2, lo2);
        mac128(m33, s3, hi3, lo3);
        
        s0 = reduce128(hi0, lo0);
        s1 = reduce128(hi1, lo1);
        s2 = reduce128(hi2, lo2);
        s3 = reduce128(hi3, lo3);
    }

    for (uint r = 0; r < r_p; ++r) {
        s0 = sbox7(gold_add(s0, shared_rc_int[r]));
        
        ulong sum = gold_add(gold_add(s0, s1), gold_add(s2, s3));
        
        ulong m0 = gold_mul(d0, s0);
        ulong m1 = gold_mul(d1, s1);
        ulong m2 = gold_mul(d2, s2);
        ulong m3 = gold_mul(d3, s3);
        
        s0 = gold_add(sum, m0);
        s1 = gold_add(sum, m1);
        s2 = gold_add(sum, m2);
        s3 = gold_add(sum, m3);
    }

    for (uint r = half_f; r < r_f; ++r) {
        ulong x0 = gold_add(s0, shared_rc_ext[rc_idx]);
        ulong x1 = gold_add(s1, shared_rc_ext[rc_idx + 1]);
        ulong x2 = gold_add(s2, shared_rc_ext[rc_idx + 2]);
        ulong x3 = gold_add(s3, shared_rc_ext[rc_idx + 3]);
        rc_idx += 4;
        
        ulong x0_2 = gold_sqr(x0);
        ulong x1_2 = gold_sqr(x1);
        ulong x2_2 = gold_sqr(x2);
        ulong x3_2 = gold_sqr(x3);
        
        ulong x0_4 = gold_sqr(x0_2);
        ulong x1_4 = gold_sqr(x1_2);
        ulong x2_4 = gold_sqr(x2_2);
        ulong x3_4 = gold_sqr(x3_2);
        
        ulong x0_6 = gold_mul(x0_4, x0_2);
        ulong x1_6 = gold_mul(x1_4, x1_2);
        ulong x2_6 = gold_mul(x2_4, x2_2);
        ulong x3_6 = gold_mul(x3_4, x3_2);
        
        s0 = gold_mul(x0_6, x0);
        s1 = gold_mul(x1_6, x1);
        s2 = gold_mul(x2_6, x2);
        s3 = gold_mul(x3_6, x3);
        
        ulong hi0 = 0, lo0 = 0;
        ulong hi1 = 0, lo1 = 0;
        ulong hi2 = 0, lo2 = 0;
        ulong hi3 = 0, lo3 = 0;
        
        mac128(m00, s0, hi0, lo0);
        mac128(m10, s0, hi1, lo1);
        mac128(m20, s0, hi2, lo2);
        mac128(m30, s0, hi3, lo3);
        
        mac128(m01, s1, hi0, lo0);
        mac128(m11, s1, hi1, lo1);
        mac128(m21, s1, hi2, lo2);
        mac128(m31, s1, hi3, lo3);
        
        mac128(m02, s2, hi0, lo0);
        mac128(m12, s2, hi1, lo1);
        mac128(m22, s2, hi2, lo2);
        mac128(m32, s2, hi3, lo3);
        
        mac128(m03, s3, hi0, lo0);
        mac128(m13, s3, hi1, lo1);
        mac128(m23, s3, hi2, lo2);
        mac128(m33, s3, hi3, lo3);
        
        s0 = reduce128(hi0, lo0);
        s1 = reduce128(hi1, lo1);
        s2 = reduce128(hi2, lo2);
        s3 = reduce128(hi3, lo3);
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
    uint p                           [[thread_position_in_grid]],
    uint tid                         [[thread_position_in_threadgroup]],
    uint block_dim                   [[threads_per_threadgroup]])
{
    uint parent_count = (child_count + arity - 1u) / arity;
    
    threadgroup ulong shared_rc_ext[32];
    threadgroup ulong shared_rc_int[32];
    threadgroup ulong shared_ext_mds[16];
    threadgroup ulong shared_int_diag[4];
    
    for (uint i = tid; i < r_f * t; i += block_dim) shared_rc_ext[i] = rc_ext[i];
    for (uint i = tid; i < r_p;     i += block_dim) shared_rc_int[i] = rc_int[i];
    for (uint i = tid; i < t * t;   i += block_dim) shared_ext_mds[i] = ext_mds[i];
    for (uint i = tid; i < t;       i += block_dim) shared_int_diag[i] = int_diag[i];
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (p >= parent_count) return;

    if (t <= 3) {
        ulong s0 = 0, s1 = 0, s2 = 0;
        uint base = p * arity;
        
        if (arity > 0 && base + 0 < child_count) s0 = tree[in_offset + base + 0];
        if (arity > 1 && base + 1 < child_count) s1 = tree[in_offset + base + 1];
        if (arity > 2 && base + 2 < child_count) s2 = tree[in_offset + base + 2];
        
        poseidon2_permute_3(s0, s1, s2, shared_rc_ext, shared_rc_int, shared_ext_mds, shared_int_diag, r_f, r_p);
        tree[out_offset + p] = s0;
    } else {
        ulong s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        uint base = p * arity;
        
        if (arity > 0 && base + 0 < child_count) s0 = tree[in_offset + base + 0];
        if (arity > 1 && base + 1 < child_count) s1 = tree[in_offset + base + 1];
        if (arity > 2 && base + 2 < child_count) s2 = tree[in_offset + base + 2];
        if (arity > 3 && base + 3 < child_count) s3 = tree[in_offset + base + 3];
        
        poseidon2_permute_4(s0, s1, s2, s3, shared_rc_ext, shared_rc_int, shared_ext_mds, shared_int_diag, r_f, r_p);
        tree[out_offset + p] = s0;
    }
}
```