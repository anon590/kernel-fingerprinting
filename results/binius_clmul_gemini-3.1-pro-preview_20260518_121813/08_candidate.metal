#include <metal_stdlib>
using namespace metal;

// ----------------------------------------------------------------------
// 16x16-bit carry-less multiply into 32-bit result
// Explicit balanced XOR tree reduces critical path latency from 16 to 4,
// vastly improving ILP for the deeply inlined Karatsuba operations.
// ----------------------------------------------------------------------
inline __attribute__((always_inline))
uint clmul16(uint a, uint b) {
    uint r0 = select(0u, a,       bool(b & (1u << 0)));
    uint r1 = select(0u, a << 1,  bool(b & (1u << 1)));
    uint r2 = select(0u, a << 2,  bool(b & (1u << 2)));
    uint r3 = select(0u, a << 3,  bool(b & (1u << 3)));
    
    uint r4 = select(0u, a << 4,  bool(b & (1u << 4)));
    uint r5 = select(0u, a << 5,  bool(b & (1u << 5)));
    uint r6 = select(0u, a << 6,  bool(b & (1u << 6)));
    uint r7 = select(0u, a << 7,  bool(b & (1u << 7)));
    
    uint r8 = select(0u, a << 8,  bool(b & (1u << 8)));
    uint r9 = select(0u, a << 9,  bool(b & (1u << 9)));
    uint rA = select(0u, a << 10, bool(b & (1u << 10)));
    uint rB = select(0u, a << 11, bool(b & (1u << 11)));
    
    uint rC = select(0u, a << 12, bool(b & (1u << 12)));
    uint rD = select(0u, a << 13, bool(b & (1u << 13)));
    uint rE = select(0u, a << 14, bool(b & (1u << 14)));
    uint rF = select(0u, a << 15, bool(b & (1u << 15)));
    
    uint s0 = (r0 ^ r1) ^ (r2 ^ r3);
    uint s1 = (r4 ^ r5) ^ (r6 ^ r7);
    uint s2 = (r8 ^ r9) ^ (rA ^ rB);
    uint s3 = (rC ^ rD) ^ (rE ^ rF);
    
    return (s0 ^ s1) ^ (s2 ^ s3);
}

// ----------------------------------------------------------------------
// 32x32-bit carry-less multiply
// ----------------------------------------------------------------------
inline __attribute__((always_inline))
uint2 clmul32(uint a, uint b) {
    uint a_lo = a & 0xFFFFu;
    uint a_hi = a >> 16;
    uint b_lo = b & 0xFFFFu;
    uint b_hi = b >> 16;
    
    uint L = clmul16(a_lo, b_lo);
    uint H = clmul16(a_hi, b_hi);
    uint M = clmul16(a_lo ^ a_hi, b_lo ^ b_hi) ^ (L ^ H);
    
    uint res_lo = L ^ (M << 16);
    uint res_hi = H ^ (M >> 16);
    return uint2(res_lo, res_hi);
}

// ----------------------------------------------------------------------
// 64x64-bit carry-less multiply
// ----------------------------------------------------------------------
inline __attribute__((always_inline))
void clmul64(uint2 a, uint2 b, thread uint2 &r_lo, thread uint2 &r_hi) {
    uint2 L = clmul32(a.x, b.x);
    uint2 H = clmul32(a.y, b.y);
    uint2 M = clmul32(a.x ^ a.y, b.x ^ b.y);
    
    M ^= (L ^ H);
    
    r_lo = uint2(L.x, L.y ^ M.x);
    r_hi = uint2(H.x ^ M.y, H.y);
}

// ----------------------------------------------------------------------
// 128x128-bit unreduced multiply
// ----------------------------------------------------------------------
inline __attribute__((always_inline))
void clmul128_unreduced(
    uint2 a_lo, uint2 a_hi, uint2 b_lo, uint2 b_hi,
    thread uint2 &t0, thread uint2 &t1,
    thread uint2 &t2, thread uint2 &t3)
{
    uint2 L_lo, L_hi;
    clmul64(a_lo, b_lo, L_lo, L_hi);
    
    uint2 H_lo, H_hi;
    clmul64(a_hi, b_hi, H_lo, H_hi);
    
    uint2 M_lo, M_hi;
    clmul64(a_lo ^ a_hi, b_lo ^ b_hi, M_lo, M_hi);
    
    uint2 LH_lo = L_lo ^ H_lo;
    uint2 LH_hi = L_hi ^ H_hi;
    
    M_lo ^= LH_lo;
    M_hi ^= LH_hi;
    
    t0 = L_lo;
    t1 = L_hi ^ M_lo;
    t2 = H_lo ^ M_hi;
    t3 = H_hi;
}

// ----------------------------------------------------------------------
// Two-stage GCM-style reduction modulo R(x) = x^128 + x^7 + x^2 + x + 1
// ----------------------------------------------------------------------
inline __attribute__((always_inline))
ulong2 gcm_reduce(ulong t0, ulong t1, ulong t2, ulong t3)
{
    ulong d_lo0 = t2 ^ (t2 << 1u) ^ (t2 << 2u) ^ (t2 << 7u);
    ulong d_lo1 = t3
                ^ ((t3 << 1u) | (t2 >> 63u))
                ^ ((t3 << 2u) | (t2 >> 62u))
                ^ ((t3 << 7u) | (t2 >> 57u));
    ulong d_hi  = (t3 >> 63u) ^ (t3 >> 62u) ^ (t3 >> 57u);

    t0 ^= d_lo0;
    t1 ^= d_lo1;
    t0 ^= d_hi ^ (d_hi << 1u) ^ (d_hi << 2u) ^ (d_hi << 7u);

    return ulong2(t0, t1);
}

// ----------------------------------------------------------------------
// Kernel
// ----------------------------------------------------------------------
kernel void binius_clmul(
    device const ulong *a         [[buffer(0)]],
    device const ulong *b         [[buffer(1)]],
    device       ulong *c         [[buffer(2)]],
    constant ulong     &alpha_lo  [[buffer(3)]],
    constant ulong     &alpha_hi  [[buffer(4)]],
    constant uint      &tower     [[buffer(5)]],
    constant uint      &batch     [[buffer(6)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= batch) return;

    if (tower == 0u) {
        device const ulong2 *a2 = (device const ulong2 *)a;
        device const ulong2 *b2 = (device const ulong2 *)b;
        device       ulong2 *c2 = (device       ulong2 *)c;
        
        ulong2 av = a2[idx];
        ulong2 bv = b2[idx];

        uint2 a_lo = as_type<uint2>(av.x), a_hi = as_type<uint2>(av.y);
        uint2 b_lo = as_type<uint2>(bv.x), b_hi = as_type<uint2>(bv.y);

        uint2 t0, t1, t2, t3;
        clmul128_unreduced(a_lo, a_hi, b_lo, b_hi, t0, t1, t2, t3);

        ulong2 c_res = gcm_reduce(as_type<ulong>(t0), as_type<ulong>(t1), 
                                  as_type<ulong>(t2), as_type<ulong>(t3));
        c2[idx] = c_res;
    } else {
        device const ulong4 *a4 = (device const ulong4 *)a;
        device const ulong4 *b4 = (device const ulong4 *)b;
        device       ulong4 *c4 = (device       ulong4 *)c;
        
        ulong4 av = a4[idx];
        ulong4 bv = b4[idx];

        uint2 a0_lo = as_type<uint2>(av.x), a0_hi = as_type<uint2>(av.y);
        uint2 a1_lo = as_type<uint2>(av.z), a1_hi = as_type<uint2>(av.w);
        uint2 b0_lo = as_type<uint2>(bv.x), b0_hi = as_type<uint2>(bv.y);
        uint2 b1_lo = as_type<uint2>(bv.z), b1_hi = as_type<uint2>(bv.w);

        uint2 t0_00, t1_00, t2_00, t3_00;
        clmul128_unreduced(a0_lo, a0_hi, b0_lo, b0_hi, t0_00, t1_00, t2_00, t3_00);

        uint2 t0_11, t1_11, t2_11, t3_11;
        clmul128_unreduced(a1_lo, a1_hi, b1_lo, b1_hi, t0_11, t1_11, t2_11, t3_11);

        uint2 t0_sum, t1_sum, t2_sum, t3_sum;
        clmul128_unreduced(
            a0_lo ^ a1_lo, a0_hi ^ a1_hi,
            b0_lo ^ b1_lo, b0_hi ^ b1_hi,
            t0_sum, t1_sum, t2_sum, t3_sum);

        // Reduce m11 before multiplying by alpha
        ulong2 m11 = gcm_reduce(as_type<ulong>(t0_11), as_type<ulong>(t1_11), 
                                as_type<ulong>(t2_11), as_type<ulong>(t3_11));

        uint2 t0_am, t1_am, t2_am, t3_am;
        // alpha_lo / alpha_hi placed as operand 'b' so unrolled bit-tests compile to grid-uniform uniform branches
        clmul128_unreduced(
            as_type<uint2>(m11.x),    as_type<uint2>(m11.y), 
            as_type<uint2>(alpha_lo), as_type<uint2>(alpha_hi), 
            t0_am, t1_am, t2_am, t3_am);

        ulong2 c0_res = gcm_reduce(
            as_type<ulong>(t0_00 ^ t0_am),
            as_type<ulong>(t1_00 ^ t1_am),
            as_type<ulong>(t2_00 ^ t2_am),
            as_type<ulong>(t3_00 ^ t3_am));

        ulong2 c1_res = gcm_reduce(
            as_type<ulong>(t0_sum ^ t0_00),
            as_type<ulong>(t1_sum ^ t1_00),
            as_type<ulong>(t2_sum ^ t2_00),
            as_type<ulong>(t3_sum ^ t3_00));

        c4[idx] = ulong4(c0_res.x, c0_res.y, c1_res.x, c1_res.y);
    }
}