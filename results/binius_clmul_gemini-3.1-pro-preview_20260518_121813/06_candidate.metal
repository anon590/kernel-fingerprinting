#include <metal_stdlib>
using namespace metal;

// ----------------------------------------------------------------------
// Space 8 bits into 32 bits (1 bit every 4 bits)
// ----------------------------------------------------------------------
inline __attribute__((always_inline))
uint4 expand32(uint x) {
    uint4 y = uint4(x, x >> 8u, x >> 16u, x >> 24u) & 0xFFu;
    y = (y | (y << 12u)) & 0x000F000Fu;
    y = (y | (y <<  6u)) & 0x03030303u;
    y = (y | (y <<  3u)) & 0x11111111u;
    return y;
}

struct exp64_t { uint4 lo, hi; };
struct exp128_t { exp64_t lo, hi; };

inline __attribute__((always_inline))
exp64_t expand64(uint2 x) { return { expand32(x.x), expand32(x.y) }; }

inline __attribute__((always_inline))
exp128_t expand128(uint4 x) { return { expand64(x.xy), expand64(x.zw) }; }

// ----------------------------------------------------------------------
// Core multiplication primitives using uncompressed expanded arrays
// ----------------------------------------------------------------------
struct uncomp32_t { uint4 lo, hi; };
struct uncomp64_t { uint4 w0, w1, w2, w3; };
struct uncomp128_t { uint4 w0, w1, w2, w3, w4, w5, w6, w7; };

inline __attribute__((always_inline))
uncomp32_t mul32_uncompressed(uint4 A, uint4 B) {
    uint2 U0 = uint2(A.x * B.x, mulhi(A.x, B.x));
    uint2 U1 = uint2(A.x * B.y, mulhi(A.x, B.y)) ^ uint2(A.y * B.x, mulhi(A.y, B.x));
    uint2 U2 = uint2(A.x * B.z, mulhi(A.x, B.z)) ^ uint2(A.y * B.y, mulhi(A.y, B.y)) ^ uint2(A.z * B.x, mulhi(A.z, B.x));
    uint2 U3 = uint2(A.x * B.w, mulhi(A.x, B.w)) ^ uint2(A.y * B.z, mulhi(A.y, B.z)) ^ uint2(A.z * B.y, mulhi(A.z, B.y)) ^ uint2(A.w * B.x, mulhi(A.w, B.x));
    uint2 U4 = uint2(A.y * B.w, mulhi(A.y, B.w)) ^ uint2(A.z * B.z, mulhi(A.z, B.z)) ^ uint2(A.w * B.y, mulhi(A.w, B.y));
    uint2 U5 = uint2(A.z * B.w, mulhi(A.z, B.w)) ^ uint2(A.w * B.z, mulhi(A.w, B.z));
    uint2 U6 = uint2(A.w * B.w, mulhi(A.w, B.w));
    
    uncomp32_t res;
    res.lo = uint4(U0.x, U0.y ^ U1.x, U1.y ^ U2.x, U2.y ^ U3.x);
    res.hi = uint4(U3.y ^ U4.x, U4.y ^ U5.x, U5.y ^ U6.x, U6.y);
    return res;
}

inline __attribute__((always_inline))
uncomp64_t mul64_uncompressed(exp64_t A, exp64_t B) {
    uncomp32_t L = mul32_uncompressed(A.lo, B.lo);
    uncomp32_t H = mul32_uncompressed(A.hi, B.hi);
    uncomp32_t M = mul32_uncompressed(A.lo ^ A.hi, B.lo ^ B.hi);
    
    M.lo ^= L.lo ^ H.lo;
    M.hi ^= L.hi ^ H.hi;
    
    uncomp64_t res;
    res.w0 = L.lo;
    res.w1 = L.hi ^ M.lo;
    res.w2 = H.lo ^ M.hi;
    res.w3 = H.hi;
    return res;
}

inline __attribute__((always_inline))
uncomp128_t mul128_uncompressed(exp128_t A, exp128_t B) {
    uncomp64_t L = mul64_uncompressed(A.lo, B.lo);
    uncomp64_t H = mul64_uncompressed(A.hi, B.hi);
    
    exp64_t A_M = { A.lo.lo ^ A.hi.lo, A.lo.hi ^ A.hi.hi };
    exp64_t B_M = { B.lo.lo ^ B.hi.lo, B.lo.hi ^ B.hi.hi };
    uncomp64_t M = mul64_uncompressed(A_M, B_M);
    
    M.w0 ^= L.w0 ^ H.w0;
    M.w1 ^= L.w1 ^ H.w1;
    M.w2 ^= L.w2 ^ H.w2;
    M.w3 ^= L.w3 ^ H.w3;
    
    uncomp128_t res;
    res.w0 = L.w0;
    res.w1 = L.w1;
    res.w2 = L.w2 ^ M.w0;
    res.w3 = L.w3 ^ M.w1;
    res.w4 = H.w0 ^ M.w2;
    res.w5 = H.w1 ^ M.w3;
    res.w6 = H.w2;
    res.w7 = H.w3;
    return res;
}

// ----------------------------------------------------------------------
// Compress uncompressed bitwise sums back into bytes
// ----------------------------------------------------------------------
inline __attribute__((always_inline))
uint4 compress_uint4(uint4 W) {
    W &= 0x11111111u;
    W = (W | (W >> 3u)) & 0x03030303u;
    W = (W | (W >> 6u)) & 0x000F000Fu;
    W = (W | (W >> 12u)) & 0x000000FFu;
    return W;
}

inline __attribute__((always_inline))
uint2 compress_words(uint4 W_lo, uint4 W_hi) {
    uint4 c_lo = compress_uint4(W_lo);
    uint4 c_hi = compress_uint4(W_hi);
    
    uint res_x = c_lo.x | (c_lo.y << 8u) | (c_lo.z << 16u) | (c_lo.w << 24u);
    uint res_y = c_hi.x | (c_hi.y << 8u) | (c_hi.z << 16u) | (c_hi.w << 24u);
    return uint2(res_x, res_y);
}

inline __attribute__((always_inline))
void clmul128_unreduced(
    uint4 a, uint4 b,
    thread uint2 &t0, thread uint2 &t1,
    thread uint2 &t2, thread uint2 &t3)
{
    exp128_t A = expand128(a);
    exp128_t B = expand128(b);
    
    uncomp128_t U = mul128_uncompressed(A, B);
    
    t0 = compress_words(U.w0, U.w1);
    t1 = compress_words(U.w2, U.w3);
    t2 = compress_words(U.w4, U.w5);
    t3 = compress_words(U.w6, U.w7);
}

// ----------------------------------------------------------------------
// Two-stage GCM-style reduction modulo R(x) = x^128 + x^7 + x^2 + x + 1
// ----------------------------------------------------------------------
inline __attribute__((always_inline))
void gcm_reduce(
    ulong t0, ulong t1, ulong t2, ulong t3,
    thread ulong &r_lo, thread ulong &r_hi)
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

    r_lo = t0;
    r_hi = t1;
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

        uint2 t0, t1, t2, t3;
        clmul128_unreduced(as_type<uint4>(av), as_type<uint4>(bv), t0, t1, t2, t3);

        ulong c_lo, c_hi;
        gcm_reduce(as_type<ulong>(t0), as_type<ulong>(t1), 
                   as_type<ulong>(t2), as_type<ulong>(t3), 
                   c_lo, c_hi);

        c2[idx] = ulong2(c_lo, c_hi);
    } else {
        device const ulong4 *a4 = (device const ulong4 *)a;
        device const ulong4 *b4 = (device const ulong4 *)b;
        device       ulong4 *c4 = (device       ulong4 *)c;
        
        ulong4 av = a4[idx];
        ulong4 bv = b4[idx];

        uint4 a0 = as_type<uint4>(av.xy);
        uint4 a1 = as_type<uint4>(av.zw);
        uint4 b0 = as_type<uint4>(bv.xy);
        uint4 b1 = as_type<uint4>(bv.zw);

        uint2 t0_00, t1_00, t2_00, t3_00;
        clmul128_unreduced(a0, b0, t0_00, t1_00, t2_00, t3_00);

        uint2 t0_11, t1_11, t2_11, t3_11;
        clmul128_unreduced(a1, b1, t0_11, t1_11, t2_11, t3_11);

        uint2 t0_sum, t1_sum, t2_sum, t3_sum;
        clmul128_unreduced(a0 ^ a1, b0 ^ b1, t0_sum, t1_sum, t2_sum, t3_sum);

        ulong m11_lo, m11_hi;
        gcm_reduce(as_type<ulong>(t0_11), as_type<ulong>(t1_11), 
                   as_type<ulong>(t2_11), as_type<ulong>(t3_11), 
                   m11_lo, m11_hi);

        uint4 alpha_val = uint4(as_type<uint2>(alpha_lo), as_type<uint2>(alpha_hi));
        uint4 m11_val = uint4(as_type<uint2>(m11_lo), as_type<uint2>(m11_hi));

        uint2 t0_am, t1_am, t2_am, t3_am;
        clmul128_unreduced(alpha_val, m11_val, t0_am, t1_am, t2_am, t3_am);

        ulong c0_lo, c0_hi;
        gcm_reduce(
            as_type<ulong>(t0_00 ^ t0_am),
            as_type<ulong>(t1_00 ^ t1_am),
            as_type<ulong>(t2_00 ^ t2_am),
            as_type<ulong>(t3_00 ^ t3_am),
            c0_lo, c0_hi);

        ulong c1_lo, c1_hi;
        gcm_reduce(
            as_type<ulong>(t0_sum ^ t0_00),
            as_type<ulong>(t1_sum ^ t1_00),
            as_type<ulong>(t2_sum ^ t2_00),
            as_type<ulong>(t3_sum ^ t3_00),
            c1_lo, c1_hi);

        c4[idx] = ulong4(c0_lo, c0_hi, c1_lo, c1_hi);
    }
}