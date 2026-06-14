#include <metal_stdlib>
using namespace metal;

struct BarrettCtx {
    uint q;
    uint y_lo;
    uint y_hi;
};

// Fast 64-bit to 32-bit Barrett reduction using 32-bit multiplications.
// Computes (a * b) % q exactly, given that (a * b) < q^2 and q <= 8380417.
inline uint mod_mul_fast(uint a, uint b, thread const BarrettCtx& ctx) {
    ulong X = (ulong)a * b;
    uint x_lo = (uint)X;
    uint x_hi = (uint)(X >> 32);
    
    uint p00_hi = mulhi(x_lo, ctx.y_lo);
    ulong p01 = (ulong)x_lo * ctx.y_hi;
    ulong p10 = (ulong)x_hi * ctx.y_lo;
    ulong p11 = (ulong)x_hi * ctx.y_hi;
    
    ulong mid = p01 + p00_hi;
    ulong mid2 = p10 + (uint)mid;
    
    // Top 64 bits of (X * M)
    uint k = (uint)(p11 + (mid >> 32) + (mid2 >> 32));
    
    // Remainder
    uint r = x_lo - k * ctx.q;
    return (r >= ctx.q) ? (r - ctx.q) : r;
}

inline uint mod_add(uint a, uint b, uint q) {
    uint t = a + b;
    return (t >= q) ? (t - q) : t;
}

inline uint mod_sub(uint a, uint b, uint q) {
    return (a >= b) ? (a - b) : (a + q - b);
}

kernel void kyber_ntt(
    device       uint *coeffs     [[buffer(0)]],
    device const uint *zetas      [[buffer(1)]],
    constant uint     &q           [[buffer(2)]],
    constant uint     &n           [[buffer(3)]],
    constant uint     &n_levels    [[buffer(4)]],
    constant uint     &batch       [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ltid [[thread_position_in_threadgroup]])
{
    if (tgid >= batch) return;

    threadgroup uint a[256];
    threadgroup BarrettCtx tg_ctx;

    uint half_n = n >> 1u;
    device uint *poly = coeffs + (size_t)tgid * n;

    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];
    
    // Thread 0 computes the Barrett magic constants to save 64-bit ALU divisions
    if (ltid == 0) {
        ulong M = 0xFFFFFFFFFFFFFFFFull / q;
        tg_ctx.q = q;
        tg_ctx.y_lo = (uint)M;
        tg_ctx.y_hi = (uint)(M >> 32);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Load into fast register context
    BarrettCtx ctx = tg_ctx;

    uint length = half_n;
    uint shift = 31u - clz(half_n);
    uint k_start = 1u;
    
    for (uint level = 0u; level < n_levels; ++level) {
        // Bitwise operations replace integer division/modulo
        uint group_idx  = ltid >> shift;
        uint j_in_group = ltid & (length - 1u);
        uint j          = (group_idx << (shift + 1u)) | j_in_group;
        uint zeta       = zetas[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];
        uint t = mod_mul_fast(zeta, y, ctx);

        a[j]          = mod_add(x, t, ctx.q);
        a[j + length] = mod_sub(x, t, ctx.q);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length  >>= 1u;
        shift   -= 1u;
    }

    // Writeback
    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}