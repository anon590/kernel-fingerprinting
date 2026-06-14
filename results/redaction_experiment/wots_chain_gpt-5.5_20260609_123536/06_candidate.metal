#include <metal_stdlib>
using namespace metal;

inline uint2 split_u64(ulong x) {
    return as_type<uint2>(x);
}

inline ulong join_u64(uint2 x) {
    return as_type<ulong>(x);
}

#define ROL2_LO(v, N) uint2((((v).x << (N)) | ((v).y >> (32u - (N)))), (((v).y << (N)) | ((v).x >> (32u - (N)))))
#define ROL2_HI(v, N) uint2((((v).y << ((N) - 32u)) | ((v).x >> (64u - (N)))), (((v).x << ((N) - 32u)) | ((v).y >> (64u - (N)))))

#define KECCAK_CHI_IOTA(RCLO, RCHI) { \
    uint2 t0; uint2 t1; \
    t0 = a0;  t1 = a1; \
    a0 = t0 ^ ((~t1) & a2); \
    a1 = t1 ^ ((~a2) & a3); \
    a2 = a2 ^ ((~a3) & a4); \
    a3 = a3 ^ ((~a4) & t0); \
    a4 = a4 ^ ((~t0) & t1); \
    t0 = a5;  t1 = a6; \
    a5 = t0 ^ ((~t1) & a7); \
    a6 = t1 ^ ((~a7) & a8); \
    a7 = a7 ^ ((~a8) & a9); \
    a8 = a8 ^ ((~a9) & t0); \
    a9 = a9 ^ ((~t0) & t1); \
    t0 = a10; t1 = a11; \
    a10 = t0 ^ ((~t1) & a12); \
    a11 = t1 ^ ((~a12) & a13); \
    a12 = a12 ^ ((~a13) & a14); \
    a13 = a13 ^ ((~a14) & t0); \
    a14 = a14 ^ ((~t0) & t1); \
    t0 = a15; t1 = a16; \
    a15 = t0 ^ ((~t1) & a17); \
    a16 = t1 ^ ((~a17) & a18); \
    a17 = a17 ^ ((~a18) & a19); \
    a18 = a18 ^ ((~a19) & t0); \
    a19 = a19 ^ ((~t0) & t1); \
    t0 = a20; t1 = a21; \
    a20 = t0 ^ ((~t1) & a22); \
    a21 = t1 ^ ((~a22) & a23); \
    a22 = a22 ^ ((~a23) & a24); \
    a23 = a23 ^ ((~a24) & t0); \
    a24 = a24 ^ ((~t0) & t1); \
    a0 ^= uint2((RCLO), (RCHI)); \
}

#define KECCAK_CHI_IOTA_N2_FIRST() { \
    const uint2 C3  = uint2(0x00C00000u, 0x00000000u); \
    const uint2 C5  = uint2(0x60000000u, 0x00000000u); \
    const uint2 C12 = uint2(0x0C000000u, 0x00000000u); \
    const uint2 C19 = uint2(0x00000000u, 0x06000000u); \
    const uint2 C21 = uint2(0x00000000u, 0x03000000u); \
    uint2 b0; uint2 b1; uint2 b2; uint2 b3; uint2 b4; \
    b0 = a0; b1 = a1; b2 = a2; b4 = a4; \
    a0 = (b0 ^ ((~b1) & b2)) ^ uint2(0x00000001u, 0x00000000u); \
    a1 =  b1 ^ ((~b2) & C3); \
    a2 =  b2 ^ ((~C3) & b4); \
    a3 =  C3 ^ ((~b4) & b0); \
    a4 =  b4 ^ ((~b0) & b1); \
    b1 = a6; b2 = a7; b3 = a8; b4 = a9; \
    a5 =  C5 ^ ((~b1) & b2); \
    a6 =  b1 ^ ((~b2) & b3); \
    a7 =  b2 ^ ((~b3) & b4); \
    a8 =  b3 ^ ((~b4) & C5); \
    a9 =  b4 ^ ((~C5) & b1); \
    b0 = a10; b1 = a11; b3 = a13; b4 = a14; \
    a10 = b0 ^ ((~b1) & C12); \
    a11 = b1 ^ ((~C12) & b3); \
    a12 = C12 ^ ((~b3) & b4); \
    a13 = b3 ^ ((~b4) & b0); \
    a14 = b4 ^ ((~b0) & b1); \
    b0 = a15; b1 = a16; b2 = a17; b3 = a18; \
    a15 = b0 ^ ((~b1) & b2); \
    a16 = b1 ^ ((~b2) & b3); \
    a17 = b2 ^ ((~b3) & C19); \
    a18 = b3 ^ ((~C19) & b0); \
    a19 = C19 ^ ((~b0) & b1); \
    b0 = a20; b2 = a22; b3 = a23; b4 = a24; \
    a20 = b0 ^ ((~C21) & b2); \
    a21 = C21 ^ ((~b2) & b3); \
    a22 = b2 ^ ((~b3) & b4); \
    a23 = b3 ^ ((~b4) & b0); \
    a24 = b4 ^ ((~b0) & C21); \
}

#define KECCAK_RHO_PI_CHI_IOTA(RCLO, RCHI) { \
    uint2 t = a1; \
    a1  = ROL2_HI(a6,  44u); \
    a6  = ROL2_LO(a9,  20u); \
    a9  = ROL2_HI(a22, 61u); \
    a22 = ROL2_HI(a14, 39u); \
    a14 = ROL2_LO(a20, 18u); \
    a20 = ROL2_HI(a2,  62u); \
    a2  = ROL2_HI(a12, 43u); \
    a12 = ROL2_LO(a13, 25u); \
    a13 = ROL2_LO(a19, 8u); \
    a19 = ROL2_HI(a23, 56u); \
    a23 = ROL2_HI(a15, 41u); \
    a15 = ROL2_LO(a4,  27u); \
    a4  = ROL2_LO(a24, 14u); \
    a24 = ROL2_LO(a21, 2u); \
    a21 = ROL2_HI(a8,  55u); \
    a8  = ROL2_HI(a16, 45u); \
    a16 = ROL2_HI(a5,  36u); \
    a5  = ROL2_LO(a3,  28u); \
    a3  = ROL2_LO(a18, 21u); \
    a18 = ROL2_LO(a17, 15u); \
    a17 = ROL2_LO(a11, 10u); \
    a11 = ROL2_LO(a7,  6u); \
    a7  = ROL2_LO(a10, 3u); \
    a10 = ROL2_LO(t,   1u); \
    KECCAK_CHI_IOTA(RCLO, RCHI) \
}

#define KECCAK_ROUND(RCLO, RCHI) { \
    uint2 c0 = a0 ^ a5 ^ a10 ^ a15 ^ a20; \
    uint2 c1 = a1 ^ a6 ^ a11 ^ a16 ^ a21; \
    uint2 c2 = a2 ^ a7 ^ a12 ^ a17 ^ a22; \
    uint2 c3 = a3 ^ a8 ^ a13 ^ a18 ^ a23; \
    uint2 c4 = a4 ^ a9 ^ a14 ^ a19 ^ a24; \
    uint2 d = c4 ^ ROL2_LO(c1, 1u); \
    a0 ^= d; a5 ^= d; a10 ^= d; a15 ^= d; a20 ^= d; \
    d = c0 ^ ROL2_LO(c2, 1u); \
    a1 ^= d; a6 ^= d; a11 ^= d; a16 ^= d; a21 ^= d; \
    d = c1 ^ ROL2_LO(c3, 1u); \
    a2 ^= d; a7 ^= d; a12 ^= d; a17 ^= d; a22 ^= d; \
    d = c2 ^ ROL2_LO(c4, 1u); \
    a3 ^= d; a8 ^= d; a13 ^= d; a18 ^= d; a23 ^= d; \
    d = c3 ^ ROL2_LO(c0, 1u); \
    a4 ^= d; a9 ^= d; a14 ^= d; a19 ^= d; a24 ^= d; \
    KECCAK_RHO_PI_CHI_IOTA(RCLO, RCHI) \
}

#define KECCAK_MIDDLE_1_TO_22() \
    KECCAK_ROUND(0x00008082u, 0x00000000u) \
    KECCAK_ROUND(0x0000808Au, 0x80000000u) \
    KECCAK_ROUND(0x80008000u, 0x80000000u) \
    KECCAK_ROUND(0x0000808Bu, 0x00000000u) \
    KECCAK_ROUND(0x80000001u, 0x00000000u) \
    KECCAK_ROUND(0x80008081u, 0x80000000u) \
    KECCAK_ROUND(0x00008009u, 0x80000000u) \
    KECCAK_ROUND(0x0000008Au, 0x00000000u) \
    KECCAK_ROUND(0x00000088u, 0x00000000u) \
    KECCAK_ROUND(0x80008009u, 0x00000000u) \
    KECCAK_ROUND(0x8000000Au, 0x00000000u) \
    KECCAK_ROUND(0x8000808Bu, 0x00000000u) \
    KECCAK_ROUND(0x0000008Bu, 0x80000000u) \
    KECCAK_ROUND(0x00008089u, 0x80000000u) \
    KECCAK_ROUND(0x00008003u, 0x80000000u) \
    KECCAK_ROUND(0x00008002u, 0x80000000u) \
    KECCAK_ROUND(0x00000080u, 0x80000000u) \
    KECCAK_ROUND(0x0000800Au, 0x00000000u) \
    KECCAK_ROUND(0x8000000Au, 0x80000000u) \
    KECCAK_ROUND(0x80008081u, 0x80000000u) \
    KECCAK_ROUND(0x00008080u, 0x80000000u) \
    KECCAK_ROUND(0x80000001u, 0x00000000u)

#define KECCAK_LAST2(RCLO, RCHI) { \
    uint2 c0 = a0 ^ a5 ^ a10 ^ a15 ^ a20; \
    uint2 c1 = a1 ^ a6 ^ a11 ^ a16 ^ a21; \
    uint2 c2 = a2 ^ a7 ^ a12 ^ a17 ^ a22; \
    uint2 c3 = a3 ^ a8 ^ a13 ^ a18 ^ a23; \
    uint2 c4 = a4 ^ a9 ^ a14 ^ a19 ^ a24; \
    uint2 d0 = c4 ^ ROL2_LO(c1, 1u); \
    uint2 d1 = c0 ^ ROL2_LO(c2, 1u); \
    uint2 d2 = c1 ^ ROL2_LO(c3, 1u); \
    uint2 d3 = c2 ^ ROL2_LO(c4, 1u); \
    uint2 b0 = a0 ^ d0; \
    uint2 b1 = ROL2_HI(a6  ^ d1, 44u); \
    uint2 b2 = ROL2_HI(a12 ^ d2, 43u); \
    uint2 b3 = ROL2_LO(a18 ^ d3, 21u); \
    a0 = (b0 ^ ((~b1) & b2)) ^ uint2((RCLO), (RCHI)); \
    a1 =  b1 ^ ((~b2) & b3); \
}

#define KECCAK_LAST4(RCLO, RCHI) { \
    uint2 c0 = a0 ^ a5 ^ a10 ^ a15 ^ a20; \
    uint2 c1 = a1 ^ a6 ^ a11 ^ a16 ^ a21; \
    uint2 c2 = a2 ^ a7 ^ a12 ^ a17 ^ a22; \
    uint2 c3 = a3 ^ a8 ^ a13 ^ a18 ^ a23; \
    uint2 c4 = a4 ^ a9 ^ a14 ^ a19 ^ a24; \
    uint2 d0 = c4 ^ ROL2_LO(c1, 1u); \
    uint2 d1 = c0 ^ ROL2_LO(c2, 1u); \
    uint2 d2 = c1 ^ ROL2_LO(c3, 1u); \
    uint2 d3 = c2 ^ ROL2_LO(c4, 1u); \
    uint2 d4 = c3 ^ ROL2_LO(c0, 1u); \
    uint2 b0 = a0 ^ d0; \
    uint2 b1 = ROL2_HI(a6  ^ d1, 44u); \
    uint2 b2 = ROL2_HI(a12 ^ d2, 43u); \
    uint2 b3 = ROL2_LO(a18 ^ d3, 21u); \
    uint2 b4 = ROL2_LO(a24 ^ d4, 14u); \
    a0 = (b0 ^ ((~b1) & b2)) ^ uint2((RCLO), (RCHI)); \
    a1 =  b1 ^ ((~b2) & b3); \
    a2 =  b2 ^ ((~b3) & b4); \
    a3 =  b3 ^ ((~b4) & b0); \
}

kernel void wots_chain(
    device const ulong *seeds    [[buffer(0)]],
    device       ulong *tips     [[buffer(1)]],
    constant uint      &n_chains [[buffer(2)]],
    constant uint      &n_bytes  [[buffer(3)]],
    constant uint      &w        [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n_chains) return;

    if (n_bytes == 16u) {
        uint base = idx << 1u;
        uint2 v0 = split_u64(seeds[base + 0u]);
        uint2 v1 = split_u64(seeds[base + 1u]);

        for (uint step = w; step != 0u; --step) {
            uint2 d0 = ROL2_LO(v1, 1u) ^ uint2(0x00000001u, 0x00000000u);
            uint2 d1 = v0 ^ uint2(0x0000000Cu, 0x00000000u);
            uint2 d2 = v1 ^ uint2(0x00000000u, 0x80000000u);

            uint2 a0  = v0 ^ d0;
            uint2 a1  = ROL2_HI(d1, 44u);
            uint2 a2  = ROL2_HI(d2, 43u);
            uint2 a3;
            uint2 a4  = ROL2_LO(v0, 15u);

            uint2 a5;
            uint2 a6  = ROL2_LO(v0, 21u);
            uint2 a7  = ROL2_LO(d0, 3u);
            uint2 a8  = ROL2_HI(uint2(0x00000000u, 0x80000000u) ^ d1, 45u);
            uint2 a9  = ROL2_HI(d2, 61u);

            uint2 a10 = ROL2_LO(v1 ^ d1, 1u);
            uint2 a11 = ROL2_LO(d2, 6u);
            uint2 a12;
            uint2 a13 = ROL2_LO(v0, 9u);
            uint2 a14 = ROL2_LO(d0, 18u);

            uint2 a15 = ROL2_LO(v0, 28u);
            uint2 a16 = ROL2_HI(d0, 36u);
            uint2 a17 = ROL2_LO(d1, 10u);
            uint2 a18 = ROL2_LO(d2, 15u);
            uint2 a19;

            uint2 a20 = ROL2_HI(uint2(0x00000006u, 0x00000000u) ^ d2, 62u);
            uint2 a21;
            uint2 a22 = ROL2_HI(v0, 40u);
            uint2 a23 = ROL2_HI(d0, 41u);
            uint2 a24 = ROL2_LO(d1, 2u);

            KECCAK_CHI_IOTA_N2_FIRST()
            KECCAK_MIDDLE_1_TO_22()
            KECCAK_LAST2(0x80008008u, 0x80000000u)

            v0 = a0;
            v1 = a1;
        }

        tips[base + 0u] = join_u64(v0);
        tips[base + 1u] = join_u64(v1);
    } else {
        uint base = idx << 2u;
        uint2 v0 = split_u64(seeds[base + 0u]);
        uint2 v1 = split_u64(seeds[base + 1u]);
        uint2 v2 = split_u64(seeds[base + 2u]);
        uint2 v3 = split_u64(seeds[base + 3u]);

        for (uint step = w; step != 0u; --step) {
            uint2 d0 = ROL2_LO(v1, 1u) ^ uint2(0x00000007u, 0x00000000u);
            uint2 d1 = v0 ^ ROL2_LO(v2, 1u);
            uint2 d2 = v1 ^ uint2(0x00000000u, 0x80000000u) ^ ROL2_LO(v3, 1u);
            uint2 d3 = v2 ^ uint2(0x0000000Cu, 0x00000000u);
            uint2 d4 = v3 ^ ROL2_LO(v0, 1u);

            uint2 a0  = v0 ^ d0;
            uint2 a1  = ROL2_HI(d1, 44u);
            uint2 a2  = ROL2_HI(d2, 43u);
            uint2 a3  = ROL2_LO(d3, 21u);
            uint2 a4  = ROL2_LO(d4, 14u);

            uint2 a5  = ROL2_LO(v3 ^ d3, 28u);
            uint2 a6  = ROL2_LO(d4, 20u);
            uint2 a7  = ROL2_LO(d0, 3u);
            uint2 a8  = ROL2_HI(uint2(0x00000000u, 0x80000000u) ^ d1, 45u);
            uint2 a9  = ROL2_HI(d2, 61u);

            uint2 a10 = ROL2_LO(v1 ^ d1, 1u);
            uint2 a11 = ROL2_LO(d2, 6u);
            uint2 a12 = ROL2_LO(d3, 25u);
            uint2 a13 = ROL2_LO(d4, 8u);
            uint2 a14 = ROL2_LO(d0, 18u);

            uint2 a15 = ROL2_LO(uint2(0x00000006u, 0x00000000u) ^ d4, 27u);
            uint2 a16 = ROL2_HI(d0, 36u);
            uint2 a17 = ROL2_LO(d1, 10u);
            uint2 a18 = ROL2_LO(d2, 15u);
            uint2 a19 = ROL2_HI(d3, 56u);

            uint2 a20 = ROL2_HI(v2 ^ d2, 62u);
            uint2 a21 = ROL2_HI(d3, 55u);
            uint2 a22 = ROL2_HI(d4, 39u);
            uint2 a23 = ROL2_HI(d0, 41u);
            uint2 a24 = ROL2_LO(d1, 2u);

            KECCAK_CHI_IOTA(0x00000001u, 0x00000000u)
            KECCAK_MIDDLE_1_TO_22()
            KECCAK_LAST4(0x80008008u, 0x80000000u)

            v0 = a0;
            v1 = a1;
            v2 = a2;
            v3 = a3;
        }

        tips[base + 0u] = join_u64(v0);
        tips[base + 1u] = join_u64(v1);
        tips[base + 2u] = join_u64(v2);
        tips[base + 3u] = join_u64(v3);
    }
}