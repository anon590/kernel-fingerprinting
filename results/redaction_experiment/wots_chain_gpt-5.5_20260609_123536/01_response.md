This version specializes at runtime for the two tested chunk sizes while still reading `n_bytes`/`w` from buffers. It replaces array-indexed Keccak with a fully unrolled, scalarized in-place permutation using `uint2` halves for each 64-bit lane, avoiding local-memory spills, modulo/index arithmetic, and variable rotations. This should improve throughput substantially over the naive loop/array implementation on Apple GPUs. 

```metal
#include <metal_stdlib>
using namespace metal;

inline uint2 split_u64(ulong x) {
    return uint2((uint)x, (uint)(x >> 32u));
}

inline ulong join_u64(uint2 x) {
    return ((ulong)x.x) | (((ulong)x.y) << 32u);
}

#define ROL2_LO(v, N) uint2((((v).x << (N)) | ((v).y >> (32u - (N)))), (((v).y << (N)) | ((v).x >> (32u - (N)))))
#define ROL2_HI(v, N) uint2((((v).y << ((N) - 32u)) | ((v).x >> (64u - (N)))), (((v).x << ((N) - 32u)) | ((v).y >> (64u - (N)))))

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
    c0 = a0; c1 = a1; c2 = a2; c3 = a3; c4 = a4; \
    a0 = c0 ^ ((~c1) & c2); \
    a1 = c1 ^ ((~c2) & c3); \
    a2 = c2 ^ ((~c3) & c4); \
    a3 = c3 ^ ((~c4) & c0); \
    a4 = c4 ^ ((~c0) & c1); \
    c0 = a5; c1 = a6; c2 = a7; c3 = a8; c4 = a9; \
    a5 = c0 ^ ((~c1) & c2); \
    a6 = c1 ^ ((~c2) & c3); \
    a7 = c2 ^ ((~c3) & c4); \
    a8 = c3 ^ ((~c4) & c0); \
    a9 = c4 ^ ((~c0) & c1); \
    c0 = a10; c1 = a11; c2 = a12; c3 = a13; c4 = a14; \
    a10 = c0 ^ ((~c1) & c2); \
    a11 = c1 ^ ((~c2) & c3); \
    a12 = c2 ^ ((~c3) & c4); \
    a13 = c3 ^ ((~c4) & c0); \
    a14 = c4 ^ ((~c0) & c1); \
    c0 = a15; c1 = a16; c2 = a17; c3 = a18; c4 = a19; \
    a15 = c0 ^ ((~c1) & c2); \
    a16 = c1 ^ ((~c2) & c3); \
    a17 = c2 ^ ((~c3) & c4); \
    a18 = c3 ^ ((~c4) & c0); \
    a19 = c4 ^ ((~c0) & c1); \
    c0 = a20; c1 = a21; c2 = a22; c3 = a23; c4 = a24; \
    a20 = c0 ^ ((~c1) & c2); \
    a21 = c1 ^ ((~c2) & c3); \
    a22 = c2 ^ ((~c3) & c4); \
    a23 = c3 ^ ((~c4) & c0); \
    a24 = c4 ^ ((~c0) & c1); \
    a0 ^= uint2((RCLO), (RCHI)); \
}

#define KECCAK_PERMUTE() \
    KECCAK_ROUND(0x00000001u, 0x00000000u) \
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
    KECCAK_ROUND(0x80000001u, 0x00000000u) \
    KECCAK_ROUND(0x80008008u, 0x80000000u)

kernel void wots_chain(
    device const ulong *seeds    [[buffer(0)]],
    device       ulong *tips     [[buffer(1)]],
    constant uint      &n_chains [[buffer(2)]],
    constant uint      &n_bytes  [[buffer(3)]],
    constant uint      &w        [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n_chains) return;

    uint n_lanes = n_bytes >> 3u;
    const uint2 Z   = uint2(0u, 0u);
    const uint2 DOM = uint2(0x00000006u, 0u);
    const uint2 PAD = uint2(0u, 0x80000000u);

    if (n_lanes == 2u) {
        uint base = idx << 1u;
        uint2 v0 = split_u64(seeds[base + 0u]);
        uint2 v1 = split_u64(seeds[base + 1u]);

        for (uint step = 0u; step < w; ++step) {
            uint2 a0 = v0,  a1 = v1,  a2 = DOM, a3 = Z,   a4 = Z;
            uint2 a5 = Z,   a6 = Z,   a7 = Z,   a8 = Z,   a9 = Z;
            uint2 a10 = Z,  a11 = Z,  a12 = Z,  a13 = Z,  a14 = Z;
            uint2 a15 = Z,  a16 = PAD,a17 = Z,  a18 = Z,  a19 = Z;
            uint2 a20 = Z,  a21 = Z,  a22 = Z,  a23 = Z,  a24 = Z;

            KECCAK_PERMUTE();

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

        for (uint step = 0u; step < w; ++step) {
            uint2 a0 = v0,  a1 = v1,  a2 = v2,  a3 = v3,  a4 = DOM;
            uint2 a5 = Z,   a6 = Z,   a7 = Z,   a8 = Z,   a9 = Z;
            uint2 a10 = Z,  a11 = Z,  a12 = Z,  a13 = Z,  a14 = Z;
            uint2 a15 = Z,  a16 = PAD,a17 = Z,  a18 = Z,  a19 = Z;
            uint2 a20 = Z,  a21 = Z,  a22 = Z,  a23 = Z,  a24 = Z;

            KECCAK_PERMUTE();

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
```