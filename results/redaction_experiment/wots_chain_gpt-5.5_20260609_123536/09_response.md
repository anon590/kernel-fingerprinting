I’m switching the permutation core to native 64-bit Keccak lanes instead of explicit `uint2` half-lanes. This is structurally different: it removes split/join traffic and many vector constructors, while keeping the same first-round sparse-state specialization and final-round truncation. On Apple compilers, constant 64-bit rotates can be lowered to efficient fixed half-word sequences, potentially reducing register pressure and scheduling overhead versus manually carrying every lane as `uint2`.

```metal
#include <metal_stdlib>
using namespace metal;

#define U64(HI, LO) ((ulong(HI) << 32u) | ulong(LO))
#define ROL64(x, N) (((x) << (N)) | ((x) >> (64u - (N))))

#define KECCAK_CHI_IOTA64(RC) { \
    ulong t0; ulong t1; \
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
    a0 ^= (RC); \
}

#define KECCAK_RHO_PI_CHI_IOTA64(RC) { \
    ulong t = a1; \
    a1  = ROL64(a6,  44u); \
    a6  = ROL64(a9,  20u); \
    a9  = ROL64(a22, 61u); \
    a22 = ROL64(a14, 39u); \
    a14 = ROL64(a20, 18u); \
    a20 = ROL64(a2,  62u); \
    a2  = ROL64(a12, 43u); \
    a12 = ROL64(a13, 25u); \
    a13 = ROL64(a19, 8u); \
    a19 = ROL64(a23, 56u); \
    a23 = ROL64(a15, 41u); \
    a15 = ROL64(a4,  27u); \
    a4  = ROL64(a24, 14u); \
    a24 = ROL64(a21, 2u); \
    a21 = ROL64(a8,  55u); \
    a8  = ROL64(a16, 45u); \
    a16 = ROL64(a5,  36u); \
    a5  = ROL64(a3,  28u); \
    a3  = ROL64(a18, 21u); \
    a18 = ROL64(a17, 15u); \
    a17 = ROL64(a11, 10u); \
    a11 = ROL64(a7,  6u); \
    a7  = ROL64(a10, 3u); \
    a10 = ROL64(t,   1u); \
    KECCAK_CHI_IOTA64(RC) \
}

#define KECCAK_ROUND64(RC) { \
    ulong c0 = a0 ^ a5 ^ a10 ^ a15 ^ a20; \
    ulong c1 = a1 ^ a6 ^ a11 ^ a16 ^ a21; \
    ulong c2 = a2 ^ a7 ^ a12 ^ a17 ^ a22; \
    ulong c3 = a3 ^ a8 ^ a13 ^ a18 ^ a23; \
    ulong c4 = a4 ^ a9 ^ a14 ^ a19 ^ a24; \
    ulong d = c4 ^ ROL64(c1, 1u); \
    a0 ^= d; a5 ^= d; a10 ^= d; a15 ^= d; a20 ^= d; \
    d = c0 ^ ROL64(c2, 1u); \
    a1 ^= d; a6 ^= d; a11 ^= d; a16 ^= d; a21 ^= d; \
    d = c1 ^ ROL64(c3, 1u); \
    a2 ^= d; a7 ^= d; a12 ^= d; a17 ^= d; a22 ^= d; \
    d = c2 ^ ROL64(c4, 1u); \
    a3 ^= d; a8 ^= d; a13 ^= d; a18 ^= d; a23 ^= d; \
    d = c3 ^ ROL64(c0, 1u); \
    a4 ^= d; a9 ^= d; a14 ^= d; a19 ^= d; a24 ^= d; \
    KECCAK_RHO_PI_CHI_IOTA64(RC) \
}

#define KECCAK_MIDDLE_1_TO_22_64() \
    KECCAK_ROUND64(U64(0x00000000u, 0x00008082u)) \
    KECCAK_ROUND64(U64(0x80000000u, 0x0000808Au)) \
    KECCAK_ROUND64(U64(0x80000000u, 0x80008000u)) \
    KECCAK_ROUND64(U64(0x00000000u, 0x0000808Bu)) \
    KECCAK_ROUND64(U64(0x00000000u, 0x80000001u)) \
    KECCAK_ROUND64(U64(0x80000000u, 0x80008081u)) \
    KECCAK_ROUND64(U64(0x80000000u, 0x00008009u)) \
    KECCAK_ROUND64(U64(0x00000000u, 0x0000008Au)) \
    KECCAK_ROUND64(U64(0x00000000u, 0x00000088u)) \
    KECCAK_ROUND64(U64(0x00000000u, 0x80008009u)) \
    KECCAK_ROUND64(U64(0x00000000u, 0x8000000Au)) \
    KECCAK_ROUND64(U64(0x00000000u, 0x8000808Bu)) \
    KECCAK_ROUND64(U64(0x80000000u, 0x0000008Bu)) \
    KECCAK_ROUND64(U64(0x80000000u, 0x00008089u)) \
    KECCAK_ROUND64(U64(0x80000000u, 0x00008003u)) \
    KECCAK_ROUND64(U64(0x80000000u, 0x00008002u)) \
    KECCAK_ROUND64(U64(0x80000000u, 0x00000080u)) \
    KECCAK_ROUND64(U64(0x00000000u, 0x0000800Au)) \
    KECCAK_ROUND64(U64(0x80000000u, 0x8000000Au)) \
    KECCAK_ROUND64(U64(0x80000000u, 0x80008081u)) \
    KECCAK_ROUND64(U64(0x80000000u, 0x00008080u)) \
    KECCAK_ROUND64(U64(0x00000000u, 0x80000001u))

#define KECCAK_LAST2_64(RC) { \
    ulong c0 = a0 ^ a5 ^ a10 ^ a15 ^ a20; \
    ulong c1 = a1 ^ a6 ^ a11 ^ a16 ^ a21; \
    ulong c2 = a2 ^ a7 ^ a12 ^ a17 ^ a22; \
    ulong c3 = a3 ^ a8 ^ a13 ^ a18 ^ a23; \
    ulong c4 = a4 ^ a9 ^ a14 ^ a19 ^ a24; \
    ulong d0 = c4 ^ ROL64(c1, 1u); \
    ulong d1 = c0 ^ ROL64(c2, 1u); \
    ulong d2 = c1 ^ ROL64(c3, 1u); \
    ulong d3 = c2 ^ ROL64(c4, 1u); \
    ulong b0 = a0 ^ d0; \
    ulong b1 = ROL64(a6  ^ d1, 44u); \
    ulong b2 = ROL64(a12 ^ d2, 43u); \
    ulong b3 = ROL64(a18 ^ d3, 21u); \
    a0 = (b0 ^ ((~b1) & b2)) ^ (RC); \
    a1 =  b1 ^ ((~b2) & b3); \
}

#define KECCAK_LAST4_64(RC) { \
    ulong c0 = a0 ^ a5 ^ a10 ^ a15 ^ a20; \
    ulong c1 = a1 ^ a6 ^ a11 ^ a16 ^ a21; \
    ulong c2 = a2 ^ a7 ^ a12 ^ a17 ^ a22; \
    ulong c3 = a3 ^ a8 ^ a13 ^ a18 ^ a23; \
    ulong c4 = a4 ^ a9 ^ a14 ^ a19 ^ a24; \
    ulong d0 = c4 ^ ROL64(c1, 1u); \
    ulong d1 = c0 ^ ROL64(c2, 1u); \
    ulong d2 = c1 ^ ROL64(c3, 1u); \
    ulong d3 = c2 ^ ROL64(c4, 1u); \
    ulong d4 = c3 ^ ROL64(c0, 1u); \
    ulong b0 = a0 ^ d0; \
    ulong b1 = ROL64(a6  ^ d1, 44u); \
    ulong b2 = ROL64(a12 ^ d2, 43u); \
    ulong b3 = ROL64(a18 ^ d3, 21u); \
    ulong b4 = ROL64(a24 ^ d4, 14u); \
    a0 = (b0 ^ ((~b1) & b2)) ^ (RC); \
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

    const ulong DOM = ulong(0x00000006u);
    const ulong PAD = U64(0x80000000u, 0x00000000u);
    const ulong RC0 = U64(0x00000000u, 0x00000001u);
    const ulong RCL = U64(0x80000000u, 0x80008008u);

    if (n_bytes == 16u) {
        uint base = idx << 1u;
        ulong v0 = seeds[base + 0u];
        ulong v1 = seeds[base + 1u];

        for (uint step = w; step != 0u; --step) {
            ulong d0 = ROL64(v1, 1u) ^ ulong(0x00000001u);
            ulong d1 = v0 ^ ulong(0x0000000Cu);
            ulong d2 = v1 ^ PAD;

            ulong a0  = v0 ^ d0;
            ulong a1  = ROL64(d1, 44u);
            ulong a2  = ROL64(d2, 43u);
            ulong a3  = ROL64(DOM, 21u);
            ulong a4  = ROL64(v0, 15u);

            ulong a5  = ROL64(DOM, 28u);
            ulong a6  = ROL64(v0, 21u);
            ulong a7  = ROL64(d0, 3u);
            ulong a8  = ROL64(PAD ^ d1, 45u);
            ulong a9  = ROL64(d2, 61u);

            ulong a10 = ROL64(v1 ^ d1, 1u);
            ulong a11 = ROL64(d2, 6u);
            ulong a12 = ROL64(DOM, 25u);
            ulong a13 = ROL64(v0, 9u);
            ulong a14 = ROL64(d0, 18u);

            ulong a15 = ROL64(v0, 28u);
            ulong a16 = ROL64(d0, 36u);
            ulong a17 = ROL64(d1, 10u);
            ulong a18 = ROL64(d2, 15u);
            ulong a19 = ROL64(DOM, 56u);

            ulong a20 = ROL64(DOM ^ d2, 62u);
            ulong a21 = ROL64(DOM, 55u);
            ulong a22 = ROL64(v0, 40u);
            ulong a23 = ROL64(d0, 41u);
            ulong a24 = ROL64(d1, 2u);

            KECCAK_CHI_IOTA64(RC0)
            KECCAK_MIDDLE_1_TO_22_64()
            KECCAK_LAST2_64(RCL)

            v0 = a0;
            v1 = a1;
        }

        tips[base + 0u] = v0;
        tips[base + 1u] = v1;
    } else {
        uint base = idx << 2u;
        ulong v0 = seeds[base + 0u];
        ulong v1 = seeds[base + 1u];
        ulong v2 = seeds[base + 2u];
        ulong v3 = seeds[base + 3u];

        for (uint step = w; step != 0u; --step) {
            ulong d0 = ROL64(v1, 1u) ^ ulong(0x00000007u);
            ulong d1 = v0 ^ ROL64(v2, 1u);
            ulong d2 = v1 ^ PAD ^ ROL64(v3, 1u);
            ulong d3 = v2 ^ ulong(0x0000000Cu);
            ulong d4 = v3 ^ ROL64(v0, 1u);

            ulong a0  = v0 ^ d0;
            ulong a1  = ROL64(d1, 44u);
            ulong a2  = ROL64(d2, 43u);
            ulong a3  = ROL64(d3, 21u);
            ulong a4  = ROL64(d4, 14u);

            ulong a5  = ROL64(v3 ^ d3, 28u);
            ulong a6  = ROL64(d4, 20u);
            ulong a7  = ROL64(d0, 3u);
            ulong a8  = ROL64(PAD ^ d1, 45u);
            ulong a9  = ROL64(d2, 61u);

            ulong a10 = ROL64(v1 ^ d1, 1u);
            ulong a11 = ROL64(d2, 6u);
            ulong a12 = ROL64(d3, 25u);
            ulong a13 = ROL64(d4, 8u);
            ulong a14 = ROL64(d0, 18u);

            ulong a15 = ROL64(DOM ^ d4, 27u);
            ulong a16 = ROL64(d0, 36u);
            ulong a17 = ROL64(d1, 10u);
            ulong a18 = ROL64(d2, 15u);
            ulong a19 = ROL64(d3, 56u);

            ulong a20 = ROL64(v2 ^ d2, 62u);
            ulong a21 = ROL64(d3, 55u);
            ulong a22 = ROL64(d4, 39u);
            ulong a23 = ROL64(d0, 41u);
            ulong a24 = ROL64(d1, 2u);

            KECCAK_CHI_IOTA64(RC0)
            KECCAK_MIDDLE_1_TO_22_64()
            KECCAK_LAST4_64(RCL)

            v0 = a0;
            v1 = a1;
            v2 = a2;
            v3 = a3;
        }

        tips[base + 0u] = v0;
        tips[base + 1u] = v1;
        tips[base + 2u] = v2;
        tips[base + 3u] = v3;
    }
}
```