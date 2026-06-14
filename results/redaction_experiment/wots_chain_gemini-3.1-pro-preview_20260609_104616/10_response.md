To break out of the stagnation, we must structurally change how the state is presented to the compiler. The previous attempt and the incumbent rely on an array `ulong A[25]` with `#pragma unroll`. While LLVM's SROA (Scalar Replacement of Aggregates) tries to promote arrays to registers, Keccak's heavy variable interference often causes SROA to falter, leading to suboptimal register coloring, invisible spilling, or redundant moves. 

This kernel completely abandons the state array. Instead, it implements the "Register Renaming / Lane Complementing" technique from the Keccak Code Package (KCP) entirely with macros. We define 50 scalar variables (`a00`..`a24` and `e00`..`e24`) and explicitly unroll the 24 rounds, ping-ponging the state. This absolutely forces the compiler into perfect register allocation. It also trivially enables LLVM to mathematically constant-fold the first round (since 21 lanes are exactly 0/0x06) and dead-store eliminate the unused lanes in the final round without crossing opaque array boundaries.

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant ulong KECCAK_RC[24] = {
    0x0000000000000001ul, 0x0000000000008082ul, 0x800000000000808Aul,
    0x8000000080008000ul, 0x000000000000808Bul, 0x0000000080000001ul,
    0x8000000080008081ul, 0x8000000000008009ul, 0x000000000000008Aul,
    0x0000000000000088ul, 0x0000000080008009ul, 0x000000008000000Aul,
    0x000000008000808Bul, 0x800000000000008Bul, 0x8000000000008089ul,
    0x8000000000008003ul, 0x8000000000008002ul, 0x8000000000000080ul,
    0x000000000000800Aul, 0x800000008000000Aul, 0x8000000080008081ul,
    0x8000000000008080ul, 0x0000000080000001ul, 0x8000000080008008ul,
};

inline __attribute__((always_inline))
ulong rotl64(ulong x, uint k) {
    return (x << k) | (x >> (64u - k));
}

#define K_ROUND(a00, a01, a02, a03, a04, \
                a05, a06, a07, a08, a09, \
                a10, a11, a12, a13, a14, \
                a15, a16, a17, a18, a19, \
                a20, a21, a22, a23, a24, \
                e00, e01, e02, e03, e04, \
                e05, e06, e07, e08, e09, \
                e10, e11, e12, e13, e14, \
                e15, e16, e17, e18, e19, \
                e20, e21, e22, e23, e24, rc) \
do { \
    ulong C0 = a00 ^ a05 ^ a10 ^ a15 ^ a20; \
    ulong C1 = a01 ^ a06 ^ a11 ^ a16 ^ a21; \
    ulong C2 = a02 ^ a07 ^ a12 ^ a17 ^ a22; \
    ulong C3 = a03 ^ a08 ^ a13 ^ a18 ^ a23; \
    ulong C4 = a04 ^ a09 ^ a14 ^ a19 ^ a24; \
    \
    ulong D0 = C4 ^ rotl64(C1, 1u); \
    ulong D1 = C0 ^ rotl64(C2, 1u); \
    ulong D2 = C1 ^ rotl64(C3, 1u); \
    ulong D3 = C2 ^ rotl64(C4, 1u); \
    ulong D4 = C3 ^ rotl64(C0, 1u); \
    \
    ulong b00 = a00 ^ D0; \
    ulong b01 = rotl64(a06 ^ D1, 44u); \
    ulong b02 = rotl64(a12 ^ D2, 43u); \
    ulong b03 = rotl64(a18 ^ D3, 21u); \
    ulong b04 = rotl64(a24 ^ D4, 14u); \
    e00 = b00 ^ (b02 & ~b01) ^ rc; \
    e01 = b01 ^ (b03 & ~b02); \
    e02 = b02 ^ (b04 & ~b03); \
    e03 = b03 ^ (b00 & ~b04); \
    e04 = b04 ^ (b01 & ~b00); \
    \
    b00 = rotl64(a03 ^ D3, 28u); \
    b01 = rotl64(a09 ^ D4, 20u); \
    b02 = rotl64(a10 ^ D0, 3u); \
    b03 = rotl64(a16 ^ D1, 45u); \
    b04 = rotl64(a22 ^ D2, 61u); \
    e05 = b00 ^ (b02 & ~b01); \
    e06 = b01 ^ (b03 & ~b02); \
    e07 = b02 ^ (b04 & ~b03); \
    e08 = b03 ^ (b00 & ~b04); \
    e09 = b04 ^ (b01 & ~b00); \
    \
    b00 = rotl64(a01 ^ D1, 1u); \
    b01 = rotl64(a07 ^ D2, 6u); \
    b02 = rotl64(a13 ^ D3, 25u); \
    b03 = rotl64(a19 ^ D4, 8u); \
    b04 = rotl64(a20 ^ D0, 18u); \
    e10 = b00 ^ (b02 & ~b01); \
    e11 = b01 ^ (b03 & ~b02); \
    e12 = b02 ^ (b04 & ~b03); \
    e13 = b03 ^ (b00 & ~b04); \
    e14 = b04 ^ (b01 & ~b00); \
    \
    b00 = rotl64(a04 ^ D4, 27u); \
    b01 = rotl64(a05 ^ D0, 36u); \
    b02 = rotl64(a11 ^ D1, 10u); \
    b03 = rotl64(a17 ^ D2, 15u); \
    b04 = rotl64(a23 ^ D3, 56u); \
    e15 = b00 ^ (b02 & ~b01); \
    e16 = b01 ^ (b03 & ~b02); \
    e17 = b02 ^ (b04 & ~b03); \
    e18 = b03 ^ (b00 & ~b04); \
    e19 = b04 ^ (b01 & ~b00); \
    \
    b00 = rotl64(a02 ^ D2, 62u); \
    b01 = rotl64(a08 ^ D3, 55u); \
    b02 = rotl64(a14 ^ D4, 39u); \
    b03 = rotl64(a15 ^ D0, 41u); \
    b04 = rotl64(a21 ^ D1, 2u); \
    e20 = b00 ^ (b02 & ~b01); \
    e21 = b01 ^ (b03 & ~b02); \
    e22 = b02 ^ (b04 & ~b03); \
    e23 = b03 ^ (b00 & ~b04); \
    e24 = b04 ^ (b01 & ~b00); \
} while(0)

#define K_ROUND_PAIR(r) \
    K_ROUND(a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, \
            e00, e01, e02, e03, e04, e05, e06, e07, e08, e09, e10, e11, e12, e13, e14, e15, e16, e17, e18, e19, e20, e21, e22, e23, e24, KECCAK_RC[r]); \
    K_ROUND(e00, e01, e02, e03, e04, e05, e06, e07, e08, e09, e10, e11, e12, e13, e14, e15, e16, e17, e18, e19, e20, e21, e22, e23, e24, \
            a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, KECCAK_RC[r+1])

kernel void wots_chain(
    device const ulong *seeds    [[buffer(0)]],
    device       ulong *tips     [[buffer(1)]],
    constant uint      &n_chains [[buffer(2)]],
    constant uint      &n_bytes  [[buffer(3)]],
    constant uint      &w        [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n_chains) return;

    uint n_lanes = n_bytes >> 3;
    uint w_val = w;

    ulong a00, a01, a02, a03, a04, a05, a06, a07, a08, a09;
    ulong a10, a11, a12, a13, a14, a15, a16, a17, a18, a19;
    ulong a20, a21, a22, a23, a24;
    
    ulong e00, e01, e02, e03, e04, e05, e06, e07, e08, e09;
    ulong e10, e11, e12, e13, e14, e15, e16, e17, e18, e19;
    ulong e20, e21, e22, e23, e24;

    if (n_lanes == 2u) {
        device const ulong2 *seeds_v = (device const ulong2 *)seeds;
        ulong2 seed_val = seeds_v[idx];
        a00 = seed_val.x;
        a01 = seed_val.y;
        
        for (uint step = 0u; step < w_val; ++step) {
            a02 = 0x06ul;
            a03 = 0ul; a04 = 0ul; a05 = 0ul; a06 = 0ul;
            a07 = 0ul; a08 = 0ul; a09 = 0ul; a10 = 0ul;
            a11 = 0ul; a12 = 0ul; a13 = 0ul; a14 = 0ul;
            a15 = 0ul; a16 = 0x8000000000000000ul; a17 = 0ul;
            a18 = 0ul; a19 = 0ul; a20 = 0ul; a21 = 0ul;
            a22 = 0ul; a23 = 0ul; a24 = 0ul;
            
            K_ROUND_PAIR(0);
            K_ROUND_PAIR(2);
            K_ROUND_PAIR(4);
            K_ROUND_PAIR(6);
            K_ROUND_PAIR(8);
            K_ROUND_PAIR(10);
            K_ROUND_PAIR(12);
            K_ROUND_PAIR(14);
            K_ROUND_PAIR(16);
            K_ROUND_PAIR(18);
            K_ROUND_PAIR(20);
            K_ROUND_PAIR(22);
        }
        
        device ulong2 *tips_v = (device ulong2 *)tips;
        tips_v[idx] = ulong2(a00, a01);

    } else if (n_lanes == 4u) {
        device const ulong4 *seeds_v = (device const ulong4 *)seeds;
        ulong4 seed_val = seeds_v[idx];
        a00 = seed_val.x;
        a01 = seed_val.y;
        a02 = seed_val.z;
        a03 = seed_val.w;
        
        for (uint step = 0u; step < w_val; ++step) {
            a04 = 0x06ul;
            a05 = 0ul; a06 = 0ul; a07 = 0ul; a08 = 0ul; a09 = 0ul;
            a10 = 0ul; a11 = 0ul; a12 = 0ul; a13 = 0ul; a14 = 0ul;
            a15 = 0ul; a16 = 0x8000000000000000ul; a17 = 0ul;
            a18 = 0ul; a19 = 0ul; a20 = 0ul; a21 = 0ul;
            a22 = 0ul; a23 = 0ul; a24 = 0ul;
            
            K_ROUND_PAIR(0);
            K_ROUND_PAIR(2);
            K_ROUND_PAIR(4);
            K_ROUND_PAIR(6);
            K_ROUND_PAIR(8);
            K_ROUND_PAIR(10);
            K_ROUND_PAIR(12);
            K_ROUND_PAIR(14);
            K_ROUND_PAIR(16);
            K_ROUND_PAIR(18);
            K_ROUND_PAIR(20);
            K_ROUND_PAIR(22);
        }
        
        device ulong4 *tips_v = (device ulong4 *)tips;
        tips_v[idx] = ulong4(a00, a01, a02, a03);

    } else {
        uint base = idx * n_lanes;
        a00=0; a01=0; a02=0; a03=0; a04=0;
        a05=0; a06=0; a07=0; a08=0; a09=0;
        a10=0; a11=0; a12=0; a13=0; a14=0;
        a15=0; a16=0; a17=0; a18=0; a19=0;
        a20=0; a21=0; a22=0; a23=0; a24=0;
        
        if (n_lanes > 0) a00 = seeds[base + 0];
        if (n_lanes > 1) a01 = seeds[base + 1];
        if (n_lanes > 2) a02 = seeds[base + 2];
        if (n_lanes > 3) a03 = seeds[base + 3];
        if (n_lanes > 4) a04 = seeds[base + 4];
        if (n_lanes > 5) a05 = seeds[base + 5];
        if (n_lanes > 6) a06 = seeds[base + 6];
        if (n_lanes > 7) a07 = seeds[base + 7];
        
        for (uint step = 0u; step < w_val; ++step) {
            if (n_lanes < 16) a15 = 0;
            if (n_lanes < 15) a14 = 0;
            if (n_lanes < 14) a13 = 0;
            if (n_lanes < 13) a12 = 0;
            if (n_lanes < 12) a11 = 0;
            if (n_lanes < 11) a10 = 0;
            if (n_lanes < 10) a09 = 0;
            if (n_lanes < 9)  a08 = 0;
            if (n_lanes < 8)  a07 = 0;
            if (n_lanes < 7)  a06 = 0;
            if (n_lanes < 6)  a05 = 0;
            if (n_lanes < 5)  a04 = 0;
            if (n_lanes < 4)  a03 = 0;
            if (n_lanes < 3)  a02 = 0;
            if (n_lanes < 2)  a01 = 0;
            if (n_lanes < 1)  a00 = 0;
            
            a16 = 0x8000000000000000ul;
            a17 = 0; a18 = 0; a19 = 0; a20 = 0;
            a21 = 0; a22 = 0; a23 = 0; a24 = 0;
            
            switch(n_lanes) {
                case 0: a00 ^= 0x06ul; break;
                case 1: a01 ^= 0x06ul; break;
                case 2: a02 ^= 0x06ul; break;
                case 3: a03 ^= 0x06ul; break;
                case 4: a04 ^= 0x06ul; break;
                case 5: a05 ^= 0x06ul; break;
                case 6: a06 ^= 0x06ul; break;
                case 7: a07 ^= 0x06ul; break;
                case 8: a08 ^= 0x06ul; break;
                case 9: a09 ^= 0x06ul; break;
                case 10: a10 ^= 0x06ul; break;
                case 11: a11 ^= 0x06ul; break;
                case 12: a12 ^= 0x06ul; break;
                case 13: a13 ^= 0x06ul; break;
                case 14: a14 ^= 0x06ul; break;
                case 15: a15 ^= 0x06ul; break;
                case 16: a16 ^= 0x06ul; break;
            }
            
            K_ROUND_PAIR(0);
            K_ROUND_PAIR(2);
            K_ROUND_PAIR(4);
            K_ROUND_PAIR(6);
            K_ROUND_PAIR(8);
            K_ROUND_PAIR(10);
            K_ROUND_PAIR(12);
            K_ROUND_PAIR(14);
            K_ROUND_PAIR(16);
            K_ROUND_PAIR(18);
            K_ROUND_PAIR(20);
            K_ROUND_PAIR(22);
        }
        
        if (n_lanes > 0) tips[base + 0] = a00;
        if (n_lanes > 1) tips[base + 1] = a01;
        if (n_lanes > 2) tips[base + 2] = a02;
        if (n_lanes > 3) tips[base + 3] = a03;
        if (n_lanes > 4) tips[base + 4] = a04;
        if (n_lanes > 5) tips[base + 5] = a05;
        if (n_lanes > 6) tips[base + 6] = a06;
        if (n_lanes > 7) tips[base + 7] = a07;
    }
}
```