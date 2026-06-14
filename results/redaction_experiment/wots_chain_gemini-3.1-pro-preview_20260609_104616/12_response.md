To break out of the local optimum, we employ a **structurally different approach** to the Keccak inner loop by completely eliminating array-based state tracking and the alternating `E` arrays. 

We flatten the Keccak state into exactly 25 explicit scalar variables (`A00` to `A24`) and implement an **in-place** round macro. By carefully unrolling the Pi permutation directly into the scalar variable assignments, we map the destination indices directly back to `Axx`, halving the register footprint (from 50 variables down to 25). Furthermore, explicitly zeroing and padding these exact scalars right before the fully unrolled `K_ROUND` loop guarantees that LLVM's constant folding effortlessly eliminates the redundant XORs and rotations in Round 0.

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

// In-place fully scalarized Keccak round. Variables A00...A24 must exist in scope.
#define K_ROUND(rc) do { \
    ulong C0 = (A00 ^ A05) ^ (A10 ^ A15) ^ A20; \
    ulong C1 = (A01 ^ A06) ^ (A11 ^ A16) ^ A21; \
    ulong C2 = (A02 ^ A07) ^ (A12 ^ A17) ^ A22; \
    ulong C3 = (A03 ^ A08) ^ (A13 ^ A18) ^ A23; \
    ulong C4 = (A04 ^ A09) ^ (A14 ^ A19) ^ A24; \
    \
    ulong D0 = C4 ^ rotl64(C1, 1u); \
    ulong D1 = C0 ^ rotl64(C2, 1u); \
    ulong D2 = C1 ^ rotl64(C3, 1u); \
    ulong D3 = C2 ^ rotl64(C4, 1u); \
    ulong D4 = C3 ^ rotl64(C0, 1u); \
    \
    ulong b00 = A00 ^ D0; \
    ulong b01 = rotl64(A06 ^ D1, 44u); \
    ulong b02 = rotl64(A12 ^ D2, 43u); \
    ulong b03 = rotl64(A18 ^ D3, 21u); \
    ulong b04 = rotl64(A24 ^ D4, 14u); \
    \
    ulong b05 = rotl64(A03 ^ D3, 28u); \
    ulong b06 = rotl64(A09 ^ D4, 20u); \
    ulong b07 = rotl64(A10 ^ D0, 3u); \
    ulong b08 = rotl64(A16 ^ D1, 45u); \
    ulong b09 = rotl64(A22 ^ D2, 61u); \
    \
    ulong b10 = rotl64(A01 ^ D1, 1u); \
    ulong b11 = rotl64(A07 ^ D2, 6u); \
    ulong b12 = rotl64(A13 ^ D3, 25u); \
    ulong b13 = rotl64(A19 ^ D4, 8u); \
    ulong b14 = rotl64(A20 ^ D0, 18u); \
    \
    ulong b15 = rotl64(A04 ^ D4, 27u); \
    ulong b16 = rotl64(A05 ^ D0, 36u); \
    ulong b17 = rotl64(A11 ^ D1, 10u); \
    ulong b18 = rotl64(A17 ^ D2, 15u); \
    ulong b19 = rotl64(A23 ^ D3, 56u); \
    \
    ulong b20 = rotl64(A02 ^ D2, 62u); \
    ulong b21 = rotl64(A08 ^ D3, 55u); \
    ulong b22 = rotl64(A14 ^ D4, 39u); \
    ulong b23 = rotl64(A15 ^ D0, 41u); \
    ulong b24 = rotl64(A21 ^ D1, 2u); \
    \
    A00 = b00 ^ (b02 & ~b01) ^ (rc); \
    A01 = b01 ^ (b03 & ~b02); \
    A02 = b02 ^ (b04 & ~b03); \
    A03 = b03 ^ (b00 & ~b04); \
    A04 = b04 ^ (b01 & ~b00); \
    \
    A05 = b05 ^ (b07 & ~b06); \
    A06 = b06 ^ (b08 & ~b07); \
    A07 = b07 ^ (b09 & ~b08); \
    A08 = b08 ^ (b05 & ~b09); \
    A09 = b09 ^ (b06 & ~b05); \
    \
    A10 = b10 ^ (b12 & ~b11); \
    A11 = b11 ^ (b13 & ~b12); \
    A12 = b12 ^ (b14 & ~b13); \
    A13 = b13 ^ (b10 & ~b14); \
    A14 = b14 ^ (b11 & ~b10); \
    \
    A15 = b15 ^ (b17 & ~b16); \
    A16 = b16 ^ (b18 & ~b17); \
    A17 = b17 ^ (b19 & ~b18); \
    A18 = b18 ^ (b15 & ~b19); \
    A19 = b19 ^ (b16 & ~b15); \
    \
    A20 = b20 ^ (b22 & ~b21); \
    A21 = b21 ^ (b23 & ~b22); \
    A22 = b22 ^ (b24 & ~b23); \
    A23 = b23 ^ (b20 & ~b24); \
    A24 = b24 ^ (b21 & ~b20); \
} while(0)

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

    if (n_lanes == 2u) {
        device const ulong2 *seeds_v = (device const ulong2 *)seeds;
        ulong2 seed_val = seeds_v[idx];
        
        ulong A00 = seed_val.x;
        ulong A01 = seed_val.y;
        ulong A02, A03, A04, A05, A06, A07, A08, A09;
        ulong A10, A11, A12, A13, A14, A15, A16, A17, A18, A19;
        ulong A20, A21, A22, A23, A24;

        for (uint step = 0u; step < w_val; ++step) {
            A02 = 0x06ul;
            A03 = 0ul; A04 = 0ul;
            A05 = 0ul; A06 = 0ul; A07 = 0ul; A08 = 0ul; A09 = 0ul;
            A10 = 0ul; A11 = 0ul; A12 = 0ul; A13 = 0ul; A14 = 0ul;
            A15 = 0ul; 
            A16 = 0x8000000000000000ul;
            A17 = 0ul; A18 = 0ul; A19 = 0ul;
            A20 = 0ul; A21 = 0ul; A22 = 0ul; A23 = 0ul; A24 = 0ul;

            #pragma unroll
            for (uint r = 0u; r < 24u; ++r) {
                K_ROUND(KECCAK_RC[r]);
            }
        }
        
        device ulong2 *tips_v = (device ulong2 *)tips;
        tips_v[idx] = ulong2(A00, A01);

    } else if (n_lanes == 4u) {
        device const ulong4 *seeds_v = (device const ulong4 *)seeds;
        ulong4 seed_val = seeds_v[idx];
        
        ulong A00 = seed_val.x;
        ulong A01 = seed_val.y;
        ulong A02 = seed_val.z;
        ulong A03 = seed_val.w;
        ulong A04, A05, A06, A07, A08, A09;
        ulong A10, A11, A12, A13, A14, A15, A16, A17, A18, A19;
        ulong A20, A21, A22, A23, A24;

        for (uint step = 0u; step < w_val; ++step) {
            A04 = 0x06ul;
            A05 = 0ul; A06 = 0ul; A07 = 0ul; A08 = 0ul; A09 = 0ul;
            A10 = 0ul; A11 = 0ul; A12 = 0ul; A13 = 0ul; A14 = 0ul;
            A15 = 0ul; 
            A16 = 0x8000000000000000ul;
            A17 = 0ul; A18 = 0ul; A19 = 0ul;
            A20 = 0ul; A21 = 0ul; A22 = 0ul; A23 = 0ul; A24 = 0ul;

            #pragma unroll
            for (uint r = 0u; r < 24u; ++r) {
                K_ROUND(KECCAK_RC[r]);
            }
        }
        
        device ulong4 *tips_v = (device ulong4 *)tips;
        tips_v[idx] = ulong4(A00, A01, A02, A03);

    } else {
        ulong A00=0, A01=0, A02=0, A03=0, A04=0, A05=0, A06=0, A07=0, A08=0, A09=0;
        ulong A10=0, A11=0, A12=0, A13=0, A14=0, A15=0, A16=0, A17=0, A18=0, A19=0;
        ulong A20=0, A21=0, A22=0, A23=0, A24=0;

        uint base = idx * n_lanes;
        if (n_lanes > 0) A00 = seeds[base + 0];
        if (n_lanes > 1) A01 = seeds[base + 1];
        if (n_lanes > 2) A02 = seeds[base + 2];
        if (n_lanes > 3) A03 = seeds[base + 3];
        if (n_lanes > 4) A04 = seeds[base + 4];
        if (n_lanes > 5) A05 = seeds[base + 5];
        if (n_lanes > 6) A06 = seeds[base + 6];
        if (n_lanes > 7) A07 = seeds[base + 7];
        if (n_lanes > 8) A08 = seeds[base + 8];
        if (n_lanes > 9) A09 = seeds[base + 9];
        if (n_lanes > 10) A10 = seeds[base + 10];
        if (n_lanes > 11) A11 = seeds[base + 11];
        if (n_lanes > 12) A12 = seeds[base + 12];
        if (n_lanes > 13) A13 = seeds[base + 13];
        if (n_lanes > 14) A14 = seeds[base + 14];
        if (n_lanes > 15) A15 = seeds[base + 15];
        if (n_lanes > 16) A16 = seeds[base + 16];

        for (uint step = 0u; step < w_val; ++step) {
            if (step > 0u) {
                if (n_lanes <= 0) A00 = 0ul;
                if (n_lanes <= 1) A01 = 0ul;
                if (n_lanes <= 2) A02 = 0ul;
                if (n_lanes <= 3) A03 = 0ul;
                if (n_lanes <= 4) A04 = 0ul;
                if (n_lanes <= 5) A05 = 0ul;
                if (n_lanes <= 6) A06 = 0ul;
                if (n_lanes <= 7) A07 = 0ul;
                if (n_lanes <= 8) A08 = 0ul;
                if (n_lanes <= 9) A09 = 0ul;
                if (n_lanes <= 10) A10 = 0ul;
                if (n_lanes <= 11) A11 = 0ul;
                if (n_lanes <= 12) A12 = 0ul;
                if (n_lanes <= 13) A13 = 0ul;
                if (n_lanes <= 14) A14 = 0ul;
                if (n_lanes <= 15) A15 = 0ul;
                if (n_lanes <= 16) A16 = 0ul;
                A17 = 0ul; A18 = 0ul; A19 = 0ul;
                A20 = 0ul; A21 = 0ul; A22 = 0ul; A23 = 0ul; A24 = 0ul;
            }

            if (n_lanes == 0) A00 ^= 0x06ul;
            else if (n_lanes == 1) A01 ^= 0x06ul;
            else if (n_lanes == 2) A02 ^= 0x06ul;
            else if (n_lanes == 3) A03 ^= 0x06ul;
            else if (n_lanes == 4) A04 ^= 0x06ul;
            else if (n_lanes == 5) A05 ^= 0x06ul;
            else if (n_lanes == 6) A06 ^= 0x06ul;
            else if (n_lanes == 7) A07 ^= 0x06ul;
            else if (n_lanes == 8) A08 ^= 0x06ul;
            else if (n_lanes == 9) A09 ^= 0x06ul;
            else if (n_lanes == 10) A10 ^= 0x06ul;
            else if (n_lanes == 11) A11 ^= 0x06ul;
            else if (n_lanes == 12) A12 ^= 0x06ul;
            else if (n_lanes == 13) A13 ^= 0x06ul;
            else if (n_lanes == 14) A14 ^= 0x06ul;
            else if (n_lanes == 15) A15 ^= 0x06ul;
            else if (n_lanes == 16) A16 ^= 0x06ul;

            A16 ^= 0x8000000000000000ul;

            #pragma unroll
            for (uint r = 0u; r < 24u; ++r) {
                K_ROUND(KECCAK_RC[r]);
            }
        }

        if (n_lanes > 0) tips[base + 0] = A00;
        if (n_lanes > 1) tips[base + 1] = A01;
        if (n_lanes > 2) tips[base + 2] = A02;
        if (n_lanes > 3) tips[base + 3] = A03;
        if (n_lanes > 4) tips[base + 4] = A04;
        if (n_lanes > 5) tips[base + 5] = A05;
        if (n_lanes > 6) tips[base + 6] = A06;
        if (n_lanes > 7) tips[base + 7] = A07;
        if (n_lanes > 8) tips[base + 8] = A08;
        if (n_lanes > 9) tips[base + 9] = A09;
        if (n_lanes > 10) tips[base + 10] = A10;
        if (n_lanes > 11) tips[base + 11] = A11;
        if (n_lanes > 12) tips[base + 12] = A12;
        if (n_lanes > 13) tips[base + 13] = A13;
        if (n_lanes > 14) tips[base + 14] = A14;
        if (n_lanes > 15) tips[base + 15] = A15;
        if (n_lanes > 16) tips[base + 16] = A16;
    }
}
```