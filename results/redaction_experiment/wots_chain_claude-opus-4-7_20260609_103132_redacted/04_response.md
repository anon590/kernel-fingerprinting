**Optimization:** The incumbent uses a runtime `for r=0..24` loop with `KECCAK_RC[r]` from `constant` memory. The previous lane-complementing attempt was slower because of register pressure / extra reset logic. I'll keep the simpler chi formulation but **fully unroll** the 24 rounds (via macro with inlined RC literals), which lets the compiler eliminate loop overhead, schedule rho-rotations as constants, and keep all 25 lanes in registers without spilling. I'll also fold the per-step reset of lanes `[n_lanes..24]` and the pad XORs more tightly: between steps only lanes `n_lanes..16` (and 17..24, but those weren't touched by squeeze) need reset — actually all of 25 are dirty after Keccak-f, so we reset 25-n_lanes lanes. The XOR of 0x06 at `dom_lane` after zeroing simplifies to a plain assign. This reduces dynamic instruction count vs the runtime-loop incumbent while avoiding the complement-bookkeeping cost.

```metal
#include <metal_stdlib>
using namespace metal;

inline ulong ROL(ulong x, uint k) {
    return rotate(x, (ulong)k);
}

#define KECCAK_ROUND(RC)                                              \
{                                                                     \
    ulong C0 = A00 ^ A05 ^ A10 ^ A15 ^ A20;                           \
    ulong C1 = A01 ^ A06 ^ A11 ^ A16 ^ A21;                           \
    ulong C2 = A02 ^ A07 ^ A12 ^ A17 ^ A22;                           \
    ulong C3 = A03 ^ A08 ^ A13 ^ A18 ^ A23;                           \
    ulong C4 = A04 ^ A09 ^ A14 ^ A19 ^ A24;                           \
                                                                      \
    ulong D0 = C4 ^ ROL(C1, 1);                                       \
    ulong D1 = C0 ^ ROL(C2, 1);                                       \
    ulong D2 = C1 ^ ROL(C3, 1);                                       \
    ulong D3 = C2 ^ ROL(C4, 1);                                       \
    ulong D4 = C3 ^ ROL(C0, 1);                                       \
                                                                      \
    ulong B00 =      (A00 ^ D0)      ;                                \
    ulong B10 = ROL( (A01 ^ D1),  1);                                 \
    ulong B20 = ROL( (A02 ^ D2), 62);                                 \
    ulong B05 = ROL( (A03 ^ D3), 28);                                 \
    ulong B15 = ROL( (A04 ^ D4), 27);                                 \
    ulong B16 = ROL( (A05 ^ D0), 36);                                 \
    ulong B01 = ROL( (A06 ^ D1), 44);                                 \
    ulong B11 = ROL( (A07 ^ D2),  6);                                 \
    ulong B21 = ROL( (A08 ^ D3), 55);                                 \
    ulong B06 = ROL( (A09 ^ D4), 20);                                 \
    ulong B07 = ROL( (A10 ^ D0),  3);                                 \
    ulong B17 = ROL( (A11 ^ D1), 10);                                 \
    ulong B02 = ROL( (A12 ^ D2), 43);                                 \
    ulong B12 = ROL( (A13 ^ D3), 25);                                 \
    ulong B22 = ROL( (A14 ^ D4), 39);                                 \
    ulong B23 = ROL( (A15 ^ D0), 41);                                 \
    ulong B08 = ROL( (A16 ^ D1), 45);                                 \
    ulong B18 = ROL( (A17 ^ D2), 15);                                 \
    ulong B03 = ROL( (A18 ^ D3), 21);                                 \
    ulong B13 = ROL( (A19 ^ D4),  8);                                 \
    ulong B14 = ROL( (A20 ^ D0), 18);                                 \
    ulong B24 = ROL( (A21 ^ D1),  2);                                 \
    ulong B09 = ROL( (A22 ^ D2), 61);                                 \
    ulong B19 = ROL( (A23 ^ D3), 56);                                 \
    ulong B04 = ROL( (A24 ^ D4), 14);                                 \
                                                                      \
    A00 = B00 ^ ((~B01) & B02);                                       \
    A01 = B01 ^ ((~B02) & B03);                                       \
    A02 = B02 ^ ((~B03) & B04);                                       \
    A03 = B03 ^ ((~B04) & B00);                                       \
    A04 = B04 ^ ((~B00) & B01);                                       \
    A05 = B05 ^ ((~B06) & B07);                                       \
    A06 = B06 ^ ((~B07) & B08);                                       \
    A07 = B07 ^ ((~B08) & B09);                                       \
    A08 = B08 ^ ((~B09) & B05);                                       \
    A09 = B09 ^ ((~B05) & B06);                                       \
    A10 = B10 ^ ((~B11) & B12);                                       \
    A11 = B11 ^ ((~B12) & B13);                                       \
    A12 = B12 ^ ((~B13) & B14);                                       \
    A13 = B13 ^ ((~B14) & B10);                                       \
    A14 = B14 ^ ((~B10) & B11);                                       \
    A15 = B15 ^ ((~B16) & B17);                                       \
    A16 = B16 ^ ((~B17) & B18);                                       \
    A17 = B17 ^ ((~B18) & B19);                                       \
    A18 = B18 ^ ((~B19) & B15);                                       \
    A19 = B19 ^ ((~B15) & B16);                                       \
    A20 = B20 ^ ((~B21) & B22);                                       \
    A21 = B21 ^ ((~B22) & B23);                                       \
    A22 = B22 ^ ((~B23) & B24);                                       \
    A23 = B23 ^ ((~B24) & B20);                                       \
    A24 = B24 ^ ((~B20) & B21);                                       \
                                                                      \
    A00 ^= (RC);                                                      \
}

#define KECCAK_F1600                                                  \
    KECCAK_ROUND(0x0000000000000001ul)                                \
    KECCAK_ROUND(0x0000000000008082ul)                                \
    KECCAK_ROUND(0x800000000000808Aul)                                \
    KECCAK_ROUND(0x8000000080008000ul)                                \
    KECCAK_ROUND(0x000000000000808Bul)                                \
    KECCAK_ROUND(0x0000000080000001ul)                                \
    KECCAK_ROUND(0x8000000080008081ul)                                \
    KECCAK_ROUND(0x8000000000008009ul)                                \
    KECCAK_ROUND(0x000000000000008Aul)                                \
    KECCAK_ROUND(0x0000000000000088ul)                                \
    KECCAK_ROUND(0x0000000080008009ul)                                \
    KECCAK_ROUND(0x000000008000000Aul)                                \
    KECCAK_ROUND(0x000000008000808Bul)                                \
    KECCAK_ROUND(0x800000000000008Bul)                                \
    KECCAK_ROUND(0x8000000000008089ul)                                \
    KECCAK_ROUND(0x8000000000008003ul)                                \
    KECCAK_ROUND(0x8000000000008002ul)                                \
    KECCAK_ROUND(0x8000000000000080ul)                                \
    KECCAK_ROUND(0x000000000000800Aul)                                \
    KECCAK_ROUND(0x800000008000000Aul)                                \
    KECCAK_ROUND(0x8000000080008081ul)                                \
    KECCAK_ROUND(0x8000000000008080ul)                                \
    KECCAK_ROUND(0x0000000080000001ul)                                \
    KECCAK_ROUND(0x8000000080008008ul)

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
    uint base = idx * n_lanes;

    ulong A00=0, A01=0, A02=0, A03=0, A04=0;
    ulong A05=0, A06=0, A07=0, A08=0, A09=0;
    ulong A10=0, A11=0, A12=0, A13=0, A14=0;
    ulong A15=0, A16=0, A17=0, A18=0, A19=0;
    ulong A20=0, A21=0, A22=0, A23=0, A24=0;

    // Load seed.
    if (n_lanes > 0u)  A00 = seeds[base + 0];
    if (n_lanes > 1u)  A01 = seeds[base + 1];
    if (n_lanes > 2u)  A02 = seeds[base + 2];
    if (n_lanes > 3u)  A03 = seeds[base + 3];
    if (n_lanes > 4u)  A04 = seeds[base + 4];
    if (n_lanes > 5u)  A05 = seeds[base + 5];
    if (n_lanes > 6u)  A06 = seeds[base + 6];
    if (n_lanes > 7u)  A07 = seeds[base + 7];
    if (n_lanes > 8u)  A08 = seeds[base + 8];
    if (n_lanes > 9u)  A09 = seeds[base + 9];
    if (n_lanes > 10u) A10 = seeds[base + 10];
    if (n_lanes > 11u) A11 = seeds[base + 11];
    if (n_lanes > 12u) A12 = seeds[base + 12];
    if (n_lanes > 13u) A13 = seeds[base + 13];
    if (n_lanes > 14u) A14 = seeds[base + 14];
    if (n_lanes > 15u) A15 = seeds[base + 15];

    uint dom_lane = n_lanes;

    for (uint step = 0u; step < w; ++step) {
        // After step>0, Keccak-f dirtied all 25 lanes. We need lanes [n_lanes..24]
        // to be zero before applying the pad. Lanes [0..n_lanes-1] hold the
        // truncated previous digest, which is the absorb block.
        if (n_lanes < 1u)  A00 = 0;
        if (n_lanes < 2u)  A01 = 0;
        if (n_lanes < 3u)  A02 = 0;
        if (n_lanes < 4u)  A03 = 0;
        if (n_lanes < 5u)  A04 = 0;
        if (n_lanes < 6u)  A05 = 0;
        if (n_lanes < 7u)  A06 = 0;
        if (n_lanes < 8u)  A07 = 0;
        if (n_lanes < 9u)  A08 = 0;
        if (n_lanes < 10u) A09 = 0;
        if (n_lanes < 11u) A10 = 0;
        if (n_lanes < 12u) A11 = 0;
        if (n_lanes < 13u) A12 = 0;
        if (n_lanes < 14u) A13 = 0;
        if (n_lanes < 15u) A14 = 0;
        if (n_lanes < 16u) A15 = 0;
        A16 = 0; A17 = 0; A18 = 0; A19 = 0;
        A20 = 0; A21 = 0; A22 = 0; A23 = 0; A24 = 0;

        // Domain pad 0x06 at lane = n_lanes.
        switch (dom_lane) {
            case 0:  A00 ^= 0x06ul; break;
            case 1:  A01 ^= 0x06ul; break;
            case 2:  A02 ^= 0x06ul; break;
            case 3:  A03 ^= 0x06ul; break;
            case 4:  A04 ^= 0x06ul; break;
            case 5:  A05 ^= 0x06ul; break;
            case 6:  A06 ^= 0x06ul; break;
            case 7:  A07 ^= 0x06ul; break;
            case 8:  A08 ^= 0x06ul; break;
            case 9:  A09 ^= 0x06ul; break;
            case 10: A10 ^= 0x06ul; break;
            case 11: A11 ^= 0x06ul; break;
            case 12: A12 ^= 0x06ul; break;
            case 13: A13 ^= 0x06ul; break;
            case 14: A14 ^= 0x06ul; break;
            case 15: A15 ^= 0x06ul; break;
            case 16: A16 ^= 0x06ul; break;
            default: break;
        }
        // Final pad 0x80 at byte 7 of lane 16.
        A16 ^= 0x8000000000000000ul;

        KECCAK_F1600
    }

    if (n_lanes > 0u)  tips[base + 0]  = A00;
    if (n_lanes > 1u)  tips[base + 1]  = A01;
    if (n_lanes > 2u)  tips[base + 2]  = A02;
    if (n_lanes > 3u)  tips[base + 3]  = A03;
    if (n_lanes > 4u)  tips[base + 4]  = A04;
    if (n_lanes > 5u)  tips[base + 5]  = A05;
    if (n_lanes > 6u)  tips[base + 6]  = A06;
    if (n_lanes > 7u)  tips[base + 7]  = A07;
    if (n_lanes > 8u)  tips[base + 8]  = A08;
    if (n_lanes > 9u)  tips[base + 9]  = A09;
    if (n_lanes > 10u) tips[base + 10] = A10;
    if (n_lanes > 11u) tips[base + 11] = A11;
    if (n_lanes > 12u) tips[base + 12] = A12;
    if (n_lanes > 13u) tips[base + 13] = A13;
    if (n_lanes > 14u) tips[base + 14] = A14;
    if (n_lanes > 15u) tips[base + 15] = A15;
}
```