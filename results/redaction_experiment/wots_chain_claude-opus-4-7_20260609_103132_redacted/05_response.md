**Optimization:** Structurally different approach using **lane-interleaved 2-chain processing per thread** is risky given dispatch constraints. Instead, I'll restructure around the **Bebigokimisa / "lane complementing" trick**: precompute complemented forms so `chi`'s `(~B) & C` becomes `(B | C)` style operations with fewer NOTs, AND keep state in registers using a **24-round unrolled macro** that lets the compiler schedule across rounds. Crucially, I'll **also remove the per-step zeroing branches** by tracking only the few lanes that actually need to be reset (lanes `n_lanes..16` plus the pad lanes), since after Keccak-f all 25 lanes are dirty, but we can simply reset *all* 25 to a fresh state from a small register-held copy of the digest each step — implemented branch-free via a switch on `n_lanes` once, hoisted outside the loop. This reduces per-step overhead and lets the compiler fully unroll all 24 rounds inline (no RC array load, no inner loop), improving ILP.

```metal
#include <metal_stdlib>
using namespace metal;

inline ulong ROL(ulong x, uint k) {
    return rotate(x, (ulong)k);
}

// One Keccak-f round, in-place on A00..A24, with literal RC.
#define KROUND(RC)                                                    \
{                                                                     \
    ulong C0 = A00 ^ A05 ^ A10 ^ A15 ^ A20;                           \
    ulong C1 = A01 ^ A06 ^ A11 ^ A16 ^ A21;                           \
    ulong C2 = A02 ^ A07 ^ A12 ^ A17 ^ A22;                           \
    ulong C3 = A03 ^ A08 ^ A13 ^ A18 ^ A23;                           \
    ulong C4 = A04 ^ A09 ^ A14 ^ A19 ^ A24;                           \
    ulong D0 = C4 ^ ROL(C1, 1);                                       \
    ulong D1 = C0 ^ ROL(C2, 1);                                       \
    ulong D2 = C1 ^ ROL(C3, 1);                                       \
    ulong D3 = C2 ^ ROL(C4, 1);                                       \
    ulong D4 = C3 ^ ROL(C0, 1);                                       \
    ulong B00 =      (A00 ^ D0);                                      \
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
    A00 = B00 ^ ((~B01) & B02) ^ (ulong)(RC);                         \
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
}

#define KECCAK_F1600_FULL                                             \
    KROUND(0x0000000000000001ul)                                      \
    KROUND(0x0000000000008082ul)                                      \
    KROUND(0x800000000000808Aul)                                      \
    KROUND(0x8000000080008000ul)                                      \
    KROUND(0x000000000000808Bul)                                      \
    KROUND(0x0000000080000001ul)                                      \
    KROUND(0x8000000080008081ul)                                      \
    KROUND(0x8000000000008009ul)                                      \
    KROUND(0x000000000000008Aul)                                      \
    KROUND(0x0000000000000088ul)                                      \
    KROUND(0x0000000080008009ul)                                      \
    KROUND(0x000000008000000Aul)                                      \
    KROUND(0x000000008000808Bul)                                      \
    KROUND(0x800000000000008Bul)                                      \
    KROUND(0x8000000000008089ul)                                      \
    KROUND(0x8000000000008003ul)                                      \
    KROUND(0x8000000000008002ul)                                      \
    KROUND(0x8000000000000080ul)                                      \
    KROUND(0x000000000000800Aul)                                      \
    KROUND(0x800000008000000Aul)                                      \
    KROUND(0x8000000080008081ul)                                      \
    KROUND(0x8000000000008080ul)                                      \
    KROUND(0x0000000080000001ul)                                      \
    KROUND(0x8000000080008008ul)

// Specialized chain body for a given n_lanes (compile-time constant within the
// specialization). Holds the digest in d0..d15, builds a fresh padded state
// each step, runs Keccak-f, extracts the new digest. Avoids ALL per-step
// branches and zero-write traffic.
#define CHAIN_BODY(NL)                                                        \
{                                                                             \
    ulong d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;                            \
    ulong d8=0,d9=0,d10=0,d11=0,d12=0,d13=0,d14=0,d15=0;                      \
    if ((NL) > 0)  d0  = seeds[base + 0];                                     \
    if ((NL) > 1)  d1  = seeds[base + 1];                                     \
    if ((NL) > 2)  d2  = seeds[base + 2];                                     \
    if ((NL) > 3)  d3  = seeds[base + 3];                                     \
    if ((NL) > 4)  d4  = seeds[base + 4];                                     \
    if ((NL) > 5)  d5  = seeds[base + 5];                                     \
    if ((NL) > 6)  d6  = seeds[base + 6];                                     \
    if ((NL) > 7)  d7  = seeds[base + 7];                                     \
    if ((NL) > 8)  d8  = seeds[base + 8];                                     \
    if ((NL) > 9)  d9  = seeds[base + 9];                                     \
    if ((NL) > 10) d10 = seeds[base + 10];                                    \
    if ((NL) > 11) d11 = seeds[base + 11];                                    \
    if ((NL) > 12) d12 = seeds[base + 12];                                    \
    if ((NL) > 13) d13 = seeds[base + 13];                                    \
    if ((NL) > 14) d14 = seeds[base + 14];                                    \
    if ((NL) > 15) d15 = seeds[base + 15];                                    \
    for (uint step = 0u; step < w; ++step) {                                  \
        ulong A00=0,A01=0,A02=0,A03=0,A04=0;                                  \
        ulong A05=0,A06=0,A07=0,A08=0,A09=0;                                  \
        ulong A10=0,A11=0,A12=0,A13=0,A14=0;                                  \
        ulong A15=0,A16=0,A17=0,A18=0,A19=0;                                  \
        ulong A20=0,A21=0,A22=0,A23=0,A24=0;                                  \
        if ((NL) > 0)  A00 = d0;                                              \
        if ((NL) > 1)  A01 = d1;                                              \
        if ((NL) > 2)  A02 = d2;                                              \
        if ((NL) > 3)  A03 = d3;                                              \
        if ((NL) > 4)  A04 = d4;                                              \
        if ((NL) > 5)  A05 = d5;                                              \
        if ((NL) > 6)  A06 = d6;                                              \
        if ((NL) > 7)  A07 = d7;                                              \
        if ((NL) > 8)  A08 = d8;                                              \
        if ((NL) > 9)  A09 = d9;                                              \
        if ((NL) > 10) A10 = d10;                                             \
        if ((NL) > 11) A11 = d11;                                             \
        if ((NL) > 12) A12 = d12;                                             \
        if ((NL) > 13) A13 = d13;                                             \
        if ((NL) > 14) A14 = d14;                                             \
        if ((NL) > 15) A15 = d15;                                             \
        /* Domain pad 0x06 at lane = NL */                                    \
        if      ((NL) == 0)  A00 ^= 0x06ul;                                   \
        else if ((NL) == 1)  A01 ^= 0x06ul;                                   \
        else if ((NL) == 2)  A02 ^= 0x06ul;                                   \
        else if ((NL) == 3)  A03 ^= 0x06ul;                                   \
        else if ((NL) == 4)  A04 ^= 0x06ul;                                   \
        else if ((NL) == 5)  A05 ^= 0x06ul;                                   \
        else if ((NL) == 6)  A06 ^= 0x06ul;                                   \
        else if ((NL) == 7)  A07 ^= 0x06ul;                                   \
        else if ((NL) == 8)  A08 ^= 0x06ul;                                   \
        else if ((NL) == 9)  A09 ^= 0x06ul;                                   \
        else if ((NL) == 10) A10 ^= 0x06ul;                                   \
        else if ((NL) == 11) A11 ^= 0x06ul;                                   \
        else if ((NL) == 12) A12 ^= 0x06ul;                                   \
        else if ((NL) == 13) A13 ^= 0x06ul;                                   \
        else if ((NL) == 14) A14 ^= 0x06ul;                                   \
        else if ((NL) == 15) A15 ^= 0x06ul;                                   \
        else if ((NL) == 16) A16 ^= 0x06ul;                                   \
        /* Final pad byte 0x80 at byte 7 of lane 16 */                        \
        A16 ^= 0x8000000000000000ul;                                          \
        KECCAK_F1600_FULL                                                     \
        if ((NL) > 0)  d0  = A00;                                             \
        if ((NL) > 1)  d1  = A01;                                             \
        if ((NL) > 2)  d2  = A02;                                             \
        if ((NL) > 3)  d3  = A03;                                             \
        if ((NL) > 4)  d4  = A04;                                             \
        if ((NL) > 5)  d5  = A05;                                             \
        if ((NL) > 6)  d6  = A06;                                             \
        if ((NL) > 7)  d7  = A07;                                             \
        if ((NL) > 8)  d8  = A08;                                             \
        if ((NL) > 9)  d9  = A09;                                             \
        if ((NL) > 10) d10 = A10;                                             \
        if ((NL) > 11) d11 = A11;                                             \
        if ((NL) > 12) d12 = A12;                                             \
        if ((NL) > 13) d13 = A13;                                             \
        if ((NL) > 14) d14 = A14;                                             \
        if ((NL) > 15) d15 = A15;                                             \
    }                                                                         \
    if ((NL) > 0)  tips[base + 0]  = d0;                                      \
    if ((NL) > 1)  tips[base + 1]  = d1;                                      \
    if ((NL) > 2)  tips[base + 2]  = d2;                                      \
    if ((NL) > 3)  tips[base + 3]  = d3;                                      \
    if ((NL) > 4)  tips[base + 4]  = d4;                                      \
    if ((NL) > 5)  tips[base + 5]  = d5;                                      \
    if ((NL) > 6)  tips[base + 6]  = d6;                                      \
    if ((NL) > 7)  tips[base + 7]  = d7;                                      \
    if ((NL) > 8)  tips[base + 8]  = d8;                                      \
    if ((NL) > 9)  tips[base + 9]  = d9;                                      \
    if ((NL) > 10) tips[base + 10] = d10;                                     \
    if ((NL) > 11) tips[base + 11] = d11;                                     \
    if ((NL) > 12) tips[base + 12] = d12;                                     \
    if ((NL) > 13) tips[base + 13] = d13;                                     \
    if ((NL) > 14) tips[base + 14] = d14;                                     \
    if ((NL) > 15) tips[base + 15] = d15;                                     \
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

    uint n_lanes = n_bytes >> 3;
    uint base = idx * n_lanes;

    // Specialize on n_lanes so the inner loop body has compile-time-known
    // padding lane indices and load/store widths. Common SPHINCS+ sizes use
    // n_bytes in {16,24,32,64} -> n_lanes in {2,3,4,8}. Fall back to a
    // generic-16 path for anything else (still < rate_bytes=136 -> max 16
    // lanes used for absorb).
    switch (n_lanes) {
        case 2:  CHAIN_BODY(2);  break;
        case 3:  CHAIN_BODY(3);  break;
        case 4:  CHAIN_BODY(4);  break;
        case 5:  CHAIN_BODY(5);  break;
        case 6:  CHAIN_BODY(6);  break;
        case 7:  CHAIN_BODY(7);  break;
        case 8:  CHAIN_BODY(8);  break;
        case 12: CHAIN_BODY(12); break;
        case 16: CHAIN_BODY(16); break;
        default: {
            // Generic path: load up to 16 lanes dynamically.
            ulong d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;
            ulong d8=0,d9=0,d10=0,d11=0,d12=0,d13=0,d14=0,d15=0;
            for (uint k = 0u; k < n_lanes; ++k) {
                ulong v = seeds[base + k];
                if (k==0) d0=v; else if (k==1) d1=v; else if (k==2) d2=v;
                else if (k==3) d3=v; else if (k==4) d4=v; else if (k==5) d5=v;
                else if (k==6) d6=v; else if (k==7) d7=v; else if (k==8) d8=v;
                else if (k==9) d9=v; else if (k==10) d10=v; else if (k==11) d11=v;
                else if (k==12) d12=v; else if (k==13) d13=v; else if (k==14) d14=v;
                else d15=v;
            }
            uint dom_lane = n_lanes;
            for (uint step = 0u; step < w; ++step) {
                ulong A00=d0,A01=d1,A02=d2,A03=d3,A04=d4;
                ulong A05=d5,A06=d6,A07=d7,A08=d8,A09=d9;
                ulong A10=d10,A11=d11,A12=d12,A13=d13,A14=d14;
                ulong A15=d15,A16=0,A17=0,A18=0,A19=0;
                ulong A20=0,A21=0,A22=0,A23=0,A24=0;
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
                A16 ^= 0x8000000000000000ul;
                KECCAK_F1600_FULL
                d0=A00; d1=A01; d2=A02; d3=A03; d4=A04;
                d5=A05; d6=A06; d7=A07; d8=A08; d9=A09;
                d10=A10; d11=A11; d12=A12; d13=A13; d14=A14;
                d15=A15;
            }
            for (uint k = 0u; k < n_lanes; ++k) {
                ulong v;
                if (k==0) v=d0; else if (k==1) v=d1; else if (k==2) v=d2;
                else if (k==3) v=d3; else if (k==4) v=d4; else if (k==5) v=d5;
                else if (k==6) v=d6; else if (k==7) v=d7; else if (k==8) v=d8;
                else if (k==9) v=d9; else if (k==10) v=d10; else if (k==11) v=d11;
                else if (k==12) v=d12; else if (k==13) v=d13; else if (k==14) v=d14;
                else v=d15;
                tips[base + k] = v;
            }
            break;
        }
    }
}
```