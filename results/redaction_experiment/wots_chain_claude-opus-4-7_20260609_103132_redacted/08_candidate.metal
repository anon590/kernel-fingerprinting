#include <metal_stdlib>
using namespace metal;

constant ulong KECCAK_RC[24] = {
    0x0000000000000001ul, 0x0000000000008082ul, 0x800000000000808Aul,
    0x8000000080008000ul, 0x000000000000808Bul, 0x0000000080000001ul,
    0x8000000080008081ul, 0x8000000000008009ul, 0x000000000000008Aul,
    0x0000000000000088ul, 0x0000000080008009ul, 0x000000008000000Aul,
    0x000000008000808Bul, 0x800000000000008Bul, 0x8000000000008089ul,
    0x8000000000008003ul, 0x8000000000008002ul, 0x8000000000000080ul,
    0x000000000000800Aul, 0x800000008000000Aul, 0x8000000080008081ul,
    0x8000000000008080ul, 0x0000000080000001ul, 0x8000000080008008ul,
};

#define ROL(x, k) rotate((ulong)(x), (ulong)(k))

// One Keccak round, in-place on 25 named lanes. We pass via macro to keep
// everything in registers and let the compiler schedule across rounds.
#define KECCAK_ROUND(RC)                                                    \
    {                                                                       \
        ulong C0 = A00 ^ A05 ^ A10 ^ A15 ^ A20;                             \
        ulong C1 = A01 ^ A06 ^ A11 ^ A16 ^ A21;                             \
        ulong C2 = A02 ^ A07 ^ A12 ^ A17 ^ A22;                             \
        ulong C3 = A03 ^ A08 ^ A13 ^ A18 ^ A23;                             \
        ulong C4 = A04 ^ A09 ^ A14 ^ A19 ^ A24;                             \
        ulong D0 = C4 ^ ROL(C1, 1);                                         \
        ulong D1 = C0 ^ ROL(C2, 1);                                         \
        ulong D2 = C1 ^ ROL(C3, 1);                                         \
        ulong D3 = C2 ^ ROL(C4, 1);                                         \
        ulong D4 = C3 ^ ROL(C0, 1);                                         \
        ulong T00 =      (A00 ^ D0);                                        \
        ulong T10 = ROL( (A01 ^ D1),  1);                                   \
        ulong T20 = ROL( (A02 ^ D2), 62);                                   \
        ulong T05 = ROL( (A03 ^ D3), 28);                                   \
        ulong T15 = ROL( (A04 ^ D4), 27);                                   \
        ulong T16 = ROL( (A05 ^ D0), 36);                                   \
        ulong T01 = ROL( (A06 ^ D1), 44);                                   \
        ulong T11 = ROL( (A07 ^ D2),  6);                                   \
        ulong T21 = ROL( (A08 ^ D3), 55);                                   \
        ulong T06 = ROL( (A09 ^ D4), 20);                                   \
        ulong T07 = ROL( (A10 ^ D0),  3);                                   \
        ulong T17 = ROL( (A11 ^ D1), 10);                                   \
        ulong T02 = ROL( (A12 ^ D2), 43);                                   \
        ulong T12 = ROL( (A13 ^ D3), 25);                                   \
        ulong T22 = ROL( (A14 ^ D4), 39);                                   \
        ulong T23 = ROL( (A15 ^ D0), 41);                                   \
        ulong T08 = ROL( (A16 ^ D1), 45);                                   \
        ulong T18 = ROL( (A17 ^ D2), 15);                                   \
        ulong T03 = ROL( (A18 ^ D3), 21);                                   \
        ulong T13 = ROL( (A19 ^ D4),  8);                                   \
        ulong T14 = ROL( (A20 ^ D0), 18);                                   \
        ulong T24 = ROL( (A21 ^ D1),  2);                                   \
        ulong T09 = ROL( (A22 ^ D2), 61);                                   \
        ulong T19 = ROL( (A23 ^ D3), 56);                                   \
        ulong T04 = ROL( (A24 ^ D4), 14);                                   \
        A00 = T00 ^ ((~T01) & T02) ^ (RC);                                  \
        A01 = T01 ^ ((~T02) & T03);                                         \
        A02 = T02 ^ ((~T03) & T04);                                         \
        A03 = T03 ^ ((~T04) & T00);                                         \
        A04 = T04 ^ ((~T00) & T01);                                         \
        A05 = T05 ^ ((~T06) & T07);                                         \
        A06 = T06 ^ ((~T07) & T08);                                         \
        A07 = T07 ^ ((~T08) & T09);                                         \
        A08 = T08 ^ ((~T09) & T05);                                         \
        A09 = T09 ^ ((~T05) & T06);                                         \
        A10 = T10 ^ ((~T11) & T12);                                         \
        A11 = T11 ^ ((~T12) & T13);                                         \
        A12 = T12 ^ ((~T13) & T14);                                         \
        A13 = T13 ^ ((~T14) & T10);                                         \
        A14 = T14 ^ ((~T10) & T11);                                         \
        A15 = T15 ^ ((~T16) & T17);                                         \
        A16 = T16 ^ ((~T17) & T18);                                         \
        A17 = T17 ^ ((~T18) & T19);                                         \
        A18 = T18 ^ ((~T19) & T15);                                         \
        A19 = T19 ^ ((~T15) & T16);                                         \
        A20 = T20 ^ ((~T21) & T22);                                         \
        A21 = T21 ^ ((~T22) & T23);                                         \
        A22 = T22 ^ ((~T23) & T24);                                         \
        A23 = T23 ^ ((~T24) & T20);                                         \
        A24 = T24 ^ ((~T20) & T21);                                         \
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

    const uint n_lanes = n_bytes >> 3;
    const uint base = idx * n_lanes;
    const uint dom_lane = n_lanes;
    const uint W = w;

    // Load seed into 25 named lanes, zero rest.
    ulong A00=0, A01=0, A02=0, A03=0, A04=0;
    ulong A05=0, A06=0, A07=0, A08=0, A09=0;
    ulong A10=0, A11=0, A12=0, A13=0, A14=0;
    ulong A15=0, A16=0, A17=0, A18=0, A19=0;
    ulong A20=0, A21=0, A22=0, A23=0, A24=0;

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

    // Build padded-state XOR mask once. The padded block for SHA3-256 with
    // an n_bytes-byte message is: msg || 0x06 || 0x00... || 0x80 at byte
    // rate-1 (= byte 7 of lane 16). All other rate/capacity lanes are 0.
    // Since on every step the input lanes are exactly the prior digest
    // truncated to n_lanes lanes and the rest of the state is fully
    // determined by Keccak-f output, we re-zero capacity+upper-rate each
    // step before re-XORing the pad mask.
    //
    // KEY STRUCTURAL CHANGE: precompute pad XOR per lane index outside the
    // inner round loop, and apply with a single XOR to the lane register.

    ulong pad_dom_00 = (dom_lane ==  0u) ? 0x06ul : 0ul;
    ulong pad_dom_01 = (dom_lane ==  1u) ? 0x06ul : 0ul;
    ulong pad_dom_02 = (dom_lane ==  2u) ? 0x06ul : 0ul;
    ulong pad_dom_03 = (dom_lane ==  3u) ? 0x06ul : 0ul;
    ulong pad_dom_04 = (dom_lane ==  4u) ? 0x06ul : 0ul;
    ulong pad_dom_05 = (dom_lane ==  5u) ? 0x06ul : 0ul;
    ulong pad_dom_06 = (dom_lane ==  6u) ? 0x06ul : 0ul;
    ulong pad_dom_07 = (dom_lane ==  7u) ? 0x06ul : 0ul;
    ulong pad_dom_08 = (dom_lane ==  8u) ? 0x06ul : 0ul;
    ulong pad_dom_09 = (dom_lane ==  9u) ? 0x06ul : 0ul;
    ulong pad_dom_10 = (dom_lane == 10u) ? 0x06ul : 0ul;
    ulong pad_dom_11 = (dom_lane == 11u) ? 0x06ul : 0ul;
    ulong pad_dom_12 = (dom_lane == 12u) ? 0x06ul : 0ul;
    ulong pad_dom_13 = (dom_lane == 13u) ? 0x06ul : 0ul;
    ulong pad_dom_14 = (dom_lane == 14u) ? 0x06ul : 0ul;
    ulong pad_dom_15 = (dom_lane == 15u) ? 0x06ul : 0ul;
    // dom_lane==16 case folded into pad_fin_16 below.
    ulong pad_dom_16 = (dom_lane == 16u) ? 0x06ul : 0ul;
    const ulong pad_fin_16 = 0x8000000000000000ul ^ pad_dom_16;

    // The first absorb's lanes 0..n_lanes-1 already hold the seed; we add
    // pad XOR to each of those lanes once at the start of each step (only
    // pad_dom_<dom_lane> is nonzero, others are zero, so XORing all is OK,
    // but we only need to XOR the one nonzero one, plus lane 16's final
    // pad). We just XOR the precomputed values into the lanes; identity
    // for zero values.

    for (uint step = 0u; step < W; ++step) {
        // Zero out lanes that were not the message lanes from the prior
        // step (or, on first step, the unused part of rate + capacity).
        // After Keccak-f on step >= 1, lanes n_lanes..24 hold output we
        // need to discard before next absorb; lanes 0..n_lanes-1 hold the
        // truncated digest already.
        if (n_lanes <  1u) A00 = 0;
        if (n_lanes <  2u) A01 = 0;
        if (n_lanes <  3u) A02 = 0;
        if (n_lanes <  4u) A03 = 0;
        if (n_lanes <  5u) A04 = 0;
        if (n_lanes <  6u) A05 = 0;
        if (n_lanes <  7u) A06 = 0;
        if (n_lanes <  8u) A07 = 0;
        if (n_lanes <  9u) A08 = 0;
        if (n_lanes < 10u) A09 = 0;
        if (n_lanes < 11u) A10 = 0;
        if (n_lanes < 12u) A11 = 0;
        if (n_lanes < 13u) A12 = 0;
        if (n_lanes < 14u) A13 = 0;
        if (n_lanes < 15u) A14 = 0;
        if (n_lanes < 16u) A15 = 0;
        A16 = pad_fin_16;
        A17 = 0; A18 = 0; A19 = 0;
        A20 = 0; A21 = 0; A22 = 0; A23 = 0; A24 = 0;

        // Apply domain pad. pad_dom_k is nonzero only for k == dom_lane.
        A00 ^= pad_dom_00;
        A01 ^= pad_dom_01;
        A02 ^= pad_dom_02;
        A03 ^= pad_dom_03;
        A04 ^= pad_dom_04;
        A05 ^= pad_dom_05;
        A06 ^= pad_dom_06;
        A07 ^= pad_dom_07;
        A08 ^= pad_dom_08;
        A09 ^= pad_dom_09;
        A10 ^= pad_dom_10;
        A11 ^= pad_dom_11;
        A12 ^= pad_dom_12;
        A13 ^= pad_dom_13;
        A14 ^= pad_dom_14;
        A15 ^= pad_dom_15;
        // A16 already includes pad_dom_16 if applicable.

        // Fully unroll 24 rounds.
        KECCAK_ROUND(KECCAK_RC[ 0]);
        KECCAK_ROUND(KECCAK_RC[ 1]);
        KECCAK_ROUND(KECCAK_RC[ 2]);
        KECCAK_ROUND(KECCAK_RC[ 3]);
        KECCAK_ROUND(KECCAK_RC[ 4]);
        KECCAK_ROUND(KECCAK_RC[ 5]);
        KECCAK_ROUND(KECCAK_RC[ 6]);
        KECCAK_ROUND(KECCAK_RC[ 7]);
        KECCAK_ROUND(KECCAK_RC[ 8]);
        KECCAK_ROUND(KECCAK_RC[ 9]);
        KECCAK_ROUND(KECCAK_RC[10]);
        KECCAK_ROUND(KECCAK_RC[11]);
        KECCAK_ROUND(KECCAK_RC[12]);
        KECCAK_ROUND(KECCAK_RC[13]);
        KECCAK_ROUND(KECCAK_RC[14]);
        KECCAK_ROUND(KECCAK_RC[15]);
        KECCAK_ROUND(KECCAK_RC[16]);
        KECCAK_ROUND(KECCAK_RC[17]);
        KECCAK_ROUND(KECCAK_RC[18]);
        KECCAK_ROUND(KECCAK_RC[19]);
        KECCAK_ROUND(KECCAK_RC[20]);
        KECCAK_ROUND(KECCAK_RC[21]);
        KECCAK_ROUND(KECCAK_RC[22]);
        KECCAK_ROUND(KECCAK_RC[23]);
    }

    // Squeeze: first n_lanes lanes hold the digest truncated chain tip.
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