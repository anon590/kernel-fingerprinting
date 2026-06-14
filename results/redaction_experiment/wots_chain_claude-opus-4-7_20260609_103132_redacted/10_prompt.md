## Task: wots_chain

Batched WOTS+ / SPHINCS+-style hash chains. Given ``n_chains`` independent ``n_bytes``-byte seeds, apply the Keccak-256 inner hash ``w`` times in sequence per chain (each digest truncated to ``n_bytes`` bytes before feeding into the next iteration) and write the chain tip to the output. The chains are embarrassingly parallel; the ``w``-step iteration along each chain is strictly sequential.

Inner hash: Keccak-f[1600] with the FIPS 202 SHA3-256 sponge framing -- rate = 136 bytes (17 lanes), capacity = 64 bytes, domain pad byte = 0x06. State convention: the 1600-bit state is a 5x5 array of 64-bit lanes; lane k = x + 5*y holds bytes 8*k .. 8*k + 7 of the sponge state in little-endian.

All test sizes have ``n_bytes < rate_bytes`` (``n_bytes`` is bound at runtime and varies across the configurations the kernel is scored on; rate_bytes=136), so every chain step collapses to a single-block absorb + single-block squeeze of ``n_lanes = n_bytes / 8`` state lanes:
  state                          := 0
  state[lane 0..n_lanes-1]       := previous_chunk
  state[lane n_lanes, byte 0]    ^= 0x06   # SHA3 domain
  state[lane 16, byte 7]         ^= 0x80   # FIPS 202 final pad
  state                          := Keccak-f1600(state)
  next_chunk                     := state[lane 0..n_lanes-1]

On the first chain step the absorb is the seed; on every subsequent step the absorb is the n_lanes-lane truncation of the previous Keccak-f1600 output. After ``w`` steps the first n_lanes state lanes are written to the output as the chain tip.

The kernel must read ``n_bytes`` and ``w`` from the bound device buffers rather than treating them as compile-time constants; both vary across the test sizes (``w`` in {16, 64, 256} among the baseline sizes shown; both ``w`` and ``n_bytes`` are bound at runtime and vary across the configurations the kernel is scored on). Hardcoding either value silently produces wrong output, not just slow output.

Correctness is bit-exact against ``hashlib.sha3_256`` iterated ``w`` times with ``n_bytes``-byte truncation; any mismatched output ulong rejects the candidate.

## Required kernel signature(s)

```
kernel void wots_chain(
    device const ulong *seeds    [[buffer(0)]],
    device       ulong *tips     [[buffer(1)]],
    constant uint      &n_chains [[buffer(2)]],
    constant uint      &n_bytes  [[buffer(3)]],
    constant uint      &w        [[buffer(4)]],
    uint idx [[thread_position_in_grid]]);

Dispatch (host-fixed):
  threadsPerGrid        = (n_chains, 1, 1)
  threadsPerThreadgroup = (min(n_chains, 64), 1, 1)
Each thread processes ONE chain end-to-end; guard against idx >= n_chains (the grid is rounded up to a multiple of the TG width). ``seeds`` is laid out as n_chains consecutive runs of ``n_bytes / 8`` ulongs; ``tips`` likewise. The external buffer layout above must be preserved and the per-chain sequential semantics honored: each chain's step ``j+1`` must read the digest produced by its own step ``j`` (cross-chain mixing of intermediate digests would be a correctness bug).
```

## Your previous attempt

```metal
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

// Lane-complementing mask (Bertoni et al.): we keep these lanes in
// complemented form throughout the permutation. The initial state (all
// zero before pad) is XORed with this mask; we run the permutation with
// modified chi that omits NOTs on complemented lanes; we XOR the mask
// back at the end to recover the true state.
// Standard complemented lanes for Keccak-f1600: 1, 2, 8, 12, 17, 20.
// Source: Bertoni et al., "Keccak implementation overview", §2.2.
// In our linear-index naming (lane k = x + 5y), those are:
//   lane 1 = (1,0), lane 2 = (2,0), lane 8 = (3,1),
//   lane 12 = (2,2), lane 17 = (2,3), lane 20 = (0,4).

#define ROL(x, k) rotate((ulong)(x), (ulong)(k))

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
    const uint W = w;
    const uint dom_lane = n_lanes;

    // Load seed lanes (n_lanes up to 16).
    ulong S00=0,S01=0,S02=0,S03=0,S04=0,S05=0,S06=0,S07=0;
    ulong S08=0,S09=0,S10=0,S11=0,S12=0,S13=0,S14=0,S15=0;
    if (n_lanes > 0u)  S00 = seeds[base + 0];
    if (n_lanes > 1u)  S01 = seeds[base + 1];
    if (n_lanes > 2u)  S02 = seeds[base + 2];
    if (n_lanes > 3u)  S03 = seeds[base + 3];
    if (n_lanes > 4u)  S04 = seeds[base + 4];
    if (n_lanes > 5u)  S05 = seeds[base + 5];
    if (n_lanes > 6u)  S06 = seeds[base + 6];
    if (n_lanes > 7u)  S07 = seeds[base + 7];
    if (n_lanes > 8u)  S08 = seeds[base + 8];
    if (n_lanes > 9u)  S09 = seeds[base + 9];
    if (n_lanes > 10u) S10 = seeds[base + 10];
    if (n_lanes > 11u) S11 = seeds[base + 11];
    if (n_lanes > 12u) S12 = seeds[base + 12];
    if (n_lanes > 13u) S13 = seeds[base + 13];
    if (n_lanes > 14u) S14 = seeds[base + 14];
    if (n_lanes > 15u) S15 = seeds[base + 15];

    // Precompute per-lane domain-pad XOR mask once.
    ulong dpad00=0,dpad01=0,dpad02=0,dpad03=0,dpad04=0,dpad05=0,dpad06=0,dpad07=0;
    ulong dpad08=0,dpad09=0,dpad10=0,dpad11=0,dpad12=0,dpad13=0,dpad14=0,dpad15=0;
    ulong dpad16=0;
    switch (dom_lane) {
        case 0:  dpad00 = 0x06ul; break;
        case 1:  dpad01 = 0x06ul; break;
        case 2:  dpad02 = 0x06ul; break;
        case 3:  dpad03 = 0x06ul; break;
        case 4:  dpad04 = 0x06ul; break;
        case 5:  dpad05 = 0x06ul; break;
        case 6:  dpad06 = 0x06ul; break;
        case 7:  dpad07 = 0x06ul; break;
        case 8:  dpad08 = 0x06ul; break;
        case 9:  dpad09 = 0x06ul; break;
        case 10: dpad10 = 0x06ul; break;
        case 11: dpad11 = 0x06ul; break;
        case 12: dpad12 = 0x06ul; break;
        case 13: dpad13 = 0x06ul; break;
        case 14: dpad14 = 0x06ul; break;
        case 15: dpad15 = 0x06ul; break;
        case 16: dpad16 = 0x06ul; break;
        default: break;
    }
    // Final pad lane 16 byte 7 = 0x80.
    const ulong fpad16 = 0x8000000000000000ul;

    // Complement-mask lanes: 1, 2, 8, 12, 17, 20.
    const ulong CMASK01 = ~0ul;
    const ulong CMASK02 = ~0ul;
    const ulong CMASK08 = ~0ul;
    const ulong CMASK12 = ~0ul;
    const ulong CMASK17 = ~0ul;
    const ulong CMASK20 = ~0ul;

    // State registers.
    ulong A00,A01,A02,A03,A04,A05,A06,A07,A08,A09;
    ulong A10,A11,A12,A13,A14,A15,A16,A17,A18,A19;
    ulong A20,A21,A22,A23,A24;

    // Track current chain "digest" (truncated) in registers across steps.
    // Initially equals the seed.
    ulong M00=S00,M01=S01,M02=S02,M03=S03,M04=S04,M05=S05,M06=S06,M07=S07;
    ulong M08=S08,M09=S09,M10=S10,M11=S11,M12=S12,M13=S13,M14=S14,M15=S15;

    for (uint step = 0u; step < W; ++step) {
        // Build padded state in lanes 0..24, then apply complement mask.
        // Lanes 0..n_lanes-1 = message (M*); lane dom_lane gets ^=0x06;
        // lane 16 gets ^=0x80<<56; rest = 0. Then ^= complement mask on
        // lanes {1,2,8,12,17,20}.
        A00 = M00 ^ dpad00;
        A01 = M01 ^ dpad01 ^ CMASK01;
        A02 = M02 ^ dpad02 ^ CMASK02;
        A03 = M03 ^ dpad03;
        A04 = M04 ^ dpad04;
        A05 = M05 ^ dpad05;
        A06 = M06 ^ dpad06;
        A07 = M07 ^ dpad07;
        A08 = M08 ^ dpad08 ^ CMASK08;
        A09 = M09 ^ dpad09;
        A10 = M10 ^ dpad10;
        A11 = M11 ^ dpad11;
        A12 = M12 ^ dpad12 ^ CMASK12;
        A13 = M13 ^ dpad13;
        A14 = M14 ^ dpad14;
        A15 = M15 ^ dpad15;
        A16 = dpad16 ^ fpad16;
        A17 = CMASK17;
        A18 = 0;
        A19 = 0;
        A20 = CMASK20;
        A21 = 0;
        A22 = 0;
        A23 = 0;
        A24 = 0;
        // (Lanes 0..15 outside [0..n_lanes-1] should be 0; since we used
        //  M* registers that mirror the previous truncation, lanes
        //  n_lanes..15 in M* are still whatever was loaded. We must zero
        //  those. We track validity via n_lanes: zero out the unused
        //  high message slots in A.)
        if (n_lanes < 1u)  A00 = dpad00;                  // no message
        if (n_lanes < 2u)  A01 = dpad01 ^ CMASK01;
        if (n_lanes < 3u)  A02 = dpad02 ^ CMASK02;
        if (n_lanes < 4u)  A03 = dpad03;
        if (n_lanes < 5u)  A04 = dpad04;
        if (n_lanes < 6u)  A05 = dpad05;
        if (n_lanes < 7u)  A06 = dpad06;
        if (n_lanes < 8u)  A07 = dpad07;
        if (n_lanes < 9u)  A08 = dpad08 ^ CMASK08;
        if (n_lanes < 10u) A09 = dpad09;
        if (n_lanes < 11u) A10 = dpad10;
        if (n_lanes < 12u) A11 = dpad11;
        if (n_lanes < 13u) A12 = dpad12 ^ CMASK12;
        if (n_lanes < 14u) A13 = dpad13;
        if (n_lanes < 15u) A14 = dpad14;
        if (n_lanes < 16u) A15 = dpad15;

        // 24 Keccak-f rounds.
        for (uint r = 0u; r < 24u; ++r) {
            // theta
            ulong C0 = A00 ^ A05 ^ A10 ^ A15 ^ A20;
            ulong C1 = A01 ^ A06 ^ A11 ^ A16 ^ A21;
            ulong C2 = A02 ^ A07 ^ A12 ^ A17 ^ A22;
            ulong C3 = A03 ^ A08 ^ A13 ^ A18 ^ A23;
            ulong C4 = A04 ^ A09 ^ A14 ^ A19 ^ A24;

            ulong D0 = C4 ^ ROL(C1, 1);
            ulong D1 = C0 ^ ROL(C2, 1);
            ulong D2 = C1 ^ ROL(C3, 1);
            ulong D3 = C2 ^ ROL(C4, 1);
            ulong D4 = C3 ^ ROL(C0, 1);

            // theta + rho + pi: compute B lanes (rotated, permuted) from A^D.
            ulong B00 =      (A00 ^ D0);
            ulong B10 = ROL((A01 ^ D1),  1);
            ulong B20 = ROL((A02 ^ D2), 62);
            ulong B05 = ROL((A03 ^ D3), 28);
            ulong B15 = ROL((A04 ^ D4), 27);

            ulong B16 = ROL((A05 ^ D0), 36);
            ulong B01 = ROL((A06 ^ D1), 44);
            ulong B11 = ROL((A07 ^ D2),  6);
            ulong B21 = ROL((A08 ^ D3), 55);
            ulong B06 = ROL((A09 ^ D4), 20);

            ulong B07 = ROL((A10 ^ D0),  3);
            ulong B17 = ROL((A11 ^ D1), 10);
            ulong B02 = ROL((A12 ^ D2), 43);
            ulong B12 = ROL((A13 ^ D3), 25);
            ulong B22 = ROL((A14 ^ D4), 39);

            ulong B23 = ROL((A15 ^ D0), 41);
            ulong B08 = ROL((A16 ^ D1), 45);
            ulong B18 = ROL((A17 ^ D2), 15);
            ulong B03 = ROL((A18 ^ D3), 21);
            ulong B13 = ROL((A19 ^ D4),  8);

            ulong B14 = ROL((A20 ^ D0), 18);
            ulong B24 = ROL((A21 ^ D1),  2);
            ulong B09 = ROL((A22 ^ D2), 61);
            ulong B19 = ROL((A23 ^ D3), 56);
            ulong B04 = ROL((A24 ^ D4), 14);

            // chi with lane-complementing:
            // Complemented lanes: 1, 2, 8, 12, 17, 20.
            // For each row y = 0..4: A[5y+x] = B[5y+x] XOR f(B[5y+x+1], B[5y+x+2])
            // where f normally is (~b)&c, but for complemented lanes we
            // substitute to avoid the NOT.
            //
            // We use the standard substitution rules (Bertoni "Keccak
            // implementation overview" Fig. 5):
            //   Row y=0 (lanes 0..4):  complemented at lane 1, lane 2.
            //     A0 =  B0 ^ ( B1 | B2)        (complemented-1 & 2)
            //     A1 =  B1 ^ ( B2 | B3)        (1 stays comp, 2 comp, 3 normal)
            //     A2 =  B2 ^ ( B3 & B4)        (lane 2 stays comp via XOR semantics)
            //     A3 =  B3 ^ (~B4 & B0)
            //     A4 =  B4 ^ (~B0 & B1)
            // Building these by hand for every row is error-prone; instead
            // we apply the *standard* chi to A (after un-complementing
            // post-permutation) -- the lane-complement trick is normally
            // done by precomputing which (~b)&c pairs include complemented
            // lanes and rewriting only those. To keep correctness rock
            // solid we DROP the complement trick and just do plain chi.
            //
            // (Removing the complement: undo at A* by XOR with mask, do
            // chi, then redo... too expensive. Keep plain chi, no
            // complement. Reset CMASK* to 0 above? Already baked in.
            // Simpler: don't apply CMASK at all. Set CMASKxx = 0.)

            A00 = B00 ^ ((~B01) & B02);
            A01 = B01 ^ ((~B02) & B03);
            A02 = B02 ^ ((~B03) & B04);
            A03 = B03 ^ ((~B04) & B00);
            A04 = B04 ^ ((~B00) & B01);

            A05 = B05 ^ ((~B06) & B07);
            A06 = B06 ^ ((~B07) & B08);
            A07 = B07 ^ ((~B08) & B09);
            A08 = B08 ^ ((~B09) & B05);
            A09 = B09 ^ ((~B05) & B06);

            A10 = B10 ^ ((~B11) & B12);
            A11 = B11 ^ ((~B12) & B13);
            A12 = B12 ^ ((~B13) & B14);
            A13 = B13 ^ ((~B14) & B10);
            A14 = B14 ^ ((~B10) & B11);

            A15 = B15 ^ ((~B16) & B17);
            A16 = B16 ^ ((~B17) & B18);
            A17 = B17 ^ ((~B18) & B19);
            A18 = B18 ^ ((~B19) & B15);
            A19 = B19 ^ ((~B15) & B16);

            A20 = B20 ^ ((~B21) & B22);
            A21 = B21 ^ ((~B22) & B23);
            A22 = B22 ^ ((~B23) & B24);
            A23 = B23 ^ ((~B24) & B20);
            A24 = B24 ^ ((~B20) & B21);

            // iota
            A00 ^= KECCAK_RC[r];
        }

        // Save truncated digest into M* for next step.
        M00 = A00; M01 = A01; M02 = A02; M03 = A03;
        M04 = A04; M05 = A05; M06 = A06; M07 = A07;
        M08 = A08; M09 = A09; M10 = A10; M11 = A11;
        M12 = A12; M13 = A13; M14 = A14; M15 = A15;
    }

    // Write tip: first n_lanes message lanes.
    if (n_lanes > 0u)  tips[base + 0]  = M00;
    if (n_lanes > 1u)  tips[base + 1]  = M01;
    if (n_lanes > 2u)  tips[base + 2]  = M02;
    if (n_lanes > 3u)  tips[base + 3]  = M03;
    if (n_lanes > 4u)  tips[base + 4]  = M04;
    if (n_lanes > 5u)  tips[base + 5]  = M05;
    if (n_lanes > 6u)  tips[base + 6]  = M06;
    if (n_lanes > 7u)  tips[base + 7]  = M07;
    if (n_lanes > 8u)  tips[base + 8]  = M08;
    if (n_lanes > 9u)  tips[base + 9]  = M09;
    if (n_lanes > 10u) tips[base + 10] = M10;
    if (n_lanes > 11u) tips[base + 11] = M11;
    if (n_lanes > 12u) tips[base + 12] = M12;
    if (n_lanes > 13u) tips[base + 13] = M13;
    if (n_lanes > 14u) tips[base + 14] = M14;
    if (n_lanes > 15u) tips[base + 15] = M15;
}
```

Result of previous attempt:
          w16_C64K: INCORRECT (bit_exact=131072)
  fail_reason: correctness failed at size w16_C64K: bit_exact=131072

## Current best (incumbent)

```metal
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

inline ulong ROL(ulong x, uint k) {
    return rotate(x, (ulong)k);
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

    // Named-lane state convention (FIPS 202): lane k=x+5y.
    // x \in {a,e,i,o,u} y \in {a,e,i,o,u}: Aba=A[0], Aga=A[1], ..., Asu=A[24]
    // We use the standard XKCP-style naming: row letter is y, col letter is x?
    // Actually XKCP uses: A##xy where lane = x + 5*y. We'll just index by (x,y) explicitly.

    ulong A00=0, A01=0, A02=0, A03=0, A04=0;
    ulong A05=0, A06=0, A07=0, A08=0, A09=0;
    ulong A10=0, A11=0, A12=0, A13=0, A14=0;
    ulong A15=0, A16=0, A17=0, A18=0, A19=0;
    ulong A20=0, A21=0, A22=0, A23=0, A24=0;

    // Load seed (n_lanes in {2,4} typically, but generic up to 16).
    ulong s[16];
    for (uint k = 0u; k < n_lanes; ++k) s[k] = seeds[base + k];

    // Place seed into lanes 0..n_lanes-1.
    if (n_lanes > 0u)  A00 = s[0];
    if (n_lanes > 1u)  A01 = s[1];
    if (n_lanes > 2u)  A02 = s[2];
    if (n_lanes > 3u)  A03 = s[3];
    if (n_lanes > 4u)  A04 = s[4];
    if (n_lanes > 5u)  A05 = s[5];
    if (n_lanes > 6u)  A06 = s[6];
    if (n_lanes > 7u)  A07 = s[7];
    if (n_lanes > 8u)  A08 = s[8];
    if (n_lanes > 9u)  A09 = s[9];
    if (n_lanes > 10u) A10 = s[10];
    if (n_lanes > 11u) A11 = s[11];
    if (n_lanes > 12u) A12 = s[12];
    if (n_lanes > 13u) A13 = s[13];
    if (n_lanes > 14u) A14 = s[14];
    if (n_lanes > 15u) A15 = s[15];

    // Domain pad lane index = n_lanes; final pad lane = 16.
    // We apply XOR (since other lanes are 0 between steps anyway).
    uint dom_lane = n_lanes;

    for (uint step = 0u; step < w; ++step) {
        // Apply SHA3 padding: XOR 0x06 into lane[n_lanes], XOR 0x80<<56 into lane[16].
        // Zero out lanes n_lanes..24 (they may carry residue from previous Keccak-f output).
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

        // XOR domain pad byte 0x06 at lane = n_lanes.
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
        // Final pad byte 0x80 at byte 7 of lane 16.
        A16 ^= 0x8000000000000000ul;

        // ----- Keccak-f[1600], 24 rounds, fully unrolled per-round body -----
        for (uint r = 0u; r < 24u; ++r) {
            // theta
            ulong C0 = A00 ^ A05 ^ A10 ^ A15 ^ A20;
            ulong C1 = A01 ^ A06 ^ A11 ^ A16 ^ A21;
            ulong C2 = A02 ^ A07 ^ A12 ^ A17 ^ A22;
            ulong C3 = A03 ^ A08 ^ A13 ^ A18 ^ A23;
            ulong C4 = A04 ^ A09 ^ A14 ^ A19 ^ A24;

            ulong D0 = C4 ^ ROL(C1, 1);
            ulong D1 = C0 ^ ROL(C2, 1);
            ulong D2 = C1 ^ ROL(C3, 1);
            ulong D3 = C2 ^ ROL(C4, 1);
            ulong D4 = C3 ^ ROL(C0, 1);

            A00 ^= D0; A05 ^= D0; A10 ^= D0; A15 ^= D0; A20 ^= D0;
            A01 ^= D1; A06 ^= D1; A11 ^= D1; A16 ^= D1; A21 ^= D1;
            A02 ^= D2; A07 ^= D2; A12 ^= D2; A17 ^= D2; A22 ^= D2;
            A03 ^= D3; A08 ^= D3; A13 ^= D3; A18 ^= D3; A23 ^= D3;
            A04 ^= D4; A09 ^= D4; A14 ^= D4; A19 ^= D4; A24 ^= D4;

            // rho + pi: B[dst] = ROL(A[src], rho[src])
            // mapping: (x,y) src -> (y, (2x+3y)%5) dst
            // We assign into a 25-lane temp then chi back into A.
            ulong B00 = A00;                  // (0,0) -> (0,0)
            ulong B10 = ROL(A01,  1);         // (1,0) -> (0,2) => lane 10
            ulong B20 = ROL(A02, 62);         // (2,0) -> (0,4) => lane 20
            ulong B05 = ROL(A03, 28);         // (3,0) -> (0,1) => lane 5
            ulong B15 = ROL(A04, 27);         // (4,0) -> (0,3) => lane 15

            ulong B16 = ROL(A05, 36);         // (0,1) -> (1,3) => lane 16
            ulong B01 = ROL(A06, 44);         // (1,1) -> (1,0) => lane 1
            ulong B11 = ROL(A07,  6);         // (2,1) -> (1,2) => lane 11
            ulong B21 = ROL(A08, 55);         // (3,1) -> (1,4) => lane 21
            ulong B06 = ROL(A09, 20);         // (4,1) -> (1,1) => lane 6

            ulong B07 = ROL(A10,  3);         // (0,2) -> (2,1) => lane 7
            ulong B17 = ROL(A11, 10);         // (1,2) -> (2,3) => lane 17
            ulong B02 = ROL(A12, 43);         // (2,2) -> (2,0) => lane 2
            ulong B12 = ROL(A13, 25);         // (3,2) -> (2,2) => lane 12
            ulong B22 = ROL(A14, 39);         // (4,2) -> (2,4) => lane 22

            ulong B23 = ROL(A15, 41);         // (0,3) -> (3,4) => lane 23
            ulong B08 = ROL(A16, 45);         // (1,3) -> (3,1) => lane 8
            ulong B18 = ROL(A17, 15);         // (2,3) -> (3,3) => lane 18
            ulong B03 = ROL(A18, 21);         // (3,3) -> (3,0) => lane 3
            ulong B13 = ROL(A19,  8);         // (4,3) -> (3,2) => lane 13

            ulong B14 = ROL(A20, 18);         // (0,4) -> (4,2) => lane 14
            ulong B24 = ROL(A21,  2);         // (1,4) -> (4,4) => lane 24
            ulong B09 = ROL(A22, 61);         // (2,4) -> (4,1) => lane 9
            ulong B19 = ROL(A23, 56);         // (3,4) -> (4,3) => lane 19
            ulong B04 = ROL(A24, 14);         // (4,4) -> (4,0) => lane 4

            // chi: row-wise nonlinear mix
            A00 = B00 ^ ((~B01) & B02);
            A01 = B01 ^ ((~B02) & B03);
            A02 = B02 ^ ((~B03) & B04);
            A03 = B03 ^ ((~B04) & B00);
            A04 = B04 ^ ((~B00) & B01);

            A05 = B05 ^ ((~B06) & B07);
            A06 = B06 ^ ((~B07) & B08);
            A07 = B07 ^ ((~B08) & B09);
            A08 = B08 ^ ((~B09) & B05);
            A09 = B09 ^ ((~B05) & B06);

            A10 = B10 ^ ((~B11) & B12);
            A11 = B11 ^ ((~B12) & B13);
            A12 = B12 ^ ((~B13) & B14);
            A13 = B13 ^ ((~B14) & B10);
            A14 = B14 ^ ((~B10) & B11);

            A15 = B15 ^ ((~B16) & B17);
            A16 = B16 ^ ((~B17) & B18);
            A17 = B17 ^ ((~B18) & B19);
            A18 = B18 ^ ((~B19) & B15);
            A19 = B19 ^ ((~B15) & B16);

            A20 = B20 ^ ((~B21) & B22);
            A21 = B21 ^ ((~B22) & B23);
            A22 = B22 ^ ((~B23) & B24);
            A23 = B23 ^ ((~B24) & B20);
            A24 = B24 ^ ((~B20) & B21);

            // iota
            A00 ^= KECCAK_RC[r];
        }
    }

    // Squeeze: write first n_lanes lanes.
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

Incumbent result:
          w16_C64K: correct, 5.25 ms, 742.4 Gbitops/s (u64) (66.0% of 1125 Gops/s (u64 bitop, est))
          w64_C64K: correct, 21.19 ms, 736.5 Gbitops/s (u64) (65.5% of 1125 Gops/s (u64 bitop, est))
         w256_C64K: correct, 85.98 ms, 725.9 Gbitops/s (u64) (64.5% of 1125 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.6532

## History

- iter  2: compile=OK | correct=True | score=0.6452034962028127
- iter  3: compile=OK | correct=True | score=0.6062045222347305
- iter  4: compile=OK | correct=True | score=0.6068846513062454
- iter  5: compile=OK | correct=True | score=0.632724312997265
- iter  6: compile=OK | correct=False | score=N/A
- iter  7: compile=OK | correct=True | score=0.6477403554332285
- iter  8: compile=OK | correct=True | score=0.6011693509441158
- iter  9: compile=OK | correct=False | score=N/A

## Stagnation notice

Your last 3 correct attempts all scored within 15% of
the incumbent without surpassing it. You are circling a local
optimum. STOP making incremental edits to the previous kernel and
propose a STRUCTURALLY different approach.

A reworded version of the previous kernel will not break out of
this plateau.

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
