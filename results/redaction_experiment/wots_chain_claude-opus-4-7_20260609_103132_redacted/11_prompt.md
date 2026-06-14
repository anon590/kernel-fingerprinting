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

constant ulong KRC[24] = {
    0x0000000000000001ul, 0x0000000000008082ul, 0x800000000000808Aul,
    0x8000000080008000ul, 0x000000000000808Bul, 0x0000000080000001ul,
    0x8000000080008081ul, 0x8000000000008009ul, 0x000000000000008Aul,
    0x0000000000000088ul, 0x0000000080008009ul, 0x000000008000000Aul,
    0x000000008000808Bul, 0x800000000000008Bul, 0x8000000000008089ul,
    0x8000000000008003ul, 0x8000000000008002ul, 0x8000000000000080ul,
    0x000000000000800Aul, 0x800000008000000Aul, 0x8000000080008081ul,
    0x8000000000008080ul, 0x0000000080000001ul, 0x8000000080008008ul,
};

#define ROL(x,k) rotate((ulong)(x), (ulong)(k))

// One Keccak-f round operating on 25 named lane registers.
#define KECCAK_ROUND(RC) {                                                 \
    ulong C0 = a00 ^ a05 ^ a10 ^ a15 ^ a20;                                \
    ulong C1 = a01 ^ a06 ^ a11 ^ a16 ^ a21;                                \
    ulong C2 = a02 ^ a07 ^ a12 ^ a17 ^ a22;                                \
    ulong C3 = a03 ^ a08 ^ a13 ^ a18 ^ a23;                                \
    ulong C4 = a04 ^ a09 ^ a14 ^ a19 ^ a24;                                \
    ulong D0 = C4 ^ ROL(C1,1);                                             \
    ulong D1 = C0 ^ ROL(C2,1);                                             \
    ulong D2 = C1 ^ ROL(C3,1);                                             \
    ulong D3 = C2 ^ ROL(C4,1);                                             \
    ulong D4 = C3 ^ ROL(C0,1);                                             \
    ulong B00 =     (a00 ^ D0);                                            \
    ulong B10 = ROL(a01 ^ D1,  1);                                         \
    ulong B20 = ROL(a02 ^ D2, 62);                                         \
    ulong B05 = ROL(a03 ^ D3, 28);                                         \
    ulong B15 = ROL(a04 ^ D4, 27);                                         \
    ulong B16 = ROL(a05 ^ D0, 36);                                         \
    ulong B01 = ROL(a06 ^ D1, 44);                                         \
    ulong B11 = ROL(a07 ^ D2,  6);                                         \
    ulong B21 = ROL(a08 ^ D3, 55);                                         \
    ulong B06 = ROL(a09 ^ D4, 20);                                         \
    ulong B07 = ROL(a10 ^ D0,  3);                                         \
    ulong B17 = ROL(a11 ^ D1, 10);                                         \
    ulong B02 = ROL(a12 ^ D2, 43);                                         \
    ulong B12 = ROL(a13 ^ D3, 25);                                         \
    ulong B22 = ROL(a14 ^ D4, 39);                                         \
    ulong B23 = ROL(a15 ^ D0, 41);                                         \
    ulong B08 = ROL(a16 ^ D1, 45);                                         \
    ulong B18 = ROL(a17 ^ D2, 15);                                         \
    ulong B03 = ROL(a18 ^ D3, 21);                                         \
    ulong B13 = ROL(a19 ^ D4,  8);                                         \
    ulong B14 = ROL(a20 ^ D0, 18);                                         \
    ulong B24 = ROL(a21 ^ D1,  2);                                         \
    ulong B09 = ROL(a22 ^ D2, 61);                                         \
    ulong B19 = ROL(a23 ^ D3, 56);                                         \
    ulong B04 = ROL(a24 ^ D4, 14);                                         \
    a00 = B00 ^ ((~B01) & B02) ^ (RC);                                     \
    a01 = B01 ^ ((~B02) & B03);                                            \
    a02 = B02 ^ ((~B03) & B04);                                            \
    a03 = B03 ^ ((~B04) & B00);                                            \
    a04 = B04 ^ ((~B00) & B01);                                            \
    a05 = B05 ^ ((~B06) & B07);                                            \
    a06 = B06 ^ ((~B07) & B08);                                            \
    a07 = B07 ^ ((~B08) & B09);                                            \
    a08 = B08 ^ ((~B09) & B05);                                            \
    a09 = B09 ^ ((~B05) & B06);                                            \
    a10 = B10 ^ ((~B11) & B12);                                            \
    a11 = B11 ^ ((~B12) & B13);                                            \
    a12 = B12 ^ ((~B13) & B14);                                            \
    a13 = B13 ^ ((~B14) & B10);                                            \
    a14 = B14 ^ ((~B10) & B11);                                            \
    a15 = B15 ^ ((~B16) & B17);                                            \
    a16 = B16 ^ ((~B17) & B18);                                            \
    a17 = B17 ^ ((~B18) & B19);                                            \
    a18 = B18 ^ ((~B19) & B15);                                            \
    a19 = B19 ^ ((~B15) & B16);                                            \
    a20 = B20 ^ ((~B21) & B22);                                            \
    a21 = B21 ^ ((~B22) & B23);                                            \
    a22 = B22 ^ ((~B23) & B24);                                            \
    a23 = B23 ^ ((~B24) & B20);                                            \
    a24 = B24 ^ ((~B20) & B21);                                            \
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
    const uint W = w;

    // Precompute per-lane padding masks (constant over the chain).
    // pad[k] gives bits to XOR into lane k of a zero state to encode
    // SHA3-256 padding when message occupies lanes 0..n_lanes-1.
    ulong pad00=0, pad01=0, pad02=0, pad03=0, pad04=0;
    ulong pad05=0, pad06=0, pad07=0, pad08=0, pad09=0;
    ulong pad10=0, pad11=0, pad12=0, pad13=0, pad14=0;
    ulong pad15=0, pad16=0x8000000000000000ul;
    switch (n_lanes) {
        case 0:  pad00 = 0x06ul; break;
        case 1:  pad01 = 0x06ul; break;
        case 2:  pad02 = 0x06ul; break;
        case 3:  pad03 = 0x06ul; break;
        case 4:  pad04 = 0x06ul; break;
        case 5:  pad05 = 0x06ul; break;
        case 6:  pad06 = 0x06ul; break;
        case 7:  pad07 = 0x06ul; break;
        case 8:  pad08 = 0x06ul; break;
        case 9:  pad09 = 0x06ul; break;
        case 10: pad10 = 0x06ul; break;
        case 11: pad11 = 0x06ul; break;
        case 12: pad12 = 0x06ul; break;
        case 13: pad13 = 0x06ul; break;
        case 14: pad14 = 0x06ul; break;
        case 15: pad15 = 0x06ul; break;
        case 16: pad16 ^= 0x06ul; break;
        default: break;
    }

    // Load message lanes into m00..m15.
    ulong m00=0,m01=0,m02=0,m03=0,m04=0,m05=0,m06=0,m07=0;
    ulong m08=0,m09=0,m10=0,m11=0,m12=0,m13=0,m14=0,m15=0;
    if (n_lanes > 0u)  m00 = seeds[base + 0];
    if (n_lanes > 1u)  m01 = seeds[base + 1];
    if (n_lanes > 2u)  m02 = seeds[base + 2];
    if (n_lanes > 3u)  m03 = seeds[base + 3];
    if (n_lanes > 4u)  m04 = seeds[base + 4];
    if (n_lanes > 5u)  m05 = seeds[base + 5];
    if (n_lanes > 6u)  m06 = seeds[base + 6];
    if (n_lanes > 7u)  m07 = seeds[base + 7];
    if (n_lanes > 8u)  m08 = seeds[base + 8];
    if (n_lanes > 9u)  m09 = seeds[base + 9];
    if (n_lanes > 10u) m10 = seeds[base + 10];
    if (n_lanes > 11u) m11 = seeds[base + 11];
    if (n_lanes > 12u) m12 = seeds[base + 12];
    if (n_lanes > 13u) m13 = seeds[base + 13];
    if (n_lanes > 14u) m14 = seeds[base + 14];
    if (n_lanes > 15u) m15 = seeds[base + 15];

    for (uint step = 0u; step < W; ++step) {
        // Build state with message in lanes 0..n_lanes-1 (others zero)
        // plus padding XOR.
        ulong a00 = m00 ^ pad00;
        ulong a01 = m01 ^ pad01;
        ulong a02 = m02 ^ pad02;
        ulong a03 = m03 ^ pad03;
        ulong a04 = m04 ^ pad04;
        ulong a05 = m05 ^ pad05;
        ulong a06 = m06 ^ pad06;
        ulong a07 = m07 ^ pad07;
        ulong a08 = m08 ^ pad08;
        ulong a09 = m09 ^ pad09;
        ulong a10 = m10 ^ pad10;
        ulong a11 = m11 ^ pad11;
        ulong a12 = m12 ^ pad12;
        ulong a13 = m13 ^ pad13;
        ulong a14 = m14 ^ pad14;
        ulong a15 = m15 ^ pad15;
        ulong a16 = pad16;
        ulong a17 = 0, a18 = 0, a19 = 0;
        ulong a20 = 0, a21 = 0, a22 = 0, a23 = 0, a24 = 0;

        // 24 rounds, fully unrolled to give the compiler maximum freedom.
        KECCAK_ROUND(KRC[0]);
        KECCAK_ROUND(KRC[1]);
        KECCAK_ROUND(KRC[2]);
        KECCAK_ROUND(KRC[3]);
        KECCAK_ROUND(KRC[4]);
        KECCAK_ROUND(KRC[5]);
        KECCAK_ROUND(KRC[6]);
        KECCAK_ROUND(KRC[7]);
        KECCAK_ROUND(KRC[8]);
        KECCAK_ROUND(KRC[9]);
        KECCAK_ROUND(KRC[10]);
        KECCAK_ROUND(KRC[11]);
        KECCAK_ROUND(KRC[12]);
        KECCAK_ROUND(KRC[13]);
        KECCAK_ROUND(KRC[14]);
        KECCAK_ROUND(KRC[15]);
        KECCAK_ROUND(KRC[16]);
        KECCAK_ROUND(KRC[17]);
        KECCAK_ROUND(KRC[18]);
        KECCAK_ROUND(KRC[19]);
        KECCAK_ROUND(KRC[20]);
        KECCAK_ROUND(KRC[21]);
        KECCAK_ROUND(KRC[22]);
        KECCAK_ROUND(KRC[23]);

        // Truncate to first n_lanes lanes for next iteration.
        m00 = a00; m01 = a01; m02 = a02; m03 = a03;
        m04 = a04; m05 = a05; m06 = a06; m07 = a07;
        m08 = a08; m09 = a09; m10 = a10; m11 = a11;
        m12 = a12; m13 = a13; m14 = a14; m15 = a15;
    }

    if (n_lanes > 0u)  tips[base + 0]  = m00;
    if (n_lanes > 1u)  tips[base + 1]  = m01;
    if (n_lanes > 2u)  tips[base + 2]  = m02;
    if (n_lanes > 3u)  tips[base + 3]  = m03;
    if (n_lanes > 4u)  tips[base + 4]  = m04;
    if (n_lanes > 5u)  tips[base + 5]  = m05;
    if (n_lanes > 6u)  tips[base + 6]  = m06;
    if (n_lanes > 7u)  tips[base + 7]  = m07;
    if (n_lanes > 8u)  tips[base + 8]  = m08;
    if (n_lanes > 9u)  tips[base + 9]  = m09;
    if (n_lanes > 10u) tips[base + 10] = m10;
    if (n_lanes > 11u) tips[base + 11] = m11;
    if (n_lanes > 12u) tips[base + 12] = m12;
    if (n_lanes > 13u) tips[base + 13] = m13;
    if (n_lanes > 14u) tips[base + 14] = m14;
    if (n_lanes > 15u) tips[base + 15] = m15;
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

- iter  3: compile=OK | correct=True | score=0.6062045222347305
- iter  4: compile=OK | correct=True | score=0.6068846513062454
- iter  5: compile=OK | correct=True | score=0.632724312997265
- iter  6: compile=OK | correct=False | score=N/A
- iter  7: compile=OK | correct=True | score=0.6477403554332285
- iter  8: compile=OK | correct=True | score=0.6011693509441158
- iter  9: compile=OK | correct=False | score=N/A
- iter 10: compile=OK | correct=False | score=N/A

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
