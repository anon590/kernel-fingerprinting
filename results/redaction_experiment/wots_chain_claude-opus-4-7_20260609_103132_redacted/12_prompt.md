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

// Bit-interleaved Keccak: split each 64-bit lane into even/odd bits.
// Rotations on the interleaved form become two 32-bit rotations.
// rotate by r on 64-bit = rotate halves by r/2 (if r even) or
// (rotate odd by (r+1)/2, rotate even by r/2, swap) (if r odd).
// We work entirely on uint pairs (e,o) packed as uint2.

// Forward: split ulong into (even_bits, odd_bits) where even_bits holds
// bit positions 0,2,4,... and odd_bits holds 1,3,5,...
inline uint2 interleave(ulong x) {
    uint lo = (uint)x;
    uint hi = (uint)(x >> 32);
    // Gather even bits of lo into low 16 of e, even bits of hi into high 16 of e.
    // Use bit-twiddle: standard interleave.
    auto pack = [](uint v) {
        v &= 0x55555555u;
        v = (v | (v >> 1)) & 0x33333333u;
        v = (v | (v >> 2)) & 0x0F0F0F0Fu;
        v = (v | (v >> 4)) & 0x00FF00FFu;
        v = (v | (v >> 8)) & 0x0000FFFFu;
        return v;
    };
    uint e = pack(lo) | (pack(hi) << 16);
    uint o = pack(lo >> 1) | (pack(hi >> 1) << 16);
    return uint2(e, o);
}

inline ulong deinterleave(uint2 p) {
    auto spread = [](uint v) -> uint {
        v &= 0x0000FFFFu;
        v = (v | (v << 8)) & 0x00FF00FFu;
        v = (v | (v << 4)) & 0x0F0F0F0Fu;
        v = (v | (v << 2)) & 0x33333333u;
        v = (v | (v << 1)) & 0x55555555u;
        return v;
    };
    uint e = p.x, o = p.y;
    uint lo = spread(e & 0xFFFFu) | (spread(o & 0xFFFFu) << 1);
    uint hi = spread(e >> 16)     | (spread(o >> 16) << 1);
    return ((ulong)hi << 32) | (ulong)lo;
}

// Rotate interleaved (e,o) representing 64-bit value left by r.
inline uint2 rol_i(uint2 p, uint r) {
    uint r2 = r >> 1;
    if ((r & 1u) == 0u) {
        return uint2(rotate(p.x, r2), rotate(p.y, r2));
    } else {
        // odd: new_e = ROL(o, r2+? )... standard formula:
        // value bit i goes to (i+r) mod 64.
        // If r odd: even bits of input go to odd positions of output, with extra rotate.
        // new_o = ROL(p.x, (r+1)/2)
        // new_e = ROL(p.y, (r-1)/2 + 1)? Let's derive:
        // Input bit 2k (even, in p.x at pos k) -> output bit (2k+r) mod 64.
        //   r odd -> 2k+r is odd -> in new_o at pos ((2k+r) mod 64)/2 = (k + (r-1)/2 + (carry?)) ...
        //   Actually (2k+r) mod 64, with r=2m+1 -> = 2k+2m+1 mod 64.
        //   This is odd, position in odd plane = ((2k+2m+1)-1)/2 mod 32 = (k+m) mod 32.
        //   So new_o[k+m mod 32] = p.x[k] => new_o = ROL(p.x, m) = ROL(p.x, (r-1)/2).
        // Input bit 2k+1 (odd, in p.y at pos k) -> output bit (2k+1+r) mod 64.
        //   = 2k+2m+2 mod 64, even, position = (k+m+1) mod 32.
        //   So new_e[k+m+1] = p.y[k] => new_e = ROL(p.y, m+1) = ROL(p.y, (r+1)/2).
        uint m = (r - 1u) >> 1;
        return uint2(rotate(p.y, m + 1u), rotate(p.x, m));
    }
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

    // 25 lanes in interleaved form.
    uint2 S[25];

    // Precompute interleaved seed lanes.
    uint2 M[16];
    for (uint k = 0u; k < 16u; ++k) M[k] = uint2(0u, 0u);
    for (uint k = 0u; k < n_lanes; ++k) {
        M[k] = interleave(seeds[base + k]);
    }

    // Interleaved padding constants.
    // 0x06 at lane n_lanes: bits 1 and 2 set.
    //   bit 1 (odd, pos 0 in p.y) -> p.y |= 1
    //   bit 2 (even, pos 1 in p.x) -> p.x |= 2
    uint2 dom_pad = uint2(0x2u, 0x1u);
    // 0x80 << 56 at lane 16 = bit 63 set.
    //   bit 63 (odd, pos 31 in p.y) -> p.y |= 0x80000000
    uint2 fin_pad = uint2(0u, 0x80000000u);

    for (uint step = 0u; step < W; ++step) {
        // Initialize state: lanes 0..n_lanes-1 = M, lane n_lanes ^= dom_pad,
        // lane 16 ^= fin_pad, rest zero.
        for (uint k = 0u; k < 25u; ++k) S[k] = uint2(0u, 0u);
        for (uint k = 0u; k < n_lanes; ++k) S[k] = M[k];

        // Domain pad lane
        uint dl = n_lanes;
        // We need conditional XOR into S[dl]. Use a switch.
        switch (dl) {
            case 0:  S[0]  = S[0]  ^ dom_pad; break;
            case 1:  S[1]  = S[1]  ^ dom_pad; break;
            case 2:  S[2]  = S[2]  ^ dom_pad; break;
            case 3:  S[3]  = S[3]  ^ dom_pad; break;
            case 4:  S[4]  = S[4]  ^ dom_pad; break;
            case 5:  S[5]  = S[5]  ^ dom_pad; break;
            case 6:  S[6]  = S[6]  ^ dom_pad; break;
            case 7:  S[7]  = S[7]  ^ dom_pad; break;
            case 8:  S[8]  = S[8]  ^ dom_pad; break;
            case 9:  S[9]  = S[9]  ^ dom_pad; break;
            case 10: S[10] = S[10] ^ dom_pad; break;
            case 11: S[11] = S[11] ^ dom_pad; break;
            case 12: S[12] = S[12] ^ dom_pad; break;
            case 13: S[13] = S[13] ^ dom_pad; break;
            case 14: S[14] = S[14] ^ dom_pad; break;
            case 15: S[15] = S[15] ^ dom_pad; break;
            case 16: S[16] = S[16] ^ dom_pad; break;
            default: break;
        }
        S[16] = S[16] ^ fin_pad;

        // Keccak-f[1600] in interleaved form.
        for (uint r = 0u; r < 24u; ++r) {
            // theta
            uint2 C0 = S[0]^S[5]^S[10]^S[15]^S[20];
            uint2 C1 = S[1]^S[6]^S[11]^S[16]^S[21];
            uint2 C2 = S[2]^S[7]^S[12]^S[17]^S[22];
            uint2 C3 = S[3]^S[8]^S[13]^S[18]^S[23];
            uint2 C4 = S[4]^S[9]^S[14]^S[19]^S[24];

            uint2 D0 = C4 ^ rol_i(C1, 1);
            uint2 D1 = C0 ^ rol_i(C2, 1);
            uint2 D2 = C1 ^ rol_i(C3, 1);
            uint2 D3 = C2 ^ rol_i(C4, 1);
            uint2 D4 = C3 ^ rol_i(C0, 1);

            S[0]^=D0; S[5]^=D0; S[10]^=D0; S[15]^=D0; S[20]^=D0;
            S[1]^=D1; S[6]^=D1; S[11]^=D1; S[16]^=D1; S[21]^=D1;
            S[2]^=D2; S[7]^=D2; S[12]^=D2; S[17]^=D2; S[22]^=D2;
            S[3]^=D3; S[8]^=D3; S[13]^=D3; S[18]^=D3; S[23]^=D3;
            S[4]^=D4; S[9]^=D4; S[14]^=D4; S[19]^=D4; S[24]^=D4;

            // rho + pi
            uint2 B00 = S[0];
            uint2 B10 = rol_i(S[1],  1);
            uint2 B20 = rol_i(S[2], 62);
            uint2 B05 = rol_i(S[3], 28);
            uint2 B15 = rol_i(S[4], 27);
            uint2 B16 = rol_i(S[5], 36);
            uint2 B01 = rol_i(S[6], 44);
            uint2 B11 = rol_i(S[7],  6);
            uint2 B21 = rol_i(S[8], 55);
            uint2 B06 = rol_i(S[9], 20);
            uint2 B07 = rol_i(S[10],  3);
            uint2 B17 = rol_i(S[11], 10);
            uint2 B02 = rol_i(S[12], 43);
            uint2 B12 = rol_i(S[13], 25);
            uint2 B22 = rol_i(S[14], 39);
            uint2 B23 = rol_i(S[15], 41);
            uint2 B08 = rol_i(S[16], 45);
            uint2 B18 = rol_i(S[17], 15);
            uint2 B03 = rol_i(S[18], 21);
            uint2 B13 = rol_i(S[19],  8);
            uint2 B14 = rol_i(S[20], 18);
            uint2 B24 = rol_i(S[21],  2);
            uint2 B09 = rol_i(S[22], 61);
            uint2 B19 = rol_i(S[23], 56);
            uint2 B04 = rol_i(S[24], 14);

            // chi
            S[0]  = B00 ^ ((~B01) & B02);
            S[1]  = B01 ^ ((~B02) & B03);
            S[2]  = B02 ^ ((~B03) & B04);
            S[3]  = B03 ^ ((~B04) & B00);
            S[4]  = B04 ^ ((~B00) & B01);
            S[5]  = B05 ^ ((~B06) & B07);
            S[6]  = B06 ^ ((~B07) & B08);
            S[7]  = B07 ^ ((~B08) & B09);
            S[8]  = B08 ^ ((~B09) & B05);
            S[9]  = B09 ^ ((~B05) & B06);
            S[10] = B10 ^ ((~B11) & B12);
            S[11] = B11 ^ ((~B12) & B13);
            S[12] = B12 ^ ((~B13) & B14);
            S[13] = B13 ^ ((~B14) & B10);
            S[14] = B14 ^ ((~B10) & B11);
            S[15] = B15 ^ ((~B16) & B17);
            S[16] = B16 ^ ((~B17) & B18);
            S[17] = B17 ^ ((~B18) & B19);
            S[18] = B18 ^ ((~B19) & B15);
            S[19] = B19 ^ ((~B15) & B16);
            S[20] = B20 ^ ((~B21) & B22);
            S[21] = B21 ^ ((~B22) & B23);
            S[22] = B22 ^ ((~B23) & B24);
            S[23] = B23 ^ ((~B24) & B20);
            S[24] = B24 ^ ((~B20) & B21);

            // iota: XOR interleaved RC.
            ulong rc = KRC[r];
            uint2 irc = interleave(rc);
            S[0] = S[0] ^ irc;
        }

        // Truncate: M[0..n_lanes-1] = S[0..n_lanes-1] (still interleaved).
        for (uint k = 0u; k < n_lanes; ++k) M[k] = S[k];
    }

    // Deinterleave output.
    for (uint k = 0u; k < n_lanes; ++k) {
        tips[base + k] = deinterleave(M[k]);
    }
}
```

Result of previous attempt:
  COMPILE FAILED: Error Domain=MTLLibraryErrorDomain Code=3 "program_source:28:17: error: lambda expressions are not supported in Metal
    auto pack = [](uint v) {
                ^
program_source:42:19: error: lambda expressions are not supported in Metal
    auto spread = [](uint v) -> uint {
                  ^
" UserInfo={NSLocalizedDescription=program_source:28:17: error: lambda expressions are not supported in Metal
    auto pack = [](uint v) {
                ^
program_source:42:19: error: lambda expressions are not supported in Metal
    auto spread = [](uint v) -> uint {
                  ^
}

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

- iter  4: compile=OK | correct=True | score=0.6068846513062454
- iter  5: compile=OK | correct=True | score=0.632724312997265
- iter  6: compile=OK | correct=False | score=N/A
- iter  7: compile=OK | correct=True | score=0.6477403554332285
- iter  8: compile=OK | correct=True | score=0.6011693509441158
- iter  9: compile=OK | correct=False | score=N/A
- iter 10: compile=OK | correct=False | score=N/A
- iter 11: compile=FAIL | correct=False | score=N/A

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
