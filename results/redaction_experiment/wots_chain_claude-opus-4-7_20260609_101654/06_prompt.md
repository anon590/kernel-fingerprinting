## Task: wots_chain

Batched WOTS+ / SPHINCS+-style hash chains. Given ``n_chains`` independent ``n_bytes``-byte seeds, apply the Keccak-256 inner hash ``w`` times in sequence per chain (each digest truncated to ``n_bytes`` bytes before feeding into the next iteration) and write the chain tip to the output. The chains are embarrassingly parallel; the ``w``-step iteration along each chain is strictly sequential.

Inner hash: Keccak-f[1600] with the FIPS 202 SHA3-256 sponge framing -- rate = 136 bytes (17 lanes), capacity = 64 bytes, domain pad byte = 0x06. State convention: the 1600-bit state is a 5x5 array of 64-bit lanes; lane k = x + 5*y holds bytes 8*k .. 8*k + 7 of the sponge state in little-endian.

All test sizes have ``n_bytes < rate_bytes`` (in-distribution n_bytes=16, held-out n_bytes=32; rate_bytes=136), so every chain step collapses to a single-block absorb + single-block squeeze of ``n_lanes = n_bytes / 8`` state lanes:
  state                          := 0
  state[lane 0..n_lanes-1]       := previous_chunk
  state[lane n_lanes, byte 0]    ^= 0x06   # SHA3 domain
  state[lane 16, byte 7]         ^= 0x80   # FIPS 202 final pad
  state                          := Keccak-f1600(state)
  next_chunk                     := state[lane 0..n_lanes-1]

On the first chain step the absorb is the seed; on every subsequent step the absorb is the n_lanes-lane truncation of the previous Keccak-f1600 output. After ``w`` steps the first n_lanes state lanes are written to the output as the chain tip.

The kernel must read ``n_bytes`` and ``w`` from the bound device buffers rather than treating them as compile-time constants; both vary across the test sizes (``w`` in {16, 64, 256} in the in-distribution sweep, ``n_bytes`` 16 -> 32 between in-distribution and held-out). Hardcoding either value silently produces wrong output, not just slow output.

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
    bool n2 = (n_lanes == 2u);

    ulong s0 = seeds[base + 0u];
    ulong s1 = seeds[base + 1u];
    ulong s2 = n2 ? 0ul : seeds[base + 2u];
    ulong s3 = n2 ? 0ul : seeds[base + 3u];

    const ulong PAD_FINAL = 0x8000000000000000ul;
    const ulong PAD_DOM   = 0x06ul;

    // Pre-computed padding constants for the fixed lanes 2..24.
    // If n_lanes==2: lane 2 = 0x06, lane 3 = 0, lane 4 = 0
    // If n_lanes==4: lane 2 = s2,   lane 3 = s3, lane 4 = 0x06
    // Lane 16 = PAD_FINAL always; all others 0.

    for (uint step = 0u; step < w; ++step) {
        // Initialize state with padding folded in.
        ulong a00 = s0;
        ulong a01 = s1;
        ulong a02 = n2 ? PAD_DOM : s2;
        ulong a03 = n2 ? 0ul     : s3;
        ulong a04 = n2 ? 0ul     : PAD_DOM;
        ulong a05 = 0ul, a06 = 0ul, a07 = 0ul, a08 = 0ul, a09 = 0ul;
        ulong a10 = 0ul, a11 = 0ul, a12 = 0ul, a13 = 0ul, a14 = 0ul;
        ulong a15 = 0ul;
        ulong a16 = PAD_FINAL;
        ulong a17 = 0ul, a18 = 0ul, a19 = 0ul;
        ulong a20 = 0ul, a21 = 0ul, a22 = 0ul, a23 = 0ul, a24 = 0ul;

        for (uint r = 0u; r < 24u; ++r) {
            // Theta
            ulong C0 = a00 ^ a05 ^ a10 ^ a15 ^ a20;
            ulong C1 = a01 ^ a06 ^ a11 ^ a16 ^ a21;
            ulong C2 = a02 ^ a07 ^ a12 ^ a17 ^ a22;
            ulong C3 = a03 ^ a08 ^ a13 ^ a18 ^ a23;
            ulong C4 = a04 ^ a09 ^ a14 ^ a19 ^ a24;

            ulong D0 = C4 ^ ROL(C1, 1);
            ulong D1 = C0 ^ ROL(C2, 1);
            ulong D2 = C1 ^ ROL(C3, 1);
            ulong D3 = C2 ^ ROL(C4, 1);
            ulong D4 = C3 ^ ROL(C0, 1);

            // Rho + Pi with D folded directly into the rotate operand.
            // Destination lane = (y, (2x+3y)%5) from source (x,y).
            ulong b00 = a00 ^ D0;
            ulong b10 = ROL(a01 ^ D1,  1);
            ulong b20 = ROL(a02 ^ D2, 62);
            ulong b05 = ROL(a03 ^ D3, 28);
            ulong b15 = ROL(a04 ^ D4, 27);

            ulong b16 = ROL(a05 ^ D0, 36);
            ulong b01 = ROL(a06 ^ D1, 44);
            ulong b11 = ROL(a07 ^ D2,  6);
            ulong b21 = ROL(a08 ^ D3, 55);
            ulong b06 = ROL(a09 ^ D4, 20);

            ulong b07 = ROL(a10 ^ D0,  3);
            ulong b17 = ROL(a11 ^ D1, 10);
            ulong b02 = ROL(a12 ^ D2, 43);
            ulong b12 = ROL(a13 ^ D3, 25);
            ulong b22 = ROL(a14 ^ D4, 39);

            ulong b23 = ROL(a15 ^ D0, 41);
            ulong b08 = ROL(a16 ^ D1, 45);
            ulong b18 = ROL(a17 ^ D2, 15);
            ulong b03 = ROL(a18 ^ D3, 21);
            ulong b13 = ROL(a19 ^ D4,  8);

            ulong b14 = ROL(a20 ^ D0, 18);
            ulong b24 = ROL(a21 ^ D1,  2);
            ulong b09 = ROL(a22 ^ D2, 61);
            ulong b19 = ROL(a23 ^ D3, 56);
            ulong b04 = ROL(a24 ^ D4, 14);

            // Chi + Iota
            a00 = b00 ^ ((~b01) & b02) ^ KECCAK_RC[r];
            a01 = b01 ^ ((~b02) & b03);
            a02 = b02 ^ ((~b03) & b04);
            a03 = b03 ^ ((~b04) & b00);
            a04 = b04 ^ ((~b00) & b01);

            a05 = b05 ^ ((~b06) & b07);
            a06 = b06 ^ ((~b07) & b08);
            a07 = b07 ^ ((~b08) & b09);
            a08 = b08 ^ ((~b09) & b05);
            a09 = b09 ^ ((~b05) & b06);

            a10 = b10 ^ ((~b11) & b12);
            a11 = b11 ^ ((~b12) & b13);
            a12 = b12 ^ ((~b13) & b14);
            a13 = b13 ^ ((~b14) & b10);
            a14 = b14 ^ ((~b10) & b11);

            a15 = b15 ^ ((~b16) & b17);
            a16 = b16 ^ ((~b17) & b18);
            a17 = b17 ^ ((~b18) & b19);
            a18 = b18 ^ ((~b19) & b15);
            a19 = b19 ^ ((~b15) & b16);

            a20 = b20 ^ ((~b21) & b22);
            a21 = b21 ^ ((~b22) & b23);
            a22 = b22 ^ ((~b23) & b24);
            a23 = b23 ^ ((~b24) & b20);
            a24 = b24 ^ ((~b20) & b21);
        }

        s0 = a00;
        s1 = a01;
        if (!n2) {
            s2 = a02;
            s3 = a03;
        }
    }

    tips[base + 0u] = s0;
    tips[base + 1u] = s1;
    if (n_lanes > 2u) {
        tips[base + 2u] = s2;
        tips[base + 3u] = s3;
    }
}
```

Result of previous attempt:
          w16_C64K: correct, 5.23 ms, 746.1 Gbitops/s (u64) (66.3% of 1125 Gops/s (u64 bitop, est))
          w64_C64K: correct, 21.63 ms, 721.3 Gbitops/s (u64) (64.1% of 1125 Gops/s (u64 bitop, est))
         w256_C64K: correct, 85.35 ms, 731.2 Gbitops/s (u64) (65.0% of 1125 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.6514

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

    // Load seed into the first n_lanes; rest start zero.
    ulong a00=0, a01=0, a02=0, a03=0, a04=0;
    ulong a05=0, a06=0, a07=0, a08=0, a09=0;
    ulong a10=0, a11=0, a12=0, a13=0, a14=0;
    ulong a15=0, a16=0, a17=0, a18=0, a19=0;
    ulong a20=0, a21=0, a22=0, a23=0, a24=0;

    // n_lanes is 2 or 4; load up to 4 lanes conditionally.
    a00 = seeds[base + 0u];
    a01 = seeds[base + 1u];
    if (n_lanes > 2u) {
        a02 = seeds[base + 2u];
        a03 = seeds[base + 3u];
    }

    // Pre-compute padding lane index/value.
    // SHA3 domain pad 0x06 goes at byte 0 of lane n_lanes; final pad 0x80 at byte 7 of lane 16.
    // n_lanes in {2,4}, so domain lane is a02 or a04.

    for (uint step = 0u; step < w; ++step) {
        // Zero lanes [n_lanes .. 24], apply padding.
        // n_lanes is 2 or 4. Lanes 4..24 always need zeroing except lane 16 gets 0x80<<56.
        if (n_lanes == 2u) {
            a02 = 0x06ul;
            a03 = 0ul;
            a04 = 0ul;
        } else {
            // n_lanes == 4
            a04 = 0x06ul;
        }
        a05 = 0ul; a06 = 0ul; a07 = 0ul; a08 = 0ul; a09 = 0ul;
        a10 = 0ul; a11 = 0ul; a12 = 0ul; a13 = 0ul; a14 = 0ul;
        a15 = 0ul;
        a16 = 0x8000000000000000ul;
        a17 = 0ul; a18 = 0ul; a19 = 0ul;
        a20 = 0ul; a21 = 0ul; a22 = 0ul; a23 = 0ul; a24 = 0ul;

        // 24 rounds of Keccak-f1600, fully unrolled.
        for (uint r = 0u; r < 24u; ++r) {
            // Theta
            ulong C0 = a00 ^ a05 ^ a10 ^ a15 ^ a20;
            ulong C1 = a01 ^ a06 ^ a11 ^ a16 ^ a21;
            ulong C2 = a02 ^ a07 ^ a12 ^ a17 ^ a22;
            ulong C3 = a03 ^ a08 ^ a13 ^ a18 ^ a23;
            ulong C4 = a04 ^ a09 ^ a14 ^ a19 ^ a24;

            ulong D0 = C4 ^ ROL(C1, 1);
            ulong D1 = C0 ^ ROL(C2, 1);
            ulong D2 = C1 ^ ROL(C3, 1);
            ulong D3 = C2 ^ ROL(C4, 1);
            ulong D4 = C3 ^ ROL(C0, 1);

            a00 ^= D0; a05 ^= D0; a10 ^= D0; a15 ^= D0; a20 ^= D0;
            a01 ^= D1; a06 ^= D1; a11 ^= D1; a16 ^= D1; a21 ^= D1;
            a02 ^= D2; a07 ^= D2; a12 ^= D2; a17 ^= D2; a22 ^= D2;
            a03 ^= D3; a08 ^= D3; a13 ^= D3; a18 ^= D3; a23 ^= D3;
            a04 ^= D4; a09 ^= D4; a14 ^= D4; a19 ^= D4; a24 ^= D4;

            // Rho + Pi: B[x_new + 5*y_new] = ROL(A[x + 5*y], rho[x+5*y])
            // Mapping (x,y) -> (y, (2x+3y)%5):
            // (0,0)->(0,0)  (1,0)->(0,2)  (2,0)->(0,4)  (3,0)->(0,1)  (4,0)->(0,3)
            // (0,1)->(1,3)  (1,1)->(1,0)  (2,1)->(1,2)  (3,1)->(1,4)  (4,1)->(1,1)
            // (0,2)->(2,1)  (1,2)->(2,3)  (2,2)->(2,0)  (3,2)->(2,2)  (4,2)->(2,4)
            // (0,3)->(3,4)  (1,3)->(3,1)  (2,3)->(3,3)  (3,3)->(3,0)  (4,3)->(3,2)
            // (0,4)->(4,2)  (1,4)->(4,4)  (2,4)->(4,1)  (3,4)->(4,3)  (4,4)->(4,0)
            ulong b00 = ROL(a00,  0);
            ulong b10 = ROL(a01,  1);
            ulong b20 = ROL(a02, 62);
            ulong b05 = ROL(a03, 28);
            ulong b15 = ROL(a04, 27);

            ulong b16 = ROL(a05, 36);
            ulong b01 = ROL(a06, 44);
            ulong b11 = ROL(a07,  6);
            ulong b21 = ROL(a08, 55);
            ulong b06 = ROL(a09, 20);

            ulong b07 = ROL(a10,  3);
            ulong b17 = ROL(a11, 10);
            ulong b02 = ROL(a12, 43);
            ulong b12 = ROL(a13, 25);
            ulong b22 = ROL(a14, 39);

            ulong b23 = ROL(a15, 41);
            ulong b08 = ROL(a16, 45);
            ulong b18 = ROL(a17, 15);
            ulong b03 = ROL(a18, 21);
            ulong b13 = ROL(a19,  8);

            ulong b14 = ROL(a20, 18);
            ulong b24 = ROL(a21,  2);
            ulong b09 = ROL(a22, 61);
            ulong b19 = ROL(a23, 56);
            ulong b04 = ROL(a24, 14);

            // Chi
            a00 = b00 ^ ((~b01) & b02);
            a01 = b01 ^ ((~b02) & b03);
            a02 = b02 ^ ((~b03) & b04);
            a03 = b03 ^ ((~b04) & b00);
            a04 = b04 ^ ((~b00) & b01);

            a05 = b05 ^ ((~b06) & b07);
            a06 = b06 ^ ((~b07) & b08);
            a07 = b07 ^ ((~b08) & b09);
            a08 = b08 ^ ((~b09) & b05);
            a09 = b09 ^ ((~b05) & b06);

            a10 = b10 ^ ((~b11) & b12);
            a11 = b11 ^ ((~b12) & b13);
            a12 = b12 ^ ((~b13) & b14);
            a13 = b13 ^ ((~b14) & b10);
            a14 = b14 ^ ((~b10) & b11);

            a15 = b15 ^ ((~b16) & b17);
            a16 = b16 ^ ((~b17) & b18);
            a17 = b17 ^ ((~b18) & b19);
            a18 = b18 ^ ((~b19) & b15);
            a19 = b19 ^ ((~b15) & b16);

            a20 = b20 ^ ((~b21) & b22);
            a21 = b21 ^ ((~b22) & b23);
            a22 = b22 ^ ((~b23) & b24);
            a23 = b23 ^ ((~b24) & b20);
            a24 = b24 ^ ((~b20) & b21);

            // Iota
            a00 ^= KECCAK_RC[r];
        }
    }

    tips[base + 0u] = a00;
    tips[base + 1u] = a01;
    if (n_lanes > 2u) {
        tips[base + 2u] = a02;
        tips[base + 3u] = a03;
    }
}
```

Incumbent result:
          w16_C64K: correct, 5.07 ms, 769.7 Gbitops/s (u64) (68.4% of 1125 Gops/s (u64 bitop, est))
          w64_C64K: correct, 20.25 ms, 770.5 Gbitops/s (u64) (68.5% of 1125 Gops/s (u64 bitop, est))
         w256_C64K: correct, 82.64 ms, 755.2 Gbitops/s (u64) (67.1% of 1125 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.6801

## History

- iter  0: compile=OK | correct=True | score=0.039473365596991154
- iter  1: compile=OK | correct=True | score=0.6800693564755079
- iter  2: compile=OK | correct=False | score=N/A
- iter  3: compile=OK | correct=True | score=0.6525438072005288
- iter  4: compile=OK | correct=True | score=0.6765303179374582
- iter  5: compile=OK | correct=True | score=0.6513695103243016

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
