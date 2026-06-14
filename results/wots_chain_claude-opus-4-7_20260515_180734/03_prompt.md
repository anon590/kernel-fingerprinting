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

inline ulong ROTL(ulong x, uint k) {
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

    ulong s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    s0 = seeds[base + 0];
    s1 = seeds[base + 1];
    bool wide = (n_lanes >= 4u);
    if (wide) {
        s2 = seeds[base + 2];
        s3 = seeds[base + 3];
    }

    // Precompute the two pad-lane positions for the initial state.
    // n_lanes==2: domain pad goes into lane 2 (A2 ^= 0x06)
    // n_lanes==4: domain pad goes into lane 4 (A4 ^= 0x06)
    // Final pad always at lane 16, byte 7 -> 0x8000000000000000.
    ulong pad_A2 = wide ? 0ul : 0x06ul;
    ulong pad_A4 = wide ? 0x06ul : 0ul;

    uint W = w;
    for (uint step = 0u; step < W; ++step) {
        ulong A0 = s0;
        ulong A1 = s1;
        ulong A2 = wide ? s2 : pad_A2;
        ulong A3 = wide ? s3 : 0ul;
        ulong A4 = pad_A4;
        ulong A5 = 0, A6 = 0, A7 = 0, A8 = 0, A9 = 0;
        ulong A10 = 0, A11 = 0, A12 = 0, A13 = 0, A14 = 0;
        ulong A15 = 0;
        ulong A16 = 0x8000000000000000ul;
        ulong A17 = 0, A18 = 0, A19 = 0;
        ulong A20 = 0, A21 = 0, A22 = 0, A23 = 0, A24 = 0;

        for (uint r = 0u; r < 24u; ++r) {
            // theta
            ulong C0 = A0 ^ A5 ^ A10 ^ A15 ^ A20;
            ulong C1 = A1 ^ A6 ^ A11 ^ A16 ^ A21;
            ulong C2 = A2 ^ A7 ^ A12 ^ A17 ^ A22;
            ulong C3 = A3 ^ A8 ^ A13 ^ A18 ^ A23;
            ulong C4 = A4 ^ A9 ^ A14 ^ A19 ^ A24;

            ulong D0 = C4 ^ ROTL(C1, 1);
            ulong D1 = C0 ^ ROTL(C2, 1);
            ulong D2 = C1 ^ ROTL(C3, 1);
            ulong D3 = C2 ^ ROTL(C4, 1);
            ulong D4 = C3 ^ ROTL(C0, 1);

            A0 ^= D0;  A1 ^= D1;  A2 ^= D2;  A3 ^= D3;  A4 ^= D4;
            A5 ^= D0;  A6 ^= D1;  A7 ^= D2;  A8 ^= D3;  A9 ^= D4;
            A10 ^= D0; A11 ^= D1; A12 ^= D2; A13 ^= D3; A14 ^= D4;
            A15 ^= D0; A16 ^= D1; A17 ^= D2; A18 ^= D3; A19 ^= D4;
            A20 ^= D0; A21 ^= D1; A22 ^= D2; A23 ^= D3; A24 ^= D4;

            // rho + pi
            ulong B0  = A0;
            ulong B10 = ROTL(A1,  1);
            ulong B20 = ROTL(A2,  62);
            ulong B5  = ROTL(A3,  28);
            ulong B15 = ROTL(A4,  27);
            ulong B16 = ROTL(A5,  36);
            ulong B1  = ROTL(A6,  44);
            ulong B11 = ROTL(A7,  6);
            ulong B21 = ROTL(A8,  55);
            ulong B6  = ROTL(A9,  20);
            ulong B7  = ROTL(A10, 3);
            ulong B17 = ROTL(A11, 10);
            ulong B2  = ROTL(A12, 43);
            ulong B12 = ROTL(A13, 25);
            ulong B22 = ROTL(A14, 39);
            ulong B23 = ROTL(A15, 41);
            ulong B8  = ROTL(A16, 45);
            ulong B18 = ROTL(A17, 15);
            ulong B3  = ROTL(A18, 21);
            ulong B13 = ROTL(A19, 8);
            ulong B14 = ROTL(A20, 18);
            ulong B24 = ROTL(A21, 2);
            ulong B9  = ROTL(A22, 61);
            ulong B19 = ROTL(A23, 56);
            ulong B4  = ROTL(A24, 14);

            // chi: A[i] = B[i] ^ (B[i+2] & ~B[i+1])  -- expresses andn for the compiler
            A0  = B0  ^ (B2  & ~B1);
            A1  = B1  ^ (B3  & ~B2);
            A2  = B2  ^ (B4  & ~B3);
            A3  = B3  ^ (B0  & ~B4);
            A4  = B4  ^ (B1  & ~B0);

            A5  = B5  ^ (B7  & ~B6);
            A6  = B6  ^ (B8  & ~B7);
            A7  = B7  ^ (B9  & ~B8);
            A8  = B8  ^ (B5  & ~B9);
            A9  = B9  ^ (B6  & ~B5);

            A10 = B10 ^ (B12 & ~B11);
            A11 = B11 ^ (B13 & ~B12);
            A12 = B12 ^ (B14 & ~B13);
            A13 = B13 ^ (B10 & ~B14);
            A14 = B14 ^ (B11 & ~B10);

            A15 = B15 ^ (B17 & ~B16);
            A16 = B16 ^ (B18 & ~B17);
            A17 = B17 ^ (B19 & ~B18);
            A18 = B18 ^ (B15 & ~B19);
            A19 = B19 ^ (B16 & ~B15);

            A20 = B20 ^ (B22 & ~B21);
            A21 = B21 ^ (B23 & ~B22);
            A22 = B22 ^ (B24 & ~B23);
            A23 = B23 ^ (B20 & ~B24);
            A24 = B24 ^ (B21 & ~B20);

            // iota
            A0 ^= KECCAK_RC[r];
        }

        s0 = A0;
        s1 = A1;
        if (wide) {
            s2 = A2;
            s3 = A3;
        }
    }

    tips[base + 0] = s0;
    tips[base + 1] = s1;
    if (wide) {
        tips[base + 2] = s2;
        tips[base + 3] = s3;
    }
}
```

Result of previous attempt:
          w16_C64K: correct, 5.06 ms, 770.8 Gbitops/s (u64) (133.5% of 577 Gops/s (u64 bitop, est))
          w64_C64K: correct, 20.20 ms, 772.3 Gbitops/s (u64) (133.8% of 577 Gops/s (u64 bitop, est))
         w256_C64K: correct, 80.76 ms, 772.8 Gbitops/s (u64) (133.8% of 577 Gops/s (u64 bitop, est))
  score (gmean of fraction): 1.3370

## History

- iter  0: compile=OK | correct=True | score=0.07723052060911632
- iter  1: compile=OK | correct=True | score=1.325610072035793
- iter  2: compile=OK | correct=True | score=1.3369883064267558

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
