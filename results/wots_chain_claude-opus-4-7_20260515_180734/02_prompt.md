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
    return (x << k) | (x >> (64u - k));
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

    // Load seed into first n_lanes lanes; rest start at 0.
    ulong s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    s0 = seeds[base + 0];
    s1 = seeds[base + 1];
    if (n_lanes >= 4u) {
        s2 = seeds[base + 2];
        s3 = seeds[base + 3];
    }

    uint W = w;
    for (uint step = 0u; step < W; ++step) {
        // Initialize state: lanes 0..n_lanes-1 from prev chunk; rest 0.
        // Apply SHA3 domain pad at lane n_lanes (byte 0 = 0x06) and
        // FIPS 202 final pad at lane 16 (byte 7 = 0x80).
        ulong A0 = s0;
        ulong A1 = s1;
        ulong A2, A3;
        ulong pad_lane;
        if (n_lanes == 2u) {
            A2 = 0x06ul;       // domain pad at lane 2
            A3 = 0ul;
            pad_lane = 0ul;
        } else {
            A2 = s2;
            A3 = s3;
            pad_lane = 0x06ul; // domain pad at lane 4
        }
        ulong A4 = pad_lane;
        ulong A5 = 0, A6 = 0, A7 = 0, A8 = 0, A9 = 0;
        ulong A10 = 0, A11 = 0, A12 = 0, A13 = 0, A14 = 0;
        ulong A15 = 0;
        ulong A16 = 0x8000000000000000ul; // final pad
        ulong A17 = 0, A18 = 0, A19 = 0;
        ulong A20 = 0, A21 = 0, A22 = 0, A23 = 0, A24 = 0;

        // 24 rounds of Keccak-f[1600], fully unrolled.
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

            // rho + pi: B[dst] = ROTL(A[src], rho[src])
            // Mapping: dst = y + 5*((2*x + 3*y) % 5), src = x + 5*y
            ulong B0  = A0;                  // (0,0) rho=0
            ulong B16 = ROTL(A1,  1);        // src(1,0) -> (0,2*0=0->y_new=(2+0)%5=2)? Let me recompute.
            // Recompute carefully using standard table:
            // src(x,y) -> dst(y, (2x+3y)%5), rho[x+5y]
            // We'll just enumerate all 25:
            ulong T;
            // (x=0,y=0): src=0, dst=0, rho=0
            B0 = A0;
            // (x=1,y=0): src=1, dst=(0,2)=10, rho=1
            T = ROTL(A1, 1);  ulong B10 = T;
            // (x=2,y=0): src=2, dst=(0,4)=20, rho=62
            T = ROTL(A2, 62); ulong B20 = T;
            // (x=3,y=0): src=3, dst=(0,1)=5,  rho=28
            T = ROTL(A3, 28); ulong B5 = T;
            // (x=4,y=0): src=4, dst=(0,3)=15, rho=27
            T = ROTL(A4, 27); ulong B15 = T;
            // (x=0,y=1): src=5, dst=(1,3)=16, rho=36
            T = ROTL(A5, 36); ulong B16v = T;
            // (x=1,y=1): src=6, dst=(1,0)=1,  rho=44
            T = ROTL(A6, 44); ulong B1 = T;
            // (x=2,y=1): src=7, dst=(1,2)=11, rho=6
            T = ROTL(A7, 6);  ulong B11 = T;
            // (x=3,y=1): src=8, dst=(1,4)=21, rho=55
            T = ROTL(A8, 55); ulong B21 = T;
            // (x=4,y=1): src=9, dst=(1,1)=6,  rho=20
            T = ROTL(A9, 20); ulong B6 = T;
            // (x=0,y=2): src=10, dst=(2,1)=7,  rho=3
            T = ROTL(A10, 3); ulong B7 = T;
            // (x=1,y=2): src=11, dst=(2,3)=17, rho=10
            T = ROTL(A11, 10); ulong B17 = T;
            // (x=2,y=2): src=12, dst=(2,0)=2,  rho=43
            T = ROTL(A12, 43); ulong B2 = T;
            // (x=3,y=2): src=13, dst=(2,2)=12, rho=25
            T = ROTL(A13, 25); ulong B12 = T;
            // (x=4,y=2): src=14, dst=(2,4)=22, rho=39
            T = ROTL(A14, 39); ulong B22 = T;
            // (x=0,y=3): src=15, dst=(3,4)=23, rho=41
            T = ROTL(A15, 41); ulong B23 = T;
            // (x=1,y=3): src=16, dst=(3,1)=8,  rho=45
            T = ROTL(A16, 45); ulong B8 = T;
            // (x=2,y=3): src=17, dst=(3,3)=18, rho=15
            T = ROTL(A17, 15); ulong B18 = T;
            // (x=3,y=3): src=18, dst=(3,0)=3,  rho=21
            T = ROTL(A18, 21); ulong B3 = T;
            // (x=4,y=3): src=19, dst=(3,2)=13, rho=8
            T = ROTL(A19, 8);  ulong B13 = T;
            // (x=0,y=4): src=20, dst=(4,2)=14, rho=18
            T = ROTL(A20, 18); ulong B14 = T;
            // (x=1,y=4): src=21, dst=(4,4)=24, rho=2
            T = ROTL(A21, 2);  ulong B24 = T;
            // (x=2,y=4): src=22, dst=(4,1)=9,  rho=61
            T = ROTL(A22, 61); ulong B9 = T;
            // (x=3,y=4): src=23, dst=(4,3)=19, rho=56
            T = ROTL(A23, 56); ulong B19 = T;
            // (x=4,y=4): src=24, dst=(4,0)=4,  rho=14
            T = ROTL(A24, 14); ulong B4 = T;

            // Use B16v (real value for dst 16); discard initial B16 placeholder.
            ulong B16f = B16v;

            // chi: A[i] = B[i] ^ ((~B[i+1]) & B[i+2]) within each row
            A0  = B0  ^ ((~B1)  & B2);
            A1  = B1  ^ ((~B2)  & B3);
            A2  = B2  ^ ((~B3)  & B4);
            A3  = B3  ^ ((~B4)  & B0);
            A4  = B4  ^ ((~B0)  & B1);

            A5  = B5  ^ ((~B6)  & B7);
            A6  = B6  ^ ((~B7)  & B8);
            A7  = B7  ^ ((~B8)  & B9);
            A8  = B8  ^ ((~B9)  & B5);
            A9  = B9  ^ ((~B5)  & B6);

            A10 = B10 ^ ((~B11) & B12);
            A11 = B11 ^ ((~B12) & B13);
            A12 = B12 ^ ((~B13) & B14);
            A13 = B13 ^ ((~B14) & B10);
            A14 = B14 ^ ((~B10) & B11);

            A15 = B15 ^ ((~B16f) & B17);
            A16 = B16f ^ ((~B17) & B18);
            A17 = B17 ^ ((~B18) & B19);
            A18 = B18 ^ ((~B19) & B15);
            A19 = B19 ^ ((~B15) & B16f);

            A20 = B20 ^ ((~B21) & B22);
            A21 = B21 ^ ((~B22) & B23);
            A22 = B22 ^ ((~B23) & B24);
            A23 = B23 ^ ((~B24) & B20);
            A24 = B24 ^ ((~B20) & B21);

            // iota
            A0 ^= KECCAK_RC[r];
        }

        // Carry first n_lanes lanes into next chain step.
        s0 = A0;
        s1 = A1;
        if (n_lanes >= 4u) {
            s2 = A2;
            s3 = A3;
        }
    }

    tips[base + 0] = s0;
    tips[base + 1] = s1;
    if (n_lanes >= 4u) {
        tips[base + 2] = s2;
        tips[base + 3] = s3;
    }
}
```

Result of previous attempt:
          w16_C64K: correct, 5.10 ms, 764.4 Gbitops/s (u64) (132.4% of 577 Gops/s (u64 bitop, est))
          w64_C64K: correct, 20.38 ms, 765.7 Gbitops/s (u64) (132.6% of 577 Gops/s (u64 bitop, est))
         w256_C64K: correct, 81.46 ms, 766.1 Gbitops/s (u64) (132.7% of 577 Gops/s (u64 bitop, est))
  score (gmean of fraction): 1.3256

## History

- iter  0: compile=OK | correct=True | score=0.07723052060911632
- iter  1: compile=OK | correct=True | score=1.325610072035793

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
