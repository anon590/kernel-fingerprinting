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

static inline ulong ROTL(ulong x, uint k) {
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

    ulong s0 = seeds[base + 0];
    ulong s1 = seeds[base + 1];
    ulong s2 = 0, s3 = 0;
    bool wide = (n_lanes >= 4u);
    if (wide) {
        s2 = seeds[base + 2];
        s3 = seeds[base + 3];
    }

    ulong pad_A2 = wide ? 0ul : 0x06ul;
    ulong pad_A4 = wide ? 0x06ul : 0ul;

    const ulong ONES = ~0ul;
    uint W = w;

    for (uint step = 0u; step < W; ++step) {
        // Initial absorb. Apply lane-complementing transform on entry.
        // Complemented lane set (standard Keccak): {1, 2, 8, 12, 17, 20}
        ulong A0  = s0;
        ulong A1  = ~s1;                                    // complemented
        ulong A2  = ~(wide ? s2 : pad_A2);                  // complemented
        ulong A3  = wide ? s3 : 0ul;
        ulong A4  = pad_A4;
        ulong A5  = 0, A6 = 0, A7 = 0;
        ulong A8  = ONES;                                   // complemented (was 0)
        ulong A9  = 0;
        ulong A10 = 0, A11 = 0;
        ulong A12 = ONES;                                   // complemented (was 0)
        ulong A13 = 0, A14 = 0, A15 = 0;
        ulong A16 = 0x8000000000000000ul;
        ulong A17 = ONES;                                   // complemented (was 0)
        ulong A18 = 0, A19 = 0;
        ulong A20 = ONES;                                   // complemented (was 0)
        ulong A21 = 0, A22 = 0, A23 = 0, A24 = 0;

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

            // rho + pi, fused with theta D-xor
            ulong B0  = A0  ^ D0;
            ulong B10 = ROTL(A1  ^ D1, 1);
            ulong B20 = ROTL(A2  ^ D2, 62);
            ulong B5  = ROTL(A3  ^ D3, 28);
            ulong B15 = ROTL(A4  ^ D4, 27);
            ulong B16 = ROTL(A5  ^ D0, 36);
            ulong B1  = ROTL(A6  ^ D1, 44);
            ulong B11 = ROTL(A7  ^ D2, 6);
            ulong B21 = ROTL(A8  ^ D3, 55);
            ulong B6  = ROTL(A9  ^ D4, 20);
            ulong B7  = ROTL(A10 ^ D0, 3);
            ulong B17 = ROTL(A11 ^ D1, 10);
            ulong B2  = ROTL(A12 ^ D2, 43);
            ulong B12 = ROTL(A13 ^ D3, 25);
            ulong B22 = ROTL(A14 ^ D4, 39);
            ulong B23 = ROTL(A15 ^ D0, 41);
            ulong B8  = ROTL(A16 ^ D1, 45);
            ulong B18 = ROTL(A17 ^ D2, 15);
            ulong B3  = ROTL(A18 ^ D3, 21);
            ulong B13 = ROTL(A19 ^ D4, 8);
            ulong B14 = ROTL(A20 ^ D0, 18);
            ulong B24 = ROTL(A21 ^ D1, 2);
            ulong B9  = ROTL(A22 ^ D2, 61);
            ulong B19 = ROTL(A23 ^ D3, 56);
            ulong B4  = ROTL(A24 ^ D4, 14);

            // chi with lane-complementing transform.
            // Original complemented set on B (= A image at this point under pi):
            //   On A: {1,2,8,12,17,20}. Under pi: A[x,y] -> B[y, 2x+3y].
            //   The B-indices in our (B0..B24) naming correspond directly to
            //   lane index k = x + 5y. We track which B[k] holds complemented
            //   value relative to the "true" state. Equivalently, we use the
            //   standard formulas from Keccak reference (Bertoni et al.):
            //
            //   A0  = B0  ^ ( B1 | B2)
            //   A1  = B1  ^ (~B2 | B3)
            //   A2  = B2  ^ ( B3 & B4)
            //   A3  = B3  | ( B4 ^ B0)    -- careful, see below
            //   ...
            // To avoid subtle errors, we use the well-known per-row pattern:
            //
            // Row y (lanes 5y .. 5y+4): let b0..b4 = B[5y..5y+4].
            // With complementing set {1,2,8,12,17,20} on A (before pi),
            // after pi the complemented lanes in B-space (k = x + 5y) are:
            //   from A1  (x=1,y=0) -> B[0 + 5*(2+0)] = B10
            //   from A2  (x=2,y=0) -> B[0 + 5*(4+0)] = B20
            //   from A8  (x=3,y=1) -> B[1 + 5*(2*3+3)%5] = B[1 + 5*4] = B21
            //         (2x+3y = 6+3 = 9 mod 5 = 4)
            //   from A12 (x=2,y=2) -> B[2 + 5*(4+6)%5] = B[2 + 5*0] = B2
            //   from A17 (x=2,y=3) -> B[3 + 5*(4+9)%5] = B[3 + 5*3] = B18
            //   from A20 (x=0,y=4) -> B[4 + 5*(0+12)%5] = B[4 + 5*2] = B14
            //
            // Complemented B lanes: {10, 20, 21, 2, 18, 14}.
            // chi: A'[i] = B[i] ^ ((~B[i+1]) & B[i+2])
            // If a lane is complemented, ~B[i] becomes B[i] (without ~).
            //
            // We rewrite each chi expression to absorb the NOTs into the
            // constants by toggling between AND and OR via De Morgan:
            //   ~b1 & b2  with b1 complemented -> b1 & b2  (drop the ~)
            //   ~b1 & b2  with b2 complemented -> ~(b1 | ~b2)... too messy.
            //
            // Simpler: precompute notB only for the lanes that need it, and
            // for the complemented lanes use B directly (since ~complemented=
            // true value). The output A' is in the SAME complemented domain
            // (same set of lanes carry complemented values across rounds).

            // Compute "logical NOT" view: nB[k] = ~B[k] if k NOT in compl set,
            // else B[k] (which already equals ~true_value... wait, careful).
            //
            // Define T[k] = true value of lane k. Then B[k] = T[k] for k not
            // in compl_B, and B[k] = ~T[k] for k in compl_B.
            // chi true:  T'[i] = T[i] ^ ((~T[i+1]) & T[i+2])
            // Substitute:
            //   ~T[i+1] = B[i+1] if (i+1) in compl_B, else ~B[i+1]
            //    T[i+2] = B[i+2] if (i+2) not in compl_B, else ~B[i+2]
            //    T[i]   = B[i]   if (i)   not in compl_B, else ~B[i]
            // We then RE-complement the output if i is in compl_B (since we
            // store A in complemented domain again for next round).

            // compl_B = {2, 10, 14, 18, 20, 21}
            // For simplicity and correctness, compute true values first, then
            // re-complement the {1,2,8,12,17,20} A-lanes at end-of-round.
            //
            // Convert B -> trueB:
            ulong tB0=B0, tB1=B1, tB2=~B2, tB3=B3, tB4=B4;
            ulong tB5=B5, tB6=B6, tB7=B7, tB8=B8, tB9=B9;
            ulong tB10=~B10, tB11=B11, tB12=B12, tB13=B13, tB14=~B14;
            ulong tB15=B15, tB16=B16, tB17=B17, tB18=~B18, tB19=B19;
            ulong tB20=~B20, tB21=~B21, tB22=B22, tB23=B23, tB24=B24;

            // chi (true domain)
            ulong tA0  = tB0  ^ (~tB1  & tB2);
            ulong tA1  = tB1  ^ (~tB2  & tB3);
            ulong tA2  = tB2  ^ (~tB3  & tB4);
            ulong tA3  = tB3  ^ (~tB4  & tB0);
            ulong tA4  = tB4  ^ (~tB0  & tB1);

            ulong tA5  = tB5  ^ (~tB6  & tB7);
            ulong tA6  = tB6  ^ (~tB7  & tB8);
            ulong tA7  = tB7  ^ (~tB8  & tB9);
            ulong tA8  = tB8  ^ (~tB9  & tB5);
            ulong tA9  = tB9  ^ (~tB5  & tB6);

            ulong tA10 = tB10 ^ (~tB11 & tB12);
            ulong tA11 = tB11 ^ (~tB12 & tB13);
            ulong tA12 = tB12 ^ (~tB13 & tB14);
            ulong tA13 = tB13 ^ (~tB14 & tB10);
            ulong tA14 = tB14 ^ (~tB10 & tB11);

            ulong tA15 = tB15 ^ (~tB16 & tB17);
            ulong tA16 = tB16 ^ (~tB17 & tB18);
            ulong tA17 = tB17 ^ (~tB18 & tB19);
            ulong tA18 = tB18 ^ (~tB19 & tB15);
            ulong tA19 = tB19 ^ (~tB15 & tB16);

            ulong tA20 = tB20 ^ (~tB21 & tB22);
            ulong tA21 = tB21 ^ (~tB22 & tB23);
            ulong tA22 = tB22 ^ (~tB23 & tB24);
            ulong tA23 = tB23 ^ (~tB24 & tB20);
            ulong tA24 = tB24 ^ (~tB20 & tB21);

            // iota on lane 0
            tA0 ^= KECCAK_RC[r];

            // Re-complement for next round on A-set {1,2,8,12,17,20}
            A0  = tA0;
            A1  = ~tA1;
            A2  = ~tA2;
            A3  = tA3;
            A4  = tA4;
            A5  = tA5;
            A6  = tA6;
            A7  = tA7;
            A8  = ~tA8;
            A9  = tA9;
            A10 = tA10;
            A11 = tA11;
            A12 = ~tA12;
            A13 = tA13;
            A14 = tA14;
            A15 = tA15;
            A16 = tA16;
            A17 = ~tA17;
            A18 = tA18;
            A19 = tA19;
            A20 = ~tA20;
            A21 = tA21;
            A22 = tA22;
            A23 = tA23;
            A24 = tA24;
        }

        // Un-complement on output lanes we read
        s0 = A0;
        s1 = ~A1;
        if (wide) {
            s2 = ~A2;
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

Incumbent result:
          w16_C64K: correct, 5.06 ms, 770.8 Gbitops/s (u64) (133.5% of 577 Gops/s (u64 bitop, est))
          w64_C64K: correct, 20.20 ms, 772.3 Gbitops/s (u64) (133.8% of 577 Gops/s (u64 bitop, est))
         w256_C64K: correct, 80.76 ms, 772.8 Gbitops/s (u64) (133.8% of 577 Gops/s (u64 bitop, est))
  score (gmean of fraction): 1.3370

## History

- iter  0: compile=OK | correct=True | score=0.07723052060911632
- iter  1: compile=OK | correct=True | score=1.325610072035793
- iter  2: compile=OK | correct=True | score=1.3369883064267558
- iter  3: compile=OK | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
