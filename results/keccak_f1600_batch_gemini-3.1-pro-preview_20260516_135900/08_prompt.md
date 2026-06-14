## Task: keccak_f1600_batch

Batched Keccak-f[1600] sponge over fixed-length messages. Each of ``batch`` independent instances absorbs ``msg_bytes`` bytes of input, applies the standard FIPS 202 padding, runs the 24-round Keccak-f[1600] permutation, and squeezes ``out_bytes`` bytes of output. All test sizes satisfy ``msg_bytes < rate_bytes`` (single absorb block) and ``msg_bytes``, ``rate_bytes``, ``out_bytes`` are all multiples of 8, so the host packs message and output as ``ulong`` arrays.

State convention: the 1600-bit state is a 5x5 array of 64-bit lanes; lane k (for k in 0..25) corresponds to byte positions 8*k .. 8*k + 7 of the sponge state in little-endian, i.e. lane k holds bytes at the (x, y) cell with x = k % 5 and y = k / 5. The seed shows the standard round constants ``RC[24]`` and rho offsets ``r[x][y]`` from FIPS 202.

Permutation: 24 rounds of theta -> rho -> pi -> chi -> iota as defined in FIPS 202. Concretely, with A the (5,5) state of 64-bit lanes:
  theta:  C[x]      = A[x,0] ^ A[x,1] ^ A[x,2] ^ A[x,3] ^ A[x,4];
          D[x]      = C[x-1] ^ rotl(C[x+1], 1);
          A[x,y]   ^= D[x].
  rho:    A'[x,y]   = rotl(A[x,y], r[x][y]).
  pi:     A''[y, (2*x + 3*y) %% 5] = A'[x, y]
          (equivalently A''[x, y] = A'[(x + 3*y) %% 5, x]).
  chi:    A'''[x,y] = A''[x,y] ^ ((~A''[(x+1)%%5, y]) & A''[(x+2)%%5, y]).
  iota:   A''''[0,0] = A'''[0,0] ^ RC[round].

Sponge protocol (msg_bytes < rate_bytes, single absorb block):
  1. Initialise the state to zero.
  2. XOR ``msg_bytes / 8`` input lanes into state lanes      0 .. msg_bytes/8 - 1 (little-endian byte stream).
  3. XOR the domain byte (low 8 bits of ``domain``) into      byte position ``msg_bytes`` (lane ``msg_bytes/8``,      byte 0 of that lane).
  4. XOR 0x80 into byte position ``rate_bytes - 1``      (lane ``rate_bytes/8 - 1``, byte 7 of that lane).
  5. Apply Keccak-f[1600].
  6. Output the first ``rate_bytes / 8`` lanes of state.
  7. If more output is needed, apply Keccak-f[1600] again      and output the next ``rate_bytes / 8`` lanes; repeat      until ``out_bytes / 8`` lanes have been written. The      final chunk may be shorter than the rate.

In-distribution sizes use the SHA3-256 mode (rate=136, domain=0x06, out=32); the held-out size uses SHAKE128 (rate=168, domain=0x1F, out=256, requires multiple squeeze permutations). The kernel must use the runtime values of ``rate_bytes``, ``out_bytes`` and ``domain`` rather than compile-time constants. Correctness is bit-exact against ``hashlib.sha3_256`` / ``hashlib.shake_128``; any mismatched output ulong rejects the candidate.

## Required kernel signature(s)

```
kernel void keccak_f1600_batch(
    device const ulong *in_data    [[buffer(0)]],
    device       ulong *out_data   [[buffer(1)]],
    constant uint      &batch      [[buffer(2)]],
    constant uint      &msg_bytes  [[buffer(3)]],
    constant uint      &rate_bytes [[buffer(4)]],
    constant uint      &out_bytes  [[buffer(5)]],
    constant uint      &domain     [[buffer(6)]],
    uint idx [[thread_position_in_grid]]);

Dispatch (host-fixed):
  threadsPerGrid        = (batch, 1, 1)
  threadsPerThreadgroup = (min(batch, 64), 1, 1)
Each thread processes ONE instance end-to-end; guard against idx >= batch (the grid is rounded up to a multiple of the TG width). All test sizes have msg_bytes = 32. ``in_data`` is laid out as batch consecutive runs of ``msg_bytes / 8`` ulongs; ``out_data`` as batch consecutive runs of ``out_bytes / 8`` ulongs. Threadgroup-cooperative and simdgroup-cooperative implementations are valid so long as the external buffer layout above is preserved.
```

## Your previous attempt

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant ulong KECCAK_RC[24] = {
    0x0000000000000001ul, 0x0000000000008082ul, 0x800000000000808Aul,
    0x8000000080008000ul, 0x000000000000808Bul, 0x0000000080000001ul,
    0x8000000080008081ul, 0x8000000000008009ul, 0x000000000000008Aul,
    0x0000000000000088ul, 0x0000000080008009ul, 0x000000008000000Aul,
    0x000000008000808Bul, 0x800000000000008Bul, 0x8000000000008089ul,
    0x8000000000008003ul, 0x8000000000008002ul, 0x8000000000000080ul,
    0x000000000000800Aul, 0x800000008000000Aul, 0x8000000080008081ul,
    0x8000000000008080ul, 0x0000000080000001ul, 0x8000000080008008ul,
};

inline __attribute__((always_inline)) ulong rotl64(ulong x, uint k) {
    return (x << k) | (x >> (64u - k));
}

// Computes 1 full Keccak round combining Theta, Rho, Pi, Chi, and Iota.
// Replaces in-place permutation moves with an explicit bipartite assignment.
#define ROUND(RC) do { \
    ulong C0 = A0 ^ A5 ^ A10 ^ A15 ^ A20; \
    ulong C1 = A1 ^ A6 ^ A11 ^ A16 ^ A21; \
    ulong C2 = A2 ^ A7 ^ A12 ^ A17 ^ A22; \
    ulong C3 = A3 ^ A8 ^ A13 ^ A18 ^ A23; \
    ulong C4 = A4 ^ A9 ^ A14 ^ A19 ^ A24; \
    \
    ulong D0 = C4 ^ rotl64(C1, 1u); \
    ulong D1 = C0 ^ rotl64(C2, 1u); \
    ulong D2 = C1 ^ rotl64(C3, 1u); \
    ulong D3 = C2 ^ rotl64(C4, 1u); \
    ulong D4 = C3 ^ rotl64(C0, 1u); \
    \
    ulong B0  = A0 ^ D0; \
    ulong B1  = rotl64(A1 ^ D1, 1u); \
    ulong B2  = rotl64(A2 ^ D2, 62u); \
    ulong B3  = rotl64(A3 ^ D3, 28u); \
    ulong B4  = rotl64(A4 ^ D4, 27u); \
    ulong B5  = rotl64(A5 ^ D0, 36u); \
    ulong B6  = rotl64(A6 ^ D1, 44u); \
    ulong B7  = rotl64(A7 ^ D2, 6u); \
    ulong B8  = rotl64(A8 ^ D3, 55u); \
    ulong B9  = rotl64(A9 ^ D4, 20u); \
    ulong B10 = rotl64(A10 ^ D0, 3u); \
    ulong B11 = rotl64(A11 ^ D1, 10u); \
    ulong B12 = rotl64(A12 ^ D2, 43u); \
    ulong B13 = rotl64(A13 ^ D3, 25u); \
    ulong B14 = rotl64(A14 ^ D4, 39u); \
    ulong B15 = rotl64(A15 ^ D0, 41u); \
    ulong B16 = rotl64(A16 ^ D1, 45u); \
    ulong B17 = rotl64(A17 ^ D2, 15u); \
    ulong B18 = rotl64(A18 ^ D3, 21u); \
    ulong B19 = rotl64(A19 ^ D4, 8u); \
    ulong B20 = rotl64(A20 ^ D0, 18u); \
    ulong B21 = rotl64(A21 ^ D1, 2u); \
    ulong B22 = rotl64(A22 ^ D2, 61u); \
    ulong B23 = rotl64(A23 ^ D3, 56u); \
    ulong B24 = rotl64(A24 ^ D4, 14u); \
    \
    A0 = B0 ^ (~B6 & B12) ^ RC; \
    A1 = B6 ^ (~B12 & B18); \
    A2 = B12 ^ (~B18 & B24); \
    A3 = B18 ^ (~B24 & B0); \
    A4 = B24 ^ (~B0 & B6); \
    \
    A5 = B3 ^ (~B9 & B10); \
    A6 = B9 ^ (~B10 & B16); \
    A7 = B10 ^ (~B16 & B22); \
    A8 = B16 ^ (~B22 & B3); \
    A9 = B22 ^ (~B3 & B9); \
    \
    A10 = B1 ^ (~B7 & B13); \
    A11 = B7 ^ (~B13 & B19); \
    A12 = B13 ^ (~B19 & B20); \
    A13 = B19 ^ (~B20 & B1); \
    A14 = B20 ^ (~B1 & B7); \
    \
    A15 = B4 ^ (~B5 & B11); \
    A16 = B5 ^ (~B11 & B17); \
    A17 = B11 ^ (~B17 & B23); \
    A18 = B17 ^ (~B23 & B4); \
    A19 = B23 ^ (~B4 & B5); \
    \
    A20 = B2 ^ (~B8 & B14); \
    A21 = B8 ^ (~B14 & B15); \
    A22 = B14 ^ (~B15 & B21); \
    A23 = B15 ^ (~B21 & B2); \
    A24 = B21 ^ (~B2 & B8); \
} while(0)

kernel void keccak_f1600_batch(
    device const ulong *in_data    [[buffer(0)]],
    device       ulong *out_data   [[buffer(1)]],
    constant uint      &batch      [[buffer(2)]],
    constant uint      &msg_bytes  [[buffer(3)]],
    constant uint      &rate_bytes [[buffer(4)]],
    constant uint      &out_bytes  [[buffer(5)]],
    constant uint      &domain     [[buffer(6)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= batch) return;

    ulong A0 = 0, A1 = 0, A2 = 0, A3 = 0, A4 = 0;
    ulong A5 = 0, A6 = 0, A7 = 0, A8 = 0, A9 = 0;
    ulong A10 = 0, A11 = 0, A12 = 0, A13 = 0, A14 = 0;
    ulong A15 = 0, A16 = 0, A17 = 0, A18 = 0, A19 = 0;
    ulong A20 = 0, A21 = 0, A22 = 0, A23 = 0, A24 = 0;

    uint msg_lanes  = msg_bytes >> 3;
    uint rate_lanes = rate_bytes >> 3;
    uint out_lanes  = out_bytes >> 3;
    
    // Vectorized Load Path
    if (msg_lanes == 4) {
        ulong4 in_val = ((device const ulong4 *)in_data)[idx];
        A0 = in_val[0];
        A1 = in_val[1];
        A2 = in_val[2];
        A3 = in_val[3];
    } else {
        uint in_base = idx * msg_lanes;
        switch (msg_lanes) {
            case 24: A23 = in_data[in_base + 23]; [[fallthrough]];
            case 23: A22 = in_data[in_base + 22]; [[fallthrough]];
            case 22: A21 = in_data[in_base + 21]; [[fallthrough]];
            case 21: A20 = in_data[in_base + 20]; [[fallthrough]];
            case 20: A19 = in_data[in_base + 19]; [[fallthrough]];
            case 19: A18 = in_data[in_base + 18]; [[fallthrough]];
            case 18: A17 = in_data[in_base + 17]; [[fallthrough]];
            case 17: A16 = in_data[in_base + 16]; [[fallthrough]];
            case 16: A15 = in_data[in_base + 15]; [[fallthrough]];
            case 15: A14 = in_data[in_base + 14]; [[fallthrough]];
            case 14: A13 = in_data[in_base + 13]; [[fallthrough]];
            case 13: A12 = in_data[in_base + 12]; [[fallthrough]];
            case 12: A11 = in_data[in_base + 11]; [[fallthrough]];
            case 11: A10 = in_data[in_base + 10]; [[fallthrough]];
            case 10: A9  = in_data[in_base + 9];  [[fallthrough]];
            case 9:  A8  = in_data[in_base + 8];  [[fallthrough]];
            case 8:  A7  = in_data[in_base + 7];  [[fallthrough]];
            case 7:  A6  = in_data[in_base + 6];  [[fallthrough]];
            case 6:  A5  = in_data[in_base + 5];  [[fallthrough]];
            case 5:  A4  = in_data[in_base + 4];  [[fallthrough]];
            case 4:  A3  = in_data[in_base + 3];  [[fallthrough]];
            case 3:  A2  = in_data[in_base + 2];  [[fallthrough]];
            case 2:  A1  = in_data[in_base + 1];  [[fallthrough]];
            case 1:  A0  = in_data[in_base + 0];  [[fallthrough]];
            case 0: break;
        }
    }

    // Branchless uniform-padded Domain Application
    switch (msg_lanes) {
        case 0:  A0  ^= (ulong)(domain & 0xFFu); break;
        case 1:  A1  ^= (ulong)(domain & 0xFFu); break;
        case 2:  A2  ^= (ulong)(domain & 0xFFu); break;
        case 3:  A3  ^= (ulong)(domain & 0xFFu); break;
        case 4:  A4  ^= (ulong)(domain & 0xFFu); break;
        case 5:  A5  ^= (ulong)(domain & 0xFFu); break;
        case 6:  A6  ^= (ulong)(domain & 0xFFu); break;
        case 7:  A7  ^= (ulong)(domain & 0xFFu); break;
        case 8:  A8  ^= (ulong)(domain & 0xFFu); break;
        case 9:  A9  ^= (ulong)(domain & 0xFFu); break;
        case 10: A10 ^= (ulong)(domain & 0xFFu); break;
        case 11: A11 ^= (ulong)(domain & 0xFFu); break;
        case 12: A12 ^= (ulong)(domain & 0xFFu); break;
        case 13: A13 ^= (ulong)(domain & 0xFFu); break;
        case 14: A14 ^= (ulong)(domain & 0xFFu); break;
        case 15: A15 ^= (ulong)(domain & 0xFFu); break;
        case 16: A16 ^= (ulong)(domain & 0xFFu); break;
        case 17: A17 ^= (ulong)(domain & 0xFFu); break;
        case 18: A18 ^= (ulong)(domain & 0xFFu); break;
        case 19: A19 ^= (ulong)(domain & 0xFFu); break;
        case 20: A20 ^= (ulong)(domain & 0xFFu); break;
        case 21: A21 ^= (ulong)(domain & 0xFFu); break;
        case 22: A22 ^= (ulong)(domain & 0xFFu); break;
        case 23: A23 ^= (ulong)(domain & 0xFFu); break;
        case 24: A24 ^= (ulong)(domain & 0xFFu); break;
    }

    // Branchless uniform-padded Rate End
    switch (rate_lanes - 1u) {
        case 0:  A0  ^= 0x8000000000000000ul; break;
        case 1:  A1  ^= 0x8000000000000000ul; break;
        case 2:  A2  ^= 0x8000000000000000ul; break;
        case 3:  A3  ^= 0x8000000000000000ul; break;
        case 4:  A4  ^= 0x8000000000000000ul; break;
        case 5:  A5  ^= 0x8000000000000000ul; break;
        case 6:  A6  ^= 0x8000000000000000ul; break;
        case 7:  A7  ^= 0x8000000000000000ul; break;
        case 8:  A8  ^= 0x8000000000000000ul; break;
        case 9:  A9  ^= 0x8000000000000000ul; break;
        case 10: A10 ^= 0x8000000000000000ul; break;
        case 11: A11 ^= 0x8000000000000000ul; break;
        case 12: A12 ^= 0x8000000000000000ul; break;
        case 13: A13 ^= 0x8000000000000000ul; break;
        case 14: A14 ^= 0x8000000000000000ul; break;
        case 15: A15 ^= 0x8000000000000000ul; break;
        case 16: A16 ^= 0x8000000000000000ul; break;
        case 17: A17 ^= 0x8000000000000000ul; break;
        case 18: A18 ^= 0x8000000000000000ul; break;
        case 19: A19 ^= 0x8000000000000000ul; break;
        case 20: A20 ^= 0x8000000000000000ul; break;
        case 21: A21 ^= 0x8000000000000000ul; break;
        case 22: A22 ^= 0x8000000000000000ul; break;
        case 23: A23 ^= 0x8000000000000000ul; break;
        case 24: A24 ^= 0x8000000000000000ul; break;
    }

    uint out_base = idx * out_lanes;
    uint written = 0u;

    for (;;) {
        #pragma unroll(24)
        for (uint r = 0u; r < 24u; ++r) {
            ROUND(KECCAK_RC[r]);
        }

        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;

        // Vectorized Store Paths for prevalent targets
        if (take == 4 && out_lanes == 4) {
            ((device ulong4 *)out_data)[idx] = ulong4(A0, A1, A2, A3);
        } else {
            uint out_offset = out_base + written;
            
            if (take == 21 && written == 0 && out_lanes == 32) {
                device ulong4 *ptr = (device ulong4 *)(out_data + out_offset);
                ptr[0] = ulong4(A0, A1, A2, A3);
                ptr[1] = ulong4(A4, A5, A6, A7);
                ptr[2] = ulong4(A8, A9, A10, A11);
                ptr[3] = ulong4(A12, A13, A14, A15);
                ptr[4] = ulong4(A16, A17, A18, A19);
                out_data[out_offset + 20] = A20;
            } else {
                switch (take) {
                    case 24: out_data[out_offset + 23] = A23; [[fallthrough]];
                    case 23: out_data[out_offset + 22] = A22; [[fallthrough]];
                    case 22: out_data[out_offset + 21] = A21; [[fallthrough]];
                    case 21: out_data[out_offset + 20] = A20; [[fallthrough]];
                    case 20: out_data[out_offset + 19] = A19; [[fallthrough]];
                    case 19: out_data[out_offset + 18] = A18; [[fallthrough]];
                    case 18: out_data[out_offset + 17] = A17; [[fallthrough]];
                    case 17: out_data[out_offset + 16] = A16; [[fallthrough]];
                    case 16: out_data[out_offset + 15] = A15; [[fallthrough]];
                    case 15: out_data[out_offset + 14] = A14; [[fallthrough]];
                    case 14: out_data[out_offset + 13] = A13; [[fallthrough]];
                    case 13: out_data[out_offset + 12] = A12; [[fallthrough]];
                    case 12: out_data[out_offset + 11] = A11; [[fallthrough]];
                    case 11: out_data[out_offset + 10] = A10; [[fallthrough]];
                    case 10: out_data[out_offset + 9]  = A9;  [[fallthrough]];
                    case 9:  out_data[out_offset + 8]  = A8;  [[fallthrough]];
                    case 8:  out_data[out_offset + 7]  = A7;  [[fallthrough]];
                    case 7:  out_data[out_offset + 6]  = A6;  [[fallthrough]];
                    case 6:  out_data[out_offset + 5]  = A5;  [[fallthrough]];
                    case 5:  out_data[out_offset + 4]  = A4;  [[fallthrough]];
                    case 4:  out_data[out_offset + 3]  = A3;  [[fallthrough]];
                    case 3:  out_data[out_offset + 2]  = A2;  [[fallthrough]];
                    case 2:  out_data[out_offset + 1]  = A1;  [[fallthrough]];
                    case 1:  out_data[out_offset + 0]  = A0;  [[fallthrough]];
                    case 0: break;
                }
            }
        }

        written += take;
        if (written >= out_lanes) break;
    }
}
```

Result of previous attempt:
     sha3_256_B16K: correct, 0.37 ms, 164.9 Gbitops/s (u64) (28.6% of 577 Gops/s (u64 bitop, est))
    sha3_256_B256K: correct, 1.44 ms, 678.7 Gbitops/s (u64) (117.5% of 577 Gops/s (u64 bitop, est))
      sha3_256_B4M: correct, 22.17 ms, 703.6 Gbitops/s (u64) (121.9% of 577 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.7423

## History

- iter  0: compile=OK | correct=True | score=0.07536349553634346
- iter  1: compile=OK | correct=True | score=0.12384641044567121
- iter  2: compile=OK | correct=True | score=0.16685157722217125
- iter  3: compile=OK | correct=False | score=N/A
- iter  4: compile=OK | correct=True | score=0.1979731001372354
- iter  5: compile=OK | correct=True | score=0.5393418711680308
- iter  6: compile=OK | correct=True | score=0.7127451068934515
- iter  7: compile=OK | correct=True | score=0.7423082670554976

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
