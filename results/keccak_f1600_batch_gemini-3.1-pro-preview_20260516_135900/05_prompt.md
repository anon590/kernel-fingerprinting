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

inline __attribute__((always_inline)) void keccak_round(thread ulong* A, ulong rc) {
    // Theta
    ulong C0 = A[0] ^ A[5] ^ A[10] ^ A[15] ^ A[20];
    ulong C1 = A[1] ^ A[6] ^ A[11] ^ A[16] ^ A[21];
    ulong C2 = A[2] ^ A[7] ^ A[12] ^ A[17] ^ A[22];
    ulong C3 = A[3] ^ A[8] ^ A[13] ^ A[18] ^ A[23];
    ulong C4 = A[4] ^ A[9] ^ A[14] ^ A[19] ^ A[24];

    ulong D0 = C4 ^ rotl64(C1, 1u);
    ulong D1 = C0 ^ rotl64(C2, 1u);
    ulong D2 = C1 ^ rotl64(C3, 1u);
    ulong D3 = C2 ^ rotl64(C4, 1u);
    ulong D4 = C3 ^ rotl64(C0, 1u);

    A[0] ^= D0; A[5] ^= D0; A[10] ^= D0; A[15] ^= D0; A[20] ^= D0;
    A[1] ^= D1; A[6] ^= D1; A[11] ^= D1; A[16] ^= D1; A[21] ^= D1;
    A[2] ^= D2; A[7] ^= D2; A[12] ^= D2; A[17] ^= D2; A[22] ^= D2;
    A[3] ^= D3; A[8] ^= D3; A[13] ^= D3; A[18] ^= D3; A[23] ^= D3;
    A[4] ^= D4; A[9] ^= D4; A[14] ^= D4; A[19] ^= D4; A[24] ^= D4;

    // Rho and Pi (in-place unrolled cycle tracing)
    ulong temp = A[1];
    ulong next;
    
    next = A[10]; A[10] = rotl64(temp, 1u);  temp = next;
    next = A[7];  A[7]  = rotl64(temp, 3u);  temp = next;
    next = A[11]; A[11] = rotl64(temp, 6u);  temp = next;
    next = A[17]; A[17] = rotl64(temp, 10u); temp = next;
    next = A[18]; A[18] = rotl64(temp, 15u); temp = next;
    next = A[3];  A[3]  = rotl64(temp, 21u); temp = next;
    next = A[5];  A[5]  = rotl64(temp, 28u); temp = next;
    next = A[16]; A[16] = rotl64(temp, 36u); temp = next;
    next = A[8];  A[8]  = rotl64(temp, 45u); temp = next;
    next = A[21]; A[21] = rotl64(temp, 55u); temp = next;
    next = A[24]; A[24] = rotl64(temp, 2u);  temp = next;
    next = A[4];  A[4]  = rotl64(temp, 14u); temp = next;
    next = A[15]; A[15] = rotl64(temp, 27u); temp = next;
    next = A[23]; A[23] = rotl64(temp, 41u); temp = next;
    next = A[19]; A[19] = rotl64(temp, 56u); temp = next;
    next = A[13]; A[13] = rotl64(temp, 8u);  temp = next;
    next = A[12]; A[12] = rotl64(temp, 25u); temp = next;
    next = A[2];  A[2]  = rotl64(temp, 43u); temp = next;
    next = A[20]; A[20] = rotl64(temp, 62u); temp = next;
    next = A[14]; A[14] = rotl64(temp, 18u); temp = next;
    next = A[22]; A[22] = rotl64(temp, 39u); temp = next;
    next = A[9];  A[9]  = rotl64(temp, 61u); temp = next;
    next = A[6];  A[6]  = rotl64(temp, 20u); temp = next;
    next = A[1];  A[1]  = rotl64(temp, 44u);

    // Chi
    #pragma unroll(5)
    for (uint y = 0u; y < 25u; y += 5u) {
        ulong c0 = A[y + 0u];
        ulong c1 = A[y + 1u];
        ulong c2 = A[y + 2u];
        ulong c3 = A[y + 3u];
        ulong c4 = A[y + 4u];

        A[y + 0u] = c0 ^ ((~c1) & c2);
        A[y + 1u] = c1 ^ ((~c2) & c3);
        A[y + 2u] = c2 ^ ((~c3) & c4);
        A[y + 3u] = c3 ^ ((~c4) & c0);
        A[y + 4u] = c4 ^ ((~c0) & c1);
    }

    // Iota
    A[0] ^= rc;
}

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

    ulong A[25];

    // msg_bytes = 32 is guaranteed per benchmark spec. 
    // We vectorize the 32-byte fetch to maximize bandwidth efficiency via coalescing.
    device const ulong4* in_data4 = (device const ulong4*)in_data;
    ulong4 msg = in_data4[idx];
    
    A[0] = msg.x;
    A[1] = msg.y;
    A[2] = msg.z;
    A[3] = msg.w;
    
    #pragma unroll(21)
    for (uint i = 4u; i < 25u; ++i) {
        A[i] = 0ul;
    }

    uint rate_lanes = rate_bytes >> 3;
    uint out_lanes  = out_bytes  >> 3;

    A[4] ^= (ulong)(domain & 0xFFu);
    A[rate_lanes - 1u] ^= 0x8000000000000000ul;

    uint out_base = idx * out_lanes;
    uint written = 0u;

    for (;;) {
        #pragma unroll(24)
        for (uint r = 0u; r < 24u; ++r) {
            keccak_round(A, KECCAK_RC[r]);
        }

        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;

        #pragma unroll(25)
        for (uint j = 0u; j < 25u; ++j) {
            if (j < take) {
                out_data[out_base + written + j] = A[j];
            }
        }

        written += take;
        if (written >= out_lanes) break;
    }
}
```

Result of previous attempt:
     sha3_256_B16K: correct, 0.85 ms, 72.0 Gbitops/s (u64) (12.5% of 577 Gops/s (u64 bitop, est))
    sha3_256_B256K: correct, 6.69 ms, 145.8 Gbitops/s (u64) (25.3% of 577 Gops/s (u64 bitop, est))
      sha3_256_B4M: correct, 109.68 ms, 142.3 Gbitops/s (u64) (24.6% of 577 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.1980

## History

- iter  0: compile=OK | correct=True | score=0.07536349553634346
- iter  1: compile=OK | correct=True | score=0.12384641044567121
- iter  2: compile=OK | correct=True | score=0.16685157722217125
- iter  3: compile=OK | correct=False | score=N/A
- iter  4: compile=OK | correct=True | score=0.1979731001372354

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
