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

The baseline sizes below use the SHA3-256 mode (rate=136, domain=0x06, out=32). The kernel is scored on several (rate_bytes, out_bytes, domain) parameter sets, including configurations not listed among the baseline sizes, and out_bytes may exceed rate_bytes. The kernel must use the runtime values of ``rate_bytes``, ``out_bytes`` and ``domain`` rather than compile-time constants. Correctness is bit-exact against ``hashlib.sha3_256`` / ``hashlib.shake_128``; any mismatched output ulong rejects the candidate.

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

inline ulong rotl(ulong x, uint k) {
    return (x << k) | (x >> (64u - k));
}

inline void keccak_f1600(thread ulong *A) {
    #pragma unroll 24
    for (uint r = 0u; r < 24u; ++r) {
        ulong C0 = A[0] ^ A[5] ^ A[10] ^ A[15] ^ A[20];
        ulong C1 = A[1] ^ A[6] ^ A[11] ^ A[16] ^ A[21];
        ulong C2 = A[2] ^ A[7] ^ A[12] ^ A[17] ^ A[22];
        ulong C3 = A[3] ^ A[8] ^ A[13] ^ A[18] ^ A[23];
        ulong C4 = A[4] ^ A[9] ^ A[14] ^ A[19] ^ A[24];

        ulong D0 = C4 ^ rotl(C1, 1u);
        ulong D1 = C0 ^ rotl(C2, 1u);
        ulong D2 = C1 ^ rotl(C3, 1u);
        ulong D3 = C2 ^ rotl(C4, 1u);
        ulong D4 = C3 ^ rotl(C0, 1u);

        ulong B0  = A[0]  ^ D0;
        ulong B1  = rotl(A[6]  ^ D1, 44u);
        ulong B2  = rotl(A[12] ^ D2, 43u);
        ulong B3  = rotl(A[18] ^ D3, 21u);
        ulong B4  = rotl(A[24] ^ D4, 14u);

        ulong B5  = rotl(A[3]  ^ D3, 28u);
        ulong B6  = rotl(A[9]  ^ D4, 20u);
        ulong B7  = rotl(A[10] ^ D0,  3u);
        ulong B8  = rotl(A[16] ^ D1, 45u);
        ulong B9  = rotl(A[22] ^ D2, 61u);

        ulong B10 = rotl(A[1]  ^ D1,  1u);
        ulong B11 = rotl(A[7]  ^ D2,  6u);
        ulong B12 = rotl(A[13] ^ D3, 25u);
        ulong B13 = rotl(A[19] ^ D4,  8u);
        ulong B14 = rotl(A[20] ^ D0, 18u);

        ulong B15 = rotl(A[4]  ^ D4, 27u);
        ulong B16 = rotl(A[5]  ^ D0, 36u);
        ulong B17 = rotl(A[11] ^ D1, 10u);
        ulong B18 = rotl(A[17] ^ D2, 15u);
        ulong B19 = rotl(A[23] ^ D3, 56u);

        ulong B20 = rotl(A[2]  ^ D2, 62u);
        ulong B21 = rotl(A[8]  ^ D3, 55u);
        ulong B22 = rotl(A[14] ^ D4, 39u);
        ulong B23 = rotl(A[15] ^ D0, 41u);
        ulong B24 = rotl(A[21] ^ D1,  2u);

        A[0]  = B0  ^ ((~B1)  & B2)  ^ KECCAK_RC[r];
        A[1]  = B1  ^ ((~B2)  & B3);
        A[2]  = B2  ^ ((~B3)  & B4);
        A[3]  = B3  ^ ((~B4)  & B0);
        A[4]  = B4  ^ ((~B0)  & B1);

        A[5]  = B5  ^ ((~B6)  & B7);
        A[6]  = B6  ^ ((~B7)  & B8);
        A[7]  = B7  ^ ((~B8)  & B9);
        A[8]  = B8  ^ ((~B9)  & B5);
        A[9]  = B9  ^ ((~B5)  & B6);

        A[10] = B10 ^ ((~B11) & B12);
        A[11] = B11 ^ ((~B12) & B13);
        A[12] = B12 ^ ((~B13) & B14);
        A[13] = B13 ^ ((~B14) & B10);
        A[14] = B14 ^ ((~B10) & B11);

        A[15] = B15 ^ ((~B16) & B17);
        A[16] = B16 ^ ((~B17) & B18);
        A[17] = B17 ^ ((~B18) & B19);
        A[18] = B18 ^ ((~B19) & B15);
        A[19] = B19 ^ ((~B15) & B16);

        A[20] = B20 ^ ((~B21) & B22);
        A[21] = B21 ^ ((~B22) & B23);
        A[22] = B22 ^ ((~B23) & B24);
        A[23] = B23 ^ ((~B24) & B20);
        A[24] = B24 ^ ((~B20) & B21);
    }
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

    uint msg_lanes  = msg_bytes  >> 3;
    uint rate_lanes = rate_bytes >> 3;
    uint out_lanes  = out_bytes  >> 3;

    ulong A[25];
    A[0] = 0;  A[1] = 0;  A[2] = 0;  A[3] = 0;  A[4] = 0;
    A[5] = 0;  A[6] = 0;  A[7] = 0;  A[8] = 0;  A[9] = 0;
    A[10] = 0; A[11] = 0; A[12] = 0; A[13] = 0; A[14] = 0;
    A[15] = 0; A[16] = 0; A[17] = 0; A[18] = 0; A[19] = 0;
    A[20] = 0; A[21] = 0; A[22] = 0; A[23] = 0; A[24] = 0;

    uint in_base = idx * msg_lanes;
    
    // Statically indexed cascade to force A to map to physical registers
    if (msg_lanes > 0)  A[0]  ^= in_data[in_base + 0];
    if (msg_lanes > 1)  A[1]  ^= in_data[in_base + 1];
    if (msg_lanes > 2)  A[2]  ^= in_data[in_base + 2];
    if (msg_lanes > 3)  A[3]  ^= in_data[in_base + 3];
    if (msg_lanes > 4)  A[4]  ^= in_data[in_base + 4];
    if (msg_lanes > 5)  A[5]  ^= in_data[in_base + 5];
    if (msg_lanes > 6)  A[6]  ^= in_data[in_base + 6];
    if (msg_lanes > 7)  A[7]  ^= in_data[in_base + 7];
    if (msg_lanes > 8)  A[8]  ^= in_data[in_base + 8];
    if (msg_lanes > 9)  A[9]  ^= in_data[in_base + 9];
    if (msg_lanes > 10) A[10] ^= in_data[in_base + 10];
    if (msg_lanes > 11) A[11] ^= in_data[in_base + 11];
    if (msg_lanes > 12) A[12] ^= in_data[in_base + 12];
    if (msg_lanes > 13) A[13] ^= in_data[in_base + 13];
    if (msg_lanes > 14) A[14] ^= in_data[in_base + 14];
    if (msg_lanes > 15) A[15] ^= in_data[in_base + 15];
    if (msg_lanes > 16) A[16] ^= in_data[in_base + 16];
    if (msg_lanes > 17) A[17] ^= in_data[in_base + 17];
    if (msg_lanes > 18) A[18] ^= in_data[in_base + 18];
    if (msg_lanes > 19) A[19] ^= in_data[in_base + 19];
    if (msg_lanes > 20) A[20] ^= in_data[in_base + 20];
    if (msg_lanes > 21) A[21] ^= in_data[in_base + 21];
    if (msg_lanes > 22) A[22] ^= in_data[in_base + 22];
    if (msg_lanes > 23) A[23] ^= in_data[in_base + 23];
    if (msg_lanes > 24) A[24] ^= in_data[in_base + 24];

    ulong domain_val = (ulong)(domain & 0xFFu);
    switch (msg_lanes) {
        case 0:  A[0]  ^= domain_val; break;
        case 1:  A[1]  ^= domain_val; break;
        case 2:  A[2]  ^= domain_val; break;
        case 3:  A[3]  ^= domain_val; break;
        case 4:  A[4]  ^= domain_val; break;
        case 5:  A[5]  ^= domain_val; break;
        case 6:  A[6]  ^= domain_val; break;
        case 7:  A[7]  ^= domain_val; break;
        case 8:  A[8]  ^= domain_val; break;
        case 9:  A[9]  ^= domain_val; break;
        case 10: A[10] ^= domain_val; break;
        case 11: A[11] ^= domain_val; break;
        case 12: A[12] ^= domain_val; break;
        case 13: A[13] ^= domain_val; break;
        case 14: A[14] ^= domain_val; break;
        case 15: A[15] ^= domain_val; break;
        case 16: A[16] ^= domain_val; break;
        case 17: A[17] ^= domain_val; break;
        case 18: A[18] ^= domain_val; break;
        case 19: A[19] ^= domain_val; break;
        case 20: A[20] ^= domain_val; break;
        case 21: A[21] ^= domain_val; break;
        case 22: A[22] ^= domain_val; break;
        case 23: A[23] ^= domain_val; break;
        case 24: A[24] ^= domain_val; break;
    }

    switch (rate_lanes - 1u) {
        case 0:  A[0]  ^= 0x8000000000000000ul; break;
        case 1:  A[1]  ^= 0x8000000000000000ul; break;
        case 2:  A[2]  ^= 0x8000000000000000ul; break;
        case 3:  A[3]  ^= 0x8000000000000000ul; break;
        case 4:  A[4]  ^= 0x8000000000000000ul; break;
        case 5:  A[5]  ^= 0x8000000000000000ul; break;
        case 6:  A[6]  ^= 0x8000000000000000ul; break;
        case 7:  A[7]  ^= 0x8000000000000000ul; break;
        case 8:  A[8]  ^= 0x8000000000000000ul; break;
        case 9:  A[9]  ^= 0x8000000000000000ul; break;
        case 10: A[10] ^= 0x8000000000000000ul; break;
        case 11: A[11] ^= 0x8000000000000000ul; break;
        case 12: A[12] ^= 0x8000000000000000ul; break;
        case 13: A[13] ^= 0x8000000000000000ul; break;
        case 14: A[14] ^= 0x8000000000000000ul; break;
        case 15: A[15] ^= 0x8000000000000000ul; break;
        case 16: A[16] ^= 0x8000000000000000ul; break;
        case 17: A[17] ^= 0x8000000000000000ul; break;
        case 18: A[18] ^= 0x8000000000000000ul; break;
        case 19: A[19] ^= 0x8000000000000000ul; break;
        case 20: A[20] ^= 0x8000000000000000ul; break;
        case 21: A[21] ^= 0x8000000000000000ul; break;
        case 22: A[22] ^= 0x8000000000000000ul; break;
        case 23: A[23] ^= 0x8000000000000000ul; break;
        case 24: A[24] ^= 0x8000000000000000ul; break;
    }

    uint out_base = idx * out_lanes;
    uint written = 0u;
    for (;;) {
        keccak_f1600(A);
        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        
        if (take > 0)  out_data[out_base + written + 0]  = A[0];
        if (take > 1)  out_data[out_base + written + 1]  = A[1];
        if (take > 2)  out_data[out_base + written + 2]  = A[2];
        if (take > 3)  out_data[out_base + written + 3]  = A[3];
        if (take > 4)  out_data[out_base + written + 4]  = A[4];
        if (take > 5)  out_data[out_base + written + 5]  = A[5];
        if (take > 6)  out_data[out_base + written + 6]  = A[6];
        if (take > 7)  out_data[out_base + written + 7]  = A[7];
        if (take > 8)  out_data[out_base + written + 8]  = A[8];
        if (take > 9)  out_data[out_base + written + 9]  = A[9];
        if (take > 10) out_data[out_base + written + 10] = A[10];
        if (take > 11) out_data[out_base + written + 11] = A[11];
        if (take > 12) out_data[out_base + written + 12] = A[12];
        if (take > 13) out_data[out_base + written + 13] = A[13];
        if (take > 14) out_data[out_base + written + 14] = A[14];
        if (take > 15) out_data[out_base + written + 15] = A[15];
        if (take > 16) out_data[out_base + written + 16] = A[16];
        if (take > 17) out_data[out_base + written + 17] = A[17];
        if (take > 18) out_data[out_base + written + 18] = A[18];
        if (take > 19) out_data[out_base + written + 19] = A[19];
        if (take > 20) out_data[out_base + written + 20] = A[20];
        if (take > 21) out_data[out_base + written + 21] = A[21];
        if (take > 22) out_data[out_base + written + 22] = A[22];
        if (take > 23) out_data[out_base + written + 23] = A[23];
        if (take > 24) out_data[out_base + written + 24] = A[24];

        written += take;
        if (written >= out_lanes) break;
    }
}
```

Result of previous attempt:
     sha3_256_B16K: correct, 0.41 ms, 148.6 Gbitops/s (u64) (13.2% of 1125 Gops/s (u64 bitop, est))
    sha3_256_B256K: correct, 1.93 ms, 504.2 Gbitops/s (u64) (44.8% of 1125 Gops/s (u64 bitop, est))
      sha3_256_B4M: correct, 22.84 ms, 683.0 Gbitops/s (u64) (60.7% of 1125 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.3300

## History

- iter  0: compile=OK | correct=True | score=0.03916592497170866
- iter  1: compile=OK | correct=True | score=0.3300328432407895

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
