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

inline ulong ROTL64(ulong x, uint k) {
    return (x << k) | (x >> (64 - k));
}

#define RND(RC) { \
    ulong C0 = a00 ^ a01 ^ a02 ^ a03 ^ a04; \
    ulong C1 = a10 ^ a11 ^ a12 ^ a13 ^ a14; \
    ulong C2 = a20 ^ a21 ^ a22 ^ a23 ^ a24; \
    ulong C3 = a30 ^ a31 ^ a32 ^ a33 ^ a34; \
    ulong C4 = a40 ^ a41 ^ a42 ^ a43 ^ a44; \
    ulong D0 = C4 ^ ROTL64(C1, 1); \
    ulong D1 = C0 ^ ROTL64(C2, 1); \
    ulong D2 = C1 ^ ROTL64(C3, 1); \
    ulong D3 = C2 ^ ROTL64(C4, 1); \
    ulong D4 = C3 ^ ROTL64(C0, 1); \
    ulong b00 =        (a00 ^ D0)     ; \
    ulong b02 = ROTL64(a10 ^ D1,  1); \
    ulong b04 = ROTL64(a20 ^ D2, 62); \
    ulong b01 = ROTL64(a30 ^ D3, 28); \
    ulong b03 = ROTL64(a40 ^ D4, 27); \
    ulong b13 = ROTL64(a01 ^ D0, 36); \
    ulong b10 = ROTL64(a11 ^ D1, 44); \
    ulong b12 = ROTL64(a21 ^ D2,  6); \
    ulong b14 = ROTL64(a31 ^ D3, 55); \
    ulong b11 = ROTL64(a41 ^ D4, 20); \
    ulong b21 = ROTL64(a02 ^ D0,  3); \
    ulong b23 = ROTL64(a12 ^ D1, 10); \
    ulong b20 = ROTL64(a22 ^ D2, 43); \
    ulong b22 = ROTL64(a32 ^ D3, 25); \
    ulong b24 = ROTL64(a42 ^ D4, 39); \
    ulong b34 = ROTL64(a03 ^ D0, 41); \
    ulong b31 = ROTL64(a13 ^ D1, 45); \
    ulong b33 = ROTL64(a23 ^ D2, 15); \
    ulong b30 = ROTL64(a33 ^ D3, 21); \
    ulong b32 = ROTL64(a43 ^ D4,  8); \
    ulong b42 = ROTL64(a04 ^ D0, 18); \
    ulong b44 = ROTL64(a14 ^ D1,  2); \
    ulong b41 = ROTL64(a24 ^ D2, 61); \
    ulong b43 = ROTL64(a34 ^ D3, 56); \
    ulong b40 = ROTL64(a44 ^ D4, 14); \
    a00 = b00 ^ ((~b10) & b20) ^ (ulong)(RC); \
    a10 = b10 ^ ((~b20) & b30); \
    a20 = b20 ^ ((~b30) & b40); \
    a30 = b30 ^ ((~b40) & b00); \
    a40 = b40 ^ ((~b00) & b10); \
    a01 = b01 ^ ((~b11) & b21); \
    a11 = b11 ^ ((~b21) & b31); \
    a21 = b21 ^ ((~b31) & b41); \
    a31 = b31 ^ ((~b41) & b01); \
    a41 = b41 ^ ((~b01) & b11); \
    a02 = b02 ^ ((~b12) & b22); \
    a12 = b12 ^ ((~b22) & b32); \
    a22 = b22 ^ ((~b32) & b42); \
    a32 = b32 ^ ((~b42) & b02); \
    a42 = b42 ^ ((~b02) & b12); \
    a03 = b03 ^ ((~b13) & b23); \
    a13 = b13 ^ ((~b23) & b33); \
    a23 = b23 ^ ((~b33) & b43); \
    a33 = b33 ^ ((~b43) & b03); \
    a43 = b43 ^ ((~b03) & b13); \
    a04 = b04 ^ ((~b14) & b24); \
    a14 = b14 ^ ((~b24) & b34); \
    a24 = b24 ^ ((~b34) & b44); \
    a34 = b34 ^ ((~b44) & b04); \
    a44 = b44 ^ ((~b04) & b14); \
}

#define PERMUTE() { \
    RND(0x0000000000000001ul); \
    RND(0x0000000000008082ul); \
    RND(0x800000000000808Aul); \
    RND(0x8000000080008000ul); \
    RND(0x000000000000808Bul); \
    RND(0x0000000080000001ul); \
    RND(0x8000000080008081ul); \
    RND(0x8000000000008009ul); \
    RND(0x000000000000008Aul); \
    RND(0x0000000000000088ul); \
    RND(0x0000000080008009ul); \
    RND(0x000000008000000Aul); \
    RND(0x000000008000808Bul); \
    RND(0x800000000000008Bul); \
    RND(0x8000000000008089ul); \
    RND(0x8000000000008003ul); \
    RND(0x8000000000008002ul); \
    RND(0x8000000000000080ul); \
    RND(0x000000000000800Aul); \
    RND(0x800000008000000Aul); \
    RND(0x8000000080008081ul); \
    RND(0x8000000000008080ul); \
    RND(0x0000000080000001ul); \
    RND(0x8000000080008008ul); \
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

    // 25 state lanes as named scalars to stay in registers.
    ulong a00=0, a10=0, a20=0, a30=0, a40=0;
    ulong a01=0, a11=0, a21=0, a31=0, a41=0;
    ulong a02=0, a12=0, a22=0, a32=0, a42=0;
    ulong a03=0, a13=0, a23=0, a33=0, a43=0;
    ulong a04=0, a14=0, a24=0, a34=0, a44=0;

    // Absorb single block. State lane index k = x + 5*y maps to a{x}{y}.
    uint in_base = idx * msg_lanes;
    // msg_lanes is known to be 4 in all in-distribution tests, but we still
    // honour the runtime value for held-out (SHAKE128, msg_bytes=32 -> 4).
    // Load message lanes directly into corresponding state scalars.
    // Mapping: k=0 -> a00, k=1 -> a10, k=2 -> a20, k=3 -> a30, k=4 -> a40,
    //          k=5 -> a01, k=6 -> a11, k=7 -> a21, k=8 -> a31, k=9 -> a41, ...
    // Since msg_lanes <= rate_lanes - 1, we just branch on each lane.
    {
        ulong tmp[25];
        for (uint i = 0; i < 25; ++i) tmp[i] = 0ul;
        for (uint i = 0; i < msg_lanes; ++i) tmp[i] = in_data[in_base + i];
        tmp[msg_lanes] ^= (ulong)(domain & 0xFFu);
        tmp[rate_lanes - 1u] ^= 0x8000000000000000ul;

        a00 = tmp[ 0]; a10 = tmp[ 1]; a20 = tmp[ 2]; a30 = tmp[ 3]; a40 = tmp[ 4];
        a01 = tmp[ 5]; a11 = tmp[ 6]; a21 = tmp[ 7]; a31 = tmp[ 8]; a41 = tmp[ 9];
        a02 = tmp[10]; a12 = tmp[11]; a22 = tmp[12]; a32 = tmp[13]; a42 = tmp[14];
        a03 = tmp[15]; a13 = tmp[16]; a23 = tmp[17]; a33 = tmp[18]; a43 = tmp[19];
        a04 = tmp[20]; a14 = tmp[21]; a24 = tmp[22]; a34 = tmp[23]; a44 = tmp[24];
    }

    uint out_base = idx * out_lanes;

    // First permutation.
    PERMUTE();

    // Fast single-squeeze path (SHA3-256: out_lanes=4, rate_lanes=17).
    if (out_lanes <= rate_lanes) {
        device ulong *op = out_data + out_base;
        // Write up to out_lanes lanes from state. Unroll via switch on out_lanes.
        // out_lanes can be at most rate_lanes (<=21). Write directly without
        // staging through a stack array.
        // Use straight conditional writes; compiler folds when out_lanes is
        // a runtime value but small loop bound is fine.
        // Map index j to state lane (x = j%5, y = j/5).
        // For SHA3-256 out_lanes = 4: lanes 0..3 -> a00,a10,a20,a30.
        if (out_lanes >= 1u) op[0] = a00;
        if (out_lanes >= 2u) op[1] = a10;
        if (out_lanes >= 3u) op[2] = a20;
        if (out_lanes >= 4u) op[3] = a30;
        if (out_lanes >= 5u) op[4] = a40;
        if (out_lanes >= 6u) op[5] = a01;
        if (out_lanes >= 7u) op[6] = a11;
        if (out_lanes >= 8u) op[7] = a21;
        if (out_lanes >= 9u) op[8] = a31;
        if (out_lanes >= 10u) op[9] = a41;
        if (out_lanes >= 11u) op[10] = a02;
        if (out_lanes >= 12u) op[11] = a12;
        if (out_lanes >= 13u) op[12] = a22;
        if (out_lanes >= 14u) op[13] = a32;
        if (out_lanes >= 15u) op[14] = a42;
        if (out_lanes >= 16u) op[15] = a03;
        if (out_lanes >= 17u) op[16] = a13;
        if (out_lanes >= 18u) op[17] = a23;
        if (out_lanes >= 19u) op[18] = a33;
        if (out_lanes >= 20u) op[19] = a43;
        if (out_lanes >= 21u) op[20] = a04;
        return;
    }

    // General path for SHAKE-style multi-squeeze.
    uint written = 0u;
    for (;;) {
        ulong s[25];
        s[ 0]=a00; s[ 1]=a10; s[ 2]=a20; s[ 3]=a30; s[ 4]=a40;
        s[ 5]=a01; s[ 6]=a11; s[ 7]=a21; s[ 8]=a31; s[ 9]=a41;
        s[10]=a02; s[11]=a12; s[12]=a22; s[13]=a32; s[14]=a42;
        s[15]=a03; s[16]=a13; s[17]=a23; s[18]=a33; s[19]=a43;
        s[20]=a04; s[21]=a14; s[22]=a24; s[23]=a34; s[24]=a44;

        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        for (uint j = 0; j < take; ++j) {
            out_data[out_base + written + j] = s[j];
        }
        written += take;
        if (written >= out_lanes) break;

        PERMUTE();
    }
}
```

Result of previous attempt:
     sha3_256_B16K: correct, 0.28 ms, 219.9 Gbitops/s (u64) (38.0% of 578 Gops/s (u64 bitop, est))
    sha3_256_B256K: correct, 1.46 ms, 669.4 Gbitops/s (u64) (115.7% of 578 Gops/s (u64 bitop, est))
      sha3_256_B4M: correct, 22.67 ms, 688.2 Gbitops/s (u64) (119.0% of 578 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.8061

## History

- iter  0: compile=OK | correct=True | score=0.07578622105259203
- iter  1: compile=OK | correct=True | score=0.7177106681658834
- iter  2: compile=OK | correct=True | score=0.6368853478781129
- iter  3: compile=OK | correct=True | score=0.80610701577865

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
