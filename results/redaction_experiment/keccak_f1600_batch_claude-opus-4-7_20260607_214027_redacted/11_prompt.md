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

#define ROTL64(x, k) (((x) << (k)) | ((x) >> (64 - (k))))

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

    ulong a00=0, a01=0, a02=0, a03=0, a04=0;
    ulong a05=0, a06=0, a07=0, a08=0, a09=0;
    ulong a10=0, a11=0, a12=0, a13=0, a14=0;
    ulong a15=0, a16=0, a17=0, a18=0, a19=0;
    ulong a20=0, a21=0, a22=0, a23=0, a24=0;

    // Absorb via temp array (msg_lanes is small, indices data-dependent).
    {
        ulong S[25];
        for (uint i = 0u; i < 25u; ++i) S[i] = 0ul;
        uint in_base = idx * msg_lanes;
        for (uint i = 0u; i < msg_lanes; ++i) {
            S[i] = in_data[in_base + i];
        }
        S[msg_lanes]       ^= (ulong)(domain & 0xFFu);
        S[rate_lanes - 1u] ^= 0x8000000000000000ul;
        a00=S[ 0]; a01=S[ 1]; a02=S[ 2]; a03=S[ 3]; a04=S[ 4];
        a05=S[ 5]; a06=S[ 6]; a07=S[ 7]; a08=S[ 8]; a09=S[ 9];
        a10=S[10]; a11=S[11]; a12=S[12]; a13=S[13]; a14=S[14];
        a15=S[15]; a16=S[16]; a17=S[17]; a18=S[18]; a19=S[19];
        a20=S[20]; a21=S[21]; a22=S[22]; a23=S[23]; a24=S[24];
    }

    uint out_base = idx * out_lanes;
    uint written = 0u;

    for (;;) {
        for (uint r = 0u; r < 24u; r += 2u) {
            // ----------- Round r -----------
            ulong C0 = a00 ^ a05 ^ a10 ^ a15 ^ a20;
            ulong C1 = a01 ^ a06 ^ a11 ^ a16 ^ a21;
            ulong C2 = a02 ^ a07 ^ a12 ^ a17 ^ a22;
            ulong C3 = a03 ^ a08 ^ a13 ^ a18 ^ a23;
            ulong C4 = a04 ^ a09 ^ a14 ^ a19 ^ a24;

            ulong D0 = C4 ^ ROTL64(C1, 1);
            ulong D1 = C0 ^ ROTL64(C2, 1);
            ulong D2 = C1 ^ ROTL64(C3, 1);
            ulong D3 = C2 ^ ROTL64(C4, 1);
            ulong D4 = C3 ^ ROTL64(C0, 1);

            ulong b00 =        (a00 ^ D0);
            ulong b10 = ROTL64(a01 ^ D1,  1);
            ulong b20 = ROTL64(a02 ^ D2, 62);
            ulong b05 = ROTL64(a03 ^ D3, 28);
            ulong b15 = ROTL64(a04 ^ D4, 27);
            ulong b16 = ROTL64(a05 ^ D0, 36);
            ulong b01 = ROTL64(a06 ^ D1, 44);
            ulong b11 = ROTL64(a07 ^ D2,  6);
            ulong b21 = ROTL64(a08 ^ D3, 55);
            ulong b06 = ROTL64(a09 ^ D4, 20);
            ulong b07 = ROTL64(a10 ^ D0,  3);
            ulong b17 = ROTL64(a11 ^ D1, 10);
            ulong b02 = ROTL64(a12 ^ D2, 43);
            ulong b12 = ROTL64(a13 ^ D3, 25);
            ulong b22 = ROTL64(a14 ^ D4, 39);
            ulong b23 = ROTL64(a15 ^ D0, 41);
            ulong b08 = ROTL64(a16 ^ D1, 45);
            ulong b18 = ROTL64(a17 ^ D2, 15);
            ulong b03 = ROTL64(a18 ^ D3, 21);
            ulong b13 = ROTL64(a19 ^ D4,  8);
            ulong b14 = ROTL64(a20 ^ D0, 18);
            ulong b24 = ROTL64(a21 ^ D1,  2);
            ulong b09 = ROTL64(a22 ^ D2, 61);
            ulong b19 = ROTL64(a23 ^ D3, 56);
            ulong b04 = ROTL64(a24 ^ D4, 14);

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

            // ----------- Round r+1 -----------
            C0 = a00 ^ a05 ^ a10 ^ a15 ^ a20;
            C1 = a01 ^ a06 ^ a11 ^ a16 ^ a21;
            C2 = a02 ^ a07 ^ a12 ^ a17 ^ a22;
            C3 = a03 ^ a08 ^ a13 ^ a18 ^ a23;
            C4 = a04 ^ a09 ^ a14 ^ a19 ^ a24;

            D0 = C4 ^ ROTL64(C1, 1);
            D1 = C0 ^ ROTL64(C2, 1);
            D2 = C1 ^ ROTL64(C3, 1);
            D3 = C2 ^ ROTL64(C4, 1);
            D4 = C3 ^ ROTL64(C0, 1);

            b00 =        (a00 ^ D0);
            b10 = ROTL64(a01 ^ D1,  1);
            b20 = ROTL64(a02 ^ D2, 62);
            b05 = ROTL64(a03 ^ D3, 28);
            b15 = ROTL64(a04 ^ D4, 27);
            b16 = ROTL64(a05 ^ D0, 36);
            b01 = ROTL64(a06 ^ D1, 44);
            b11 = ROTL64(a07 ^ D2,  6);
            b21 = ROTL64(a08 ^ D3, 55);
            b06 = ROTL64(a09 ^ D4, 20);
            b07 = ROTL64(a10 ^ D0,  3);
            b17 = ROTL64(a11 ^ D1, 10);
            b02 = ROTL64(a12 ^ D2, 43);
            b12 = ROTL64(a13 ^ D3, 25);
            b22 = ROTL64(a14 ^ D4, 39);
            b23 = ROTL64(a15 ^ D0, 41);
            b08 = ROTL64(a16 ^ D1, 45);
            b18 = ROTL64(a17 ^ D2, 15);
            b03 = ROTL64(a18 ^ D3, 21);
            b13 = ROTL64(a19 ^ D4,  8);
            b14 = ROTL64(a20 ^ D0, 18);
            b24 = ROTL64(a21 ^ D1,  2);
            b09 = ROTL64(a22 ^ D2, 61);
            b19 = ROTL64(a23 ^ D3, 56);
            b04 = ROTL64(a24 ^ D4, 14);

            a00 = b00 ^ ((~b01) & b02) ^ KECCAK_RC[r + 1u];
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

        // Squeeze: write up to rate_lanes lanes, capped at remaining out.
        ulong S[25];
        S[ 0]=a00; S[ 1]=a01; S[ 2]=a02; S[ 3]=a03; S[ 4]=a04;
        S[ 5]=a05; S[ 6]=a06; S[ 7]=a07; S[ 8]=a08; S[ 9]=a09;
        S[10]=a10; S[11]=a11; S[12]=a12; S[13]=a13; S[14]=a14;
        S[15]=a15; S[16]=a16; S[17]=a17; S[18]=a18; S[19]=a19;
        S[20]=a20; S[21]=a21; S[22]=a22; S[23]=a23; S[24]=a24;

        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        device ulong *op = out_data + out_base + written;
        for (uint j = 0u; j < take; ++j) {
            op[j] = S[j];
        }
        written += take;
        if (written >= out_lanes) break;
    }
}
```

Result of previous attempt:
     sha3_256_B16K: correct, 0.37 ms, 166.3 Gbitops/s (u64) (14.8% of 1125 Gops/s (u64 bitop, est))
    sha3_256_B256K: correct, 2.29 ms, 426.8 Gbitops/s (u64) (37.9% of 1125 Gops/s (u64 bitop, est))
      sha3_256_B4M: correct, 21.91 ms, 712.2 Gbitops/s (u64) (63.3% of 1125 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.3287

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

#define ROTL64(x, k) (((x) << (k)) | ((x) >> (64 - (k))))

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

    // Keep state in a flat 25-element array; the compiler will promote
    // to registers since all indices used in the permutation are
    // compile-time constants.
    ulong S[25];
    for (uint i = 0u; i < 25u; ++i) S[i] = 0ul;

    // Absorb: data-dependent loop with data-dependent index. Use a
    // device pointer offset; the loop trip count msg_lanes is small.
    uint in_base = idx * msg_lanes;
    for (uint i = 0u; i < msg_lanes; ++i) {
        S[i] = in_data[in_base + i];
    }
    // Padding: domain at byte 0 of lane msg_lanes; 0x80 at byte 7 of lane rate_lanes-1.
    S[msg_lanes]      ^= (ulong)(domain & 0xFFu);
    S[rate_lanes - 1u] ^= 0x8000000000000000ul;

    uint out_base = idx * out_lanes;
    uint written = 0u;

    for (;;) {
        // Pull the entire state into named locals so all subsequent
        // operations are register-resident with constant indices.
        ulong a00=S[ 0], a01=S[ 1], a02=S[ 2], a03=S[ 3], a04=S[ 4];
        ulong a05=S[ 5], a06=S[ 6], a07=S[ 7], a08=S[ 8], a09=S[ 9];
        ulong a10=S[10], a11=S[11], a12=S[12], a13=S[13], a14=S[14];
        ulong a15=S[15], a16=S[16], a17=S[17], a18=S[18], a19=S[19];
        ulong a20=S[20], a21=S[21], a22=S[22], a23=S[23], a24=S[24];

        // Unroll 2 rounds at a time, using b-buffer for round k and
        // ping-pong back to a for round k+1. This halves the round
        // overhead and gives the compiler more scheduling freedom.
        for (uint r = 0u; r < 24u; r += 2u) {
            // ----------- Round r -----------
            ulong C0 = a00 ^ a05 ^ a10 ^ a15 ^ a20;
            ulong C1 = a01 ^ a06 ^ a11 ^ a16 ^ a21;
            ulong C2 = a02 ^ a07 ^ a12 ^ a17 ^ a22;
            ulong C3 = a03 ^ a08 ^ a13 ^ a18 ^ a23;
            ulong C4 = a04 ^ a09 ^ a14 ^ a19 ^ a24;

            ulong D0 = C4 ^ ROTL64(C1, 1);
            ulong D1 = C0 ^ ROTL64(C2, 1);
            ulong D2 = C1 ^ ROTL64(C3, 1);
            ulong D3 = C2 ^ ROTL64(C4, 1);
            ulong D4 = C3 ^ ROTL64(C0, 1);

            ulong b00 =        (a00 ^ D0);
            ulong b10 = ROTL64(a01 ^ D1,  1);
            ulong b20 = ROTL64(a02 ^ D2, 62);
            ulong b05 = ROTL64(a03 ^ D3, 28);
            ulong b15 = ROTL64(a04 ^ D4, 27);
            ulong b16 = ROTL64(a05 ^ D0, 36);
            ulong b01 = ROTL64(a06 ^ D1, 44);
            ulong b11 = ROTL64(a07 ^ D2,  6);
            ulong b21 = ROTL64(a08 ^ D3, 55);
            ulong b06 = ROTL64(a09 ^ D4, 20);
            ulong b07 = ROTL64(a10 ^ D0,  3);
            ulong b17 = ROTL64(a11 ^ D1, 10);
            ulong b02 = ROTL64(a12 ^ D2, 43);
            ulong b12 = ROTL64(a13 ^ D3, 25);
            ulong b22 = ROTL64(a14 ^ D4, 39);
            ulong b23 = ROTL64(a15 ^ D0, 41);
            ulong b08 = ROTL64(a16 ^ D1, 45);
            ulong b18 = ROTL64(a17 ^ D2, 15);
            ulong b03 = ROTL64(a18 ^ D3, 21);
            ulong b13 = ROTL64(a19 ^ D4,  8);
            ulong b14 = ROTL64(a20 ^ D0, 18);
            ulong b24 = ROTL64(a21 ^ D1,  2);
            ulong b09 = ROTL64(a22 ^ D2, 61);
            ulong b19 = ROTL64(a23 ^ D3, 56);
            ulong b04 = ROTL64(a24 ^ D4, 14);

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

            // ----------- Round r+1 -----------
            C0 = a00 ^ a05 ^ a10 ^ a15 ^ a20;
            C1 = a01 ^ a06 ^ a11 ^ a16 ^ a21;
            C2 = a02 ^ a07 ^ a12 ^ a17 ^ a22;
            C3 = a03 ^ a08 ^ a13 ^ a18 ^ a23;
            C4 = a04 ^ a09 ^ a14 ^ a19 ^ a24;

            D0 = C4 ^ ROTL64(C1, 1);
            D1 = C0 ^ ROTL64(C2, 1);
            D2 = C1 ^ ROTL64(C3, 1);
            D3 = C2 ^ ROTL64(C4, 1);
            D4 = C3 ^ ROTL64(C0, 1);

            b00 =        (a00 ^ D0);
            b10 = ROTL64(a01 ^ D1,  1);
            b20 = ROTL64(a02 ^ D2, 62);
            b05 = ROTL64(a03 ^ D3, 28);
            b15 = ROTL64(a04 ^ D4, 27);
            b16 = ROTL64(a05 ^ D0, 36);
            b01 = ROTL64(a06 ^ D1, 44);
            b11 = ROTL64(a07 ^ D2,  6);
            b21 = ROTL64(a08 ^ D3, 55);
            b06 = ROTL64(a09 ^ D4, 20);
            b07 = ROTL64(a10 ^ D0,  3);
            b17 = ROTL64(a11 ^ D1, 10);
            b02 = ROTL64(a12 ^ D2, 43);
            b12 = ROTL64(a13 ^ D3, 25);
            b22 = ROTL64(a14 ^ D4, 39);
            b23 = ROTL64(a15 ^ D0, 41);
            b08 = ROTL64(a16 ^ D1, 45);
            b18 = ROTL64(a17 ^ D2, 15);
            b03 = ROTL64(a18 ^ D3, 21);
            b13 = ROTL64(a19 ^ D4,  8);
            b14 = ROTL64(a20 ^ D0, 18);
            b24 = ROTL64(a21 ^ D1,  2);
            b09 = ROTL64(a22 ^ D2, 61);
            b19 = ROTL64(a23 ^ D3, 56);
            b04 = ROTL64(a24 ^ D4, 14);

            a00 = b00 ^ ((~b01) & b02) ^ KECCAK_RC[r + 1u];
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

        // Spill back to S[] so we can do a data-dependent squeeze.
        S[ 0]=a00; S[ 1]=a01; S[ 2]=a02; S[ 3]=a03; S[ 4]=a04;
        S[ 5]=a05; S[ 6]=a06; S[ 7]=a07; S[ 8]=a08; S[ 9]=a09;
        S[10]=a10; S[11]=a11; S[12]=a12; S[13]=a13; S[14]=a14;
        S[15]=a15; S[16]=a16; S[17]=a17; S[18]=a18; S[19]=a19;
        S[20]=a20; S[21]=a21; S[22]=a22; S[23]=a23; S[24]=a24;

        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        for (uint j = 0u; j < take; ++j) {
            out_data[out_base + written + j] = S[j];
        }
        written += take;
        if (written >= out_lanes) break;
    }
}
```

Incumbent result:
     sha3_256_B16K: correct, 0.38 ms, 161.8 Gbitops/s (u64) (14.4% of 1125 Gops/s (u64 bitop, est))
    sha3_256_B256K: correct, 1.40 ms, 696.9 Gbitops/s (u64) (62.0% of 1125 Gops/s (u64 bitop, est))
      sha3_256_B4M: correct, 22.16 ms, 704.2 Gbitops/s (u64) (62.6% of 1125 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.3820

## History

- iter  3: compile=OK | correct=True | score=0.3404235613166802
- iter  4: compile=OK | correct=False | score=N/A
- iter  5: compile=OK | correct=True | score=0.3405687512836963
- iter  6: compile=OK | correct=True | score=0.35945115994324794
- iter  7: compile=OK | correct=True | score=0.38203009048350167
- iter  8: compile=OK | correct=True | score=0.33641367382947185
- iter  9: compile=OK | correct=False | score=N/A
- iter 10: compile=OK | correct=True | score=0.3286844231034277

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
