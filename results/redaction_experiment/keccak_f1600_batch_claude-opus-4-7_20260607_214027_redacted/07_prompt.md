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

    // Absorb. For typical case msg_lanes=4, this is just 4 XORs.
    uint in_base = idx * msg_lanes;
    {
        ulong tmp[25] = {0};
        for (uint i = 0u; i < msg_lanes; ++i) {
            tmp[i] = in_data[in_base + i];
        }
        tmp[msg_lanes]      ^= (ulong)(domain & 0xFFu);
        tmp[rate_lanes - 1u] ^= 0x8000000000000000ul;
        a00=tmp[ 0]; a01=tmp[ 1]; a02=tmp[ 2]; a03=tmp[ 3]; a04=tmp[ 4];
        a05=tmp[ 5]; a06=tmp[ 6]; a07=tmp[ 7]; a08=tmp[ 8]; a09=tmp[ 9];
        a10=tmp[10]; a11=tmp[11]; a12=tmp[12]; a13=tmp[13]; a14=tmp[14];
        a15=tmp[15]; a16=tmp[16]; a17=tmp[17]; a18=tmp[18]; a19=tmp[19];
        a20=tmp[20]; a21=tmp[21]; a22=tmp[22]; a23=tmp[23]; a24=tmp[24];
    }

    uint out_base = idx * out_lanes;
    uint written = 0u;

    for (;;) {
        for (uint r = 0u; r < 24u; ++r) {
            // theta
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

            // Fuse theta-XOR with rho+pi: compute b[x_new+5y_new] = ROTL(a[x,y] ^ D[x], rho).
            // Row y=0 (new positions from x=0..4): b00, b10, b20, b05, b15
            ulong b00 =       (a00 ^ D0);
            ulong b10 = ROTL64(a01 ^ D1,  1);
            ulong b20 = ROTL64(a02 ^ D2, 62);
            ulong b05 = ROTL64(a03 ^ D3, 28);
            ulong b15 = ROTL64(a04 ^ D4, 27);

            // Row y=1: b16, b01, b11, b21, b06
            ulong b16 = ROTL64(a05 ^ D0, 36);
            ulong b01 = ROTL64(a06 ^ D1, 44);
            ulong b11 = ROTL64(a07 ^ D2,  6);
            ulong b21 = ROTL64(a08 ^ D3, 55);
            ulong b06 = ROTL64(a09 ^ D4, 20);

            // Row y=2: b07, b17, b02, b12, b22
            ulong b07 = ROTL64(a10 ^ D0,  3);
            ulong b17 = ROTL64(a11 ^ D1, 10);
            ulong b02 = ROTL64(a12 ^ D2, 43);
            ulong b12 = ROTL64(a13 ^ D3, 25);
            ulong b22 = ROTL64(a14 ^ D4, 39);

            // Row y=3: b23, b08, b18, b03, b13
            ulong b23 = ROTL64(a15 ^ D0, 41);
            ulong b08 = ROTL64(a16 ^ D1, 45);
            ulong b18 = ROTL64(a17 ^ D2, 15);
            ulong b03 = ROTL64(a18 ^ D3, 21);
            ulong b13 = ROTL64(a19 ^ D4,  8);

            // Row y=4: b14, b24, b09, b19, b04
            ulong b14 = ROTL64(a20 ^ D0, 18);
            ulong b24 = ROTL64(a21 ^ D1,  2);
            ulong b09 = ROTL64(a22 ^ D2, 61);
            ulong b19 = ROTL64(a23 ^ D3, 56);
            ulong b04 = ROTL64(a24 ^ D4, 14);

            // chi + iota
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

        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;

        // Squeeze via local array.
        ulong O[25];
        O[ 0]=a00; O[ 1]=a01; O[ 2]=a02; O[ 3]=a03; O[ 4]=a04;
        O[ 5]=a05; O[ 6]=a06; O[ 7]=a07; O[ 8]=a08; O[ 9]=a09;
        O[10]=a10; O[11]=a11; O[12]=a12; O[13]=a13; O[14]=a14;
        O[15]=a15; O[16]=a16; O[17]=a17; O[18]=a18; O[19]=a19;
        O[20]=a20; O[21]=a21; O[22]=a22; O[23]=a23; O[24]=a24;

        for (uint j = 0u; j < take; ++j) {
            out_data[out_base + written + j] = O[j];
        }
        written += take;
        if (written >= out_lanes) break;
    }
}
```

Result of previous attempt:
     sha3_256_B16K: correct, 0.36 ms, 168.4 Gbitops/s (u64) (15.0% of 1125 Gops/s (u64 bitop, est))
    sha3_256_B256K: correct, 1.80 ms, 541.8 Gbitops/s (u64) (48.2% of 1125 Gops/s (u64 bitop, est))
      sha3_256_B4M: correct, 21.53 ms, 724.7 Gbitops/s (u64) (64.4% of 1125 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.3595

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

    // 25 lanes: a[x + 5*y]
    ulong a00=0, a01=0, a02=0, a03=0, a04=0;
    ulong a05=0, a06=0, a07=0, a08=0, a09=0;
    ulong a10=0, a11=0, a12=0, a13=0, a14=0;
    ulong a15=0, a16=0, a17=0, a18=0, a19=0;
    ulong a20=0, a21=0, a22=0, a23=0, a24=0;

    // Absorb single block.
    uint in_base = idx * msg_lanes;
    // Write into a temporary array on the stack, then copy out.
    // Use a switch-style fan-out so we don't need pointer arrays.
    for (uint i = 0u; i < msg_lanes; ++i) {
        ulong v = in_data[in_base + i];
        switch (i) {
            case  0: a00 ^= v; break; case  1: a01 ^= v; break;
            case  2: a02 ^= v; break; case  3: a03 ^= v; break;
            case  4: a04 ^= v; break; case  5: a05 ^= v; break;
            case  6: a06 ^= v; break; case  7: a07 ^= v; break;
            case  8: a08 ^= v; break; case  9: a09 ^= v; break;
            case 10: a10 ^= v; break; case 11: a11 ^= v; break;
            case 12: a12 ^= v; break; case 13: a13 ^= v; break;
            case 14: a14 ^= v; break; case 15: a15 ^= v; break;
            case 16: a16 ^= v; break;
            default: break;
        }
    }
    // Domain byte at lane msg_lanes, byte 0.
    {
        ulong dv = (ulong)(domain & 0xFFu);
        switch (msg_lanes) {
            case  0: a00 ^= dv; break; case  1: a01 ^= dv; break;
            case  2: a02 ^= dv; break; case  3: a03 ^= dv; break;
            case  4: a04 ^= dv; break; case  5: a05 ^= dv; break;
            case  6: a06 ^= dv; break; case  7: a07 ^= dv; break;
            case  8: a08 ^= dv; break; case  9: a09 ^= dv; break;
            case 10: a10 ^= dv; break; case 11: a11 ^= dv; break;
            case 12: a12 ^= dv; break; case 13: a13 ^= dv; break;
            case 14: a14 ^= dv; break; case 15: a15 ^= dv; break;
            case 16: a16 ^= dv; break;
            default: break;
        }
    }
    // 0x80 at byte position rate_bytes-1: lane rate_lanes-1, byte 7.
    {
        ulong pv = 0x8000000000000000ul;
        uint  pl = rate_lanes - 1u;
        switch (pl) {
            case  0: a00 ^= pv; break; case  1: a01 ^= pv; break;
            case  2: a02 ^= pv; break; case  3: a03 ^= pv; break;
            case  4: a04 ^= pv; break; case  5: a05 ^= pv; break;
            case  6: a06 ^= pv; break; case  7: a07 ^= pv; break;
            case  8: a08 ^= pv; break; case  9: a09 ^= pv; break;
            case 10: a10 ^= pv; break; case 11: a11 ^= pv; break;
            case 12: a12 ^= pv; break; case 13: a13 ^= pv; break;
            case 14: a14 ^= pv; break; case 15: a15 ^= pv; break;
            case 16: a16 ^= pv; break; case 17: a17 ^= pv; break;
            case 18: a18 ^= pv; break; case 19: a19 ^= pv; break;
            case 20: a20 ^= pv; break;
            default: break;
        }
    }

    uint out_base = idx * out_lanes;
    uint written = 0u;

    for (;;) {
        // 24 rounds, fully scalarised.
        for (uint r = 0u; r < 24u; ++r) {
            // theta
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

            ulong t00 = a00 ^ D0;
            ulong t01 = a01 ^ D1;
            ulong t02 = a02 ^ D2;
            ulong t03 = a03 ^ D3;
            ulong t04 = a04 ^ D4;
            ulong t05 = a05 ^ D0;
            ulong t06 = a06 ^ D1;
            ulong t07 = a07 ^ D2;
            ulong t08 = a08 ^ D3;
            ulong t09 = a09 ^ D4;
            ulong t10 = a10 ^ D0;
            ulong t11 = a11 ^ D1;
            ulong t12 = a12 ^ D2;
            ulong t13 = a13 ^ D3;
            ulong t14 = a14 ^ D4;
            ulong t15 = a15 ^ D0;
            ulong t16 = a16 ^ D1;
            ulong t17 = a17 ^ D2;
            ulong t18 = a18 ^ D3;
            ulong t19 = a19 ^ D4;
            ulong t20 = a20 ^ D0;
            ulong t21 = a21 ^ D1;
            ulong t22 = a22 ^ D2;
            ulong t23 = a23 ^ D3;
            ulong t24 = a24 ^ D4;

            // rho + pi: B[x_new + 5*y_new] = rotl(t[x+5y], rho[x+5y])
            // with x_new = y, y_new = (2x+3y) % 5.
            ulong b00 = t00;
            ulong b10 = ROTL64(t01,  1);
            ulong b20 = ROTL64(t02, 62);
            ulong b05 = ROTL64(t03, 28);
            ulong b15 = ROTL64(t04, 27);

            ulong b16 = ROTL64(t05, 36);
            ulong b01 = ROTL64(t06, 44);
            ulong b11 = ROTL64(t07,  6);
            ulong b21 = ROTL64(t08, 55);
            ulong b06 = ROTL64(t09, 20);

            ulong b07 = ROTL64(t10,  3);
            ulong b17 = ROTL64(t11, 10);
            ulong b02 = ROTL64(t12, 43);
            ulong b12 = ROTL64(t13, 25);
            ulong b22 = ROTL64(t14, 39);

            ulong b23 = ROTL64(t15, 41);
            ulong b08 = ROTL64(t16, 45);
            ulong b18 = ROTL64(t17, 15);
            ulong b03 = ROTL64(t18, 21);
            ulong b13 = ROTL64(t19,  8);

            ulong b14 = ROTL64(t20, 18);
            ulong b24 = ROTL64(t21,  2);
            ulong b09 = ROTL64(t22, 61);
            ulong b19 = ROTL64(t23, 56);
            ulong b04 = ROTL64(t24, 14);

            // chi
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

            // iota
            a00 ^= KECCAK_RC[r];
        }

        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        for (uint j = 0u; j < take; ++j) {
            ulong v;
            switch (j) {
                case  0: v = a00; break; case  1: v = a01; break;
                case  2: v = a02; break; case  3: v = a03; break;
                case  4: v = a04; break; case  5: v = a05; break;
                case  6: v = a06; break; case  7: v = a07; break;
                case  8: v = a08; break; case  9: v = a09; break;
                case 10: v = a10; break; case 11: v = a11; break;
                case 12: v = a12; break; case 13: v = a13; break;
                case 14: v = a14; break; case 15: v = a15; break;
                case 16: v = a16; break; case 17: v = a17; break;
                case 18: v = a18; break; case 19: v = a19; break;
                case 20: v = a20; break; case 21: v = a21; break;
                case 22: v = a22; break; case 23: v = a23; break;
                default: v = a24; break;
            }
            out_data[out_base + written + j] = v;
        }
        written += take;
        if (written >= out_lanes) break;
    }
}
```

Incumbent result:
     sha3_256_B16K: correct, 0.34 ms, 180.1 Gbitops/s (u64) (16.0% of 1125 Gops/s (u64 bitop, est))
    sha3_256_B256K: correct, 1.84 ms, 529.7 Gbitops/s (u64) (47.1% of 1125 Gops/s (u64 bitop, est))
      sha3_256_B4M: correct, 22.04 ms, 707.9 Gbitops/s (u64) (62.9% of 1125 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.3620

## History

- iter  0: compile=OK | correct=True | score=0.03921608618417052
- iter  1: compile=FAIL | correct=False | score=N/A
- iter  2: compile=OK | correct=True | score=0.3619794242470567
- iter  3: compile=OK | correct=True | score=0.3404235613166802
- iter  4: compile=OK | correct=False | score=N/A
- iter  5: compile=OK | correct=True | score=0.3405687512836963
- iter  6: compile=OK | correct=True | score=0.35945115994324794

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
