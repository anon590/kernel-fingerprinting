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

    ulong S[25];
    for (uint i = 0u; i < 25u; ++i) S[i] = 0ul;

    uint in_base = idx * msg_lanes;
    for (uint i = 0u; i < msg_lanes; ++i) {
        S[i] = in_data[in_base + i];
    }
    S[msg_lanes]       ^= (ulong)(domain & 0xFFu);
    S[rate_lanes - 1u] ^= 0x8000000000000000ul;

    uint out_base = idx * out_lanes;
    uint written = 0u;

    for (;;) {
        ulong a00=S[ 0], a01=S[ 1], a02=S[ 2], a03=S[ 3], a04=S[ 4];
        ulong a05=S[ 5], a06=S[ 6], a07=S[ 7], a08=S[ 8], a09=S[ 9];
        ulong a10=S[10], a11=S[11], a12=S[12], a13=S[13], a14=S[14];
        ulong a15=S[15], a16=S[16], a17=S[17], a18=S[18], a19=S[19];
        ulong a20=S[20], a21=S[21], a22=S[22], a23=S[23], a24=S[24];

        // Lane-complement transform: invert lanes 1, 2, 8, 12, 17, 20.
        // After the permutation these are inverted again before output.
        a01 = ~a01;
        a02 = ~a02;
        a08 = ~a08;
        a12 = ~a12;
        a17 = ~a17;
        a20 = ~a20;

        for (uint r = 0u; r < 24u; r += 2u) {
            // ============ Round r ============
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

            // chi with lane-complement. Inverted lanes after rho+pi:
            // pi: A''[y, (2x+3y) mod 5] = A'[x,y]
            // The inverted input lanes (1,2,8,12,17,20) map under pi to
            // b-positions. Compute pi mapping for each:
            //   (x=1,y=0)->b10  ; (x=2,y=0)->b20 ; (x=3,y=1)->b21 (a08)
            //   (x=2,y=2)->b02 (a12); (x=2,y=3)->b18 (a17); (x=0,y=4)->b14 (a20)
            // So inverted b-lanes are: b10, b20, b21, b02, b18, b14.
            // For chi row (x,y): out[x] = b[x] ^ ((~b[x+1]) & b[x+2])
            // We need to track which b's are inverted and rewrite:
            //   if b[x+1] inverted: (~b[x+1]) = original, so use b[x+1] & b[x+2] -> rewrite
            //   The standard pattern: 6 specific lanes inverted gives a fixed rewrite.
            //
            // Per-row analysis (b indexed by lane k = x + 5y):
            // y=0: b00 b01 b02* b03 b04   (b02 inverted)
            //   out00 = b00 ^ (~b01 & b02')  = b00 ^ (~b01 & ~b02) = b00 ^ ~(b01|b02)... no
            //   Let me use: ~b02' = b02 (since b02' = ~b02 stored). So if stored is inverted,
            //   (~stored) = original_value.
            //   Convention: let B = stored value. If lane is "inverted-stored", true_value = ~B.
            //   chi true: out_true[x] = true[x] ^ ((~true[x+1]) & true[x+2])
            //   We want to compute out_stored which keeps the same lane-inversion pattern
            //   for the NEXT round. After pi, the inverted set is fixed (b02,b10,b14,b18,b20,b21).
            //   After chi+iota, the inverted set for a-lanes should again be {1,2,8,12,17,20}.

            // Row y=0, lanes b00,b01,b02,b03,b04 -> a00,a01,a02,a03,a04
            // Stored-inverted in b: {b02}. Required inverted in a: {a01,a02}.
            // out_true[0] = b00 ^ (~b01 & ~b02_stored)  [since true(b02)=~b02_stored]
            //             = b00 ^ (~b01 & ~b02_s)
            //             = b00 ^ ~(b01 | b02_s)   -- but we want stored value for a00 (not inv)
            // a00_stored = out_true[0] = b00 ^ (~(b01 | b02_s))
            //
            // a01 needs inverted-stored: a01_s = ~out_true[1]
            //   out_true[1] = true(b01) ^ (~true(b02) & true(b03))
            //               = b01 ^ (~(~b02_s) & b03) = b01 ^ (b02_s & b03)
            //   a01_s = ~(b01 ^ (b02_s & b03))
            //
            // a02 needs inverted-stored: a02_s = ~out_true[2]
            //   out_true[2] = true(b02) ^ (~true(b03) & true(b04))
            //               = ~b02_s ^ (~b03 & b04)
            //   a02_s = ~(~b02_s ^ (~b03 & b04)) = b02_s ^ (~b03 & b04)
            //
            // a03 normal: out_true[3] = b03 ^ (~b04 & true(b00)) = b03 ^ (~b04 & b00)
            // a04 normal: out_true[4] = b04 ^ (~true(b00) & true(b01)) = b04 ^ (~b00 & b01)

            // Row y=1, b05..b09 -> a05..a09. Stored-inv in b: {}. Required inv in a: {a08}.
            // a05 = b05 ^ (~b06 & b07)
            // a06 = b06 ^ (~b07 & b08)
            // a07 = b07 ^ (~b08 & b09)
            // a08_s = ~(b08 ^ (~b09 & b05)) = b08 ^ ~(~b09 & b05) = b08 ^ (b09 | ~b05)
            //   simpler: a08_s = ~out_true[3 in row] -> compute then invert
            // a09 = b09 ^ (~b05 & b06)

            // Row y=2, b10..b14 -> a10..a14. Stored-inv in b: {b10, b14}. Required inv in a: {a12}.
            // true(b10) = ~b10_s, true(b14) = ~b14_s
            // a10_t = true(b10) ^ (~b11 & b12) = ~b10_s ^ (~b11 & b12)
            //   a10_stored (not inv) = a10_t = ~b10_s ^ (~b11 & b12)
            // a11_t = b11 ^ (~b12 & b13)
            // a12_t = b12 ^ (~b13 & true(b14)) = b12 ^ (~b13 & ~b14_s) = b12 ^ ~(b13 | b14_s)
            //   a12_s = ~a12_t = ~(b12 ^ ~(b13|b14_s)) = b12 ^ (b13 | b14_s)
            // a13_t = b13 ^ (~true(b14) & true(b10)) = b13 ^ (b14_s & ~b10_s)
            // a14_t = true(b14) ^ (~true(b10) & b11) = ~b14_s ^ (b10_s & b11)
            //   a14_stored = a14_t = ~b14_s ^ (b10_s & b11)

            // Row y=3, b15..b19 -> a15..a19. Stored-inv in b: {b18}. Required inv in a: {a17}.
            // a15 = b15 ^ (~b16 & b17)
            // a16 = b16 ^ (~b17 & true(b18)) = b16 ^ (~b17 & ~b18_s) = b16 ^ ~(b17|b18_s)
            //   a16_s_actually not inv -> a16_stored = b16 ^ ~(b17|b18_s)
            //   Hmm, but a16 is NOT in inverted set. So a16_stored = a16_true = b16 ^ ~(b17|b18_s).
            //   That's expensive. Alternative: just compute everything with true values then re-invert.

            // Given complexity, simplest correct approach: compute true values
            // (un-inverting on read) and re-invert on write. This still saves
            // because we batch the NOTs into the chi computation.

            // Compute true b-values for the 6 inverted positions:
            ulong tb02 = ~b02;
            ulong tb10 = ~b10;
            ulong tb14 = ~b14;
            ulong tb18 = ~b18;
            ulong tb20 = ~b20;
            ulong tb21 = ~b21;

            // chi using true values, producing true a-values, then we
            // re-invert the 6 stored-inverted a-lanes (1,2,8,12,17,20).
            ulong ta00 = b00  ^ ((~b01)  & tb02);
            ulong ta01 = b01  ^ ((~tb02) & b03);
            ulong ta02 = tb02 ^ ((~b03)  & b04);
            ulong ta03 = b03  ^ ((~b04)  & b00);
            ulong ta04 = b04  ^ ((~b00)  & b01);

            ulong ta05 = b05  ^ ((~b06)  & b07);
            ulong ta06 = b06  ^ ((~b07)  & b08);
            ulong ta07 = b07  ^ ((~b08)  & b09);
            ulong ta08 = b08  ^ ((~b09)  & b05);
            ulong ta09 = b09  ^ ((~b05)  & b06);

            ulong ta10 = tb10 ^ ((~b11)  & b12);
            ulong ta11 = b11  ^ ((~b12)  & b13);
            ulong ta12 = b12  ^ ((~b13)  & tb14);
            ulong ta13 = b13  ^ ((~tb14) & tb10);
            ulong ta14 = tb14 ^ ((~tb10) & b11);

            ulong ta15 = b15  ^ ((~b16)  & b17);
            ulong ta16 = b16  ^ ((~b17)  & tb18);
            ulong ta17 = b17  ^ ((~tb18) & b19);
            ulong ta18 = tb18 ^ ((~b19)  & b15);
            ulong ta19 = b19  ^ ((~b15)  & b16);

            ulong ta20 = tb20 ^ ((~tb21) & b22);
            ulong ta21 = tb21 ^ ((~b22)  & b23);
            ulong ta22 = b22  ^ ((~b23)  & b24);
            ulong ta23 = b23  ^ ((~b24)  & tb20);
            ulong ta24 = b24  ^ ((~tb20) & tb21);

            a00 = ta00 ^ KECCAK_RC[r];
            a01 = ~ta01;
            a02 = ~ta02;
            a03 = ta03;
            a04 = ta04;
            a05 = ta05;
            a06 = ta06;
            a07 = ta07;
            a08 = ~ta08;
            a09 = ta09;
            a10 = ta10;
            a11 = ta11;
            a12 = ~ta12;
            a13 = ta13;
            a14 = ta14;
            a15 = ta15;
            a16 = ta16;
            a17 = ~ta17;
            a18 = ta18;
            a19 = ta19;
            a20 = ~ta20;
            a21 = ta21;
            a22 = ta22;
            a23 = ta23;
            a24 = ta24;

            // ============ Round r+1 ============
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

            tb02 = ~b02;
            tb10 = ~b10;
            tb14 = ~b14;
            tb18 = ~b18;
            tb20 = ~b20;
            tb21 = ~b21;

            ta00 = b00  ^ ((~b01)  & tb02);
            ta01 = b01  ^ ((~tb02) & b03);
            ta02 = tb02 ^ ((~b03)  & b04);
            ta03 = b03  ^ ((~b04)  & b00);
            ta04 = b04  ^ ((~b00)  & b01);

            ta05 = b05  ^ ((~b06)  & b07);
            ta06 = b06  ^ ((~b07)  & b08);
            ta07 = b07  ^ ((~b08)  & b09);
            ta08 = b08  ^ ((~b09)  & b05);
            ta09 = b09  ^ ((~b05)  & b06);

            ta10 = tb10 ^ ((~b11)  & b12);
            ta11 = b11  ^ ((~b12)  & b13);
            ta12 = b12  ^ ((~b13)  & tb14);
            ta13 = b13  ^ ((~tb14) & tb10);
            ta14 = tb14 ^ ((~tb10) & b11);

            ta15 = b15  ^ ((~b16)  & b17);
            ta16 = b16  ^ ((~b17)  & tb18);
            ta17 = b17  ^ ((~tb18) & b19);
            ta18 = tb18 ^ ((~b19)  & b15);
            ta19 = b19  ^ ((~b15)  & b16);

            ta20 = tb20 ^ ((~tb21) & b22);
            ta21 = tb21 ^ ((~b22)  & b23);
            ta22 = b22  ^ ((~b23)  & b24);
            ta23 = b23  ^ ((~b24)  & tb20);
            ta24 = b24  ^ ((~tb20) & tb21);

            a00 = ta00 ^ KECCAK_RC[r + 1u];
            a01 = ~ta01;
            a02 = ~ta02;
            a03 = ta03;
            a04 = ta04;
            a05 = ta05;
            a06 = ta06;
            a07 = ta07;
            a08 = ~ta08;
            a09 = ta09;
            a10 = ta10;
            a11 = ta11;
            a12 = ~ta12;
            a13 = ta13;
            a14 = ta14;
            a15 = ta15;
            a16 = ta16;
            a17 = ~ta17;
            a18 = ta18;
            a19 = ta19;
            a20 = ~ta20;
            a21 = ta21;
            a22 = ta22;
            a23 = ta23;
            a24 = ta24;
        }

        // Un-invert lanes before squeezing.
        a01 = ~a01;
        a02 = ~a02;
        a08 = ~a08;
        a12 = ~a12;
        a17 = ~a17;
        a20 = ~a20;

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

Result of previous attempt:
     sha3_256_B16K: INCORRECT (bit_exact=65536)
  fail_reason: correctness failed at size sha3_256_B16K: bit_exact=65536

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

- iter  2: compile=OK | correct=True | score=0.3619794242470567
- iter  3: compile=OK | correct=True | score=0.3404235613166802
- iter  4: compile=OK | correct=False | score=N/A
- iter  5: compile=OK | correct=True | score=0.3405687512836963
- iter  6: compile=OK | correct=True | score=0.35945115994324794
- iter  7: compile=OK | correct=True | score=0.38203009048350167
- iter  8: compile=OK | correct=True | score=0.33641367382947185
- iter  9: compile=OK | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
