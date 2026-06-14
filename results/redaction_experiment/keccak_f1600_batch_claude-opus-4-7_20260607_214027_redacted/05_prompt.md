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

    // Build state via a small local array (msg_lanes is typically small)
    // then move into scalar registers for the permutation.
    ulong s[25];
    #pragma unroll
    for (uint i = 0u; i < 25u; ++i) s[i] = 0ul;

    uint in_base = idx * msg_lanes;
    for (uint i = 0u; i < msg_lanes; ++i) {
        s[i] = in_data[in_base + i];
    }
    s[msg_lanes]      ^= (ulong)(domain & 0xFFu);
    s[rate_lanes - 1u] ^= 0x8000000000000000ul;

    // Lane-complement trick: pre-complement lanes
    //   1, 2, 8, 12, 17, 20 (Bertoni et al., "Keccak implementation overview").
    // We carry the state with these lanes stored as their bitwise complements,
    // and adjust the chi step to remove ~ on those positions.
    ulong a00 = s[ 0];
    ulong a01 = ~s[ 1];
    ulong a02 = ~s[ 2];
    ulong a03 = s[ 3];
    ulong a04 = s[ 4];
    ulong a05 = s[ 5];
    ulong a06 = s[ 6];
    ulong a07 = s[ 7];
    ulong a08 = ~s[ 8];
    ulong a09 = s[ 9];
    ulong a10 = s[10];
    ulong a11 = s[11];
    ulong a12 = ~s[12];
    ulong a13 = s[13];
    ulong a14 = s[14];
    ulong a15 = s[15];
    ulong a16 = s[16];
    ulong a17 = ~s[17];
    ulong a18 = s[18];
    ulong a19 = s[19];
    ulong a20 = ~s[20];
    ulong a21 = s[21];
    ulong a22 = s[22];
    ulong a23 = s[23];
    ulong a24 = s[24];

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

            // chi with lane-complement trick.
            // Complemented lanes after rho/pi (which preserves bitwise complement):
            //   original complemented set on input lanes (x+5y):
            //     1, 2, 8, 12, 17, 20.
            //   After pi: A''[y, (2x+3y)%5] = A'[x,y]; mapping:
            //     (1,0)->(0,2)  lane 10  (b10)
            //     (2,0)->(0,4)  lane 20  (b20)
            //     (3,1)->(1,4)  lane 21  (b21)  -- lane 8 was (3,1)
            //     (2,2)->(2,1)  lane  7  (b07)  -- lane 12 was (2,2)
            //     (2,3)->(3,2)  lane 17  (b17)  -- lane 17 was (2,3)
            //     (0,4)->(4,2)  lane 14  (b14)  -- lane 20 was (0,4)
            // We perform chi accordingly. Then in the output, we want lanes
            //  {1,2,8,12,17,20} (post-chi positions) to be stored as
            // their complements again to maintain the invariant for the next round.
            //
            // Standard chi:   a[x,y] = b[x] ^ ((~b[x+1]) & b[x+2])
            // If b[i] is stored as ~B[i] (true value is ~b[i]), then
            //   ~b[x+1] in "true logic" becomes b[x+1] (the stored value),
            // so the AND term may use OR-NOT identities.
            //
            // We just write out the row-by-row chi explicitly with the right
            // operator choices so that the OUTPUT a is the complement of the
            // true value at positions {1,2,8,12,17,20}, and the true value
            // elsewhere.

            // Row y=0: b00, b01, b02, b03, b04 with b10, b20 from this row?
            // Wait: row y=0 means lanes 0..4. Complemented inputs in this row:
            //   b01? lane 1 -> b01 from pi mapping of lane 6 (t06). Not complemented.
            //   b02? lane 2 -> b02 from t12 (lane 12). YES complemented.
            //   b00,b03,b04 -> not complemented.
            // Row y=0 complement mask: {2}
            //
            // Row y=1: lanes 5..9.
            //   b05 from t03 (lane 3): no.
            //   b06 from t09 (lane 9): no.
            //   b07 from t10 (lane 10)? No, b07 comes from t10 mapping; t10 is
            //     lane 10 (not in complemented set). But our complemented post-pi
            //     set includes b07 (from lane 12 originally). Let me recompute.
            //
            // Actually I made an error above. Let's recompute the post-pi
            // complement set carefully.
            //
            // The pi step writes:  B[ y, (2x+3y) mod 5 ] = rotl(A[x,y], rho[x,y])
            // i.e. new (X,Y) = (y, (2x+3y) mod 5). The lane index of the source
            // is L = x + 5y; the destination lane is X + 5Y = y + 5*((2x+3y)%5).
            //
            // Originally complemented source lanes L in {1,2,8,12,17,20}:
            //   L=1  -> (x,y)=(1,0); (X,Y)=(0,2); dest lane = 10. -> b10
            //   L=2  -> (2,0); (X,Y)=(0,4); dest = 20. -> b20
            //   L=8  -> (3,1); (X,Y)=(1,(6+3)%5=4); dest = 1+20=21. -> b21
            //   L=12 -> (2,2); (X,Y)=(2,(4+6)%5=0); dest = 2+0=2. -> b02
            //   L=17 -> (2,3); (X,Y)=(3,(4+9)%5=3); dest = 3+15=18. -> b18
            //   L=20 -> (0,4); (X,Y)=(4,(0+12)%5=2); dest = 4+10=14. -> b14
            //
            // So post-pi complemented set: {b10, b20, b21, b02, b18, b14}.

            // Now chi rows (y is fixed; lanes are X=0..4 for that y):
            // y=0: lanes 0..4: {b00,b01,b02,b03,b04}; complemented: {b02}
            // y=1: lanes 5..9: {b05,b06,b07,b08,b09}; complemented: {} (none of 7,8,9,5,6)
            // y=2: lanes 10..14:{b10,b11,b12,b13,b14}; complemented: {b10,b14}
            // y=3: lanes 15..19:{b15,b16,b17,b18,b19}; complemented: {b18}
            // y=4: lanes 20..24:{b20,b21,b22,b23,b24}; complemented: {b20,b21}

            // Output complement mask (so that result lanes
            // {1,2,8,12,17,20} are stored as complements again):
            //   need a01, a02, a08, a12, a17, a20 to be stored as ~true.

            // Helper: chi formula true_a[i] = true_b[i] XOR ((~true_b[i+1]) AND true_b[i+2])
            // We'll write each lane carefully using De Morgan as needed.
            //
            // For a lane where b is stored as B_s and true_b = ~B_s, we substitute.
            // Let cN = complemented flag for input b at position i+0,i+1,i+2.
            // We want to express
            //   true_a = true_b0 XOR ((~true_b1) AND true_b2)
            // and possibly invert the result if output should be stored
            // complemented.

            // ---- Row y=0: b00,b01,b02,b03,b04; comp set {b02}.
            // Output complement set: a01, a02.
            //
            // a00 (out: true): true_a00 = b00 ^ (~b01 & ~B02)
            //   = b00 ^ (~b01 & ~B02). Note ~B02 = true_b02. So
            //   a00 = b00 ^ (~b01 & true_b02). But we don't have true_b02
            //   separately. Use: ~b01 & true_b02 = ~b01 & ~B02 = ~(b01 | B02).
            //   So a00 = b00 ^ ~(b01 | b02).
            // a01 (out: comp): we want stored ~true_a01.
            //   true_a01 = b01 ^ (~true_b02 & b03) = b01 ^ (B02 & b03).
            //   stored a01 = ~(b01 ^ (B02 & b03)) = b01 ^ ~(B02 & b03).
            // a02 (out: comp): stored ~true_a02.
            //   true_a02 = true_b02 ^ (~b03 & b04) = ~B02 ^ (~b03 & b04).
            //   stored a02 = B02 ^ (~b03 & b04).
            //   (since flipping the first XOR operand flips the result.)
            // a03 (out: true): true_a03 = b03 ^ (~b04 & b00).
            //   = b03 ^ (~b04 & b00).
            // a04 (out: true): true_a04 = b04 ^ (~b00 & b01).

            ulong na00 = b00 ^ (~(b01 | b02));
            ulong na01 = b01 ^ ~(b02 & b03);
            ulong na02 = b02 ^ (~b03 & b04);
            ulong na03 = b03 ^ (~b04 & b00);
            ulong na04 = b04 ^ (~b00 & b01);

            // ---- Row y=1: b05..b09; comp set {} (none).
            // Output complement set: a08.
            // All inputs are "true" representations here.
            // a05 = b05 ^ (~b06 & b07)
            // a06 = b06 ^ (~b07 & b08)
            // a07 = b07 ^ (~b08 & b09)
            // a08 (out comp): stored = ~true_a08
            //   true_a08 = b08 ^ (~b09 & b05); stored = b08 ^ ~(~b09 & b05)
            //            = b08 ^ (b09 | ~b05).
            // a09 = b09 ^ (~b05 & b06)

            ulong na05 = b05 ^ (~b06 & b07);
            ulong na06 = b06 ^ (~b07 & b08);
            ulong na07 = b07 ^ (~b08 & b09);
            ulong na08 = b08 ^ (b09 | ~b05);
            ulong na09 = b09 ^ (~b05 & b06);

            // ---- Row y=2: b10..b14; comp set {b10, b14}.
            // Output complement set: a12.
            //
            // a10 (out: true): true_a10 = true_b10 ^ (~b11 & b12)
            //   = ~B10 ^ (~b11 & b12).
            //   In stored terms: a10 = ~B10 ^ (~b11 & b12) = (b10 ^ ~0) ^ (~b11 & b12)
            //     i.e. a10 = ~(b10) ^ (~b11 & b12).
            //   But we want a10 stored as TRUE (not complemented). So
            //   a10 = ~b10 ^ (~b11 & b12).
            // a11 (out: true): true_a11 = b11 ^ (~b12 & b13).
            //   a11 = b11 ^ (~b12 & b13).
            // a12 (out: comp): stored = ~true_a12
            //   true_a12 = b12 ^ (~b13 & true_b14) = b12 ^ (~b13 & ~B14)
            //            = b12 ^ ~(b13 | B14).
            //   stored a12 = ~true_a12 = b12 ^ (b13 | b14).
            // a13 (out: true): true_a13 = b13 ^ (~true_b14 & true_b10)
            //   = b13 ^ (B14 & ~B10).
            //   a13 = b13 ^ (b14 & ~b10).
            // a14 (out: true): true_a14 = true_b14 ^ (~true_b10 & b11)
            //   = ~B14 ^ (B10 & b11)
            //   a14 = ~b14 ^ (b10 & b11).

            ulong na10 = ~b10 ^ (~b11 & b12);
            ulong na11 = b11 ^ (~b12 & b13);
            ulong na12 = b12 ^ (b13 | b14);
            ulong na13 = b13 ^ (b14 & ~b10);
            ulong na14 = ~b14 ^ (b10 & b11);

            // ---- Row y=3: b15..b19; comp set {b18}.
            // Output complement set: a17.
            //
            // a15 = b15 ^ (~b16 & b17)
            // a16 = b16 ^ (~b17 & true_b18) = b16 ^ (~b17 & ~B18)
            //     = b16 ^ ~(b17 | b18).
            // a17 (out comp): stored = ~true_a17
            //   true_a17 = b17 ^ (~true_b18 & b19) = b17 ^ (B18 & b19).
            //   stored a17 = b17 ^ ~(b18 & b19).
            // a18 (out true): true_a18 = true_b18 ^ (~b19 & b15)
            //   = ~B18 ^ (~b19 & b15). To store true: a18 = ~b18 ^ (~b19 & b15).
            // a19 = b19 ^ (~b15 & b16)

            ulong na15 = b15 ^ (~b16 & b17);
            ulong na16 = b16 ^ ~(b17 | b18);
            ulong na17 = b17 ^ ~(b18 & b19);
            ulong na18 = ~b18 ^ (~b19 & b15);
            ulong na19 = b19 ^ (~b15 & b16);

            // ---- Row y=4: b20..b24; comp set {b20, b21}.
            // Output complement set: a20.
            //
            // a20 (out comp): stored = ~true_a20.
            //   true_a20 = true_b20 ^ (~true_b21 & b22) = ~B20 ^ (B21 & b22).
            //   stored a20 = ~true_a20 = B20 ^ (B21 & b22) = b20 ^ (b21 & b22).
            // a21 (out true): true_a21 = true_b21 ^ (~b22 & b23) = ~B21 ^ (~b22 & b23).
            //   stored true: a21 = ~b21 ^ (~b22 & b23).
            // a22 = b22 ^ (~b23 & b24)
            // a23 (out true): true_a23 = b23 ^ (~b24 & true_b20)
            //   = b23 ^ (~b24 & ~B20) = b23 ^ ~(b24 | B20)
            //   a23 = b23 ^ ~(b24 | b20).
            // a24 (out true): true_a24 = b24 ^ (~true_b20 & true_b21)
            //   = b24 ^ (B20 & ~B21) = b24 ^ (b20 & ~b21).

            ulong na20 = b20 ^ (b21 & b22);
            ulong na21 = ~b21 ^ (~b22 & b23);
            ulong na22 = b22 ^ (~b23 & b24);
            ulong na23 = b23 ^ ~(b24 | b20);
            ulong na24 = b24 ^ (b20 & ~b21);

            // iota: XOR RC into lane 0 (which is stored as true).
            na00 ^= KECCAK_RC[r];

            a00 = na00; a01 = na01; a02 = na02; a03 = na03; a04 = na04;
            a05 = na05; a06 = na06; a07 = na07; a08 = na08; a09 = na09;
            a10 = na10; a11 = na11; a12 = na12; a13 = na13; a14 = na14;
            a15 = na15; a16 = na16; a17 = na17; a18 = na18; a19 = na19;
            a20 = na20; a21 = na21; a22 = na22; a23 = na23; a24 = na24;
        }

        // Squeeze: undo the complement on lanes {1,2,8,12,17,20} for output.
        // We materialise to a small local array so the variable-length squeeze
        // can be a simple loop.
        ulong o[25];
        o[ 0] = a00;
        o[ 1] = ~a01;
        o[ 2] = ~a02;
        o[ 3] = a03;
        o[ 4] = a04;
        o[ 5] = a05;
        o[ 6] = a06;
        o[ 7] = a07;
        o[ 8] = ~a08;
        o[ 9] = a09;
        o[10] = a10;
        o[11] = a11;
        o[12] = ~a12;
        o[13] = a13;
        o[14] = a14;
        o[15] = a15;
        o[16] = a16;
        o[17] = ~a17;
        o[18] = a18;
        o[19] = a19;
        o[20] = ~a20;
        o[21] = a21;
        o[22] = a22;
        o[23] = a23;
        o[24] = a24;

        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        for (uint j = 0u; j < take; ++j) {
            out_data[out_base + written + j] = o[j];
        }
        written += take;
        if (written >= out_lanes) break;

        // For another squeeze block, re-load the state from o[] (already true
        // values) and re-apply the complement-mask for the next permutation.
        a00 = o[ 0];
        a01 = ~o[ 1];
        a02 = ~o[ 2];
        a03 = o[ 3];
        a04 = o[ 4];
        a05 = o[ 5];
        a06 = o[ 6];
        a07 = o[ 7];
        a08 = ~o[ 8];
        a09 = o[ 9];
        a10 = o[10];
        a11 = o[11];
        a12 = ~o[12];
        a13 = o[13];
        a14 = o[14];
        a15 = o[15];
        a16 = o[16];
        a17 = ~o[17];
        a18 = o[18];
        a19 = o[19];
        a20 = ~o[20];
        a21 = o[21];
        a22 = o[22];
        a23 = o[23];
        a24 = o[24];
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

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
