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

    // 25 state lanes laid out as (x + 5*y).
    ulong a00=0, a01=0, a02=0, a03=0, a04=0;
    ulong a05=0, a06=0, a07=0, a08=0, a09=0;
    ulong a10=0, a11=0, a12=0, a13=0, a14=0;
    ulong a15=0, a16=0, a17=0, a18=0, a19=0;
    ulong a20=0, a21=0, a22=0, a23=0, a24=0;

    // Absorb (single block).
    uint in_base = idx * msg_lanes;
    ulong *Aarr[25] = {
        &a00,&a01,&a02,&a03,&a04,
        &a05,&a06,&a07,&a08,&a09,
        &a10,&a11,&a12,&a13,&a14,
        &a15,&a16,&a17,&a18,&a19,
        &a20,&a21,&a22,&a23,&a24
    };
    for (uint i = 0u; i < msg_lanes; ++i) {
        *Aarr[i] ^= in_data[in_base + i];
    }
    *Aarr[msg_lanes] ^= (ulong)(domain & 0xFFu);
    *Aarr[rate_lanes - 1u] ^= 0x8000000000000000ul;

    uint out_base = idx * out_lanes;
    uint written = 0u;

    for (;;) {
        // ---- 24 rounds of Keccak-f[1600], fully unrolled per round ----
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

            // rho + pi: B[y, (2x+3y)%5] = rotl(t[x,y], r[x][y])
            // i.e. B[x_new + 5*y_new] with x_new=y, y_new=(2x+3y)%5
            // Rho offsets r[x][y] indexed by (x+5y):
            //  0  1 62 28 27
            // 36 44  6 55 20
            //  3 10 43 25 39
            // 41 45 15 21  8
            // 18  2 61 56 14
            ulong b00 = t00;                 // (0,0)->(0,0), rot 0
            ulong b10 = ROTL64(t01,  1);     // (1,0)->(0,2)
            ulong b20 = ROTL64(t02, 62);     // (2,0)->(0,4)
            ulong b05 = ROTL64(t03, 28);     // (3,0)->(0,1)
            ulong b15 = ROTL64(t04, 27);     // (4,0)->(0,3)

            ulong b16 = ROTL64(t05, 36);     // (0,1)->(1,3)
            ulong b01 = ROTL64(t06, 44);     // (1,1)->(1,0)
            ulong b11 = ROTL64(t07,  6);     // (2,1)->(1,2)
            ulong b21 = ROTL64(t08, 55);     // (3,1)->(1,4)
            ulong b06 = ROTL64(t09, 20);     // (4,1)->(1,1)

            ulong b07 = ROTL64(t10,  3);     // (0,2)->(2,1)
            ulong b17 = ROTL64(t11, 10);     // (1,2)->(2,3)
            ulong b02 = ROTL64(t12, 43);     // (2,2)->(2,0)
            ulong b12 = ROTL64(t13, 25);     // (3,2)->(2,2)
            ulong b22 = ROTL64(t14, 39);     // (4,2)->(2,4)

            ulong b23 = ROTL64(t15, 41);     // (0,3)->(3,4)
            ulong b08 = ROTL64(t16, 45);     // (1,3)->(3,1)
            ulong b18 = ROTL64(t17, 15);     // (2,3)->(3,3)
            ulong b03 = ROTL64(t18, 21);     // (3,3)->(3,0)
            ulong b13 = ROTL64(t19,  8);     // (4,3)->(3,2)

            ulong b14 = ROTL64(t20, 18);     // (0,4)->(4,2)
            ulong b24 = ROTL64(t21,  2);     // (1,4)->(4,4)
            ulong b09 = ROTL64(t22, 61);     // (2,4)->(4,1)
            ulong b19 = ROTL64(t23, 56);     // (3,4)->(4,3)
            ulong b04 = ROTL64(t24, 14);     // (4,4)->(4,0)

            // chi: A[x,y] = B[x,y] ^ ((~B[x+1,y]) & B[x+2,y])
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

        // Squeeze rate_lanes lanes (or fewer for the final chunk).
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

Result of previous attempt:
  COMPILE FAILED: Error Domain=MTLLibraryErrorDomain Code=3 "program_source:42:11: error: pointer type must have explicit address space qualifier
    ulong *Aarr[25] = {
          ^
" UserInfo={NSLocalizedDescription=program_source:42:11: error: pointer type must have explicit address space qualifier
    ulong *Aarr[25] = {
          ^
}

## Current best (incumbent)

```metal
// Naive seed for batched Keccak-f[1600] sponge (one thread per instance).
//
// Each thread runs the full pipeline for one independent sponge:
// initialise a 25-lane 64-bit state to zero, XOR ``msg_bytes / 8``
// input lanes into the state, apply the FIPS 202 padding, then
// alternate Keccak-f[1600] permutations with ``rate_bytes / 8`` lane
// writes to the output until ``out_bytes / 8`` lanes have been
// emitted (the last chunk may be shorter than the rate).
//
// All test sizes have ``msg_bytes < rate_bytes`` (one absorb block)
// and ``msg_bytes``, ``rate_bytes``, ``out_bytes`` all multiples of 8.
// Lane k of the 5x5 state corresponds to the (x, y) cell with
// x = k % 5, y = k / 5 (lane index k = x + 5*y).
//
// Buffer layout (host-fixed; preserved by candidate):
//   buffer 0: device const ulong *in_data    (batch * msg_bytes/8)
//   buffer 1: device       ulong *out_data   (batch * out_bytes/8)
//   buffer 2: constant uint &batch
//   buffer 3: constant uint &msg_bytes
//   buffer 4: constant uint &rate_bytes
//   buffer 5: constant uint &out_bytes
//   buffer 6: constant uint &domain          (low 8 bits = padding domain byte)
//
// Dispatch (host-provided):
//   threadsPerGrid        = (batch, 1, 1)
//   threadsPerThreadgroup = (min(batch, 64), 1, 1)

#include <metal_stdlib>
using namespace metal;

// FIPS 202 round constants for Keccak-f[1600].
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

// FIPS 202 rho offsets, indexed by lane (x + 5*y) for x,y in 0..5.
constant uint KECCAK_RHO[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
};

inline ulong rotl64(ulong x, uint k) {
    k &= 63u;
    if (k == 0u) return x;
    return (x << k) | (x >> (64u - k));
}

inline void keccak_f1600(thread ulong *A) {
    ulong C[5];
    ulong D[5];
    ulong B[25];
    for (uint r = 0u; r < 24u; ++r) {
        // theta: column XOR + 1-bit-rotated lateral mix.
        for (uint x = 0u; x < 5u; ++x) {
            C[x] = A[x] ^ A[x + 5u] ^ A[x + 10u] ^ A[x + 15u] ^ A[x + 20u];
        }
        for (uint x = 0u; x < 5u; ++x) {
            D[x] = C[(x + 4u) % 5u] ^ rotl64(C[(x + 1u) % 5u], 1u);
        }
        for (uint y = 0u; y < 5u; ++y) {
            for (uint x = 0u; x < 5u; ++x) {
                A[x + 5u * y] ^= D[x];
            }
        }

        // rho + pi: rotate each lane by r[x][y] and scatter to
        // destination cell (x_new, y_new) = (y, (2*x + 3*y) % 5).
        for (uint y = 0u; y < 5u; ++y) {
            for (uint x = 0u; x < 5u; ++x) {
                uint src = x + 5u * y;
                uint x_new = y;
                uint y_new = (2u * x + 3u * y) % 5u;
                B[x_new + 5u * y_new] = rotl64(A[src], KECCAK_RHO[src]);
            }
        }

        // chi: nonlinear row mix.
        for (uint y = 0u; y < 5u; ++y) {
            for (uint x = 0u; x < 5u; ++x) {
                uint i  = x + 5u * y;
                uint i1 = ((x + 1u) % 5u) + 5u * y;
                uint i2 = ((x + 2u) % 5u) + 5u * y;
                A[i] = B[i] ^ ((~B[i1]) & B[i2]);
            }
        }

        // iota.
        A[0] ^= KECCAK_RC[r];
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

    uint msg_lanes  = msg_bytes  >> 3;   // msg_bytes  / 8
    uint rate_lanes = rate_bytes >> 3;   // rate_bytes / 8
    uint out_lanes  = out_bytes  >> 3;   // out_bytes  / 8

    thread ulong A[25];
    for (uint i = 0u; i < 25u; ++i) A[i] = 0ul;

    // Absorb single block: XOR message lanes into state lanes 0..msg_lanes-1.
    uint in_base = idx * msg_lanes;
    for (uint i = 0u; i < msg_lanes; ++i) {
        A[i] ^= in_data[in_base + i];
    }
    // Pad domain byte at byte position msg_bytes
    // (lane msg_lanes, byte 0 of that lane).
    A[msg_lanes] ^= (ulong)(domain & 0xFFu);
    // Pad 0x80 at byte position rate_bytes - 1
    // (lane rate_lanes - 1, byte 7 of that lane).
    A[rate_lanes - 1u] ^= 0x8000000000000000ul;

    // Permute then squeeze; if out_lanes > rate_lanes, run additional
    // permutations and emit further chunks until done.
    uint out_base = idx * out_lanes;
    uint written = 0u;
    for (;;) {
        keccak_f1600(A);
        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        for (uint j = 0u; j < take; ++j) {
            out_data[out_base + written + j] = A[j];
        }
        written += take;
        if (written >= out_lanes) break;
    }
}
```

Incumbent result:
     sha3_256_B16K: correct, 1.39 ms, 43.9 Gbitops/s (u64) (3.9% of 1125 Gops/s (u64 bitop, est))
    sha3_256_B256K: correct, 22.01 ms, 44.3 Gbitops/s (u64) (3.9% of 1125 Gops/s (u64 bitop, est))
      sha3_256_B4M: correct, 353.45 ms, 44.1 Gbitops/s (u64) (3.9% of 1125 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.0392

## History

- iter  0: compile=OK | correct=True | score=0.03921608618417052
- iter  1: compile=FAIL | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
