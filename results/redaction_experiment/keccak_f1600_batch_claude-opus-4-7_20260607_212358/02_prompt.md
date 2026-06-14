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

inline ulong ROTL64(ulong x, uint k) {
    return (x << k) | (x >> (64u - k));
}

// One Keccak round on 25 named lanes. Lane naming: a{x}{y}.
#define KECCAK_ROUND(rc)                                                     \
    {                                                                        \
        ulong c0 = a00 ^ a01 ^ a02 ^ a03 ^ a04;                              \
        ulong c1 = a10 ^ a11 ^ a12 ^ a13 ^ a14;                              \
        ulong c2 = a20 ^ a21 ^ a22 ^ a23 ^ a24;                              \
        ulong c3 = a30 ^ a31 ^ a32 ^ a33 ^ a34;                              \
        ulong c4 = a40 ^ a41 ^ a42 ^ a43 ^ a44;                              \
        ulong d0 = c4 ^ ROTL64(c1, 1);                                       \
        ulong d1 = c0 ^ ROTL64(c2, 1);                                       \
        ulong d2 = c1 ^ ROTL64(c3, 1);                                       \
        ulong d3 = c2 ^ ROTL64(c4, 1);                                       \
        ulong d4 = c3 ^ ROTL64(c0, 1);                                       \
        /* theta + rho + pi: compute rotated lanes b{x}{y} (destination layout). */ \
        /* B[x_new + 5*y_new] = rotl(A[x + 5*y] ^ D[x], r[x][y])             */ \
        /* with (x_new, y_new) = (y, (2x+3y)%5)                              */ \
        ulong b00 = (a00 ^ d0);                                              \
        ulong b10 = ROTL64(a30 ^ d3, 28);                                    \
        ulong b20 = ROTL64(a10 ^ d1, 1);                                     \
        ulong b30 = ROTL64(a40 ^ d4, 27);                                    \
        ulong b40 = ROTL64(a20 ^ d2, 62);                                    \
        ulong b01 = ROTL64(a11 ^ d1, 44);                                    \
        ulong b11 = ROTL64(a41 ^ d4, 20);                                    \
        ulong b21 = ROTL64(a21 ^ d2, 6);                                     \
        ulong b31 = ROTL64(a01 ^ d0, 36);                                    \
        ulong b41 = ROTL64(a31 ^ d3, 55);                                    \
        ulong b02 = ROTL64(a22 ^ d2, 43);                                    \
        ulong b12 = ROTL64(a02 ^ d0, 3);                                     \
        ulong b22 = ROTL64(a32 ^ d3, 25);                                    \
        ulong b32 = ROTL64(a12 ^ d1, 10);                                    \
        ulong b42 = ROTL64(a42 ^ d4, 39);                                    \
        ulong b03 = ROTL64(a33 ^ d3, 21);                                    \
        ulong b13 = ROTL64(a13 ^ d1, 45);                                    \
        ulong b23 = ROTL64(a43 ^ d4, 8);                                     \
        ulong b33 = ROTL64(a23 ^ d2, 15);                                    \
        ulong b43 = ROTL64(a03 ^ d0, 41);                                    \
        ulong b04 = ROTL64(a44 ^ d4, 14);                                    \
        ulong b14 = ROTL64(a24 ^ d2, 61);                                    \
        ulong b24 = ROTL64(a04 ^ d0, 18);                                    \
        ulong b34 = ROTL64(a34 ^ d3, 56);                                    \
        ulong b44 = ROTL64(a14 ^ d1, 2);                                     \
        /* chi + iota */                                                     \
        a00 = b00 ^ ((~b10) & b20) ^ (rc);                                   \
        a10 = b10 ^ ((~b20) & b30);                                          \
        a20 = b20 ^ ((~b30) & b40);                                          \
        a30 = b30 ^ ((~b40) & b00);                                          \
        a40 = b40 ^ ((~b00) & b10);                                          \
        a01 = b01 ^ ((~b11) & b21);                                          \
        a11 = b11 ^ ((~b21) & b31);                                          \
        a21 = b21 ^ ((~b31) & b41);                                          \
        a31 = b31 ^ ((~b41) & b01);                                          \
        a41 = b41 ^ ((~b01) & b11);                                          \
        a02 = b02 ^ ((~b12) & b22);                                          \
        a12 = b12 ^ ((~b22) & b32);                                          \
        a22 = b22 ^ ((~b32) & b42);                                          \
        a32 = b32 ^ ((~b42) & b02);                                          \
        a42 = b42 ^ ((~b02) & b12);                                          \
        a03 = b03 ^ ((~b13) & b23);                                          \
        a13 = b13 ^ ((~b23) & b33);                                          \
        a23 = b23 ^ ((~b33) & b43);                                          \
        a33 = b33 ^ ((~b43) & b03);                                          \
        a43 = b43 ^ ((~b03) & b13);                                          \
        a04 = b04 ^ ((~b14) & b24);                                          \
        a14 = b14 ^ ((~b24) & b34);                                          \
        a24 = b24 ^ ((~b34) & b44);                                          \
        a34 = b34 ^ ((~b44) & b04);                                          \
        a44 = b44 ^ ((~b04) & b14);                                          \
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

    // Initialise 25 lanes in registers.
    ulong a00 = 0, a10 = 0, a20 = 0, a30 = 0, a40 = 0;
    ulong a01 = 0, a11 = 0, a21 = 0, a31 = 0, a41 = 0;
    ulong a02 = 0, a12 = 0, a22 = 0, a32 = 0, a42 = 0;
    ulong a03 = 0, a13 = 0, a23 = 0, a33 = 0, a43 = 0;
    ulong a04 = 0, a14 = 0, a24 = 0, a34 = 0, a44 = 0;

    // Absorb msg_lanes from in_data into lanes 0..msg_lanes-1.
    // Common test path: msg_bytes = 32 (4 lanes).
    uint in_base = idx * msg_lanes;
    device const ulong *msg = in_data + in_base;

    // Generic absorb via a small dispatch (msg_lanes is at most rate_lanes-1).
    // We use a temp array small enough to live in registers/threadlocal.
    {
        ulong tmp[25];
        for (uint i = 0u; i < 25u; ++i) tmp[i] = 0ul;
        for (uint i = 0u; i < msg_lanes; ++i) tmp[i] = msg[i];
        tmp[msg_lanes]     ^= (ulong)(domain & 0xFFu);
        tmp[rate_lanes - 1u] ^= 0x8000000000000000ul;
        a00 = tmp[0];  a10 = tmp[1];  a20 = tmp[2];  a30 = tmp[3];  a40 = tmp[4];
        a01 = tmp[5];  a11 = tmp[6];  a21 = tmp[7];  a31 = tmp[8];  a41 = tmp[9];
        a02 = tmp[10]; a12 = tmp[11]; a22 = tmp[12]; a32 = tmp[13]; a42 = tmp[14];
        a03 = tmp[15]; a13 = tmp[16]; a23 = tmp[17]; a33 = tmp[18]; a43 = tmp[19];
        a04 = tmp[20]; a14 = tmp[21]; a24 = tmp[22]; a34 = tmp[23]; a44 = tmp[24];
    }

    uint out_base = idx * out_lanes;
    uint written = 0u;

    for (;;) {
        // 24 unrolled rounds.
        KECCAK_ROUND(KECCAK_RC[0])
        KECCAK_ROUND(KECCAK_RC[1])
        KECCAK_ROUND(KECCAK_RC[2])
        KECCAK_ROUND(KECCAK_RC[3])
        KECCAK_ROUND(KECCAK_RC[4])
        KECCAK_ROUND(KECCAK_RC[5])
        KECCAK_ROUND(KECCAK_RC[6])
        KECCAK_ROUND(KECCAK_RC[7])
        KECCAK_ROUND(KECCAK_RC[8])
        KECCAK_ROUND(KECCAK_RC[9])
        KECCAK_ROUND(KECCAK_RC[10])
        KECCAK_ROUND(KECCAK_RC[11])
        KECCAK_ROUND(KECCAK_RC[12])
        KECCAK_ROUND(KECCAK_RC[13])
        KECCAK_ROUND(KECCAK_RC[14])
        KECCAK_ROUND(KECCAK_RC[15])
        KECCAK_ROUND(KECCAK_RC[16])
        KECCAK_ROUND(KECCAK_RC[17])
        KECCAK_ROUND(KECCAK_RC[18])
        KECCAK_ROUND(KECCAK_RC[19])
        KECCAK_ROUND(KECCAK_RC[20])
        KECCAK_ROUND(KECCAK_RC[21])
        KECCAK_ROUND(KECCAK_RC[22])
        KECCAK_ROUND(KECCAK_RC[23])

        // Squeeze up to rate_lanes lanes (or fewer for last chunk).
        ulong squeezed[25];
        squeezed[0]  = a00; squeezed[1]  = a10; squeezed[2]  = a20; squeezed[3]  = a30; squeezed[4]  = a40;
        squeezed[5]  = a01; squeezed[6]  = a11; squeezed[7]  = a21; squeezed[8]  = a31; squeezed[9]  = a41;
        squeezed[10] = a02; squeezed[11] = a12; squeezed[12] = a22; squeezed[13] = a32; squeezed[14] = a42;
        squeezed[15] = a03; squeezed[16] = a13; squeezed[17] = a23; squeezed[18] = a33; squeezed[19] = a43;
        squeezed[20] = a04; squeezed[21] = a14; squeezed[22] = a24; squeezed[23] = a34; squeezed[24] = a44;

        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        for (uint j = 0u; j < take; ++j) {
            out_data[out_base + written + j] = squeezed[j];
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
     sha3_256_B16K: correct, 1.40 ms, 43.5 Gbitops/s (u64) (3.9% of 1125 Gops/s (u64 bitop, est))
    sha3_256_B256K: correct, 23.80 ms, 41.0 Gbitops/s (u64) (3.6% of 1125 Gops/s (u64 bitop, est))
      sha3_256_B4M: correct, 354.39 ms, 44.0 Gbitops/s (u64) (3.9% of 1125 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.0381

## History

- iter  0: compile=OK | correct=True | score=0.03807011258151357
- iter  1: compile=OK | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
