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

#define ROTL64(x, k) (((x) << (k)) | ((x) >> (64 - (k))))

// One round operating on 25 named lane registers a##x##y (x = col, y = row).
// Pi mapping verified:
//   B[X,Y] = rotl(A[src_x, src_y], rho[src_x][src_y])
//   src_x = (X + 3Y) mod 5, src_y = X
// rho table (indexed [x][y]):
//   x=0: { 0,36, 3,41,18}
//   x=1: { 1,44,10,45, 2}
//   x=2: {62, 6,43,15,61}
//   x=3: {28,55,25,21,56}
//   x=4: {27,20,39, 8,14}
#define KECCAK_ROUND(rc) {                                                   \
    /* theta */                                                              \
    ulong _c0 = a00 ^ a01 ^ a02 ^ a03 ^ a04;                                 \
    ulong _c1 = a10 ^ a11 ^ a12 ^ a13 ^ a14;                                 \
    ulong _c2 = a20 ^ a21 ^ a22 ^ a23 ^ a24;                                 \
    ulong _c3 = a30 ^ a31 ^ a32 ^ a33 ^ a34;                                 \
    ulong _c4 = a40 ^ a41 ^ a42 ^ a43 ^ a44;                                 \
    ulong _d0 = _c4 ^ ROTL64(_c1, 1);                                        \
    ulong _d1 = _c0 ^ ROTL64(_c2, 1);                                        \
    ulong _d2 = _c1 ^ ROTL64(_c3, 1);                                        \
    ulong _d3 = _c2 ^ ROTL64(_c4, 1);                                        \
    ulong _d4 = _c3 ^ ROTL64(_c0, 1);                                        \
    ulong t00 = a00 ^ _d0, t10 = a10 ^ _d1, t20 = a20 ^ _d2,                 \
          t30 = a30 ^ _d3, t40 = a40 ^ _d4;                                  \
    ulong t01 = a01 ^ _d0, t11 = a11 ^ _d1, t21 = a21 ^ _d2,                 \
          t31 = a31 ^ _d3, t41 = a41 ^ _d4;                                  \
    ulong t02 = a02 ^ _d0, t12 = a12 ^ _d1, t22 = a22 ^ _d2,                 \
          t32 = a32 ^ _d3, t42 = a42 ^ _d4;                                  \
    ulong t03 = a03 ^ _d0, t13 = a13 ^ _d1, t23 = a23 ^ _d2,                 \
          t33 = a33 ^ _d3, t43 = a43 ^ _d4;                                  \
    ulong t04 = a04 ^ _d0, t14 = a14 ^ _d1, t24 = a24 ^ _d2,                 \
          t34 = a34 ^ _d3, t44 = a44 ^ _d4;                                  \
    /* rho + pi: B[X][Y] = rotl(t[src_x][src_y], rho[src_x][src_y]) */       \
    /* Y=0: src_x=X, src_y=0 -> rho[X][0] */                                 \
    ulong b00 = t00;                                                         \
    ulong b10 = ROTL64(t10,  1);                                             \
    ulong b20 = ROTL64(t20, 62);                                             \
    ulong b30 = ROTL64(t30, 28);                                             \
    ulong b40 = ROTL64(t40, 27);                                             \
    /* Y=1: src_x=(X+3)%5, src_y=X */                                        \
    ulong b01 = ROTL64(t30, 28);                                             \
    ulong b11 = ROTL64(t41, 20);                                             \
    ulong b21 = ROTL64(t02,  3);                                             \
    ulong b31 = ROTL64(t13, 45);                                             \
    ulong b41 = ROTL64(t24, 61);                                             \
    /* Y=2: src_x=(X+1)%5, src_y=X */                                        \
    ulong b02 = ROTL64(t10,  1);                                             \
    ulong b12 = ROTL64(t21,  6);                                             \
    ulong b22 = ROTL64(t32, 25);                                             \
    ulong b32 = ROTL64(t43,  8);                                             \
    ulong b42 = ROTL64(t04, 18);                                             \
    /* Y=3: src_x=(X+4)%5, src_y=X */                                        \
    ulong b03 = ROTL64(t40, 27);                                             \
    ulong b13 = ROTL64(t01, 36);                                             \
    ulong b23 = ROTL64(t12, 10);                                             \
    ulong b33 = ROTL64(t23, 15);                                             \
    ulong b43 = ROTL64(t34, 56);                                             \
    /* Y=4: src_x=(X+2)%5, src_y=X */                                        \
    ulong b04 = ROTL64(t20, 62);                                             \
    ulong b14 = ROTL64(t31, 55);                                             \
    ulong b24 = ROTL64(t42, 39);                                             \
    ulong b34 = ROTL64(t03, 41);                                             \
    ulong b44 = ROTL64(t14,  2);                                             \
    /* chi + iota */                                                         \
    a00 = b00 ^ ((~b10) & b20) ^ (rc);                                       \
    a10 = b10 ^ ((~b20) & b30);                                              \
    a20 = b20 ^ ((~b30) & b40);                                              \
    a30 = b30 ^ ((~b40) & b00);                                              \
    a40 = b40 ^ ((~b00) & b10);                                              \
    a01 = b01 ^ ((~b11) & b21);                                              \
    a11 = b11 ^ ((~b21) & b31);                                              \
    a21 = b21 ^ ((~b31) & b41);                                              \
    a31 = b31 ^ ((~b41) & b01);                                              \
    a41 = b41 ^ ((~b01) & b11);                                              \
    a02 = b02 ^ ((~b12) & b22);                                              \
    a12 = b12 ^ ((~b22) & b32);                                              \
    a22 = b22 ^ ((~b32) & b42);                                              \
    a32 = b32 ^ ((~b42) & b02);                                              \
    a42 = b42 ^ ((~b02) & b12);                                              \
    a03 = b03 ^ ((~b13) & b23);                                              \
    a13 = b13 ^ ((~b23) & b33);                                              \
    a23 = b23 ^ ((~b33) & b43);                                              \
    a33 = b33 ^ ((~b43) & b03);                                              \
    a43 = b43 ^ ((~b03) & b13);                                              \
    a04 = b04 ^ ((~b14) & b24);                                              \
    a14 = b14 ^ ((~b24) & b34);                                              \
    a24 = b24 ^ ((~b34) & b44);                                              \
    a34 = b34 ^ ((~b44) & b04);                                              \
    a44 = b44 ^ ((~b04) & b14);                                              \
}

#define KECCAK_PERMUTE() {                  \
    KECCAK_ROUND(KECCAK_RC[0])              \
    KECCAK_ROUND(KECCAK_RC[1])              \
    KECCAK_ROUND(KECCAK_RC[2])              \
    KECCAK_ROUND(KECCAK_RC[3])              \
    KECCAK_ROUND(KECCAK_RC[4])              \
    KECCAK_ROUND(KECCAK_RC[5])              \
    KECCAK_ROUND(KECCAK_RC[6])              \
    KECCAK_ROUND(KECCAK_RC[7])              \
    KECCAK_ROUND(KECCAK_RC[8])              \
    KECCAK_ROUND(KECCAK_RC[9])              \
    KECCAK_ROUND(KECCAK_RC[10])             \
    KECCAK_ROUND(KECCAK_RC[11])             \
    KECCAK_ROUND(KECCAK_RC[12])             \
    KECCAK_ROUND(KECCAK_RC[13])             \
    KECCAK_ROUND(KECCAK_RC[14])             \
    KECCAK_ROUND(KECCAK_RC[15])             \
    KECCAK_ROUND(KECCAK_RC[16])             \
    KECCAK_ROUND(KECCAK_RC[17])             \
    KECCAK_ROUND(KECCAK_RC[18])             \
    KECCAK_ROUND(KECCAK_RC[19])             \
    KECCAK_ROUND(KECCAK_RC[20])             \
    KECCAK_ROUND(KECCAK_RC[21])             \
    KECCAK_ROUND(KECCAK_RC[22])             \
    KECCAK_ROUND(KECCAK_RC[23])             \
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

    // Lane k = x + 5*y -> aXY register. Initialize to 0 then absorb.
    ulong a00=0, a10=0, a20=0, a30=0, a40=0;
    ulong a01=0, a11=0, a21=0, a31=0, a41=0;
    ulong a02=0, a12=0, a22=0, a32=0, a42=0;
    ulong a03=0, a13=0, a23=0, a33=0, a43=0;
    ulong a04=0, a14=0, a24=0, a34=0, a44=0;

    // Absorb via a small array indexed by lane, then unpack to registers.
    uint in_base = idx * msg_lanes;
    ulong s[25];
    for (uint i = 0u; i < 25u; ++i) s[i] = 0ul;
    for (uint i = 0u; i < msg_lanes; ++i) s[i] = in_data[in_base + i];
    s[msg_lanes]        ^= (ulong)(domain & 0xFFu);
    s[rate_lanes - 1u]  ^= 0x8000000000000000ul;

    a00 = s[ 0]; a10 = s[ 1]; a20 = s[ 2]; a30 = s[ 3]; a40 = s[ 4];
    a01 = s[ 5]; a11 = s[ 6]; a21 = s[ 7]; a31 = s[ 8]; a41 = s[ 9];
    a02 = s[10]; a12 = s[11]; a22 = s[12]; a32 = s[13]; a42 = s[14];
    a03 = s[15]; a13 = s[16]; a23 = s[17]; a33 = s[18]; a43 = s[19];
    a04 = s[20]; a14 = s[21]; a24 = s[22]; a34 = s[23]; a44 = s[24];

    uint out_base = idx * out_lanes;
    uint written = 0u;

    for (;;) {
        KECCAK_PERMUTE()

        // Pack current state lanes (in lane-index order) for emission.
        ulong out_buf[25];
        out_buf[ 0]=a00; out_buf[ 1]=a10; out_buf[ 2]=a20; out_buf[ 3]=a30; out_buf[ 4]=a40;
        out_buf[ 5]=a01; out_buf[ 6]=a11; out_buf[ 7]=a21; out_buf[ 8]=a31; out_buf[ 9]=a41;
        out_buf[10]=a02; out_buf[11]=a12; out_buf[12]=a22; out_buf[13]=a32; out_buf[14]=a42;
        out_buf[15]=a03; out_buf[16]=a13; out_buf[17]=a23; out_buf[18]=a33; out_buf[19]=a43;
        out_buf[20]=a04; out_buf[21]=a14; out_buf[22]=a24; out_buf[23]=a34; out_buf[24]=a44;

        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        for (uint j = 0u; j < take; ++j) {
            out_data[out_base + written + j] = out_buf[j];
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
- iter  2: compile=OK | correct=False | score=N/A
- iter  3: compile=OK | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
