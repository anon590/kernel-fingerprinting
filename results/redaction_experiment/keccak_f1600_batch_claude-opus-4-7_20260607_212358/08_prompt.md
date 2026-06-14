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

// Lane-complemented round constants: RC[i] for lane (0,0), which is kept
// in normal form. (Standard RC values.)
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

#define ROTL(x,k) (((x) << (k)) | ((x) >> (64 - (k))))

// Lane complementing: lanes that are stored INVERTED throughout the
// permutation are (using (x,y) coords): (1,0), (2,0), (3,1), (2,2), (2,3), (0,4).
// In our register naming a{x}{y}:
//   a10, a20, a31, a22, a23, a04   are complemented.
//
// Chi step formula rewrites. Standard chi:
//   a[x,y] = B[x,y] ^ ((~B[(x+1)%5,y]) & B[(x+2)%5,y])
//
// For each row y, considering which of B[0..4,y] are inverted (denote ' for inverted):
//
// Row y=0:  inverted: B10', B20'
//   a00 = B00 ^ (~B10  & B20 )   -> B00 ^ ( B10' & B20')   [both inv]
//   a10 = B10 ^ (~B20  & B30 )   -> want a10 inverted (output).
//         Output (true) = B10' ^ (~B20'  & B30) = B10' ^ (B20 & B30); we want stored = ~Output = B10 ^ (B20 & B30)
//         Equivalently: store = B10' ^ (B20'|~B30)? Let's derive cleanly below.
//   We use identity: for chi pieces where the operand is inverted, use:
//      ~(~x) = x. (~Ba & Bb) with Ba stored inverted Ba' = ~Ba: ~Ba = Ba', so ~Ba & Bb = Ba' & Bb.
//      (~Ba & Bb) with Bb stored inverted Bb' = ~Bb: Bb = ~Bb', so ~Ba & ~Bb' = ~(Ba | Bb').
//      (~Ba & Bb) with both inverted: ~Ba & ~Bb' = Ba' & ~Bb' = ~(~Ba' | Bb') = ~(Ba | Bb').
//   For outputs where the destination is inverted-stored, we XOR with ~mask instead of mask
//   (since storing ~result = B' ^ mask where result = B ^ ~mask... carefully: stored = ~(B ^ m) = ~B ^ m
//   but B may itself be stored inverted. We'll work it out per-lane.)
//
// Rather than derive symbolically here, I use the well-known Bertoni table directly
// (see Keccak reference code, "lane complementing" optimization). The transformation
// converts 5 of the 25 chi instances per round from (B ^ (~X & Y)) to (B ^ (X | Y)).
//
// Inverted set I = { a[1,0], a[2,0], a[3,1], a[2,2], a[2,3], a[0,4] }.
// Per-row rewrites (verified against Keccak ref impl):
//   y=0:
//     a00 = B00 ^ ( B10 |  B20)    [B10,B20 are stored inv -> formula uses OR]
//     a10 = B10 ^ ( B20 &  B30)    [B10 stays inv-stored]
//     a20 = B20 ^ (~B30 | ~B40) -> = B20 ^ ~(B30 & B40)? - we keep simple form below
//     a30 = B30 ^ (~B40 &  B00)
//     a40 = B40 ^ (~B00 &  B10)    [B10 inv-stored]
//
// This is getting error-prone to write in a comment; instead, we derive each row's
// 5 expressions by computer-checked substitution and inline them.

// To make this provably correct, I implement chi WITHOUT lane complementing.
// Instead, the optimization I apply is: keep the heavily-tuned round, but inline
// absorb directly into registers (no S[] staging array), and inline the squeeze
// for the common SHA3-256 case (out_lanes == 4) to avoid the OUT[25] spill.

#define KECCAK_ROUND(RC) {                                            \
    ulong C0 = a00 ^ a01 ^ a02 ^ a03 ^ a04;                           \
    ulong C1 = a10 ^ a11 ^ a12 ^ a13 ^ a14;                           \
    ulong C2 = a20 ^ a21 ^ a22 ^ a23 ^ a24;                           \
    ulong C3 = a30 ^ a31 ^ a32 ^ a33 ^ a34;                           \
    ulong C4 = a40 ^ a41 ^ a42 ^ a43 ^ a44;                           \
    ulong D0 = C4 ^ ROTL(C1, 1);                                      \
    ulong D1 = C0 ^ ROTL(C2, 1);                                      \
    ulong D2 = C1 ^ ROTL(C3, 1);                                      \
    ulong D3 = C2 ^ ROTL(C4, 1);                                      \
    ulong D4 = C3 ^ ROTL(C0, 1);                                      \
    a00 ^= D0; a10 ^= D1; a20 ^= D2; a30 ^= D3; a40 ^= D4;            \
    a01 ^= D0; a11 ^= D1; a21 ^= D2; a31 ^= D3; a41 ^= D4;            \
    a02 ^= D0; a12 ^= D1; a22 ^= D2; a32 ^= D3; a42 ^= D4;            \
    a03 ^= D0; a13 ^= D1; a23 ^= D2; a33 ^= D3; a43 ^= D4;            \
    a04 ^= D0; a14 ^= D1; a24 ^= D2; a34 ^= D3; a44 ^= D4;            \
    ulong B00 = a00;                                                  \
    ulong B02 = ROTL(a10, 1);                                         \
    ulong B04 = ROTL(a20, 62);                                        \
    ulong B01 = ROTL(a30, 28);                                        \
    ulong B03 = ROTL(a40, 27);                                        \
    ulong B13 = ROTL(a01, 36);                                        \
    ulong B10 = ROTL(a11, 44);                                        \
    ulong B12 = ROTL(a21, 6);                                         \
    ulong B14 = ROTL(a31, 55);                                        \
    ulong B11 = ROTL(a41, 20);                                        \
    ulong B21 = ROTL(a02, 3);                                         \
    ulong B23 = ROTL(a12, 10);                                        \
    ulong B20 = ROTL(a22, 43);                                        \
    ulong B22 = ROTL(a32, 25);                                        \
    ulong B24 = ROTL(a42, 39);                                        \
    ulong B34 = ROTL(a03, 41);                                        \
    ulong B31 = ROTL(a13, 45);                                        \
    ulong B33 = ROTL(a23, 15);                                        \
    ulong B30 = ROTL(a33, 21);                                        \
    ulong B32 = ROTL(a43, 8);                                         \
    ulong B42 = ROTL(a04, 18);                                        \
    ulong B44 = ROTL(a14, 2);                                         \
    ulong B41 = ROTL(a24, 61);                                        \
    ulong B43 = ROTL(a34, 56);                                        \
    ulong B40 = ROTL(a44, 14);                                        \
    a00 = B00 ^ ((~B10) & B20) ^ (RC);                                \
    a10 = B10 ^ ((~B20) & B30);                                       \
    a20 = B20 ^ ((~B30) & B40);                                       \
    a30 = B30 ^ ((~B40) & B00);                                       \
    a40 = B40 ^ ((~B00) & B10);                                       \
    a01 = B01 ^ ((~B11) & B21);                                       \
    a11 = B11 ^ ((~B21) & B31);                                       \
    a21 = B21 ^ ((~B31) & B41);                                       \
    a31 = B31 ^ ((~B41) & B01);                                       \
    a41 = B41 ^ ((~B01) & B11);                                       \
    a02 = B02 ^ ((~B12) & B22);                                       \
    a12 = B12 ^ ((~B22) & B32);                                       \
    a22 = B22 ^ ((~B32) & B42);                                       \
    a32 = B32 ^ ((~B42) & B02);                                       \
    a42 = B42 ^ ((~B02) & B12);                                       \
    a03 = B03 ^ ((~B13) & B23);                                       \
    a13 = B13 ^ ((~B23) & B33);                                       \
    a23 = B23 ^ ((~B33) & B43);                                       \
    a33 = B33 ^ ((~B43) & B03);                                       \
    a43 = B43 ^ ((~B03) & B13);                                       \
    a04 = B04 ^ ((~B14) & B24);                                       \
    a14 = B14 ^ ((~B24) & B34);                                       \
    a24 = B24 ^ ((~B34) & B44);                                       \
    a34 = B34 ^ ((~B44) & B04);                                       \
    a44 = B44 ^ ((~B04) & B14);                                       \
}

#define KECCAK_PERMUTE() do {              \
    KECCAK_ROUND(KECCAK_RC[ 0])            \
    KECCAK_ROUND(KECCAK_RC[ 1])            \
    KECCAK_ROUND(KECCAK_RC[ 2])            \
    KECCAK_ROUND(KECCAK_RC[ 3])            \
    KECCAK_ROUND(KECCAK_RC[ 4])            \
    KECCAK_ROUND(KECCAK_RC[ 5])            \
    KECCAK_ROUND(KECCAK_RC[ 6])            \
    KECCAK_ROUND(KECCAK_RC[ 7])            \
    KECCAK_ROUND(KECCAK_RC[ 8])            \
    KECCAK_ROUND(KECCAK_RC[ 9])            \
    KECCAK_ROUND(KECCAK_RC[10])            \
    KECCAK_ROUND(KECCAK_RC[11])            \
    KECCAK_ROUND(KECCAK_RC[12])            \
    KECCAK_ROUND(KECCAK_RC[13])            \
    KECCAK_ROUND(KECCAK_RC[14])            \
    KECCAK_ROUND(KECCAK_RC[15])            \
    KECCAK_ROUND(KECCAK_RC[16])            \
    KECCAK_ROUND(KECCAK_RC[17])            \
    KECCAK_ROUND(KECCAK_RC[18])            \
    KECCAK_ROUND(KECCAK_RC[19])            \
    KECCAK_ROUND(KECCAK_RC[20])            \
    KECCAK_ROUND(KECCAK_RC[21])            \
    KECCAK_ROUND(KECCAK_RC[22])            \
    KECCAK_ROUND(KECCAK_RC[23])            \
} while (0)

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

    ulong a00=0, a10=0, a20=0, a30=0, a40=0;
    ulong a01=0, a11=0, a21=0, a31=0, a41=0;
    ulong a02=0, a12=0, a22=0, a32=0, a42=0;
    ulong a03=0, a13=0, a23=0, a33=0, a43=0;
    ulong a04=0, a14=0, a24=0, a34=0, a44=0;

    uint in_base = idx * msg_lanes;
    device const ulong *in_ptr = in_data + in_base;

    // Fast path: msg_lanes == 4 (SHA3-256 test sizes: msg_bytes = 32).
    if (msg_lanes == 4u) {
        a00 = in_ptr[0];
        a10 = in_ptr[1];
        a20 = in_ptr[2];
        a30 = in_ptr[3];
        a40 = (ulong)(domain & 0xFFu);          // lane 4 = a[4,0] = a40, byte 0
        // 0x80 at rate_bytes-1: lane rate_lanes-1, byte 7.
        // For SHA3-256, rate_lanes=17 -> lane 16 = a[1,3] = a13.
        // For SHAKE128, rate_lanes=21 -> lane 20 = a[0,4] = a04.
        // General path needed for held-out (SHAKE) case (msg_lanes still 4? Yes spec
        // says all test sizes have msg_bytes = 32, so this fast path always applies).
        uint plast = rate_lanes - 1u;
        ulong pad_hi = 0x8000000000000000ul;
        // Branch sparingly using switch on plast.
        switch (plast) {
            case 16: a13 ^= pad_hi; break;  // SHA3-256
            case 20: a04 ^= pad_hi; break;  // SHAKE128
            case  0: a00 ^= pad_hi; break;
            case  1: a10 ^= pad_hi; break;
            case  2: a20 ^= pad_hi; break;
            case  3: a30 ^= pad_hi; break;
            case  4: a40 ^= pad_hi; break;
            case  5: a01 ^= pad_hi; break;
            case  6: a11 ^= pad_hi; break;
            case  7: a21 ^= pad_hi; break;
            case  8: a31 ^= pad_hi; break;
            case  9: a41 ^= pad_hi; break;
            case 10: a02 ^= pad_hi; break;
            case 11: a12 ^= pad_hi; break;
            case 12: a22 ^= pad_hi; break;
            case 13: a32 ^= pad_hi; break;
            case 14: a42 ^= pad_hi; break;
            case 15: a03 ^= pad_hi; break;
            case 17: a23 ^= pad_hi; break;
            case 18: a33 ^= pad_hi; break;
            case 19: a43 ^= pad_hi; break;
            case 21: a14 ^= pad_hi; break;
            case 22: a24 ^= pad_hi; break;
            case 23: a34 ^= pad_hi; break;
            case 24: a44 ^= pad_hi; break;
            default: break;
        }
    } else {
        // Generic absorb path (not used by current test sizes, but kept for safety).
        ulong S[25];
        for (uint i = 0u; i < 25u; ++i) S[i] = 0ul;
        for (uint i = 0u; i < msg_lanes; ++i) S[i] = in_ptr[i];
        S[msg_lanes]       ^= (ulong)(domain & 0xFFu);
        S[rate_lanes - 1u] ^= 0x8000000000000000ul;
        a00 = S[ 0]; a10 = S[ 1]; a20 = S[ 2]; a30 = S[ 3]; a40 = S[ 4];
        a01 = S[ 5]; a11 = S[ 6]; a21 = S[ 7]; a31 = S[ 8]; a41 = S[ 9];
        a02 = S[10]; a12 = S[11]; a22 = S[12]; a32 = S[13]; a42 = S[14];
        a03 = S[15]; a13 = S[16]; a23 = S[17]; a33 = S[18]; a43 = S[19];
        a04 = S[20]; a14 = S[21]; a24 = S[22]; a34 = S[23]; a44 = S[24];
    }

    uint out_base = idx * out_lanes;
    device ulong *op = out_data + out_base;

    // Fast path: SHA3-256 single squeeze, out_lanes == 4.
    if (out_lanes == 4u) {
        KECCAK_PERMUTE();
        op[0] = a00;
        op[1] = a10;
        op[2] = a20;
        op[3] = a30;
        return;
    }

    // Single-squeeze path when output fits in one block.
    if (out_lanes <= rate_lanes) {
        KECCAK_PERMUTE();
        ulong OUT[25];
        OUT[ 0]=a00; OUT[ 1]=a10; OUT[ 2]=a20; OUT[ 3]=a30; OUT[ 4]=a40;
        OUT[ 5]=a01; OUT[ 6]=a11; OUT[ 7]=a21; OUT[ 8]=a31; OUT[ 9]=a41;
        OUT[10]=a02; OUT[11]=a12; OUT[12]=a22; OUT[13]=a32; OUT[14]=a42;
        OUT[15]=a03; OUT[16]=a13; OUT[17]=a23; OUT[18]=a33; OUT[19]=a43;
        OUT[20]=a04; OUT[21]=a14; OUT[22]=a24; OUT[23]=a34; OUT[24]=a44;
        for (uint j = 0u; j < out_lanes; ++j) op[j] = OUT[j];
        return;
    }

    // Multi-squeeze (SHAKE128 held-out case).
    uint written = 0u;
    for (;;) {
        KECCAK_PERMUTE();
        ulong OUT[25];
        OUT[ 0]=a00; OUT[ 1]=a10; OUT[ 2]=a20; OUT[ 3]=a30; OUT[ 4]=a40;
        OUT[ 5]=a01; OUT[ 6]=a11; OUT[ 7]=a21; OUT[ 8]=a31; OUT[ 9]=a41;
        OUT[10]=a02; OUT[11]=a12; OUT[12]=a22; OUT[13]=a32; OUT[14]=a42;
        OUT[15]=a03; OUT[16]=a13; OUT[17]=a23; OUT[18]=a33; OUT[19]=a43;
        OUT[20]=a04; OUT[21]=a14; OUT[22]=a24; OUT[23]=a34; OUT[24]=a44;

        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        for (uint j = 0u; j < take; ++j) {
            op[written + j] = OUT[j];
        }
        written += take;
        if (written >= out_lanes) break;
    }
}
```

Result of previous attempt:
     sha3_256_B16K: correct, 0.43 ms, 142.8 Gbitops/s (u64) (12.7% of 1125 Gops/s (u64 bitop, est))
    sha3_256_B256K: correct, 1.58 ms, 618.5 Gbitops/s (u64) (55.0% of 1125 Gops/s (u64 bitop, est))
      sha3_256_B4M: correct, 22.51 ms, 693.0 Gbitops/s (u64) (61.6% of 1125 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.3503

## History

- iter  0: compile=OK | correct=True | score=0.03807011258151357
- iter  1: compile=OK | correct=False | score=N/A
- iter  2: compile=OK | correct=False | score=N/A
- iter  3: compile=OK | correct=False | score=N/A
- iter  4: compile=OK | correct=False | score=N/A
- iter  5: compile=OK | correct=True | score=0.33718968173758335
- iter  6: compile=OK | correct=True | score=0.32103969653022396
- iter  7: compile=OK | correct=True | score=0.35034610412022826

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
