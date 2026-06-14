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

#define KECCAK_ROUND(RCV)                                            \
{                                                                    \
    ulong C0 = a0 ^ a5 ^ a10 ^ a15 ^ a20;                            \
    ulong C1 = a1 ^ a6 ^ a11 ^ a16 ^ a21;                            \
    ulong C2 = a2 ^ a7 ^ a12 ^ a17 ^ a22;                            \
    ulong C3 = a3 ^ a8 ^ a13 ^ a18 ^ a23;                            \
    ulong C4 = a4 ^ a9 ^ a14 ^ a19 ^ a24;                            \
    ulong D0 = C4 ^ ROTL64(C1, 1);                                   \
    ulong D1 = C0 ^ ROTL64(C2, 1);                                   \
    ulong D2 = C1 ^ ROTL64(C3, 1);                                   \
    ulong D3 = C2 ^ ROTL64(C4, 1);                                   \
    ulong D4 = C3 ^ ROTL64(C0, 1);                                   \
    ulong b00 =        (a0  ^ D0)      ;                             \
    ulong b02 = ROTL64(a1  ^ D1,  1);                                \
    ulong b04 = ROTL64(a2  ^ D2, 62);                                \
    ulong b01 = ROTL64(a3  ^ D3, 28);                                \
    ulong b03 = ROTL64(a4  ^ D4, 27);                                \
    ulong b13 = ROTL64(a5  ^ D0, 36);                                \
    ulong b10 = ROTL64(a6  ^ D1, 44);                                \
    ulong b12 = ROTL64(a7  ^ D2,  6);                                \
    ulong b14 = ROTL64(a8  ^ D3, 55);                                \
    ulong b11 = ROTL64(a9  ^ D4, 20);                                \
    ulong b21 = ROTL64(a10 ^ D0,  3);                                \
    ulong b23 = ROTL64(a11 ^ D1, 10);                                \
    ulong b20 = ROTL64(a12 ^ D2, 43);                                \
    ulong b22 = ROTL64(a13 ^ D3, 25);                                \
    ulong b24 = ROTL64(a14 ^ D4, 39);                                \
    ulong b34 = ROTL64(a15 ^ D0, 41);                                \
    ulong b31 = ROTL64(a16 ^ D1, 45);                                \
    ulong b33 = ROTL64(a17 ^ D2, 15);                                \
    ulong b30 = ROTL64(a18 ^ D3, 21);                                \
    ulong b32 = ROTL64(a19 ^ D4,  8);                                \
    ulong b42 = ROTL64(a20 ^ D0, 18);                                \
    ulong b44 = ROTL64(a21 ^ D1,  2);                                \
    ulong b41 = ROTL64(a22 ^ D2, 61);                                \
    ulong b43 = ROTL64(a23 ^ D3, 56);                                \
    ulong b40 = ROTL64(a24 ^ D4, 14);                                \
    a0  = b00 ^ ((~b10) & b20) ^ (RCV);                              \
    a1  = b10 ^ ((~b20) & b30);                                      \
    a2  = b20 ^ ((~b30) & b40);                                      \
    a3  = b30 ^ ((~b40) & b00);                                      \
    a4  = b40 ^ ((~b00) & b10);                                      \
    a5  = b01 ^ ((~b11) & b21);                                      \
    a6  = b11 ^ ((~b21) & b31);                                      \
    a7  = b21 ^ ((~b31) & b41);                                      \
    a8  = b31 ^ ((~b41) & b01);                                      \
    a9  = b41 ^ ((~b01) & b11);                                      \
    a10 = b02 ^ ((~b12) & b22);                                      \
    a11 = b12 ^ ((~b22) & b32);                                      \
    a12 = b22 ^ ((~b32) & b42);                                      \
    a13 = b32 ^ ((~b42) & b02);                                      \
    a14 = b42 ^ ((~b02) & b12);                                      \
    a15 = b03 ^ ((~b13) & b23);                                      \
    a16 = b13 ^ ((~b23) & b33);                                      \
    a17 = b23 ^ ((~b33) & b43);                                      \
    a18 = b33 ^ ((~b43) & b03);                                      \
    a19 = b43 ^ ((~b03) & b13);                                      \
    a20 = b04 ^ ((~b14) & b24);                                      \
    a21 = b14 ^ ((~b24) & b34);                                      \
    a22 = b24 ^ ((~b34) & b44);                                      \
    a23 = b34 ^ ((~b44) & b04);                                      \
    a24 = b44 ^ ((~b04) & b14);                                      \
}

static inline void keccak_f1600_state(thread ulong *a) {
    ulong a0=a[0],a1=a[1],a2=a[2],a3=a[3],a4=a[4];
    ulong a5=a[5],a6=a[6],a7=a[7],a8=a[8],a9=a[9];
    ulong a10=a[10],a11=a[11],a12=a[12],a13=a[13],a14=a[14];
    ulong a15=a[15],a16=a[16],a17=a[17],a18=a[18],a19=a[19];
    ulong a20=a[20],a21=a[21],a22=a[22],a23=a[23],a24=a[24];

    KECCAK_ROUND(0x0000000000000001ul);
    KECCAK_ROUND(0x0000000000008082ul);
    KECCAK_ROUND(0x800000000000808Aul);
    KECCAK_ROUND(0x8000000080008000ul);
    KECCAK_ROUND(0x000000000000808Bul);
    KECCAK_ROUND(0x0000000080000001ul);
    KECCAK_ROUND(0x8000000080008081ul);
    KECCAK_ROUND(0x8000000000008009ul);
    KECCAK_ROUND(0x000000000000008Aul);
    KECCAK_ROUND(0x0000000000000088ul);
    KECCAK_ROUND(0x0000000080008009ul);
    KECCAK_ROUND(0x000000008000000Aul);
    KECCAK_ROUND(0x000000008000808Bul);
    KECCAK_ROUND(0x800000000000008Bul);
    KECCAK_ROUND(0x8000000000008089ul);
    KECCAK_ROUND(0x8000000000008003ul);
    KECCAK_ROUND(0x8000000000008002ul);
    KECCAK_ROUND(0x8000000000000080ul);
    KECCAK_ROUND(0x000000000000800Aul);
    KECCAK_ROUND(0x800000008000000Aul);
    KECCAK_ROUND(0x8000000080008081ul);
    KECCAK_ROUND(0x8000000000008080ul);
    KECCAK_ROUND(0x0000000080000001ul);
    KECCAK_ROUND(0x8000000080008008ul);

    a[0]=a0;a[1]=a1;a[2]=a2;a[3]=a3;a[4]=a4;
    a[5]=a5;a[6]=a6;a[7]=a7;a[8]=a8;a[9]=a9;
    a[10]=a10;a[11]=a11;a[12]=a12;a[13]=a13;a[14]=a14;
    a[15]=a15;a[16]=a16;a[17]=a17;a[18]=a18;a[19]=a19;
    a[20]=a20;a[21]=a21;a[22]=a22;a[23]=a23;a[24]=a24;
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

    ulong a[25];
    #pragma unroll
    for (uint i = 0; i < 25; ++i) a[i] = 0ul;

    uint in_base = idx * msg_lanes;
    for (uint i = 0; i < msg_lanes; ++i) a[i] = in_data[in_base + i];
    a[msg_lanes]       ^= (ulong)(domain & 0xFFu);
    a[rate_lanes - 1u] ^= 0x8000000000000000ul;

    uint out_base = idx * out_lanes;

    keccak_f1600_state(a);

    if (out_lanes <= rate_lanes) {
        for (uint j = 0; j < out_lanes; ++j) {
            out_data[out_base + j] = a[j];
        }
        return;
    }

    uint written = 0u;
    for (;;) {
        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        for (uint j = 0; j < take; ++j) {
            out_data[out_base + written + j] = a[j];
        }
        written += take;
        if (written >= out_lanes) break;
        keccak_f1600_state(a);
    }
}
```

Result of previous attempt:
     sha3_256_B16K: correct, 0.15 ms, 400.4 Gbitops/s (u64) (69.2% of 578 Gops/s (u64 bitop, est))
    sha3_256_B256K: correct, 1.47 ms, 663.4 Gbitops/s (u64) (114.7% of 578 Gops/s (u64 bitop, est))
      sha3_256_B4M: correct, 23.94 ms, 651.9 Gbitops/s (u64) (112.7% of 578 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.9638

## History

- iter  0: compile=OK | correct=True | score=0.07578622105259203
- iter  1: compile=OK | correct=True | score=0.7177106681658834
- iter  2: compile=OK | correct=True | score=0.6368853478781129
- iter  3: compile=OK | correct=True | score=0.80610701577865
- iter  4: compile=OK | correct=True | score=0.8430644752236637
- iter  5: compile=OK | correct=True | score=0.9637705614303859

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
