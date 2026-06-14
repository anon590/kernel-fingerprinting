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

#define ROL64(x, n) (((x) << (n)) | ((x) >> (64 - (n))))

#define KECCAK_ROUND(RCVAL) do {                                      \
    ulong c0 = a0 ^ a5 ^ a10 ^ a15 ^ a20;                             \
    ulong c1 = a1 ^ a6 ^ a11 ^ a16 ^ a21;                             \
    ulong c2 = a2 ^ a7 ^ a12 ^ a17 ^ a22;                             \
    ulong c3 = a3 ^ a8 ^ a13 ^ a18 ^ a23;                             \
    ulong c4 = a4 ^ a9 ^ a14 ^ a19 ^ a24;                             \
                                                                       \
    ulong d0 = c4 ^ ROL64(c1, 1);                                      \
    ulong d1 = c0 ^ ROL64(c2, 1);                                      \
    ulong d2 = c1 ^ ROL64(c3, 1);                                      \
    ulong d3 = c2 ^ ROL64(c4, 1);                                      \
    ulong d4 = c3 ^ ROL64(c0, 1);                                      \
                                                                       \
    a0 ^= d0;  a5 ^= d0;  a10 ^= d0;  a15 ^= d0;  a20 ^= d0;          \
    a1 ^= d1;  a6 ^= d1;  a11 ^= d1;  a16 ^= d1;  a21 ^= d1;          \
    a2 ^= d2;  a7 ^= d2;  a12 ^= d2;  a17 ^= d2;  a22 ^= d2;          \
    a3 ^= d3;  a8 ^= d3;  a13 ^= d3;  a18 ^= d3;  a23 ^= d3;          \
    a4 ^= d4;  a9 ^= d4;  a14 ^= d4;  a19 ^= d4;  a24 ^= d4;          \
                                                                       \
    ulong t = a1;                                                      \
    ulong u = a10; a10 = ROL64(t,  1); t = u;                          \
          u = a7;  a7  = ROL64(t,  3); t = u;                          \
          u = a11; a11 = ROL64(t,  6); t = u;                          \
          u = a17; a17 = ROL64(t, 10); t = u;                          \
          u = a18; a18 = ROL64(t, 15); t = u;                          \
          u = a3;  a3  = ROL64(t, 21); t = u;                          \
          u = a5;  a5  = ROL64(t, 28); t = u;                          \
          u = a16; a16 = ROL64(t, 36); t = u;                          \
          u = a8;  a8  = ROL64(t, 45); t = u;                          \
          u = a21; a21 = ROL64(t, 55); t = u;                          \
          u = a24; a24 = ROL64(t,  2); t = u;                          \
          u = a4;  a4  = ROL64(t, 14); t = u;                          \
          u = a15; a15 = ROL64(t, 27); t = u;                          \
          u = a23; a23 = ROL64(t, 41); t = u;                          \
          u = a19; a19 = ROL64(t, 56); t = u;                          \
          u = a13; a13 = ROL64(t,  8); t = u;                          \
          u = a12; a12 = ROL64(t, 25); t = u;                          \
          u = a2;  a2  = ROL64(t, 43); t = u;                          \
          u = a20; a20 = ROL64(t, 62); t = u;                          \
          u = a14; a14 = ROL64(t, 18); t = u;                          \
          u = a22; a22 = ROL64(t, 39); t = u;                          \
          u = a9;  a9  = ROL64(t, 61); t = u;                          \
          u = a6;  a6  = ROL64(t, 20); t = u;                          \
    a1 = ROL64(t, 44);                                                 \
                                                                       \
    c0 = a0; c1 = a1; c2 = a2; c3 = a3; c4 = a4;                       \
    a0 = c0 ^ ((~c1) & c2);                                            \
    a1 = c1 ^ ((~c2) & c3);                                            \
    a2 = c2 ^ ((~c3) & c4);                                            \
    a3 = c3 ^ ((~c4) & c0);                                            \
    a4 = c4 ^ ((~c0) & c1);                                            \
                                                                       \
    c0 = a5; c1 = a6; c2 = a7; c3 = a8; c4 = a9;                       \
    a5 = c0 ^ ((~c1) & c2);                                            \
    a6 = c1 ^ ((~c2) & c3);                                            \
    a7 = c2 ^ ((~c3) & c4);                                            \
    a8 = c3 ^ ((~c4) & c0);                                            \
    a9 = c4 ^ ((~c0) & c1);                                            \
                                                                       \
    c0 = a10; c1 = a11; c2 = a12; c3 = a13; c4 = a14;                  \
    a10 = c0 ^ ((~c1) & c2);                                           \
    a11 = c1 ^ ((~c2) & c3);                                           \
    a12 = c2 ^ ((~c3) & c4);                                           \
    a13 = c3 ^ ((~c4) & c0);                                           \
    a14 = c4 ^ ((~c0) & c1);                                           \
                                                                       \
    c0 = a15; c1 = a16; c2 = a17; c3 = a18; c4 = a19;                  \
    a15 = c0 ^ ((~c1) & c2);                                           \
    a16 = c1 ^ ((~c2) & c3);                                           \
    a17 = c2 ^ ((~c3) & c4);                                           \
    a18 = c3 ^ ((~c4) & c0);                                           \
    a19 = c4 ^ ((~c0) & c1);                                           \
                                                                       \
    c0 = a20; c1 = a21; c2 = a22; c3 = a23; c4 = a24;                  \
    a20 = c0 ^ ((~c1) & c2);                                           \
    a21 = c1 ^ ((~c2) & c3);                                           \
    a22 = c2 ^ ((~c3) & c4);                                           \
    a23 = c3 ^ ((~c4) & c0);                                           \
    a24 = c4 ^ ((~c0) & c1);                                           \
                                                                       \
    a0 ^= (ulong)(RCVAL);                                              \
} while (false)

#define KECCAK_PERMUTE() do {                                          \
    KECCAK_ROUND(0x0000000000000001ul);                                \
    KECCAK_ROUND(0x0000000000008082ul);                                \
    KECCAK_ROUND(0x800000000000808Aul);                                \
    KECCAK_ROUND(0x8000000080008000ul);                                \
    KECCAK_ROUND(0x000000000000808Bul);                                \
    KECCAK_ROUND(0x0000000080000001ul);                                \
    KECCAK_ROUND(0x8000000080008081ul);                                \
    KECCAK_ROUND(0x8000000000008009ul);                                \
    KECCAK_ROUND(0x000000000000008Aul);                                \
    KECCAK_ROUND(0x0000000000000088ul);                                \
    KECCAK_ROUND(0x0000000080008009ul);                                \
    KECCAK_ROUND(0x000000008000000Aul);                                \
    KECCAK_ROUND(0x000000008000808Bul);                                \
    KECCAK_ROUND(0x800000000000008Bul);                                \
    KECCAK_ROUND(0x8000000000008089ul);                                \
    KECCAK_ROUND(0x8000000000008003ul);                                \
    KECCAK_ROUND(0x8000000000008002ul);                                \
    KECCAK_ROUND(0x8000000000000080ul);                                \
    KECCAK_ROUND(0x000000000000800Aul);                                \
    KECCAK_ROUND(0x800000008000000Aul);                                \
    KECCAK_ROUND(0x8000000080008081ul);                                \
    KECCAK_ROUND(0x8000000000008080ul);                                \
    KECCAK_ROUND(0x0000000080000001ul);                                \
    KECCAK_ROUND(0x8000000080008008ul);                                \
} while (false)

#define XOR_DOMAIN_TO_LANE(LANE, VAL) do {                             \
    switch (LANE) {                                                     \
        case 0u:  a0  ^= (VAL); break;                                  \
        case 1u:  a1  ^= (VAL); break;                                  \
        case 2u:  a2  ^= (VAL); break;                                  \
        case 3u:  a3  ^= (VAL); break;                                  \
        case 4u:  a4  ^= (VAL); break;                                  \
        case 5u:  a5  ^= (VAL); break;                                  \
        case 6u:  a6  ^= (VAL); break;                                  \
        case 7u:  a7  ^= (VAL); break;                                  \
        case 8u:  a8  ^= (VAL); break;                                  \
        case 9u:  a9  ^= (VAL); break;                                  \
        case 10u: a10 ^= (VAL); break;                                  \
        case 11u: a11 ^= (VAL); break;                                  \
        case 12u: a12 ^= (VAL); break;                                  \
        case 13u: a13 ^= (VAL); break;                                  \
        case 14u: a14 ^= (VAL); break;                                  \
        case 15u: a15 ^= (VAL); break;                                  \
        case 16u: a16 ^= (VAL); break;                                  \
        case 17u: a17 ^= (VAL); break;                                  \
        case 18u: a18 ^= (VAL); break;                                  \
        case 19u: a19 ^= (VAL); break;                                  \
        case 20u: a20 ^= (VAL); break;                                  \
        case 21u: a21 ^= (VAL); break;                                  \
        case 22u: a22 ^= (VAL); break;                                  \
        case 23u: a23 ^= (VAL); break;                                  \
        default:  a24 ^= (VAL); break;                                  \
    }                                                                   \
} while (false)

#define STORE_RATE_PREFIX(BASE, OFFSET, LIMIT) do {                     \
    uint _lim = (LIMIT);                                                \
    uint _pos = (OFFSET);                                               \
    if (_lim > 0u)  out_data[(BASE) + _pos + 0u]  = a0;                 \
    if (_lim > 1u)  out_data[(BASE) + _pos + 1u]  = a1;                 \
    if (_lim > 2u)  out_data[(BASE) + _pos + 2u]  = a2;                 \
    if (_lim > 3u)  out_data[(BASE) + _pos + 3u]  = a3;                 \
    if (_lim > 4u)  out_data[(BASE) + _pos + 4u]  = a4;                 \
    if (_lim > 5u)  out_data[(BASE) + _pos + 5u]  = a5;                 \
    if (_lim > 6u)  out_data[(BASE) + _pos + 6u]  = a6;                 \
    if (_lim > 7u)  out_data[(BASE) + _pos + 7u]  = a7;                 \
    if (_lim > 8u)  out_data[(BASE) + _pos + 8u]  = a8;                 \
    if (_lim > 9u)  out_data[(BASE) + _pos + 9u]  = a9;                 \
    if (_lim > 10u) out_data[(BASE) + _pos + 10u] = a10;                \
    if (_lim > 11u) out_data[(BASE) + _pos + 11u] = a11;                \
    if (_lim > 12u) out_data[(BASE) + _pos + 12u] = a12;                \
    if (_lim > 13u) out_data[(BASE) + _pos + 13u] = a13;                \
    if (_lim > 14u) out_data[(BASE) + _pos + 14u] = a14;                \
    if (_lim > 15u) out_data[(BASE) + _pos + 15u] = a15;                \
    if (_lim > 16u) out_data[(BASE) + _pos + 16u] = a16;                \
    if (_lim > 17u) out_data[(BASE) + _pos + 17u] = a17;                \
    if (_lim > 18u) out_data[(BASE) + _pos + 18u] = a18;                \
    if (_lim > 19u) out_data[(BASE) + _pos + 19u] = a19;                \
    if (_lim > 20u) out_data[(BASE) + _pos + 20u] = a20;                \
    if (_lim > 21u) out_data[(BASE) + _pos + 21u] = a21;                \
    if (_lim > 22u) out_data[(BASE) + _pos + 22u] = a22;                \
    if (_lim > 23u) out_data[(BASE) + _pos + 23u] = a23;                \
    if (_lim > 24u) out_data[(BASE) + _pos + 24u] = a24;                \
} while (false)

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

    if (msg_bytes == 32u && rate_bytes == 136u && out_bytes == 32u && ((domain & 0xFFu) == 0x06u)) {
        const uint in_base = idx << 2;
        ulong a0 = in_data[in_base + 0u];
        ulong a1 = in_data[in_base + 1u];
        ulong a2 = in_data[in_base + 2u];
        ulong a3 = in_data[in_base + 3u];
        ulong a4 = 0x0000000000000006ul;
        ulong a5 = 0ul,  a6 = 0ul,  a7 = 0ul,  a8 = 0ul,  a9 = 0ul;
        ulong a10 = 0ul, a11 = 0ul, a12 = 0ul, a13 = 0ul, a14 = 0ul;
        ulong a15 = 0ul, a16 = 0x8000000000000000ul, a17 = 0ul, a18 = 0ul, a19 = 0ul;
        ulong a20 = 0ul, a21 = 0ul, a22 = 0ul, a23 = 0ul, a24 = 0ul;

        KECCAK_PERMUTE();

        const uint out_base = idx << 2;
        out_data[out_base + 0u] = a0;
        out_data[out_base + 1u] = a1;
        out_data[out_base + 2u] = a2;
        out_data[out_base + 3u] = a3;
        return;
    }

    uint msg_lanes  = msg_bytes >> 3;
    uint rate_lanes = rate_bytes >> 3;
    uint out_lanes  = out_bytes >> 3;

    uint in_base  = idx * msg_lanes;
    uint out_base = idx * out_lanes;

    ulong a0 = 0ul,  a1 = 0ul,  a2 = 0ul,  a3 = 0ul,  a4 = 0ul;
    ulong a5 = 0ul,  a6 = 0ul,  a7 = 0ul,  a8 = 0ul,  a9 = 0ul;
    ulong a10 = 0ul, a11 = 0ul, a12 = 0ul, a13 = 0ul, a14 = 0ul;
    ulong a15 = 0ul, a16 = 0ul, a17 = 0ul, a18 = 0ul, a19 = 0ul;
    ulong a20 = 0ul, a21 = 0ul, a22 = 0ul, a23 = 0ul, a24 = 0ul;

    if (msg_lanes > 0u)  a0  = in_data[in_base + 0u];
    if (msg_lanes > 1u)  a1  = in_data[in_base + 1u];
    if (msg_lanes > 2u)  a2  = in_data[in_base + 2u];
    if (msg_lanes > 3u)  a3  = in_data[in_base + 3u];
    if (msg_lanes > 4u)  a4  = in_data[in_base + 4u];
    if (msg_lanes > 5u)  a5  = in_data[in_base + 5u];
    if (msg_lanes > 6u)  a6  = in_data[in_base + 6u];
    if (msg_lanes > 7u)  a7  = in_data[in_base + 7u];
    if (msg_lanes > 8u)  a8  = in_data[in_base + 8u];
    if (msg_lanes > 9u)  a9  = in_data[in_base + 9u];
    if (msg_lanes > 10u) a10 = in_data[in_base + 10u];
    if (msg_lanes > 11u) a11 = in_data[in_base + 11u];
    if (msg_lanes > 12u) a12 = in_data[in_base + 12u];
    if (msg_lanes > 13u) a13 = in_data[in_base + 13u];
    if (msg_lanes > 14u) a14 = in_data[in_base + 14u];
    if (msg_lanes > 15u) a15 = in_data[in_base + 15u];
    if (msg_lanes > 16u) a16 = in_data[in_base + 16u];
    if (msg_lanes > 17u) a17 = in_data[in_base + 17u];
    if (msg_lanes > 18u) a18 = in_data[in_base + 18u];
    if (msg_lanes > 19u) a19 = in_data[in_base + 19u];
    if (msg_lanes > 20u) a20 = in_data[in_base + 20u];
    if (msg_lanes > 21u) a21 = in_data[in_base + 21u];
    if (msg_lanes > 22u) a22 = in_data[in_base + 22u];
    if (msg_lanes > 23u) a23 = in_data[in_base + 23u];

    XOR_DOMAIN_TO_LANE(msg_lanes, (ulong)(domain & 0xFFu));
    XOR_DOMAIN_TO_LANE(rate_lanes - 1u, 0x8000000000000000ul);

    KECCAK_PERMUTE();

    if (out_lanes == 4u && rate_lanes >= 4u) {
        out_data[out_base + 0u] = a0;
        out_data[out_base + 1u] = a1;
        out_data[out_base + 2u] = a2;
        out_data[out_base + 3u] = a3;
        return;
    }

    uint written = 0u;
    while (written < out_lanes) {
        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        STORE_RATE_PREFIX(out_base, written, take);
        written += take;
        if (written >= out_lanes) break;
        KECCAK_PERMUTE();
    }
}

#undef STORE_RATE_PREFIX
#undef XOR_DOMAIN_TO_LANE
#undef KECCAK_PERMUTE
#undef KECCAK_ROUND
#undef ROL64
```

Result of previous attempt:
     sha3_256_B16K: correct, 0.35 ms, 173.1 Gbitops/s (u64) (30.0% of 577 Gops/s (u64 bitop, est))
    sha3_256_B256K: correct, 1.94 ms, 501.5 Gbitops/s (u64) (86.9% of 577 Gops/s (u64 bitop, est))
      sha3_256_B4M: correct, 21.00 ms, 743.1 Gbitops/s (u64) (128.7% of 577 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.6946

## History

- iter  1: compile=OK | correct=True | score=0.6787960616779275
- iter  2: compile=OK | correct=True | score=0.6500687158019971
- iter  3: compile=OK | correct=True | score=0.6672808901210725
- iter  4: compile=OK | correct=True | score=0.6340098150247799
- iter  5: compile=OK | correct=True | score=0.6773371937102108
- iter  6: compile=OK | correct=True | score=0.6312737831809283
- iter  7: compile=OK | correct=True | score=0.667931537799551
- iter  8: compile=OK | correct=True | score=0.6946029958059002

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
