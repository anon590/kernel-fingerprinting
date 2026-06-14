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

#define ROL64(x, n) (((x) << (n)) | ((x) >> (64u - (n))))

#define RHOPI(dst, rot) do {              \
    ulong _tmp = (dst);                   \
    (dst) = ROL64(t, rot);                \
    t = _tmp;                             \
} while (0)

#define KECCAK_ROUND(rc_) do {                                                    \
    ulong c0 = a00 ^ a01 ^ a02 ^ a03 ^ a04;                                       \
    ulong c1 = a10 ^ a11 ^ a12 ^ a13 ^ a14;                                       \
    ulong c2 = a20 ^ a21 ^ a22 ^ a23 ^ a24;                                       \
    ulong c3 = a30 ^ a31 ^ a32 ^ a33 ^ a34;                                       \
    ulong c4 = a40 ^ a41 ^ a42 ^ a43 ^ a44;                                       \
                                                                                   \
    ulong d = c4 ^ ROL64(c1, 1u);                                                  \
    a00 ^= d; a01 ^= d; a02 ^= d; a03 ^= d; a04 ^= d;                              \
    d = c0 ^ ROL64(c2, 1u);                                                        \
    a10 ^= d; a11 ^= d; a12 ^= d; a13 ^= d; a14 ^= d;                              \
    d = c1 ^ ROL64(c3, 1u);                                                        \
    a20 ^= d; a21 ^= d; a22 ^= d; a23 ^= d; a24 ^= d;                              \
    d = c2 ^ ROL64(c4, 1u);                                                        \
    a30 ^= d; a31 ^= d; a32 ^= d; a33 ^= d; a34 ^= d;                              \
    d = c3 ^ ROL64(c0, 1u);                                                        \
    a40 ^= d; a41 ^= d; a42 ^= d; a43 ^= d; a44 ^= d;                              \
                                                                                   \
    ulong t = a10;                                                                 \
    RHOPI(a02,  1u);                                                               \
    RHOPI(a21,  3u);                                                               \
    RHOPI(a12,  6u);                                                               \
    RHOPI(a23, 10u);                                                               \
    RHOPI(a33, 15u);                                                               \
    RHOPI(a30, 21u);                                                               \
    RHOPI(a01, 28u);                                                               \
    RHOPI(a13, 36u);                                                               \
    RHOPI(a31, 45u);                                                               \
    RHOPI(a14, 55u);                                                               \
    RHOPI(a44,  2u);                                                               \
    RHOPI(a40, 14u);                                                               \
    RHOPI(a03, 27u);                                                               \
    RHOPI(a34, 41u);                                                               \
    RHOPI(a43, 56u);                                                               \
    RHOPI(a32,  8u);                                                               \
    RHOPI(a22, 25u);                                                               \
    RHOPI(a20, 43u);                                                               \
    RHOPI(a04, 62u);                                                               \
    RHOPI(a42, 18u);                                                               \
    RHOPI(a24, 39u);                                                               \
    RHOPI(a41, 61u);                                                               \
    RHOPI(a11, 20u);                                                               \
    RHOPI(a10, 44u);                                                               \
                                                                                   \
    ulong b0, b1, b2, b3, b4;                                                      \
                                                                                   \
    b0 = a00; b1 = a10; b2 = a20; b3 = a30; b4 = a40;                              \
    a00 = b0 ^ ((~b1) & b2);                                                       \
    a10 = b1 ^ ((~b2) & b3);                                                       \
    a20 = b2 ^ ((~b3) & b4);                                                       \
    a30 = b3 ^ ((~b4) & b0);                                                       \
    a40 = b4 ^ ((~b0) & b1);                                                       \
                                                                                   \
    b0 = a01; b1 = a11; b2 = a21; b3 = a31; b4 = a41;                              \
    a01 = b0 ^ ((~b1) & b2);                                                       \
    a11 = b1 ^ ((~b2) & b3);                                                       \
    a21 = b2 ^ ((~b3) & b4);                                                       \
    a31 = b3 ^ ((~b4) & b0);                                                       \
    a41 = b4 ^ ((~b0) & b1);                                                       \
                                                                                   \
    b0 = a02; b1 = a12; b2 = a22; b3 = a32; b4 = a42;                              \
    a02 = b0 ^ ((~b1) & b2);                                                       \
    a12 = b1 ^ ((~b2) & b3);                                                       \
    a22 = b2 ^ ((~b3) & b4);                                                       \
    a32 = b3 ^ ((~b4) & b0);                                                       \
    a42 = b4 ^ ((~b0) & b1);                                                       \
                                                                                   \
    b0 = a03; b1 = a13; b2 = a23; b3 = a33; b4 = a43;                              \
    a03 = b0 ^ ((~b1) & b2);                                                       \
    a13 = b1 ^ ((~b2) & b3);                                                       \
    a23 = b2 ^ ((~b3) & b4);                                                       \
    a33 = b3 ^ ((~b4) & b0);                                                       \
    a43 = b4 ^ ((~b0) & b1);                                                       \
                                                                                   \
    b0 = a04; b1 = a14; b2 = a24; b3 = a34; b4 = a44;                              \
    a04 = b0 ^ ((~b1) & b2);                                                       \
    a14 = b1 ^ ((~b2) & b3);                                                       \
    a24 = b2 ^ ((~b3) & b4);                                                       \
    a34 = b3 ^ ((~b4) & b0);                                                       \
    a44 = b4 ^ ((~b0) & b1);                                                       \
                                                                                   \
    a00 ^= (rc_);                                                                  \
} while (0)

#define KECCAK_PERMUTE() do {                              \
    KECCAK_ROUND(0x0000000000000001ul);                    \
    KECCAK_ROUND(0x0000000000008082ul);                    \
    KECCAK_ROUND(0x800000000000808Aul);                    \
    KECCAK_ROUND(0x8000000080008000ul);                    \
    KECCAK_ROUND(0x000000000000808Bul);                    \
    KECCAK_ROUND(0x0000000080000001ul);                    \
    KECCAK_ROUND(0x8000000080008081ul);                    \
    KECCAK_ROUND(0x8000000000008009ul);                    \
    KECCAK_ROUND(0x000000000000008Aul);                    \
    KECCAK_ROUND(0x0000000000000088ul);                    \
    KECCAK_ROUND(0x0000000080008009ul);                    \
    KECCAK_ROUND(0x000000008000000Aul);                    \
    KECCAK_ROUND(0x000000008000808Bul);                    \
    KECCAK_ROUND(0x800000000000008Bul);                    \
    KECCAK_ROUND(0x8000000000008089ul);                    \
    KECCAK_ROUND(0x8000000000008003ul);                    \
    KECCAK_ROUND(0x8000000000008002ul);                    \
    KECCAK_ROUND(0x8000000000000080ul);                    \
    KECCAK_ROUND(0x000000000000800Aul);                    \
    KECCAK_ROUND(0x800000008000000Aul);                    \
    KECCAK_ROUND(0x8000000080008081ul);                    \
    KECCAK_ROUND(0x8000000000008080ul);                    \
    KECCAK_ROUND(0x0000000080000001ul);                    \
    KECCAK_ROUND(0x8000000080008008ul);                    \
} while (0)

#define LOAD_MSG_LANE(n_, var_) do {                       \
    if (msg_lanes > (uint)(n_)) {                           \
        (var_) = in_data[in_base + (uint)(n_)];             \
    }                                                       \
} while (0)

#define STORE_PREFIX(base_, cnt_) do {                      \
    uint _base = (base_);                                   \
    uint _cnt  = (cnt_);                                    \
    if (_cnt >  0u) out_data[_base +  0u] = a00;            \
    if (_cnt >  1u) out_data[_base +  1u] = a10;            \
    if (_cnt >  2u) out_data[_base +  2u] = a20;            \
    if (_cnt >  3u) out_data[_base +  3u] = a30;            \
    if (_cnt >  4u) out_data[_base +  4u] = a40;            \
    if (_cnt >  5u) out_data[_base +  5u] = a01;            \
    if (_cnt >  6u) out_data[_base +  6u] = a11;            \
    if (_cnt >  7u) out_data[_base +  7u] = a21;            \
    if (_cnt >  8u) out_data[_base +  8u] = a31;            \
    if (_cnt >  9u) out_data[_base +  9u] = a41;            \
    if (_cnt > 10u) out_data[_base + 10u] = a02;            \
    if (_cnt > 11u) out_data[_base + 11u] = a12;            \
    if (_cnt > 12u) out_data[_base + 12u] = a22;            \
    if (_cnt > 13u) out_data[_base + 13u] = a32;            \
    if (_cnt > 14u) out_data[_base + 14u] = a42;            \
    if (_cnt > 15u) out_data[_base + 15u] = a03;            \
    if (_cnt > 16u) out_data[_base + 16u] = a13;            \
    if (_cnt > 17u) out_data[_base + 17u] = a23;            \
    if (_cnt > 18u) out_data[_base + 18u] = a33;            \
    if (_cnt > 19u) out_data[_base + 19u] = a43;            \
    if (_cnt > 20u) out_data[_base + 20u] = a04;            \
    if (_cnt > 21u) out_data[_base + 21u] = a14;            \
    if (_cnt > 22u) out_data[_base + 22u] = a24;            \
    if (_cnt > 23u) out_data[_base + 23u] = a34;            \
    if (_cnt > 24u) out_data[_base + 24u] = a44;            \
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

    uint in_base = idx * msg_lanes;

    ulong a00 = 0ul, a10 = 0ul, a20 = 0ul, a30 = 0ul, a40 = 0ul;
    ulong a01 = 0ul, a11 = 0ul, a21 = 0ul, a31 = 0ul, a41 = 0ul;
    ulong a02 = 0ul, a12 = 0ul, a22 = 0ul, a32 = 0ul, a42 = 0ul;
    ulong a03 = 0ul, a13 = 0ul, a23 = 0ul, a33 = 0ul, a43 = 0ul;
    ulong a04 = 0ul, a14 = 0ul, a24 = 0ul, a34 = 0ul, a44 = 0ul;

    ulong dom = (ulong)(domain & 0xFFu);

    if (msg_lanes == 4u) {
        a00 = in_data[in_base + 0u];
        a10 = in_data[in_base + 1u];
        a20 = in_data[in_base + 2u];
        a30 = in_data[in_base + 3u];
        a40 = dom;
    } else {
        LOAD_MSG_LANE( 0, a00);
        LOAD_MSG_LANE( 1, a10);
        LOAD_MSG_LANE( 2, a20);
        LOAD_MSG_LANE( 3, a30);
        LOAD_MSG_LANE( 4, a40);
        LOAD_MSG_LANE( 5, a01);
        LOAD_MSG_LANE( 6, a11);
        LOAD_MSG_LANE( 7, a21);
        LOAD_MSG_LANE( 8, a31);
        LOAD_MSG_LANE( 9, a41);
        LOAD_MSG_LANE(10, a02);
        LOAD_MSG_LANE(11, a12);
        LOAD_MSG_LANE(12, a22);
        LOAD_MSG_LANE(13, a32);
        LOAD_MSG_LANE(14, a42);
        LOAD_MSG_LANE(15, a03);
        LOAD_MSG_LANE(16, a13);
        LOAD_MSG_LANE(17, a23);
        LOAD_MSG_LANE(18, a33);
        LOAD_MSG_LANE(19, a43);
        LOAD_MSG_LANE(20, a04);
        LOAD_MSG_LANE(21, a14);
        LOAD_MSG_LANE(22, a24);
        LOAD_MSG_LANE(23, a34);
        LOAD_MSG_LANE(24, a44);

        switch (msg_lanes) {
            case  0u: a00 ^= dom; break;
            case  1u: a10 ^= dom; break;
            case  2u: a20 ^= dom; break;
            case  3u: a30 ^= dom; break;
            case  4u: a40 ^= dom; break;
            case  5u: a01 ^= dom; break;
            case  6u: a11 ^= dom; break;
            case  7u: a21 ^= dom; break;
            case  8u: a31 ^= dom; break;
            case  9u: a41 ^= dom; break;
            case 10u: a02 ^= dom; break;
            case 11u: a12 ^= dom; break;
            case 12u: a22 ^= dom; break;
            case 13u: a32 ^= dom; break;
            case 14u: a42 ^= dom; break;
            case 15u: a03 ^= dom; break;
            case 16u: a13 ^= dom; break;
            case 17u: a23 ^= dom; break;
            case 18u: a33 ^= dom; break;
            case 19u: a43 ^= dom; break;
            case 20u: a04 ^= dom; break;
            case 21u: a14 ^= dom; break;
            case 22u: a24 ^= dom; break;
            case 23u: a34 ^= dom; break;
            case 24u: a44 ^= dom; break;
            default: break;
        }
    }

    const ulong pad80 = 0x8000000000000000ul;
    if (rate_lanes == 17u) {
        a13 ^= pad80;
    } else if (rate_lanes == 21u) {
        a04 ^= pad80;
    } else {
        switch (rate_lanes - 1u) {
            case  0u: a00 ^= pad80; break;
            case  1u: a10 ^= pad80; break;
            case  2u: a20 ^= pad80; break;
            case  3u: a30 ^= pad80; break;
            case  4u: a40 ^= pad80; break;
            case  5u: a01 ^= pad80; break;
            case  6u: a11 ^= pad80; break;
            case  7u: a21 ^= pad80; break;
            case  8u: a31 ^= pad80; break;
            case  9u: a41 ^= pad80; break;
            case 10u: a02 ^= pad80; break;
            case 11u: a12 ^= pad80; break;
            case 12u: a22 ^= pad80; break;
            case 13u: a32 ^= pad80; break;
            case 14u: a42 ^= pad80; break;
            case 15u: a03 ^= pad80; break;
            case 16u: a13 ^= pad80; break;
            case 17u: a23 ^= pad80; break;
            case 18u: a33 ^= pad80; break;
            case 19u: a43 ^= pad80; break;
            case 20u: a04 ^= pad80; break;
            case 21u: a14 ^= pad80; break;
            case 22u: a24 ^= pad80; break;
            case 23u: a34 ^= pad80; break;
            case 24u: a44 ^= pad80; break;
            default: break;
        }
    }

    uint out_base = idx * out_lanes;
    uint written = 0u;

    for (;;) {
        KECCAK_PERMUTE();

        if (out_lanes == 4u && rate_lanes >= 4u) {
            out_data[out_base + 0u] = a00;
            out_data[out_base + 1u] = a10;
            out_data[out_base + 2u] = a20;
            out_data[out_base + 3u] = a30;
            return;
        }

        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        STORE_PREFIX(out_base + written, take);
        written += take;

        if (written >= out_lanes) return;
    }
}

#undef STORE_PREFIX
#undef LOAD_MSG_LANE
#undef KECCAK_PERMUTE
#undef KECCAK_ROUND
#undef RHOPI
#undef ROL64
```

Result of previous attempt:
     sha3_256_B16K: correct, 0.36 ms, 167.9 Gbitops/s (u64) (14.9% of 1125 Gops/s (u64 bitop, est))
    sha3_256_B256K: correct, 2.20 ms, 442.3 Gbitops/s (u64) (39.3% of 1125 Gops/s (u64 bitop, est))
      sha3_256_B4M: correct, 21.50 ms, 725.7 Gbitops/s (u64) (64.5% of 1125 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.3358

## History

- iter  0: compile=OK | correct=True | score=0.039286301511018605
- iter  1: compile=OK | correct=True | score=0.33577025080587536

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
