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

inline ulong ROTL(ulong x, uint k) {
    return (x << k) | (x >> (64u - k));
}

#define KECCAK_ROUND(RCV)                                              \
    {                                                                  \
        ulong C0 = s00 ^ s01 ^ s02 ^ s03 ^ s04;                        \
        ulong C1 = s10 ^ s11 ^ s12 ^ s13 ^ s14;                        \
        ulong C2 = s20 ^ s21 ^ s22 ^ s23 ^ s24;                        \
        ulong C3 = s30 ^ s31 ^ s32 ^ s33 ^ s34;                        \
        ulong C4 = s40 ^ s41 ^ s42 ^ s43 ^ s44;                        \
        ulong D0 = C4 ^ ROTL(C1, 1);                                   \
        ulong D1 = C0 ^ ROTL(C2, 1);                                   \
        ulong D2 = C1 ^ ROTL(C3, 1);                                   \
        ulong D3 = C2 ^ ROTL(C4, 1);                                   \
        ulong D4 = C3 ^ ROTL(C0, 1);                                   \
        ulong b00 = (s00 ^ D0);                                        \
        ulong b02 = ROTL(s10 ^ D1,  1);                                \
        ulong b04 = ROTL(s20 ^ D2, 62);                                \
        ulong b01 = ROTL(s30 ^ D3, 28);                                \
        ulong b03 = ROTL(s40 ^ D4, 27);                                \
        ulong b13 = ROTL(s01 ^ D0, 36);                                \
        ulong b10 = ROTL(s11 ^ D1, 44);                                \
        ulong b12 = ROTL(s21 ^ D2,  6);                                \
        ulong b14 = ROTL(s31 ^ D3, 55);                                \
        ulong b11 = ROTL(s41 ^ D4, 20);                                \
        ulong b21 = ROTL(s02 ^ D0,  3);                                \
        ulong b23 = ROTL(s12 ^ D1, 10);                                \
        ulong b20 = ROTL(s22 ^ D2, 43);                                \
        ulong b22 = ROTL(s32 ^ D3, 25);                                \
        ulong b24 = ROTL(s42 ^ D4, 39);                                \
        ulong b34 = ROTL(s03 ^ D0, 41);                                \
        ulong b31 = ROTL(s13 ^ D1, 45);                                \
        ulong b33 = ROTL(s23 ^ D2, 15);                                \
        ulong b30 = ROTL(s33 ^ D3, 21);                                \
        ulong b32 = ROTL(s43 ^ D4,  8);                                \
        ulong b42 = ROTL(s04 ^ D0, 18);                                \
        ulong b44 = ROTL(s14 ^ D1,  2);                                \
        ulong b41 = ROTL(s24 ^ D2, 61);                                \
        ulong b43 = ROTL(s34 ^ D3, 56);                                \
        ulong b40 = ROTL(s44 ^ D4, 14);                                \
        s00 = b00 ^ ((~b10) & b20) ^ (RCV);                            \
        s10 = b10 ^ ((~b20) & b30);                                    \
        s20 = b20 ^ ((~b30) & b40);                                    \
        s30 = b30 ^ ((~b40) & b00);                                    \
        s40 = b40 ^ ((~b00) & b10);                                    \
        s01 = b01 ^ ((~b11) & b21);                                    \
        s11 = b11 ^ ((~b21) & b31);                                    \
        s21 = b21 ^ ((~b31) & b41);                                    \
        s31 = b31 ^ ((~b41) & b01);                                    \
        s41 = b41 ^ ((~b01) & b11);                                    \
        s02 = b02 ^ ((~b12) & b22);                                    \
        s12 = b12 ^ ((~b22) & b32);                                    \
        s22 = b22 ^ ((~b32) & b42);                                    \
        s32 = b32 ^ ((~b42) & b02);                                    \
        s42 = b42 ^ ((~b02) & b12);                                    \
        s03 = b03 ^ ((~b13) & b23);                                    \
        s13 = b13 ^ ((~b23) & b33);                                    \
        s23 = b23 ^ ((~b33) & b43);                                    \
        s33 = b33 ^ ((~b43) & b03);                                    \
        s43 = b43 ^ ((~b03) & b13);                                    \
        s04 = b04 ^ ((~b14) & b24);                                    \
        s14 = b14 ^ ((~b24) & b34);                                    \
        s24 = b24 ^ ((~b34) & b44);                                    \
        s34 = b34 ^ ((~b44) & b04);                                    \
        s44 = b44 ^ ((~b04) & b14);                                    \
    }

#define KECCAK_F1600()                  \
    KECCAK_ROUND(KECCAK_RC[ 0]);        \
    KECCAK_ROUND(KECCAK_RC[ 1]);        \
    KECCAK_ROUND(KECCAK_RC[ 2]);        \
    KECCAK_ROUND(KECCAK_RC[ 3]);        \
    KECCAK_ROUND(KECCAK_RC[ 4]);        \
    KECCAK_ROUND(KECCAK_RC[ 5]);        \
    KECCAK_ROUND(KECCAK_RC[ 6]);        \
    KECCAK_ROUND(KECCAK_RC[ 7]);        \
    KECCAK_ROUND(KECCAK_RC[ 8]);        \
    KECCAK_ROUND(KECCAK_RC[ 9]);        \
    KECCAK_ROUND(KECCAK_RC[10]);        \
    KECCAK_ROUND(KECCAK_RC[11]);        \
    KECCAK_ROUND(KECCAK_RC[12]);        \
    KECCAK_ROUND(KECCAK_RC[13]);        \
    KECCAK_ROUND(KECCAK_RC[14]);        \
    KECCAK_ROUND(KECCAK_RC[15]);        \
    KECCAK_ROUND(KECCAK_RC[16]);        \
    KECCAK_ROUND(KECCAK_RC[17]);        \
    KECCAK_ROUND(KECCAK_RC[18]);        \
    KECCAK_ROUND(KECCAK_RC[19]);        \
    KECCAK_ROUND(KECCAK_RC[20]);        \
    KECCAK_ROUND(KECCAK_RC[21]);        \
    KECCAK_ROUND(KECCAK_RC[22]);        \
    KECCAK_ROUND(KECCAK_RC[23]);

kernel void keccak_f1600_batch(
    device const ulong *in_data    [[buffer(0)]],
    device       ulong *out_data   [[buffer(1)]],
    constant uint      &batch      [[buffer(2)]],
    constant uint      &msg_bytes  [[buffer(3)]],
    constant uint      &rate_bytes [[buffer(4)]],
    constant uint      &out_bytes  [[buffer(5)]],
    constant uint      &domain     [[buffer(6)]],
    uint idx     [[thread_position_in_grid]],
    uint tid_tg  [[thread_position_in_threadgroup]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint msg_lanes  = msg_bytes  >> 3;
    uint rate_lanes = rate_bytes >> 3;
    uint out_lanes  = out_bytes  >> 3;

    ulong s00=0, s10=0, s20=0, s30=0, s40=0;
    ulong s01=0, s11=0, s21=0, s31=0, s41=0;
    ulong s02=0, s12=0, s22=0, s32=0, s42=0;
    ulong s03=0, s13=0, s23=0, s33=0, s43=0;
    ulong s04=0, s14=0, s24=0, s34=0, s44=0;

    // Hot fast path: SHA3-256 with 32-byte input (msg_lanes=4, rate=17, out=4).
    if (msg_lanes == 4u && rate_lanes == 17u && out_lanes == 4u) {
        threadgroup ulong tg_buf[64 * 4];  // up to tg_size=64, 4 lanes each

        uint tg_base = tg_id * tg_size * 4u;
        uint valid_threads = min(tg_size, batch - tg_id * tg_size);
        uint valid_in = valid_threads * 4u;

        // Coalesced load: stride tg_size, each thread reads 4 strided ulongs.
        for (uint k = 0u; k < 4u; ++k) {
            uint off = k * tg_size + tid_tg;
            if (off < valid_in) {
                tg_buf[off] = in_data[tg_base + off];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (idx < batch) {
            uint lb = tid_tg * 4u;
            s00 = tg_buf[lb + 0u];
            s10 = tg_buf[lb + 1u];
            s20 = tg_buf[lb + 2u];
            s30 = tg_buf[lb + 3u];
            s40 = (ulong)(domain & 0xFFu);
            // rate_lanes-1 = 16 -> (x=1,y=3) = s13
            s13 = 0x8000000000000000ul;

            KECCAK_F1600();

            // Direct device write (no extra TG roundtrip): writes are
            // 4 ulongs per thread = 32 bytes, naturally vector-sized.
            device ulong *out_ptr = out_data + idx * 4u;
            out_ptr[0] = s00;
            out_ptr[1] = s10;
            out_ptr[2] = s20;
            out_ptr[3] = s30;
        }
        return;
    }

    if (idx >= batch) return;

    // General path: fold absorb directly into named registers.
    uint in_base = idx * msg_lanes;
    ulong lane_arr[25];
    for (uint i = 0u; i < 25u; ++i) lane_arr[i] = 0ul;
    for (uint i = 0u; i < msg_lanes; ++i) lane_arr[i] = in_data[in_base + i];
    lane_arr[msg_lanes]       ^= (ulong)(domain & 0xFFu);
    lane_arr[rate_lanes - 1u] ^= 0x8000000000000000ul;

    s00 = lane_arr[ 0]; s10 = lane_arr[ 1]; s20 = lane_arr[ 2]; s30 = lane_arr[ 3]; s40 = lane_arr[ 4];
    s01 = lane_arr[ 5]; s11 = lane_arr[ 6]; s21 = lane_arr[ 7]; s31 = lane_arr[ 8]; s41 = lane_arr[ 9];
    s02 = lane_arr[10]; s12 = lane_arr[11]; s22 = lane_arr[12]; s32 = lane_arr[13]; s42 = lane_arr[14];
    s03 = lane_arr[15]; s13 = lane_arr[16]; s23 = lane_arr[17]; s33 = lane_arr[18]; s43 = lane_arr[19];
    s04 = lane_arr[20]; s14 = lane_arr[21]; s24 = lane_arr[22]; s34 = lane_arr[23]; s44 = lane_arr[24];

    uint out_base = idx * out_lanes;
    uint written  = 0u;
    for (;;) {
        KECCAK_F1600();

        ulong lanes[25] = {
            s00, s10, s20, s30, s40,
            s01, s11, s21, s31, s41,
            s02, s12, s22, s32, s42,
            s03, s13, s23, s33, s43,
            s04, s14, s24, s34, s44
        };
        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        for (uint j = 0u; j < take; ++j) {
            out_data[out_base + written + j] = lanes[j];
        }
        written += take;
        if (written >= out_lanes) break;
    }
}
```

Result of previous attempt:
     sha3_256_B16K: correct, 0.41 ms, 149.1 Gbitops/s (u64) (13.3% of 1125 Gops/s (u64 bitop, est))
    sha3_256_B256K: correct, 1.84 ms, 531.1 Gbitops/s (u64) (47.2% of 1125 Gops/s (u64 bitop, est))
      sha3_256_B4M: correct, 21.61 ms, 722.1 Gbitops/s (u64) (64.2% of 1125 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.3424

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

inline ulong ROTL(ulong x, uint k) {
    return (x << k) | (x >> (64u - k));
}

#define KECCAK_ROUND(RCV)                                              \
    {                                                                  \
        ulong C0 = s00 ^ s01 ^ s02 ^ s03 ^ s04;                        \
        ulong C1 = s10 ^ s11 ^ s12 ^ s13 ^ s14;                        \
        ulong C2 = s20 ^ s21 ^ s22 ^ s23 ^ s24;                        \
        ulong C3 = s30 ^ s31 ^ s32 ^ s33 ^ s34;                        \
        ulong C4 = s40 ^ s41 ^ s42 ^ s43 ^ s44;                        \
        ulong D0 = C4 ^ ROTL(C1, 1);                                   \
        ulong D1 = C0 ^ ROTL(C2, 1);                                   \
        ulong D2 = C1 ^ ROTL(C3, 1);                                   \
        ulong D3 = C2 ^ ROTL(C4, 1);                                   \
        ulong D4 = C3 ^ ROTL(C0, 1);                                   \
        ulong b00 = (s00 ^ D0);                                        \
        ulong b02 = ROTL(s10 ^ D1,  1);                                \
        ulong b04 = ROTL(s20 ^ D2, 62);                                \
        ulong b01 = ROTL(s30 ^ D3, 28);                                \
        ulong b03 = ROTL(s40 ^ D4, 27);                                \
        ulong b13 = ROTL(s01 ^ D0, 36);                                \
        ulong b10 = ROTL(s11 ^ D1, 44);                                \
        ulong b12 = ROTL(s21 ^ D2,  6);                                \
        ulong b14 = ROTL(s31 ^ D3, 55);                                \
        ulong b11 = ROTL(s41 ^ D4, 20);                                \
        ulong b21 = ROTL(s02 ^ D0,  3);                                \
        ulong b23 = ROTL(s12 ^ D1, 10);                                \
        ulong b20 = ROTL(s22 ^ D2, 43);                                \
        ulong b22 = ROTL(s32 ^ D3, 25);                                \
        ulong b24 = ROTL(s42 ^ D4, 39);                                \
        ulong b34 = ROTL(s03 ^ D0, 41);                                \
        ulong b31 = ROTL(s13 ^ D1, 45);                                \
        ulong b33 = ROTL(s23 ^ D2, 15);                                \
        ulong b30 = ROTL(s33 ^ D3, 21);                                \
        ulong b32 = ROTL(s43 ^ D4,  8);                                \
        ulong b42 = ROTL(s04 ^ D0, 18);                                \
        ulong b44 = ROTL(s14 ^ D1,  2);                                \
        ulong b41 = ROTL(s24 ^ D2, 61);                                \
        ulong b43 = ROTL(s34 ^ D3, 56);                                \
        ulong b40 = ROTL(s44 ^ D4, 14);                                \
        s00 = b00 ^ ((~b10) & b20) ^ (RCV);                            \
        s10 = b10 ^ ((~b20) & b30);                                    \
        s20 = b20 ^ ((~b30) & b40);                                    \
        s30 = b30 ^ ((~b40) & b00);                                    \
        s40 = b40 ^ ((~b00) & b10);                                    \
        s01 = b01 ^ ((~b11) & b21);                                    \
        s11 = b11 ^ ((~b21) & b31);                                    \
        s21 = b21 ^ ((~b31) & b41);                                    \
        s31 = b31 ^ ((~b41) & b01);                                    \
        s41 = b41 ^ ((~b01) & b11);                                    \
        s02 = b02 ^ ((~b12) & b22);                                    \
        s12 = b12 ^ ((~b22) & b32);                                    \
        s22 = b22 ^ ((~b32) & b42);                                    \
        s32 = b32 ^ ((~b42) & b02);                                    \
        s42 = b42 ^ ((~b02) & b12);                                    \
        s03 = b03 ^ ((~b13) & b23);                                    \
        s13 = b13 ^ ((~b23) & b33);                                    \
        s23 = b23 ^ ((~b33) & b43);                                    \
        s33 = b33 ^ ((~b43) & b03);                                    \
        s43 = b43 ^ ((~b03) & b13);                                    \
        s04 = b04 ^ ((~b14) & b24);                                    \
        s14 = b14 ^ ((~b24) & b34);                                    \
        s24 = b24 ^ ((~b34) & b44);                                    \
        s34 = b34 ^ ((~b44) & b04);                                    \
        s44 = b44 ^ ((~b04) & b14);                                    \
    }

#define KECCAK_F1600()                  \
    KECCAK_ROUND(KECCAK_RC[ 0]);        \
    KECCAK_ROUND(KECCAK_RC[ 1]);        \
    KECCAK_ROUND(KECCAK_RC[ 2]);        \
    KECCAK_ROUND(KECCAK_RC[ 3]);        \
    KECCAK_ROUND(KECCAK_RC[ 4]);        \
    KECCAK_ROUND(KECCAK_RC[ 5]);        \
    KECCAK_ROUND(KECCAK_RC[ 6]);        \
    KECCAK_ROUND(KECCAK_RC[ 7]);        \
    KECCAK_ROUND(KECCAK_RC[ 8]);        \
    KECCAK_ROUND(KECCAK_RC[ 9]);        \
    KECCAK_ROUND(KECCAK_RC[10]);        \
    KECCAK_ROUND(KECCAK_RC[11]);        \
    KECCAK_ROUND(KECCAK_RC[12]);        \
    KECCAK_ROUND(KECCAK_RC[13]);        \
    KECCAK_ROUND(KECCAK_RC[14]);        \
    KECCAK_ROUND(KECCAK_RC[15]);        \
    KECCAK_ROUND(KECCAK_RC[16]);        \
    KECCAK_ROUND(KECCAK_RC[17]);        \
    KECCAK_ROUND(KECCAK_RC[18]);        \
    KECCAK_ROUND(KECCAK_RC[19]);        \
    KECCAK_ROUND(KECCAK_RC[20]);        \
    KECCAK_ROUND(KECCAK_RC[21]);        \
    KECCAK_ROUND(KECCAK_RC[22]);        \
    KECCAK_ROUND(KECCAK_RC[23]);

kernel void keccak_f1600_batch(
    device const ulong *in_data    [[buffer(0)]],
    device       ulong *out_data   [[buffer(1)]],
    constant uint      &batch      [[buffer(2)]],
    constant uint      &msg_bytes  [[buffer(3)]],
    constant uint      &rate_bytes [[buffer(4)]],
    constant uint      &out_bytes  [[buffer(5)]],
    constant uint      &domain     [[buffer(6)]],
    uint idx     [[thread_position_in_grid]],
    uint tid_tg  [[thread_position_in_threadgroup]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint msg_lanes  = msg_bytes  >> 3;
    uint rate_lanes = rate_bytes >> 3;
    uint out_lanes  = out_bytes  >> 3;

    ulong s00=0, s10=0, s20=0, s30=0, s40=0;
    ulong s01=0, s11=0, s21=0, s31=0, s41=0;
    ulong s02=0, s12=0, s22=0, s32=0, s42=0;
    ulong s03=0, s13=0, s23=0, s33=0, s43=0;
    ulong s04=0, s14=0, s24=0, s34=0, s44=0;

    // Hot fast path: SHA3-256 with 32-byte input (msg_lanes=4, rate=17, out=4).
    // Use threadgroup-cooperative coalesced loads/stores.
    if (msg_lanes == 4u && rate_lanes == 17u && out_lanes == 4u) {
        threadgroup ulong tg_buf[64 * 4];  // up to tg_size=64, 4 lanes each

        uint tg_base_in = tg_id * tg_size * 4u;
        // Cooperative coalesced load: each thread loads 4 contiguous ulongs.
        // Total ulongs to load = tg_size * 4. Each thread loads 4 of them
        // at stride tg_size for coalescing.
        uint total_in = tg_size * 4u;
        // Guard: last TG may be partial.
        uint valid_threads = min(tg_size, batch - tg_id * tg_size);
        uint valid_in = valid_threads * 4u;

        for (uint k = 0u; k < 4u; ++k) {
            uint off = k * tg_size + tid_tg;
            if (off < valid_in) {
                tg_buf[off] = in_data[tg_base_in + off];
            }
            (void)total_in;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (idx < batch) {
            uint lb = tid_tg * 4u;
            s00 = tg_buf[lb + 0u];
            s10 = tg_buf[lb + 1u];
            s20 = tg_buf[lb + 2u];
            s30 = tg_buf[lb + 3u];
            s40 = (ulong)(domain & 0xFFu);
            // rate_lanes-1 = 16 -> lane index 16 = s13
            s13 = 0x8000000000000000ul;

            KECCAK_F1600();
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Cooperative coalesced store: write 4 output lanes per thread.
        if (idx < batch) {
            uint lb = tid_tg * 4u;
            tg_buf[lb + 0u] = s00;
            tg_buf[lb + 1u] = s10;
            tg_buf[lb + 2u] = s20;
            tg_buf[lb + 3u] = s30;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint tg_base_out = tg_id * tg_size * 4u;
        uint valid_out = valid_threads * 4u;
        for (uint k = 0u; k < 4u; ++k) {
            uint off = k * tg_size + tid_tg;
            if (off < valid_out) {
                out_data[tg_base_out + off] = tg_buf[off];
            }
        }
        return;
    }

    if (idx >= batch) return;

    // General path
    {
        ulong tmp[25];
        for (uint i = 0u; i < 25u; ++i) tmp[i] = 0ul;
        uint in_base = idx * msg_lanes;
        for (uint i = 0u; i < msg_lanes; ++i) tmp[i] = in_data[in_base + i];
        tmp[msg_lanes]       ^= (ulong)(domain & 0xFFu);
        tmp[rate_lanes - 1u] ^= 0x8000000000000000ul;

        s00 = tmp[ 0]; s10 = tmp[ 1]; s20 = tmp[ 2]; s30 = tmp[ 3]; s40 = tmp[ 4];
        s01 = tmp[ 5]; s11 = tmp[ 6]; s21 = tmp[ 7]; s31 = tmp[ 8]; s41 = tmp[ 9];
        s02 = tmp[10]; s12 = tmp[11]; s22 = tmp[12]; s32 = tmp[13]; s42 = tmp[14];
        s03 = tmp[15]; s13 = tmp[16]; s23 = tmp[17]; s33 = tmp[18]; s43 = tmp[19];
        s04 = tmp[20]; s14 = tmp[21]; s24 = tmp[22]; s34 = tmp[23]; s44 = tmp[24];
    }

    uint out_base = idx * out_lanes;
    uint written  = 0u;
    for (;;) {
        KECCAK_F1600();

        ulong lanes[25] = {
            s00, s10, s20, s30, s40,
            s01, s11, s21, s31, s41,
            s02, s12, s22, s32, s42,
            s03, s13, s23, s33, s43,
            s04, s14, s24, s34, s44
        };
        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        for (uint j = 0u; j < take; ++j) {
            out_data[out_base + written + j] = lanes[j];
        }
        written += take;
        if (written >= out_lanes) break;
    }
}
```

Incumbent result:
     sha3_256_B16K: correct, 0.21 ms, 288.9 Gbitops/s (u64) (25.7% of 1125 Gops/s (u64 bitop, est))
    sha3_256_B256K: correct, 1.37 ms, 710.9 Gbitops/s (u64) (63.2% of 1125 Gops/s (u64 bitop, est))
      sha3_256_B4M: correct, 21.28 ms, 733.3 Gbitops/s (u64) (65.2% of 1125 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.4729

## History

- iter  3: compile=OK | correct=True | score=0.30244444577489676
- iter  4: compile=OK | correct=True | score=0.35939762940062103
- iter  5: compile=OK | correct=True | score=0.3776191693522675
- iter  6: compile=OK | correct=True | score=0.38452581384184564
- iter  7: compile=OK | correct=True | score=0.47294261596443365
- iter  8: compile=OK | correct=True | score=0.3817890738017575
- iter  9: compile=OK | correct=True | score=0.3200972958052429
- iter 10: compile=OK | correct=True | score=0.3424460885910696

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
