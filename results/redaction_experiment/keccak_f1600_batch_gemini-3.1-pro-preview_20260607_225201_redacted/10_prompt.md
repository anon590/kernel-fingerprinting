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

inline ulong rotl(ulong x, uint k) {
    return (x << k) | (x >> (64u - k));
}

#define KECCAK_ROUND(RC) \
do { \
    ulong C0 = s0 ^ s5 ^ s10 ^ s15 ^ s20; \
    ulong C1 = s1 ^ s6 ^ s11 ^ s16 ^ s21; \
    ulong C2 = s2 ^ s7 ^ s12 ^ s17 ^ s22; \
    ulong C3 = s3 ^ s8 ^ s13 ^ s18 ^ s23; \
    ulong C4 = s4 ^ s9 ^ s14 ^ s19 ^ s24; \
    ulong D0 = C4 ^ rotl(C1, 1u); \
    ulong D1 = C0 ^ rotl(C2, 1u); \
    ulong D2 = C1 ^ rotl(C3, 1u); \
    ulong D3 = C2 ^ rotl(C4, 1u); \
    ulong D4 = C3 ^ rotl(C0, 1u); \
    ulong B0 = s0 ^ D0; \
    ulong B1 = rotl(s6 ^ D1, 44u); \
    ulong B2 = rotl(s12 ^ D2, 43u); \
    ulong B3 = rotl(s18 ^ D3, 21u); \
    ulong B4 = rotl(s24 ^ D4, 14u); \
    ulong n0 = B0 ^ (~B1 & B2) ^ (RC); \
    ulong n1 = B1 ^ (~B2 & B3); \
    ulong n2 = B2 ^ (~B3 & B4); \
    ulong n3 = B3 ^ (~B4 & B0); \
    ulong n4 = B4 ^ (~B0 & B1); \
    B0 = rotl(s3 ^ D3, 28u); \
    B1 = rotl(s9 ^ D4, 20u); \
    B2 = rotl(s10 ^ D0, 3u); \
    B3 = rotl(s16 ^ D1, 45u); \
    B4 = rotl(s22 ^ D2, 61u); \
    ulong n5 = B0 ^ (~B1 & B2); \
    ulong n6 = B1 ^ (~B2 & B3); \
    ulong n7 = B2 ^ (~B3 & B4); \
    ulong n8 = B3 ^ (~B4 & B0); \
    ulong n9 = B4 ^ (~B0 & B1); \
    B0 = rotl(s1 ^ D1, 1u); \
    B1 = rotl(s7 ^ D2, 6u); \
    B2 = rotl(s13 ^ D3, 25u); \
    B3 = rotl(s19 ^ D4, 8u); \
    B4 = rotl(s20 ^ D0, 18u); \
    ulong n10 = B0 ^ (~B1 & B2); \
    ulong n11 = B1 ^ (~B2 & B3); \
    ulong n12 = B2 ^ (~B3 & B4); \
    ulong n13 = B3 ^ (~B4 & B0); \
    ulong n14 = B4 ^ (~B0 & B1); \
    B0 = rotl(s4 ^ D4, 27u); \
    B1 = rotl(s5 ^ D0, 36u); \
    B2 = rotl(s11 ^ D1, 10u); \
    B3 = rotl(s17 ^ D2, 15u); \
    B4 = rotl(s23 ^ D3, 56u); \
    ulong n15 = B0 ^ (~B1 & B2); \
    ulong n16 = B1 ^ (~B2 & B3); \
    ulong n17 = B2 ^ (~B3 & B4); \
    ulong n18 = B3 ^ (~B4 & B0); \
    ulong n19 = B4 ^ (~B0 & B1); \
    B0 = rotl(s2 ^ D2, 62u); \
    B1 = rotl(s8 ^ D3, 55u); \
    B2 = rotl(s14 ^ D4, 39u); \
    B3 = rotl(s15 ^ D0, 41u); \
    B4 = rotl(s21 ^ D1, 2u); \
    ulong n20 = B0 ^ (~B1 & B2); \
    ulong n21 = B1 ^ (~B2 & B3); \
    ulong n22 = B2 ^ (~B3 & B4); \
    ulong n23 = B3 ^ (~B4 & B0); \
    ulong n24 = B4 ^ (~B0 & B1); \
    s0=n0; s1=n1; s2=n2; s3=n3; s4=n4; \
    s5=n5; s6=n6; s7=n7; s8=n8; s9=n9; \
    s10=n10; s11=n11; s12=n12; s13=n13; s14=n14; \
    s15=n15; s16=n16; s17=n17; s18=n18; s19=n19; \
    s20=n20; s21=n21; s22=n22; s23=n23; s24=n24; \
} while(0)

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
    if (out_bytes == 0) return;

    ulong s0=0, s1=0, s2=0, s3=0, s4=0;
    ulong s5=0, s6=0, s7=0, s8=0, s9=0;
    ulong s10=0, s11=0, s12=0, s13=0, s14=0;
    ulong s15=0, s16=0, s17=0, s18=0, s19=0;
    ulong s20=0, s21=0, s22=0, s23=0, s24=0;

    uint msg_lanes  = msg_bytes  >> 3;
    uint rate_lanes = rate_bytes >> 3;
    uint out_lanes  = out_bytes  >> 3;
    ulong domain_val = (ulong)(domain & 0xFFu);

    // Initial Absorb Phase
    if (msg_lanes == 4) {
        ulong4 msg = ((device const ulong4*)in_data)[idx];
        s0 = msg.x;
        s1 = msg.y;
        s2 = msg.z;
        s3 = msg.w;
        s4 = domain_val;
    } else {
        uint in_base = idx * msg_lanes;
        switch (msg_lanes) {
            case 24: s23 ^= in_data[in_base + 23];
            case 23: s22 ^= in_data[in_base + 22];
            case 22: s21 ^= in_data[in_base + 21];
            case 21: s20 ^= in_data[in_base + 20];
            case 20: s19 ^= in_data[in_base + 19];
            case 19: s18 ^= in_data[in_base + 18];
            case 18: s17 ^= in_data[in_base + 17];
            case 17: s16 ^= in_data[in_base + 16];
            case 16: s15 ^= in_data[in_base + 15];
            case 15: s14 ^= in_data[in_base + 14];
            case 14: s13 ^= in_data[in_base + 13];
            case 13: s12 ^= in_data[in_base + 12];
            case 12: s11 ^= in_data[in_base + 11];
            case 11: s10 ^= in_data[in_base + 10];
            case 10: s9  ^= in_data[in_base +  9];
            case  9: s8  ^= in_data[in_base +  8];
            case  8: s7  ^= in_data[in_base +  7];
            case  7: s6  ^= in_data[in_base +  6];
            case  6: s5  ^= in_data[in_base +  5];
            case  5: s4  ^= in_data[in_base +  4];
            case  4: s3  ^= in_data[in_base +  3];
            case  3: s2  ^= in_data[in_base +  2];
            case  2: s1  ^= in_data[in_base +  1];
            case  1: s0  ^= in_data[in_base +  0]; break;
            default: break;
        }
        switch (msg_lanes) {
            case 24: s24 ^= domain_val; break;
            case 23: s23 ^= domain_val; break;
            case 22: s22 ^= domain_val; break;
            case 21: s21 ^= domain_val; break;
            case 20: s20 ^= domain_val; break;
            case 19: s19 ^= domain_val; break;
            case 18: s18 ^= domain_val; break;
            case 17: s17 ^= domain_val; break;
            case 16: s16 ^= domain_val; break;
            case 15: s15 ^= domain_val; break;
            case 14: s14 ^= domain_val; break;
            case 13: s13 ^= domain_val; break;
            case 12: s12 ^= domain_val; break;
            case 11: s11 ^= domain_val; break;
            case 10: s10 ^= domain_val; break;
            case  9: s9  ^= domain_val; break;
            case  8: s8  ^= domain_val; break;
            case  7: s7  ^= domain_val; break;
            case  6: s6  ^= domain_val; break;
            case  5: s5  ^= domain_val; break;
            case  4: s4  ^= domain_val; break;
            case  3: s3  ^= domain_val; break;
            case  2: s2  ^= domain_val; break;
            case  1: s1  ^= domain_val; break;
            case  0: s0  ^= domain_val; break;
            default: break;
        }
    }

    if (rate_lanes == 17) {
        s16 ^= 0x8000000000000000ul;
    } else if (rate_lanes == 21) {
        s20 ^= 0x8000000000000000ul;
    } else if (rate_lanes == 9) {
        s8 ^= 0x8000000000000000ul;
    } else if (rate_lanes == 13) {
        s12 ^= 0x8000000000000000ul;
    } else {
        switch (rate_lanes - 1) {
            case 24: s24 ^= 0x8000000000000000ul; break;
            case 23: s23 ^= 0x8000000000000000ul; break;
            case 22: s22 ^= 0x8000000000000000ul; break;
            case 21: s21 ^= 0x8000000000000000ul; break;
            case 20: s20 ^= 0x8000000000000000ul; break;
            case 19: s19 ^= 0x8000000000000000ul; break;
            case 18: s18 ^= 0x8000000000000000ul; break;
            case 17: s17 ^= 0x8000000000000000ul; break;
            case 16: s16 ^= 0x8000000000000000ul; break;
            case 15: s15 ^= 0x8000000000000000ul; break;
            case 14: s14 ^= 0x8000000000000000ul; break;
            case 13: s13 ^= 0x8000000000000000ul; break;
            case 12: s12 ^= 0x8000000000000000ul; break;
            case 11: s11 ^= 0x8000000000000000ul; break;
            case 10: s10 ^= 0x8000000000000000ul; break;
            case  9: s9  ^= 0x8000000000000000ul; break;
            case  8: s8  ^= 0x8000000000000000ul; break;
            case  7: s7  ^= 0x8000000000000000ul; break;
            case  6: s6  ^= 0x8000000000000000ul; break;
            case  5: s5  ^= 0x8000000000000000ul; break;
            case  4: s4  ^= 0x8000000000000000ul; break;
            case  3: s3  ^= 0x8000000000000000ul; break;
            case  2: s2  ^= 0x8000000000000000ul; break;
            case  1: s1  ^= 0x8000000000000000ul; break;
            case  0: s0  ^= 0x8000000000000000ul; break;
            default: break;
        }
    }

    uint out_base = idx * out_lanes;
    uint written = 0;

    for (;;) {
        // Apply exactly 24 straightline rounds
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

        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;

        if (take == 4 && written == 0 && out_lanes == 4) {
            ((device ulong4*)out_data)[idx] = ulong4(s0, s1, s2, s3);
        } else if (take == 8 && written == 0 && out_lanes == 8) {
            ((device ulong4*)out_data)[idx * 2 + 0] = ulong4(s0, s1, s2, s3);
            ((device ulong4*)out_data)[idx * 2 + 1] = ulong4(s4, s5, s6, s7);
        } else if (take == 16 && written == 0 && out_lanes == 16) {
            ((device ulong4*)out_data)[idx * 4 + 0] = ulong4(s0, s1, s2, s3);
            ((device ulong4*)out_data)[idx * 4 + 1] = ulong4(s4, s5, s6, s7);
            ((device ulong4*)out_data)[idx * 4 + 2] = ulong4(s8, s9, s10, s11);
            ((device ulong4*)out_data)[idx * 4 + 3] = ulong4(s12, s13, s14, s15);
        } else {
            switch (take) {
                case 25: out_data[out_base + written + 24] = s24;
                case 24: out_data[out_base + written + 23] = s23;
                case 23: out_data[out_base + written + 22] = s22;
                case 22: out_data[out_base + written + 21] = s21;
                case 21: out_data[out_base + written + 20] = s20;
                case 20: out_data[out_base + written + 19] = s19;
                case 19: out_data[out_base + written + 18] = s18;
                case 18: out_data[out_base + written + 17] = s17;
                case 17: out_data[out_base + written + 16] = s16;
                case 16: out_data[out_base + written + 15] = s15;
                case 15: out_data[out_base + written + 14] = s14;
                case 14: out_data[out_base + written + 13] = s13;
                case 13: out_data[out_base + written + 12] = s12;
                case 12: out_data[out_base + written + 11] = s11;
                case 11: out_data[out_base + written + 10] = s10;
                case 10: out_data[out_base + written +  9] = s9;
                case  9: out_data[out_base + written +  8] = s8;
                case  8: out_data[out_base + written +  7] = s7;
                case  7: out_data[out_base + written +  6] = s6;
                case  6: out_data[out_base + written +  5] = s5;
                case  5: out_data[out_base + written +  4] = s4;
                case  4: out_data[out_base + written +  3] = s3;
                case  3: out_data[out_base + written +  2] = s2;
                case  2: out_data[out_base + written +  1] = s1;
                case  1: out_data[out_base + written +  0] = s0; break;
                default: break;
            }
        }

        written += take;
        if (written >= out_lanes) break;
    }
}
```

Result of previous attempt:
     sha3_256_B16K: correct, 0.37 ms, 166.5 Gbitops/s (u64) (14.8% of 1125 Gops/s (u64 bitop, est))
    sha3_256_B256K: correct, 1.88 ms, 517.6 Gbitops/s (u64) (46.0% of 1125 Gops/s (u64 bitop, est))
      sha3_256_B4M: correct, 22.16 ms, 704.1 Gbitops/s (u64) (62.6% of 1125 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.3493

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

inline ulong rotl(ulong x, uint k) {
    return (x << k) | (x >> (64u - k));
}

inline void keccak_f1600(thread ulong *A) {
    ulong E[25];

    #pragma unroll 12
    for (uint r = 0; r < 24; r += 2) {
        // --- Round r: A -> E ---
        ulong C0 = A[0] ^ A[5] ^ A[10] ^ A[15] ^ A[20];
        ulong C1 = A[1] ^ A[6] ^ A[11] ^ A[16] ^ A[21];
        ulong C2 = A[2] ^ A[7] ^ A[12] ^ A[17] ^ A[22];
        ulong C3 = A[3] ^ A[8] ^ A[13] ^ A[18] ^ A[23];
        ulong C4 = A[4] ^ A[9] ^ A[14] ^ A[19] ^ A[24];

        ulong D0 = C4 ^ rotl(C1, 1u);
        ulong D1 = C0 ^ rotl(C2, 1u);
        ulong D2 = C1 ^ rotl(C3, 1u);
        ulong D3 = C2 ^ rotl(C4, 1u);
        ulong D4 = C3 ^ rotl(C0, 1u);

        ulong B0 = A[0] ^ D0;
        ulong B1 = rotl(A[6] ^ D1, 44u);
        ulong B2 = rotl(A[12] ^ D2, 43u);
        ulong B3 = rotl(A[18] ^ D3, 21u);
        ulong B4 = rotl(A[24] ^ D4, 14u);
        E[0] = B0 ^ (~B1 & B2) ^ KECCAK_RC[r];
        E[1] = B1 ^ (~B2 & B3);
        E[2] = B2 ^ (~B3 & B4);
        E[3] = B3 ^ (~B4 & B0);
        E[4] = B4 ^ (~B0 & B1);

        B0 = rotl(A[3] ^ D3, 28u);
        B1 = rotl(A[9] ^ D4, 20u);
        B2 = rotl(A[10] ^ D0, 3u);
        B3 = rotl(A[16] ^ D1, 45u);
        B4 = rotl(A[22] ^ D2, 61u);
        E[5] = B0 ^ (~B1 & B2);
        E[6] = B1 ^ (~B2 & B3);
        E[7] = B2 ^ (~B3 & B4);
        E[8] = B3 ^ (~B4 & B0);
        E[9] = B4 ^ (~B0 & B1);

        B0 = rotl(A[1] ^ D1, 1u);
        B1 = rotl(A[7] ^ D2, 6u);
        B2 = rotl(A[13] ^ D3, 25u);
        B3 = rotl(A[19] ^ D4, 8u);
        B4 = rotl(A[20] ^ D0, 18u);
        E[10] = B0 ^ (~B1 & B2);
        E[11] = B1 ^ (~B2 & B3);
        E[12] = B2 ^ (~B3 & B4);
        E[13] = B3 ^ (~B4 & B0);
        E[14] = B4 ^ (~B0 & B1);

        B0 = rotl(A[4] ^ D4, 27u);
        B1 = rotl(A[5] ^ D0, 36u);
        B2 = rotl(A[11] ^ D1, 10u);
        B3 = rotl(A[17] ^ D2, 15u);
        B4 = rotl(A[23] ^ D3, 56u);
        E[15] = B0 ^ (~B1 & B2);
        E[16] = B1 ^ (~B2 & B3);
        E[17] = B2 ^ (~B3 & B4);
        E[18] = B3 ^ (~B4 & B0);
        E[19] = B4 ^ (~B0 & B1);

        B0 = rotl(A[2] ^ D2, 62u);
        B1 = rotl(A[8] ^ D3, 55u);
        B2 = rotl(A[14] ^ D4, 39u);
        B3 = rotl(A[15] ^ D0, 41u);
        B4 = rotl(A[21] ^ D1, 2u);
        E[20] = B0 ^ (~B1 & B2);
        E[21] = B1 ^ (~B2 & B3);
        E[22] = B2 ^ (~B3 & B4);
        E[23] = B3 ^ (~B4 & B0);
        E[24] = B4 ^ (~B0 & B1);

        // --- Round r+1: E -> A ---
        C0 = E[0] ^ E[5] ^ E[10] ^ E[15] ^ E[20];
        C1 = E[1] ^ E[6] ^ E[11] ^ E[16] ^ E[21];
        C2 = E[2] ^ E[7] ^ E[12] ^ E[17] ^ E[22];
        C3 = E[3] ^ E[8] ^ E[13] ^ E[18] ^ E[23];
        C4 = E[4] ^ E[9] ^ E[14] ^ E[19] ^ E[24];

        D0 = C4 ^ rotl(C1, 1u);
        D1 = C0 ^ rotl(C2, 1u);
        D2 = C1 ^ rotl(C3, 1u);
        D3 = C2 ^ rotl(C4, 1u);
        D4 = C3 ^ rotl(C0, 1u);

        B0 = E[0] ^ D0;
        B1 = rotl(E[6] ^ D1, 44u);
        B2 = rotl(E[12] ^ D2, 43u);
        B3 = rotl(E[18] ^ D3, 21u);
        B4 = rotl(E[24] ^ D4, 14u);
        A[0] = B0 ^ (~B1 & B2) ^ KECCAK_RC[r+1];
        A[1] = B1 ^ (~B2 & B3);
        A[2] = B2 ^ (~B3 & B4);
        A[3] = B3 ^ (~B4 & B0);
        A[4] = B4 ^ (~B0 & B1);

        B0 = rotl(E[3] ^ D3, 28u);
        B1 = rotl(E[9] ^ D4, 20u);
        B2 = rotl(E[10] ^ D0, 3u);
        B3 = rotl(E[16] ^ D1, 45u);
        B4 = rotl(E[22] ^ D2, 61u);
        A[5] = B0 ^ (~B1 & B2);
        A[6] = B1 ^ (~B2 & B3);
        A[7] = B2 ^ (~B3 & B4);
        A[8] = B3 ^ (~B4 & B0);
        A[9] = B4 ^ (~B0 & B1);

        B0 = rotl(E[1] ^ D1, 1u);
        B1 = rotl(E[7] ^ D2, 6u);
        B2 = rotl(E[13] ^ D3, 25u);
        B3 = rotl(E[19] ^ D4, 8u);
        B4 = rotl(E[20] ^ D0, 18u);
        A[10] = B0 ^ (~B1 & B2);
        A[11] = B1 ^ (~B2 & B3);
        A[12] = B2 ^ (~B3 & B4);
        A[13] = B3 ^ (~B4 & B0);
        A[14] = B4 ^ (~B0 & B1);

        B0 = rotl(E[4] ^ D4, 27u);
        B1 = rotl(E[5] ^ D0, 36u);
        B2 = rotl(E[11] ^ D1, 10u);
        B3 = rotl(E[17] ^ D2, 15u);
        B4 = rotl(E[23] ^ D3, 56u);
        A[15] = B0 ^ (~B1 & B2);
        A[16] = B1 ^ (~B2 & B3);
        A[17] = B2 ^ (~B3 & B4);
        A[18] = B3 ^ (~B4 & B0);
        A[19] = B4 ^ (~B0 & B1);

        B0 = rotl(E[2] ^ D2, 62u);
        B1 = rotl(E[8] ^ D3, 55u);
        B2 = rotl(E[14] ^ D4, 39u);
        B3 = rotl(E[15] ^ D0, 41u);
        B4 = rotl(E[21] ^ D1, 2u);
        A[20] = B0 ^ (~B1 & B2);
        A[21] = B1 ^ (~B2 & B3);
        A[22] = B2 ^ (~B3 & B4);
        A[23] = B3 ^ (~B4 & B0);
        A[24] = B4 ^ (~B0 & B1);
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

    ulong A[25];

    uint msg_lanes  = msg_bytes  >> 3;
    uint rate_lanes = rate_bytes >> 3;
    uint out_lanes  = out_bytes  >> 3;
    ulong domain_val = (ulong)(domain & 0xFFu);

    // Common fast-path (vectorized 32-byte load and direct initialization)
    if (msg_lanes == 4) {
        ulong4 msg = ((device const ulong4*)in_data)[idx];
        A[0] = msg.x;
        A[1] = msg.y;
        A[2] = msg.z;
        A[3] = msg.w;
        A[4] = domain_val;
        
        #pragma unroll
        for (uint i = 5; i < 25; ++i) A[i] = 0;
    } else {
        #pragma unroll
        for (uint i = 0; i < 25; ++i) A[i] = 0;

        uint in_base = idx * msg_lanes;
        switch (msg_lanes) {
            case 24: A[23] ^= in_data[in_base + 23];
            case 23: A[22] ^= in_data[in_base + 22];
            case 22: A[21] ^= in_data[in_base + 21];
            case 21: A[20] ^= in_data[in_base + 20];
            case 20: A[19] ^= in_data[in_base + 19];
            case 19: A[18] ^= in_data[in_base + 18];
            case 18: A[17] ^= in_data[in_base + 17];
            case 17: A[16] ^= in_data[in_base + 16];
            case 16: A[15] ^= in_data[in_base + 15];
            case 15: A[14] ^= in_data[in_base + 14];
            case 14: A[13] ^= in_data[in_base + 13];
            case 13: A[12] ^= in_data[in_base + 12];
            case 12: A[11] ^= in_data[in_base + 11];
            case 11: A[10] ^= in_data[in_base + 10];
            case 10: A[ 9] ^= in_data[in_base +  9];
            case  9: A[ 8] ^= in_data[in_base +  8];
            case  8: A[ 7] ^= in_data[in_base +  7];
            case  7: A[ 6] ^= in_data[in_base +  6];
            case  6: A[ 5] ^= in_data[in_base +  5];
            case  5: A[ 4] ^= in_data[in_base +  4];
            case  4: A[ 3] ^= in_data[in_base +  3];
            case  3: A[ 2] ^= in_data[in_base +  2];
            case  2: A[ 1] ^= in_data[in_base +  1];
            case  1: A[ 0] ^= in_data[in_base +  0]; break;
            default: break;
        }
        switch (msg_lanes) {
            case 24: A[24] ^= domain_val; break;
            case 23: A[23] ^= domain_val; break;
            case 22: A[22] ^= domain_val; break;
            case 21: A[21] ^= domain_val; break;
            case 20: A[20] ^= domain_val; break;
            case 19: A[19] ^= domain_val; break;
            case 18: A[18] ^= domain_val; break;
            case 17: A[17] ^= domain_val; break;
            case 16: A[16] ^= domain_val; break;
            case 15: A[15] ^= domain_val; break;
            case 14: A[14] ^= domain_val; break;
            case 13: A[13] ^= domain_val; break;
            case 12: A[12] ^= domain_val; break;
            case 11: A[11] ^= domain_val; break;
            case 10: A[10] ^= domain_val; break;
            case  9: A[ 9] ^= domain_val; break;
            case  8: A[ 8] ^= domain_val; break;
            case  7: A[ 7] ^= domain_val; break;
            case  6: A[ 6] ^= domain_val; break;
            case  5: A[ 5] ^= domain_val; break;
            case  4: A[ 4] ^= domain_val; break;
            case  3: A[ 3] ^= domain_val; break;
            case  2: A[ 2] ^= domain_val; break;
            case  1: A[ 1] ^= domain_val; break;
            case  0: A[ 0] ^= domain_val; break;
            default: break;
        }
    }

    // Rate padding with explicit fast checks to bypass switch
    if (rate_lanes == 17) {
        A[16] ^= 0x8000000000000000ul;
    } else if (rate_lanes == 21) {
        A[20] ^= 0x8000000000000000ul;
    } else if (rate_lanes == 9) {
        A[8] ^= 0x8000000000000000ul;
    } else if (rate_lanes == 13) {
        A[12] ^= 0x8000000000000000ul;
    } else {
        switch (rate_lanes - 1) {
            case 24: A[24] ^= 0x8000000000000000ul; break;
            case 23: A[23] ^= 0x8000000000000000ul; break;
            case 22: A[22] ^= 0x8000000000000000ul; break;
            case 21: A[21] ^= 0x8000000000000000ul; break;
            case 20: A[20] ^= 0x8000000000000000ul; break;
            case 19: A[19] ^= 0x8000000000000000ul; break;
            case 18: A[18] ^= 0x8000000000000000ul; break;
            case 17: A[17] ^= 0x8000000000000000ul; break;
            case 16: A[16] ^= 0x8000000000000000ul; break;
            case 15: A[15] ^= 0x8000000000000000ul; break;
            case 14: A[14] ^= 0x8000000000000000ul; break;
            case 13: A[13] ^= 0x8000000000000000ul; break;
            case 12: A[12] ^= 0x8000000000000000ul; break;
            case 11: A[11] ^= 0x8000000000000000ul; break;
            case 10: A[10] ^= 0x8000000000000000ul; break;
            case  9: A[ 9] ^= 0x8000000000000000ul; break;
            case  8: A[ 8] ^= 0x8000000000000000ul; break;
            case  7: A[ 7] ^= 0x8000000000000000ul; break;
            case  6: A[ 6] ^= 0x8000000000000000ul; break;
            case  5: A[ 5] ^= 0x8000000000000000ul; break;
            case  4: A[ 4] ^= 0x8000000000000000ul; break;
            case  3: A[ 3] ^= 0x8000000000000000ul; break;
            case  2: A[ 2] ^= 0x8000000000000000ul; break;
            case  1: A[ 1] ^= 0x8000000000000000ul; break;
            case  0: A[ 0] ^= 0x8000000000000000ul; break;
            default: break;
        }
    }

    uint out_base = idx * out_lanes;
    uint written = 0;

    // Single unified extraction loop (avoids multiple inline clones of keccak_f1600)
    for (;;) {
        keccak_f1600(A);
        
        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        
        // Coalesced vectorized stores for exact multiples (32, 64, 128 bytes)
        if (take == 4 && written == 0 && out_lanes == 4) {
            ((device ulong4*)out_data)[idx] = { A[0], A[1], A[2], A[3] };
        } else if (take == 8 && written == 0 && out_lanes == 8) {
            ((device ulong4*)out_data)[idx * 2 + 0] = { A[0], A[1], A[2], A[3] };
            ((device ulong4*)out_data)[idx * 2 + 1] = { A[4], A[5], A[6], A[7] };
        } else if (take == 16 && written == 0 && out_lanes == 16) {
            ((device ulong4*)out_data)[idx * 4 + 0] = { A[0], A[1], A[2], A[3] };
            ((device ulong4*)out_data)[idx * 4 + 1] = { A[4], A[5], A[6], A[7] };
            ((device ulong4*)out_data)[idx * 4 + 2] = { A[8], A[9], A[10], A[11] };
            ((device ulong4*)out_data)[idx * 4 + 3] = { A[12], A[13], A[14], A[15] };
        } else {
            // Register-safe writeback using switch fall-through
            switch (take) {
                case 25: out_data[out_base + written + 24] = A[24];
                case 24: out_data[out_base + written + 23] = A[23];
                case 23: out_data[out_base + written + 22] = A[22];
                case 22: out_data[out_base + written + 21] = A[21];
                case 21: out_data[out_base + written + 20] = A[20];
                case 20: out_data[out_base + written + 19] = A[19];
                case 19: out_data[out_base + written + 18] = A[18];
                case 18: out_data[out_base + written + 17] = A[17];
                case 17: out_data[out_base + written + 16] = A[16];
                case 16: out_data[out_base + written + 15] = A[15];
                case 15: out_data[out_base + written + 14] = A[14];
                case 14: out_data[out_base + written + 13] = A[13];
                case 13: out_data[out_base + written + 12] = A[12];
                case 12: out_data[out_base + written + 11] = A[11];
                case 11: out_data[out_base + written + 10] = A[10];
                case 10: out_data[out_base + written +  9] = A[ 9];
                case  9: out_data[out_base + written +  8] = A[ 8];
                case  8: out_data[out_base + written +  7] = A[ 7];
                case  7: out_data[out_base + written +  6] = A[ 6];
                case  6: out_data[out_base + written +  5] = A[ 5];
                case  5: out_data[out_base + written +  4] = A[ 4];
                case  4: out_data[out_base + written +  3] = A[ 3];
                case  3: out_data[out_base + written +  2] = A[ 2];
                case  2: out_data[out_base + written +  1] = A[ 1];
                case  1: out_data[out_base + written +  0] = A[ 0]; break;
                default: break;
            }
        }

        written += take;
        if (written >= out_lanes) break;
    }
}
```

Incumbent result:
     sha3_256_B16K: correct, 0.37 ms, 164.8 Gbitops/s (u64) (14.7% of 1125 Gops/s (u64 bitop, est))
    sha3_256_B256K: correct, 1.41 ms, 690.6 Gbitops/s (u64) (61.4% of 1125 Gops/s (u64 bitop, est))
      sha3_256_B4M: correct, 21.88 ms, 713.2 Gbitops/s (u64) (63.4% of 1125 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.3849

## History

- iter  2: compile=OK | correct=True | score=0.37367744092898847
- iter  3: compile=OK | correct=True | score=0.34766545781404606
- iter  4: compile=OK | correct=True | score=0.3517096437888823
- iter  5: compile=OK | correct=True | score=0.3734132102472631
- iter  6: compile=OK | correct=True | score=0.29594792770962075
- iter  7: compile=OK | correct=True | score=0.34900504924473275
- iter  8: compile=OK | correct=True | score=0.3848960643922841
- iter  9: compile=OK | correct=True | score=0.3492773819019227

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
