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

struct KeccakState {
    ulong A00, A10, A20, A30, A40;
    ulong A01, A11, A21, A31, A41;
    ulong A02, A12, A22, A32, A42;
    ulong A03, A13, A23, A33, A43;
    ulong A04, A14, A24, A34, A44;
};

inline void keccak_f1600(thread KeccakState &S) {
    constexpr ulong RC[24] = {
        0x0000000000000001ul, 0x0000000000008082ul, 0x800000000000808Aul,
        0x8000000080008000ul, 0x000000000000808Bul, 0x0000000080000001ul,
        0x8000000080008081ul, 0x8000000000008009ul, 0x000000000000008Aul,
        0x0000000000000088ul, 0x0000000080008009ul, 0x000000008000000Aul,
        0x000000008000808Bul, 0x800000000000008Bul, 0x8000000000008089ul,
        0x8000000000008003ul, 0x8000000000008002ul, 0x8000000000000080ul,
        0x000000000000800Aul, 0x800000008000000Aul, 0x8000000080008081ul,
        0x8000000000008080ul, 0x0000000080000001ul, 0x8000000080008008ul,
    };

    #pragma unroll 24
    for (int r = 0; r < 24; ++r) {
        ulong C0 = S.A00 ^ S.A01 ^ S.A02 ^ S.A03 ^ S.A04;
        ulong C1 = S.A10 ^ S.A11 ^ S.A12 ^ S.A13 ^ S.A14;
        ulong C2 = S.A20 ^ S.A21 ^ S.A22 ^ S.A23 ^ S.A24;
        ulong C3 = S.A30 ^ S.A31 ^ S.A32 ^ S.A33 ^ S.A34;
        ulong C4 = S.A40 ^ S.A41 ^ S.A42 ^ S.A43 ^ S.A44;

        ulong D0 = C4 ^ rotl(C1, 1u);
        ulong D1 = C0 ^ rotl(C2, 1u);
        ulong D2 = C1 ^ rotl(C3, 1u);
        ulong D3 = C2 ^ rotl(C4, 1u);
        ulong D4 = C3 ^ rotl(C0, 1u);

        ulong B00 = S.A00 ^ D0;
        ulong B01 = rotl(S.A11 ^ D1, 44u);
        ulong B02 = rotl(S.A22 ^ D2, 43u);
        ulong B03 = rotl(S.A33 ^ D3, 21u);
        ulong B04 = rotl(S.A44 ^ D4, 14u);

        ulong B10 = rotl(S.A30 ^ D3, 28u);
        ulong B11 = rotl(S.A41 ^ D4, 20u);
        ulong B12 = rotl(S.A02 ^ D0, 3u);
        ulong B13 = rotl(S.A13 ^ D1, 45u);
        ulong B14 = rotl(S.A24 ^ D2, 61u);

        ulong B20 = rotl(S.A10 ^ D1, 1u);
        ulong B21 = rotl(S.A21 ^ D2, 6u);
        ulong B22 = rotl(S.A32 ^ D3, 25u);
        ulong B23 = rotl(S.A43 ^ D4, 8u);
        ulong B24 = rotl(S.A04 ^ D0, 18u);

        ulong B30 = rotl(S.A40 ^ D4, 27u);
        ulong B31 = rotl(S.A01 ^ D0, 36u);
        ulong B32 = rotl(S.A12 ^ D1, 10u);
        ulong B33 = rotl(S.A23 ^ D2, 15u);
        ulong B34 = rotl(S.A34 ^ D3, 56u);

        ulong B40 = rotl(S.A20 ^ D2, 62u);
        ulong B41 = rotl(S.A31 ^ D3, 55u);
        ulong B42 = rotl(S.A42 ^ D4, 39u);
        ulong B43 = rotl(S.A03 ^ D0, 41u);
        ulong B44 = rotl(S.A14 ^ D1, 2u);

        S.A00 = B00 ^ (~B01 & B02) ^ RC[r];
        S.A10 = B01 ^ (~B02 & B03);
        S.A20 = B02 ^ (~B03 & B04);
        S.A30 = B03 ^ (~B04 & B00);
        S.A40 = B04 ^ (~B00 & B01);

        S.A01 = B10 ^ (~B11 & B12);
        S.A11 = B11 ^ (~B12 & B13);
        S.A21 = B12 ^ (~B13 & B14);
        S.A31 = B13 ^ (~B14 & B10);
        S.A41 = B14 ^ (~B10 & B11);

        S.A02 = B20 ^ (~B21 & B22);
        S.A12 = B21 ^ (~B22 & B23);
        S.A22 = B22 ^ (~B23 & B24);
        S.A32 = B23 ^ (~B24 & B20);
        S.A42 = B24 ^ (~B20 & B21);

        S.A03 = B30 ^ (~B31 & B32);
        S.A13 = B31 ^ (~B32 & B33);
        S.A23 = B32 ^ (~B33 & B34);
        S.A33 = B33 ^ (~B34 & B30);
        S.A43 = B34 ^ (~B30 & B31);

        S.A04 = B40 ^ (~B41 & B42);
        S.A14 = B41 ^ (~B42 & B43);
        S.A24 = B42 ^ (~B43 & B44);
        S.A34 = B43 ^ (~B44 & B40);
        S.A44 = B44 ^ (~B40 & B41);
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

    KeccakState S;
    S.A00 = 0; S.A10 = 0; S.A20 = 0; S.A30 = 0; S.A40 = 0;
    S.A01 = 0; S.A11 = 0; S.A21 = 0; S.A31 = 0; S.A41 = 0;
    S.A02 = 0; S.A12 = 0; S.A22 = 0; S.A32 = 0; S.A42 = 0;
    S.A03 = 0; S.A13 = 0; S.A23 = 0; S.A33 = 0; S.A43 = 0;
    S.A04 = 0; S.A14 = 0; S.A24 = 0; S.A34 = 0; S.A44 = 0;

    uint msg_lanes  = msg_bytes  >> 3;
    uint rate_lanes = rate_bytes >> 3;
    uint out_lanes  = out_bytes  >> 3;

    uint in_base = idx * msg_lanes;
    ulong domain_val = (ulong)(domain & 0xFFu);

    if (msg_lanes == 4) {
        S.A00 ^= in_data[in_base + 0];
        S.A10 ^= in_data[in_base + 1];
        S.A20 ^= in_data[in_base + 2];
        S.A30 ^= in_data[in_base + 3];
        S.A40 ^= domain_val;
    } else {
        switch (msg_lanes) {
            case 24: S.A34 ^= in_data[in_base + 23];
            case 23: S.A24 ^= in_data[in_base + 22];
            case 22: S.A14 ^= in_data[in_base + 21];
            case 21: S.A04 ^= in_data[in_base + 20];
            case 20: S.A43 ^= in_data[in_base + 19];
            case 19: S.A33 ^= in_data[in_base + 18];
            case 18: S.A23 ^= in_data[in_base + 17];
            case 17: S.A13 ^= in_data[in_base + 16];
            case 16: S.A03 ^= in_data[in_base + 15];
            case 15: S.A42 ^= in_data[in_base + 14];
            case 14: S.A32 ^= in_data[in_base + 13];
            case 13: S.A22 ^= in_data[in_base + 12];
            case 12: S.A12 ^= in_data[in_base + 11];
            case 11: S.A02 ^= in_data[in_base + 10];
            case 10: S.A41 ^= in_data[in_base +  9];
            case  9: S.A31 ^= in_data[in_base +  8];
            case  8: S.A21 ^= in_data[in_base +  7];
            case  7: S.A11 ^= in_data[in_base +  6];
            case  6: S.A01 ^= in_data[in_base +  5];
            case  5: S.A40 ^= in_data[in_base +  4];
            case  4: S.A30 ^= in_data[in_base +  3];
            case  3: S.A20 ^= in_data[in_base +  2];
            case  2: S.A10 ^= in_data[in_base +  1];
            case  1: S.A00 ^= in_data[in_base +  0]; break;
            default: break;
        }
        switch (msg_lanes) {
            case 24: S.A44 ^= domain_val; break;
            case 23: S.A34 ^= domain_val; break;
            case 22: S.A24 ^= domain_val; break;
            case 21: S.A14 ^= domain_val; break;
            case 20: S.A04 ^= domain_val; break;
            case 19: S.A43 ^= domain_val; break;
            case 18: S.A33 ^= domain_val; break;
            case 17: S.A23 ^= domain_val; break;
            case 16: S.A13 ^= domain_val; break;
            case 15: S.A03 ^= domain_val; break;
            case 14: S.A42 ^= domain_val; break;
            case 13: S.A32 ^= domain_val; break;
            case 12: S.A22 ^= domain_val; break;
            case 11: S.A12 ^= domain_val; break;
            case 10: S.A02 ^= domain_val; break;
            case  9: S.A41 ^= domain_val; break;
            case  8: S.A31 ^= domain_val; break;
            case  7: S.A21 ^= domain_val; break;
            case  6: S.A11 ^= domain_val; break;
            case  5: S.A01 ^= domain_val; break;
            case  4: S.A40 ^= domain_val; break;
            case  3: S.A30 ^= domain_val; break;
            case  2: S.A20 ^= domain_val; break;
            case  1: S.A10 ^= domain_val; break;
            case  0: S.A00 ^= domain_val; break;
            default: break;
        }
    }

    if (rate_lanes == 17) {
        S.A13 ^= 0x8000000000000000ul;
    } else if (rate_lanes == 21) {
        S.A04 ^= 0x8000000000000000ul;
    } else {
        switch (rate_lanes - 1) {
            case 24: S.A44 ^= 0x8000000000000000ul; break;
            case 23: S.A34 ^= 0x8000000000000000ul; break;
            case 22: S.A24 ^= 0x8000000000000000ul; break;
            case 21: S.A14 ^= 0x8000000000000000ul; break;
            case 20: S.A04 ^= 0x8000000000000000ul; break;
            case 19: S.A43 ^= 0x8000000000000000ul; break;
            case 18: S.A33 ^= 0x8000000000000000ul; break;
            case 17: S.A23 ^= 0x8000000000000000ul; break;
            case 16: S.A13 ^= 0x8000000000000000ul; break;
            case 15: S.A03 ^= 0x8000000000000000ul; break;
            case 14: S.A42 ^= 0x8000000000000000ul; break;
            case 13: S.A32 ^= 0x8000000000000000ul; break;
            case 12: S.A22 ^= 0x8000000000000000ul; break;
            case 11: S.A12 ^= 0x8000000000000000ul; break;
            case 10: S.A02 ^= 0x8000000000000000ul; break;
            case  9: S.A41 ^= 0x8000000000000000ul; break;
            case  8: S.A31 ^= 0x8000000000000000ul; break;
            case  7: S.A21 ^= 0x8000000000000000ul; break;
            case  6: S.A11 ^= 0x8000000000000000ul; break;
            case  5: S.A01 ^= 0x8000000000000000ul; break;
            case  4: S.A40 ^= 0x8000000000000000ul; break;
            case  3: S.A30 ^= 0x8000000000000000ul; break;
            case  2: S.A20 ^= 0x8000000000000000ul; break;
            case  1: S.A10 ^= 0x8000000000000000ul; break;
            case  0: S.A00 ^= 0x8000000000000000ul; break;
            default: break;
        }
    }

    uint out_base = idx * out_lanes;

    if (out_lanes == 4) {
        keccak_f1600(S);
        out_data[out_base + 0] = S.A00;
        out_data[out_base + 1] = S.A10;
        out_data[out_base + 2] = S.A20;
        out_data[out_base + 3] = S.A30;
    } else {
        uint written = 0;
        for (;;) {
            keccak_f1600(S);
            uint remaining = out_lanes - written;
            uint take = remaining < rate_lanes ? remaining : rate_lanes;
            
            switch (take) {
                case 25: out_data[out_base + written + 24] = S.A44;
                case 24: out_data[out_base + written + 23] = S.A34;
                case 23: out_data[out_base + written + 22] = S.A24;
                case 22: out_data[out_base + written + 21] = S.A14;
                case 21: out_data[out_base + written + 20] = S.A04;
                case 20: out_data[out_base + written + 19] = S.A43;
                case 19: out_data[out_base + written + 18] = S.A33;
                case 18: out_data[out_base + written + 17] = S.A23;
                case 17: out_data[out_base + written + 16] = S.A13;
                case 16: out_data[out_base + written + 15] = S.A03;
                case 15: out_data[out_base + written + 14] = S.A42;
                case 14: out_data[out_base + written + 13] = S.A32;
                case 13: out_data[out_base + written + 12] = S.A22;
                case 12: out_data[out_base + written + 11] = S.A12;
                case 11: out_data[out_base + written + 10] = S.A02;
                case 10: out_data[out_base + written +  9] = S.A41;
                case  9: out_data[out_base + written +  8] = S.A31;
                case  8: out_data[out_base + written +  7] = S.A21;
                case  7: out_data[out_base + written +  6] = S.A11;
                case  6: out_data[out_base + written +  5] = S.A01;
                case  5: out_data[out_base + written +  4] = S.A40;
                case  4: out_data[out_base + written +  3] = S.A30;
                case  3: out_data[out_base + written +  2] = S.A20;
                case  2: out_data[out_base + written +  1] = S.A10;
                case  1: out_data[out_base + written +  0] = S.A00; break;
                default: break;
            }

            written += take;
            if (written >= out_lanes) break;
        }
    }
}
```

Result of previous attempt:
     sha3_256_B16K: correct, 0.38 ms, 160.7 Gbitops/s (u64) (14.3% of 1125 Gops/s (u64 bitop, est))
    sha3_256_B256K: correct, 1.85 ms, 526.3 Gbitops/s (u64) (46.8% of 1125 Gops/s (u64 bitop, est))
      sha3_256_B4M: correct, 21.80 ms, 715.8 Gbitops/s (u64) (63.6% of 1125 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.3490

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

inline ulong rotl(ulong x, uint k) {
    return (x << k) | (x >> (64u - k));
}

inline void keccak_f1600(thread ulong *A) {
    constexpr ulong RC[24] = {
        0x0000000000000001ul, 0x0000000000008082ul, 0x800000000000808Aul,
        0x8000000080008000ul, 0x000000000000808Bul, 0x0000000080000001ul,
        0x8000000080008081ul, 0x8000000000008009ul, 0x000000000000008Aul,
        0x0000000000000088ul, 0x0000000080008009ul, 0x000000008000000Aul,
        0x000000008000808Bul, 0x800000000000008Bul, 0x8000000000008089ul,
        0x8000000000008003ul, 0x8000000000008002ul, 0x8000000000000080ul,
        0x000000000000800Aul, 0x800000008000000Aul, 0x8000000080008081ul,
        0x8000000000008080ul, 0x0000000080000001ul, 0x8000000080008008ul,
    };

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
        E[0] = B0 ^ (~B1 & B2) ^ RC[r];
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
        A[0] = B0 ^ (~B1 & B2) ^ RC[r+1];
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
    #pragma unroll
    for (uint i = 0; i < 25; ++i) {
        A[i] = 0;
    }

    uint msg_lanes  = msg_bytes  >> 3;
    uint rate_lanes = rate_bytes >> 3;
    uint out_lanes  = out_bytes  >> 3;

    uint in_base = idx * msg_lanes;
    ulong domain_val = (ulong)(domain & 0xFFu);

    // Fast-path for common input length
    if (msg_lanes == 4) {
        A[0] ^= in_data[in_base + 0];
        A[1] ^= in_data[in_base + 1];
        A[2] ^= in_data[in_base + 2];
        A[3] ^= in_data[in_base + 3];
        A[4] ^= domain_val;
    } else {
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

    // Fast-paths for common SHA3-256 (rate_lanes=17) and SHAKE-128 (rate_lanes=21) paddings
    if (rate_lanes == 17) {
        A[16] ^= 0x8000000000000000ul;
    } else if (rate_lanes == 21) {
        A[20] ^= 0x8000000000000000ul;
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

    // Fast-path for standard 32-byte outputs
    if (out_lanes == 4) {
        keccak_f1600(A);
        out_data[out_base + 0] = A[0];
        out_data[out_base + 1] = A[1];
        out_data[out_base + 2] = A[2];
        out_data[out_base + 3] = A[3];
    } else {
        uint written = 0;
        for (;;) {
            keccak_f1600(A);
            uint remaining = out_lanes - written;
            uint take = remaining < rate_lanes ? remaining : rate_lanes;
            
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

            written += take;
            if (written >= out_lanes) break;
        }
    }
}
```

Incumbent result:
     sha3_256_B16K: correct, 0.31 ms, 199.7 Gbitops/s (u64) (17.8% of 1125 Gops/s (u64 bitop, est))
    sha3_256_B256K: correct, 1.86 ms, 525.1 Gbitops/s (u64) (46.7% of 1125 Gops/s (u64 bitop, est))
      sha3_256_B4M: correct, 22.02 ms, 708.5 Gbitops/s (u64) (63.0% of 1125 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.3737

## History

- iter  0: compile=OK | correct=True | score=0.03916592497170866
- iter  1: compile=OK | correct=True | score=0.3300328432407895
- iter  2: compile=OK | correct=True | score=0.37367744092898847
- iter  3: compile=OK | correct=True | score=0.34766545781404606
- iter  4: compile=OK | correct=True | score=0.3517096437888823
- iter  5: compile=OK | correct=True | score=0.3734132102472631
- iter  6: compile=OK | correct=True | score=0.29594792770962075
- iter  7: compile=OK | correct=True | score=0.34900504924473275

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
