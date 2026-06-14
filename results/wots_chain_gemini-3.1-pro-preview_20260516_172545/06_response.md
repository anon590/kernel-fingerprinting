I am applying an algorithmic optimization: explicitly truncating the computation of dead state lanes in the 24th round of Keccak for `n_bytes == 16` and `n_bytes == 32`.

WOTS+ chaining only feeds the truncated `n_bytes` digest into the next chain step; the rest of the capacity and rate lanes are zeroed/overwritten. However, because `A` is mutated across loop boundaries, LLVM's alias analysis often fails to apply Dead Store Elimination (DSE) through the loop backedge. By specializing the final round to compute only `A[0..1]` (for 16 bytes) or `A[0..3]` (for 32 bytes), we structurally omit computing unused outputs in `C`, `D`, `B`, and `A`. This removes ~45 logical operations per Keccak block without relying on the compiler, safely improving throughput over the incumbent's full 24-round execution.

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant ulong SHA3_DOMAIN_WORD  = 0x06ul;
constexpr constant ulong SHA3_FINAL_PAD    = 0x8000000000000000ul;

constant ulong KECCAK_RC[24] = {
    0x0000000000000001ul, 0x0000000000008082ul, 0x800000000000808Aul,
    0x8000000080008000ul, 0x000000000000808Bul, 0x0000000080000001ul,
    0x8000000080008081ul, 0x8000000000008009ul, 0x000000000000008Aul,
    0x0000000000000088ul, 0x0000000080008009ul, 0x000000008000000Aul,
    0x000000008000808Bul, 0x800000000000008Bul, 0x8000000000008089ul,
    0x8000000000008003ul, 0x8000000000008002ul, 0x8000000000000080ul,
    0x000000000000800Aul, 0x800000008000000Aul, 0x8000000080008081ul,
    0x8000000000008080ul, 0x0000000080000001ul, 0x8000000080008008ul
};

inline ulong rotl64_opt(ulong x, uint k) {
    if (k == 0u) return x;
    return (x << k) | (x >> (64u - k));
}

inline __attribute__((always_inline))
void keccak_f1600_23_rounds(thread ulong (&A)[25]) {
    #pragma unroll
    for (uint r = 0; r < 23; ++r) {
        ulong C[5];
        C[0] = A[0] ^ A[5] ^ A[10] ^ A[15] ^ A[20];
        C[1] = A[1] ^ A[6] ^ A[11] ^ A[16] ^ A[21];
        C[2] = A[2] ^ A[7] ^ A[12] ^ A[17] ^ A[22];
        C[3] = A[3] ^ A[8] ^ A[13] ^ A[18] ^ A[23];
        C[4] = A[4] ^ A[9] ^ A[14] ^ A[19] ^ A[24];

        ulong D[5];
        D[0] = C[4] ^ rotl64_opt(C[1], 1u);
        D[1] = C[0] ^ rotl64_opt(C[2], 1u);
        D[2] = C[1] ^ rotl64_opt(C[3], 1u);
        D[3] = C[2] ^ rotl64_opt(C[4], 1u);
        D[4] = C[3] ^ rotl64_opt(C[0], 1u);

        ulong B[25];
        B[ 0] = A[ 0] ^ D[0];
        B[10] = rotl64_opt(A[ 1] ^ D[1],  1u);
        B[20] = rotl64_opt(A[ 2] ^ D[2], 62u);
        B[ 5] = rotl64_opt(A[ 3] ^ D[3], 28u);
        B[15] = rotl64_opt(A[ 4] ^ D[4], 27u);

        B[16] = rotl64_opt(A[ 5] ^ D[0], 36u);
        B[ 1] = rotl64_opt(A[ 6] ^ D[1], 44u);
        B[11] = rotl64_opt(A[ 7] ^ D[2],  6u);
        B[21] = rotl64_opt(A[ 8] ^ D[3], 55u);
        B[ 6] = rotl64_opt(A[ 9] ^ D[4], 20u);

        B[ 7] = rotl64_opt(A[10] ^ D[0],  3u);
        B[17] = rotl64_opt(A[11] ^ D[1], 10u);
        B[ 2] = rotl64_opt(A[12] ^ D[2], 43u);
        B[12] = rotl64_opt(A[13] ^ D[3], 25u);
        B[22] = rotl64_opt(A[14] ^ D[4], 39u);

        B[23] = rotl64_opt(A[15] ^ D[0], 41u);
        B[ 8] = rotl64_opt(A[16] ^ D[1], 45u);
        B[18] = rotl64_opt(A[17] ^ D[2], 15u);
        B[ 3] = rotl64_opt(A[18] ^ D[3], 21u);
        B[13] = rotl64_opt(A[19] ^ D[4],  8u);

        B[14] = rotl64_opt(A[20] ^ D[0], 18u);
        B[24] = rotl64_opt(A[21] ^ D[1],  2u);
        B[ 9] = rotl64_opt(A[22] ^ D[2], 61u);
        B[19] = rotl64_opt(A[23] ^ D[3], 56u);
        B[ 4] = rotl64_opt(A[24] ^ D[4], 14u);

        #pragma unroll
        for (uint y = 0; y < 25; y += 5) {
            A[y + 0] = B[y + 0] ^ (~B[y + 1] & B[y + 2]);
            A[y + 1] = B[y + 1] ^ (~B[y + 2] & B[y + 3]);
            A[y + 2] = B[y + 2] ^ (~B[y + 3] & B[y + 4]);
            A[y + 3] = B[y + 3] ^ (~B[y + 4] & B[y + 0]);
            A[y + 4] = B[y + 4] ^ (~B[y + 0] & B[y + 1]);
        }

        A[0] ^= KECCAK_RC[r];
    }
}

inline __attribute__((always_inline))
void keccak_f1600_16(thread ulong (&A)[25]) {
    keccak_f1600_23_rounds(A);

    // Round 23: Truncate computation, we only need A[0] and A[1]
    ulong C[5];
    C[0] = A[0] ^ A[5] ^ A[10] ^ A[15] ^ A[20];
    C[1] = A[1] ^ A[6] ^ A[11] ^ A[16] ^ A[21];
    C[2] = A[2] ^ A[7] ^ A[12] ^ A[17] ^ A[22];
    C[3] = A[3] ^ A[8] ^ A[13] ^ A[18] ^ A[23];
    C[4] = A[4] ^ A[9] ^ A[14] ^ A[19] ^ A[24];

    ulong D[4];
    D[0] = C[4] ^ rotl64_opt(C[1], 1u);
    D[1] = C[0] ^ rotl64_opt(C[2], 1u);
    D[2] = C[1] ^ rotl64_opt(C[3], 1u);
    D[3] = C[2] ^ rotl64_opt(C[4], 1u);

    ulong B0 = A[0] ^ D[0];
    ulong B1 = rotl64_opt(A[6] ^ D[1], 44u);
    ulong B2 = rotl64_opt(A[12] ^ D[2], 43u);
    ulong B3 = rotl64_opt(A[18] ^ D[3], 21u);

    A[0] = (B0 ^ (~B1 & B2)) ^ 0x8000000080008008ul; // KECCAK_RC[23]
    A[1] = B1 ^ (~B2 & B3);
}

inline __attribute__((always_inline))
void keccak_f1600_32(thread ulong (&A)[25]) {
    keccak_f1600_23_rounds(A);

    // Round 23: Truncate computation, we only need A[0..3]
    ulong C[5];
    C[0] = A[0] ^ A[5] ^ A[10] ^ A[15] ^ A[20];
    C[1] = A[1] ^ A[6] ^ A[11] ^ A[16] ^ A[21];
    C[2] = A[2] ^ A[7] ^ A[12] ^ A[17] ^ A[22];
    C[3] = A[3] ^ A[8] ^ A[13] ^ A[18] ^ A[23];
    C[4] = A[4] ^ A[9] ^ A[14] ^ A[19] ^ A[24];

    ulong D[5];
    D[0] = C[4] ^ rotl64_opt(C[1], 1u);
    D[1] = C[0] ^ rotl64_opt(C[2], 1u);
    D[2] = C[1] ^ rotl64_opt(C[3], 1u);
    D[3] = C[2] ^ rotl64_opt(C[4], 1u);
    D[4] = C[3] ^ rotl64_opt(C[0], 1u);

    ulong B0 = A[0] ^ D[0];
    ulong B1 = rotl64_opt(A[6] ^ D[1], 44u);
    ulong B2 = rotl64_opt(A[12] ^ D[2], 43u);
    ulong B3 = rotl64_opt(A[18] ^ D[3], 21u);
    ulong B4 = rotl64_opt(A[24] ^ D[4], 14u);

    A[0] = (B0 ^ (~B1 & B2)) ^ 0x8000000080008008ul; // KECCAK_RC[23]
    A[1] = B1 ^ (~B2 & B3);
    A[2] = B2 ^ (~B3 & B4);
    A[3] = B3 ^ (~B4 & B0);
}

inline __attribute__((always_inline))
void keccak_f1600_full(thread ulong (&A)[25]) {
    keccak_f1600_23_rounds(A);

    // Round 23 full
    ulong C[5];
    C[0] = A[0] ^ A[5] ^ A[10] ^ A[15] ^ A[20];
    C[1] = A[1] ^ A[6] ^ A[11] ^ A[16] ^ A[21];
    C[2] = A[2] ^ A[7] ^ A[12] ^ A[17] ^ A[22];
    C[3] = A[3] ^ A[8] ^ A[13] ^ A[18] ^ A[23];
    C[4] = A[4] ^ A[9] ^ A[14] ^ A[19] ^ A[24];

    ulong D[5];
    D[0] = C[4] ^ rotl64_opt(C[1], 1u);
    D[1] = C[0] ^ rotl64_opt(C[2], 1u);
    D[2] = C[1] ^ rotl64_opt(C[3], 1u);
    D[3] = C[2] ^ rotl64_opt(C[4], 1u);
    D[4] = C[3] ^ rotl64_opt(C[0], 1u);

    ulong B[25];
    B[ 0] = A[ 0] ^ D[0];
    B[10] = rotl64_opt(A[ 1] ^ D[1],  1u);
    B[20] = rotl64_opt(A[ 2] ^ D[2], 62u);
    B[ 5] = rotl64_opt(A[ 3] ^ D[3], 28u);
    B[15] = rotl64_opt(A[ 4] ^ D[4], 27u);

    B[16] = rotl64_opt(A[ 5] ^ D[0], 36u);
    B[ 1] = rotl64_opt(A[ 6] ^ D[1], 44u);
    B[11] = rotl64_opt(A[ 7] ^ D[2],  6u);
    B[21] = rotl64_opt(A[ 8] ^ D[3], 55u);
    B[ 6] = rotl64_opt(A[ 9] ^ D[4], 20u);

    B[ 7] = rotl64_opt(A[10] ^ D[0],  3u);
    B[17] = rotl64_opt(A[11] ^ D[1], 10u);
    B[ 2] = rotl64_opt(A[12] ^ D[2], 43u);
    B[12] = rotl64_opt(A[13] ^ D[3], 25u);
    B[22] = rotl64_opt(A[14] ^ D[4], 39u);

    B[23] = rotl64_opt(A[15] ^ D[0], 41u);
    B[ 8] = rotl64_opt(A[16] ^ D[1], 45u);
    B[18] = rotl64_opt(A[17] ^ D[2], 15u);
    B[ 3] = rotl64_opt(A[18] ^ D[3], 21u);
    B[13] = rotl64_opt(A[19] ^ D[4],  8u);

    B[14] = rotl64_opt(A[20] ^ D[0], 18u);
    B[24] = rotl64_opt(A[21] ^ D[1],  2u);
    B[ 9] = rotl64_opt(A[22] ^ D[2], 61u);
    B[19] = rotl64_opt(A[23] ^ D[3], 56u);
    B[ 4] = rotl64_opt(A[24] ^ D[4], 14u);

    #pragma unroll
    for (uint y = 0; y < 25; y += 5) {
        A[y + 0] = B[y + 0] ^ (~B[y + 1] & B[y + 2]);
        A[y + 1] = B[y + 1] ^ (~B[y + 2] & B[y + 3]);
        A[y + 2] = B[y + 2] ^ (~B[y + 3] & B[y + 4]);
        A[y + 3] = B[y + 3] ^ (~B[y + 4] & B[y + 0]);
        A[y + 4] = B[y + 4] ^ (~B[y + 0] & B[y + 1]);
    }

    A[0] ^= 0x8000000080008008ul; // KECCAK_RC[23]
}

kernel void wots_chain(
    device const ulong *seeds    [[buffer(0)]],
    device       ulong *tips     [[buffer(1)]],
    constant uint      &n_chains [[buffer(2)]],
    constant uint      &n_bytes  [[buffer(3)]],
    constant uint      &w        [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n_chains) return;
    
    uint chain_len = w;

    if (n_bytes == 16) {
        ulong A[25];
        uint base = idx * 2;
        
        A[0] = seeds[base + 0];
        A[1] = seeds[base + 1];
        
        for (uint step = 0; step < chain_len; ++step) {
            #pragma unroll
            for (uint i = 2; i < 25; ++i) A[i] = 0ul;
            A[2] = SHA3_DOMAIN_WORD;
            A[16] = SHA3_FINAL_PAD;
            keccak_f1600_16(A);
        }
        
        tips[base + 0] = A[0];
        tips[base + 1] = A[1];
        
    } else if (n_bytes == 32) {
        ulong A[25];
        uint base = idx * 4;
        
        A[0] = seeds[base + 0];
        A[1] = seeds[base + 1];
        A[2] = seeds[base + 2];
        A[3] = seeds[base + 3];
        
        for (uint step = 0; step < chain_len; ++step) {
            #pragma unroll
            for (uint i = 4; i < 25; ++i) A[i] = 0ul;
            A[4] = SHA3_DOMAIN_WORD;
            A[16] = SHA3_FINAL_PAD;
            keccak_f1600_32(A);
        }
        
        tips[base + 0] = A[0];
        tips[base + 1] = A[1];
        tips[base + 2] = A[2];
        tips[base + 3] = A[3];
        
    } else {
        uint n_lanes = n_bytes >> 3;
        ulong A[25];
        uint base = idx * n_lanes;
        
        for (uint k = 0; k < n_lanes; ++k) {
            A[k] = seeds[base + k];
        }
        
        for (uint step = 0; step < chain_len; ++step) {
            for (uint i = n_lanes; i < 25; ++i) A[i] = 0ul;
            A[n_lanes] ^= SHA3_DOMAIN_WORD;
            A[16] ^= SHA3_FINAL_PAD;
            keccak_f1600_full(A);
        }
        
        for (uint k = 0; k < n_lanes; ++k) {
            tips[base + k] = A[k];
        }
    }
}
```