#include <metal_stdlib>
using namespace metal;

constexpr constant ulong KECCAK_RC[24] = {
    0x0000000000000001ul, 0x0000000000008082ul, 0x800000000000808Aul,
    0x8000000080008000ul, 0x000000000000808Bul, 0x0000000080000001ul,
    0x8000000080008081ul, 0x8000000000008009ul, 0x000000000000008Aul,
    0x0000000000000088ul, 0x0000000080008009ul, 0x000000008000000Aul,
    0x000000008000808Bul, 0x800000000000008Bul, 0x8000000000008089ul,
    0x8000000000008003ul, 0x8000000000008002ul, 0x8000000000000080ul,
    0x000000000000800Aul, 0x800000008000000Aul, 0x8000000080008081ul,
    0x8000000000008080ul, 0x0000000080000001ul, 0x8000000080008008ul,
};

inline __attribute__((always_inline)) ulong rotl64(ulong x, uint k) {
    return (x << k) | (x >> (64u - k));
}

inline __attribute__((always_inline)) void keccak_f1600(thread ulong *A) {
    #pragma clang loop unroll(full)
    for (uint r = 0u; r < 24u; ++r) {
        // Theta C
        ulong C[5];
        C[0] = A[0] ^ A[5] ^ A[10] ^ A[15] ^ A[20];
        C[1] = A[1] ^ A[6] ^ A[11] ^ A[16] ^ A[21];
        C[2] = A[2] ^ A[7] ^ A[12] ^ A[17] ^ A[22];
        C[3] = A[3] ^ A[8] ^ A[13] ^ A[18] ^ A[23];
        C[4] = A[4] ^ A[9] ^ A[14] ^ A[19] ^ A[24];

        // Theta D
        ulong D[5];
        D[0] = C[4] ^ rotl64(C[1], 1u);
        D[1] = C[0] ^ rotl64(C[2], 1u);
        D[2] = C[1] ^ rotl64(C[3], 1u);
        D[3] = C[2] ^ rotl64(C[4], 1u);
        D[4] = C[3] ^ rotl64(C[0], 1u);

        // Rho and Pi (Explicit parallel assignment mapping)
        ulong B[25];
        B[ 0] = A[ 0] ^ D[0];
        B[ 1] = rotl64(A[ 6] ^ D[1], 44u);
        B[ 2] = rotl64(A[12] ^ D[2], 43u);
        B[ 3] = rotl64(A[18] ^ D[3], 21u);
        B[ 4] = rotl64(A[24] ^ D[4], 14u);

        B[ 5] = rotl64(A[ 3] ^ D[3], 28u);
        B[ 6] = rotl64(A[ 9] ^ D[4], 20u);
        B[ 7] = rotl64(A[10] ^ D[0],  3u);
        B[ 8] = rotl64(A[16] ^ D[1], 45u);
        B[ 9] = rotl64(A[22] ^ D[2], 61u);

        B[10] = rotl64(A[ 1] ^ D[1],  1u);
        B[11] = rotl64(A[ 7] ^ D[2], 10u);
        B[12] = rotl64(A[13] ^ D[3], 25u);
        B[13] = rotl64(A[19] ^ D[4],  8u);
        B[14] = rotl64(A[20] ^ D[0], 18u);

        B[15] = rotl64(A[ 4] ^ D[4], 27u);
        B[16] = rotl64(A[ 5] ^ D[0], 36u);
        B[17] = rotl64(A[11] ^ D[1],  6u);
        B[18] = rotl64(A[17] ^ D[2], 15u);
        B[19] = rotl64(A[23] ^ D[3], 56u);

        B[20] = rotl64(A[ 2] ^ D[2], 62u);
        B[21] = rotl64(A[ 8] ^ D[3], 55u);
        B[22] = rotl64(A[14] ^ D[4], 39u);
        B[23] = rotl64(A[15] ^ D[0], 41u);
        B[24] = rotl64(A[21] ^ D[1],  2u);

        // Chi
        #pragma clang loop unroll(full)
        for (uint y = 0u; y < 5u; ++y) {
            uint row = 5u * y;
            A[0 + row] = B[0 + row] ^ (~B[1 + row] & B[2 + row]);
            A[1 + row] = B[1 + row] ^ (~B[2 + row] & B[3 + row]);
            A[2 + row] = B[2 + row] ^ (~B[3 + row] & B[4 + row]);
            A[3 + row] = B[3 + row] ^ (~B[4 + row] & B[0 + row]);
            A[4 + row] = B[4 + row] ^ (~B[0 + row] & B[1 + row]);
        }

        // Iota
        A[0] ^= KECCAK_RC[r];
    }
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

    uint n_lanes = n_bytes >> 3;

    thread ulong A[25] = {0};
    
    uint base = idx * n_lanes;
    // Initial absorb (guarded statically unrolled loop prevents divergence)
    #pragma clang loop unroll(full)
    for (uint k = 0u; k < 17u; ++k) {
        if (k < n_lanes) A[k] = seeds[base + k];
    }

    for (uint step = 0u; step < w; ++step) {
        // Safe domain padding via conditional overwrites mapped exclusively to registers
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < 25u; ++i) {
            if (i >= n_lanes) A[i] = 0ul;
            if (i == n_lanes) A[i] ^= 0x06ul;
        }
        
        A[16] ^= 0x8000000000000000ul;
        
        keccak_f1600(A);
    }

    // Export chain tip
    #pragma clang loop unroll(full)
    for (uint k = 0u; k < 17u; ++k) {
        if (k < n_lanes) tips[base + k] = A[k];
    }
}