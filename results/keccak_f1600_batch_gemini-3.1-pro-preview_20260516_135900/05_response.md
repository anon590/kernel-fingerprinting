To maximize Keccak permutation throughput, we must overcome two primary bottlenecks in the previous iteration: instruction-level parallelism (ILP) and register spilling. 

1. **Eliminating Serial Dependencies**: The previous attempt used in-place cycle tracing for the `Rho` and `Pi` steps. This formed a 24-cycle dependency chain per round, heavily stalling the GPU's ALUs. We break this by using a dual-state array approach (`A` and `E`) and merging Theta, Rho, Pi, and Chi into a fully parallel assignment block, unrolled pairwise to avoid memory copies.
2. **Preventing State Register Spills**: Using dynamic loop indices for padding (e.g., `A[rate_lanes - 1]`) or squeezing forces the compiler to demote the 1600-bit state array from fast registers to slow thread-local memory. By wrapping these accesses in `#pragma unroll(25)` loops, we guarantee that all array indices are evaluated as compile-time constants. This ensures LLVM scalarizes the entire state matrix into 50 physical registers, keeping memory accesses strictly out of the permutation loop.

```metal
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

// Fully parallel round transformation merging Theta, Rho, Pi, and Chi.
// Calculates all rows independently, completely avoiding serial dependency chains.
#define ROUND(A, E, rc) do { \
    ulong C0 = A[0] ^ A[5] ^ A[10] ^ A[15] ^ A[20]; \
    ulong C1 = A[1] ^ A[6] ^ A[11] ^ A[16] ^ A[21]; \
    ulong C2 = A[2] ^ A[7] ^ A[12] ^ A[17] ^ A[22]; \
    ulong C3 = A[3] ^ A[8] ^ A[13] ^ A[18] ^ A[23]; \
    ulong C4 = A[4] ^ A[9] ^ A[14] ^ A[19] ^ A[24]; \
    \
    ulong D0 = C4 ^ rotl64(C1, 1u); \
    ulong D1 = C0 ^ rotl64(C2, 1u); \
    ulong D2 = C1 ^ rotl64(C3, 1u); \
    ulong D3 = C2 ^ rotl64(C4, 1u); \
    ulong D4 = C3 ^ rotl64(C0, 1u); \
    \
    ulong B0, B1, B2, B3, B4; \
    \
    B0 = A[0] ^ D0; \
    B1 = rotl64(A[6] ^ D1, 44u); \
    B2 = rotl64(A[12] ^ D2, 43u); \
    B3 = rotl64(A[18] ^ D3, 21u); \
    B4 = rotl64(A[24] ^ D4, 14u); \
    E[0] = B0 ^ ((~B1) & B2) ^ rc; \
    E[1] = B1 ^ ((~B2) & B3); \
    E[2] = B2 ^ ((~B3) & B4); \
    E[3] = B3 ^ ((~B4) & B0); \
    E[4] = B4 ^ ((~B0) & B1); \
    \
    B0 = rotl64(A[3] ^ D3, 28u); \
    B1 = rotl64(A[9] ^ D4, 20u); \
    B2 = rotl64(A[10] ^ D0, 3u); \
    B3 = rotl64(A[16] ^ D1, 45u); \
    B4 = rotl64(A[22] ^ D2, 61u); \
    E[5] = B0 ^ ((~B1) & B2); \
    E[6] = B1 ^ ((~B2) & B3); \
    E[7] = B2 ^ ((~B3) & B4); \
    E[8] = B3 ^ ((~B4) & B0); \
    E[9] = B4 ^ ((~B0) & B1); \
    \
    B0 = rotl64(A[1] ^ D1, 1u); \
    B1 = rotl64(A[7] ^ D2, 6u); \
    B2 = rotl64(A[13] ^ D3, 25u); \
    B3 = rotl64(A[19] ^ D4, 8u); \
    B4 = rotl64(A[20] ^ D0, 18u); \
    E[10] = B0 ^ ((~B1) & B2); \
    E[11] = B1 ^ ((~B2) & B3); \
    E[12] = B2 ^ ((~B3) & B4); \
    E[13] = B3 ^ ((~B4) & B0); \
    E[14] = B4 ^ ((~B0) & B1); \
    \
    B0 = rotl64(A[4] ^ D4, 27u); \
    B1 = rotl64(A[5] ^ D0, 36u); \
    B2 = rotl64(A[11] ^ D1, 10u); \
    B3 = rotl64(A[17] ^ D2, 15u); \
    B4 = rotl64(A[23] ^ D3, 56u); \
    E[15] = B0 ^ ((~B1) & B2); \
    E[16] = B1 ^ ((~B2) & B3); \
    E[17] = B2 ^ ((~B3) & B4); \
    E[18] = B3 ^ ((~B4) & B0); \
    E[19] = B4 ^ ((~B0) & B1); \
    \
    B0 = rotl64(A[2] ^ D2, 62u); \
    B1 = rotl64(A[8] ^ D3, 55u); \
    B2 = rotl64(A[14] ^ D4, 39u); \
    B3 = rotl64(A[15] ^ D0, 41u); \
    B4 = rotl64(A[21] ^ D1, 2u); \
    E[20] = B0 ^ ((~B1) & B2); \
    E[21] = B1 ^ ((~B2) & B3); \
    E[22] = B2 ^ ((~B3) & B4); \
    E[23] = B3 ^ ((~B4) & B0); \
    E[24] = B4 ^ ((~B0) & B1); \
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

    ulong A[25];
    ulong E[25];

    // Load guaranteed 32 bytes of message sequentially as ulong4 block
    device const ulong4* in_data4 = (device const ulong4*)in_data;
    ulong4 msg = in_data4[idx];
    
    A[0] = msg.x;
    A[1] = msg.y;
    A[2] = msg.z;
    A[3] = msg.w;
    
    #pragma unroll(21)
    for (uint i = 4u; i < 25u; ++i) {
        A[i] = 0ul;
    }

    uint msg_lanes  = msg_bytes >> 3;
    uint rate_lanes = rate_bytes >> 3;
    uint out_lanes  = out_bytes >> 3;

    // Use unrolled loops over constant indices to enforce compiler scalarization into registers
    #pragma unroll(25)
    for (uint i = 0u; i < 25u; i++) {
        if (i == msg_lanes) A[i] ^= (ulong)(domain & 0xFFu);
    }

    #pragma unroll(25)
    for (uint i = 0u; i < 25u; i++) {
        if (i == rate_lanes - 1u) A[i] ^= 0x8000000000000000ul;
    }

    uint out_base = idx * out_lanes;
    uint written = 0u;

    for (;;) {
        // Completely unroll permutation: alternate arrays to remove copies
        #pragma unroll(12)
        for (uint r = 0u; r < 24u; r += 2u) {
            ROUND(A, E, KECCAK_RC[r]);
            ROUND(E, A, KECCAK_RC[r+1]);
        }

        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;

        // Fully unrolled branch mapping prevents dynamic indexing & register eviction
        #pragma unroll(25)
        for (uint j = 0u; j < 25u; ++j) {
            if (j < take) {
                out_data[out_base + written + j] = A[j];
            }
        }

        written += take;
        if (written >= out_lanes) break;
    }
}
```