The core optimization is eliminating all dynamic array indexing of the 1600-bit state `A`, combined with `#pragma unroll` on all permutation loops. By replacing dynamic-length loops with statically bounded unrolled loops and predicated execution (e.g., `if (i < msg_lanes)`), the compiler's SROA (Scalar Replacement of Aggregates) pass can completely scalarize the `A` and `B` arrays into registers. This prevents the state from spilling to slow local memory, which is a major performance killer in cryptographic kernels.

Furthermore, fully unrolling the 24 rounds allows the compiler to resolve all FIPS 202 constants (`RC` and `RHO`) at compile time. This is especially crucial for the 64-bit rotations, which the compiler optimizes from variable-shift library calls into highly efficient 32-bit immediate shift-and-insert instructions native to Apple Silicon ALUs.

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

constant uint KECCAK_RHO[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
};

inline ulong rotl64(ulong x, uint k) {
    if (k == 0u) return x;
    return (x << k) | (x >> (64u - k));
}

inline void keccak_f1600(thread ulong *A) {
    #pragma unroll
    for (uint r = 0u; r < 24u; ++r) {
        ulong C[5];
        #pragma unroll
        for (uint x = 0u; x < 5u; ++x) {
            C[x] = A[x] ^ A[x + 5u] ^ A[x + 10u] ^ A[x + 15u] ^ A[x + 20u];
        }
        
        ulong D[5];
        #pragma unroll
        for (uint x = 0u; x < 5u; ++x) {
            D[x] = C[(x + 4u) % 5u] ^ rotl64(C[(x + 1u) % 5u], 1u);
        }

        ulong B[25];
        #pragma unroll
        for (uint x = 0u; x < 5u; ++x) {
            #pragma unroll
            for (uint y = 0u; y < 5u; ++y) {
                uint src = x + 5u * y;
                uint dest = y + 5u * ((2u * x + 3u * y) % 5u);
                ulong val = A[src] ^ D[x];
                B[dest] = rotl64(val, KECCAK_RHO[src]);
            }
        }

        #pragma unroll
        for (uint y = 0u; y < 5u; ++y) {
            #pragma unroll
            for (uint x = 0u; x < 5u; ++x) {
                uint i  = x + 5u * y;
                uint i1 = ((x + 1u) % 5u) + 5u * y;
                uint i2 = ((x + 2u) % 5u) + 5u * y;
                A[i] = B[i] ^ ((~B[i1]) & B[i2]);
            }
        }

        A[0] ^= KECCAK_RC[r];
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

    uint msg_lanes  = msg_bytes  >> 3;
    uint rate_lanes = rate_bytes >> 3;
    uint out_lanes  = out_bytes  >> 3;

    ulong A[25] = {0};

    // Absorb: statically unrolled to prevent the A array from spilling to local memory
    uint in_base = idx * msg_lanes;
    #pragma unroll
    for (uint i = 0u; i < 25u; ++i) {
        if (i < msg_lanes) {
            A[i] ^= in_data[in_base + i];
        }
    }

    // Pad
    #pragma unroll
    for (uint i = 0u; i < 25u; ++i) {
        if (i == msg_lanes) {
            A[i] ^= (ulong)(domain & 0xFFu);
        }
        if (i == rate_lanes - 1u) {
            A[i] ^= 0x8000000000000000ul;
        }
    }

    // Squeeze
    uint out_base = idx * out_lanes;
    uint written = 0u;
    for (;;) {
        keccak_f1600(A);
        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        
        // Output writes are statically unrolled for perfect scalarization of A
        #pragma unroll
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