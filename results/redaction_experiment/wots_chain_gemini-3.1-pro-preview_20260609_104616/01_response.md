This optimization targets the primary bottleneck: the baseline stores the Keccak state (`A`, `B`, `C`, `D`) in arrays with dynamic or loop-based indexing. This prevents the compiler from performing Scalar Replacement of Aggregates (SROA), forcing the 200-byte state to spill into thread-local device memory, which takes hundreds of cycles per access. 

By fully unrolling the Keccak inner loops and rewriting all state accesses to use strictly compile-time constant indices, the compiler natively promotes the entire Keccak state into high-speed registers. Additionally, we explicitly branch on the known `n_lanes` sizes (2 and 4). This lets the compiler's constant-folding statically propagate the zero-padded lanes directly into the first round of the Keccak permutation inside the step loop, completely eliminating a significant portion of the XOR operations for that round.

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint  SHA3_RATE_LANES  = 17u;
constexpr constant ulong SHA3_DOMAIN_WORD = 0x06ul;
constexpr constant ulong SHA3_FINAL_PAD   = 0x8000000000000000ul;

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

constexpr constant uint KECCAK_RHO[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
};

inline __attribute__((always_inline))
ulong rotl64(ulong x, uint k) {
    return (k == 0u) ? x : ((x << k) | (x >> (64u - k)));
}

inline __attribute__((always_inline))
void keccak_f1600(thread ulong *A) {
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
        for (uint i = 0u; i < 25u; ++i) {
            uint x = i % 5u;
            uint y = i / 5u;
            uint dest_x = y;
            uint dest_y = (2u * x + 3u * y) % 5u;
            // Directly inline A^D step into rotation
            B[dest_x + 5u * dest_y] = rotl64(A[i] ^ D[x], KECCAK_RHO[i]);
        }

        #pragma unroll
        for (uint i = 0u; i < 25u; ++i) {
            uint x = i % 5u;
            uint y = i / 5u;
            uint i1 = ((x + 1u) % 5u) + 5u * y;
            uint i2 = ((x + 2u) % 5u) + 5u * y;
            A[i] = B[i] ^ ((~B[i1]) & B[i2]);
        }

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
    uint base = idx * n_lanes;

    thread ulong A[25];
    #pragma unroll
    for (uint i = 0u; i < 25u; ++i) {
        A[i] = 0ul;
    }

    // Explicit manual indexing to force full compiler scalarization
    if (n_lanes == 2u) {
        A[0] = seeds[base + 0u];
        A[1] = seeds[base + 1u];
    } else if (n_lanes == 4u) {
        A[0] = seeds[base + 0u];
        A[1] = seeds[base + 1u];
        A[2] = seeds[base + 2u];
        A[3] = seeds[base + 3u];
    } else {
        #pragma unroll
        for (uint k = 0u; k < 17u; ++k) {
            if (k < n_lanes) A[k] = seeds[base + k];
        }
    }

    // Specialize branches by expected sizes. Hard-coding zeroes as pure constants allows 
    // LLVM to fold them deeply into the first round of the Keccak core.
    if (n_lanes == 2u) {
        for (uint step = 0u; step < w; ++step) {
            #pragma unroll
            for (uint i = 2u; i < 25u; ++i) {
                if (i == 2u) A[i] = SHA3_DOMAIN_WORD;
                else if (i == 16u) A[i] = SHA3_FINAL_PAD;
                else A[i] = 0ul;
            }
            keccak_f1600(A);
        }
    } else if (n_lanes == 4u) {
        for (uint step = 0u; step < w; ++step) {
            #pragma unroll
            for (uint i = 4u; i < 25u; ++i) {
                if (i == 4u) A[i] = SHA3_DOMAIN_WORD;
                else if (i == 16u) A[i] = SHA3_FINAL_PAD;
                else A[i] = 0ul;
            }
            keccak_f1600(A);
        }
    } else {
        for (uint step = 0u; step < w; ++step) {
            #pragma unroll
            for (uint i = 0u; i < 25u; ++i) {
                if (i >= n_lanes) A[i] = 0ul;
            }
            #pragma unroll
            for (uint i = 0u; i < 25u; ++i) {
                if (i == n_lanes) A[i] ^= SHA3_DOMAIN_WORD;
            }
            A[16u] ^= SHA3_FINAL_PAD;
            keccak_f1600(A);
        }
    }

    if (n_lanes == 2u) {
        tips[base + 0u] = A[0];
        tips[base + 1u] = A[1];
    } else if (n_lanes == 4u) {
        tips[base + 0u] = A[0];
        tips[base + 1u] = A[1];
        tips[base + 2u] = A[2];
        tips[base + 3u] = A[3];
    } else {
        #pragma unroll
        for (uint k = 0u; k < 17u; ++k) {
            if (k < n_lanes) tips[base + k] = A[k];
        }
    }
}
```