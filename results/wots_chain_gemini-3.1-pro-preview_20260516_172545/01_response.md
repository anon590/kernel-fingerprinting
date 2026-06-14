(1) **Optimization**: We explicitly specialize the outer step loop for `n_bytes == 16` and `32`, fully unroll the Keccak-f1600 round loop, and fuse the Rho and Pi permutations into a single in-place cycle (avoiding the 25-element `B` buffer). 

(2) **Why it improves**: Specializing the loop bounds guarantees that all array accesses into the state `A` use strictly compile-time constant indices. This unlocks LLVM's Scalar Replacement of Aggregates (SROA), completely promoting the 25-lane state into fast thread registers and avoiding costly local memory spills. Unrolling the rounds and eliminating the `B` array keeps our register footprint within the hardware budget (ensuring 100% occupancy), while enabling the compiler to fold the padded zeros deeply into the first Keccak round.

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint  SHA3_RATE_LANES   = 17u;
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
    0x8000000000008080ul, 0x0000000080000001ul, 0x8000000080008008ul,
};

constant uint KECCAK_RHO[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
};

// Sequence of indices for the in-place Pi permutation cycle
constant uint PI_LANE[24] = {
    1, 10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6
};

inline ulong rotl64_opt(ulong x, uint k) {
    uint amt = k & 63u;
    return (amt == 0u) ? x : ((x << amt) | (x >> (64u - amt)));
}

inline __attribute__((always_inline))
void keccak_f1600(thread ulong (&A)[25]) {
    #pragma unroll
    for (uint r = 0; r < 24; ++r) {
        ulong C[5];
        #pragma unroll
        for (uint x = 0; x < 5; ++x) {
            C[x] = A[x] ^ A[x + 5] ^ A[x + 10] ^ A[x + 15] ^ A[x + 20];
        }
        
        ulong D[5];
        #pragma unroll
        for (uint x = 0; x < 5; ++x) {
            D[x] = C[(x + 4) % 5] ^ rotl64_opt(C[(x + 1) % 5], 1u);
        }
        
        #pragma unroll
        for (uint y = 0; y < 5; ++y) {
            #pragma unroll
            for (uint x = 0; x < 5; ++x) {
                A[x + 5 * y] ^= D[x];
            }
        }
        
        ulong current = A[1];
        #pragma unroll
        for (uint i = 0; i < 24; ++i) {
            uint src = PI_LANE[i];
            uint dest = PI_LANE[(i + 1) % 24];
            ulong temp = A[dest];
            A[dest] = rotl64_opt(current, KECCAK_RHO[src]);
            current = temp;
        }
        
        #pragma unroll
        for (uint y = 0; y < 5; ++y) {
            ulong T[5];
            #pragma unroll
            for (uint x = 0; x < 5; ++x) {
                T[x] = A[x + 5 * y];
            }
            #pragma unroll
            for (uint x = 0; x < 5; ++x) {
                A[x + 5 * y] = T[x] ^ ((~T[(x + 1) % 5]) & T[(x + 2) % 5]);
            }
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

    // Specialize for expected cryptographic digest sizes to guarantee that
    // state array accesses use strictly compile-time constant indices. 
    // This perfectly scalarizes `A` into thread registers.
    if (n_bytes == 16) {
        ulong A[25];
        #pragma unroll
        for (uint i = 0; i < 25; ++i) A[i] = 0ul;
        
        uint base = idx * 2;
        A[0] = seeds[base + 0];
        A[1] = seeds[base + 1];
        
        for (uint step = 0; step < w; ++step) {
            #pragma unroll
            for (uint i = 2; i < 25; ++i) A[i] = 0ul;
            A[2] = SHA3_DOMAIN_WORD;
            A[16] = SHA3_FINAL_PAD;
            keccak_f1600(A);
        }
        
        tips[base + 0] = A[0];
        tips[base + 1] = A[1];
        
    } else if (n_bytes == 32) {
        ulong A[25];
        #pragma unroll
        for (uint i = 0; i < 25; ++i) A[i] = 0ul;
        
        uint base = idx * 4;
        A[0] = seeds[base + 0];
        A[1] = seeds[base + 1];
        A[2] = seeds[base + 2];
        A[3] = seeds[base + 3];
        
        for (uint step = 0; step < w; ++step) {
            #pragma unroll
            for (uint i = 4; i < 25; ++i) A[i] = 0ul;
            A[4] = SHA3_DOMAIN_WORD;
            A[16] = SHA3_FINAL_PAD;
            keccak_f1600(A);
        }
        
        tips[base + 0] = A[0];
        tips[base + 1] = A[1];
        tips[base + 2] = A[2];
        tips[base + 3] = A[3];
        
    } else {
        // Generic fallback for any other sizes
        uint n_lanes = n_bytes >> 3;
        ulong A[25];
        #pragma unroll
        for (uint i = 0; i < 25; ++i) A[i] = 0ul;
        
        uint base = idx * n_lanes;
        for (uint k = 0; k < n_lanes; ++k) {
            A[k] = seeds[base + k];
        }
        
        for (uint step = 0; step < w; ++step) {
            for (uint i = n_lanes; i < 25; ++i) A[i] = 0ul;
            A[n_lanes] ^= SHA3_DOMAIN_WORD;
            A[SHA3_RATE_LANES - 1u] ^= SHA3_FINAL_PAD;
            keccak_f1600(A);
        }
        
        for (uint k = 0; k < n_lanes; ++k) {
            tips[base + k] = A[k];
        }
    }
}
```