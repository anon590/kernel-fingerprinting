To maximize performance, the entire Keccak-f1600 state `A` must be kept in registers. The baseline kernel spills the state to thread-local memory because of dynamic array indexing (e.g., zeroing `A[n_lanes..24]`). 

The key optimization is eliminating all dynamic array indices. We fully unroll all internal loops (`#pragma clang loop unroll(full)`) and replace variable-length padding loops with constant-bound loops using uniform conditional assignments (`if (i >= n_lanes) A[i] = 0ul;`). Additionally, the Rho and Pi permutations are implemented in-place using a single mathematical cycle of 24 elements. This completely removes the baseline's 25-element temporary `B` array, halving register pressure. 

By forcing completely static indexing and minimizing temporary variables, the state securely fits into the Apple Silicon registers, eliminating all L1/memory spilling and reducing the kernel to pure register-to-register ALUs.

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint  SHA3_RATE_LANES   = 17u;
constexpr constant ulong SHA3_DOMAIN_WORD  = 0x06ul;
constexpr constant ulong SHA3_FINAL_PAD    = 0x8000000000000000ul;

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

inline ulong rotl64(ulong x, uint k) {
    return (x << k) | (x >> (64u - k));
}

inline __attribute__((always_inline)) void keccak_f1600(thread ulong *A) {
    #pragma clang loop unroll(full)
    for (uint r = 0u; r < 24u; ++r) {
        // Theta
        ulong C[5];
        #pragma clang loop unroll(full)
        for (uint x = 0u; x < 5u; ++x) {
            C[x] = A[x] ^ A[x + 5u] ^ A[x + 10u] ^ A[x + 15u] ^ A[x + 20u];
        }
        
        ulong D[5];
        #pragma clang loop unroll(full)
        for (uint x = 0u; x < 5u; ++x) {
            D[x] = C[(x + 4u) % 5u] ^ rotl64(C[(x + 1u) % 5u], 1u);
        }
        
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < 25u; ++i) {
            A[i] ^= D[i % 5u];
        }

        // Rho and Pi (in-place cycle)
        ulong current = A[1];
        uint x_pos = 1u;
        uint y_pos = 0u;
        
        #pragma clang loop unroll(full)
        for (uint t = 0u; t < 24u; ++t) {
            uint src = x_pos + 5u * y_pos;
            uint x_new = y_pos;
            uint y_new = (2u * x_pos + 3u * y_pos) % 5u;
            uint dst = x_new + 5u * y_new;
            
            ulong temp = A[dst];
            A[dst] = rotl64(current, KECCAK_RHO[src]);
            current = temp;
            
            x_pos = x_new;
            y_pos = y_new;
        }

        // Chi
        #pragma clang loop unroll(full)
        for (uint y = 0u; y < 5u; ++y) {
            ulong T[5];
            #pragma clang loop unroll(full)
            for (uint x = 0u; x < 5u; ++x) {
                T[x] = A[x + 5u * y];
            }
            #pragma clang loop unroll(full)
            for (uint x = 0u; x < 5u; ++x) {
                A[x + 5u * y] = T[x] ^ ((~T[(x + 1u) % 5u]) & T[(x + 2u) % 5u]);
            }
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

    // Allocate state array cleanly to avoid uninitialized values
    thread ulong A[25] = {0};
    
    // Initial absorb (avoid dynamic indexing to keep A purely in registers)
    uint base = idx * n_lanes;
    #pragma clang loop unroll(full)
    for (uint k = 0u; k < 17u; ++k) {
        if (k < n_lanes) A[k] = seeds[base + k];
    }

    for (uint step = 0u; step < w; ++step) {
        // Zeroing and domain padding via unrolled conditional execution
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < 25u; ++i) {
            if (i >= n_lanes) A[i] = 0ul;
            if (i == n_lanes) A[i] ^= SHA3_DOMAIN_WORD;
        }
        
        A[16] ^= SHA3_FINAL_PAD; // SHA3_RATE_LANES - 1u == 16
        keccak_f1600(A);
    }

    // Export chain tip via static indexing bounds check
    #pragma clang loop unroll(full)
    for (uint k = 0u; k < 17u; ++k) {
        if (k < n_lanes) tips[base + k] = A[k];
    }
}
```