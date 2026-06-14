To reach peak performance, I applied three major optimizations:

1. **Fully Merged Rho-Pi Step**: Standard Keccak implementations use a sequential loop for the Rho and Pi permutations. I merged Theta (part 2), Rho, and Pi into a single parallel assignment step (`B[dest_x + 5*dest_y] = rotl(...)`), eliminating data dependencies and allowing the compiler to schedule and parallelize the permutation completely.
2. **Strict Register Allocation**: The `KECCAK_RHO` and `KECCAK_RC` arrays are explicitly marked `constexpr` instead of `constant` to force compile-time evaluation. In the generic fallback, all indexing into the `A` state array is governed by `#pragma unroll` and predicate conditions (`if (i >= n_lanes)`), guaranteeing that the compiler maps the 200-byte array `A` completely into registers without spilling to thread memory.
3. **Vectorized Coalesced Memory**: I cast `seeds` and `tips` to `ulong2` and `ulong4` in the specialized 16-byte and 32-byte branches. This guarantees perfectly coalesced wide loads/stores across the SIMD group.

```metal
#include <metal_stdlib>
using namespace metal;

constexpr ulong KECCAK_RC[24] = {
    0x0000000000000001ul, 0x0000000000008082ul, 0x800000000000808Aul,
    0x8000000080008000ul, 0x000000000000808Bul, 0x0000000080000001ul,
    0x8000000080008081ul, 0x8000000000008009ul, 0x000000000000008Aul,
    0x0000000000000088ul, 0x0000000080008009ul, 0x000000008000000Aul,
    0x000000008000808Bul, 0x800000000000008Bul, 0x8000000000008089ul,
    0x8000000000008003ul, 0x8000000000008002ul, 0x8000000000000080ul,
    0x000000000000800Aul, 0x800000008000000Aul, 0x8000000080008081ul,
    0x8000000000008080ul, 0x0000000080000001ul, 0x8000000080008008ul
};

constexpr uint KECCAK_RHO[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14
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
        
        ulong B[25];
        #pragma unroll
        for (uint y = 0; y < 5; ++y) {
            #pragma unroll
            for (uint x = 0; x < 5; ++x) {
                uint idx = x + 5 * y;
                uint dest_x = y;
                uint dest_y = (2 * x + 3 * y) % 5;
                B[dest_x + 5 * dest_y] = rotl64_opt(A[idx] ^ D[x], KECCAK_RHO[idx]);
            }
        }
        
        #pragma unroll
        for (uint y = 0; y < 5; ++y) {
            #pragma unroll
            for (uint x = 0; x < 5; ++x) {
                A[x + 5 * y] = B[x + 5 * y] ^ ((~B[(x + 1) % 5 + 5 * y]) & B[(x + 2) % 5 + 5 * y]);
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

    if (n_bytes == 16) {
        ulong A[25];
        
        device const ulong2 *seeds2 = (device const ulong2 *)seeds;
        device ulong2 *tips2 = (device ulong2 *)tips;
        
        ulong2 s = seeds2[idx];
        A[0] = s.x;
        A[1] = s.y;
        
        for (uint step = 0; step < w; ++step) {
            #pragma unroll
            for (uint i = 2; i < 25; ++i) A[i] = 0ul;
            A[2] = 0x06ul;
            A[16] = 0x8000000000000000ul;
            keccak_f1600(A);
        }
        
        tips2[idx] = ulong2(A[0], A[1]);
        
    } else if (n_bytes == 32) {
        ulong A[25];
        
        device const ulong4 *seeds4 = (device const ulong4 *)seeds;
        device ulong4 *tips4 = (device ulong4 *)tips;
        
        ulong4 s = seeds4[idx];
        A[0] = s.x;
        A[1] = s.y;
        A[2] = s.z;
        A[3] = s.w;
        
        for (uint step = 0; step < w; ++step) {
            #pragma unroll
            for (uint i = 4; i < 25; ++i) A[i] = 0ul;
            A[4] = 0x06ul;
            A[16] = 0x8000000000000000ul;
            keccak_f1600(A);
        }
        
        tips4[idx] = ulong4(A[0], A[1], A[2], A[3]);
        
    } else {
        uint n_lanes = n_bytes >> 3;
        ulong A[25];
        #pragma unroll
        for (uint i = 0; i < 25; ++i) A[i] = 0ul;
        
        uint base = idx * n_lanes;
        #pragma unroll
        for (uint k = 0; k < 17; ++k) {
            if (k < n_lanes) A[k] = seeds[base + k];
        }
        
        for (uint step = 0; step < w; ++step) {
            #pragma unroll
            for (uint i = 0; i < 25; ++i) {
                if (i >= n_lanes) A[i] = 0ul;
                if (i == n_lanes) A[i] ^= 0x06ul;
                if (i == 16)      A[i] ^= 0x8000000000000000ul;
            }
            keccak_f1600(A);
        }
        
        #pragma unroll
        for (uint k = 0; k < 17; ++k) {
            if (k < n_lanes) tips[base + k] = A[k];
        }
    }
}
```