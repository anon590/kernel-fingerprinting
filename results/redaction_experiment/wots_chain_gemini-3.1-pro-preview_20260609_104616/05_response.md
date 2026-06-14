To optimize the kernel, I manually unrolled the 24 Keccak rounds and tightly scoped the state array `A` into the branch blocks to guarantee perfect SROA (Scalar Replacement of Aggregates). In the previous version, the `thread ulong A[25];` was declared at the function level, and a dynamic array index (`A[n_lanes]`) in the fallback `else` branch forced the LLVM compiler to place the state array in thread memory across all branches. 

By isolating `A` into each branch and rewriting the dynamic indexing into compile-time unrolled loops, we guarantee that all accesses use strictly constant indices. This ensures the entire 200-byte Keccak state and its intermediate arrays live exclusively in registers. Furthermore, because it's now fully scalarized, the compiler's dead code elimination (DCE) can successfully prune the unused trailing `Chi` step computations from the final round of each step.

```metal
#include <metal_stdlib>
using namespace metal;

inline __attribute__((always_inline))
ulong rotl64(ulong x, uint k) {
    return (x << k) | (x >> (64u - k));
}

#define ROUND(A, E, rc) do { \
    ulong C0 = (A[0]  ^ A[5])  ^ (A[10] ^ A[15]) ^ A[20]; \
    ulong C1 = (A[1]  ^ A[6])  ^ (A[11] ^ A[16]) ^ A[21]; \
    ulong C2 = (A[2]  ^ A[7])  ^ (A[12] ^ A[17]) ^ A[22]; \
    ulong C3 = (A[3]  ^ A[8])  ^ (A[13] ^ A[18]) ^ A[23]; \
    ulong C4 = (A[4]  ^ A[9])  ^ (A[14] ^ A[19]) ^ A[24]; \
    \
    ulong D0 = C4 ^ rotl64(C1, 1u); \
    ulong D1 = C0 ^ rotl64(C2, 1u); \
    ulong D2 = C1 ^ rotl64(C3, 1u); \
    ulong D3 = C2 ^ rotl64(C4, 1u); \
    ulong D4 = C3 ^ rotl64(C0, 1u); \
    \
    ulong b0 = A[0] ^ D0; \
    ulong b1 = rotl64(A[6] ^ D1, 44u); \
    ulong b2 = rotl64(A[12] ^ D2, 43u); \
    ulong b3 = rotl64(A[18] ^ D3, 21u); \
    ulong b4 = rotl64(A[24] ^ D4, 14u); \
    E[0] = b0 ^ ((~b1) & b2) ^ rc; \
    E[1] = b1 ^ ((~b2) & b3); \
    E[2] = b2 ^ ((~b3) & b4); \
    E[3] = b3 ^ ((~b4) & b0); \
    E[4] = b4 ^ ((~b0) & b1); \
    \
    b0 = rotl64(A[3] ^ D3, 28u); \
    b1 = rotl64(A[9] ^ D4, 20u); \
    b2 = rotl64(A[10] ^ D0, 3u); \
    b3 = rotl64(A[16] ^ D1, 45u); \
    b4 = rotl64(A[22] ^ D2, 61u); \
    E[5] = b0 ^ ((~b1) & b2); \
    E[6] = b1 ^ ((~b2) & b3); \
    E[7] = b2 ^ ((~b3) & b4); \
    E[8] = b3 ^ ((~b4) & b0); \
    E[9] = b4 ^ ((~b0) & b1); \
    \
    b0 = rotl64(A[1] ^ D1, 1u); \
    b1 = rotl64(A[7] ^ D2, 6u); \
    b2 = rotl64(A[13] ^ D3, 25u); \
    b3 = rotl64(A[19] ^ D4, 8u); \
    b4 = rotl64(A[20] ^ D0, 18u); \
    E[10] = b0 ^ ((~b1) & b2); \
    E[11] = b1 ^ ((~b2) & b3); \
    E[12] = b2 ^ ((~b3) & b4); \
    E[13] = b3 ^ ((~b4) & b0); \
    E[14] = b4 ^ ((~b0) & b1); \
    \
    b0 = rotl64(A[4] ^ D4, 27u); \
    b1 = rotl64(A[5] ^ D0, 36u); \
    b2 = rotl64(A[11] ^ D1, 10u); \
    b3 = rotl64(A[17] ^ D2, 15u); \
    b4 = rotl64(A[23] ^ D3, 56u); \
    E[15] = b0 ^ ((~b1) & b2); \
    E[16] = b1 ^ ((~b2) & b3); \
    E[17] = b2 ^ ((~b3) & b4); \
    E[18] = b3 ^ ((~b4) & b0); \
    E[19] = b4 ^ ((~b0) & b1); \
    \
    b0 = rotl64(A[2] ^ D2, 62u); \
    b1 = rotl64(A[8] ^ D3, 55u); \
    b2 = rotl64(A[14] ^ D4, 39u); \
    b3 = rotl64(A[15] ^ D0, 41u); \
    b4 = rotl64(A[21] ^ D1, 2u); \
    E[20] = b0 ^ ((~b1) & b2); \
    E[21] = b1 ^ ((~b2) & b3); \
    E[22] = b2 ^ ((~b3) & b4); \
    E[23] = b3 ^ ((~b4) & b0); \
    E[24] = b4 ^ ((~b0) & b1); \
} while(0)

inline __attribute__((always_inline))
void keccak_f1600(thread ulong (&A)[25]) {
    ulong E[25];
    ROUND(A, E, 0x0000000000000001ul);
    ROUND(E, A, 0x0000000000008082ul);
    ROUND(A, E, 0x800000000000808Aul);
    ROUND(E, A, 0x8000000080008000ul);
    ROUND(A, E, 0x000000000000808Bul);
    ROUND(E, A, 0x0000000080000001ul);
    ROUND(A, E, 0x8000000080008081ul);
    ROUND(E, A, 0x8000000000008009ul);
    ROUND(A, E, 0x000000000000008Aul);
    ROUND(E, A, 0x0000000000000088ul);
    ROUND(A, E, 0x0000000080008009ul);
    ROUND(E, A, 0x000000008000000Aul);
    ROUND(A, E, 0x000000008000808Bul);
    ROUND(E, A, 0x800000000000008Bul);
    ROUND(A, E, 0x8000000000008089ul);
    ROUND(E, A, 0x8000000000008003ul);
    ROUND(A, E, 0x8000000000008002ul);
    ROUND(E, A, 0x8000000000000080ul);
    ROUND(A, E, 0x000000000000800Aul);
    ROUND(E, A, 0x800000008000000Aul);
    ROUND(A, E, 0x8000000080008081ul);
    ROUND(E, A, 0x8000000000008080ul);
    ROUND(A, E, 0x0000000080000001ul);
    ROUND(E, A, 0x8000000080008008ul);
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
    uint w_val = w;

    if (n_lanes == 2u) {
        ulong A[25];
        device const ulong2 *seeds_v = (device const ulong2 *)seeds;
        ulong2 seed_val = seeds_v[idx];
        A[0] = seed_val.x;
        A[1] = seed_val.y;
        
        for (uint step = 0u; step < w_val; ++step) {
            A[2]  = 0x06ul;
            A[3]  = 0ul; A[4]  = 0ul; A[5]  = 0ul; A[6]  = 0ul;
            A[7]  = 0ul; A[8]  = 0ul; A[9]  = 0ul; A[10] = 0ul;
            A[11] = 0ul; A[12] = 0ul; A[13] = 0ul; A[14] = 0ul;
            A[15] = 0ul; 
            A[16] = 0x8000000000000000ul;
            A[17] = 0ul; A[18] = 0ul; A[19] = 0ul; A[20] = 0ul;
            A[21] = 0ul; A[22] = 0ul; A[23] = 0ul; A[24] = 0ul;
            
            keccak_f1600(A);
        }
        
        device ulong2 *tips_v = (device ulong2 *)tips;
        tips_v[idx] = ulong2(A[0], A[1]);

    } else if (n_lanes == 4u) {
        ulong A[25];
        device const ulong4 *seeds_v = (device const ulong4 *)seeds;
        ulong4 seed_val = seeds_v[idx];
        A[0] = seed_val.x;
        A[1] = seed_val.y;
        A[2] = seed_val.z;
        A[3] = seed_val.w;
        
        for (uint step = 0u; step < w_val; ++step) {
            A[4]  = 0x06ul;
            A[5]  = 0ul; A[6]  = 0ul; A[7]  = 0ul; A[8]  = 0ul; 
            A[9]  = 0ul; A[10] = 0ul; A[11] = 0ul; A[12] = 0ul; 
            A[13] = 0ul; A[14] = 0ul; A[15] = 0ul; 
            A[16] = 0x8000000000000000ul;
            A[17] = 0ul; A[18] = 0ul; A[19] = 0ul; A[20] = 0ul;
            A[21] = 0ul; A[22] = 0ul; A[23] = 0ul; A[24] = 0ul;
            
            keccak_f1600(A);
        }
        
        device ulong4 *tips_v = (device ulong4 *)tips;
        tips_v[idx] = ulong4(A[0], A[1], A[2], A[3]);

    } else {
        ulong A[25];
        uint base = idx * n_lanes;
        #pragma unroll
        for (uint i = 0u; i < 25u; ++i) {
            A[i] = 0ul;
        }
        #pragma unroll
        for (uint i = 0u; i < 25u; ++i) {
            if (i < n_lanes) A[i] = seeds[base + i];
        }
        
        for (uint step = 0u; step < w_val; ++step) {
            if (step > 0u) {
                #pragma unroll
                for (uint i = 0u; i < 25u; ++i) {
                    if (i >= n_lanes) A[i] = 0ul;
                }
            }
            #pragma unroll
            for (uint i = 0u; i < 25u; ++i) {
                if (i == n_lanes) A[i] ^= 0x06ul;
            }
            A[16u] ^= 0x8000000000000000ul;
            
            keccak_f1600(A);
        }
        
        #pragma unroll
        for (uint i = 0u; i < 17u; ++i) {
            if (i < n_lanes) tips[base + i] = A[i];
        }
    }
}
```