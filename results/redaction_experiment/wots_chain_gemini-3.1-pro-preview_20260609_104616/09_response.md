To optimize the Keccak-f[1600] permutation for the WOTS+ chains, we leverage the Metal compiler's aggressive SROA (Scalar Replacement of Aggregates) and constant folding. By restructuring the zero-initialization of the Keccak state using `KeccakState s = {}`, we reduce IR bloat and guarantee that LLVM perfectly folds the sparse `0` lanes into the first round of every chain step. Furthermore, we tweak the critical path of the Theta/Chi steps (e.g., re-associating `E[0] = (b0 ^ rc) ^ (b2 & ~b1)`) to expose more Instruction Level Parallelism (ILP) and allow the GPU's dual-issue ALUs to overlap the constant XOR with the bit-clear `bic` operations. We specifically preserve the separate branches for statically-known `n_lanes` sizes (2 and 4) to ensure the dynamic array indexing does not force the Keccak state to spill from registers to thread-local memory.

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
    E[0] = (b0 ^ rc) ^ (b2 & ~b1); \
    E[1] = b1 ^ (b3 & ~b2); \
    E[2] = b2 ^ (b4 & ~b3); \
    E[3] = b3 ^ (b0 & ~b4); \
    E[4] = b4 ^ (b1 & ~b0); \
    \
    b0 = rotl64(A[3] ^ D3, 28u); \
    b1 = rotl64(A[9] ^ D4, 20u); \
    b2 = rotl64(A[10] ^ D0, 3u); \
    b3 = rotl64(A[16] ^ D1, 45u); \
    b4 = rotl64(A[22] ^ D2, 61u); \
    E[5] = b0 ^ (b2 & ~b1); \
    E[6] = b1 ^ (b3 & ~b2); \
    E[7] = b2 ^ (b4 & ~b3); \
    E[8] = b3 ^ (b0 & ~b4); \
    E[9] = b4 ^ (b1 & ~b0); \
    \
    b0 = rotl64(A[1] ^ D1, 1u); \
    b1 = rotl64(A[7] ^ D2, 6u); \
    b2 = rotl64(A[13] ^ D3, 25u); \
    b3 = rotl64(A[19] ^ D4, 8u); \
    b4 = rotl64(A[20] ^ D0, 18u); \
    E[10] = b0 ^ (b2 & ~b1); \
    E[11] = b1 ^ (b3 & ~b2); \
    E[12] = b2 ^ (b4 & ~b3); \
    E[13] = b3 ^ (b0 & ~b4); \
    E[14] = b4 ^ (b1 & ~b0); \
    \
    b0 = rotl64(A[4] ^ D4, 27u); \
    b1 = rotl64(A[5] ^ D0, 36u); \
    b2 = rotl64(A[11] ^ D1, 10u); \
    b3 = rotl64(A[17] ^ D2, 15u); \
    b4 = rotl64(A[23] ^ D3, 56u); \
    E[15] = b0 ^ (b2 & ~b1); \
    E[16] = b1 ^ (b3 & ~b2); \
    E[17] = b2 ^ (b4 & ~b3); \
    E[18] = b3 ^ (b0 & ~b4); \
    E[19] = b4 ^ (b1 & ~b0); \
    \
    b0 = rotl64(A[2] ^ D2, 62u); \
    b1 = rotl64(A[8] ^ D3, 55u); \
    b2 = rotl64(A[14] ^ D4, 39u); \
    b3 = rotl64(A[15] ^ D0, 41u); \
    b4 = rotl64(A[21] ^ D1, 2u); \
    E[20] = b0 ^ (b2 & ~b1); \
    E[21] = b1 ^ (b3 & ~b2); \
    E[22] = b2 ^ (b4 & ~b3); \
    E[23] = b3 ^ (b0 & ~b4); \
    E[24] = b4 ^ (b1 & ~b0); \
} while(0)

struct KeccakState {
    ulong A[25];
};

inline __attribute__((always_inline))
KeccakState keccak_f1600(KeccakState s) {
    KeccakState e;
    #pragma unroll(12)
    for (uint r = 0u; r < 24u; r += 2u) {
        ROUND(s.A, e.A, KECCAK_RC[r]);
        ROUND(e.A, s.A, KECCAK_RC[r+1]);
    }
    return s;
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
        device const ulong2 *seeds_v = (device const ulong2 *)seeds;
        ulong2 seed_val = seeds_v[idx];
        
        ulong A0 = seed_val.x;
        ulong A1 = seed_val.y;
        
        for (uint step = 0u; step < w_val; ++step) {
            KeccakState s = {};
            s.A[0] = A0;
            s.A[1] = A1;
            s.A[2] = 0x06ul;
            s.A[16] = 0x8000000000000000ul;
            
            s = keccak_f1600(s);
            
            A0 = s.A[0];
            A1 = s.A[1];
        }
        
        device ulong2 *tips_v = (device ulong2 *)tips;
        tips_v[idx] = ulong2(A0, A1);

    } else if (n_lanes == 4u) {
        device const ulong4 *seeds_v = (device const ulong4 *)seeds;
        ulong4 seed_val = seeds_v[idx];
        
        ulong A0 = seed_val.x;
        ulong A1 = seed_val.y;
        ulong A2 = seed_val.z;
        ulong A3 = seed_val.w;
        
        for (uint step = 0u; step < w_val; ++step) {
            KeccakState s = {};
            s.A[0] = A0;
            s.A[1] = A1;
            s.A[2] = A2;
            s.A[3] = A3;
            s.A[4] = 0x06ul;
            s.A[16] = 0x8000000000000000ul;
            
            s = keccak_f1600(s);
            
            A0 = s.A[0];
            A1 = s.A[1];
            A2 = s.A[2];
            A3 = s.A[3];
        }
        
        device ulong4 *tips_v = (device ulong4 *)tips;
        tips_v[idx] = ulong4(A0, A1, A2, A3);

    } else {
        uint base = idx * n_lanes;
        KeccakState s;
        #pragma unroll 24
        for (uint i = 0u; i < 25u; ++i) {
            s.A[i] = (i < n_lanes) ? seeds[base + i] : 0ul;
        }
        
        for (uint step = 0u; step < w_val; ++step) {
            if (step > 0u) {
                #pragma unroll 24
                for (uint i = 0u; i < 25u; ++i) {
                    if (i >= n_lanes) s.A[i] = 0ul;
                }
            }
            s.A[n_lanes] ^= 0x06ul;
            s.A[16u]     ^= 0x8000000000000000ul;
            
            s = keccak_f1600(s);
        }
        
        #pragma unroll 24
        for (uint i = 0u; i < 17u; ++i) {
            if (i < n_lanes) tips[base + i] = s.A[i];
        }
    }
}
```