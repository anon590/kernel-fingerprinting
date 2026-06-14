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

        // Mutating walk for Theta + Rho + Pi without temporary array B
        ulong current = A[1] ^ D[1];
        uint curr_x = 1u;
        uint curr_y = 0u;
        ulong A00 = A[0] ^ D[0];

        #pragma unroll
        for (uint t = 0u; t < 24u; ++t) {
            uint dest_x = curr_y;
            uint dest_y = (2u * curr_x + 3u * curr_y) % 5u;
            uint dest_idx = dest_x + 5u * dest_y;
            uint orig_idx = curr_x + 5u * curr_y;
            
            ulong saved = A[dest_idx] ^ D[dest_x];
            A[dest_idx] = rotl64(current, KECCAK_RHO[orig_idx]);
            
            current = saved;
            curr_x = dest_x;
            curr_y = dest_y;
        }
        A[0] = A00;

        #pragma unroll
        for (uint y = 0u; y < 5u; ++y) {
            uint base_y = y * 5u;
            ulong T0 = A[base_y + 0u];
            ulong T1 = A[base_y + 1u];
            ulong T2 = A[base_y + 2u];
            ulong T3 = A[base_y + 3u];
            ulong T4 = A[base_y + 4u];

            A[base_y + 0u] = T0 ^ ((~T1) & T2);
            A[base_y + 1u] = T1 ^ ((~T2) & T3);
            A[base_y + 2u] = T2 ^ ((~T3) & T4);
            A[base_y + 3u] = T3 ^ ((~T4) & T0);
            A[base_y + 4u] = T4 ^ ((~T0) & T1);
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
    uint w_val = w;

    thread ulong A[25];

    if (n_lanes == 2u) {
        A[0] = seeds[base + 0u];
        A[1] = seeds[base + 1u];
        
        for (uint step = 0u; step < w_val; ++step) {
            A[2] = 0x06ul;
            A[3] = 0ul; A[4] = 0ul; A[5] = 0ul; A[6] = 0ul;
            A[7] = 0ul; A[8] = 0ul; A[9] = 0ul; A[10] = 0ul;
            A[11] = 0ul; A[12] = 0ul; A[13] = 0ul; A[14] = 0ul;
            A[15] = 0ul; 
            A[16] = 0x8000000000000000ul;
            A[17] = 0ul; A[18] = 0ul; A[19] = 0ul; A[20] = 0ul;
            A[21] = 0ul; A[22] = 0ul; A[23] = 0ul; A[24] = 0ul;
            
            keccak_f1600(A);
        }
        
        tips[base + 0u] = A[0];
        tips[base + 1u] = A[1];

    } else if (n_lanes == 4u) {
        A[0] = seeds[base + 0u];
        A[1] = seeds[base + 1u];
        A[2] = seeds[base + 2u];
        A[3] = seeds[base + 3u];
        
        for (uint step = 0u; step < w_val; ++step) {
            A[4] = 0x06ul;
            A[5] = 0ul; A[6] = 0ul;
            A[7] = 0ul; A[8] = 0ul; A[9] = 0ul; A[10] = 0ul;
            A[11] = 0ul; A[12] = 0ul; A[13] = 0ul; A[14] = 0ul;
            A[15] = 0ul; 
            A[16] = 0x8000000000000000ul;
            A[17] = 0ul; A[18] = 0ul; A[19] = 0ul; A[20] = 0ul;
            A[21] = 0ul; A[22] = 0ul; A[23] = 0ul; A[24] = 0ul;
            
            keccak_f1600(A);
        }
        
        tips[base + 0u] = A[0];
        tips[base + 1u] = A[1];
        tips[base + 2u] = A[2];
        tips[base + 3u] = A[3];

    } else {
        #pragma unroll
        for (uint i = 0u; i < 25u; ++i) {
            A[i] = (i < n_lanes) ? seeds[base + i] : 0ul;
        }
        
        for (uint step = 0u; step < w_val; ++step) {
            if (step > 0u) {
                #pragma unroll
                for (uint i = 0u; i < 25u; ++i) {
                    if (i >= n_lanes) A[i] = 0ul;
                }
            }
            A[n_lanes] ^= 0x06ul;
            A[16u] ^= 0x8000000000000000ul;
            
            keccak_f1600(A);
        }
        
        #pragma unroll
        for (uint i = 0u; i < 17u; ++i) {
            if (i < n_lanes) tips[base + i] = A[i];
        }
    }
}