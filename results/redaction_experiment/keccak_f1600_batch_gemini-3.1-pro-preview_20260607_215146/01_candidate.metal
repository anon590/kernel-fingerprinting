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

#define ROTL64(x, k) (((k) == 0u) ? (x) : (((x) << (k)) | ((x) >> (64u - (k)))))

inline void keccak_round(thread const ulong* A, thread ulong* B, ulong rc) {
    ulong C[5];
    #pragma unroll
    for (uint x = 0u; x < 5u; ++x) {
        C[x] = A[x] ^ A[x + 5u] ^ A[x + 10u] ^ A[x + 15u] ^ A[x + 20u];
    }
    
    ulong D[5];
    #pragma unroll
    for (uint x = 0u; x < 5u; ++x) {
        D[x] = C[(x + 4u) % 5u] ^ ROTL64(C[(x + 1u) % 5u], 1u);
    }
    
    #pragma unroll
    for (uint y = 0u; y < 5u; ++y) {
        ulong T_row[5];
        #pragma unroll
        for (uint x = 0u; x < 5u; ++x) {
            uint x_old = (x + 3u * y) % 5u;
            uint y_old = x;
            uint src = x_old + 5u * y_old;
            T_row[x] = ROTL64(A[src] ^ D[x_old], KECCAK_RHO[src]);
        }
        
        #pragma unroll
        for (uint x = 0u; x < 5u; ++x) {
            uint i1 = (x + 1u) % 5u;
            uint i2 = (x + 2u) % 5u;
            B[x + 5u * y] = T_row[x] ^ ((~T_row[i1]) & T_row[i2]);
        }
    }
    B[0] ^= rc;
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

    thread ulong A[25];
    #pragma unroll
    for (uint i = 0u; i < 25u; ++i) {
        A[i] = 0ul;
    }

    uint in_base = idx * msg_lanes;
    
    // Absorb + Pad (using static bounds to keep A in registers)
    #pragma unroll
    for (uint i = 0u; i < 25u; ++i) {
        if (i < msg_lanes) {
            A[i] ^= in_data[in_base + i];
        }
        if (i == msg_lanes) {
            A[i] ^= (ulong)(domain & 0xFFu);
        }
        if (i == rate_lanes - 1u) {
            A[i] ^= 0x8000000000000000ul;
        }
    }

    uint out_base = idx * out_lanes;
    uint written = 0u;
    thread ulong B[25];

    for (;;) {
        #pragma unroll
        for (uint r = 0u; r < 24u; r += 2u) {
            keccak_round(A, B, KECCAK_RC[r]);
            keccak_round(B, A, KECCAK_RC[r + 1u]);
        }

        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        
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