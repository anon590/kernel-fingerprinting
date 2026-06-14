#include <metal_stdlib>
using namespace metal;

constant uint2 KECCAK_RC[24] = {
    uint2(0x00000001, 0x00000000), uint2(0x00008082, 0x00000000), uint2(0x0000808A, 0x80000000),
    uint2(0x80008000, 0x80000000), uint2(0x0000808B, 0x00000000), uint2(0x80000001, 0x00000000),
    uint2(0x80008081, 0x80000000), uint2(0x00008009, 0x80000000), uint2(0x0000008A, 0x00000000),
    uint2(0x00000088, 0x00000000), uint2(0x80008009, 0x00000000), uint2(0x8000000A, 0x00000000),
    uint2(0x8000808B, 0x00000000), uint2(0x0000008B, 0x80000000), uint2(0x00008089, 0x80000000),
    uint2(0x00008003, 0x80000000), uint2(0x00008002, 0x80000000), uint2(0x00000080, 0x80000000),
    uint2(0x0000800A, 0x00000000), uint2(0x8000000A, 0x80000000), uint2(0x80008081, 0x80000000),
    uint2(0x00008080, 0x80000000), uint2(0x80000001, 0x00000000), uint2(0x80008008, 0x80000000)
};

inline __attribute__((always_inline)) uint2 rotl_constant(uint2 v, uint k) {
    if (k == 0) return v;
    if (k < 32) return uint2((v.x << k) | (v.y >> (32 - k)), (v.y << k) | (v.x >> (32 - k)));
    if (k == 32) return uint2(v.y, v.x);
    uint j = k - 32;
    return uint2((v.y << j) | (v.x >> (32 - j)), (v.x << j) | (v.y >> (32 - j)));
}

inline __attribute__((always_inline)) void keccak_f1600(thread uint2 *A) {
    uint2 E[25];

    #pragma unroll
    for (uint r = 0; r < 24; r += 2) {
        // --- Round r: A -> E ---
        uint2 C0 = A[0] ^ A[5] ^ A[10] ^ A[15] ^ A[20];
        uint2 C1 = A[1] ^ A[6] ^ A[11] ^ A[16] ^ A[21];
        uint2 C2 = A[2] ^ A[7] ^ A[12] ^ A[17] ^ A[22];
        uint2 C3 = A[3] ^ A[8] ^ A[13] ^ A[18] ^ A[23];
        uint2 C4 = A[4] ^ A[9] ^ A[14] ^ A[19] ^ A[24];

        uint2 D0 = C4 ^ rotl_constant(C1, 1);
        uint2 D1 = C0 ^ rotl_constant(C2, 1);
        uint2 D2 = C1 ^ rotl_constant(C3, 1);
        uint2 D3 = C2 ^ rotl_constant(C4, 1);
        uint2 D4 = C3 ^ rotl_constant(C0, 1);

        uint2 B0 = A[0] ^ D0;
        uint2 B1 = rotl_constant(A[6] ^ D1, 44);
        uint2 B2 = rotl_constant(A[12] ^ D2, 43);
        uint2 B3 = rotl_constant(A[18] ^ D3, 21);
        uint2 B4 = rotl_constant(A[24] ^ D4, 14);
        E[0] = B0 ^ (~B1 & B2) ^ KECCAK_RC[r];
        E[1] = B1 ^ (~B2 & B3);
        E[2] = B2 ^ (~B3 & B4);
        E[3] = B3 ^ (~B4 & B0);
        E[4] = B4 ^ (~B0 & B1);

        B0 = rotl_constant(A[3] ^ D3, 28);
        B1 = rotl_constant(A[9] ^ D4, 20);
        B2 = rotl_constant(A[10] ^ D0, 3);
        B3 = rotl_constant(A[16] ^ D1, 45);
        B4 = rotl_constant(A[22] ^ D2, 61);
        E[5] = B0 ^ (~B1 & B2);
        E[6] = B1 ^ (~B2 & B3);
        E[7] = B2 ^ (~B3 & B4);
        E[8] = B3 ^ (~B4 & B0);
        E[9] = B4 ^ (~B0 & B1);

        B0 = rotl_constant(A[1] ^ D1, 1);
        B1 = rotl_constant(A[7] ^ D2, 6);
        B2 = rotl_constant(A[13] ^ D3, 25);
        B3 = rotl_constant(A[19] ^ D4, 8);
        B4 = rotl_constant(A[20] ^ D0, 18);
        E[10] = B0 ^ (~B1 & B2);
        E[11] = B1 ^ (~B2 & B3);
        E[12] = B2 ^ (~B3 & B4);
        E[13] = B3 ^ (~B4 & B0);
        E[14] = B4 ^ (~B0 & B1);

        B0 = rotl_constant(A[4] ^ D4, 27);
        B1 = rotl_constant(A[5] ^ D0, 36);
        B2 = rotl_constant(A[11] ^ D1, 10);
        B3 = rotl_constant(A[17] ^ D2, 15);
        B4 = rotl_constant(A[23] ^ D3, 56);
        E[15] = B0 ^ (~B1 & B2);
        E[16] = B1 ^ (~B2 & B3);
        E[17] = B2 ^ (~B3 & B4);
        E[18] = B3 ^ (~B4 & B0);
        E[19] = B4 ^ (~B0 & B1);

        B0 = rotl_constant(A[2] ^ D2, 62);
        B1 = rotl_constant(A[8] ^ D3, 55);
        B2 = rotl_constant(A[14] ^ D4, 39);
        B3 = rotl_constant(A[15] ^ D0, 41);
        B4 = rotl_constant(A[21] ^ D1, 2);
        E[20] = B0 ^ (~B1 & B2);
        E[21] = B1 ^ (~B2 & B3);
        E[22] = B2 ^ (~B3 & B4);
        E[23] = B3 ^ (~B4 & B0);
        E[24] = B4 ^ (~B0 & B1);

        // --- Round r+1: E -> A ---
        C0 = E[0] ^ E[5] ^ E[10] ^ E[15] ^ E[20];
        C1 = E[1] ^ E[6] ^ E[11] ^ E[16] ^ E[21];
        C2 = E[2] ^ E[7] ^ E[12] ^ E[17] ^ E[22];
        C3 = E[3] ^ E[8] ^ E[13] ^ E[18] ^ E[23];
        C4 = E[4] ^ E[9] ^ E[14] ^ E[19] ^ E[24];

        D0 = C4 ^ rotl_constant(C1, 1);
        D1 = C0 ^ rotl_constant(C2, 1);
        D2 = C1 ^ rotl_constant(C3, 1);
        D3 = C2 ^ rotl_constant(C4, 1);
        D4 = C3 ^ rotl_constant(C0, 1);

        B0 = E[0] ^ D0;
        B1 = rotl_constant(E[6] ^ D1, 44);
        B2 = rotl_constant(E[12] ^ D2, 43);
        B3 = rotl_constant(E[18] ^ D3, 21);
        B4 = rotl_constant(E[24] ^ D4, 14);
        A[0] = B0 ^ (~B1 & B2) ^ KECCAK_RC[r+1];
        A[1] = B1 ^ (~B2 & B3);
        A[2] = B2 ^ (~B3 & B4);
        A[3] = B3 ^ (~B4 & B0);
        A[4] = B4 ^ (~B0 & B1);

        B0 = rotl_constant(E[3] ^ D3, 28);
        B1 = rotl_constant(E[9] ^ D4, 20);
        B2 = rotl_constant(E[10] ^ D0, 3);
        B3 = rotl_constant(E[16] ^ D1, 45);
        B4 = rotl_constant(E[22] ^ D2, 61);
        A[5] = B0 ^ (~B1 & B2);
        A[6] = B1 ^ (~B2 & B3);
        A[7] = B2 ^ (~B3 & B4);
        A[8] = B3 ^ (~B4 & B0);
        A[9] = B4 ^ (~B0 & B1);

        B0 = rotl_constant(E[1] ^ D1, 1);
        B1 = rotl_constant(E[7] ^ D2, 6);
        B2 = rotl_constant(E[13] ^ D3, 25);
        B3 = rotl_constant(E[19] ^ D4, 8);
        B4 = rotl_constant(E[20] ^ D0, 18);
        A[10] = B0 ^ (~B1 & B2);
        A[11] = B1 ^ (~B2 & B3);
        A[12] = B2 ^ (~B3 & B4);
        A[13] = B3 ^ (~B4 & B0);
        A[14] = B4 ^ (~B0 & B1);

        B0 = rotl_constant(E[4] ^ D4, 27);
        B1 = rotl_constant(E[5] ^ D0, 36);
        B2 = rotl_constant(E[11] ^ D1, 10);
        B3 = rotl_constant(E[17] ^ D2, 15);
        B4 = rotl_constant(E[23] ^ D3, 56);
        A[15] = B0 ^ (~B1 & B2);
        A[16] = B1 ^ (~B2 & B3);
        A[17] = B2 ^ (~B3 & B4);
        A[18] = B3 ^ (~B4 & B0);
        A[19] = B4 ^ (~B0 & B1);

        B0 = rotl_constant(E[2] ^ D2, 62);
        B1 = rotl_constant(E[8] ^ D3, 55);
        B2 = rotl_constant(E[14] ^ D4, 39);
        B3 = rotl_constant(E[15] ^ D0, 41);
        B4 = rotl_constant(E[21] ^ D1, 2);
        A[20] = B0 ^ (~B1 & B2);
        A[21] = B1 ^ (~B2 & B3);
        A[22] = B2 ^ (~B3 & B4);
        A[23] = B3 ^ (~B4 & B0);
        A[24] = B4 ^ (~B0 & B1);
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

    uint2 A[25];
    #pragma unroll
    for (uint i = 0; i < 25; ++i) {
        A[i] = uint2(0, 0);
    }

    uint msg_lanes  = msg_bytes  >> 3;
    uint rate_lanes = rate_bytes >> 3;
    uint out_lanes  = out_bytes  >> 3;

    uint in_base = idx * msg_lanes;

    // Standard Fast-path bypassing unrolled conditionals
    if (msg_lanes == 4) {
        A[0] ^= as_type<uint2>(in_data[in_base + 0]);
        A[1] ^= as_type<uint2>(in_data[in_base + 1]);
        A[2] ^= as_type<uint2>(in_data[in_base + 2]);
        A[3] ^= as_type<uint2>(in_data[in_base + 3]);
        A[4].x ^= (domain & 0xFFu);
    } else {
        #pragma unroll
        for (uint i = 0; i < 25; ++i) {
            if (i < msg_lanes) {
                A[i] ^= as_type<uint2>(in_data[in_base + i]);
            }
        }
        #pragma unroll
        for (uint i = 0; i < 25; ++i) {
            if (i == msg_lanes) {
                A[i].x ^= (domain & 0xFFu);
            }
        }
    }

    if (rate_lanes == 17) {
        A[16].y ^= 0x80000000u;
    } else if (rate_lanes == 21) {
        A[20].y ^= 0x80000000u;
    } else {
        #pragma unroll
        for (uint i = 0; i < 25; ++i) {
            if (i == rate_lanes - 1) {
                A[i].y ^= 0x80000000u;
            }
        }
    }

    uint out_base = idx * out_lanes;

    if (out_lanes == 4) {
        keccak_f1600(A);
        out_data[out_base + 0] = as_type<ulong>(A[0]);
        out_data[out_base + 1] = as_type<ulong>(A[1]);
        out_data[out_base + 2] = as_type<ulong>(A[2]);
        out_data[out_base + 3] = as_type<ulong>(A[3]);
    } else {
        uint written = 0;
        for (;;) {
            keccak_f1600(A);
            uint remaining = out_lanes - written;
            uint take = remaining < rate_lanes ? remaining : rate_lanes;
            
            #pragma unroll
            for (uint i = 0; i < 25; ++i) {
                if (i < take) {
                    out_data[out_base + written + i] = as_type<ulong>(A[i]);
                }
            }

            written += take;
            if (written >= out_lanes) break;
        }
    }
}