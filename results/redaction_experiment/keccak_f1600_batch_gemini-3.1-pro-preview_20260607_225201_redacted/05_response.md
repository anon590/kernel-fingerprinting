To push beyond the incumbent's performance limit, we must optimize the arithmetic for the underlying hardware. Apple Silicon GPUs are natively 32-bit and lack a single-instruction 64-bit bitwise rotate. If we rely on `ulong` rotations, LLVM may fall back to suboptimal multi-instruction sequences.

This optimization explicitly splits the 64-bit Keccak state array into a 32-bit `uint2` representation. We define a custom `rotl_constant` function that perfectly maps to the native 32-bit `extr` (funnel shift) instructions. Since all rotation offsets in Keccak are constants, the compiler evaluates the shift branches at compile-time, guaranteeing exactly two 32-bit instructions per 64-bit rotate. Metal's native element-wise vector operations cleanly handle the `^`, `&`, and `~` logic on `uint2` without any 64-bit emulation overhead. The fallthrough switch statements are preserved to guarantee that the `uint2 A[25]` state remains un-indexed by variables, enforcing optimal register allocation without spilling to thread memory.

```metal
#include <metal_stdlib>
using namespace metal;

inline __attribute__((always_inline)) uint2 rotl_constant(uint2 v, uint k) {
    if (k == 0) {
        return v;
    } else if (k < 32) {
        return uint2((v.x << k) | (v.y >> (32 - k)), (v.y << k) | (v.x >> (32 - k)));
    } else if (k == 32) {
        return uint2(v.y, v.x);
    } else {
        uint j = k - 32;
        return uint2((v.y << j) | (v.x >> (32 - j)), (v.x << j) | (v.y >> (32 - j)));
    }
}

inline __attribute__((always_inline)) void keccak_f1600(thread uint2 *A) {
    constexpr uint2 RC[24] = {
        uint2(0x00000001, 0x00000000), uint2(0x00008082, 0x00000000), uint2(0x0000808A, 0x80000000),
        uint2(0x80008000, 0x80000000), uint2(0x0000808B, 0x00000000), uint2(0x80000001, 0x00000000),
        uint2(0x80008081, 0x80000000), uint2(0x00008009, 0x80000000), uint2(0x0000008A, 0x00000000),
        uint2(0x00000088, 0x00000000), uint2(0x80008009, 0x00000000), uint2(0x8000000A, 0x00000000),
        uint2(0x8000808B, 0x00000000), uint2(0x0000008B, 0x80000000), uint2(0x00008089, 0x80000000),
        uint2(0x00008003, 0x80000000), uint2(0x00008002, 0x80000000), uint2(0x00000080, 0x80000000),
        uint2(0x0000800A, 0x00000000), uint2(0x8000000A, 0x80000000), uint2(0x80008081, 0x80000000),
        uint2(0x00008080, 0x80000000), uint2(0x80000001, 0x00000000), uint2(0x80008008, 0x80000000),
    };

    uint2 E[25];

    #pragma unroll 12
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
        E[0] = B0 ^ (~B1 & B2) ^ RC[r];
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
        A[0] = B0 ^ (~B1 & B2) ^ RC[r+1];
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

    // Fast-path for common input length
    if (msg_lanes == 4) {
        A[0] ^= as_type<uint2>(in_data[in_base + 0]);
        A[1] ^= as_type<uint2>(in_data[in_base + 1]);
        A[2] ^= as_type<uint2>(in_data[in_base + 2]);
        A[3] ^= as_type<uint2>(in_data[in_base + 3]);
        A[4].x ^= (domain & 0xFFu);
    } else {
        switch (msg_lanes) {
            case 24: A[23] ^= as_type<uint2>(in_data[in_base + 23]);
            case 23: A[22] ^= as_type<uint2>(in_data[in_base + 22]);
            case 22: A[21] ^= as_type<uint2>(in_data[in_base + 21]);
            case 21: A[20] ^= as_type<uint2>(in_data[in_base + 20]);
            case 20: A[19] ^= as_type<uint2>(in_data[in_base + 19]);
            case 19: A[18] ^= as_type<uint2>(in_data[in_base + 18]);
            case 18: A[17] ^= as_type<uint2>(in_data[in_base + 17]);
            case 17: A[16] ^= as_type<uint2>(in_data[in_base + 16]);
            case 16: A[15] ^= as_type<uint2>(in_data[in_base + 15]);
            case 15: A[14] ^= as_type<uint2>(in_data[in_base + 14]);
            case 14: A[13] ^= as_type<uint2>(in_data[in_base + 13]);
            case 13: A[12] ^= as_type<uint2>(in_data[in_base + 12]);
            case 12: A[11] ^= as_type<uint2>(in_data[in_base + 11]);
            case 11: A[10] ^= as_type<uint2>(in_data[in_base + 10]);
            case 10: A[ 9] ^= as_type<uint2>(in_data[in_base +  9]);
            case  9: A[ 8] ^= as_type<uint2>(in_data[in_base +  8]);
            case  8: A[ 7] ^= as_type<uint2>(in_data[in_base +  7]);
            case  7: A[ 6] ^= as_type<uint2>(in_data[in_base +  6]);
            case  6: A[ 5] ^= as_type<uint2>(in_data[in_base +  5]);
            case  5: A[ 4] ^= as_type<uint2>(in_data[in_base +  4]);
            case  4: A[ 3] ^= as_type<uint2>(in_data[in_base +  3]);
            case  3: A[ 2] ^= as_type<uint2>(in_data[in_base +  2]);
            case  2: A[ 1] ^= as_type<uint2>(in_data[in_base +  1]);
            case  1: A[ 0] ^= as_type<uint2>(in_data[in_base +  0]); break;
            default: break;
        }
        
        switch (msg_lanes) {
            case 24: A[24].x ^= (domain & 0xFFu); break;
            case 23: A[23].x ^= (domain & 0xFFu); break;
            case 22: A[22].x ^= (domain & 0xFFu); break;
            case 21: A[21].x ^= (domain & 0xFFu); break;
            case 20: A[20].x ^= (domain & 0xFFu); break;
            case 19: A[19].x ^= (domain & 0xFFu); break;
            case 18: A[18].x ^= (domain & 0xFFu); break;
            case 17: A[17].x ^= (domain & 0xFFu); break;
            case 16: A[16].x ^= (domain & 0xFFu); break;
            case 15: A[15].x ^= (domain & 0xFFu); break;
            case 14: A[14].x ^= (domain & 0xFFu); break;
            case 13: A[13].x ^= (domain & 0xFFu); break;
            case 12: A[12].x ^= (domain & 0xFFu); break;
            case 11: A[11].x ^= (domain & 0xFFu); break;
            case 10: A[10].x ^= (domain & 0xFFu); break;
            case  9: A[ 9].x ^= (domain & 0xFFu); break;
            case  8: A[ 8].x ^= (domain & 0xFFu); break;
            case  7: A[ 7].x ^= (domain & 0xFFu); break;
            case  6: A[ 6].x ^= (domain & 0xFFu); break;
            case  5: A[ 5].x ^= (domain & 0xFFu); break;
            case  4: A[ 4].x ^= (domain & 0xFFu); break;
            case  3: A[ 3].x ^= (domain & 0xFFu); break;
            case  2: A[ 2].x ^= (domain & 0xFFu); break;
            case  1: A[ 1].x ^= (domain & 0xFFu); break;
            case  0: A[ 0].x ^= (domain & 0xFFu); break;
            default: break;
        }
    }

    // Fast-paths for common SHA3-256 (rate_lanes=17) and SHAKE-128 (rate_lanes=21) paddings
    if (rate_lanes == 17) {
        A[16].y ^= 0x80000000u;
    } else if (rate_lanes == 21) {
        A[20].y ^= 0x80000000u;
    } else {
        switch (rate_lanes - 1) {
            case 24: A[24].y ^= 0x80000000u; break;
            case 23: A[23].y ^= 0x80000000u; break;
            case 22: A[22].y ^= 0x80000000u; break;
            case 21: A[21].y ^= 0x80000000u; break;
            case 20: A[20].y ^= 0x80000000u; break;
            case 19: A[19].y ^= 0x80000000u; break;
            case 18: A[18].y ^= 0x80000000u; break;
            case 17: A[17].y ^= 0x80000000u; break;
            case 16: A[16].y ^= 0x80000000u; break;
            case 15: A[15].y ^= 0x80000000u; break;
            case 14: A[14].y ^= 0x80000000u; break;
            case 13: A[13].y ^= 0x80000000u; break;
            case 12: A[12].y ^= 0x80000000u; break;
            case 11: A[11].y ^= 0x80000000u; break;
            case 10: A[10].y ^= 0x80000000u; break;
            case  9: A[ 9].y ^= 0x80000000u; break;
            case  8: A[ 8].y ^= 0x80000000u; break;
            case  7: A[ 7].y ^= 0x80000000u; break;
            case  6: A[ 6].y ^= 0x80000000u; break;
            case  5: A[ 5].y ^= 0x80000000u; break;
            case  4: A[ 4].y ^= 0x80000000u; break;
            case  3: A[ 3].y ^= 0x80000000u; break;
            case  2: A[ 2].y ^= 0x80000000u; break;
            case  1: A[ 1].y ^= 0x80000000u; break;
            case  0: A[ 0].y ^= 0x80000000u; break;
            default: break;
        }
    }

    uint out_base = idx * out_lanes;

    // Fast-path for standard 32-byte outputs
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
            
            switch (take) {
                case 25: out_data[out_base + written + 24] = as_type<ulong>(A[24]);
                case 24: out_data[out_base + written + 23] = as_type<ulong>(A[23]);
                case 23: out_data[out_base + written + 22] = as_type<ulong>(A[22]);
                case 22: out_data[out_base + written + 21] = as_type<ulong>(A[21]);
                case 21: out_data[out_base + written + 20] = as_type<ulong>(A[20]);
                case 20: out_data[out_base + written + 19] = as_type<ulong>(A[19]);
                case 19: out_data[out_base + written + 18] = as_type<ulong>(A[18]);
                case 18: out_data[out_base + written + 17] = as_type<ulong>(A[17]);
                case 17: out_data[out_base + written + 16] = as_type<ulong>(A[16]);
                case 16: out_data[out_base + written + 15] = as_type<ulong>(A[15]);
                case 15: out_data[out_base + written + 14] = as_type<ulong>(A[14]);
                case 14: out_data[out_base + written + 13] = as_type<ulong>(A[13]);
                case 13: out_data[out_base + written + 12] = as_type<ulong>(A[12]);
                case 12: out_data[out_base + written + 11] = as_type<ulong>(A[11]);
                case 11: out_data[out_base + written + 10] = as_type<ulong>(A[10]);
                case 10: out_data[out_base + written +  9] = as_type<ulong>(A[ 9]);
                case  9: out_data[out_base + written +  8] = as_type<ulong>(A[ 8]);
                case  8: out_data[out_base + written +  7] = as_type<ulong>(A[ 7]);
                case  7: out_data[out_base + written +  6] = as_type<ulong>(A[ 6]);
                case  6: out_data[out_base + written +  5] = as_type<ulong>(A[ 5]);
                case  5: out_data[out_base + written +  4] = as_type<ulong>(A[ 4]);
                case  4: out_data[out_base + written +  3] = as_type<ulong>(A[ 3]);
                case  3: out_data[out_base + written +  2] = as_type<ulong>(A[ 2]);
                case  2: out_data[out_base + written +  1] = as_type<ulong>(A[ 1]);
                case  1: out_data[out_base + written +  0] = as_type<ulong>(A[ 0]); break;
                default: break;
            }

            written += take;
            if (written >= out_lanes) break;
        }
    }
}
```