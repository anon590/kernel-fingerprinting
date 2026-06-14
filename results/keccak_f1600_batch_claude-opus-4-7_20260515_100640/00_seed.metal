// Naive seed for batched Keccak-f[1600] sponge (one thread per instance).
//
// Each thread runs the full pipeline for one independent sponge:
// initialise a 25-lane 64-bit state to zero, XOR ``msg_bytes / 8``
// input lanes into the state, apply the FIPS 202 padding, then
// alternate Keccak-f[1600] permutations with ``rate_bytes / 8`` lane
// writes to the output until ``out_bytes / 8`` lanes have been
// emitted (the last chunk may be shorter than the rate).
//
// All test sizes have ``msg_bytes < rate_bytes`` (one absorb block)
// and ``msg_bytes``, ``rate_bytes``, ``out_bytes`` all multiples of 8.
// Lane k of the 5x5 state corresponds to the (x, y) cell with
// x = k % 5, y = k / 5 (lane index k = x + 5*y).
//
// Buffer layout (host-fixed; preserved by candidate):
//   buffer 0: device const ulong *in_data    (batch * msg_bytes/8)
//   buffer 1: device       ulong *out_data   (batch * out_bytes/8)
//   buffer 2: constant uint &batch
//   buffer 3: constant uint &msg_bytes
//   buffer 4: constant uint &rate_bytes
//   buffer 5: constant uint &out_bytes
//   buffer 6: constant uint &domain          (low 8 bits = padding domain byte)
//
// Dispatch (host-provided):
//   threadsPerGrid        = (batch, 1, 1)
//   threadsPerThreadgroup = (min(batch, 64), 1, 1)

#include <metal_stdlib>
using namespace metal;

// FIPS 202 round constants for Keccak-f[1600].
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

// FIPS 202 rho offsets, indexed by lane (x + 5*y) for x,y in 0..5.
constant uint KECCAK_RHO[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
};

inline ulong rotl64(ulong x, uint k) {
    k &= 63u;
    if (k == 0u) return x;
    return (x << k) | (x >> (64u - k));
}

inline void keccak_f1600(thread ulong *A) {
    ulong C[5];
    ulong D[5];
    ulong B[25];
    for (uint r = 0u; r < 24u; ++r) {
        // theta: column XOR + 1-bit-rotated lateral mix.
        for (uint x = 0u; x < 5u; ++x) {
            C[x] = A[x] ^ A[x + 5u] ^ A[x + 10u] ^ A[x + 15u] ^ A[x + 20u];
        }
        for (uint x = 0u; x < 5u; ++x) {
            D[x] = C[(x + 4u) % 5u] ^ rotl64(C[(x + 1u) % 5u], 1u);
        }
        for (uint y = 0u; y < 5u; ++y) {
            for (uint x = 0u; x < 5u; ++x) {
                A[x + 5u * y] ^= D[x];
            }
        }

        // rho + pi: rotate each lane by r[x][y] and scatter to
        // destination cell (x_new, y_new) = (y, (2*x + 3*y) % 5).
        for (uint y = 0u; y < 5u; ++y) {
            for (uint x = 0u; x < 5u; ++x) {
                uint src = x + 5u * y;
                uint x_new = y;
                uint y_new = (2u * x + 3u * y) % 5u;
                B[x_new + 5u * y_new] = rotl64(A[src], KECCAK_RHO[src]);
            }
        }

        // chi: nonlinear row mix.
        for (uint y = 0u; y < 5u; ++y) {
            for (uint x = 0u; x < 5u; ++x) {
                uint i  = x + 5u * y;
                uint i1 = ((x + 1u) % 5u) + 5u * y;
                uint i2 = ((x + 2u) % 5u) + 5u * y;
                A[i] = B[i] ^ ((~B[i1]) & B[i2]);
            }
        }

        // iota.
        A[0] ^= KECCAK_RC[r];
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

    uint msg_lanes  = msg_bytes  >> 3;   // msg_bytes  / 8
    uint rate_lanes = rate_bytes >> 3;   // rate_bytes / 8
    uint out_lanes  = out_bytes  >> 3;   // out_bytes  / 8

    thread ulong A[25];
    for (uint i = 0u; i < 25u; ++i) A[i] = 0ul;

    // Absorb single block: XOR message lanes into state lanes 0..msg_lanes-1.
    uint in_base = idx * msg_lanes;
    for (uint i = 0u; i < msg_lanes; ++i) {
        A[i] ^= in_data[in_base + i];
    }
    // Pad domain byte at byte position msg_bytes
    // (lane msg_lanes, byte 0 of that lane).
    A[msg_lanes] ^= (ulong)(domain & 0xFFu);
    // Pad 0x80 at byte position rate_bytes - 1
    // (lane rate_lanes - 1, byte 7 of that lane).
    A[rate_lanes - 1u] ^= 0x8000000000000000ul;

    // Permute then squeeze; if out_lanes > rate_lanes, run additional
    // permutations and emit further chunks until done.
    uint out_base = idx * out_lanes;
    uint written = 0u;
    for (;;) {
        keccak_f1600(A);
        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        for (uint j = 0u; j < take; ++j) {
            out_data[out_base + written + j] = A[j];
        }
        written += take;
        if (written >= out_lanes) break;
    }
}
