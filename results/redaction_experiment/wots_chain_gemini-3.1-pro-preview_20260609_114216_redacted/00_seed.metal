// Naive seed for the WOTS+ / SPHINCS+ chain build (one thread per chain).
//
// Each thread evolves one independent chain end-to-end: load the
// n_bytes-byte seed into a thread-private state, then iterate the
// Keccak-256 inner hash w times, truncating each digest back to
// n_bytes bytes before feeding it into the next step. The final
// truncated digest is written to the output as the chain tip.
//
// Inner hash: Keccak-f[1600] with the FIPS 202 SHA3-256 sponge framing
// (rate = 136 bytes, capacity = 64 bytes, domain pad byte = 0x06).
// All test sizes have n_bytes < rate_bytes (in-distribution n_bytes=16,
// held-out n_bytes=32; rate_bytes=136), so every chain step is a
// single-block absorb + single-block squeeze of n_bytes/8 lanes.
//
// Chain step (n_lanes = n_bytes / 8 <= 4, rate_lanes = 17):
//     state                          := 0
//     state[lane 0..n_lanes - 1]     := previous_chunk
//     state[lane n_lanes, byte 0]    ^= 0x06     // SHA3 domain
//     state[lane 16, byte 7]         ^= 0x80     // FIPS 202 final pad
//     state                          := Keccak-f[1600](state)
//     next_chunk                     := state[lane 0..n_lanes - 1]
//
// State convention: the 1600-bit state is a 5x5 array of 64-bit lanes;
// lane k (for k in 0..25) corresponds to byte positions 8*k .. 8*k + 7
// of the sponge state in little-endian, i.e. lane k holds bytes at the
// (x, y) cell with x = k % 5, y = k / 5.
//
// Buffer layout (host-fixed; preserved by candidate):
//   buffer 0: device const ulong *seeds       (n_chains * n_bytes/8)
//   buffer 1: device       ulong *tips        (n_chains * n_bytes/8)
//   buffer 2: constant uint &n_chains
//   buffer 3: constant uint &n_bytes          (chunk size; 16 in-dist, 32 held-out)
//   buffer 4: constant uint &w                (chain length)
//
// Dispatch (host-provided):
//   threadsPerGrid        = (n_chains, 1, 1)
//   threadsPerThreadgroup = (min(n_chains, 64), 1, 1)

#include <metal_stdlib>
using namespace metal;

// FIPS 202 SHA3 sponge framing for the inner hash.
constexpr constant uint  SHA3_RATE_LANES   = 17u;          // 136 bytes / 8
constexpr constant ulong SHA3_DOMAIN_WORD  = 0x06ul;       // domain byte at lane msb-side 0
constexpr constant ulong SHA3_FINAL_PAD    = 0x8000000000000000ul;  // 0x80 at byte 7 of lane 16

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

// FIPS 202 rho offsets, indexed by lane (x + 5*y).
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

kernel void wots_chain(
    device const ulong *seeds    [[buffer(0)]],
    device       ulong *tips     [[buffer(1)]],
    constant uint      &n_chains [[buffer(2)]],
    constant uint      &n_bytes  [[buffer(3)]],
    constant uint      &w        [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n_chains) return;

    uint n_lanes = n_bytes >> 3;       // n_bytes / 8; in {2, 4} at test sizes.

    // Initial absorb: load the seed chunk into state lanes 0..n_lanes-1.
    thread ulong A[25];
    for (uint i = 0u; i < 25u; ++i) A[i] = 0ul;
    uint base = idx * n_lanes;
    for (uint k = 0u; k < n_lanes; ++k) {
        A[k] = seeds[base + k];
    }

    // w sequential chain steps. The squeeze of step j sits in lanes
    // 0..n_lanes-1 and becomes the absorb of step j+1; we reset lanes
    // n_lanes..24 to zero and re-apply the SHA3 padding each time.
    for (uint step = 0u; step < w; ++step) {
        for (uint i = n_lanes; i < 25u; ++i) A[i] = 0ul;
        A[n_lanes] ^= SHA3_DOMAIN_WORD;
        A[SHA3_RATE_LANES - 1u] ^= SHA3_FINAL_PAD;
        keccak_f1600(A);
    }

    // Emit the chain tip (first n_lanes lanes of the final state).
    for (uint k = 0u; k < n_lanes; ++k) {
        tips[base + k] = A[k];
    }
}
