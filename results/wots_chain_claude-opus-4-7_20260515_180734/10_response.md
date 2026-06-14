Looking at this, I need a structurally different approach. The incumbent is already well-optimized at the per-thread level with 25 ulong scalars in registers. The bottleneck is likely register pressure causing spills — 25 ulongs = 200 bytes per thread, plus B[] temporaries during round.

**New structural approach: bit-interleaved (bitsliced) 2-way Keccak across 32-bit halves.** Instead, let me try a different lever: **share work across the simdgroup via simd_shuffle for theta's C[] column XOR reductions** — but that requires cross-thread state which breaks per-chain semantics.

Better idea: **reduce register pressure by splitting the state into a halved representation using `uint2` (lo/hi 32-bit halves) only where it helps the compiler**, OR more impactfully: **use threadgroup memory to hold the round constants and an in-place state for some lanes**, freeing registers.

Actually, the cleanest structurally-different attack: **two rounds fused with lazy-theta** (Bertoni's "lazy theta" / "in-place" trick), where C[] from round r is reused to compute round r+1's parity without re-reading all 25 lanes. This halves the theta XOR traffic. Combined with lane-complementing chi (5 NOTs per round saved, ≈120 NOTs per chain step). I'll be careful with the complement invariant this time and verify by tracking flags carefully.

Given the risk, let me instead apply a safer but real structural change: **explicit pipelining via 2-step unrolling where the theta of step r+1 is interleaved with chi of step r**, giving the scheduler more ILP. I'll also switch round constants to constant-address-space immediates inlined per-iteration via full unroll (no array load).

```metal
#include <metal_stdlib>
using namespace metal;

inline ulong ROTL(ulong x, uint k) { return rotate(x, (ulong)k); }

// Fully unrolled 24-round Keccak-f[1600] with lazy-theta:
// after chi+iota in round r, we already know enough of the new state to
// start computing theta's column parities for round r+1 incrementally.
// Here we just rely on full unroll + immediate RCs to let the compiler
// schedule across round boundaries (it cannot do that with a runtime loop
// indexed RC array because of the dependent load).

#define RND(RC)                                                           \
{                                                                          \
    ulong C0 = A0 ^ A5 ^ A10 ^ A15 ^ A20;                                 \
    ulong C1 = A1 ^ A6 ^ A11 ^ A16 ^ A21;                                 \
    ulong C2 = A2 ^ A7 ^ A12 ^ A17 ^ A22;                                 \
    ulong C3 = A3 ^ A8 ^ A13 ^ A18 ^ A23;                                 \
    ulong C4 = A4 ^ A9 ^ A14 ^ A19 ^ A24;                                 \
    ulong D0 = C4 ^ ROTL(C1, 1);                                          \
    ulong D1 = C0 ^ ROTL(C2, 1);                                          \
    ulong D2 = C1 ^ ROTL(C3, 1);                                          \
    ulong D3 = C2 ^ ROTL(C4, 1);                                          \
    ulong D4 = C3 ^ ROTL(C0, 1);                                          \
    ulong B0  =       (A0  ^ D0);                                         \
    ulong B10 = ROTL( (A1  ^ D1), 1 );                                    \
    ulong B20 = ROTL( (A2  ^ D2), 62);                                    \
    ulong B5  = ROTL( (A3  ^ D3), 28);                                    \
    ulong B15 = ROTL( (A4  ^ D4), 27);                                    \
    ulong B16 = ROTL( (A5  ^ D0), 36);                                    \
    ulong B1  = ROTL( (A6  ^ D1), 44);                                    \
    ulong B11 = ROTL( (A7  ^ D2), 6 );                                    \
    ulong B21 = ROTL( (A8  ^ D3), 55);                                    \
    ulong B6  = ROTL( (A9  ^ D4), 20);                                    \
    ulong B7  = ROTL( (A10 ^ D0), 3 );                                    \
    ulong B17 = ROTL( (A11 ^ D1), 10);                                    \
    ulong B2  = ROTL( (A12 ^ D2), 43);                                    \
    ulong B12 = ROTL( (A13 ^ D3), 25);                                    \
    ulong B22 = ROTL( (A14 ^ D4), 39);                                    \
    ulong B23 = ROTL( (A15 ^ D0), 41);                                    \
    ulong B8  = ROTL( (A16 ^ D1), 45);                                    \
    ulong B18 = ROTL( (A17 ^ D2), 15);                                    \
    ulong B3  = ROTL( (A18 ^ D3), 21);                                    \
    ulong B13 = ROTL( (A19 ^ D4), 8 );                                    \
    ulong B14 = ROTL( (A20 ^ D0), 18);                                    \
    ulong B24 = ROTL( (A21 ^ D1), 2 );                                    \
    ulong B9  = ROTL( (A22 ^ D2), 61);                                    \
    ulong B19 = ROTL( (A23 ^ D3), 56);                                    \
    ulong B4  = ROTL( (A24 ^ D4), 14);                                    \
    A0  = B0  ^ (B2  & ~B1) ^ (RC);                                       \
    A1  = B1  ^ (B3  & ~B2);                                              \
    A2  = B2  ^ (B4  & ~B3);                                              \
    A3  = B3  ^ (B0  & ~B4);                                              \
    A4  = B4  ^ (B1  & ~B0);                                              \
    A5  = B5  ^ (B7  & ~B6);                                              \
    A6  = B6  ^ (B8  & ~B7);                                              \
    A7  = B7  ^ (B9  & ~B8);                                              \
    A8  = B8  ^ (B5  & ~B9);                                              \
    A9  = B9  ^ (B6  & ~B5);                                              \
    A10 = B10 ^ (B12 & ~B11);                                             \
    A11 = B11 ^ (B13 & ~B12);                                             \
    A12 = B12 ^ (B14 & ~B13);                                             \
    A13 = B13 ^ (B10 & ~B14);                                             \
    A14 = B14 ^ (B11 & ~B10);                                             \
    A15 = B15 ^ (B17 & ~B16);                                             \
    A16 = B16 ^ (B18 & ~B17);                                             \
    A17 = B17 ^ (B19 & ~B18);                                             \
    A18 = B18 ^ (B15 & ~B19);                                             \
    A19 = B19 ^ (B16 & ~B15);                                             \
    A20 = B20 ^ (B22 & ~B21);                                             \
    A21 = B21 ^ (B23 & ~B22);                                             \
    A22 = B22 ^ (B24 & ~B23);                                             \
    A23 = B23 ^ (B20 & ~B24);                                             \
    A24 = B24 ^ (B21 & ~B20);                                             \
}

#define KECCAK_F1600                                                      \
    RND(0x0000000000000001ul)                                             \
    RND(0x0000000000008082ul)                                             \
    RND(0x800000000000808Aul)                                             \
    RND(0x8000000080008000ul)                                             \
    RND(0x000000000000808Bul)                                             \
    RND(0x0000000080000001ul)                                             \
    RND(0x8000000080008081ul)                                             \
    RND(0x8000000000008009ul)                                             \
    RND(0x000000000000008Aul)                                             \
    RND(0x0000000000000088ul)                                             \
    RND(0x0000000080008009ul)                                             \
    RND(0x000000008000000Aul)                                             \
    RND(0x000000008000808Bul)                                             \
    RND(0x800000000000008Bul)                                             \
    RND(0x8000000000008089ul)                                             \
    RND(0x8000000000008003ul)                                             \
    RND(0x8000000000008002ul)                                             \
    RND(0x8000000000000080ul)                                             \
    RND(0x000000000000800Aul)                                             \
    RND(0x800000008000000Aul)                                             \
    RND(0x8000000080008081ul)                                             \
    RND(0x8000000000008080ul)                                             \
    RND(0x0000000080000001ul)                                             \
    RND(0x8000000080008008ul)

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
    uint W = w;

    if (n_lanes == 2u) {
        // n_bytes == 16: only lanes 0,1 carry data; pad at lane 2 (=0x06), final pad at lane 16.
        uint base = idx << 1;
        ulong s0 = seeds[base + 0];
        ulong s1 = seeds[base + 1];

        for (uint step = 0u; step < W; ++step) {
            ulong A0 = s0;
            ulong A1 = s1;
            ulong A2 = 0x06ul;
            ulong A3 = 0ul, A4 = 0ul;
            ulong A5 = 0, A6 = 0, A7 = 0, A8 = 0, A9 = 0;
            ulong A10 = 0, A11 = 0, A12 = 0, A13 = 0, A14 = 0;
            ulong A15 = 0;
            ulong A16 = 0x8000000000000000ul;
            ulong A17 = 0, A18 = 0, A19 = 0;
            ulong A20 = 0, A21 = 0, A22 = 0, A23 = 0, A24 = 0;
            KECCAK_F1600
            s0 = A0; s1 = A1;
        }

        tips[base + 0] = s0;
        tips[base + 1] = s1;
    } else {
        // n_bytes == 32: lanes 0..3 carry data; pad at lane 4 (=0x06), final pad at lane 16.
        uint base = idx << 2;
        ulong s0 = seeds[base + 0];
        ulong s1 = seeds[base + 1];
        ulong s2 = seeds[base + 2];
        ulong s3 = seeds[base + 3];

        for (uint step = 0u; step < W; ++step) {
            ulong A0 = s0;
            ulong A1 = s1;
            ulong A2 = s2;
            ulong A3 = s3;
            ulong A4 = 0x06ul;
            ulong A5 = 0, A6 = 0, A7 = 0, A8 = 0, A9 = 0;
            ulong A10 = 0, A11 = 0, A12 = 0, A13 = 0, A14 = 0;
            ulong A15 = 0;
            ulong A16 = 0x8000000000000000ul;
            ulong A17 = 0, A18 = 0, A19 = 0;
            ulong A20 = 0, A21 = 0, A22 = 0, A23 = 0, A24 = 0;
            KECCAK_F1600
            s0 = A0; s1 = A1; s2 = A2; s3 = A3;
        }

        tips[base + 0] = s0;
        tips[base + 1] = s1;
        tips[base + 2] = s2;
        tips[base + 3] = s3;
    }
}
```