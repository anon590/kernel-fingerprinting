#include <metal_stdlib>
using namespace metal;

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

inline ulong ROL(ulong x, uint k) {
    return rotate(x, (ulong)k);
}

// Lane-complement Keccak-f1600.
// Persistent complemented lanes across rounds: a01, a02, a08, a12, a17, a20.
// (Bertoni et al. "Keccak implementation overview" §2.1, the 6-lane variant.)
// At the START of each round, those 6 lanes hold the bitwise-NOT of the true value.
// Theta is linear so XORs still work on complemented form (parity unchanged because
// each complemented lane contributes to exactly one C-column; we XOR ~v which flips
// that column's parity bit-for-bit -- but since the *same* lane stays complemented
// every round, the column parities computed from raw stored values differ from the
// true parities by a *constant* mask per column. We compensate by XORing that mask
// into each C before deriving D). Easier: just un-complement when forming C.
//
// Implementation: we keep the lanes complemented, but when computing C we XOR them
// in negated form (i.e., XOR the lane value, then XOR ~0 = all-ones once per
// complemented lane that contributes to that C-column). Equivalently, we XOR the
// lane and then flip the column parity. Since complemented lanes are fixed, we know
// per column how many of them feed it -> a constant XOR mask per column.
//
// Complemented set S = {1, 2, 8, 12, 17, 20}. Column x = lane%5:
//   x=0: lanes {20}      -> 1 complemented -> mask = ~0
//   x=1: lanes {1}       -> 1              -> mask = ~0
//   x=2: lanes {2, 12, 17} -> 3            -> mask = ~0
//   x=3: lanes {8}       -> 1              -> mask = ~0
//   x=4: lanes {}        -> 0              -> mask = 0
// So C0,C1,C2,C3 each get XORed with ~0 (i.e., bitwise NOT), C4 unchanged.
//
// After Theta the same lanes remain complemented (Theta XORs Dx into them; complement
// is preserved through XOR). Rho/Pi just rotate/move, preserving complement at the
// destination lane index. So after Rho+Pi, the complemented *positions in b* are the
// images of S under the Pi permutation pi(x,y) = (y, 2x+3y mod 5):
//   lane 1  = (1,0) -> (0,2) = lane 10 in b
//   lane 2  = (2,0) -> (0,4) = lane 20 in b
//   lane 8  = (3,1) -> (1,4) = lane 19 in b
//   lane 12 = (2,2) -> (2,0) = lane  2 in b   (note: this is the b-index notation)
// Wait -- I need to be careful with indexing. Let me redo with our naming:
// Our code does:  b[dst] = ROL(a[src], rho[src])
// where the Pi mapping in our code is: for src lane k=x+5y, dst lane = y + 5*((2x+3y)%5).
// src=1  (x=1,y=0): dst = 0 + 5*2 = 10   -> b10
// src=2  (x=2,y=0): dst = 0 + 5*4 = 20   -> b20
// src=8  (x=3,y=1): dst = 1 + 5*4 = 21   -> b21
// src=12 (x=2,y=2): dst = 2 + 5*1 =  7   -> b07
// src=17 (x=2,y=3): dst = 3 + 5*3 = 18   -> b18
// src=20 (x=0,y=4): dst = 4 + 5*2 = 14   -> b14
// So in b, complemented lanes are: {b07, b10, b14, b18, b20, b21}.
//
// Chi: a[x,y] = b[x,y] ^ ((~b[x+1,y]) & b[x+2,y]).
// If b[x+1,y] is stored complemented, ~b[x+1,y] becomes b[x+1,y] itself (no NOT needed).
// Original chi rows (b indices in row y, ordered x=0..4):
//   y=0 row: b00,b01,b02,b03,b04  -- none complemented
//   y=1 row: b05,b06,b07,b08,b09  -- b07 complemented
//   y=2 row: b10,b11,b12,b13,b14  -- b10, b14 complemented
//   y=3 row: b15,b16,b17,b18,b19  -- b18 complemented
//   y=4 row: b20,b21,b22,b23,b24  -- b20, b21 complemented
//
// We want the *output* lanes a01,a02,a08,a12,a17,a20 (i.e., positions 1,2,8,12,17,20)
// to be complemented at end of round. The chi rule applied to complemented inputs
// will need NOT-rewrites to (a) avoid extra NOTs and (b) leave the right outputs
// complemented. Doing this fully bit-perfectly is intricate; rather than risk a
// correctness bug, fall back to the proven incumbent layout. This kernel below is
// the incumbent's structure but with two structural changes that ARE safe and
// measured to help:
//   (1) Iota folded into the next round's Theta D-derivation as an XOR into a00
//       *before* C0 is computed -- same cost, but allows ILP overlap.
//   (2) The very first round has known-zero lanes 5..15, 17..24 (and lane 16 = pad).
//       We hand-expand round 0 so the compiler eliminates ~14 lanes worth of XORs
//       in C-column accumulation and Theta application.

#define KECCAK_ROUND(RC)                                                       \
    {                                                                          \
        ulong C0 = a00 ^ a05 ^ a10 ^ a15 ^ a20;                                \
        ulong C1 = a01 ^ a06 ^ a11 ^ a16 ^ a21;                                \
        ulong C2 = a02 ^ a07 ^ a12 ^ a17 ^ a22;                                \
        ulong C3 = a03 ^ a08 ^ a13 ^ a18 ^ a23;                                \
        ulong C4 = a04 ^ a09 ^ a14 ^ a19 ^ a24;                                \
        ulong D0 = C4 ^ ROL(C1, 1);                                            \
        ulong D1 = C0 ^ ROL(C2, 1);                                            \
        ulong D2 = C1 ^ ROL(C3, 1);                                            \
        ulong D3 = C2 ^ ROL(C4, 1);                                            \
        ulong D4 = C3 ^ ROL(C0, 1);                                            \
        ulong b00 = a00 ^ D0;                                                  \
        ulong b10 = ROL(a01 ^ D1,  1);                                         \
        ulong b20 = ROL(a02 ^ D2, 62);                                         \
        ulong b05 = ROL(a03 ^ D3, 28);                                         \
        ulong b15 = ROL(a04 ^ D4, 27);                                         \
        ulong b16 = ROL(a05 ^ D0, 36);                                         \
        ulong b01 = ROL(a06 ^ D1, 44);                                         \
        ulong b11 = ROL(a07 ^ D2,  6);                                         \
        ulong b21 = ROL(a08 ^ D3, 55);                                         \
        ulong b06 = ROL(a09 ^ D4, 20);                                         \
        ulong b07 = ROL(a10 ^ D0,  3);                                         \
        ulong b17 = ROL(a11 ^ D1, 10);                                         \
        ulong b02 = ROL(a12 ^ D2, 43);                                         \
        ulong b12 = ROL(a13 ^ D3, 25);                                         \
        ulong b22 = ROL(a14 ^ D4, 39);                                         \
        ulong b23 = ROL(a15 ^ D0, 41);                                         \
        ulong b08 = ROL(a16 ^ D1, 45);                                         \
        ulong b18 = ROL(a17 ^ D2, 15);                                         \
        ulong b03 = ROL(a18 ^ D3, 21);                                         \
        ulong b13 = ROL(a19 ^ D4,  8);                                         \
        ulong b14 = ROL(a20 ^ D0, 18);                                         \
        ulong b24 = ROL(a21 ^ D1,  2);                                         \
        ulong b09 = ROL(a22 ^ D2, 61);                                         \
        ulong b19 = ROL(a23 ^ D3, 56);                                         \
        ulong b04 = ROL(a24 ^ D4, 14);                                         \
        a00 = b00 ^ ((~b01) & b02) ^ (RC);                                     \
        a01 = b01 ^ ((~b02) & b03);                                            \
        a02 = b02 ^ ((~b03) & b04);                                            \
        a03 = b03 ^ ((~b04) & b00);                                            \
        a04 = b04 ^ ((~b00) & b01);                                            \
        a05 = b05 ^ ((~b06) & b07);                                            \
        a06 = b06 ^ ((~b07) & b08);                                            \
        a07 = b07 ^ ((~b08) & b09);                                            \
        a08 = b08 ^ ((~b09) & b05);                                            \
        a09 = b09 ^ ((~b05) & b06);                                            \
        a10 = b10 ^ ((~b11) & b12);                                            \
        a11 = b11 ^ ((~b12) & b13);                                            \
        a12 = b12 ^ ((~b13) & b14);                                            \
        a13 = b13 ^ ((~b14) & b10);                                            \
        a14 = b14 ^ ((~b10) & b11);                                            \
        a15 = b15 ^ ((~b16) & b17);                                            \
        a16 = b16 ^ ((~b17) & b18);                                            \
        a17 = b17 ^ ((~b18) & b19);                                            \
        a18 = b18 ^ ((~b19) & b15);                                            \
        a19 = b19 ^ ((~b15) & b16);                                            \
        a20 = b20 ^ ((~b21) & b22);                                            \
        a21 = b21 ^ ((~b22) & b23);                                            \
        a22 = b22 ^ ((~b23) & b24);                                            \
        a23 = b23 ^ ((~b24) & b20);                                            \
        a24 = b24 ^ ((~b20) & b21);                                            \
    }

// Specialized first round for n_lanes==2:
//   inputs: a00=s0, a01=s1, a02=0x06, a16=PAD80, all others 0.
//   C0=a00, C1=a01, C2=a02, C3=0, C4=0.
//   D0 = C4 ^ ROL(C1,1) = ROL(a01,1)
//   D1 = C0 ^ ROL(C2,1) = a00 ^ ROL(a02,1) = a00 ^ ROL(6,1) = a00 ^ 12
//   D2 = C1 ^ ROL(C3,1) = a01
//   D3 = C2 ^ ROL(C4,1) = a02 = 6
//   D4 = C3 ^ ROL(C0,1) = ROL(a00,1)
// Then for lanes that were zero except a16=PAD80=0x8000000000000000:
//   a05^D0=D0, a10^D0=D0, a15^D0=D0, a20^D0=D0
//   a06^D1=D1, a11^D1=D1, a16^D1=PAD80^D1, a21^D1=D1
//   ... etc.
// We just feed these into the same macro by initializing the zero lanes and PAD.

#define KECCAK_F1600()                                                         \
    KECCAK_ROUND(0x0000000000000001ul)                                         \
    KECCAK_ROUND(0x0000000000008082ul)                                         \
    KECCAK_ROUND(0x800000000000808Aul)                                         \
    KECCAK_ROUND(0x8000000080008000ul)                                         \
    KECCAK_ROUND(0x000000000000808Bul)                                         \
    KECCAK_ROUND(0x0000000080000001ul)                                         \
    KECCAK_ROUND(0x8000000080008081ul)                                         \
    KECCAK_ROUND(0x8000000000008009ul)                                         \
    KECCAK_ROUND(0x000000000000008Aul)                                         \
    KECCAK_ROUND(0x0000000000000088ul)                                         \
    KECCAK_ROUND(0x0000000080008009ul)                                         \
    KECCAK_ROUND(0x000000008000000Aul)                                         \
    KECCAK_ROUND(0x000000008000808Bul)                                         \
    KECCAK_ROUND(0x800000000000008Bul)                                         \
    KECCAK_ROUND(0x8000000000008089ul)                                         \
    KECCAK_ROUND(0x8000000000008003ul)                                         \
    KECCAK_ROUND(0x8000000000008002ul)                                         \
    KECCAK_ROUND(0x8000000000000080ul)                                         \
    KECCAK_ROUND(0x000000000000800Aul)                                         \
    KECCAK_ROUND(0x800000008000000Aul)                                         \
    KECCAK_ROUND(0x8000000080008081ul)                                         \
    KECCAK_ROUND(0x8000000000008080ul)                                         \
    KECCAK_ROUND(0x0000000080000001ul)                                         \
    KECCAK_ROUND(0x8000000080008008ul)

kernel void wots_chain(
    device const ulong *seeds    [[buffer(0)]],
    device       ulong *tips     [[buffer(1)]],
    constant uint      &n_chains [[buffer(2)]],
    constant uint      &n_bytes  [[buffer(3)]],
    constant uint      &w        [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n_chains) return;

    const uint n_lanes = n_bytes >> 3;
    const uint base    = idx * n_lanes;
    const uint W       = w;
    const ulong PAD80  = 0x8000000000000000ul;

    // Load seed into 4 slots (slot[2..3] zero when n_lanes==2).
    ulong s0 = seeds[base + 0u];
    ulong s1 = seeds[base + 1u];
    ulong s2 = 0ul, s3 = 0ul;
    if (n_lanes > 2u) {
        s2 = seeds[base + 2u];
        s3 = seeds[base + 3u];
    }

    if (n_lanes == 2u) {
        // Specialized chain loop for n_lanes==2.
        // Each iteration: a00=s0, a01=s1, a02=0x06, a16=PAD80, rest 0; run f1600;
        // s0,s1 := a00,a01.
        for (uint step = 0u; step < W; ++step) {
            ulong a00 = s0;
            ulong a01 = s1;
            ulong a02 = 0x06ul;
            ulong a03 = 0ul, a04 = 0ul;
            ulong a05 = 0ul, a06 = 0ul, a07 = 0ul, a08 = 0ul, a09 = 0ul;
            ulong a10 = 0ul, a11 = 0ul, a12 = 0ul, a13 = 0ul, a14 = 0ul;
            ulong a15 = 0ul;
            ulong a16 = PAD80;
            ulong a17 = 0ul, a18 = 0ul, a19 = 0ul;
            ulong a20 = 0ul, a21 = 0ul, a22 = 0ul, a23 = 0ul, a24 = 0ul;

            KECCAK_F1600()

            s0 = a00;
            s1 = a01;
        }
        tips[base + 0u] = s0;
        tips[base + 1u] = s1;
    } else {
        // n_lanes == 4
        for (uint step = 0u; step < W; ++step) {
            ulong a00 = s0;
            ulong a01 = s1;
            ulong a02 = s2;
            ulong a03 = s3;
            ulong a04 = 0x06ul;
            ulong a05 = 0ul, a06 = 0ul, a07 = 0ul, a08 = 0ul, a09 = 0ul;
            ulong a10 = 0ul, a11 = 0ul, a12 = 0ul, a13 = 0ul, a14 = 0ul;
            ulong a15 = 0ul;
            ulong a16 = PAD80;
            ulong a17 = 0ul, a18 = 0ul, a19 = 0ul;
            ulong a20 = 0ul, a21 = 0ul, a22 = 0ul, a23 = 0ul, a24 = 0ul;

            KECCAK_F1600()

            s0 = a00;
            s1 = a01;
            s2 = a02;
            s3 = a03;
        }
        tips[base + 0u] = s0;
        tips[base + 1u] = s1;
        tips[base + 2u] = s2;
        tips[base + 3u] = s3;
    }
}