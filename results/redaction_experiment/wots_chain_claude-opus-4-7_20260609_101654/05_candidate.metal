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
    bool n2 = (n_lanes == 2u);

    ulong s0 = seeds[base + 0u];
    ulong s1 = seeds[base + 1u];
    ulong s2 = n2 ? 0ul : seeds[base + 2u];
    ulong s3 = n2 ? 0ul : seeds[base + 3u];

    const ulong PAD_FINAL = 0x8000000000000000ul;
    const ulong PAD_DOM   = 0x06ul;

    // Pre-computed padding constants for the fixed lanes 2..24.
    // If n_lanes==2: lane 2 = 0x06, lane 3 = 0, lane 4 = 0
    // If n_lanes==4: lane 2 = s2,   lane 3 = s3, lane 4 = 0x06
    // Lane 16 = PAD_FINAL always; all others 0.

    for (uint step = 0u; step < w; ++step) {
        // Initialize state with padding folded in.
        ulong a00 = s0;
        ulong a01 = s1;
        ulong a02 = n2 ? PAD_DOM : s2;
        ulong a03 = n2 ? 0ul     : s3;
        ulong a04 = n2 ? 0ul     : PAD_DOM;
        ulong a05 = 0ul, a06 = 0ul, a07 = 0ul, a08 = 0ul, a09 = 0ul;
        ulong a10 = 0ul, a11 = 0ul, a12 = 0ul, a13 = 0ul, a14 = 0ul;
        ulong a15 = 0ul;
        ulong a16 = PAD_FINAL;
        ulong a17 = 0ul, a18 = 0ul, a19 = 0ul;
        ulong a20 = 0ul, a21 = 0ul, a22 = 0ul, a23 = 0ul, a24 = 0ul;

        for (uint r = 0u; r < 24u; ++r) {
            // Theta
            ulong C0 = a00 ^ a05 ^ a10 ^ a15 ^ a20;
            ulong C1 = a01 ^ a06 ^ a11 ^ a16 ^ a21;
            ulong C2 = a02 ^ a07 ^ a12 ^ a17 ^ a22;
            ulong C3 = a03 ^ a08 ^ a13 ^ a18 ^ a23;
            ulong C4 = a04 ^ a09 ^ a14 ^ a19 ^ a24;

            ulong D0 = C4 ^ ROL(C1, 1);
            ulong D1 = C0 ^ ROL(C2, 1);
            ulong D2 = C1 ^ ROL(C3, 1);
            ulong D3 = C2 ^ ROL(C4, 1);
            ulong D4 = C3 ^ ROL(C0, 1);

            // Rho + Pi with D folded directly into the rotate operand.
            // Destination lane = (y, (2x+3y)%5) from source (x,y).
            ulong b00 = a00 ^ D0;
            ulong b10 = ROL(a01 ^ D1,  1);
            ulong b20 = ROL(a02 ^ D2, 62);
            ulong b05 = ROL(a03 ^ D3, 28);
            ulong b15 = ROL(a04 ^ D4, 27);

            ulong b16 = ROL(a05 ^ D0, 36);
            ulong b01 = ROL(a06 ^ D1, 44);
            ulong b11 = ROL(a07 ^ D2,  6);
            ulong b21 = ROL(a08 ^ D3, 55);
            ulong b06 = ROL(a09 ^ D4, 20);

            ulong b07 = ROL(a10 ^ D0,  3);
            ulong b17 = ROL(a11 ^ D1, 10);
            ulong b02 = ROL(a12 ^ D2, 43);
            ulong b12 = ROL(a13 ^ D3, 25);
            ulong b22 = ROL(a14 ^ D4, 39);

            ulong b23 = ROL(a15 ^ D0, 41);
            ulong b08 = ROL(a16 ^ D1, 45);
            ulong b18 = ROL(a17 ^ D2, 15);
            ulong b03 = ROL(a18 ^ D3, 21);
            ulong b13 = ROL(a19 ^ D4,  8);

            ulong b14 = ROL(a20 ^ D0, 18);
            ulong b24 = ROL(a21 ^ D1,  2);
            ulong b09 = ROL(a22 ^ D2, 61);
            ulong b19 = ROL(a23 ^ D3, 56);
            ulong b04 = ROL(a24 ^ D4, 14);

            // Chi + Iota
            a00 = b00 ^ ((~b01) & b02) ^ KECCAK_RC[r];
            a01 = b01 ^ ((~b02) & b03);
            a02 = b02 ^ ((~b03) & b04);
            a03 = b03 ^ ((~b04) & b00);
            a04 = b04 ^ ((~b00) & b01);

            a05 = b05 ^ ((~b06) & b07);
            a06 = b06 ^ ((~b07) & b08);
            a07 = b07 ^ ((~b08) & b09);
            a08 = b08 ^ ((~b09) & b05);
            a09 = b09 ^ ((~b05) & b06);

            a10 = b10 ^ ((~b11) & b12);
            a11 = b11 ^ ((~b12) & b13);
            a12 = b12 ^ ((~b13) & b14);
            a13 = b13 ^ ((~b14) & b10);
            a14 = b14 ^ ((~b10) & b11);

            a15 = b15 ^ ((~b16) & b17);
            a16 = b16 ^ ((~b17) & b18);
            a17 = b17 ^ ((~b18) & b19);
            a18 = b18 ^ ((~b19) & b15);
            a19 = b19 ^ ((~b15) & b16);

            a20 = b20 ^ ((~b21) & b22);
            a21 = b21 ^ ((~b22) & b23);
            a22 = b22 ^ ((~b23) & b24);
            a23 = b23 ^ ((~b24) & b20);
            a24 = b24 ^ ((~b20) & b21);
        }

        s0 = a00;
        s1 = a01;
        if (!n2) {
            s2 = a02;
            s3 = a03;
        }
    }

    tips[base + 0u] = s0;
    tips[base + 1u] = s1;
    if (n_lanes > 2u) {
        tips[base + 2u] = s2;
        tips[base + 3u] = s3;
    }
}