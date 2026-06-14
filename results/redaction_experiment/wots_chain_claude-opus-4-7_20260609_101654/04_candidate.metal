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

    ulong s0, s1, s2 = 0ul, s3 = 0ul;
    s0 = seeds[base + 0u];
    s1 = seeds[base + 1u];
    if (n_lanes > 2u) {
        s2 = seeds[base + 2u];
        s3 = seeds[base + 3u];
    }

    const ulong PAD_FINAL = 0x8000000000000000ul;
    const ulong PAD_DOM   = 0x06ul;
    bool n2 = (n_lanes == 2u);

    for (uint step = 0u; step < w; ++step) {
        // Round-1 specialization. State at entry:
        //   a00 = s0, a01 = s1
        //   if n_lanes==2: a02 = 0x06, a03 = a04 = 0
        //   if n_lanes==4: a02 = s2, a03 = s3, a04 = 0x06
        //   a05..a15 = 0, a16 = 0x80<<56, a17..a24 = 0
        ulong a00 = s0;
        ulong a01 = s1;
        ulong a02 = n2 ? PAD_DOM : s2;
        ulong a03 = n2 ? 0ul     : s3;
        ulong a04 = n2 ? 0ul     : PAD_DOM;
        ulong a16 = PAD_FINAL;

        // Theta on the sparse initial state.
        // Column XORs reduce to just the lanes that are nonzero.
        // C0 = a00; C1 = a01 ^ a16; C2 = a02; C3 = a03; C4 = a04.
        ulong C0 = a00;
        ulong C1 = a01 ^ a16;
        ulong C2 = a02;
        ulong C3 = a03;
        ulong C4 = a04;

        ulong D0 = C4 ^ ROL(C1, 1);
        ulong D1 = C0 ^ ROL(C2, 1);
        ulong D2 = C1 ^ ROL(C3, 1);
        ulong D3 = C2 ^ ROL(C4, 1);
        ulong D4 = C3 ^ ROL(C0, 1);

        // Apply D and Rho+Pi. Most lanes are 0 or PAD_FINAL.
        ulong b00 = a00 ^ D0;                 // rot 0
        ulong b10 = ROL(a01 ^ D1, 1);
        ulong b20 = ROL(a02 ^ D2, 62);
        ulong b05 = ROL(a03 ^ D3, 28);
        ulong b15 = ROL(a04 ^ D4, 27);

        ulong b16 = ROL(D0, 36);
        ulong b01 = ROL(D1, 44);
        ulong b11 = ROL(D2,  6);
        ulong b21 = ROL(D3, 55);
        ulong b06 = ROL(D4, 20);

        ulong b07 = ROL(D0,  3);
        ulong b17 = ROL(D1, 10);
        ulong b02 = ROL(D2, 43);
        ulong b12 = ROL(D3, 25);
        ulong b22 = ROL(D4, 39);

        ulong b23 = ROL(D0, 41);
        ulong b08 = ROL(a16 ^ D1, 45);
        ulong b18 = ROL(D2, 15);
        ulong b03 = ROL(D3, 21);
        ulong b13 = ROL(D4,  8);

        ulong b14 = ROL(D0, 18);
        ulong b24 = ROL(D1,  2);
        ulong b09 = ROL(D2, 61);
        ulong b19 = ROL(D3, 56);
        ulong b04 = ROL(D4, 14);

        // Chi + Iota for round 0
        a00 = b00 ^ ((~b01) & b02) ^ KECCAK_RC[0];
        a01 = b01 ^ ((~b02) & b03);
        a02 = b02 ^ ((~b03) & b04);
        a03 = b03 ^ ((~b04) & b00);
        a04 = b04 ^ ((~b00) & b01);

        ulong a05 = b05 ^ ((~b06) & b07);
        ulong a06 = b06 ^ ((~b07) & b08);
        ulong a07 = b07 ^ ((~b08) & b09);
        ulong a08 = b08 ^ ((~b09) & b05);
        ulong a09 = b09 ^ ((~b05) & b06);

        ulong a10 = b10 ^ ((~b11) & b12);
        ulong a11 = b11 ^ ((~b12) & b13);
        ulong a12 = b12 ^ ((~b13) & b14);
        ulong a13 = b13 ^ ((~b14) & b10);
        ulong a14 = b14 ^ ((~b10) & b11);

        ulong a15 = b15 ^ ((~b16) & b17);
        a16       = b16 ^ ((~b17) & b18);
        ulong a17 = b17 ^ ((~b18) & b19);
        ulong a18 = b18 ^ ((~b19) & b15);
        ulong a19 = b19 ^ ((~b15) & b16);

        ulong a20 = b20 ^ ((~b21) & b22);
        ulong a21 = b21 ^ ((~b22) & b23);
        ulong a22 = b22 ^ ((~b23) & b24);
        ulong a23 = b23 ^ ((~b24) & b20);
        ulong a24 = b24 ^ ((~b20) & b21);

        // Rounds 1..23, full Keccak-f.
        for (uint r = 1u; r < 24u; ++r) {
            ulong CC0 = a00 ^ a05 ^ a10 ^ a15 ^ a20;
            ulong CC1 = a01 ^ a06 ^ a11 ^ a16 ^ a21;
            ulong CC2 = a02 ^ a07 ^ a12 ^ a17 ^ a22;
            ulong CC3 = a03 ^ a08 ^ a13 ^ a18 ^ a23;
            ulong CC4 = a04 ^ a09 ^ a14 ^ a19 ^ a24;

            ulong DD0 = CC4 ^ ROL(CC1, 1);
            ulong DD1 = CC0 ^ ROL(CC2, 1);
            ulong DD2 = CC1 ^ ROL(CC3, 1);
            ulong DD3 = CC2 ^ ROL(CC4, 1);
            ulong DD4 = CC3 ^ ROL(CC0, 1);

            ulong c00 = a00 ^ DD0;
            ulong c10 = ROL(a01 ^ DD1,  1);
            ulong c20 = ROL(a02 ^ DD2, 62);
            ulong c05 = ROL(a03 ^ DD3, 28);
            ulong c15 = ROL(a04 ^ DD4, 27);

            ulong c16 = ROL(a05 ^ DD0, 36);
            ulong c01 = ROL(a06 ^ DD1, 44);
            ulong c11 = ROL(a07 ^ DD2,  6);
            ulong c21 = ROL(a08 ^ DD3, 55);
            ulong c06 = ROL(a09 ^ DD4, 20);

            ulong c07 = ROL(a10 ^ DD0,  3);
            ulong c17 = ROL(a11 ^ DD1, 10);
            ulong c02 = ROL(a12 ^ DD2, 43);
            ulong c12 = ROL(a13 ^ DD3, 25);
            ulong c22 = ROL(a14 ^ DD4, 39);

            ulong c23 = ROL(a15 ^ DD0, 41);
            ulong c08 = ROL(a16 ^ DD1, 45);
            ulong c18 = ROL(a17 ^ DD2, 15);
            ulong c03 = ROL(a18 ^ DD3, 21);
            ulong c13 = ROL(a19 ^ DD4,  8);

            ulong c14 = ROL(a20 ^ DD0, 18);
            ulong c24 = ROL(a21 ^ DD1,  2);
            ulong c09 = ROL(a22 ^ DD2, 61);
            ulong c19 = ROL(a23 ^ DD3, 56);
            ulong c04 = ROL(a24 ^ DD4, 14);

            a00 = c00 ^ ((~c01) & c02) ^ KECCAK_RC[r];
            a01 = c01 ^ ((~c02) & c03);
            a02 = c02 ^ ((~c03) & c04);
            a03 = c03 ^ ((~c04) & c00);
            a04 = c04 ^ ((~c00) & c01);

            a05 = c05 ^ ((~c06) & c07);
            a06 = c06 ^ ((~c07) & c08);
            a07 = c07 ^ ((~c08) & c09);
            a08 = c08 ^ ((~c09) & c05);
            a09 = c09 ^ ((~c05) & c06);

            a10 = c10 ^ ((~c11) & c12);
            a11 = c11 ^ ((~c12) & c13);
            a12 = c12 ^ ((~c13) & c14);
            a13 = c13 ^ ((~c14) & c10);
            a14 = c14 ^ ((~c10) & c11);

            a15 = c15 ^ ((~c16) & c17);
            a16 = c16 ^ ((~c17) & c18);
            a17 = c17 ^ ((~c18) & c19);
            a18 = c18 ^ ((~c19) & c15);
            a19 = c19 ^ ((~c15) & c16);

            a20 = c20 ^ ((~c21) & c22);
            a21 = c21 ^ ((~c22) & c23);
            a22 = c22 ^ ((~c23) & c24);
            a23 = c23 ^ ((~c24) & c20);
            a24 = c24 ^ ((~c20) & c21);
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