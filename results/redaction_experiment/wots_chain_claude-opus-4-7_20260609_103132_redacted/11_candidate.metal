#include <metal_stdlib>
using namespace metal;

constant ulong KRC[24] = {
    0x0000000000000001ul, 0x0000000000008082ul, 0x800000000000808Aul,
    0x8000000080008000ul, 0x000000000000808Bul, 0x0000000080000001ul,
    0x8000000080008081ul, 0x8000000000008009ul, 0x000000000000008Aul,
    0x0000000000000088ul, 0x0000000080008009ul, 0x000000008000000Aul,
    0x000000008000808Bul, 0x800000000000008Bul, 0x8000000000008089ul,
    0x8000000000008003ul, 0x8000000000008002ul, 0x8000000000000080ul,
    0x000000000000800Aul, 0x800000008000000Aul, 0x8000000080008081ul,
    0x8000000000008080ul, 0x0000000080000001ul, 0x8000000080008008ul,
};

// Bit-interleaved Keccak: split each 64-bit lane into even/odd bits.
// Rotations on the interleaved form become two 32-bit rotations.
// rotate by r on 64-bit = rotate halves by r/2 (if r even) or
// (rotate odd by (r+1)/2, rotate even by r/2, swap) (if r odd).
// We work entirely on uint pairs (e,o) packed as uint2.

// Forward: split ulong into (even_bits, odd_bits) where even_bits holds
// bit positions 0,2,4,... and odd_bits holds 1,3,5,...
inline uint2 interleave(ulong x) {
    uint lo = (uint)x;
    uint hi = (uint)(x >> 32);
    // Gather even bits of lo into low 16 of e, even bits of hi into high 16 of e.
    // Use bit-twiddle: standard interleave.
    auto pack = [](uint v) {
        v &= 0x55555555u;
        v = (v | (v >> 1)) & 0x33333333u;
        v = (v | (v >> 2)) & 0x0F0F0F0Fu;
        v = (v | (v >> 4)) & 0x00FF00FFu;
        v = (v | (v >> 8)) & 0x0000FFFFu;
        return v;
    };
    uint e = pack(lo) | (pack(hi) << 16);
    uint o = pack(lo >> 1) | (pack(hi >> 1) << 16);
    return uint2(e, o);
}

inline ulong deinterleave(uint2 p) {
    auto spread = [](uint v) -> uint {
        v &= 0x0000FFFFu;
        v = (v | (v << 8)) & 0x00FF00FFu;
        v = (v | (v << 4)) & 0x0F0F0F0Fu;
        v = (v | (v << 2)) & 0x33333333u;
        v = (v | (v << 1)) & 0x55555555u;
        return v;
    };
    uint e = p.x, o = p.y;
    uint lo = spread(e & 0xFFFFu) | (spread(o & 0xFFFFu) << 1);
    uint hi = spread(e >> 16)     | (spread(o >> 16) << 1);
    return ((ulong)hi << 32) | (ulong)lo;
}

// Rotate interleaved (e,o) representing 64-bit value left by r.
inline uint2 rol_i(uint2 p, uint r) {
    uint r2 = r >> 1;
    if ((r & 1u) == 0u) {
        return uint2(rotate(p.x, r2), rotate(p.y, r2));
    } else {
        // odd: new_e = ROL(o, r2+? )... standard formula:
        // value bit i goes to (i+r) mod 64.
        // If r odd: even bits of input go to odd positions of output, with extra rotate.
        // new_o = ROL(p.x, (r+1)/2)
        // new_e = ROL(p.y, (r-1)/2 + 1)? Let's derive:
        // Input bit 2k (even, in p.x at pos k) -> output bit (2k+r) mod 64.
        //   r odd -> 2k+r is odd -> in new_o at pos ((2k+r) mod 64)/2 = (k + (r-1)/2 + (carry?)) ...
        //   Actually (2k+r) mod 64, with r=2m+1 -> = 2k+2m+1 mod 64.
        //   This is odd, position in odd plane = ((2k+2m+1)-1)/2 mod 32 = (k+m) mod 32.
        //   So new_o[k+m mod 32] = p.x[k] => new_o = ROL(p.x, m) = ROL(p.x, (r-1)/2).
        // Input bit 2k+1 (odd, in p.y at pos k) -> output bit (2k+1+r) mod 64.
        //   = 2k+2m+2 mod 64, even, position = (k+m+1) mod 32.
        //   So new_e[k+m+1] = p.y[k] => new_e = ROL(p.y, m+1) = ROL(p.y, (r+1)/2).
        uint m = (r - 1u) >> 1;
        return uint2(rotate(p.y, m + 1u), rotate(p.x, m));
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

    const uint n_lanes = n_bytes >> 3;
    const uint base = idx * n_lanes;
    const uint W = w;

    // 25 lanes in interleaved form.
    uint2 S[25];

    // Precompute interleaved seed lanes.
    uint2 M[16];
    for (uint k = 0u; k < 16u; ++k) M[k] = uint2(0u, 0u);
    for (uint k = 0u; k < n_lanes; ++k) {
        M[k] = interleave(seeds[base + k]);
    }

    // Interleaved padding constants.
    // 0x06 at lane n_lanes: bits 1 and 2 set.
    //   bit 1 (odd, pos 0 in p.y) -> p.y |= 1
    //   bit 2 (even, pos 1 in p.x) -> p.x |= 2
    uint2 dom_pad = uint2(0x2u, 0x1u);
    // 0x80 << 56 at lane 16 = bit 63 set.
    //   bit 63 (odd, pos 31 in p.y) -> p.y |= 0x80000000
    uint2 fin_pad = uint2(0u, 0x80000000u);

    for (uint step = 0u; step < W; ++step) {
        // Initialize state: lanes 0..n_lanes-1 = M, lane n_lanes ^= dom_pad,
        // lane 16 ^= fin_pad, rest zero.
        for (uint k = 0u; k < 25u; ++k) S[k] = uint2(0u, 0u);
        for (uint k = 0u; k < n_lanes; ++k) S[k] = M[k];

        // Domain pad lane
        uint dl = n_lanes;
        // We need conditional XOR into S[dl]. Use a switch.
        switch (dl) {
            case 0:  S[0]  = S[0]  ^ dom_pad; break;
            case 1:  S[1]  = S[1]  ^ dom_pad; break;
            case 2:  S[2]  = S[2]  ^ dom_pad; break;
            case 3:  S[3]  = S[3]  ^ dom_pad; break;
            case 4:  S[4]  = S[4]  ^ dom_pad; break;
            case 5:  S[5]  = S[5]  ^ dom_pad; break;
            case 6:  S[6]  = S[6]  ^ dom_pad; break;
            case 7:  S[7]  = S[7]  ^ dom_pad; break;
            case 8:  S[8]  = S[8]  ^ dom_pad; break;
            case 9:  S[9]  = S[9]  ^ dom_pad; break;
            case 10: S[10] = S[10] ^ dom_pad; break;
            case 11: S[11] = S[11] ^ dom_pad; break;
            case 12: S[12] = S[12] ^ dom_pad; break;
            case 13: S[13] = S[13] ^ dom_pad; break;
            case 14: S[14] = S[14] ^ dom_pad; break;
            case 15: S[15] = S[15] ^ dom_pad; break;
            case 16: S[16] = S[16] ^ dom_pad; break;
            default: break;
        }
        S[16] = S[16] ^ fin_pad;

        // Keccak-f[1600] in interleaved form.
        for (uint r = 0u; r < 24u; ++r) {
            // theta
            uint2 C0 = S[0]^S[5]^S[10]^S[15]^S[20];
            uint2 C1 = S[1]^S[6]^S[11]^S[16]^S[21];
            uint2 C2 = S[2]^S[7]^S[12]^S[17]^S[22];
            uint2 C3 = S[3]^S[8]^S[13]^S[18]^S[23];
            uint2 C4 = S[4]^S[9]^S[14]^S[19]^S[24];

            uint2 D0 = C4 ^ rol_i(C1, 1);
            uint2 D1 = C0 ^ rol_i(C2, 1);
            uint2 D2 = C1 ^ rol_i(C3, 1);
            uint2 D3 = C2 ^ rol_i(C4, 1);
            uint2 D4 = C3 ^ rol_i(C0, 1);

            S[0]^=D0; S[5]^=D0; S[10]^=D0; S[15]^=D0; S[20]^=D0;
            S[1]^=D1; S[6]^=D1; S[11]^=D1; S[16]^=D1; S[21]^=D1;
            S[2]^=D2; S[7]^=D2; S[12]^=D2; S[17]^=D2; S[22]^=D2;
            S[3]^=D3; S[8]^=D3; S[13]^=D3; S[18]^=D3; S[23]^=D3;
            S[4]^=D4; S[9]^=D4; S[14]^=D4; S[19]^=D4; S[24]^=D4;

            // rho + pi
            uint2 B00 = S[0];
            uint2 B10 = rol_i(S[1],  1);
            uint2 B20 = rol_i(S[2], 62);
            uint2 B05 = rol_i(S[3], 28);
            uint2 B15 = rol_i(S[4], 27);
            uint2 B16 = rol_i(S[5], 36);
            uint2 B01 = rol_i(S[6], 44);
            uint2 B11 = rol_i(S[7],  6);
            uint2 B21 = rol_i(S[8], 55);
            uint2 B06 = rol_i(S[9], 20);
            uint2 B07 = rol_i(S[10],  3);
            uint2 B17 = rol_i(S[11], 10);
            uint2 B02 = rol_i(S[12], 43);
            uint2 B12 = rol_i(S[13], 25);
            uint2 B22 = rol_i(S[14], 39);
            uint2 B23 = rol_i(S[15], 41);
            uint2 B08 = rol_i(S[16], 45);
            uint2 B18 = rol_i(S[17], 15);
            uint2 B03 = rol_i(S[18], 21);
            uint2 B13 = rol_i(S[19],  8);
            uint2 B14 = rol_i(S[20], 18);
            uint2 B24 = rol_i(S[21],  2);
            uint2 B09 = rol_i(S[22], 61);
            uint2 B19 = rol_i(S[23], 56);
            uint2 B04 = rol_i(S[24], 14);

            // chi
            S[0]  = B00 ^ ((~B01) & B02);
            S[1]  = B01 ^ ((~B02) & B03);
            S[2]  = B02 ^ ((~B03) & B04);
            S[3]  = B03 ^ ((~B04) & B00);
            S[4]  = B04 ^ ((~B00) & B01);
            S[5]  = B05 ^ ((~B06) & B07);
            S[6]  = B06 ^ ((~B07) & B08);
            S[7]  = B07 ^ ((~B08) & B09);
            S[8]  = B08 ^ ((~B09) & B05);
            S[9]  = B09 ^ ((~B05) & B06);
            S[10] = B10 ^ ((~B11) & B12);
            S[11] = B11 ^ ((~B12) & B13);
            S[12] = B12 ^ ((~B13) & B14);
            S[13] = B13 ^ ((~B14) & B10);
            S[14] = B14 ^ ((~B10) & B11);
            S[15] = B15 ^ ((~B16) & B17);
            S[16] = B16 ^ ((~B17) & B18);
            S[17] = B17 ^ ((~B18) & B19);
            S[18] = B18 ^ ((~B19) & B15);
            S[19] = B19 ^ ((~B15) & B16);
            S[20] = B20 ^ ((~B21) & B22);
            S[21] = B21 ^ ((~B22) & B23);
            S[22] = B22 ^ ((~B23) & B24);
            S[23] = B23 ^ ((~B24) & B20);
            S[24] = B24 ^ ((~B20) & B21);

            // iota: XOR interleaved RC.
            ulong rc = KRC[r];
            uint2 irc = interleave(rc);
            S[0] = S[0] ^ irc;
        }

        // Truncate: M[0..n_lanes-1] = S[0..n_lanes-1] (still interleaved).
        for (uint k = 0u; k < n_lanes; ++k) M[k] = S[k];
    }

    // Deinterleave output.
    for (uint k = 0u; k < n_lanes; ++k) {
        tips[base + k] = deinterleave(M[k]);
    }
}