#include <metal_stdlib>
using namespace metal;

inline ulong ROL(ulong x, uint k) {
    return rotate(x, (ulong)k);
}

#define THETA_RHO_PI_CHI_IOTA(RC)                                          \
    {                                                                      \
        ulong C0 = a00 ^ a05 ^ a10 ^ a15 ^ a20;                            \
        ulong C1 = a01 ^ a06 ^ a11 ^ a16 ^ a21;                            \
        ulong C2 = a02 ^ a07 ^ a12 ^ a17 ^ a22;                            \
        ulong C3 = a03 ^ a08 ^ a13 ^ a18 ^ a23;                            \
        ulong C4 = a04 ^ a09 ^ a14 ^ a19 ^ a24;                            \
        ulong D0 = C4 ^ ROL(C1, 1);                                        \
        ulong D1 = C0 ^ ROL(C2, 1);                                        \
        ulong D2 = C1 ^ ROL(C3, 1);                                        \
        ulong D3 = C2 ^ ROL(C4, 1);                                        \
        ulong D4 = C3 ^ ROL(C0, 1);                                        \
        ulong t00 = a00 ^ D0;                                              \
        ulong t01 = a01 ^ D1;                                              \
        ulong t02 = a02 ^ D2;                                              \
        ulong t03 = a03 ^ D3;                                              \
        ulong t04 = a04 ^ D4;                                              \
        ulong t05 = a05 ^ D0;                                              \
        ulong t06 = a06 ^ D1;                                              \
        ulong t07 = a07 ^ D2;                                              \
        ulong t08 = a08 ^ D3;                                              \
        ulong t09 = a09 ^ D4;                                              \
        ulong t10 = a10 ^ D0;                                              \
        ulong t11 = a11 ^ D1;                                              \
        ulong t12 = a12 ^ D2;                                              \
        ulong t13 = a13 ^ D3;                                              \
        ulong t14 = a14 ^ D4;                                              \
        ulong t15 = a15 ^ D0;                                              \
        ulong t16 = a16 ^ D1;                                              \
        ulong t17 = a17 ^ D2;                                              \
        ulong t18 = a18 ^ D3;                                              \
        ulong t19 = a19 ^ D4;                                              \
        ulong t20 = a20 ^ D0;                                              \
        ulong t21 = a21 ^ D1;                                              \
        ulong t22 = a22 ^ D2;                                              \
        ulong t23 = a23 ^ D3;                                              \
        ulong t24 = a24 ^ D4;                                              \
        ulong b00 = t00;                                                   \
        ulong b10 = ROL(t01,  1);                                          \
        ulong b20 = ROL(t02, 62);                                          \
        ulong b05 = ROL(t03, 28);                                          \
        ulong b15 = ROL(t04, 27);                                          \
        ulong b16 = ROL(t05, 36);                                          \
        ulong b01 = ROL(t06, 44);                                          \
        ulong b11 = ROL(t07,  6);                                          \
        ulong b21 = ROL(t08, 55);                                          \
        ulong b06 = ROL(t09, 20);                                          \
        ulong b07 = ROL(t10,  3);                                          \
        ulong b17 = ROL(t11, 10);                                          \
        ulong b02 = ROL(t12, 43);                                          \
        ulong b12 = ROL(t13, 25);                                          \
        ulong b22 = ROL(t14, 39);                                          \
        ulong b23 = ROL(t15, 41);                                          \
        ulong b08 = ROL(t16, 45);                                          \
        ulong b18 = ROL(t17, 15);                                          \
        ulong b03 = ROL(t18, 21);                                          \
        ulong b13 = ROL(t19,  8);                                          \
        ulong b14 = ROL(t20, 18);                                          \
        ulong b24 = ROL(t21,  2);                                          \
        ulong b09 = ROL(t22, 61);                                          \
        ulong b19 = ROL(t23, 56);                                          \
        ulong b04 = ROL(t24, 14);                                          \
        a00 = b00 ^ ((~b01) & b02) ^ (RC);                                 \
        a01 = b01 ^ ((~b02) & b03);                                        \
        a02 = b02 ^ ((~b03) & b04);                                        \
        a03 = b03 ^ ((~b04) & b00);                                        \
        a04 = b04 ^ ((~b00) & b01);                                        \
        a05 = b05 ^ ((~b06) & b07);                                        \
        a06 = b06 ^ ((~b07) & b08);                                        \
        a07 = b07 ^ ((~b08) & b09);                                        \
        a08 = b08 ^ ((~b09) & b05);                                        \
        a09 = b09 ^ ((~b05) & b06);                                        \
        a10 = b10 ^ ((~b11) & b12);                                        \
        a11 = b11 ^ ((~b12) & b13);                                        \
        a12 = b12 ^ ((~b13) & b14);                                        \
        a13 = b13 ^ ((~b14) & b10);                                        \
        a14 = b14 ^ ((~b10) & b11);                                        \
        a15 = b15 ^ ((~b16) & b17);                                        \
        a16 = b16 ^ ((~b17) & b18);                                        \
        a17 = b17 ^ ((~b18) & b19);                                        \
        a18 = b18 ^ ((~b19) & b15);                                        \
        a19 = b19 ^ ((~b15) & b16);                                        \
        a20 = b20 ^ ((~b21) & b22);                                        \
        a21 = b21 ^ ((~b22) & b23);                                        \
        a22 = b22 ^ ((~b23) & b24);                                        \
        a23 = b23 ^ ((~b24) & b20);                                        \
        a24 = b24 ^ ((~b20) & b21);                                        \
    }

#define KECCAK_F1600()                                                     \
    THETA_RHO_PI_CHI_IOTA(0x0000000000000001ul)                            \
    THETA_RHO_PI_CHI_IOTA(0x0000000000008082ul)                            \
    THETA_RHO_PI_CHI_IOTA(0x800000000000808Aul)                            \
    THETA_RHO_PI_CHI_IOTA(0x8000000080008000ul)                            \
    THETA_RHO_PI_CHI_IOTA(0x000000000000808Bul)                            \
    THETA_RHO_PI_CHI_IOTA(0x0000000080000001ul)                            \
    THETA_RHO_PI_CHI_IOTA(0x8000000080008081ul)                            \
    THETA_RHO_PI_CHI_IOTA(0x8000000000008009ul)                            \
    THETA_RHO_PI_CHI_IOTA(0x000000000000008Aul)                            \
    THETA_RHO_PI_CHI_IOTA(0x0000000000000088ul)                            \
    THETA_RHO_PI_CHI_IOTA(0x0000000080008009ul)                            \
    THETA_RHO_PI_CHI_IOTA(0x000000008000000Aul)                            \
    THETA_RHO_PI_CHI_IOTA(0x000000008000808Bul)                            \
    THETA_RHO_PI_CHI_IOTA(0x800000000000008Bul)                            \
    THETA_RHO_PI_CHI_IOTA(0x8000000000008089ul)                            \
    THETA_RHO_PI_CHI_IOTA(0x8000000000008003ul)                            \
    THETA_RHO_PI_CHI_IOTA(0x8000000000008002ul)                            \
    THETA_RHO_PI_CHI_IOTA(0x8000000000000080ul)                            \
    THETA_RHO_PI_CHI_IOTA(0x000000000000800Aul)                            \
    THETA_RHO_PI_CHI_IOTA(0x800000008000000Aul)                            \
    THETA_RHO_PI_CHI_IOTA(0x8000000080008081ul)                            \
    THETA_RHO_PI_CHI_IOTA(0x8000000000008080ul)                            \
    THETA_RHO_PI_CHI_IOTA(0x0000000080000001ul)                            \
    THETA_RHO_PI_CHI_IOTA(0x8000000080008008ul)

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

    // Load seed lanes (n_lanes is 2 or 4).
    ulong s0 = seeds[base + 0u];
    ulong s1 = seeds[base + 1u];
    ulong s2 = 0ul, s3 = 0ul;
    if (n_lanes > 2u) {
        s2 = seeds[base + 2u];
        s3 = seeds[base + 3u];
    }

    // Specialize the inner loop on n_lanes to make the padding pattern compile-time
    // within each branch. Lanes 5..24 (except 16) are always zero on absorb; lane 16
    // is always PAD80. Domain pad lane is n_lanes (a02 for n=2, a04 for n=4).
    if (n_lanes == 2u) {
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