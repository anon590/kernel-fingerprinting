#include <metal_stdlib>
using namespace metal;

#define WOTS_SHA3_FINAL_PAD 0x8000000000000000ul
#define WOTS_SHA3_DOMAIN    0x0000000000000006ul

#define ROL64(x, n) (((x) << (n)) | ((x) >> (64u - (n))))

#define KECCAK_ROUND(RCVAL) do {                                      \
    ulong c0 = a0 ^ a5 ^ a10 ^ a15 ^ a20;                             \
    ulong c1 = a1 ^ a6 ^ a11 ^ a16 ^ a21;                             \
    ulong c2 = a2 ^ a7 ^ a12 ^ a17 ^ a22;                             \
    ulong c3 = a3 ^ a8 ^ a13 ^ a18 ^ a23;                             \
    ulong c4 = a4 ^ a9 ^ a14 ^ a19 ^ a24;                             \
                                                                       \
    ulong d0 = c4 ^ ROL64(c1, 1u);                                    \
    ulong d1 = c0 ^ ROL64(c2, 1u);                                    \
    ulong d2 = c1 ^ ROL64(c3, 1u);                                    \
    ulong d3 = c2 ^ ROL64(c4, 1u);                                    \
    ulong d4 = c3 ^ ROL64(c0, 1u);                                    \
                                                                       \
    a0 ^= d0;  a5 ^= d0;  a10 ^= d0; a15 ^= d0; a20 ^= d0;            \
    a1 ^= d1;  a6 ^= d1;  a11 ^= d1; a16 ^= d1; a21 ^= d1;            \
    a2 ^= d2;  a7 ^= d2;  a12 ^= d2; a17 ^= d2; a22 ^= d2;            \
    a3 ^= d3;  a8 ^= d3;  a13 ^= d3; a18 ^= d3; a23 ^= d3;            \
    a4 ^= d4;  a9 ^= d4;  a14 ^= d4; a19 ^= d4; a24 ^= d4;            \
                                                                       \
    ulong t = a1;                                                      \
    ulong u = a10; a10 = ROL64(t,  1u); t = u;                        \
          u = a7;  a7  = ROL64(t,  3u); t = u;                        \
          u = a11; a11 = ROL64(t,  6u); t = u;                        \
          u = a17; a17 = ROL64(t, 10u); t = u;                        \
          u = a18; a18 = ROL64(t, 15u); t = u;                        \
          u = a3;  a3  = ROL64(t, 21u); t = u;                        \
          u = a5;  a5  = ROL64(t, 28u); t = u;                        \
          u = a16; a16 = ROL64(t, 36u); t = u;                        \
          u = a8;  a8  = ROL64(t, 45u); t = u;                        \
          u = a21; a21 = ROL64(t, 55u); t = u;                        \
          u = a24; a24 = ROL64(t,  2u); t = u;                        \
          u = a4;  a4  = ROL64(t, 14u); t = u;                        \
          u = a15; a15 = ROL64(t, 27u); t = u;                        \
          u = a23; a23 = ROL64(t, 41u); t = u;                        \
          u = a19; a19 = ROL64(t, 56u); t = u;                        \
          u = a13; a13 = ROL64(t,  8u); t = u;                        \
          u = a12; a12 = ROL64(t, 25u); t = u;                        \
          u = a2;  a2  = ROL64(t, 43u); t = u;                        \
          u = a20; a20 = ROL64(t, 62u); t = u;                        \
          u = a14; a14 = ROL64(t, 18u); t = u;                        \
          u = a22; a22 = ROL64(t, 39u); t = u;                        \
          u = a9;  a9  = ROL64(t, 61u); t = u;                        \
          u = a6;  a6  = ROL64(t, 20u); t = u;                        \
                    a1 = ROL64(t, 44u);                               \
                                                                       \
    c0 = a0; c1 = a1; c2 = a2; c3 = a3; c4 = a4;                      \
    a0 = c0 ^ ((~c1) & c2);                                           \
    a1 = c1 ^ ((~c2) & c3);                                           \
    a2 = c2 ^ ((~c3) & c4);                                           \
    a3 = c3 ^ ((~c4) & c0);                                           \
    a4 = c4 ^ ((~c0) & c1);                                           \
                                                                       \
    c0 = a5; c1 = a6; c2 = a7; c3 = a8; c4 = a9;                      \
    a5 = c0 ^ ((~c1) & c2);                                           \
    a6 = c1 ^ ((~c2) & c3);                                           \
    a7 = c2 ^ ((~c3) & c4);                                           \
    a8 = c3 ^ ((~c4) & c0);                                           \
    a9 = c4 ^ ((~c0) & c1);                                           \
                                                                       \
    c0 = a10; c1 = a11; c2 = a12; c3 = a13; c4 = a14;                 \
    a10 = c0 ^ ((~c1) & c2);                                          \
    a11 = c1 ^ ((~c2) & c3);                                          \
    a12 = c2 ^ ((~c3) & c4);                                          \
    a13 = c3 ^ ((~c4) & c0);                                          \
    a14 = c4 ^ ((~c0) & c1);                                          \
                                                                       \
    c0 = a15; c1 = a16; c2 = a17; c3 = a18; c4 = a19;                 \
    a15 = c0 ^ ((~c1) & c2);                                          \
    a16 = c1 ^ ((~c2) & c3);                                          \
    a17 = c2 ^ ((~c3) & c4);                                          \
    a18 = c3 ^ ((~c4) & c0);                                          \
    a19 = c4 ^ ((~c0) & c1);                                          \
                                                                       \
    c0 = a20; c1 = a21; c2 = a22; c3 = a23; c4 = a24;                 \
    a20 = c0 ^ ((~c1) & c2);                                          \
    a21 = c1 ^ ((~c2) & c3);                                          \
    a22 = c2 ^ ((~c3) & c4);                                          \
    a23 = c3 ^ ((~c4) & c0);                                          \
    a24 = c4 ^ ((~c0) & c1);                                          \
                                                                       \
    a0 ^= (RCVAL);                                                     \
} while (false)

#define KECCAK_F1600_SCALAR() do {                                    \
    KECCAK_ROUND(0x0000000000000001ul);                               \
    KECCAK_ROUND(0x0000000000008082ul);                               \
    KECCAK_ROUND(0x800000000000808Aul);                               \
    KECCAK_ROUND(0x8000000080008000ul);                               \
    KECCAK_ROUND(0x000000000000808Bul);                               \
    KECCAK_ROUND(0x0000000080000001ul);                               \
    KECCAK_ROUND(0x8000000080008081ul);                               \
    KECCAK_ROUND(0x8000000000008009ul);                               \
    KECCAK_ROUND(0x000000000000008Aul);                               \
    KECCAK_ROUND(0x0000000000000088ul);                               \
    KECCAK_ROUND(0x0000000080008009ul);                               \
    KECCAK_ROUND(0x000000008000000Aul);                               \
    KECCAK_ROUND(0x000000008000808Bul);                               \
    KECCAK_ROUND(0x800000000000008Bul);                               \
    KECCAK_ROUND(0x8000000000008089ul);                               \
    KECCAK_ROUND(0x8000000000008003ul);                               \
    KECCAK_ROUND(0x8000000000008002ul);                               \
    KECCAK_ROUND(0x8000000000000080ul);                               \
    KECCAK_ROUND(0x000000000000800Aul);                               \
    KECCAK_ROUND(0x800000008000000Aul);                               \
    KECCAK_ROUND(0x8000000080008081ul);                               \
    KECCAK_ROUND(0x8000000000008080ul);                               \
    KECCAK_ROUND(0x0000000080000001ul);                               \
    KECCAK_ROUND(0x8000000080008008ul);                               \
} while (false)

constant ulong WOTS_KECCAK_RC_REF[24] = {
    0x0000000000000001ul, 0x0000000000008082ul, 0x800000000000808Aul,
    0x8000000080008000ul, 0x000000000000808Bul, 0x0000000080000001ul,
    0x8000000080008081ul, 0x8000000000008009ul, 0x000000000000008Aul,
    0x0000000000000088ul, 0x0000000080008009ul, 0x000000008000000Aul,
    0x000000008000808Bul, 0x800000000000008Bul, 0x8000000000008089ul,
    0x8000000000008003ul, 0x8000000000008002ul, 0x8000000000000080ul,
    0x000000000000800Aul, 0x800000008000000Aul, 0x8000000080008081ul,
    0x8000000000008080ul, 0x0000000080000001ul, 0x8000000080008008ul,
};

constant uint WOTS_KECCAK_RHO_REF[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
};

inline ulong wots_rotl64_var(ulong x, uint k) {
    k &= 63u;
    if (k == 0u) return x;
    return (x << k) | (x >> (64u - k));
}

inline void wots_keccak_f1600_ref(thread ulong *A) {
    ulong C[5];
    ulong D[5];
    ulong B[25];

    for (uint r = 0u; r < 24u; ++r) {
        for (uint x = 0u; x < 5u; ++x) {
            C[x] = A[x] ^ A[x + 5u] ^ A[x + 10u] ^ A[x + 15u] ^ A[x + 20u];
        }

        D[0] = C[4] ^ wots_rotl64_var(C[1], 1u);
        D[1] = C[0] ^ wots_rotl64_var(C[2], 1u);
        D[2] = C[1] ^ wots_rotl64_var(C[3], 1u);
        D[3] = C[2] ^ wots_rotl64_var(C[4], 1u);
        D[4] = C[3] ^ wots_rotl64_var(C[0], 1u);

        for (uint y = 0u; y < 5u; ++y) {
            A[5u * y + 0u] ^= D[0];
            A[5u * y + 1u] ^= D[1];
            A[5u * y + 2u] ^= D[2];
            A[5u * y + 3u] ^= D[3];
            A[5u * y + 4u] ^= D[4];
        }

        for (uint y = 0u; y < 5u; ++y) {
            for (uint x = 0u; x < 5u; ++x) {
                uint src = x + 5u * y;
                uint dst = y + 5u * ((2u * x + 3u * y) % 5u);
                B[dst] = wots_rotl64_var(A[src], WOTS_KECCAK_RHO_REF[src]);
            }
        }

        for (uint y = 0u; y < 5u; ++y) {
            uint b = 5u * y;
            A[b + 0u] = B[b + 0u] ^ ((~B[b + 1u]) & B[b + 2u]);
            A[b + 1u] = B[b + 1u] ^ ((~B[b + 2u]) & B[b + 3u]);
            A[b + 2u] = B[b + 2u] ^ ((~B[b + 3u]) & B[b + 4u]);
            A[b + 3u] = B[b + 3u] ^ ((~B[b + 4u]) & B[b + 0u]);
            A[b + 4u] = B[b + 4u] ^ ((~B[b + 0u]) & B[b + 1u]);
        }

        A[0] ^= WOTS_KECCAK_RC_REF[r];
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

    uint n_lanes = n_bytes >> 3;

    if (n_lanes == 2u) {
        uint base = idx << 1;

        ulong a0 = seeds[base + 0u];
        ulong a1 = seeds[base + 1u];
        ulong a2, a3, a4, a5, a6, a7, a8, a9;
        ulong a10, a11, a12, a13, a14, a15, a16, a17, a18, a19;
        ulong a20, a21, a22, a23, a24;

        for (uint step = 0u; step < w; ++step) {
            a2  = WOTS_SHA3_DOMAIN;
            a3  = 0ul; a4  = 0ul; a5  = 0ul; a6  = 0ul; a7  = 0ul;
            a8  = 0ul; a9  = 0ul; a10 = 0ul; a11 = 0ul; a12 = 0ul;
            a13 = 0ul; a14 = 0ul; a15 = 0ul; a16 = WOTS_SHA3_FINAL_PAD;
            a17 = 0ul; a18 = 0ul; a19 = 0ul; a20 = 0ul; a21 = 0ul;
            a22 = 0ul; a23 = 0ul; a24 = 0ul;

            KECCAK_F1600_SCALAR();
        }

        tips[base + 0u] = a0;
        tips[base + 1u] = a1;
        return;
    }

    if (n_lanes == 4u) {
        uint base = idx << 2;

        ulong a0 = seeds[base + 0u];
        ulong a1 = seeds[base + 1u];
        ulong a2 = seeds[base + 2u];
        ulong a3 = seeds[base + 3u];
        ulong a4, a5, a6, a7, a8, a9;
        ulong a10, a11, a12, a13, a14, a15, a16, a17, a18, a19;
        ulong a20, a21, a22, a23, a24;

        for (uint step = 0u; step < w; ++step) {
            a4  = WOTS_SHA3_DOMAIN;
            a5  = 0ul; a6  = 0ul; a7  = 0ul; a8  = 0ul; a9  = 0ul;
            a10 = 0ul; a11 = 0ul; a12 = 0ul; a13 = 0ul; a14 = 0ul;
            a15 = 0ul; a16 = WOTS_SHA3_FINAL_PAD; a17 = 0ul; a18 = 0ul;
            a19 = 0ul; a20 = 0ul; a21 = 0ul; a22 = 0ul; a23 = 0ul;
            a24 = 0ul;

            KECCAK_F1600_SCALAR();
        }

        tips[base + 0u] = a0;
        tips[base + 1u] = a1;
        tips[base + 2u] = a2;
        tips[base + 3u] = a3;
        return;
    }

    uint base = idx * n_lanes;
    thread ulong A[25];

    for (uint i = 0u; i < 25u; ++i) A[i] = 0ul;
    for (uint k = 0u; k < n_lanes; ++k) A[k] = seeds[base + k];

    for (uint step = 0u; step < w; ++step) {
        for (uint i = n_lanes; i < 25u; ++i) A[i] = 0ul;
        A[n_lanes] ^= WOTS_SHA3_DOMAIN;
        A[16] ^= WOTS_SHA3_FINAL_PAD;
        wots_keccak_f1600_ref(A);
    }

    for (uint k = 0u; k < n_lanes; ++k) {
        tips[base + k] = A[k];
    }
}

#undef KECCAK_F1600_SCALAR
#undef KECCAK_ROUND
#undef ROL64