I specialize the first Keccak round for the only hot input shapes (`n_bytes` = 16/32). Each chain step starts from a sparse, freshly padded state, so this avoids per-step zero-filling and removes much of the first-round theta work while preserving the full 24-round permutation. The remaining 23 rounds stay fully unrolled in the proven `uint2` 32-bit-limb form, which should reduce instruction count without changing bit-exact SHA3 semantics.

```metal
#include <metal_stdlib>
using namespace metal;

#define WOTS_ZERO2     uint2(0x00000000u, 0x00000000u)
#define WOTS_ONE2      uint2(0x00000001u, 0x00000000u)
#define WOTS_DOMAIN2   uint2(0x00000006u, 0x00000000u)
#define WOTS_DOMAIN_R1 uint2(0x0000000Cu, 0x00000000u)
#define WOTS_FINAL2    uint2(0x00000000u, 0x80000000u)

#define WOTS_D_R21     uint2(0x00C00000u, 0x00000000u)
#define WOTS_D_R25     uint2(0x0C000000u, 0x00000000u)
#define WOTS_D_R28     uint2(0x60000000u, 0x00000000u)
#define WOTS_D_R55     uint2(0x00000000u, 0x03000000u)
#define WOTS_D_R56     uint2(0x00000000u, 0x06000000u)
#define WOTS_D_R62     uint2(0x00000001u, 0x80000000u)
#define WOTS_F_R45     uint2(0x00000000u, 0x00001000u)

#define WOTS_ROL2L(v, n) uint2(((((v).x) << (n)) | (((v).y) >> (32u - (n)))), \
                               ((((v).y) << (n)) | (((v).x) >> (32u - (n)))))

#define WOTS_ROL2G(v, n) uint2(((((v).y) << ((n) - 32u)) | (((v).x) >> (64u - (n)))), \
                               ((((v).x) << ((n) - 32u)) | (((v).y) >> (64u - (n)))))

inline uint2 wots_split_u64(ulong x) {
    return uint2((uint)x, (uint)(x >> 32));
}

inline ulong wots_join_u64(uint2 v) {
    return (((ulong)v.y) << 32) | ((ulong)v.x);
}

#define WOTS_KECCAK_ROUND2(RCLO, RCHI) do {                            \
    uint2 c0 = a0 ^ a5 ^ a10 ^ a15 ^ a20;                              \
    uint2 c1 = a1 ^ a6 ^ a11 ^ a16 ^ a21;                              \
    uint2 c2 = a2 ^ a7 ^ a12 ^ a17 ^ a22;                              \
    uint2 c3 = a3 ^ a8 ^ a13 ^ a18 ^ a23;                              \
    uint2 c4 = a4 ^ a9 ^ a14 ^ a19 ^ a24;                              \
                                                                        \
    uint2 d0 = c4 ^ WOTS_ROL2L(c1, 1u);                                \
    uint2 d1 = c0 ^ WOTS_ROL2L(c2, 1u);                                \
    uint2 d2 = c1 ^ WOTS_ROL2L(c3, 1u);                                \
    uint2 d3 = c2 ^ WOTS_ROL2L(c4, 1u);                                \
    uint2 d4 = c3 ^ WOTS_ROL2L(c0, 1u);                                \
                                                                        \
    a0 ^= d0;  a5 ^= d0;  a10 ^= d0; a15 ^= d0; a20 ^= d0;             \
    a1 ^= d1;  a6 ^= d1;  a11 ^= d1; a16 ^= d1; a21 ^= d1;             \
    a2 ^= d2;  a7 ^= d2;  a12 ^= d2; a17 ^= d2; a22 ^= d2;             \
    a3 ^= d3;  a8 ^= d3;  a13 ^= d3; a18 ^= d3; a23 ^= d3;             \
    a4 ^= d4;  a9 ^= d4;  a14 ^= d4; a19 ^= d4; a24 ^= d4;             \
                                                                        \
    uint2 t = a1;                                                       \
    uint2 u = a10; a10 = WOTS_ROL2L(t,  1u); t = u;                    \
          u = a7;  a7  = WOTS_ROL2L(t,  3u); t = u;                    \
          u = a11; a11 = WOTS_ROL2L(t,  6u); t = u;                    \
          u = a17; a17 = WOTS_ROL2L(t, 10u); t = u;                    \
          u = a18; a18 = WOTS_ROL2L(t, 15u); t = u;                    \
          u = a3;  a3  = WOTS_ROL2L(t, 21u); t = u;                    \
          u = a5;  a5  = WOTS_ROL2L(t, 28u); t = u;                    \
          u = a16; a16 = WOTS_ROL2G(t, 36u); t = u;                    \
          u = a8;  a8  = WOTS_ROL2G(t, 45u); t = u;                    \
          u = a21; a21 = WOTS_ROL2G(t, 55u); t = u;                    \
          u = a24; a24 = WOTS_ROL2L(t,  2u); t = u;                    \
          u = a4;  a4  = WOTS_ROL2L(t, 14u); t = u;                    \
          u = a15; a15 = WOTS_ROL2L(t, 27u); t = u;                    \
          u = a23; a23 = WOTS_ROL2G(t, 41u); t = u;                    \
          u = a19; a19 = WOTS_ROL2G(t, 56u); t = u;                    \
          u = a13; a13 = WOTS_ROL2L(t,  8u); t = u;                    \
          u = a12; a12 = WOTS_ROL2L(t, 25u); t = u;                    \
          u = a2;  a2  = WOTS_ROL2G(t, 43u); t = u;                    \
          u = a20; a20 = WOTS_ROL2G(t, 62u); t = u;                    \
          u = a14; a14 = WOTS_ROL2L(t, 18u); t = u;                    \
          u = a22; a22 = WOTS_ROL2G(t, 39u); t = u;                    \
          u = a9;  a9  = WOTS_ROL2G(t, 61u); t = u;                    \
          u = a6;  a6  = WOTS_ROL2L(t, 20u); t = u;                    \
                    a1 = WOTS_ROL2G(t, 44u);                           \
                                                                        \
    c0 = a0; c1 = a1; c2 = a2; c3 = a3; c4 = a4;                       \
    a0 = c0 ^ ((~c1) & c2);                                            \
    a1 = c1 ^ ((~c2) & c3);                                            \
    a2 = c2 ^ ((~c3) & c4);                                            \
    a3 = c3 ^ ((~c4) & c0);                                            \
    a4 = c4 ^ ((~c0) & c1);                                            \
                                                                        \
    c0 = a5; c1 = a6; c2 = a7; c3 = a8; c4 = a9;                       \
    a5 = c0 ^ ((~c1) & c2);                                            \
    a6 = c1 ^ ((~c2) & c3);                                            \
    a7 = c2 ^ ((~c3) & c4);                                            \
    a8 = c3 ^ ((~c4) & c0);                                            \
    a9 = c4 ^ ((~c0) & c1);                                            \
                                                                        \
    c0 = a10; c1 = a11; c2 = a12; c3 = a13; c4 = a14;                  \
    a10 = c0 ^ ((~c1) & c2);                                           \
    a11 = c1 ^ ((~c2) & c3);                                           \
    a12 = c2 ^ ((~c3) & c4);                                           \
    a13 = c3 ^ ((~c4) & c0);                                           \
    a14 = c4 ^ ((~c0) & c1);                                           \
                                                                        \
    c0 = a15; c1 = a16; c2 = a17; c3 = a18; c4 = a19;                  \
    a15 = c0 ^ ((~c1) & c2);                                           \
    a16 = c1 ^ ((~c2) & c3);                                           \
    a17 = c2 ^ ((~c3) & c4);                                           \
    a18 = c3 ^ ((~c4) & c0);                                           \
    a19 = c4 ^ ((~c0) & c1);                                           \
                                                                        \
    c0 = a20; c1 = a21; c2 = a22; c3 = a23; c4 = a24;                  \
    a20 = c0 ^ ((~c1) & c2);                                           \
    a21 = c1 ^ ((~c2) & c3);                                           \
    a22 = c2 ^ ((~c3) & c4);                                           \
    a23 = c3 ^ ((~c4) & c0);                                           \
    a24 = c4 ^ ((~c0) & c1);                                           \
                                                                        \
    a0 ^= uint2((RCLO), (RCHI));                                       \
} while (false)

#define WOTS_KECCAK_FIRST_N2() do {                                    \
    uint2 d2 = a1 ^ WOTS_FINAL2;                                       \
    uint2 d0 = WOTS_ROL2L(a1, 1u) ^ WOTS_ONE2;                         \
    uint2 d1 = a0 ^ WOTS_DOMAIN_R1;                                    \
    uint2 d4 = WOTS_ROL2L(a0, 1u);                                     \
    uint2 b0, b1, b2, b3, b4;                                          \
                                                                        \
    b0 = a0 ^ d0;                                                      \
    b1 = WOTS_ROL2G(d1, 44u);                                         \
    b2 = WOTS_ROL2G(d2, 43u);                                         \
    b3 = WOTS_D_R21;                                                   \
    b4 = WOTS_ROL2L(d4, 14u);                                         \
    a0 = (b0 ^ ((~b1) & b2)) ^ WOTS_ONE2;                              \
    a1 =  b1 ^ ((~b2) & b3);                                          \
    a2 =  b2 ^ ((~b3) & b4);                                          \
    a3 =  b3 ^ ((~b4) & b0);                                          \
    a4 =  b4 ^ ((~b0) & b1);                                          \
                                                                        \
    b0 = WOTS_D_R28;                                                   \
    b1 = WOTS_ROL2L(d4, 20u);                                         \
    b2 = WOTS_ROL2L(d0, 3u);                                          \
    b3 = WOTS_ROL2G(d1, 45u) ^ WOTS_F_R45;                            \
    b4 = WOTS_ROL2G(d2, 61u);                                         \
    a5 = b0 ^ ((~b1) & b2);                                           \
    a6 = b1 ^ ((~b2) & b3);                                           \
    a7 = b2 ^ ((~b3) & b4);                                           \
    a8 = b3 ^ ((~b4) & b0);                                           \
    a9 = b4 ^ ((~b0) & b1);                                           \
                                                                        \
    b0 = (d0 ^ WOTS_ONE2) ^ WOTS_ROL2L(d1, 1u);                       \
    b1 = WOTS_ROL2L(d2, 6u);                                          \
    b2 = WOTS_D_R25;                                                   \
    b3 = WOTS_ROL2L(d4, 8u);                                          \
    b4 = WOTS_ROL2L(d0, 18u);                                         \
    a10 = b0 ^ ((~b1) & b2);                                          \
    a11 = b1 ^ ((~b2) & b3);                                          \
    a12 = b2 ^ ((~b3) & b4);                                          \
    a13 = b3 ^ ((~b4) & b0);                                          \
    a14 = b4 ^ ((~b0) & b1);                                          \
                                                                        \
    b0 = WOTS_ROL2L(d4, 27u);                                         \
    b1 = WOTS_ROL2G(d0, 36u);                                         \
    b2 = WOTS_ROL2L(d1, 10u);                                         \
    b3 = WOTS_ROL2L(d2, 15u);                                         \
    b4 = WOTS_D_R56;                                                   \
    a15 = b0 ^ ((~b1) & b2);                                          \
    a16 = b1 ^ ((~b2) & b3);                                          \
    a17 = b2 ^ ((~b3) & b4);                                          \
    a18 = b3 ^ ((~b4) & b0);                                          \
    a19 = b4 ^ ((~b0) & b1);                                          \
                                                                        \
    b0 = WOTS_ROL2G(d2, 62u) ^ WOTS_D_R62;                            \
    b1 = WOTS_D_R55;                                                   \
    b2 = WOTS_ROL2G(d4, 39u);                                         \
    b3 = WOTS_ROL2G(d0, 41u);                                         \
    b4 = WOTS_ROL2L(d1, 2u);                                          \
    a20 = b0 ^ ((~b1) & b2);                                          \
    a21 = b1 ^ ((~b2) & b3);                                          \
    a22 = b2 ^ ((~b3) & b4);                                          \
    a23 = b3 ^ ((~b4) & b0);                                          \
    a24 = b4 ^ ((~b0) & b1);                                          \
} while (false)

#define WOTS_KECCAK_FIRST_N4() do {                                    \
    uint2 d0 = WOTS_ROL2L(a1, 1u) ^ uint2(0x00000007u, 0x00000000u);   \
    uint2 d1 = a0 ^ WOTS_ROL2L(a2, 1u);                                \
    uint2 d2 = a1 ^ WOTS_FINAL2 ^ WOTS_ROL2L(a3, 1u);                  \
    uint2 d3 = a2 ^ WOTS_DOMAIN_R1;                                    \
    uint2 d4 = a3 ^ WOTS_ROL2L(a0, 1u);                                \
    uint2 p0 = a0 ^ d0;                                                \
    uint2 p1 = a1 ^ d1;                                                \
    uint2 p2 = a2 ^ d2;                                                \
    uint2 p3 = a3 ^ d3;                                                \
    uint2 p4 = WOTS_DOMAIN2 ^ d4;                                      \
    uint2 p16 = WOTS_FINAL2 ^ d1;                                      \
    uint2 b0, b1, b2, b3, b4;                                          \
                                                                        \
    b0 = p0;                                                           \
    b1 = WOTS_ROL2G(d1, 44u);                                         \
    b2 = WOTS_ROL2G(d2, 43u);                                         \
    b3 = WOTS_ROL2L(d3, 21u);                                         \
    b4 = WOTS_ROL2L(d4, 14u);                                         \
    a0 = (b0 ^ ((~b1) & b2)) ^ WOTS_ONE2;                              \
    a1 =  b1 ^ ((~b2) & b3);                                          \
    a2 =  b2 ^ ((~b3) & b4);                                          \
    a3 =  b3 ^ ((~b4) & b0);                                          \
    a4 =  b4 ^ ((~b0) & b1);                                          \
                                                                        \
    b0 = WOTS_ROL2L(p3, 28u);                                         \
    b1 = WOTS_ROL2L(d4, 20u);                                         \
    b2 = WOTS_ROL2L(d0, 3u);                                          \
    b3 = WOTS_ROL2G(p16, 45u);                                        \
    b4 = WOTS_ROL2G(d2, 61u);                                         \
    a5 = b0 ^ ((~b1) & b2);                                           \
    a6 = b1 ^ ((~b2) & b3);                                           \
    a7 = b2 ^ ((~b3) & b4);                                           \
    a8 = b3 ^ ((~b4) & b0);                                           \
    a9 = b4 ^ ((~b0) & b1);                                           \
                                                                        \
    b0 = WOTS_ROL2L(p1, 1u);                                          \
    b1 = WOTS_ROL2L(d2, 6u);                                          \
    b2 = WOTS_ROL2L(d3, 25u);                                         \
    b3 = WOTS_ROL2L(d4, 8u);                                          \
    b4 = WOTS_ROL2L(d0, 18u);                                         \
    a10 = b0 ^ ((~b1) & b2);                                          \
    a11 = b1 ^ ((~b2) & b3);                                          \
    a12 = b2 ^ ((~b3) & b4);                                          \
    a13 = b3 ^ ((~b4) & b0);                                          \
    a14 = b4 ^ ((~b0) & b1);                                          \
                                                                        \
    b0 = WOTS_ROL2L(p4, 27u);                                         \
    b1 = WOTS_ROL2G(d0, 36u);                                         \
    b2 = WOTS_ROL2L(d1, 10u);                                         \
    b3 = WOTS_ROL2L(d2, 15u);                                         \
    b4 = WOTS_ROL2G(d3, 56u);                                         \
    a15 = b0 ^ ((~b1) & b2);                                          \
    a16 = b1 ^ ((~b2) & b3);                                          \
    a17 = b2 ^ ((~b3) & b4);                                          \
    a18 = b3 ^ ((~b4) & b0);                                          \
    a19 = b4 ^ ((~b0) & b1);                                          \
                                                                        \
    b0 = WOTS_ROL2G(p2, 62u);                                         \
    b1 = WOTS_ROL2G(d3, 55u);                                         \
    b2 = WOTS_ROL2G(d4, 39u);                                         \
    b3 = WOTS_ROL2G(d0, 41u);                                         \
    b4 = WOTS_ROL2L(d1, 2u);                                          \
    a20 = b0 ^ ((~b1) & b2);                                          \
    a21 = b1 ^ ((~b2) & b3);                                          \
    a22 = b2 ^ ((~b3) & b4);                                          \
    a23 = b3 ^ ((~b4) & b0);                                          \
    a24 = b4 ^ ((~b0) & b1);                                          \
} while (false)

#define WOTS_KECCAK_TAIL_1_23() do {                                   \
    WOTS_KECCAK_ROUND2(0x00008082u, 0x00000000u);                      \
    WOTS_KECCAK_ROUND2(0x0000808Au, 0x80000000u);                      \
    WOTS_KECCAK_ROUND2(0x80008000u, 0x80000000u);                      \
    WOTS_KECCAK_ROUND2(0x0000808Bu, 0x00000000u);                      \
    WOTS_KECCAK_ROUND2(0x80000001u, 0x00000000u);                      \
    WOTS_KECCAK_ROUND2(0x80008081u, 0x80000000u);                      \
    WOTS_KECCAK_ROUND2(0x00008009u, 0x80000000u);                      \
    WOTS_KECCAK_ROUND2(0x0000008Au, 0x00000000u);                      \
    WOTS_KECCAK_ROUND2(0x00000088u, 0x00000000u);                      \
    WOTS_KECCAK_ROUND2(0x80008009u, 0x00000000u);                      \
    WOTS_KECCAK_ROUND2(0x8000000Au, 0x00000000u);                      \
    WOTS_KECCAK_ROUND2(0x8000808Bu, 0x00000000u);                      \
    WOTS_KECCAK_ROUND2(0x0000008Bu, 0x80000000u);                      \
    WOTS_KECCAK_ROUND2(0x00008089u, 0x80000000u);                      \
    WOTS_KECCAK_ROUND2(0x00008003u, 0x80000000u);                      \
    WOTS_KECCAK_ROUND2(0x00008002u, 0x80000000u);                      \
    WOTS_KECCAK_ROUND2(0x00000080u, 0x80000000u);                      \
    WOTS_KECCAK_ROUND2(0x0000800Au, 0x00000000u);                      \
    WOTS_KECCAK_ROUND2(0x8000000Au, 0x80000000u);                      \
    WOTS_KECCAK_ROUND2(0x80008081u, 0x80000000u);                      \
    WOTS_KECCAK_ROUND2(0x00008080u, 0x80000000u);                      \
    WOTS_KECCAK_ROUND2(0x80000001u, 0x00000000u);                      \
    WOTS_KECCAK_ROUND2(0x80008008u, 0x80000000u);                      \
} while (false)

#define WOTS_KECCAK_F1600_N2() do {                                    \
    WOTS_KECCAK_FIRST_N2();                                            \
    WOTS_KECCAK_TAIL_1_23();                                           \
} while (false)

#define WOTS_KECCAK_F1600_N4() do {                                    \
    WOTS_KECCAK_FIRST_N4();                                            \
    WOTS_KECCAK_TAIL_1_23();                                           \
} while (false)

constant uint2 WOTS_KECCAK_RC2_REF[24] = {
    uint2(0x00000001u, 0x00000000u), uint2(0x00008082u, 0x00000000u),
    uint2(0x0000808Au, 0x80000000u), uint2(0x80008000u, 0x80000000u),
    uint2(0x0000808Bu, 0x00000000u), uint2(0x80000001u, 0x00000000u),
    uint2(0x80008081u, 0x80000000u), uint2(0x00008009u, 0x80000000u),
    uint2(0x0000008Au, 0x00000000u), uint2(0x00000088u, 0x00000000u),
    uint2(0x80008009u, 0x00000000u), uint2(0x8000000Au, 0x00000000u),
    uint2(0x8000808Bu, 0x00000000u), uint2(0x0000008Bu, 0x80000000u),
    uint2(0x00008089u, 0x80000000u), uint2(0x00008003u, 0x80000000u),
    uint2(0x00008002u, 0x80000000u), uint2(0x00000080u, 0x80000000u),
    uint2(0x0000800Au, 0x00000000u), uint2(0x8000000Au, 0x80000000u),
    uint2(0x80008081u, 0x80000000u), uint2(0x00008080u, 0x80000000u),
    uint2(0x80000001u, 0x00000000u), uint2(0x80008008u, 0x80000000u)
};

constant uint WOTS_KECCAK_RHO_REF[25] = {
     0u,  1u, 62u, 28u, 27u,
    36u, 44u,  6u, 55u, 20u,
     3u, 10u, 43u, 25u, 39u,
    41u, 45u, 15u, 21u,  8u,
    18u,  2u, 61u, 56u, 14u
};

inline uint2 wots_rol2_var(uint2 v, uint k) {
    k &= 63u;
    if (k == 0u) {
        return v;
    }
    if (k < 32u) {
        return uint2((v.x << k) | (v.y >> (32u - k)),
                     (v.y << k) | (v.x >> (32u - k)));
    }
    if (k == 32u) {
        return uint2(v.y, v.x);
    }
    uint s = k - 32u;
    return uint2((v.y << s) | (v.x >> (32u - s)),
                 (v.x << s) | (v.y >> (32u - s)));
}

inline void wots_keccak_f1600_ref2(thread uint2 *A) {
    uint2 C[5];
    uint2 D[5];
    uint2 B[25];

    for (uint r = 0u; r < 24u; ++r) {
        for (uint x = 0u; x < 5u; ++x) {
            C[x] = A[x] ^ A[x + 5u] ^ A[x + 10u] ^ A[x + 15u] ^ A[x + 20u];
        }

        D[0] = C[4] ^ wots_rol2_var(C[1], 1u);
        D[1] = C[0] ^ wots_rol2_var(C[2], 1u);
        D[2] = C[1] ^ wots_rol2_var(C[3], 1u);
        D[3] = C[2] ^ wots_rol2_var(C[4], 1u);
        D[4] = C[3] ^ wots_rol2_var(C[0], 1u);

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
                B[dst] = wots_rol2_var(A[src], WOTS_KECCAK_RHO_REF[src]);
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

        A[0] ^= WOTS_KECCAK_RC2_REF[r];
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
    if (idx >= n_chains) {
        return;
    }

    uint n_lanes = n_bytes >> 3;
    uint steps = w;

    if (n_lanes == 2u) {
        uint base = idx << 1;

        uint2 a0 = wots_split_u64(seeds[base + 0u]);
        uint2 a1 = wots_split_u64(seeds[base + 1u]);
        uint2 a2, a3, a4, a5, a6, a7, a8, a9;
        uint2 a10, a11, a12, a13, a14, a15, a16, a17, a18, a19;
        uint2 a20, a21, a22, a23, a24;

        for (uint step = 0u; step < steps; ++step) {
            WOTS_KECCAK_F1600_N2();
        }

        tips[base + 0u] = wots_join_u64(a0);
        tips[base + 1u] = wots_join_u64(a1);
        return;
    }

    if (n_lanes == 4u) {
        uint base = idx << 2;

        uint2 a0 = wots_split_u64(seeds[base + 0u]);
        uint2 a1 = wots_split_u64(seeds[base + 1u]);
        uint2 a2 = wots_split_u64(seeds[base + 2u]);
        uint2 a3 = wots_split_u64(seeds[base + 3u]);
        uint2 a4, a5, a6, a7, a8, a9;
        uint2 a10, a11, a12, a13, a14, a15, a16, a17, a18, a19;
        uint2 a20, a21, a22, a23, a24;

        for (uint step = 0u; step < steps; ++step) {
            WOTS_KECCAK_F1600_N4();
        }

        tips[base + 0u] = wots_join_u64(a0);
        tips[base + 1u] = wots_join_u64(a1);
        tips[base + 2u] = wots_join_u64(a2);
        tips[base + 3u] = wots_join_u64(a3);
        return;
    }

    uint base = idx * n_lanes;
    thread uint2 A[25];

    for (uint i = 0u; i < 25u; ++i) {
        A[i] = WOTS_ZERO2;
    }
    for (uint k = 0u; k < n_lanes; ++k) {
        A[k] = wots_split_u64(seeds[base + k]);
    }

    for (uint step = 0u; step < steps; ++step) {
        for (uint i = n_lanes; i < 25u; ++i) {
            A[i] = WOTS_ZERO2;
        }
        A[n_lanes].x ^= 0x00000006u;
        A[16].y ^= 0x80000000u;
        wots_keccak_f1600_ref2(A);
    }

    for (uint k = 0u; k < n_lanes; ++k) {
        tips[base + k] = wots_join_u64(A[k]);
    }
}

#undef WOTS_KECCAK_F1600_N4
#undef WOTS_KECCAK_F1600_N2
#undef WOTS_KECCAK_TAIL_1_23
#undef WOTS_KECCAK_FIRST_N4
#undef WOTS_KECCAK_FIRST_N2
#undef WOTS_KECCAK_ROUND2
#undef WOTS_ROL2G
#undef WOTS_ROL2L
#undef WOTS_F_R45
#undef WOTS_D_R62
#undef WOTS_D_R56
#undef WOTS_D_R55
#undef WOTS_D_R28
#undef WOTS_D_R25
#undef WOTS_D_R21
#undef WOTS_FINAL2
#undef WOTS_DOMAIN_R1
#undef WOTS_DOMAIN2
#undef WOTS_ONE2
#undef WOTS_ZERO2
```