#include <metal_stdlib>
using namespace metal;

#define WOTS_ZERO2     uint2(0x00000000u, 0x00000000u)
#define WOTS_DOMAIN2   uint2(0x00000006u, 0x00000000u)
#define WOTS_FINAL2    uint2(0x00000000u, 0x80000000u)

#define WOTS_ROL2L(v, n) uint2(((((v).x) << (n)) | (((v).y) >> (32u - (n)))), \
                               ((((v).y) << (n)) | (((v).x) >> (32u - (n)))))

#define WOTS_ROL2G(v, n) uint2(((((v).y) << ((n) - 32u)) | (((v).x) >> (64u - (n)))), \
                               ((((v).x) << ((n) - 32u)) | (((v).y) >> (64u - (n)))))

inline uint2 wots_split_u64(ulong x) {
    return as_type<uint2>(x);
}

inline ulong wots_join_u64(uint2 v) {
    return as_type<ulong>(v);
}

#define WOTS_KECCAK_ROUND2(RCLO, RCHI) do {                            \
    uint2 c0 = ((a0 ^ a5)  ^ (a10 ^ a15)) ^ a20;                       \
    uint2 c1 = ((a1 ^ a6)  ^ (a11 ^ a16)) ^ a21;                       \
    uint2 c2 = ((a2 ^ a7)  ^ (a12 ^ a17)) ^ a22;                       \
    uint2 c3 = ((a3 ^ a8)  ^ (a13 ^ a18)) ^ a23;                       \
    uint2 c4 = ((a4 ^ a9)  ^ (a14 ^ a19)) ^ a24;                       \
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
    a1  = WOTS_ROL2G(a6,  44u);                                        \
    a6  = WOTS_ROL2L(a9,  20u);                                        \
    a9  = WOTS_ROL2G(a22, 61u);                                        \
    a22 = WOTS_ROL2G(a14, 39u);                                        \
    a14 = WOTS_ROL2L(a20, 18u);                                        \
    a20 = WOTS_ROL2G(a2,  62u);                                        \
    a2  = WOTS_ROL2G(a12, 43u);                                        \
    a12 = WOTS_ROL2L(a13, 25u);                                        \
    a13 = WOTS_ROL2L(a19,  8u);                                        \
    a19 = WOTS_ROL2G(a23, 56u);                                        \
    a23 = WOTS_ROL2G(a15, 41u);                                        \
    a15 = WOTS_ROL2L(a4,  27u);                                        \
    a4  = WOTS_ROL2L(a24, 14u);                                        \
    a24 = WOTS_ROL2L(a21,  2u);                                        \
    a21 = WOTS_ROL2G(a8,  55u);                                        \
    a8  = WOTS_ROL2G(a16, 45u);                                        \
    a16 = WOTS_ROL2G(a5,  36u);                                        \
    a5  = WOTS_ROL2L(a3,  28u);                                        \
    a3  = WOTS_ROL2L(a18, 21u);                                        \
    a18 = WOTS_ROL2L(a17, 15u);                                        \
    a17 = WOTS_ROL2L(a11, 10u);                                        \
    a11 = WOTS_ROL2L(a7,   6u);                                        \
    a7  = WOTS_ROL2L(a10,  3u);                                        \
    a10 = WOTS_ROL2L(t,    1u);                                        \
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

#define WOTS_KECCAK_F1600_2() do {                                     \
    WOTS_KECCAK_ROUND2(0x00000001u, 0x00000000u);                      \
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

    uint steps = w;

    if (n_bytes == 16u) {
        uint base = idx << 1;

        uint2 a0 = wots_split_u64(seeds[base + 0u]);
        uint2 a1 = wots_split_u64(seeds[base + 1u]);

        for (uint step = steps; step != 0u; --step) {
            uint2 a2  = WOTS_DOMAIN2;
            uint2 a3  = WOTS_ZERO2, a4  = WOTS_ZERO2, a5  = WOTS_ZERO2;
            uint2 a6  = WOTS_ZERO2, a7  = WOTS_ZERO2, a8  = WOTS_ZERO2;
            uint2 a9  = WOTS_ZERO2, a10 = WOTS_ZERO2, a11 = WOTS_ZERO2;
            uint2 a12 = WOTS_ZERO2, a13 = WOTS_ZERO2, a14 = WOTS_ZERO2;
            uint2 a15 = WOTS_ZERO2, a16 = WOTS_FINAL2,  a17 = WOTS_ZERO2;
            uint2 a18 = WOTS_ZERO2, a19 = WOTS_ZERO2, a20 = WOTS_ZERO2;
            uint2 a21 = WOTS_ZERO2, a22 = WOTS_ZERO2, a23 = WOTS_ZERO2;
            uint2 a24 = WOTS_ZERO2;

            WOTS_KECCAK_F1600_2();
        }

        tips[base + 0u] = wots_join_u64(a0);
        tips[base + 1u] = wots_join_u64(a1);
        return;
    }

    if (n_bytes == 32u) {
        uint base = idx << 2;

        uint2 a0 = wots_split_u64(seeds[base + 0u]);
        uint2 a1 = wots_split_u64(seeds[base + 1u]);
        uint2 a2 = wots_split_u64(seeds[base + 2u]);
        uint2 a3 = wots_split_u64(seeds[base + 3u]);

        for (uint step = steps; step != 0u; --step) {
            uint2 a4  = WOTS_DOMAIN2;
            uint2 a5  = WOTS_ZERO2, a6  = WOTS_ZERO2, a7  = WOTS_ZERO2;
            uint2 a8  = WOTS_ZERO2, a9  = WOTS_ZERO2, a10 = WOTS_ZERO2;
            uint2 a11 = WOTS_ZERO2, a12 = WOTS_ZERO2, a13 = WOTS_ZERO2;
            uint2 a14 = WOTS_ZERO2, a15 = WOTS_ZERO2, a16 = WOTS_FINAL2;
            uint2 a17 = WOTS_ZERO2, a18 = WOTS_ZERO2, a19 = WOTS_ZERO2;
            uint2 a20 = WOTS_ZERO2, a21 = WOTS_ZERO2, a22 = WOTS_ZERO2;
            uint2 a23 = WOTS_ZERO2, a24 = WOTS_ZERO2;

            WOTS_KECCAK_F1600_2();
        }

        tips[base + 0u] = wots_join_u64(a0);
        tips[base + 1u] = wots_join_u64(a1);
        tips[base + 2u] = wots_join_u64(a2);
        tips[base + 3u] = wots_join_u64(a3);
        return;
    }

    uint n_lanes = n_bytes >> 3;
    uint base = idx * n_lanes;
    thread uint2 A[25];

    for (uint i = 0u; i < 25u; ++i) {
        A[i] = WOTS_ZERO2;
    }
    for (uint k = 0u; k < n_lanes; ++k) {
        A[k] = wots_split_u64(seeds[base + k]);
    }

    for (uint step = steps; step != 0u; --step) {
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

#undef WOTS_KECCAK_F1600_2
#undef WOTS_KECCAK_ROUND2
#undef WOTS_ROL2G
#undef WOTS_ROL2L
#undef WOTS_FINAL2
#undef WOTS_DOMAIN2
#undef WOTS_ZERO2