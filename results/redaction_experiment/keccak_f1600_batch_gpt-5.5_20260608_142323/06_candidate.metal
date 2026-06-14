#include <metal_stdlib>
using namespace metal;

#define U64(lo, hi) uint2((uint)(lo), (uint)(hi))
#define Z64 U64(0u, 0u)

#define LOAD64(p, i) as_type<uint2>((p)[(i)])
#define STORE64(p, i, v) do { (p)[(i)] = as_type<ulong>((v)); } while (0)

#define ROL64(x, n) ( ((n) < 32u) ? \
    uint2( (((x).x << ((n) & 31u)) | ((x).y >> ((32u - ((n) & 31u)) & 31u))), \
           (((x).y << ((n) & 31u)) | ((x).x >> ((32u - ((n) & 31u)) & 31u))) ) : \
    uint2( (((x).y << ((n) & 31u)) | ((x).x >> ((32u - ((n) & 31u)) & 31u))), \
           (((x).x << ((n) & 31u)) | ((x).y >> ((32u - ((n) & 31u)) & 31u))) ) )

#define RHOPI_STEP(tvar, dst, rot) do {        \
    uint2 _tmp = (dst);                        \
    (dst) = ROL64((tvar), (rot));              \
    (tvar) = _tmp;                             \
} while (0)

#define KECCAK_RHOPI_CHAIN() do {              \
    uint2 t0 = a10;                            \
    uint2 t1 = a40;                            \
    RHOPI_STEP(t0, a02,  1u);                  \
    RHOPI_STEP(t1, a03, 27u);                  \
    RHOPI_STEP(t0, a21,  3u);                  \
    RHOPI_STEP(t1, a34, 41u);                  \
    RHOPI_STEP(t0, a12,  6u);                  \
    RHOPI_STEP(t1, a43, 56u);                  \
    RHOPI_STEP(t0, a23, 10u);                  \
    RHOPI_STEP(t1, a32,  8u);                  \
    RHOPI_STEP(t0, a33, 15u);                  \
    RHOPI_STEP(t1, a22, 25u);                  \
    RHOPI_STEP(t0, a30, 21u);                  \
    RHOPI_STEP(t1, a20, 43u);                  \
    RHOPI_STEP(t0, a01, 28u);                  \
    RHOPI_STEP(t1, a04, 62u);                  \
    RHOPI_STEP(t0, a13, 36u);                  \
    RHOPI_STEP(t1, a42, 18u);                  \
    RHOPI_STEP(t0, a31, 45u);                  \
    RHOPI_STEP(t1, a24, 39u);                  \
    RHOPI_STEP(t0, a14, 55u);                  \
    RHOPI_STEP(t1, a41, 61u);                  \
    RHOPI_STEP(t0, a44,  2u);                  \
    RHOPI_STEP(t1, a11, 20u);                  \
    RHOPI_STEP(t0, a40, 14u);                  \
    RHOPI_STEP(t1, a10, 44u);                  \
} while (0)

#define KECCAK_RHOPI_CHI_IOTA(rc_) do {                                             \
    KECCAK_RHOPI_CHAIN();                                                           \
                                                                                    \
    uint2 b0, b1, b2, b3, b4;                                                       \
                                                                                    \
    b0 = a00; b1 = a10; b2 = a20; b3 = a30; b4 = a40;                               \
    a00 = b0 ^ (b2 & ~b1);                                                          \
    a10 = b1 ^ (b3 & ~b2);                                                          \
    a20 = b2 ^ (b4 & ~b3);                                                          \
    a30 = b3 ^ (b0 & ~b4);                                                          \
    a40 = b4 ^ (b1 & ~b0);                                                          \
                                                                                    \
    b0 = a01; b1 = a11; b2 = a21; b3 = a31; b4 = a41;                               \
    a01 = b0 ^ (b2 & ~b1);                                                          \
    a11 = b1 ^ (b3 & ~b2);                                                          \
    a21 = b2 ^ (b4 & ~b3);                                                          \
    a31 = b3 ^ (b0 & ~b4);                                                          \
    a41 = b4 ^ (b1 & ~b0);                                                          \
                                                                                    \
    b0 = a02; b1 = a12; b2 = a22; b3 = a32; b4 = a42;                               \
    a02 = b0 ^ (b2 & ~b1);                                                          \
    a12 = b1 ^ (b3 & ~b2);                                                          \
    a22 = b2 ^ (b4 & ~b3);                                                          \
    a32 = b3 ^ (b0 & ~b4);                                                          \
    a42 = b4 ^ (b1 & ~b0);                                                          \
                                                                                    \
    b0 = a03; b1 = a13; b2 = a23; b3 = a33; b4 = a43;                               \
    a03 = b0 ^ (b2 & ~b1);                                                          \
    a13 = b1 ^ (b3 & ~b2);                                                          \
    a23 = b2 ^ (b4 & ~b3);                                                          \
    a33 = b3 ^ (b0 & ~b4);                                                          \
    a43 = b4 ^ (b1 & ~b0);                                                          \
                                                                                    \
    b0 = a04; b1 = a14; b2 = a24; b3 = a34; b4 = a44;                               \
    a04 = b0 ^ (b2 & ~b1);                                                          \
    a14 = b1 ^ (b3 & ~b2);                                                          \
    a24 = b2 ^ (b4 & ~b3);                                                          \
    a34 = b3 ^ (b0 & ~b4);                                                          \
    a44 = b4 ^ (b1 & ~b0);                                                          \
                                                                                    \
    a00 ^= (rc_);                                                                   \
} while (0)

#define KECCAK_ROUND(rc_) do {                                                     \
    uint2 c0 = a00 ^ a01 ^ a02 ^ a03 ^ a04;                                        \
    uint2 c1 = a10 ^ a11 ^ a12 ^ a13 ^ a14;                                        \
    uint2 c2 = a20 ^ a21 ^ a22 ^ a23 ^ a24;                                        \
    uint2 c3 = a30 ^ a31 ^ a32 ^ a33 ^ a34;                                        \
    uint2 c4 = a40 ^ a41 ^ a42 ^ a43 ^ a44;                                        \
                                                                                   \
    uint2 d = c4 ^ ROL64(c1, 1u);                                                  \
    a00 ^= d; a01 ^= d; a02 ^= d; a03 ^= d; a04 ^= d;                              \
    d = c0 ^ ROL64(c2, 1u);                                                        \
    a10 ^= d; a11 ^= d; a12 ^= d; a13 ^= d; a14 ^= d;                              \
    d = c1 ^ ROL64(c3, 1u);                                                        \
    a20 ^= d; a21 ^= d; a22 ^= d; a23 ^= d; a24 ^= d;                              \
    d = c2 ^ ROL64(c4, 1u);                                                        \
    a30 ^= d; a31 ^= d; a32 ^= d; a33 ^= d; a34 ^= d;                              \
    d = c3 ^ ROL64(c0, 1u);                                                        \
    a40 ^= d; a41 ^= d; a42 ^= d; a43 ^= d; a44 ^= d;                              \
                                                                                   \
    KECCAK_RHOPI_CHI_IOTA(rc_);                                                    \
} while (0)

#define KECCAK_FIRST_ROUND_SHA3_256() do {                                         \
    uint2 c0 = a00;                                                                \
    uint2 c1 = a10 ^ a13;                                                          \
    uint2 c2 = a20;                                                                \
    uint2 c3 = a30;                                                                \
    uint2 c4 = a40;                                                                \
                                                                                   \
    uint2 d0 = c4 ^ ROL64(c1, 1u);                                                 \
    uint2 d1 = c0 ^ ROL64(c2, 1u);                                                 \
    uint2 d2 = c1 ^ ROL64(c3, 1u);                                                 \
    uint2 d3 = c2 ^ ROL64(c4, 1u);                                                 \
    uint2 d4 = c3 ^ ROL64(c0, 1u);                                                 \
                                                                                   \
    a00 ^= d0; a01 = d0; a02 = d0; a03 = d0; a04 = d0;                             \
    a10 ^= d1; a11 = d1; a12 = d1; a13 ^= d1; a14 = d1;                            \
    a20 ^= d2; a21 = d2; a22 = d2; a23 = d2; a24 = d2;                             \
    a30 ^= d3; a31 = d3; a32 = d3; a33 = d3; a34 = d3;                             \
    a40 ^= d4; a41 = d4; a42 = d4; a43 = d4; a44 = d4;                             \
                                                                                   \
    KECCAK_RHOPI_CHI_IOTA(U64(0x00000001u, 0x00000000u));                          \
} while (0)

#define KECCAK_FIRST_ROUND_SHAKE128() do {                                         \
    uint2 c0 = a00 ^ a04;                                                          \
    uint2 c1 = a10;                                                                \
    uint2 c2 = a20;                                                                \
    uint2 c3 = a30;                                                                \
    uint2 c4 = a40;                                                                \
                                                                                   \
    uint2 d0 = c4 ^ ROL64(c1, 1u);                                                 \
    uint2 d1 = c0 ^ ROL64(c2, 1u);                                                 \
    uint2 d2 = c1 ^ ROL64(c3, 1u);                                                 \
    uint2 d3 = c2 ^ ROL64(c4, 1u);                                                 \
    uint2 d4 = c3 ^ ROL64(c0, 1u);                                                 \
                                                                                   \
    a00 ^= d0; a01 = d0; a02 = d0; a03 = d0; a04 ^= d0;                            \
    a10 ^= d1; a11 = d1; a12 = d1; a13 = d1; a14 = d1;                             \
    a20 ^= d2; a21 = d2; a22 = d2; a23 = d2; a24 = d2;                             \
    a30 ^= d3; a31 = d3; a32 = d3; a33 = d3; a34 = d3;                             \
    a40 ^= d4; a41 = d4; a42 = d4; a43 = d4; a44 = d4;                             \
                                                                                   \
    KECCAK_RHOPI_CHI_IOTA(U64(0x00000001u, 0x00000000u));                          \
} while (0)

#define KECCAK_ROUNDS_1_TO_21() do {                       \
    KECCAK_ROUND(U64(0x00008082u, 0x00000000u));            \
    KECCAK_ROUND(U64(0x0000808Au, 0x80000000u));            \
    KECCAK_ROUND(U64(0x80008000u, 0x80000000u));            \
    KECCAK_ROUND(U64(0x0000808Bu, 0x00000000u));            \
    KECCAK_ROUND(U64(0x80000001u, 0x00000000u));            \
    KECCAK_ROUND(U64(0x80008081u, 0x80000000u));            \
    KECCAK_ROUND(U64(0x00008009u, 0x80000000u));            \
    KECCAK_ROUND(U64(0x0000008Au, 0x00000000u));            \
    KECCAK_ROUND(U64(0x00000088u, 0x00000000u));            \
    KECCAK_ROUND(U64(0x80008009u, 0x00000000u));            \
    KECCAK_ROUND(U64(0x8000000Au, 0x00000000u));            \
    KECCAK_ROUND(U64(0x8000808Bu, 0x00000000u));            \
    KECCAK_ROUND(U64(0x0000008Bu, 0x80000000u));            \
    KECCAK_ROUND(U64(0x00008089u, 0x80000000u));            \
    KECCAK_ROUND(U64(0x00008003u, 0x80000000u));            \
    KECCAK_ROUND(U64(0x00008002u, 0x80000000u));            \
    KECCAK_ROUND(U64(0x00000080u, 0x80000000u));            \
    KECCAK_ROUND(U64(0x0000800Au, 0x00000000u));            \
    KECCAK_ROUND(U64(0x8000000Au, 0x80000000u));            \
    KECCAK_ROUND(U64(0x80008081u, 0x80000000u));            \
    KECCAK_ROUND(U64(0x00008080u, 0x80000000u));            \
} while (0)

#define KECCAK_ROUNDS_1_TO_22() do {                       \
    KECCAK_ROUNDS_1_TO_21();                               \
    KECCAK_ROUND(U64(0x80000001u, 0x00000000u));            \
} while (0)

#define KECCAK_ROUNDS_0_TO_22() do {                       \
    KECCAK_ROUND(U64(0x00000001u, 0x00000000u));            \
    KECCAK_ROUNDS_1_TO_22();                               \
} while (0)

#define KECCAK_PERMUTE() do {                              \
    KECCAK_ROUND(U64(0x00000001u, 0x00000000u));            \
    KECCAK_ROUNDS_1_TO_22();                               \
    KECCAK_ROUND(U64(0x80008008u, 0x80000000u));            \
} while (0)

#define KECCAK_PENULTIMATE_FINAL4_STORE(rc22_, rc23_, base_) do {                  \
    uint2 c0 = a00 ^ a01 ^ a02 ^ a03 ^ a04;                                        \
    uint2 c1 = a10 ^ a11 ^ a12 ^ a13 ^ a14;                                        \
    uint2 c2 = a20 ^ a21 ^ a22 ^ a23 ^ a24;                                        \
    uint2 c3 = a30 ^ a31 ^ a32 ^ a33 ^ a34;                                        \
    uint2 c4 = a40 ^ a41 ^ a42 ^ a43 ^ a44;                                        \
                                                                                   \
    uint2 d0 = c4 ^ ROL64(c1, 1u);                                                 \
    uint2 d1 = c0 ^ ROL64(c2, 1u);                                                 \
    uint2 d2 = c1 ^ ROL64(c3, 1u);                                                 \
    uint2 d3 = c2 ^ ROL64(c4, 1u);                                                 \
    uint2 d4 = c3 ^ ROL64(c0, 1u);                                                 \
                                                                                   \
    uint2 b0, b1, b2, b3, b4;                                                      \
    uint2 o0, o1, o2, o3, o4;                                                      \
                                                                                   \
    b0 = a00 ^ d0;                                                                 \
    b1 = ROL64(a11 ^ d1, 44u);                                                     \
    b2 = ROL64(a22 ^ d2, 43u);                                                     \
    b3 = ROL64(a33 ^ d3, 21u);                                                     \
    b4 = ROL64(a44 ^ d4, 14u);                                                     \
    o0 = (b0 ^ (b2 & ~b1)) ^ (rc22_);                                              \
    o1 =  b1 ^ (b3 & ~b2);                                                         \
    o2 =  b2 ^ (b4 & ~b3);                                                         \
    o3 =  b3 ^ (b0 & ~b4);                                                         \
    o4 =  b4 ^ (b1 & ~b0);                                                         \
    uint2 nc0 = o0, nc1 = o1, nc2 = o2, nc3 = o3, nc4 = o4;                        \
    uint2 g0 = o0;                                                                 \
                                                                                   \
    b0 = ROL64(a30 ^ d3, 28u);                                                     \
    b1 = ROL64(a41 ^ d4, 20u);                                                     \
    b2 = ROL64(a02 ^ d0,  3u);                                                     \
    b3 = ROL64(a13 ^ d1, 45u);                                                     \
    b4 = ROL64(a24 ^ d2, 61u);                                                     \
    o0 = b0 ^ (b2 & ~b1);                                                          \
    o1 = b1 ^ (b3 & ~b2);                                                          \
    o2 = b2 ^ (b4 & ~b3);                                                          \
    o3 = b3 ^ (b0 & ~b4);                                                          \
    o4 = b4 ^ (b1 & ~b0);                                                          \
    nc0 ^= o0; nc1 ^= o1; nc2 ^= o2; nc3 ^= o3; nc4 ^= o4;                         \
    uint2 g1 = o1;                                                                 \
                                                                                   \
    b0 = ROL64(a10 ^ d1,  1u);                                                     \
    b1 = ROL64(a21 ^ d2,  6u);                                                     \
    b2 = ROL64(a32 ^ d3, 25u);                                                     \
    b3 = ROL64(a43 ^ d4,  8u);                                                     \
    b4 = ROL64(a04 ^ d0, 18u);                                                     \
    o0 = b0 ^ (b2 & ~b1);                                                          \
    o1 = b1 ^ (b3 & ~b2);                                                          \
    o2 = b2 ^ (b4 & ~b3);                                                          \
    o3 = b3 ^ (b0 & ~b4);                                                          \
    o4 = b4 ^ (b1 & ~b0);                                                          \
    nc0 ^= o0; nc1 ^= o1; nc2 ^= o2; nc3 ^= o3; nc4 ^= o4;                         \
    uint2 g2 = o2;                                                                 \
                                                                                   \
    b0 = ROL64(a40 ^ d4, 27u);                                                     \
    b1 = ROL64(a01 ^ d0, 36u);                                                     \
    b2 = ROL64(a12 ^ d1, 10u);                                                     \
    b3 = ROL64(a23 ^ d2, 15u);                                                     \
    b4 = ROL64(a34 ^ d3, 56u);                                                     \
    o0 = b0 ^ (b2 & ~b1);                                                          \
    o1 = b1 ^ (b3 & ~b2);                                                          \
    o2 = b2 ^ (b4 & ~b3);                                                          \
    o3 = b3 ^ (b0 & ~b4);                                                          \
    o4 = b4 ^ (b1 & ~b0);                                                          \
    nc0 ^= o0; nc1 ^= o1; nc2 ^= o2; nc3 ^= o3; nc4 ^= o4;                         \
    uint2 g3 = o3;                                                                 \
                                                                                   \
    b0 = ROL64(a20 ^ d2, 62u);                                                     \
    b1 = ROL64(a31 ^ d3, 55u);                                                     \
    b2 = ROL64(a42 ^ d4, 39u);                                                     \
    b3 = ROL64(a03 ^ d0, 41u);                                                     \
    b4 = ROL64(a14 ^ d1,  2u);                                                     \
    o0 = b0 ^ (b2 & ~b1);                                                          \
    o1 = b1 ^ (b3 & ~b2);                                                          \
    o2 = b2 ^ (b4 & ~b3);                                                          \
    o3 = b3 ^ (b0 & ~b4);                                                          \
    o4 = b4 ^ (b1 & ~b0);                                                          \
    nc0 ^= o0; nc1 ^= o1; nc2 ^= o2; nc3 ^= o3; nc4 ^= o4;                         \
    uint2 g4 = o4;                                                                 \
                                                                                   \
    uint2 fd0 = nc4 ^ ROL64(nc1, 1u);                                              \
    uint2 fd1 = nc0 ^ ROL64(nc2, 1u);                                              \
    uint2 fd2 = nc1 ^ ROL64(nc3, 1u);                                              \
    uint2 fd3 = nc2 ^ ROL64(nc4, 1u);                                              \
    uint2 fd4 = nc3 ^ ROL64(nc0, 1u);                                              \
                                                                                   \
    b0 = g0 ^ fd0;                                                                 \
    b1 = ROL64(g1 ^ fd1, 44u);                                                     \
    b2 = ROL64(g2 ^ fd2, 43u);                                                     \
    b3 = ROL64(g3 ^ fd3, 21u);                                                     \
    b4 = ROL64(g4 ^ fd4, 14u);                                                     \
                                                                                   \
    uint _base = (base_);                                                          \
    STORE64(out_data, _base + 0u, (b0 ^ (b2 & ~b1)) ^ (rc23_));                    \
    STORE64(out_data, _base + 1u,  b1 ^ (b3 & ~b2));                               \
    STORE64(out_data, _base + 2u,  b2 ^ (b4 & ~b3));                               \
    STORE64(out_data, _base + 3u,  b3 ^ (b0 & ~b4));                               \
} while (0)

#define KECCAK_FINAL11_STORE(rc_, base_) do {                                      \
    uint2 c0 = a00 ^ a01 ^ a02 ^ a03 ^ a04;                                        \
    uint2 c1 = a10 ^ a11 ^ a12 ^ a13 ^ a14;                                        \
    uint2 c2 = a20 ^ a21 ^ a22 ^ a23 ^ a24;                                        \
    uint2 c3 = a30 ^ a31 ^ a32 ^ a33 ^ a34;                                        \
    uint2 c4 = a40 ^ a41 ^ a42 ^ a43 ^ a44;                                        \
                                                                                   \
    uint2 d0 = c4 ^ ROL64(c1, 1u);                                                 \
    uint2 d1 = c0 ^ ROL64(c2, 1u);                                                 \
    uint2 d2 = c1 ^ ROL64(c3, 1u);                                                 \
    uint2 d3 = c2 ^ ROL64(c4, 1u);                                                 \
    uint2 d4 = c3 ^ ROL64(c0, 1u);                                                 \
                                                                                   \
    uint _base = (base_);                                                          \
    uint2 b0, b1, b2, b3, b4;                                                      \
                                                                                   \
    b0 = a00 ^ d0;                                                                 \
    b1 = ROL64(a11 ^ d1, 44u);                                                     \
    b2 = ROL64(a22 ^ d2, 43u);                                                     \
    b3 = ROL64(a33 ^ d3, 21u);                                                     \
    b4 = ROL64(a44 ^ d4, 14u);                                                     \
    STORE64(out_data, _base + 0u, (b0 ^ (b2 & ~b1)) ^ (rc_));                      \
    STORE64(out_data, _base + 1u,  b1 ^ (b3 & ~b2));                               \
    STORE64(out_data, _base + 2u,  b2 ^ (b4 & ~b3));                               \
    STORE64(out_data, _base + 3u,  b3 ^ (b0 & ~b4));                               \
    STORE64(out_data, _base + 4u,  b4 ^ (b1 & ~b0));                               \
                                                                                   \
    b0 = ROL64(a30 ^ d3, 28u);                                                     \
    b1 = ROL64(a41 ^ d4, 20u);                                                     \
    b2 = ROL64(a02 ^ d0,  3u);                                                     \
    b3 = ROL64(a13 ^ d1, 45u);                                                     \
    b4 = ROL64(a24 ^ d2, 61u);                                                     \
    STORE64(out_data, _base + 5u,  b0 ^ (b2 & ~b1));                               \
    STORE64(out_data, _base + 6u,  b1 ^ (b3 & ~b2));                               \
    STORE64(out_data, _base + 7u,  b2 ^ (b4 & ~b3));                               \
    STORE64(out_data, _base + 8u,  b3 ^ (b0 & ~b4));                               \
    STORE64(out_data, _base + 9u,  b4 ^ (b1 & ~b0));                               \
                                                                                   \
    b0 = ROL64(a10 ^ d1,  1u);                                                     \
    b1 = ROL64(a21 ^ d2,  6u);                                                     \
    b2 = ROL64(a32 ^ d3, 25u);                                                     \
    STORE64(out_data, _base +10u,  b0 ^ (b2 & ~b1));                               \
} while (0)

#define LOAD_MSG_LANE(n_, var_) do {                       \
    if (msg_lanes > (uint)(n_)) {                           \
        (var_) = LOAD64(in_data, in_base + (uint)(n_));     \
    }                                                       \
} while (0)

#define STORE_PREFIX(base_, cnt_) do {                      \
    uint _base = (base_);                                   \
    uint _cnt  = (cnt_);                                    \
    if (_cnt >  0u) STORE64(out_data, _base +  0u, a00);    \
    if (_cnt >  1u) STORE64(out_data, _base +  1u, a10);    \
    if (_cnt >  2u) STORE64(out_data, _base +  2u, a20);    \
    if (_cnt >  3u) STORE64(out_data, _base +  3u, a30);    \
    if (_cnt >  4u) STORE64(out_data, _base +  4u, a40);    \
    if (_cnt >  5u) STORE64(out_data, _base +  5u, a01);    \
    if (_cnt >  6u) STORE64(out_data, _base +  6u, a11);    \
    if (_cnt >  7u) STORE64(out_data, _base +  7u, a21);    \
    if (_cnt >  8u) STORE64(out_data, _base +  8u, a31);    \
    if (_cnt >  9u) STORE64(out_data, _base +  9u, a41);    \
    if (_cnt > 10u) STORE64(out_data, _base + 10u, a02);    \
    if (_cnt > 11u) STORE64(out_data, _base + 11u, a12);    \
    if (_cnt > 12u) STORE64(out_data, _base + 12u, a22);    \
    if (_cnt > 13u) STORE64(out_data, _base + 13u, a32);    \
    if (_cnt > 14u) STORE64(out_data, _base + 14u, a42);    \
    if (_cnt > 15u) STORE64(out_data, _base + 15u, a03);    \
    if (_cnt > 16u) STORE64(out_data, _base + 16u, a13);    \
    if (_cnt > 17u) STORE64(out_data, _base + 17u, a23);    \
    if (_cnt > 18u) STORE64(out_data, _base + 18u, a33);    \
    if (_cnt > 19u) STORE64(out_data, _base + 19u, a43);    \
    if (_cnt > 20u) STORE64(out_data, _base + 20u, a04);    \
    if (_cnt > 21u) STORE64(out_data, _base + 21u, a14);    \
    if (_cnt > 22u) STORE64(out_data, _base + 22u, a24);    \
    if (_cnt > 23u) STORE64(out_data, _base + 23u, a34);    \
    if (_cnt > 24u) STORE64(out_data, _base + 24u, a44);    \
} while (0)

#define STORE_21(base_) do {                                \
    uint _base = (base_);                                   \
    STORE64(out_data, _base +  0u, a00);                    \
    STORE64(out_data, _base +  1u, a10);                    \
    STORE64(out_data, _base +  2u, a20);                    \
    STORE64(out_data, _base +  3u, a30);                    \
    STORE64(out_data, _base +  4u, a40);                    \
    STORE64(out_data, _base +  5u, a01);                    \
    STORE64(out_data, _base +  6u, a11);                    \
    STORE64(out_data, _base +  7u, a21);                    \
    STORE64(out_data, _base +  8u, a31);                    \
    STORE64(out_data, _base +  9u, a41);                    \
    STORE64(out_data, _base + 10u, a02);                    \
    STORE64(out_data, _base + 11u, a12);                    \
    STORE64(out_data, _base + 12u, a22);                    \
    STORE64(out_data, _base + 13u, a32);                    \
    STORE64(out_data, _base + 14u, a42);                    \
    STORE64(out_data, _base + 15u, a03);                    \
    STORE64(out_data, _base + 16u, a13);                    \
    STORE64(out_data, _base + 17u, a23);                    \
    STORE64(out_data, _base + 18u, a33);                    \
    STORE64(out_data, _base + 19u, a43);                    \
    STORE64(out_data, _base + 20u, a04);                    \
} while (0)

kernel void keccak_f1600_batch(
    device const ulong *in_data    [[buffer(0)]],
    device       ulong *out_data   [[buffer(1)]],
    constant uint      &batch      [[buffer(2)]],
    constant uint      &msg_bytes  [[buffer(3)]],
    constant uint      &rate_bytes [[buffer(4)]],
    constant uint      &out_bytes  [[buffer(5)]],
    constant uint      &domain     [[buffer(6)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= batch) return;

    if (msg_bytes == 32u && rate_bytes == 136u && out_bytes == 32u && ((domain & 0xFFu) == 0x06u)) {
        uint in_base  = idx << 2;
        uint out_base = idx << 2;

        uint2 a00 = LOAD64(in_data, in_base + 0u);
        uint2 a10 = LOAD64(in_data, in_base + 1u);
        uint2 a20 = LOAD64(in_data, in_base + 2u);
        uint2 a30 = LOAD64(in_data, in_base + 3u);
        uint2 a40 = U64(0x00000006u, 0x00000000u);

        uint2 a01 = Z64, a11 = Z64, a21 = Z64, a31 = Z64, a41 = Z64;
        uint2 a02 = Z64, a12 = Z64, a22 = Z64, a32 = Z64, a42 = Z64;
        uint2 a03 = Z64, a13 = U64(0x00000000u, 0x80000000u), a23 = Z64, a33 = Z64, a43 = Z64;
        uint2 a04 = Z64, a14 = Z64, a24 = Z64, a34 = Z64, a44 = Z64;

        KECCAK_FIRST_ROUND_SHA3_256();
        KECCAK_ROUNDS_1_TO_21();
        KECCAK_PENULTIMATE_FINAL4_STORE(U64(0x80000001u, 0x00000000u),
                                        U64(0x80008008u, 0x80000000u),
                                        out_base);
        return;
    }

    if (msg_bytes == 32u && rate_bytes == 168u && out_bytes == 256u && ((domain & 0xFFu) == 0x1Fu)) {
        uint in_base  = idx << 2;
        uint out_base = idx << 5;

        uint2 a00 = LOAD64(in_data, in_base + 0u);
        uint2 a10 = LOAD64(in_data, in_base + 1u);
        uint2 a20 = LOAD64(in_data, in_base + 2u);
        uint2 a30 = LOAD64(in_data, in_base + 3u);
        uint2 a40 = U64(0x0000001Fu, 0x00000000u);

        uint2 a01 = Z64, a11 = Z64, a21 = Z64, a31 = Z64, a41 = Z64;
        uint2 a02 = Z64, a12 = Z64, a22 = Z64, a32 = Z64, a42 = Z64;
        uint2 a03 = Z64, a13 = Z64, a23 = Z64, a33 = Z64, a43 = Z64;
        uint2 a04 = U64(0x00000000u, 0x80000000u), a14 = Z64, a24 = Z64, a34 = Z64, a44 = Z64;

        KECCAK_FIRST_ROUND_SHAKE128();
        KECCAK_ROUNDS_1_TO_22();
        KECCAK_ROUND(U64(0x80008008u, 0x80000000u));
        STORE_21(out_base);

        KECCAK_ROUNDS_0_TO_22();
        KECCAK_FINAL11_STORE(U64(0x80008008u, 0x80000000u), out_base + 21u);
        return;
    }

    uint msg_lanes  = msg_bytes  >> 3;
    uint rate_lanes = rate_bytes >> 3;
    uint out_lanes  = out_bytes  >> 3;

    uint in_base = idx * msg_lanes;

    uint2 a00 = Z64, a10 = Z64, a20 = Z64, a30 = Z64, a40 = Z64;
    uint2 a01 = Z64, a11 = Z64, a21 = Z64, a31 = Z64, a41 = Z64;
    uint2 a02 = Z64, a12 = Z64, a22 = Z64, a32 = Z64, a42 = Z64;
    uint2 a03 = Z64, a13 = Z64, a23 = Z64, a33 = Z64, a43 = Z64;
    uint2 a04 = Z64, a14 = Z64, a24 = Z64, a34 = Z64, a44 = Z64;

    uint2 dom = U64(domain & 0xFFu, 0u);

    if (msg_lanes == 4u) {
        a00 = LOAD64(in_data, in_base + 0u);
        a10 = LOAD64(in_data, in_base + 1u);
        a20 = LOAD64(in_data, in_base + 2u);
        a30 = LOAD64(in_data, in_base + 3u);
        a40 = dom;
    } else {
        LOAD_MSG_LANE( 0, a00);
        LOAD_MSG_LANE( 1, a10);
        LOAD_MSG_LANE( 2, a20);
        LOAD_MSG_LANE( 3, a30);
        LOAD_MSG_LANE( 4, a40);
        LOAD_MSG_LANE( 5, a01);
        LOAD_MSG_LANE( 6, a11);
        LOAD_MSG_LANE( 7, a21);
        LOAD_MSG_LANE( 8, a31);
        LOAD_MSG_LANE( 9, a41);
        LOAD_MSG_LANE(10, a02);
        LOAD_MSG_LANE(11, a12);
        LOAD_MSG_LANE(12, a22);
        LOAD_MSG_LANE(13, a32);
        LOAD_MSG_LANE(14, a42);
        LOAD_MSG_LANE(15, a03);
        LOAD_MSG_LANE(16, a13);
        LOAD_MSG_LANE(17, a23);
        LOAD_MSG_LANE(18, a33);
        LOAD_MSG_LANE(19, a43);
        LOAD_MSG_LANE(20, a04);
        LOAD_MSG_LANE(21, a14);
        LOAD_MSG_LANE(22, a24);
        LOAD_MSG_LANE(23, a34);
        LOAD_MSG_LANE(24, a44);

        switch (msg_lanes) {
            case  0u: a00 ^= dom; break;
            case  1u: a10 ^= dom; break;
            case  2u: a20 ^= dom; break;
            case  3u: a30 ^= dom; break;
            case  4u: a40 ^= dom; break;
            case  5u: a01 ^= dom; break;
            case  6u: a11 ^= dom; break;
            case  7u: a21 ^= dom; break;
            case  8u: a31 ^= dom; break;
            case  9u: a41 ^= dom; break;
            case 10u: a02 ^= dom; break;
            case 11u: a12 ^= dom; break;
            case 12u: a22 ^= dom; break;
            case 13u: a32 ^= dom; break;
            case 14u: a42 ^= dom; break;
            case 15u: a03 ^= dom; break;
            case 16u: a13 ^= dom; break;
            case 17u: a23 ^= dom; break;
            case 18u: a33 ^= dom; break;
            case 19u: a43 ^= dom; break;
            case 20u: a04 ^= dom; break;
            case 21u: a14 ^= dom; break;
            case 22u: a24 ^= dom; break;
            case 23u: a34 ^= dom; break;
            case 24u: a44 ^= dom; break;
            default: break;
        }
    }

    const uint2 pad80 = U64(0x00000000u, 0x80000000u);
    if (rate_lanes == 17u) {
        a13 ^= pad80;
    } else if (rate_lanes == 21u) {
        a04 ^= pad80;
    } else {
        switch (rate_lanes - 1u) {
            case  0u: a00 ^= pad80; break;
            case  1u: a10 ^= pad80; break;
            case  2u: a20 ^= pad80; break;
            case  3u: a30 ^= pad80; break;
            case  4u: a40 ^= pad80; break;
            case  5u: a01 ^= pad80; break;
            case  6u: a11 ^= pad80; break;
            case  7u: a21 ^= pad80; break;
            case  8u: a31 ^= pad80; break;
            case  9u: a41 ^= pad80; break;
            case 10u: a02 ^= pad80; break;
            case 11u: a12 ^= pad80; break;
            case 12u: a22 ^= pad80; break;
            case 13u: a32 ^= pad80; break;
            case 14u: a42 ^= pad80; break;
            case 15u: a03 ^= pad80; break;
            case 16u: a13 ^= pad80; break;
            case 17u: a23 ^= pad80; break;
            case 18u: a33 ^= pad80; break;
            case 19u: a43 ^= pad80; break;
            case 20u: a04 ^= pad80; break;
            case 21u: a14 ^= pad80; break;
            case 22u: a24 ^= pad80; break;
            case 23u: a34 ^= pad80; break;
            case 24u: a44 ^= pad80; break;
            default: break;
        }
    }

    uint out_base = idx * out_lanes;
    uint written = 0u;

    for (;;) {
        KECCAK_PERMUTE();

        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        STORE_PREFIX(out_base + written, take);
        written += take;

        if (written >= out_lanes) return;
    }
}

#undef STORE_21
#undef STORE_PREFIX
#undef LOAD_MSG_LANE
#undef KECCAK_FINAL11_STORE
#undef KECCAK_PENULTIMATE_FINAL4_STORE
#undef KECCAK_PERMUTE
#undef KECCAK_ROUNDS_0_TO_22
#undef KECCAK_ROUNDS_1_TO_22
#undef KECCAK_ROUNDS_1_TO_21
#undef KECCAK_FIRST_ROUND_SHAKE128
#undef KECCAK_FIRST_ROUND_SHA3_256
#undef KECCAK_ROUND
#undef KECCAK_RHOPI_CHI_IOTA
#undef KECCAK_RHOPI_CHAIN
#undef RHOPI_STEP
#undef ROL64
#undef STORE64
#undef LOAD64
#undef Z64
#undef U64