#include <metal_stdlib>
using namespace metal;

#define ROL64(x, n) (((x) << (n)) | ((x) >> (64u - (n))))

#define RHOPI_STEP(tvar, dst, rot) do {        \
    ulong _tmp = (dst);                        \
    (dst) = ROL64((tvar), (rot));              \
    (tvar) = _tmp;                             \
} while (0)

#define KECCAK_RHOPI_CHAIN() do {              \
    ulong t0 = a10;                            \
    ulong t1 = a40;                            \
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
    ulong b0, b1, b2, b3, b4;                                                       \
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
    ulong c0 = a00 ^ a01 ^ a02 ^ a03 ^ a04;                                        \
    ulong c1 = a10 ^ a11 ^ a12 ^ a13 ^ a14;                                        \
    ulong c2 = a20 ^ a21 ^ a22 ^ a23 ^ a24;                                        \
    ulong c3 = a30 ^ a31 ^ a32 ^ a33 ^ a34;                                        \
    ulong c4 = a40 ^ a41 ^ a42 ^ a43 ^ a44;                                        \
                                                                                   \
    ulong d = c4 ^ ROL64(c1, 1u);                                                  \
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
    ulong c0 = a00;                                                                \
    ulong c1 = a10 ^ a13;                                                          \
    ulong c2 = a20;                                                                \
    ulong c3 = a30;                                                                \
    ulong c4 = a40;                                                                \
                                                                                   \
    ulong d0 = c4 ^ ROL64(c1, 1u);                                                 \
    ulong d1 = c0 ^ ROL64(c2, 1u);                                                 \
    ulong d2 = c1 ^ ROL64(c3, 1u);                                                 \
    ulong d3 = c2 ^ ROL64(c4, 1u);                                                 \
    ulong d4 = c3 ^ ROL64(c0, 1u);                                                 \
                                                                                   \
    a00 ^= d0; a01 = d0; a02 = d0; a03 = d0; a04 = d0;                             \
    a10 ^= d1; a11 = d1; a12 = d1; a13 ^= d1; a14 = d1;                            \
    a20 ^= d2; a21 = d2; a22 = d2; a23 = d2; a24 = d2;                             \
    a30 ^= d3; a31 = d3; a32 = d3; a33 = d3; a34 = d3;                             \
    a40 ^= d4; a41 = d4; a42 = d4; a43 = d4; a44 = d4;                             \
                                                                                   \
    KECCAK_RHOPI_CHI_IOTA(0x0000000000000001ul);                                  \
} while (0)

#define KECCAK_FIRST_ROUND_SHAKE128() do {                                         \
    ulong c0 = a00 ^ a04;                                                          \
    ulong c1 = a10;                                                                \
    ulong c2 = a20;                                                                \
    ulong c3 = a30;                                                                \
    ulong c4 = a40;                                                                \
                                                                                   \
    ulong d0 = c4 ^ ROL64(c1, 1u);                                                 \
    ulong d1 = c0 ^ ROL64(c2, 1u);                                                 \
    ulong d2 = c1 ^ ROL64(c3, 1u);                                                 \
    ulong d3 = c2 ^ ROL64(c4, 1u);                                                 \
    ulong d4 = c3 ^ ROL64(c0, 1u);                                                 \
                                                                                   \
    a00 ^= d0; a01 = d0; a02 = d0; a03 = d0; a04 ^= d0;                            \
    a10 ^= d1; a11 = d1; a12 = d1; a13 = d1; a14 = d1;                             \
    a20 ^= d2; a21 = d2; a22 = d2; a23 = d2; a24 = d2;                             \
    a30 ^= d3; a31 = d3; a32 = d3; a33 = d3; a34 = d3;                             \
    a40 ^= d4; a41 = d4; a42 = d4; a43 = d4; a44 = d4;                             \
                                                                                   \
    KECCAK_RHOPI_CHI_IOTA(0x0000000000000001ul);                                  \
} while (0)

#define KECCAK_ROUNDS_1_TO_21() do {                       \
    KECCAK_ROUND(0x0000000000008082ul);                    \
    KECCAK_ROUND(0x800000000000808Aul);                    \
    KECCAK_ROUND(0x8000000080008000ul);                    \
    KECCAK_ROUND(0x000000000000808Bul);                    \
    KECCAK_ROUND(0x0000000080000001ul);                    \
    KECCAK_ROUND(0x8000000080008081ul);                    \
    KECCAK_ROUND(0x8000000000008009ul);                    \
    KECCAK_ROUND(0x000000000000008Aul);                    \
    KECCAK_ROUND(0x0000000000000088ul);                    \
    KECCAK_ROUND(0x0000000080008009ul);                    \
    KECCAK_ROUND(0x000000008000000Aul);                    \
    KECCAK_ROUND(0x000000008000808Bul);                    \
    KECCAK_ROUND(0x800000000000008Bul);                    \
    KECCAK_ROUND(0x8000000000008089ul);                    \
    KECCAK_ROUND(0x8000000000008003ul);                    \
    KECCAK_ROUND(0x8000000000008002ul);                    \
    KECCAK_ROUND(0x8000000000000080ul);                    \
    KECCAK_ROUND(0x000000000000800Aul);                    \
    KECCAK_ROUND(0x800000008000000Aul);                    \
    KECCAK_ROUND(0x8000000080008081ul);                    \
    KECCAK_ROUND(0x8000000000008080ul);                    \
} while (0)

#define KECCAK_ROUNDS_1_TO_22() do {                       \
    KECCAK_ROUNDS_1_TO_21();                               \
    KECCAK_ROUND(0x0000000080000001ul);                    \
} while (0)

#define KECCAK_ROUNDS_0_TO_22() do {                       \
    KECCAK_ROUND(0x0000000000000001ul);                    \
    KECCAK_ROUNDS_1_TO_22();                               \
} while (0)

#define KECCAK_PERMUTE() do {                              \
    KECCAK_ROUND(0x0000000000000001ul);                    \
    KECCAK_ROUNDS_1_TO_22();                               \
    KECCAK_ROUND(0x8000000080008008ul);                    \
} while (0)

#define KECCAK_PENULTIMATE_FINAL4_STORE(rc22_, rc23_, base_) do {                  \
    ulong c0 = a00 ^ a01 ^ a02 ^ a03 ^ a04;                                        \
    ulong c1 = a10 ^ a11 ^ a12 ^ a13 ^ a14;                                        \
    ulong c2 = a20 ^ a21 ^ a22 ^ a23 ^ a24;                                        \
    ulong c3 = a30 ^ a31 ^ a32 ^ a33 ^ a34;                                        \
    ulong c4 = a40 ^ a41 ^ a42 ^ a43 ^ a44;                                        \
                                                                                   \
    ulong d0 = c4 ^ ROL64(c1, 1u);                                                 \
    ulong d1 = c0 ^ ROL64(c2, 1u);                                                 \
    ulong d2 = c1 ^ ROL64(c3, 1u);                                                 \
    ulong d3 = c2 ^ ROL64(c4, 1u);                                                 \
    ulong d4 = c3 ^ ROL64(c0, 1u);                                                 \
                                                                                   \
    ulong b0, b1, b2, b3, b4;                                                      \
    ulong o0, o1, o2, o3, o4;                                                      \
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
    ulong nc0 = o0, nc1 = o1, nc2 = o2, nc3 = o3, nc4 = o4;                        \
    ulong g0 = o0;                                                                 \
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
    ulong g1 = o1;                                                                 \
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
    ulong g2 = o2;                                                                 \
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
    ulong g3 = o3;                                                                 \
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
    ulong g4 = o4;                                                                 \
                                                                                   \
    ulong fd0 = nc4 ^ ROL64(nc1, 1u);                                              \
    ulong fd1 = nc0 ^ ROL64(nc2, 1u);                                              \
    ulong fd2 = nc1 ^ ROL64(nc3, 1u);                                              \
    ulong fd3 = nc2 ^ ROL64(nc4, 1u);                                              \
    ulong fd4 = nc3 ^ ROL64(nc0, 1u);                                              \
                                                                                   \
    b0 = g0 ^ fd0;                                                                 \
    b1 = ROL64(g1 ^ fd1, 44u);                                                     \
    b2 = ROL64(g2 ^ fd2, 43u);                                                     \
    b3 = ROL64(g3 ^ fd3, 21u);                                                     \
    b4 = ROL64(g4 ^ fd4, 14u);                                                     \
                                                                                   \
    uint _base = (base_);                                                          \
    out_data[_base + 0u] = (b0 ^ (b2 & ~b1)) ^ (rc23_);                            \
    out_data[_base + 1u] =  b1 ^ (b3 & ~b2);                                       \
    out_data[_base + 2u] =  b2 ^ (b4 & ~b3);                                       \
    out_data[_base + 3u] =  b3 ^ (b0 & ~b4);                                       \
} while (0)

#define KECCAK_FINAL11_STORE(rc_, base_) do {                                      \
    ulong c0 = a00 ^ a01 ^ a02 ^ a03 ^ a04;                                        \
    ulong c1 = a10 ^ a11 ^ a12 ^ a13 ^ a14;                                        \
    ulong c2 = a20 ^ a21 ^ a22 ^ a23 ^ a24;                                        \
    ulong c3 = a30 ^ a31 ^ a32 ^ a33 ^ a34;                                        \
    ulong c4 = a40 ^ a41 ^ a42 ^ a43 ^ a44;                                        \
                                                                                   \
    ulong d0 = c4 ^ ROL64(c1, 1u);                                                 \
    ulong d1 = c0 ^ ROL64(c2, 1u);                                                 \
    ulong d2 = c1 ^ ROL64(c3, 1u);                                                 \
    ulong d3 = c2 ^ ROL64(c4, 1u);                                                 \
    ulong d4 = c3 ^ ROL64(c0, 1u);                                                 \
                                                                                   \
    uint _base = (base_);                                                          \
    ulong b0, b1, b2, b3, b4;                                                      \
                                                                                   \
    b0 = a00 ^ d0;                                                                 \
    b1 = ROL64(a11 ^ d1, 44u);                                                     \
    b2 = ROL64(a22 ^ d2, 43u);                                                     \
    b3 = ROL64(a33 ^ d3, 21u);                                                     \
    b4 = ROL64(a44 ^ d4, 14u);                                                     \
    out_data[_base + 0u] = (b0 ^ (b2 & ~b1)) ^ (rc_);                              \
    out_data[_base + 1u] =  b1 ^ (b3 & ~b2);                                       \
    out_data[_base + 2u] =  b2 ^ (b4 & ~b3);                                       \
    out_data[_base + 3u] =  b3 ^ (b0 & ~b4);                                       \
    out_data[_base + 4u] =  b4 ^ (b1 & ~b0);                                       \
                                                                                   \
    b0 = ROL64(a30 ^ d3, 28u);                                                     \
    b1 = ROL64(a41 ^ d4, 20u);                                                     \
    b2 = ROL64(a02 ^ d0,  3u);                                                     \
    b3 = ROL64(a13 ^ d1, 45u);                                                     \
    b4 = ROL64(a24 ^ d2, 61u);                                                     \
    out_data[_base + 5u] =  b0 ^ (b2 & ~b1);                                       \
    out_data[_base + 6u] =  b1 ^ (b3 & ~b2);                                       \
    out_data[_base + 7u] =  b2 ^ (b4 & ~b3);                                       \
    out_data[_base + 8u] =  b3 ^ (b0 & ~b4);                                       \
    out_data[_base + 9u] =  b4 ^ (b1 & ~b0);                                       \
                                                                                   \
    b0 = ROL64(a10 ^ d1,  1u);                                                     \
    b1 = ROL64(a21 ^ d2,  6u);                                                     \
    b2 = ROL64(a32 ^ d3, 25u);                                                     \
    out_data[_base +10u] =  b0 ^ (b2 & ~b1);                                       \
} while (0)

#define LOAD_MSG_LANE(n_, var_) do {                       \
    if (msg_lanes > (uint)(n_)) {                           \
        (var_) = in_data[in_base + (uint)(n_)];             \
    }                                                       \
} while (0)

#define STORE_PREFIX(base_, cnt_) do {                      \
    uint _base = (base_);                                   \
    uint _cnt  = (cnt_);                                    \
    if (_cnt >  0u) out_data[_base +  0u] = a00;            \
    if (_cnt >  1u) out_data[_base +  1u] = a10;            \
    if (_cnt >  2u) out_data[_base +  2u] = a20;            \
    if (_cnt >  3u) out_data[_base +  3u] = a30;            \
    if (_cnt >  4u) out_data[_base +  4u] = a40;            \
    if (_cnt >  5u) out_data[_base +  5u] = a01;            \
    if (_cnt >  6u) out_data[_base +  6u] = a11;            \
    if (_cnt >  7u) out_data[_base +  7u] = a21;            \
    if (_cnt >  8u) out_data[_base +  8u] = a31;            \
    if (_cnt >  9u) out_data[_base +  9u] = a41;            \
    if (_cnt > 10u) out_data[_base + 10u] = a02;            \
    if (_cnt > 11u) out_data[_base + 11u] = a12;            \
    if (_cnt > 12u) out_data[_base + 12u] = a22;            \
    if (_cnt > 13u) out_data[_base + 13u] = a32;            \
    if (_cnt > 14u) out_data[_base + 14u] = a42;            \
    if (_cnt > 15u) out_data[_base + 15u] = a03;            \
    if (_cnt > 16u) out_data[_base + 16u] = a13;            \
    if (_cnt > 17u) out_data[_base + 17u] = a23;            \
    if (_cnt > 18u) out_data[_base + 18u] = a33;            \
    if (_cnt > 19u) out_data[_base + 19u] = a43;            \
    if (_cnt > 20u) out_data[_base + 20u] = a04;            \
    if (_cnt > 21u) out_data[_base + 21u] = a14;            \
    if (_cnt > 22u) out_data[_base + 22u] = a24;            \
    if (_cnt > 23u) out_data[_base + 23u] = a34;            \
    if (_cnt > 24u) out_data[_base + 24u] = a44;            \
} while (0)

#define STORE_21(base_) do {                                \
    uint _base = (base_);                                   \
    out_data[_base +  0u] = a00;                            \
    out_data[_base +  1u] = a10;                            \
    out_data[_base +  2u] = a20;                            \
    out_data[_base +  3u] = a30;                            \
    out_data[_base +  4u] = a40;                            \
    out_data[_base +  5u] = a01;                            \
    out_data[_base +  6u] = a11;                            \
    out_data[_base +  7u] = a21;                            \
    out_data[_base +  8u] = a31;                            \
    out_data[_base +  9u] = a41;                            \
    out_data[_base + 10u] = a02;                            \
    out_data[_base + 11u] = a12;                            \
    out_data[_base + 12u] = a22;                            \
    out_data[_base + 13u] = a32;                            \
    out_data[_base + 14u] = a42;                            \
    out_data[_base + 15u] = a03;                            \
    out_data[_base + 16u] = a13;                            \
    out_data[_base + 17u] = a23;                            \
    out_data[_base + 18u] = a33;                            \
    out_data[_base + 19u] = a43;                            \
    out_data[_base + 20u] = a04;                            \
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

        ulong a00 = in_data[in_base + 0u];
        ulong a10 = in_data[in_base + 1u];
        ulong a20 = in_data[in_base + 2u];
        ulong a30 = in_data[in_base + 3u];
        ulong a40 = 0x0000000000000006ul;

        ulong a01 = 0ul, a11 = 0ul, a21 = 0ul, a31 = 0ul, a41 = 0ul;
        ulong a02 = 0ul, a12 = 0ul, a22 = 0ul, a32 = 0ul, a42 = 0ul;
        ulong a03 = 0ul, a13 = 0x8000000000000000ul, a23 = 0ul, a33 = 0ul, a43 = 0ul;
        ulong a04 = 0ul, a14 = 0ul, a24 = 0ul, a34 = 0ul, a44 = 0ul;

        KECCAK_FIRST_ROUND_SHA3_256();
        KECCAK_ROUNDS_1_TO_21();
        KECCAK_PENULTIMATE_FINAL4_STORE(0x0000000080000001ul, 0x8000000080008008ul, out_base);
        return;
    }

    if (msg_bytes == 32u && rate_bytes == 168u && out_bytes == 256u && ((domain & 0xFFu) == 0x1Fu)) {
        uint in_base  = idx << 2;
        uint out_base = idx << 5;

        ulong a00 = in_data[in_base + 0u];
        ulong a10 = in_data[in_base + 1u];
        ulong a20 = in_data[in_base + 2u];
        ulong a30 = in_data[in_base + 3u];
        ulong a40 = 0x000000000000001Ful;

        ulong a01 = 0ul, a11 = 0ul, a21 = 0ul, a31 = 0ul, a41 = 0ul;
        ulong a02 = 0ul, a12 = 0ul, a22 = 0ul, a32 = 0ul, a42 = 0ul;
        ulong a03 = 0ul, a13 = 0ul, a23 = 0ul, a33 = 0ul, a43 = 0ul;
        ulong a04 = 0x8000000000000000ul, a14 = 0ul, a24 = 0ul, a34 = 0ul, a44 = 0ul;

        KECCAK_FIRST_ROUND_SHAKE128();
        KECCAK_ROUNDS_1_TO_22();
        KECCAK_ROUND(0x8000000080008008ul);
        STORE_21(out_base);

        KECCAK_ROUNDS_0_TO_22();
        KECCAK_FINAL11_STORE(0x8000000080008008ul, out_base + 21u);
        return;
    }

    uint msg_lanes  = msg_bytes  >> 3;
    uint rate_lanes = rate_bytes >> 3;
    uint out_lanes  = out_bytes  >> 3;

    uint in_base = idx * msg_lanes;

    ulong a00 = 0ul, a10 = 0ul, a20 = 0ul, a30 = 0ul, a40 = 0ul;
    ulong a01 = 0ul, a11 = 0ul, a21 = 0ul, a31 = 0ul, a41 = 0ul;
    ulong a02 = 0ul, a12 = 0ul, a22 = 0ul, a32 = 0ul, a42 = 0ul;
    ulong a03 = 0ul, a13 = 0ul, a23 = 0ul, a33 = 0ul, a43 = 0ul;
    ulong a04 = 0ul, a14 = 0ul, a24 = 0ul, a34 = 0ul, a44 = 0ul;

    ulong dom = (ulong)(domain & 0xFFu);

    if (msg_lanes == 4u) {
        a00 = in_data[in_base + 0u];
        a10 = in_data[in_base + 1u];
        a20 = in_data[in_base + 2u];
        a30 = in_data[in_base + 3u];
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

    const ulong pad80 = 0x8000000000000000ul;
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