The kernel is rewritten around explicit 32-bit lane pairs (`uint2`) instead of relying on compiler-lowered 64-bit rotates/shifts. Keccak-f is bitwise-heavy, and Apple GPUs execute 32-bit integer operations more naturally; spelling rotates as two 32-bit halves exposes this directly while preserving exact 64-bit lane semantics. I also keep the common `msg_bytes=32` path direct, avoiding generic absorb switches for the SHA3/SHAKE workloads.  

```metal
#include <metal_stdlib>
using namespace metal;

static inline uint2 u64_to_u2(ulong x)
{
    return uint2((uint)x, (uint)(x >> 32));
}

static inline ulong u2_to_u64(uint2 v)
{
    return (((ulong)v.y) << 32) | ((ulong)v.x);
}

#define ROL2L(v, n) uint2((((v).x) << (n)) | (((v).y) >> (32 - (n))), \
                          (((v).y) << (n)) | (((v).x) >> (32 - (n))))

#define ROL2H(v, n) uint2((((v).y) << ((n) - 32)) | (((v).x) >> (64 - (n))), \
                          (((v).x) << ((n) - 32)) | (((v).y) >> (64 - (n))))

#define KECCAK_ROUND2(RCLO, RCHI) do {                                \
    uint2 c0 = a0 ^ a5 ^ a10 ^ a15 ^ a20;                              \
    uint2 c1 = a1 ^ a6 ^ a11 ^ a16 ^ a21;                              \
    uint2 c2 = a2 ^ a7 ^ a12 ^ a17 ^ a22;                              \
    uint2 c3 = a3 ^ a8 ^ a13 ^ a18 ^ a23;                              \
    uint2 c4 = a4 ^ a9 ^ a14 ^ a19 ^ a24;                              \
                                                                        \
    uint2 d0 = c4 ^ ROL2L(c1, 1);                                      \
    uint2 d1 = c0 ^ ROL2L(c2, 1);                                      \
    uint2 d2 = c1 ^ ROL2L(c3, 1);                                      \
    uint2 d3 = c2 ^ ROL2L(c4, 1);                                      \
    uint2 d4 = c3 ^ ROL2L(c0, 1);                                      \
                                                                        \
    a0 ^= d0;  a5 ^= d0;  a10 ^= d0;  a15 ^= d0;  a20 ^= d0;           \
    a1 ^= d1;  a6 ^= d1;  a11 ^= d1;  a16 ^= d1;  a21 ^= d1;           \
    a2 ^= d2;  a7 ^= d2;  a12 ^= d2;  a17 ^= d2;  a22 ^= d2;           \
    a3 ^= d3;  a8 ^= d3;  a13 ^= d3;  a18 ^= d3;  a23 ^= d3;           \
    a4 ^= d4;  a9 ^= d4;  a14 ^= d4;  a19 ^= d4;  a24 ^= d4;           \
                                                                        \
    uint2 t = a1;                                                       \
    uint2 u = a10; a10 = ROL2L(t,  1); t = u;                          \
          u = a7;  a7  = ROL2L(t,  3); t = u;                          \
          u = a11; a11 = ROL2L(t,  6); t = u;                          \
          u = a17; a17 = ROL2L(t, 10); t = u;                          \
          u = a18; a18 = ROL2L(t, 15); t = u;                          \
          u = a3;  a3  = ROL2L(t, 21); t = u;                          \
          u = a5;  a5  = ROL2L(t, 28); t = u;                          \
          u = a16; a16 = ROL2H(t, 36); t = u;                          \
          u = a8;  a8  = ROL2H(t, 45); t = u;                          \
          u = a21; a21 = ROL2H(t, 55); t = u;                          \
          u = a24; a24 = ROL2L(t,  2); t = u;                          \
          u = a4;  a4  = ROL2L(t, 14); t = u;                          \
          u = a15; a15 = ROL2L(t, 27); t = u;                          \
          u = a23; a23 = ROL2H(t, 41); t = u;                          \
          u = a19; a19 = ROL2H(t, 56); t = u;                          \
          u = a13; a13 = ROL2L(t,  8); t = u;                          \
          u = a12; a12 = ROL2L(t, 25); t = u;                          \
          u = a2;  a2  = ROL2H(t, 43); t = u;                          \
          u = a20; a20 = ROL2H(t, 62); t = u;                          \
          u = a14; a14 = ROL2L(t, 18); t = u;                          \
          u = a22; a22 = ROL2H(t, 39); t = u;                          \
          u = a9;  a9  = ROL2H(t, 61); t = u;                          \
          u = a6;  a6  = ROL2L(t, 20); t = u;                          \
    a1 = ROL2H(t, 44);                                                  \
                                                                        \
    c0 = a0; c1 = a1; c2 = a2; c3 = a3; c4 = a4;                       \
    a0 = c0 ^ ((~c1) & c2);                                             \
    a1 = c1 ^ ((~c2) & c3);                                             \
    a2 = c2 ^ ((~c3) & c4);                                             \
    a3 = c3 ^ ((~c4) & c0);                                             \
    a4 = c4 ^ ((~c0) & c1);                                             \
                                                                        \
    c0 = a5; c1 = a6; c2 = a7; c3 = a8; c4 = a9;                       \
    a5 = c0 ^ ((~c1) & c2);                                             \
    a6 = c1 ^ ((~c2) & c3);                                             \
    a7 = c2 ^ ((~c3) & c4);                                             \
    a8 = c3 ^ ((~c4) & c0);                                             \
    a9 = c4 ^ ((~c0) & c1);                                             \
                                                                        \
    c0 = a10; c1 = a11; c2 = a12; c3 = a13; c4 = a14;                  \
    a10 = c0 ^ ((~c1) & c2);                                            \
    a11 = c1 ^ ((~c2) & c3);                                            \
    a12 = c2 ^ ((~c3) & c4);                                            \
    a13 = c3 ^ ((~c4) & c0);                                            \
    a14 = c4 ^ ((~c0) & c1);                                            \
                                                                        \
    c0 = a15; c1 = a16; c2 = a17; c3 = a18; c4 = a19;                  \
    a15 = c0 ^ ((~c1) & c2);                                            \
    a16 = c1 ^ ((~c2) & c3);                                            \
    a17 = c2 ^ ((~c3) & c4);                                            \
    a18 = c3 ^ ((~c4) & c0);                                            \
    a19 = c4 ^ ((~c0) & c1);                                            \
                                                                        \
    c0 = a20; c1 = a21; c2 = a22; c3 = a23; c4 = a24;                  \
    a20 = c0 ^ ((~c1) & c2);                                            \
    a21 = c1 ^ ((~c2) & c3);                                            \
    a22 = c2 ^ ((~c3) & c4);                                            \
    a23 = c3 ^ ((~c4) & c0);                                            \
    a24 = c4 ^ ((~c0) & c1);                                            \
                                                                        \
    a0 ^= uint2((uint)(RCLO), (uint)(RCHI));                            \
} while (false)

#define KECCAK_PERMUTE2() do {                                         \
    KECCAK_ROUND2(0x00000001u, 0x00000000u);                           \
    KECCAK_ROUND2(0x00008082u, 0x00000000u);                           \
    KECCAK_ROUND2(0x0000808Au, 0x80000000u);                           \
    KECCAK_ROUND2(0x80008000u, 0x80000000u);                           \
    KECCAK_ROUND2(0x0000808Bu, 0x00000000u);                           \
    KECCAK_ROUND2(0x80000001u, 0x00000000u);                           \
    KECCAK_ROUND2(0x80008081u, 0x80000000u);                           \
    KECCAK_ROUND2(0x00008009u, 0x80000000u);                           \
    KECCAK_ROUND2(0x0000008Au, 0x00000000u);                           \
    KECCAK_ROUND2(0x00000088u, 0x00000000u);                           \
    KECCAK_ROUND2(0x80008009u, 0x00000000u);                           \
    KECCAK_ROUND2(0x8000000Au, 0x00000000u);                           \
    KECCAK_ROUND2(0x8000808Bu, 0x00000000u);                           \
    KECCAK_ROUND2(0x0000008Bu, 0x80000000u);                           \
    KECCAK_ROUND2(0x00008089u, 0x80000000u);                           \
    KECCAK_ROUND2(0x00008003u, 0x80000000u);                           \
    KECCAK_ROUND2(0x00008002u, 0x80000000u);                           \
    KECCAK_ROUND2(0x00000080u, 0x80000000u);                           \
    KECCAK_ROUND2(0x0000800Au, 0x00000000u);                           \
    KECCAK_ROUND2(0x8000000Au, 0x80000000u);                           \
    KECCAK_ROUND2(0x80008081u, 0x80000000u);                           \
    KECCAK_ROUND2(0x00008080u, 0x80000000u);                           \
    KECCAK_ROUND2(0x80000001u, 0x00000000u);                           \
    KECCAK_ROUND2(0x80008008u, 0x80000000u);                           \
} while (false)

#define XOR_TO_LANE2(LANE, VAL) do {                                   \
    switch (LANE) {                                                     \
        case 0u:  a0  ^= (VAL); break;                                  \
        case 1u:  a1  ^= (VAL); break;                                  \
        case 2u:  a2  ^= (VAL); break;                                  \
        case 3u:  a3  ^= (VAL); break;                                  \
        case 4u:  a4  ^= (VAL); break;                                  \
        case 5u:  a5  ^= (VAL); break;                                  \
        case 6u:  a6  ^= (VAL); break;                                  \
        case 7u:  a7  ^= (VAL); break;                                  \
        case 8u:  a8  ^= (VAL); break;                                  \
        case 9u:  a9  ^= (VAL); break;                                  \
        case 10u: a10 ^= (VAL); break;                                  \
        case 11u: a11 ^= (VAL); break;                                  \
        case 12u: a12 ^= (VAL); break;                                  \
        case 13u: a13 ^= (VAL); break;                                  \
        case 14u: a14 ^= (VAL); break;                                  \
        case 15u: a15 ^= (VAL); break;                                  \
        case 16u: a16 ^= (VAL); break;                                  \
        case 17u: a17 ^= (VAL); break;                                  \
        case 18u: a18 ^= (VAL); break;                                  \
        case 19u: a19 ^= (VAL); break;                                  \
        case 20u: a20 ^= (VAL); break;                                  \
        case 21u: a21 ^= (VAL); break;                                  \
        case 22u: a22 ^= (VAL); break;                                  \
        case 23u: a23 ^= (VAL); break;                                  \
        default:  a24 ^= (VAL); break;                                  \
    }                                                                   \
} while (false)

#define STORE_RATE_PREFIX2(BASE, OFFSET, LIMIT) do {                   \
    uint _lim = (LIMIT);                                                \
    uint _pos = (OFFSET);                                               \
    if (_lim > 0u)  out_data[(BASE) + _pos + 0u]  = u2_to_u64(a0);      \
    if (_lim > 1u)  out_data[(BASE) + _pos + 1u]  = u2_to_u64(a1);      \
    if (_lim > 2u)  out_data[(BASE) + _pos + 2u]  = u2_to_u64(a2);      \
    if (_lim > 3u)  out_data[(BASE) + _pos + 3u]  = u2_to_u64(a3);      \
    if (_lim > 4u)  out_data[(BASE) + _pos + 4u]  = u2_to_u64(a4);      \
    if (_lim > 5u)  out_data[(BASE) + _pos + 5u]  = u2_to_u64(a5);      \
    if (_lim > 6u)  out_data[(BASE) + _pos + 6u]  = u2_to_u64(a6);      \
    if (_lim > 7u)  out_data[(BASE) + _pos + 7u]  = u2_to_u64(a7);      \
    if (_lim > 8u)  out_data[(BASE) + _pos + 8u]  = u2_to_u64(a8);      \
    if (_lim > 9u)  out_data[(BASE) + _pos + 9u]  = u2_to_u64(a9);      \
    if (_lim > 10u) out_data[(BASE) + _pos + 10u] = u2_to_u64(a10);     \
    if (_lim > 11u) out_data[(BASE) + _pos + 11u] = u2_to_u64(a11);     \
    if (_lim > 12u) out_data[(BASE) + _pos + 12u] = u2_to_u64(a12);     \
    if (_lim > 13u) out_data[(BASE) + _pos + 13u] = u2_to_u64(a13);     \
    if (_lim > 14u) out_data[(BASE) + _pos + 14u] = u2_to_u64(a14);     \
    if (_lim > 15u) out_data[(BASE) + _pos + 15u] = u2_to_u64(a15);     \
    if (_lim > 16u) out_data[(BASE) + _pos + 16u] = u2_to_u64(a16);     \
    if (_lim > 17u) out_data[(BASE) + _pos + 17u] = u2_to_u64(a17);     \
    if (_lim > 18u) out_data[(BASE) + _pos + 18u] = u2_to_u64(a18);     \
    if (_lim > 19u) out_data[(BASE) + _pos + 19u] = u2_to_u64(a19);     \
    if (_lim > 20u) out_data[(BASE) + _pos + 20u] = u2_to_u64(a20);     \
    if (_lim > 21u) out_data[(BASE) + _pos + 21u] = u2_to_u64(a21);     \
    if (_lim > 22u) out_data[(BASE) + _pos + 22u] = u2_to_u64(a22);     \
    if (_lim > 23u) out_data[(BASE) + _pos + 23u] = u2_to_u64(a23);     \
    if (_lim > 24u) out_data[(BASE) + _pos + 24u] = u2_to_u64(a24);     \
} while (false)

#define STORE_21_2(BASE) do {                                          \
    out_data[(BASE) + 0u]  = u2_to_u64(a0);                             \
    out_data[(BASE) + 1u]  = u2_to_u64(a1);                             \
    out_data[(BASE) + 2u]  = u2_to_u64(a2);                             \
    out_data[(BASE) + 3u]  = u2_to_u64(a3);                             \
    out_data[(BASE) + 4u]  = u2_to_u64(a4);                             \
    out_data[(BASE) + 5u]  = u2_to_u64(a5);                             \
    out_data[(BASE) + 6u]  = u2_to_u64(a6);                             \
    out_data[(BASE) + 7u]  = u2_to_u64(a7);                             \
    out_data[(BASE) + 8u]  = u2_to_u64(a8);                             \
    out_data[(BASE) + 9u]  = u2_to_u64(a9);                             \
    out_data[(BASE) + 10u] = u2_to_u64(a10);                            \
    out_data[(BASE) + 11u] = u2_to_u64(a11);                            \
    out_data[(BASE) + 12u] = u2_to_u64(a12);                            \
    out_data[(BASE) + 13u] = u2_to_u64(a13);                            \
    out_data[(BASE) + 14u] = u2_to_u64(a14);                            \
    out_data[(BASE) + 15u] = u2_to_u64(a15);                            \
    out_data[(BASE) + 16u] = u2_to_u64(a16);                            \
    out_data[(BASE) + 17u] = u2_to_u64(a17);                            \
    out_data[(BASE) + 18u] = u2_to_u64(a18);                            \
    out_data[(BASE) + 19u] = u2_to_u64(a19);                            \
    out_data[(BASE) + 20u] = u2_to_u64(a20);                            \
} while (false)

#define STORE_11_AT_2(BASE, OFF) do {                                  \
    out_data[(BASE) + (OFF) + 0u]  = u2_to_u64(a0);                     \
    out_data[(BASE) + (OFF) + 1u]  = u2_to_u64(a1);                     \
    out_data[(BASE) + (OFF) + 2u]  = u2_to_u64(a2);                     \
    out_data[(BASE) + (OFF) + 3u]  = u2_to_u64(a3);                     \
    out_data[(BASE) + (OFF) + 4u]  = u2_to_u64(a4);                     \
    out_data[(BASE) + (OFF) + 5u]  = u2_to_u64(a5);                     \
    out_data[(BASE) + (OFF) + 6u]  = u2_to_u64(a6);                     \
    out_data[(BASE) + (OFF) + 7u]  = u2_to_u64(a7);                     \
    out_data[(BASE) + (OFF) + 8u]  = u2_to_u64(a8);                     \
    out_data[(BASE) + (OFF) + 9u]  = u2_to_u64(a9);                     \
    out_data[(BASE) + (OFF) + 10u] = u2_to_u64(a10);                    \
} while (false)

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

    const uint msg_lanes  = msg_bytes >> 3;
    const uint rate_lanes = rate_bytes >> 3;
    const uint out_lanes  = out_bytes >> 3;

    const uint2 z = uint2(0u, 0u);
    uint2 a0 = z,  a1 = z,  a2 = z,  a3 = z,  a4 = z;
    uint2 a5 = z,  a6 = z,  a7 = z,  a8 = z,  a9 = z;
    uint2 a10 = z, a11 = z, a12 = z, a13 = z, a14 = z;
    uint2 a15 = z, a16 = z, a17 = z, a18 = z, a19 = z;
    uint2 a20 = z, a21 = z, a22 = z, a23 = z, a24 = z;

    if (msg_lanes == 4u) {
        const uint in_base4 = idx << 2;
        a0 = u64_to_u2(in_data[in_base4 + 0u]);
        a1 = u64_to_u2(in_data[in_base4 + 1u]);
        a2 = u64_to_u2(in_data[in_base4 + 2u]);
        a3 = u64_to_u2(in_data[in_base4 + 3u]);
        a4 = uint2(domain & 0xFFu, 0u);
    } else {
        const uint in_base = idx * msg_lanes;

        if (msg_lanes > 0u)  a0  = u64_to_u2(in_data[in_base + 0u]);
        if (msg_lanes > 1u)  a1  = u64_to_u2(in_data[in_base + 1u]);
        if (msg_lanes > 2u)  a2  = u64_to_u2(in_data[in_base + 2u]);
        if (msg_lanes > 3u)  a3  = u64_to_u2(in_data[in_base + 3u]);
        if (msg_lanes > 4u)  a4  = u64_to_u2(in_data[in_base + 4u]);
        if (msg_lanes > 5u)  a5  = u64_to_u2(in_data[in_base + 5u]);
        if (msg_lanes > 6u)  a6  = u64_to_u2(in_data[in_base + 6u]);
        if (msg_lanes > 7u)  a7  = u64_to_u2(in_data[in_base + 7u]);
        if (msg_lanes > 8u)  a8  = u64_to_u2(in_data[in_base + 8u]);
        if (msg_lanes > 9u)  a9  = u64_to_u2(in_data[in_base + 9u]);
        if (msg_lanes > 10u) a10 = u64_to_u2(in_data[in_base + 10u]);
        if (msg_lanes > 11u) a11 = u64_to_u2(in_data[in_base + 11u]);
        if (msg_lanes > 12u) a12 = u64_to_u2(in_data[in_base + 12u]);
        if (msg_lanes > 13u) a13 = u64_to_u2(in_data[in_base + 13u]);
        if (msg_lanes > 14u) a14 = u64_to_u2(in_data[in_base + 14u]);
        if (msg_lanes > 15u) a15 = u64_to_u2(in_data[in_base + 15u]);
        if (msg_lanes > 16u) a16 = u64_to_u2(in_data[in_base + 16u]);
        if (msg_lanes > 17u) a17 = u64_to_u2(in_data[in_base + 17u]);
        if (msg_lanes > 18u) a18 = u64_to_u2(in_data[in_base + 18u]);
        if (msg_lanes > 19u) a19 = u64_to_u2(in_data[in_base + 19u]);
        if (msg_lanes > 20u) a20 = u64_to_u2(in_data[in_base + 20u]);
        if (msg_lanes > 21u) a21 = u64_to_u2(in_data[in_base + 21u]);
        if (msg_lanes > 22u) a22 = u64_to_u2(in_data[in_base + 22u]);
        if (msg_lanes > 23u) a23 = u64_to_u2(in_data[in_base + 23u]);

        XOR_TO_LANE2(msg_lanes, uint2(domain & 0xFFu, 0u));
    }

    const uint2 pad = uint2(0u, 0x80000000u);
    if (rate_lanes == 17u) {
        a16 ^= pad;
    } else if (rate_lanes == 21u) {
        a20 ^= pad;
    } else {
        XOR_TO_LANE2(rate_lanes - 1u, pad);
    }

    KECCAK_PERMUTE2();

    if (out_lanes == 4u && rate_lanes >= 4u) {
        const uint out_base4 = idx << 2;
        out_data[out_base4 + 0u] = u2_to_u64(a0);
        out_data[out_base4 + 1u] = u2_to_u64(a1);
        out_data[out_base4 + 2u] = u2_to_u64(a2);
        out_data[out_base4 + 3u] = u2_to_u64(a3);
        return;
    }

    if (out_lanes == 32u && rate_lanes == 21u) {
        const uint out_base32 = idx << 5;
        STORE_21_2(out_base32);
        KECCAK_PERMUTE2();
        STORE_11_AT_2(out_base32, 21u);
        return;
    }

    const uint out_base = idx * out_lanes;
    uint written = 0u;

    while (written < out_lanes) {
        const uint remaining = out_lanes - written;
        const uint take = remaining < rate_lanes ? remaining : rate_lanes;

        STORE_RATE_PREFIX2(out_base, written, take);

        written += take;
        if (written >= out_lanes) break;

        KECCAK_PERMUTE2();
    }
}

#undef STORE_11_AT_2
#undef STORE_21_2
#undef STORE_RATE_PREFIX2
#undef XOR_TO_LANE2
#undef KECCAK_PERMUTE2
#undef KECCAK_ROUND2
#undef ROL2H
#undef ROL2L
```