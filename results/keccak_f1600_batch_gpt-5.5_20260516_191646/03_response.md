Specializing the guaranteed 32-byte input removes the dynamic message-lane load chain and domain-lane switch from the hot SHA3/SHAKE paths, while still using runtime rate/output/domain. The permutation structure remains the incumbent’s low-register in-place Keccak round, avoiding the previous attempt’s duplicated permutation code that likely hurt instruction-cache behavior. 

```metal
#include <metal_stdlib>
using namespace metal;

#define ROL64(x, n) (((x) << (n)) | ((x) >> (64 - (n))))

#define KECCAK_ROUND(RCVAL) do {                                      \
    ulong c0 = a0 ^ a5 ^ a10 ^ a15 ^ a20;                             \
    ulong c1 = a1 ^ a6 ^ a11 ^ a16 ^ a21;                             \
    ulong c2 = a2 ^ a7 ^ a12 ^ a17 ^ a22;                             \
    ulong c3 = a3 ^ a8 ^ a13 ^ a18 ^ a23;                             \
    ulong c4 = a4 ^ a9 ^ a14 ^ a19 ^ a24;                             \
                                                                       \
    ulong d0 = c4 ^ ROL64(c1, 1);                                      \
    ulong d1 = c0 ^ ROL64(c2, 1);                                      \
    ulong d2 = c1 ^ ROL64(c3, 1);                                      \
    ulong d3 = c2 ^ ROL64(c4, 1);                                      \
    ulong d4 = c3 ^ ROL64(c0, 1);                                      \
                                                                       \
    a0 ^= d0;  a5 ^= d0;  a10 ^= d0;  a15 ^= d0;  a20 ^= d0;          \
    a1 ^= d1;  a6 ^= d1;  a11 ^= d1;  a16 ^= d1;  a21 ^= d1;          \
    a2 ^= d2;  a7 ^= d2;  a12 ^= d2;  a17 ^= d2;  a22 ^= d2;          \
    a3 ^= d3;  a8 ^= d3;  a13 ^= d3;  a18 ^= d3;  a23 ^= d3;          \
    a4 ^= d4;  a9 ^= d4;  a14 ^= d4;  a19 ^= d4;  a24 ^= d4;          \
                                                                       \
    ulong t = a1;                                                      \
    ulong u = a10; a10 = ROL64(t,  1); t = u;                          \
          u = a7;  a7  = ROL64(t,  3); t = u;                          \
          u = a11; a11 = ROL64(t,  6); t = u;                          \
          u = a17; a17 = ROL64(t, 10); t = u;                          \
          u = a18; a18 = ROL64(t, 15); t = u;                          \
          u = a3;  a3  = ROL64(t, 21); t = u;                          \
          u = a5;  a5  = ROL64(t, 28); t = u;                          \
          u = a16; a16 = ROL64(t, 36); t = u;                          \
          u = a8;  a8  = ROL64(t, 45); t = u;                          \
          u = a21; a21 = ROL64(t, 55); t = u;                          \
          u = a24; a24 = ROL64(t,  2); t = u;                          \
          u = a4;  a4  = ROL64(t, 14); t = u;                          \
          u = a15; a15 = ROL64(t, 27); t = u;                          \
          u = a23; a23 = ROL64(t, 41); t = u;                          \
          u = a19; a19 = ROL64(t, 56); t = u;                          \
          u = a13; a13 = ROL64(t,  8); t = u;                          \
          u = a12; a12 = ROL64(t, 25); t = u;                          \
          u = a2;  a2  = ROL64(t, 43); t = u;                          \
          u = a20; a20 = ROL64(t, 62); t = u;                          \
          u = a14; a14 = ROL64(t, 18); t = u;                          \
          u = a22; a22 = ROL64(t, 39); t = u;                          \
          u = a9;  a9  = ROL64(t, 61); t = u;                          \
          u = a6;  a6  = ROL64(t, 20); t = u;                          \
    a1 = ROL64(t, 44);                                                 \
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
    a0 ^= (ulong)(RCVAL);                                              \
} while (false)

#define KECCAK_PERMUTE() do {                                          \
    KECCAK_ROUND(0x0000000000000001ul);                                \
    KECCAK_ROUND(0x0000000000008082ul);                                \
    KECCAK_ROUND(0x800000000000808Aul);                                \
    KECCAK_ROUND(0x8000000080008000ul);                                \
    KECCAK_ROUND(0x000000000000808Bul);                                \
    KECCAK_ROUND(0x0000000080000001ul);                                \
    KECCAK_ROUND(0x8000000080008081ul);                                \
    KECCAK_ROUND(0x8000000000008009ul);                                \
    KECCAK_ROUND(0x000000000000008Aul);                                \
    KECCAK_ROUND(0x0000000000000088ul);                                \
    KECCAK_ROUND(0x0000000080008009ul);                                \
    KECCAK_ROUND(0x000000008000000Aul);                                \
    KECCAK_ROUND(0x000000008000808Bul);                                \
    KECCAK_ROUND(0x800000000000008Bul);                                \
    KECCAK_ROUND(0x8000000000008089ul);                                \
    KECCAK_ROUND(0x8000000000008003ul);                                \
    KECCAK_ROUND(0x8000000000008002ul);                                \
    KECCAK_ROUND(0x8000000000000080ul);                                \
    KECCAK_ROUND(0x000000000000800Aul);                                \
    KECCAK_ROUND(0x800000008000000Aul);                                \
    KECCAK_ROUND(0x8000000080008081ul);                                \
    KECCAK_ROUND(0x8000000000008080ul);                                \
    KECCAK_ROUND(0x0000000080000001ul);                                \
    KECCAK_ROUND(0x8000000080008008ul);                                \
} while (false)

#define XOR_TO_LANE(LANE, VAL) do {                                    \
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

#define STORE_RATE_PREFIX(BASE, OFFSET, LIMIT) do {                     \
    uint _lim = (LIMIT);                                                \
    uint _pos = (OFFSET);                                               \
    if (_lim > 0u)  out_data[(BASE) + _pos + 0u]  = a0;                 \
    if (_lim > 1u)  out_data[(BASE) + _pos + 1u]  = a1;                 \
    if (_lim > 2u)  out_data[(BASE) + _pos + 2u]  = a2;                 \
    if (_lim > 3u)  out_data[(BASE) + _pos + 3u]  = a3;                 \
    if (_lim > 4u)  out_data[(BASE) + _pos + 4u]  = a4;                 \
    if (_lim > 5u)  out_data[(BASE) + _pos + 5u]  = a5;                 \
    if (_lim > 6u)  out_data[(BASE) + _pos + 6u]  = a6;                 \
    if (_lim > 7u)  out_data[(BASE) + _pos + 7u]  = a7;                 \
    if (_lim > 8u)  out_data[(BASE) + _pos + 8u]  = a8;                 \
    if (_lim > 9u)  out_data[(BASE) + _pos + 9u]  = a9;                 \
    if (_lim > 10u) out_data[(BASE) + _pos + 10u] = a10;                \
    if (_lim > 11u) out_data[(BASE) + _pos + 11u] = a11;                \
    if (_lim > 12u) out_data[(BASE) + _pos + 12u] = a12;                \
    if (_lim > 13u) out_data[(BASE) + _pos + 13u] = a13;                \
    if (_lim > 14u) out_data[(BASE) + _pos + 14u] = a14;                \
    if (_lim > 15u) out_data[(BASE) + _pos + 15u] = a15;                \
    if (_lim > 16u) out_data[(BASE) + _pos + 16u] = a16;                \
    if (_lim > 17u) out_data[(BASE) + _pos + 17u] = a17;                \
    if (_lim > 18u) out_data[(BASE) + _pos + 18u] = a18;                \
    if (_lim > 19u) out_data[(BASE) + _pos + 19u] = a19;                \
    if (_lim > 20u) out_data[(BASE) + _pos + 20u] = a20;                \
    if (_lim > 21u) out_data[(BASE) + _pos + 21u] = a21;                \
    if (_lim > 22u) out_data[(BASE) + _pos + 22u] = a22;                \
    if (_lim > 23u) out_data[(BASE) + _pos + 23u] = a23;                \
    if (_lim > 24u) out_data[(BASE) + _pos + 24u] = a24;                \
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
    (void)msg_bytes;

    uint rate_lanes = rate_bytes >> 3;
    uint out_lanes  = out_bytes  >> 3;

    uint in_base = idx << 2;

    ulong a0 = in_data[in_base + 0u];
    ulong a1 = in_data[in_base + 1u];
    ulong a2 = in_data[in_base + 2u];
    ulong a3 = in_data[in_base + 3u];
    ulong a4 = (ulong)(domain & 0xFFu);

    ulong a5 = 0ul,  a6 = 0ul,  a7 = 0ul,  a8 = 0ul,  a9 = 0ul;
    ulong a10 = 0ul, a11 = 0ul, a12 = 0ul, a13 = 0ul, a14 = 0ul;
    ulong a15 = 0ul, a16 = 0ul, a17 = 0ul, a18 = 0ul, a19 = 0ul;
    ulong a20 = 0ul, a21 = 0ul, a22 = 0ul, a23 = 0ul, a24 = 0ul;

    const ulong pad = 0x8000000000000000ul;
    if (rate_lanes == 17u) {
        a16 ^= pad;
    } else if (rate_lanes == 21u) {
        a20 ^= pad;
    } else {
        XOR_TO_LANE(rate_lanes - 1u, pad);
    }

    KECCAK_PERMUTE();

    if (out_lanes == 4u && rate_lanes >= 4u) {
        uint out_base4 = idx << 2;
        out_data[out_base4 + 0u] = a0;
        out_data[out_base4 + 1u] = a1;
        out_data[out_base4 + 2u] = a2;
        out_data[out_base4 + 3u] = a3;
        return;
    }

    uint out_base;
    if (out_lanes == 32u) {
        out_base = idx << 5;
    } else {
        out_base = idx * out_lanes;
    }

    uint written = 0u;
    while (written < out_lanes) {
        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;

        STORE_RATE_PREFIX(out_base, written, take);

        written += take;
        if (written >= out_lanes) break;

        KECCAK_PERMUTE();
    }
}

#undef STORE_RATE_PREFIX
#undef XOR_TO_LANE
#undef KECCAK_PERMUTE
#undef KECCAK_ROUND
#undef ROL64
```