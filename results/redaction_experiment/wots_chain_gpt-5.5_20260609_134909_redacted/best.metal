#include <metal_stdlib>
using namespace metal;

#define Z2  uint2(0u, 0u)

/* Bit-interleaved lane representation:
   x = even bits of the original 64-bit lane, y = odd bits. */
#define D2  uint2(0x00000002u, 0x00000001u)   /* original 0x06 */
#define F2  uint2(0x00000000u, 0x80000000u)   /* original 1 << 63 */
#define DF2 uint2(0x00000002u, 0x80000001u)

inline uint compact1by1(uint x) {
    x &= 0x55555555u;
    x = (x | (x >> 1)) & 0x33333333u;
    x = (x | (x >> 2)) & 0x0f0f0f0fu;
    x = (x | (x >> 4)) & 0x00ff00ffu;
    x = (x | (x >> 8)) & 0x0000ffffu;
    return x;
}

inline uint spread1by1(uint x) {
    x &= 0x0000ffffu;
    x = (x | (x << 8)) & 0x00ff00ffu;
    x = (x | (x << 4)) & 0x0f0f0f0fu;
    x = (x | (x << 2)) & 0x33333333u;
    x = (x | (x << 1)) & 0x55555555u;
    return x;
}

inline uint2 load64le(ulong v) {
    uint lo = (uint)v;
    uint hi = (uint)(v >> 32);
    uint e = compact1by1(lo) | (compact1by1(hi) << 16);
    uint o = compact1by1(lo >> 1) | (compact1by1(hi >> 1) << 16);
    return uint2(e, o);
}

inline ulong store64le(uint2 v) {
    uint lo = spread1by1(v.x) | (spread1by1(v.y) << 1);
    uint hi = spread1by1(v.x >> 16) | (spread1by1(v.y >> 16) << 1);
    return ((ulong)hi << 32) | (ulong)lo;
}

#define ROL32(x,k) (((x) << (k)) | ((x) >> (32u - (k))))

/* Rotate original 64-bit lane left by 1 in bit-interleaved form. */
#define RI1(v) uint2(ROL32((v).y, 1u), (v).x)

/* Rotate original 64-bit lane left by an even amount 2*s. */
#define RIE(v,s) uint2(ROL32((v).x, (s)), ROL32((v).y, (s)))

/* Rotate original 64-bit lane left by an odd amount 2*s+1, s >= 1 here. */
#define RIO(v,s) uint2(ROL32((v).y, ((s) + 1u)), ROL32((v).x, (s)))

#define KECCAK_ROUND(RCE,RCO) do { \
    uint2 C0 = A00 ^ A01 ^ A02 ^ A03 ^ A04; \
    uint2 C1 = A10 ^ A11 ^ A12 ^ A13 ^ A14; \
    uint2 C2 = A20 ^ A21 ^ A22 ^ A23 ^ A24; \
    uint2 C3 = A30 ^ A31 ^ A32 ^ A33 ^ A34; \
    uint2 C4 = A40 ^ A41 ^ A42 ^ A43 ^ A44; \
    uint2 D0v = C4 ^ RI1(C1); \
    uint2 D1v = C0 ^ RI1(C2); \
    uint2 D2v = C1 ^ RI1(C3); \
    uint2 D3v = C2 ^ RI1(C4); \
    uint2 D4v = C3 ^ RI1(C0); \
    A00 ^= D0v; A01 ^= D0v; A02 ^= D0v; A03 ^= D0v; A04 ^= D0v; \
    A10 ^= D1v; A11 ^= D1v; A12 ^= D1v; A13 ^= D1v; A14 ^= D1v; \
    A20 ^= D2v; A21 ^= D2v; A22 ^= D2v; A23 ^= D2v; A24 ^= D2v; \
    A30 ^= D3v; A31 ^= D3v; A32 ^= D3v; A33 ^= D3v; A34 ^= D3v; \
    A40 ^= D4v; A41 ^= D4v; A42 ^= D4v; A43 ^= D4v; A44 ^= D4v; \
    uint2 T = A10; \
    A10 = RIE(A11, 22u); \
    A11 = RIE(A41, 10u); \
    A41 = RIO(A24, 30u); \
    A24 = RIO(A42, 19u); \
    A42 = RIE(A04,  9u); \
    A04 = RIE(A20, 31u); \
    A20 = RIO(A22, 21u); \
    A22 = RIO(A32, 12u); \
    A32 = RIE(A43,  4u); \
    A43 = RIE(A34, 28u); \
    A34 = RIO(A03, 20u); \
    A03 = RIO(A40, 13u); \
    A40 = RIE(A44,  7u); \
    A44 = RIE(A14,  1u); \
    A14 = RIO(A31, 27u); \
    A31 = RIO(A13, 22u); \
    A13 = RIE(A01, 18u); \
    A01 = RIE(A30, 14u); \
    A30 = RIO(A33, 10u); \
    A33 = RIO(A23,  7u); \
    A23 = RIE(A12,  5u); \
    A12 = RIE(A21,  3u); \
    A21 = RIO(A02,  1u); \
    A02 = RI1(T); \
    uint2 T0, T1, T2, T3, T4; \
    T0 = A00; T1 = A10; T2 = A20; T3 = A30; T4 = A40; \
    A00 = T0 ^ ((~T1) & T2) ^ uint2((RCE), (RCO)); \
    A10 = T1 ^ ((~T2) & T3); \
    A20 = T2 ^ ((~T3) & T4); \
    A30 = T3 ^ ((~T4) & T0); \
    A40 = T4 ^ ((~T0) & T1); \
    T0 = A01; T1 = A11; T2 = A21; T3 = A31; T4 = A41; \
    A01 = T0 ^ ((~T1) & T2); \
    A11 = T1 ^ ((~T2) & T3); \
    A21 = T2 ^ ((~T3) & T4); \
    A31 = T3 ^ ((~T4) & T0); \
    A41 = T4 ^ ((~T0) & T1); \
    T0 = A02; T1 = A12; T2 = A22; T3 = A32; T4 = A42; \
    A02 = T0 ^ ((~T1) & T2); \
    A12 = T1 ^ ((~T2) & T3); \
    A22 = T2 ^ ((~T3) & T4); \
    A32 = T3 ^ ((~T4) & T0); \
    A42 = T4 ^ ((~T0) & T1); \
    T0 = A03; T1 = A13; T2 = A23; T3 = A33; T4 = A43; \
    A03 = T0 ^ ((~T1) & T2); \
    A13 = T1 ^ ((~T2) & T3); \
    A23 = T2 ^ ((~T3) & T4); \
    A33 = T3 ^ ((~T4) & T0); \
    A43 = T4 ^ ((~T0) & T1); \
    T0 = A04; T1 = A14; T2 = A24; T3 = A34; T4 = A44; \
    A04 = T0 ^ ((~T1) & T2); \
    A14 = T1 ^ ((~T2) & T3); \
    A24 = T2 ^ ((~T3) & T4); \
    A34 = T3 ^ ((~T4) & T0); \
    A44 = T4 ^ ((~T0) & T1); \
} while (0)

#define KECCAK_F1600() do { \
    KECCAK_ROUND(0x00000001u, 0x00000000u); \
    KECCAK_ROUND(0x00000000u, 0x00000089u); \
    KECCAK_ROUND(0x00000000u, 0x8000008Bu); \
    KECCAK_ROUND(0x00000000u, 0x80008080u); \
    KECCAK_ROUND(0x00000001u, 0x0000008Bu); \
    KECCAK_ROUND(0x00000001u, 0x00008000u); \
    KECCAK_ROUND(0x00000001u, 0x80008088u); \
    KECCAK_ROUND(0x00000001u, 0x80000082u); \
    KECCAK_ROUND(0x00000000u, 0x0000000Bu); \
    KECCAK_ROUND(0x00000000u, 0x0000000Au); \
    KECCAK_ROUND(0x00000001u, 0x00008082u); \
    KECCAK_ROUND(0x00000000u, 0x00008003u); \
    KECCAK_ROUND(0x00000001u, 0x0000808Bu); \
    KECCAK_ROUND(0x00000001u, 0x8000000Bu); \
    KECCAK_ROUND(0x00000001u, 0x8000008Au); \
    KECCAK_ROUND(0x00000001u, 0x80000081u); \
    KECCAK_ROUND(0x00000000u, 0x80000081u); \
    KECCAK_ROUND(0x00000000u, 0x80000008u); \
    KECCAK_ROUND(0x00000000u, 0x00000083u); \
    KECCAK_ROUND(0x00000000u, 0x80008003u); \
    KECCAK_ROUND(0x00000001u, 0x80008088u); \
    KECCAK_ROUND(0x00000000u, 0x80000088u); \
    KECCAK_ROUND(0x00000001u, 0x00008000u); \
    KECCAK_ROUND(0x00000000u, 0x80008082u); \
} while (0)

#define RESET_N2() do { \
    A20 = D2; A30 = Z2; A40 = Z2; \
    A01 = Z2; A11 = Z2; A21 = Z2; A31 = Z2; A41 = Z2; \
    A02 = Z2; A12 = Z2; A22 = Z2; A32 = Z2; A42 = Z2; \
    A03 = Z2; A13 = F2; A23 = Z2; A33 = Z2; A43 = Z2; \
    A04 = Z2; A14 = Z2; A24 = Z2; A34 = Z2; A44 = Z2; \
} while (0)

#define RESET_N3() do { \
    A30 = D2; A40 = Z2; \
    A01 = Z2; A11 = Z2; A21 = Z2; A31 = Z2; A41 = Z2; \
    A02 = Z2; A12 = Z2; A22 = Z2; A32 = Z2; A42 = Z2; \
    A03 = Z2; A13 = F2; A23 = Z2; A33 = Z2; A43 = Z2; \
    A04 = Z2; A14 = Z2; A24 = Z2; A34 = Z2; A44 = Z2; \
} while (0)

#define RESET_N4() do { \
    A40 = D2; \
    A01 = Z2; A11 = Z2; A21 = Z2; A31 = Z2; A41 = Z2; \
    A02 = Z2; A12 = Z2; A22 = Z2; A32 = Z2; A42 = Z2; \
    A03 = Z2; A13 = F2; A23 = Z2; A33 = Z2; A43 = Z2; \
    A04 = Z2; A14 = Z2; A24 = Z2; A34 = Z2; A44 = Z2; \
} while (0)

#define RESET_N8() do { \
    A31 = D2; A41 = Z2; \
    A02 = Z2; A12 = Z2; A22 = Z2; A32 = Z2; A42 = Z2; \
    A03 = Z2; A13 = F2; A23 = Z2; A33 = Z2; A43 = Z2; \
    A04 = Z2; A14 = Z2; A24 = Z2; A34 = Z2; A44 = Z2; \
} while (0)

#define RESET_N16() do { \
    A13 = DF2; A23 = Z2; A33 = Z2; A43 = Z2; \
    A04 = Z2; A14 = Z2; A24 = Z2; A34 = Z2; A44 = Z2; \
} while (0)

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
    const uint ww = w;
    if (n_lanes == 0u) return;

    if (n_lanes == 4u) {
        uint base = idx << 2;

        uint2 A00 = load64le(seeds[base + 0u]);
        uint2 A10 = load64le(seeds[base + 1u]);
        uint2 A20 = load64le(seeds[base + 2u]);
        uint2 A30 = load64le(seeds[base + 3u]);
        uint2 A40, A01, A11, A21, A31, A41;
        uint2 A02, A12, A22, A32, A42;
        uint2 A03, A13, A23, A33, A43;
        uint2 A04, A14, A24, A34, A44;

        for (uint step = 0u; step < ww; ++step) {
            RESET_N4();
            KECCAK_F1600();
        }

        tips[base + 0u] = store64le(A00);
        tips[base + 1u] = store64le(A10);
        tips[base + 2u] = store64le(A20);
        tips[base + 3u] = store64le(A30);
        return;
    }

    if (n_lanes == 2u) {
        uint base = idx << 1;

        uint2 A00 = load64le(seeds[base + 0u]);
        uint2 A10 = load64le(seeds[base + 1u]);
        uint2 A20, A30, A40, A01, A11, A21, A31, A41;
        uint2 A02, A12, A22, A32, A42;
        uint2 A03, A13, A23, A33, A43;
        uint2 A04, A14, A24, A34, A44;

        for (uint step = 0u; step < ww; ++step) {
            RESET_N2();
            KECCAK_F1600();
        }

        tips[base + 0u] = store64le(A00);
        tips[base + 1u] = store64le(A10);
        return;
    }

    if (n_lanes == 3u) {
        uint base = idx * 3u;

        uint2 A00 = load64le(seeds[base + 0u]);
        uint2 A10 = load64le(seeds[base + 1u]);
        uint2 A20 = load64le(seeds[base + 2u]);
        uint2 A30, A40, A01, A11, A21, A31, A41;
        uint2 A02, A12, A22, A32, A42;
        uint2 A03, A13, A23, A33, A43;
        uint2 A04, A14, A24, A34, A44;

        for (uint step = 0u; step < ww; ++step) {
            RESET_N3();
            KECCAK_F1600();
        }

        tips[base + 0u] = store64le(A00);
        tips[base + 1u] = store64le(A10);
        tips[base + 2u] = store64le(A20);
        return;
    }

    if (n_lanes == 8u) {
        uint base = idx << 3;

        uint2 A00 = load64le(seeds[base + 0u]);
        uint2 A10 = load64le(seeds[base + 1u]);
        uint2 A20 = load64le(seeds[base + 2u]);
        uint2 A30 = load64le(seeds[base + 3u]);
        uint2 A40 = load64le(seeds[base + 4u]);
        uint2 A01 = load64le(seeds[base + 5u]);
        uint2 A11 = load64le(seeds[base + 6u]);
        uint2 A21 = load64le(seeds[base + 7u]);
        uint2 A31, A41;
        uint2 A02, A12, A22, A32, A42;
        uint2 A03, A13, A23, A33, A43;
        uint2 A04, A14, A24, A34, A44;

        for (uint step = 0u; step < ww; ++step) {
            RESET_N8();
            KECCAK_F1600();
        }

        tips[base + 0u] = store64le(A00);
        tips[base + 1u] = store64le(A10);
        tips[base + 2u] = store64le(A20);
        tips[base + 3u] = store64le(A30);
        tips[base + 4u] = store64le(A40);
        tips[base + 5u] = store64le(A01);
        tips[base + 6u] = store64le(A11);
        tips[base + 7u] = store64le(A21);
        return;
    }

    if (n_lanes == 16u) {
        uint base = idx << 4;

        uint2 A00 = load64le(seeds[base +  0u]);
        uint2 A10 = load64le(seeds[base +  1u]);
        uint2 A20 = load64le(seeds[base +  2u]);
        uint2 A30 = load64le(seeds[base +  3u]);
        uint2 A40 = load64le(seeds[base +  4u]);
        uint2 A01 = load64le(seeds[base +  5u]);
        uint2 A11 = load64le(seeds[base +  6u]);
        uint2 A21 = load64le(seeds[base +  7u]);
        uint2 A31 = load64le(seeds[base +  8u]);
        uint2 A41 = load64le(seeds[base +  9u]);
        uint2 A02 = load64le(seeds[base + 10u]);
        uint2 A12 = load64le(seeds[base + 11u]);
        uint2 A22 = load64le(seeds[base + 12u]);
        uint2 A32 = load64le(seeds[base + 13u]);
        uint2 A42 = load64le(seeds[base + 14u]);
        uint2 A03 = load64le(seeds[base + 15u]);
        uint2 A13, A23, A33, A43;
        uint2 A04, A14, A24, A34, A44;

        for (uint step = 0u; step < ww; ++step) {
            RESET_N16();
            KECCAK_F1600();
        }

        tips[base +  0u] = store64le(A00);
        tips[base +  1u] = store64le(A10);
        tips[base +  2u] = store64le(A20);
        tips[base +  3u] = store64le(A30);
        tips[base +  4u] = store64le(A40);
        tips[base +  5u] = store64le(A01);
        tips[base +  6u] = store64le(A11);
        tips[base +  7u] = store64le(A21);
        tips[base +  8u] = store64le(A31);
        tips[base +  9u] = store64le(A41);
        tips[base + 10u] = store64le(A02);
        tips[base + 11u] = store64le(A12);
        tips[base + 12u] = store64le(A22);
        tips[base + 13u] = store64le(A32);
        tips[base + 14u] = store64le(A42);
        tips[base + 15u] = store64le(A03);
        return;
    }

    uint base = idx * n_lanes;

    uint2 A00 = Z2, A10 = Z2, A20 = Z2, A30 = Z2, A40 = Z2;
    uint2 A01 = Z2, A11 = Z2, A21 = Z2, A31 = Z2, A41 = Z2;
    uint2 A02 = Z2, A12 = Z2, A22 = Z2, A32 = Z2, A42 = Z2;
    uint2 A03 = Z2, A13 = Z2, A23 = Z2, A33 = Z2, A43 = Z2;
    uint2 A04 = Z2, A14 = Z2, A24 = Z2, A34 = Z2, A44 = Z2;

    if (n_lanes >  0u) A00 = load64le(seeds[base +  0u]);
    if (n_lanes >  1u) A10 = load64le(seeds[base +  1u]);
    if (n_lanes >  2u) A20 = load64le(seeds[base +  2u]);
    if (n_lanes >  3u) A30 = load64le(seeds[base +  3u]);
    if (n_lanes >  4u) A40 = load64le(seeds[base +  4u]);
    if (n_lanes >  5u) A01 = load64le(seeds[base +  5u]);
    if (n_lanes >  6u) A11 = load64le(seeds[base +  6u]);
    if (n_lanes >  7u) A21 = load64le(seeds[base +  7u]);
    if (n_lanes >  8u) A31 = load64le(seeds[base +  8u]);
    if (n_lanes >  9u) A41 = load64le(seeds[base +  9u]);
    if (n_lanes > 10u) A02 = load64le(seeds[base + 10u]);
    if (n_lanes > 11u) A12 = load64le(seeds[base + 11u]);
    if (n_lanes > 12u) A22 = load64le(seeds[base + 12u]);
    if (n_lanes > 13u) A32 = load64le(seeds[base + 13u]);
    if (n_lanes > 14u) A42 = load64le(seeds[base + 14u]);
    if (n_lanes > 15u) A03 = load64le(seeds[base + 15u]);

    for (uint step = 0u; step < ww; ++step) {
        if (n_lanes <=  1u) A10 = Z2;
        if (n_lanes <=  2u) A20 = Z2;
        if (n_lanes <=  3u) A30 = Z2;
        if (n_lanes <=  4u) A40 = Z2;
        if (n_lanes <=  5u) A01 = Z2;
        if (n_lanes <=  6u) A11 = Z2;
        if (n_lanes <=  7u) A21 = Z2;
        if (n_lanes <=  8u) A31 = Z2;
        if (n_lanes <=  9u) A41 = Z2;
        if (n_lanes <= 10u) A02 = Z2;
        if (n_lanes <= 11u) A12 = Z2;
        if (n_lanes <= 12u) A22 = Z2;
        if (n_lanes <= 13u) A32 = Z2;
        if (n_lanes <= 14u) A42 = Z2;
        if (n_lanes <= 15u) A03 = Z2;

        A13 = Z2; A23 = Z2; A33 = Z2; A43 = Z2;
        A04 = Z2; A14 = Z2; A24 = Z2; A34 = Z2; A44 = Z2;

        switch (n_lanes) {
            case 1u:  A10 = D2; break;
            case 5u:  A01 = D2; break;
            case 6u:  A11 = D2; break;
            case 7u:  A21 = D2; break;
            case 9u:  A41 = D2; break;
            case 10u: A02 = D2; break;
            case 11u: A12 = D2; break;
            case 12u: A22 = D2; break;
            case 13u: A32 = D2; break;
            case 14u: A42 = D2; break;
            case 15u: A03 = D2; break;
            default:  A00 = D2; break;
        }

        A13 ^= F2;

        KECCAK_F1600();
    }

    if (n_lanes >  0u) tips[base +  0u] = store64le(A00);
    if (n_lanes >  1u) tips[base +  1u] = store64le(A10);
    if (n_lanes >  2u) tips[base +  2u] = store64le(A20);
    if (n_lanes >  3u) tips[base +  3u] = store64le(A30);
    if (n_lanes >  4u) tips[base +  4u] = store64le(A40);
    if (n_lanes >  5u) tips[base +  5u] = store64le(A01);
    if (n_lanes >  6u) tips[base +  6u] = store64le(A11);
    if (n_lanes >  7u) tips[base +  7u] = store64le(A21);
    if (n_lanes >  8u) tips[base +  8u] = store64le(A31);
    if (n_lanes >  9u) tips[base +  9u] = store64le(A41);
    if (n_lanes > 10u) tips[base + 10u] = store64le(A02);
    if (n_lanes > 11u) tips[base + 11u] = store64le(A12);
    if (n_lanes > 12u) tips[base + 12u] = store64le(A22);
    if (n_lanes > 13u) tips[base + 13u] = store64le(A32);
    if (n_lanes > 14u) tips[base + 14u] = store64le(A42);
    if (n_lanes > 15u) tips[base + 15u] = store64le(A03);
}