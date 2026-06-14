## Task: wots_chain

Batched WOTS+ / SPHINCS+-style hash chains. Given ``n_chains`` independent ``n_bytes``-byte seeds, apply the Keccak-256 inner hash ``w`` times in sequence per chain (each digest truncated to ``n_bytes`` bytes before feeding into the next iteration) and write the chain tip to the output. The chains are embarrassingly parallel; the ``w``-step iteration along each chain is strictly sequential.

Inner hash: Keccak-f[1600] with the FIPS 202 SHA3-256 sponge framing -- rate = 136 bytes (17 lanes), capacity = 64 bytes, domain pad byte = 0x06. State convention: the 1600-bit state is a 5x5 array of 64-bit lanes; lane k = x + 5*y holds bytes 8*k .. 8*k + 7 of the sponge state in little-endian.

All test sizes have ``n_bytes < rate_bytes`` (``n_bytes`` is bound at runtime and varies across the configurations the kernel is scored on; rate_bytes=136), so every chain step collapses to a single-block absorb + single-block squeeze of ``n_lanes = n_bytes / 8`` state lanes:
  state                          := 0
  state[lane 0..n_lanes-1]       := previous_chunk
  state[lane n_lanes, byte 0]    ^= 0x06   # SHA3 domain
  state[lane 16, byte 7]         ^= 0x80   # FIPS 202 final pad
  state                          := Keccak-f1600(state)
  next_chunk                     := state[lane 0..n_lanes-1]

On the first chain step the absorb is the seed; on every subsequent step the absorb is the n_lanes-lane truncation of the previous Keccak-f1600 output. After ``w`` steps the first n_lanes state lanes are written to the output as the chain tip.

The kernel must read ``n_bytes`` and ``w`` from the bound device buffers rather than treating them as compile-time constants; both vary across the test sizes (``w`` in {16, 64, 256} among the baseline sizes shown; both ``w`` and ``n_bytes`` are bound at runtime and vary across the configurations the kernel is scored on). Hardcoding either value silently produces wrong output, not just slow output.

Correctness is bit-exact against ``hashlib.sha3_256`` iterated ``w`` times with ``n_bytes``-byte truncation; any mismatched output ulong rejects the candidate.

## Required kernel signature(s)

```
kernel void wots_chain(
    device const ulong *seeds    [[buffer(0)]],
    device       ulong *tips     [[buffer(1)]],
    constant uint      &n_chains [[buffer(2)]],
    constant uint      &n_bytes  [[buffer(3)]],
    constant uint      &w        [[buffer(4)]],
    uint idx [[thread_position_in_grid]]);

Dispatch (host-fixed):
  threadsPerGrid        = (n_chains, 1, 1)
  threadsPerThreadgroup = (min(n_chains, 64), 1, 1)
Each thread processes ONE chain end-to-end; guard against idx >= n_chains (the grid is rounded up to a multiple of the TG width). ``seeds`` is laid out as n_chains consecutive runs of ``n_bytes / 8`` ulongs; ``tips`` likewise. The external buffer layout above must be preserved and the per-chain sequential semantics honored: each chain's step ``j+1`` must read the digest produced by its own step ``j`` (cross-chain mixing of intermediate digests would be a correctness bug).
```

## Your previous attempt

```metal
#include <metal_stdlib>
using namespace metal;

#define Z2  uint2(0u, 0u)
#define D2  uint2(0x00000006u, 0u)
#define F2  uint2(0u, 0x80000000u)
#define DF2 uint2(0x00000006u, 0x80000000u)

inline uint2 load64le(ulong x) {
    return uint2((uint)x, (uint)(x >> 32));
}

inline ulong store64le(uint2 v) {
    return ((ulong)v.y << 32) | (ulong)v.x;
}

#define ROL_LT(v,k) uint2((((v).x << (k)) | ((v).y >> (32u - (k)))), \
                          (((v).y << (k)) | ((v).x >> (32u - (k)))))

#define ROL_GT(v,k) uint2((((v).y << ((k) - 32u)) | ((v).x >> (64u - (k)))), \
                          (((v).x << ((k) - 32u)) | ((v).y >> (64u - (k)))))

#define KECCAK_ROUND(RCLO,RCHI) do { \
    uint2 C0 = A00 ^ A01 ^ A02 ^ A03 ^ A04; \
    uint2 C1 = A10 ^ A11 ^ A12 ^ A13 ^ A14; \
    uint2 C2 = A20 ^ A21 ^ A22 ^ A23 ^ A24; \
    uint2 C3 = A30 ^ A31 ^ A32 ^ A33 ^ A34; \
    uint2 C4 = A40 ^ A41 ^ A42 ^ A43 ^ A44; \
    uint2 D0v = C4 ^ ROL_LT(C1, 1u); \
    uint2 D1v = C0 ^ ROL_LT(C2, 1u); \
    uint2 D2v = C1 ^ ROL_LT(C3, 1u); \
    uint2 D3v = C2 ^ ROL_LT(C4, 1u); \
    uint2 D4v = C3 ^ ROL_LT(C0, 1u); \
    A00 ^= D0v; A01 ^= D0v; A02 ^= D0v; A03 ^= D0v; A04 ^= D0v; \
    A10 ^= D1v; A11 ^= D1v; A12 ^= D1v; A13 ^= D1v; A14 ^= D1v; \
    A20 ^= D2v; A21 ^= D2v; A22 ^= D2v; A23 ^= D2v; A24 ^= D2v; \
    A30 ^= D3v; A31 ^= D3v; A32 ^= D3v; A33 ^= D3v; A34 ^= D3v; \
    A40 ^= D4v; A41 ^= D4v; A42 ^= D4v; A43 ^= D4v; A44 ^= D4v; \
    uint2 T = A10; \
    A10 = ROL_GT(A11, 44u); \
    A11 = ROL_LT(A41, 20u); \
    A41 = ROL_GT(A24, 61u); \
    A24 = ROL_GT(A42, 39u); \
    A42 = ROL_LT(A04, 18u); \
    A04 = ROL_GT(A20, 62u); \
    A20 = ROL_GT(A22, 43u); \
    A22 = ROL_LT(A32, 25u); \
    A32 = ROL_LT(A43,  8u); \
    A43 = ROL_GT(A34, 56u); \
    A34 = ROL_GT(A03, 41u); \
    A03 = ROL_LT(A40, 27u); \
    A40 = ROL_LT(A44, 14u); \
    A44 = ROL_LT(A14,  2u); \
    A14 = ROL_GT(A31, 55u); \
    A31 = ROL_GT(A13, 45u); \
    A13 = ROL_GT(A01, 36u); \
    A01 = ROL_LT(A30, 28u); \
    A30 = ROL_LT(A33, 21u); \
    A33 = ROL_LT(A23, 15u); \
    A23 = ROL_LT(A12, 10u); \
    A12 = ROL_LT(A21,  6u); \
    A21 = ROL_LT(A02,  3u); \
    A02 = ROL_LT(T,    1u); \
    uint2 T0, T1, T2, T3, T4; \
    T0 = A00; T1 = A10; T2 = A20; T3 = A30; T4 = A40; \
    A00 = T0 ^ ((~T1) & T2) ^ uint2((RCLO), (RCHI)); \
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

#define KECCAK_ROUND_LAST_N4(RCLO,RCHI) do { \
    uint2 C0 = A00 ^ A01 ^ A02 ^ A03 ^ A04; \
    uint2 C1 = A10 ^ A11 ^ A12 ^ A13 ^ A14; \
    uint2 C2 = A20 ^ A21 ^ A22 ^ A23 ^ A24; \
    uint2 C3 = A30 ^ A31 ^ A32 ^ A33 ^ A34; \
    uint2 C4 = A40 ^ A41 ^ A42 ^ A43 ^ A44; \
    uint2 D0v = C4 ^ ROL_LT(C1, 1u); \
    uint2 D1v = C0 ^ ROL_LT(C2, 1u); \
    uint2 D2v = C1 ^ ROL_LT(C3, 1u); \
    uint2 D3v = C2 ^ ROL_LT(C4, 1u); \
    uint2 D4v = C3 ^ ROL_LT(C0, 1u); \
    uint2 B0 = A00 ^ D0v; \
    uint2 B1 = ROL_GT(A11 ^ D1v, 44u); \
    uint2 B2 = ROL_GT(A22 ^ D2v, 43u); \
    uint2 B3 = ROL_LT(A33 ^ D3v, 21u); \
    uint2 B4 = ROL_LT(A44 ^ D4v, 14u); \
    A00 = B0 ^ ((~B1) & B2) ^ uint2((RCLO), (RCHI)); \
    A10 = B1 ^ ((~B2) & B3); \
    A20 = B2 ^ ((~B3) & B4); \
    A30 = B3 ^ ((~B4) & B0); \
} while (0)

#define KECCAK_ROUND_LAST_N8(RCLO,RCHI) do { \
    uint2 C0 = A00 ^ A01 ^ A02 ^ A03 ^ A04; \
    uint2 C1 = A10 ^ A11 ^ A12 ^ A13 ^ A14; \
    uint2 C2 = A20 ^ A21 ^ A22 ^ A23 ^ A24; \
    uint2 C3 = A30 ^ A31 ^ A32 ^ A33 ^ A34; \
    uint2 C4 = A40 ^ A41 ^ A42 ^ A43 ^ A44; \
    uint2 D0v = C4 ^ ROL_LT(C1, 1u); \
    uint2 D1v = C0 ^ ROL_LT(C2, 1u); \
    uint2 D2v = C1 ^ ROL_LT(C3, 1u); \
    uint2 D3v = C2 ^ ROL_LT(C4, 1u); \
    uint2 D4v = C3 ^ ROL_LT(C0, 1u); \
    uint2 B0, B1, B2, B3, B4; \
    B0 = A00 ^ D0v; \
    B1 = ROL_GT(A11 ^ D1v, 44u); \
    B2 = ROL_GT(A22 ^ D2v, 43u); \
    B3 = ROL_LT(A33 ^ D3v, 21u); \
    B4 = ROL_LT(A44 ^ D4v, 14u); \
    uint2 O00 = B0 ^ ((~B1) & B2) ^ uint2((RCLO), (RCHI)); \
    uint2 O10 = B1 ^ ((~B2) & B3); \
    uint2 O20 = B2 ^ ((~B3) & B4); \
    uint2 O30 = B3 ^ ((~B4) & B0); \
    uint2 O40 = B4 ^ ((~B0) & B1); \
    B0 = ROL_LT(A30 ^ D3v, 28u); \
    B1 = ROL_LT(A41 ^ D4v, 20u); \
    B2 = ROL_LT(A02 ^ D0v,  3u); \
    B3 = ROL_GT(A13 ^ D1v, 45u); \
    B4 = ROL_GT(A24 ^ D2v, 61u); \
    uint2 O01 = B0 ^ ((~B1) & B2); \
    uint2 O11 = B1 ^ ((~B2) & B3); \
    uint2 O21 = B2 ^ ((~B3) & B4); \
    A00 = O00; A10 = O10; A20 = O20; A30 = O30; A40 = O40; \
    A01 = O01; A11 = O11; A21 = O21; \
} while (0)

#define KECCAK_F1600() do { \
    KECCAK_ROUND(0x00000001u, 0x00000000u); \
    KECCAK_ROUND(0x00008082u, 0x00000000u); \
    KECCAK_ROUND(0x0000808Au, 0x80000000u); \
    KECCAK_ROUND(0x80008000u, 0x80000000u); \
    KECCAK_ROUND(0x0000808Bu, 0x00000000u); \
    KECCAK_ROUND(0x80000001u, 0x00000000u); \
    KECCAK_ROUND(0x80008081u, 0x80000000u); \
    KECCAK_ROUND(0x00008009u, 0x80000000u); \
    KECCAK_ROUND(0x0000008Au, 0x00000000u); \
    KECCAK_ROUND(0x00000088u, 0x00000000u); \
    KECCAK_ROUND(0x80008009u, 0x00000000u); \
    KECCAK_ROUND(0x8000000Au, 0x00000000u); \
    KECCAK_ROUND(0x8000808Bu, 0x00000000u); \
    KECCAK_ROUND(0x0000008Bu, 0x80000000u); \
    KECCAK_ROUND(0x00008089u, 0x80000000u); \
    KECCAK_ROUND(0x00008003u, 0x80000000u); \
    KECCAK_ROUND(0x00008002u, 0x80000000u); \
    KECCAK_ROUND(0x00000080u, 0x80000000u); \
    KECCAK_ROUND(0x0000800Au, 0x00000000u); \
    KECCAK_ROUND(0x8000000Au, 0x80000000u); \
    KECCAK_ROUND(0x80008081u, 0x80000000u); \
    KECCAK_ROUND(0x00008080u, 0x80000000u); \
    KECCAK_ROUND(0x80000001u, 0x00000000u); \
    KECCAK_ROUND(0x80008008u, 0x80000000u); \
} while (0)

#define KECCAK_F1600_TRUNC4() do { \
    KECCAK_ROUND(0x00000001u, 0x00000000u); \
    KECCAK_ROUND(0x00008082u, 0x00000000u); \
    KECCAK_ROUND(0x0000808Au, 0x80000000u); \
    KECCAK_ROUND(0x80008000u, 0x80000000u); \
    KECCAK_ROUND(0x0000808Bu, 0x00000000u); \
    KECCAK_ROUND(0x80000001u, 0x00000000u); \
    KECCAK_ROUND(0x80008081u, 0x80000000u); \
    KECCAK_ROUND(0x00008009u, 0x80000000u); \
    KECCAK_ROUND(0x0000008Au, 0x00000000u); \
    KECCAK_ROUND(0x00000088u, 0x00000000u); \
    KECCAK_ROUND(0x80008009u, 0x00000000u); \
    KECCAK_ROUND(0x8000000Au, 0x00000000u); \
    KECCAK_ROUND(0x8000808Bu, 0x00000000u); \
    KECCAK_ROUND(0x0000008Bu, 0x80000000u); \
    KECCAK_ROUND(0x00008089u, 0x80000000u); \
    KECCAK_ROUND(0x00008003u, 0x80000000u); \
    KECCAK_ROUND(0x00008002u, 0x80000000u); \
    KECCAK_ROUND(0x00000080u, 0x80000000u); \
    KECCAK_ROUND(0x0000800Au, 0x00000000u); \
    KECCAK_ROUND(0x8000000Au, 0x80000000u); \
    KECCAK_ROUND(0x80008081u, 0x80000000u); \
    KECCAK_ROUND(0x00008080u, 0x80000000u); \
    KECCAK_ROUND(0x80000001u, 0x00000000u); \
    KECCAK_ROUND_LAST_N4(0x80008008u, 0x80000000u); \
} while (0)

#define KECCAK_F1600_TRUNC8() do { \
    KECCAK_ROUND(0x00000001u, 0x00000000u); \
    KECCAK_ROUND(0x00008082u, 0x00000000u); \
    KECCAK_ROUND(0x0000808Au, 0x80000000u); \
    KECCAK_ROUND(0x80008000u, 0x80000000u); \
    KECCAK_ROUND(0x0000808Bu, 0x00000000u); \
    KECCAK_ROUND(0x80000001u, 0x00000000u); \
    KECCAK_ROUND(0x80008081u, 0x80000000u); \
    KECCAK_ROUND(0x00008009u, 0x80000000u); \
    KECCAK_ROUND(0x0000008Au, 0x00000000u); \
    KECCAK_ROUND(0x00000088u, 0x00000000u); \
    KECCAK_ROUND(0x80008009u, 0x00000000u); \
    KECCAK_ROUND(0x8000000Au, 0x00000000u); \
    KECCAK_ROUND(0x8000808Bu, 0x00000000u); \
    KECCAK_ROUND(0x0000008Bu, 0x80000000u); \
    KECCAK_ROUND(0x00008089u, 0x80000000u); \
    KECCAK_ROUND(0x00008003u, 0x80000000u); \
    KECCAK_ROUND(0x00008002u, 0x80000000u); \
    KECCAK_ROUND(0x00000080u, 0x80000000u); \
    KECCAK_ROUND(0x0000800Au, 0x00000000u); \
    KECCAK_ROUND(0x8000000Au, 0x80000000u); \
    KECCAK_ROUND(0x80008081u, 0x80000000u); \
    KECCAK_ROUND(0x00008080u, 0x80000000u); \
    KECCAK_ROUND(0x80000001u, 0x00000000u); \
    KECCAK_ROUND_LAST_N8(0x80008008u, 0x80000000u); \
} while (0)

#define RESET_N1() do { \
    A10 = D2; A20 = Z2; A30 = Z2; A40 = Z2; \
    A01 = Z2; A11 = Z2; A21 = Z2; A31 = Z2; A41 = Z2; \
    A02 = Z2; A12 = Z2; A22 = Z2; A32 = Z2; A42 = Z2; \
    A03 = Z2; A13 = F2; A23 = Z2; A33 = Z2; A43 = Z2; \
    A04 = Z2; A14 = Z2; A24 = Z2; A34 = Z2; A44 = Z2; \
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
            KECCAK_F1600_TRUNC4();
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
            KECCAK_F1600_TRUNC4();
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
            KECCAK_F1600_TRUNC4();
        }

        tips[base + 0u] = store64le(A00);
        tips[base + 1u] = store64le(A10);
        tips[base + 2u] = store64le(A20);
        return;
    }

    if (n_lanes == 1u) {
        uint base = idx;

        uint2 A00 = load64le(seeds[base]);
        uint2 A10, A20, A30, A40, A01, A11, A21, A31, A41;
        uint2 A02, A12, A22, A32, A42;
        uint2 A03, A13, A23, A33, A43;
        uint2 A04, A14, A24, A34, A44;

        for (uint step = 0u; step < ww; ++step) {
            RESET_N1();
            KECCAK_F1600_TRUNC4();
        }

        tips[base] = store64le(A00);
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
            KECCAK_F1600_TRUNC8();
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
```

Result of previous attempt:
          w16_C64K: correct, 5.61 ms, 695.2 Gbitops/s (u64) (61.8% of 1125 Gops/s (u64 bitop, est))
          w64_C64K: correct, 21.98 ms, 709.9 Gbitops/s (u64) (63.1% of 1125 Gops/s (u64 bitop, est))
         w256_C64K: correct, 86.04 ms, 725.4 Gbitops/s (u64) (64.5% of 1125 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.6311

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

#define Z2  uint2(0u, 0u)
#define D2  uint2(0x00000006u, 0u)
#define F2  uint2(0u, 0x80000000u)
#define DF2 uint2(0x00000006u, 0x80000000u)

inline uint2 load64le(ulong x) {
    return uint2((uint)x, (uint)(x >> 32));
}

inline ulong store64le(uint2 v) {
    return ((ulong)v.y << 32) | (ulong)v.x;
}

#define ROL_LT(v,k) uint2((((v).x << (k)) | ((v).y >> (32u - (k)))), \
                          (((v).y << (k)) | ((v).x >> (32u - (k)))))

#define ROL_GT(v,k) uint2((((v).y << ((k) - 32u)) | ((v).x >> (64u - (k)))), \
                          (((v).x << ((k) - 32u)) | ((v).y >> (64u - (k)))))

#define KECCAK_ROUND(RCLO,RCHI) do { \
    uint2 C0 = A00 ^ A01 ^ A02 ^ A03 ^ A04; \
    uint2 C1 = A10 ^ A11 ^ A12 ^ A13 ^ A14; \
    uint2 C2 = A20 ^ A21 ^ A22 ^ A23 ^ A24; \
    uint2 C3 = A30 ^ A31 ^ A32 ^ A33 ^ A34; \
    uint2 C4 = A40 ^ A41 ^ A42 ^ A43 ^ A44; \
    uint2 D0v = C4 ^ ROL_LT(C1, 1u); \
    uint2 D1v = C0 ^ ROL_LT(C2, 1u); \
    uint2 D2v = C1 ^ ROL_LT(C3, 1u); \
    uint2 D3v = C2 ^ ROL_LT(C4, 1u); \
    uint2 D4v = C3 ^ ROL_LT(C0, 1u); \
    A00 ^= D0v; A01 ^= D0v; A02 ^= D0v; A03 ^= D0v; A04 ^= D0v; \
    A10 ^= D1v; A11 ^= D1v; A12 ^= D1v; A13 ^= D1v; A14 ^= D1v; \
    A20 ^= D2v; A21 ^= D2v; A22 ^= D2v; A23 ^= D2v; A24 ^= D2v; \
    A30 ^= D3v; A31 ^= D3v; A32 ^= D3v; A33 ^= D3v; A34 ^= D3v; \
    A40 ^= D4v; A41 ^= D4v; A42 ^= D4v; A43 ^= D4v; A44 ^= D4v; \
    uint2 T = A10; \
    A10 = ROL_GT(A11, 44u); \
    A11 = ROL_LT(A41, 20u); \
    A41 = ROL_GT(A24, 61u); \
    A24 = ROL_GT(A42, 39u); \
    A42 = ROL_LT(A04, 18u); \
    A04 = ROL_GT(A20, 62u); \
    A20 = ROL_GT(A22, 43u); \
    A22 = ROL_LT(A32, 25u); \
    A32 = ROL_LT(A43,  8u); \
    A43 = ROL_GT(A34, 56u); \
    A34 = ROL_GT(A03, 41u); \
    A03 = ROL_LT(A40, 27u); \
    A40 = ROL_LT(A44, 14u); \
    A44 = ROL_LT(A14,  2u); \
    A14 = ROL_GT(A31, 55u); \
    A31 = ROL_GT(A13, 45u); \
    A13 = ROL_GT(A01, 36u); \
    A01 = ROL_LT(A30, 28u); \
    A30 = ROL_LT(A33, 21u); \
    A33 = ROL_LT(A23, 15u); \
    A23 = ROL_LT(A12, 10u); \
    A12 = ROL_LT(A21,  6u); \
    A21 = ROL_LT(A02,  3u); \
    A02 = ROL_LT(T,    1u); \
    uint2 T0, T1, T2, T3, T4; \
    T0 = A00; T1 = A10; T2 = A20; T3 = A30; T4 = A40; \
    A00 = T0 ^ ((~T1) & T2) ^ uint2((RCLO), (RCHI)); \
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
    KECCAK_ROUND(0x00008082u, 0x00000000u); \
    KECCAK_ROUND(0x0000808Au, 0x80000000u); \
    KECCAK_ROUND(0x80008000u, 0x80000000u); \
    KECCAK_ROUND(0x0000808Bu, 0x00000000u); \
    KECCAK_ROUND(0x80000001u, 0x00000000u); \
    KECCAK_ROUND(0x80008081u, 0x80000000u); \
    KECCAK_ROUND(0x00008009u, 0x80000000u); \
    KECCAK_ROUND(0x0000008Au, 0x00000000u); \
    KECCAK_ROUND(0x00000088u, 0x00000000u); \
    KECCAK_ROUND(0x80008009u, 0x00000000u); \
    KECCAK_ROUND(0x8000000Au, 0x00000000u); \
    KECCAK_ROUND(0x8000808Bu, 0x00000000u); \
    KECCAK_ROUND(0x0000008Bu, 0x80000000u); \
    KECCAK_ROUND(0x00008089u, 0x80000000u); \
    KECCAK_ROUND(0x00008003u, 0x80000000u); \
    KECCAK_ROUND(0x00008002u, 0x80000000u); \
    KECCAK_ROUND(0x00000080u, 0x80000000u); \
    KECCAK_ROUND(0x0000800Au, 0x00000000u); \
    KECCAK_ROUND(0x8000000Au, 0x80000000u); \
    KECCAK_ROUND(0x80008081u, 0x80000000u); \
    KECCAK_ROUND(0x00008080u, 0x80000000u); \
    KECCAK_ROUND(0x80000001u, 0x00000000u); \
    KECCAK_ROUND(0x80008008u, 0x80000000u); \
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
```

Incumbent result:
          w16_C64K: correct, 5.62 ms, 694.2 Gbitops/s (u64) (61.7% of 1125 Gops/s (u64 bitop, est))
          w64_C64K: correct, 21.18 ms, 736.5 Gbitops/s (u64) (65.5% of 1125 Gops/s (u64 bitop, est))
         w256_C64K: correct, 84.43 ms, 739.2 Gbitops/s (u64) (65.7% of 1125 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.6427

## History

- iter  1: compile=OK | correct=True | score=0.6210732308823792
- iter  2: compile=OK | correct=True | score=0.6402770966318182
- iter  3: compile=OK | correct=True | score=0.6363022834872183
- iter  4: compile=OK | correct=True | score=0.6426646689906004
- iter  5: compile=OK | correct=True | score=0.6327562516367025
- iter  6: compile=OK | correct=True | score=0.6325826358360042
- iter  7: compile=OK | correct=True | score=0.6367353383986895
- iter  8: compile=OK | correct=True | score=0.6311390152826103

## Stagnation notice

Your last 3 correct attempts all scored within 15% of
the incumbent without surpassing it. You are circling a local
optimum. STOP making incremental edits to the previous kernel and
propose a STRUCTURALLY different approach.

A reworded version of the previous kernel will not break out of
this plateau.

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
