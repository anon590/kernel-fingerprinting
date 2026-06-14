## Task: keccak_f1600_batch

Batched Keccak-f[1600] sponge over fixed-length messages. Each of ``batch`` independent instances absorbs ``msg_bytes`` bytes of input, applies the standard FIPS 202 padding, runs the 24-round Keccak-f[1600] permutation, and squeezes ``out_bytes`` bytes of output. All test sizes satisfy ``msg_bytes < rate_bytes`` (single absorb block) and ``msg_bytes``, ``rate_bytes``, ``out_bytes`` are all multiples of 8, so the host packs message and output as ``ulong`` arrays.

State convention: the 1600-bit state is a 5x5 array of 64-bit lanes; lane k (for k in 0..25) corresponds to byte positions 8*k .. 8*k + 7 of the sponge state in little-endian, i.e. lane k holds bytes at the (x, y) cell with x = k % 5 and y = k / 5. The seed shows the standard round constants ``RC[24]`` and rho offsets ``r[x][y]`` from FIPS 202.

Permutation: 24 rounds of theta -> rho -> pi -> chi -> iota as defined in FIPS 202. Concretely, with A the (5,5) state of 64-bit lanes:
  theta:  C[x]      = A[x,0] ^ A[x,1] ^ A[x,2] ^ A[x,3] ^ A[x,4];
          D[x]      = C[x-1] ^ rotl(C[x+1], 1);
          A[x,y]   ^= D[x].
  rho:    A'[x,y]   = rotl(A[x,y], r[x][y]).
  pi:     A''[y, (2*x + 3*y) %% 5] = A'[x, y]
          (equivalently A''[x, y] = A'[(x + 3*y) %% 5, x]).
  chi:    A'''[x,y] = A''[x,y] ^ ((~A''[(x+1)%%5, y]) & A''[(x+2)%%5, y]).
  iota:   A''''[0,0] = A'''[0,0] ^ RC[round].

Sponge protocol (msg_bytes < rate_bytes, single absorb block):
  1. Initialise the state to zero.
  2. XOR ``msg_bytes / 8`` input lanes into state lanes      0 .. msg_bytes/8 - 1 (little-endian byte stream).
  3. XOR the domain byte (low 8 bits of ``domain``) into      byte position ``msg_bytes`` (lane ``msg_bytes/8``,      byte 0 of that lane).
  4. XOR 0x80 into byte position ``rate_bytes - 1``      (lane ``rate_bytes/8 - 1``, byte 7 of that lane).
  5. Apply Keccak-f[1600].
  6. Output the first ``rate_bytes / 8`` lanes of state.
  7. If more output is needed, apply Keccak-f[1600] again      and output the next ``rate_bytes / 8`` lanes; repeat      until ``out_bytes / 8`` lanes have been written. The      final chunk may be shorter than the rate.

In-distribution sizes use the SHA3-256 mode (rate=136, domain=0x06, out=32); the held-out size uses SHAKE128 (rate=168, domain=0x1F, out=256, requires multiple squeeze permutations). The kernel must use the runtime values of ``rate_bytes``, ``out_bytes`` and ``domain`` rather than compile-time constants. Correctness is bit-exact against ``hashlib.sha3_256`` / ``hashlib.shake_128``; any mismatched output ulong rejects the candidate.

## Required kernel signature(s)

```
kernel void keccak_f1600_batch(
    device const ulong *in_data    [[buffer(0)]],
    device       ulong *out_data   [[buffer(1)]],
    constant uint      &batch      [[buffer(2)]],
    constant uint      &msg_bytes  [[buffer(3)]],
    constant uint      &rate_bytes [[buffer(4)]],
    constant uint      &out_bytes  [[buffer(5)]],
    constant uint      &domain     [[buffer(6)]],
    uint idx [[thread_position_in_grid]]);

Dispatch (host-fixed):
  threadsPerGrid        = (batch, 1, 1)
  threadsPerThreadgroup = (min(batch, 64), 1, 1)
Each thread processes ONE instance end-to-end; guard against idx >= batch (the grid is rounded up to a multiple of the TG width). All test sizes have msg_bytes = 32. ``in_data`` is laid out as batch consecutive runs of ``msg_bytes / 8`` ulongs; ``out_data`` as batch consecutive runs of ``out_bytes / 8`` ulongs. Threadgroup-cooperative and simdgroup-cooperative implementations are valid so long as the external buffer layout above is preserved.
```

## Your previous attempt

```metal
#include <metal_stdlib>
using namespace metal;

inline ulong ROL64(ulong a, uint offset) {
    return (a << offset) | (a >> (64u - offset));
}

constexpr constant ulong KECCAK_RC[24] = {
    0x0000000000000001ul, 0x0000000000008082ul, 0x800000000000808Aul,
    0x8000000080008000ul, 0x000000000000808Bul, 0x0000000080000001ul,
    0x8000000080008081ul, 0x8000000000008009ul, 0x000000000000008Aul,
    0x0000000000000088ul, 0x0000000080008009ul, 0x000000008000000Aul,
    0x000000008000808Bul, 0x800000000000008Bul, 0x8000000000008089ul,
    0x8000000000008003ul, 0x8000000000008002ul, 0x8000000000000080ul,
    0x000000000000800Aul, 0x800000008000000Aul, 0x8000000080008081ul,
    0x8000000000008080ul, 0x0000000080000001ul, 0x8000000080008008ul,
};

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

    uint msg_lanes  = msg_bytes  >> 3;
    uint rate_lanes = rate_bytes >> 3;
    uint out_lanes  = out_bytes  >> 3;

    device const ulong *in_ptr = in_data + idx * msg_lanes;
    device       ulong *out_ptr = out_data + idx * out_lanes;
    
    ulong Aba = 0, Abe = 0, Abi = 0, Abo = 0, Abu = 0;
    ulong Aga = 0, Age = 0, Agi = 0, Ago = 0, Agu = 0;
    ulong Aka = 0, Ake = 0, Aki = 0, Ako = 0, Aku = 0;
    ulong Ama = 0, Ame = 0, Ami = 0, Amo = 0, Amu = 0;
    ulong Asa = 0, Ase = 0, Asi = 0, Aso = 0, Asu = 0;

    if (msg_bytes == 32) {
        Aba ^= in_ptr[0];
        Abe ^= in_ptr[1];
        Abi ^= in_ptr[2];
        Abo ^= in_ptr[3];
        Abu ^= (ulong)(domain & 0xFFu);
    } else {
        if (msg_lanes > 0) Aba ^= in_ptr[0];
        if (msg_lanes > 1) Abe ^= in_ptr[1];
        if (msg_lanes > 2) Abi ^= in_ptr[2];
        if (msg_lanes > 3) Abo ^= in_ptr[3];
        if (msg_lanes > 4) Abu ^= in_ptr[4];
        if (msg_lanes > 5) Aga ^= in_ptr[5];
        if (msg_lanes > 6) Age ^= in_ptr[6];
        if (msg_lanes > 7) Agi ^= in_ptr[7];
        if (msg_lanes > 8) Ago ^= in_ptr[8];
        if (msg_lanes > 9) Agu ^= in_ptr[9];
        if (msg_lanes > 10) Aka ^= in_ptr[10];
        if (msg_lanes > 11) Ake ^= in_ptr[11];
        if (msg_lanes > 12) Aki ^= in_ptr[12];
        if (msg_lanes > 13) Ako ^= in_ptr[13];
        if (msg_lanes > 14) Aku ^= in_ptr[14];
        if (msg_lanes > 15) Ama ^= in_ptr[15];
        if (msg_lanes > 16) Ame ^= in_ptr[16];
        if (msg_lanes > 17) Ami ^= in_ptr[17];
        if (msg_lanes > 18) Amo ^= in_ptr[18];
        if (msg_lanes > 19) Amu ^= in_ptr[19];
        if (msg_lanes > 20) Asa ^= in_ptr[20];

        ulong dom = (ulong)(domain & 0xFFu);
        if (msg_lanes == 0) Aba ^= dom;
        else if (msg_lanes == 1) Abe ^= dom;
        else if (msg_lanes == 2) Abi ^= dom;
        else if (msg_lanes == 3) Abo ^= dom;
        else if (msg_lanes == 4) Abu ^= dom;
        else if (msg_lanes == 5) Aga ^= dom;
        else if (msg_lanes == 6) Age ^= dom;
        else if (msg_lanes == 7) Agi ^= dom;
        else if (msg_lanes == 8) Ago ^= dom;
        else if (msg_lanes == 9) Agu ^= dom;
        else if (msg_lanes == 10) Aka ^= dom;
        else if (msg_lanes == 11) Ake ^= dom;
        else if (msg_lanes == 12) Aki ^= dom;
        else if (msg_lanes == 13) Ako ^= dom;
        else if (msg_lanes == 14) Aku ^= dom;
        else if (msg_lanes == 15) Ama ^= dom;
        else if (msg_lanes == 16) Ame ^= dom;
        else if (msg_lanes == 17) Ami ^= dom;
        else if (msg_lanes == 18) Amo ^= dom;
        else if (msg_lanes == 19) Amu ^= dom;
        else if (msg_lanes == 20) Asa ^= dom;
    }

    ulong pad = 0x8000000000000000ul;
    uint r_idx = rate_lanes - 1u;
    if (r_idx == 0) Aba ^= pad;
    else if (r_idx == 1) Abe ^= pad;
    else if (r_idx == 2) Abi ^= pad;
    else if (r_idx == 3) Abo ^= pad;
    else if (r_idx == 4) Abu ^= pad;
    else if (r_idx == 5) Aga ^= pad;
    else if (r_idx == 6) Age ^= pad;
    else if (r_idx == 7) Agi ^= pad;
    else if (r_idx == 8) Ago ^= pad;
    else if (r_idx == 9) Agu ^= pad;
    else if (r_idx == 10) Aka ^= pad;
    else if (r_idx == 11) Ake ^= pad;
    else if (r_idx == 12) Aki ^= pad;
    else if (r_idx == 13) Ako ^= pad;
    else if (r_idx == 14) Aku ^= pad;
    else if (r_idx == 15) Ama ^= pad;
    else if (r_idx == 16) Ame ^= pad;
    else if (r_idx == 17) Ami ^= pad;
    else if (r_idx == 18) Amo ^= pad;
    else if (r_idx == 19) Amu ^= pad;
    else if (r_idx == 20) Asa ^= pad;

    uint written = 0u;

    for (;;) {
        #pragma unroll
        for (uint r = 0u; r < 24u; r += 2u) {
            ulong BCa = Aba^Aga^Aka^Ama^Asa;
            ulong BCe = Abe^Age^Ake^Ame^Ase;
            ulong BCi = Abi^Agi^Aki^Ami^Asi;
            ulong BCo = Abo^Ago^Ako^Amo^Aso;
            ulong BCu = Abu^Agu^Aku^Amu^Asu;
            
            ulong Da = BCu^ROL64(BCe, 1u);
            ulong De = BCa^ROL64(BCi, 1u);
            ulong Di = BCe^ROL64(BCo, 1u);
            ulong Do = BCi^ROL64(BCu, 1u);
            ulong Du = BCo^ROL64(BCa, 1u);
            
            Aba ^= Da; BCa = Aba;
            Age ^= De; BCe = ROL64(Age, 44u);
            Aki ^= Di; BCi = ROL64(Aki, 43u);
            Amo ^= Do; BCo = ROL64(Amo, 21u);
            Asu ^= Du; BCu = ROL64(Asu, 14u);
            ulong Eba = BCa ^ ((~BCe) & BCi) ^ KECCAK_RC[r];
            ulong Ebe = BCe ^ ((~BCi) & BCo);
            ulong Ebi = BCi ^ ((~BCo) & BCu);
            ulong Ebo = BCo ^ ((~BCu) & BCa);
            ulong Ebu = BCu ^ ((~BCa) & BCe);

            Abo ^= Do; BCa = ROL64(Abo, 28u);
            Agu ^= Du; BCe = ROL64(Agu, 20u);
            Aka ^= Da; BCi = ROL64(Aka, 3u);
            Ame ^= De; BCo = ROL64(Ame, 45u);
            Asi ^= Di; BCu = ROL64(Asi, 61u);
            ulong Ega = BCa ^ ((~BCe) & BCi);
            ulong Ege = BCe ^ ((~BCi) & BCo);
            ulong Egi = BCi ^ ((~BCo) & BCu);
            ulong Ego = BCo ^ ((~BCu) & BCa);
            ulong Egu = BCu ^ ((~BCa) & BCe);

            Abe ^= De; BCa = ROL64(Abe, 1u);
            Agi ^= Di; BCe = ROL64(Agi, 6u);
            Ako ^= Do; BCi = ROL64(Ako, 25u);
            Amu ^= Du; BCo = ROL64(Amu, 8u);
            Asa ^= Da; BCu = ROL64(Asa, 18u);
            ulong Eka = BCa ^ ((~BCe) & BCi);
            ulong Eke = BCe ^ ((~BCi) & BCo);
            ulong Eki = BCi ^ ((~BCo) & BCu);
            ulong Eko = BCo ^ ((~BCu) & BCa);
            ulong Eku = BCu ^ ((~BCa) & BCe);

            Abu ^= Du; BCa = ROL64(Abu, 27u);
            Aga ^= Da; BCe = ROL64(Aga, 36u);
            Ake ^= De; BCi = ROL64(Ake, 10u);
            Ami ^= Di; BCo = ROL64(Ami, 15u);
            Aso ^= Do; BCu = ROL64(Aso, 56u);
            ulong Ema = BCa ^ ((~BCe) & BCi);
            ulong Eme = BCe ^ ((~BCi) & BCo);
            ulong Emi = BCi ^ ((~BCo) & BCu);
            ulong Emo = BCo ^ ((~BCu) & BCa);
            ulong Emu = BCu ^ ((~BCa) & BCe);

            Abi ^= Di; BCa = ROL64(Abi, 62u);
            Ago ^= Do; BCe = ROL64(Ago, 55u);
            Aku ^= Du; BCi = ROL64(Aku, 39u);
            Ama ^= Da; BCo = ROL64(Ama, 41u);
            Ase ^= De; BCu = ROL64(Ase, 2u);
            ulong Esa = BCa ^ ((~BCe) & BCi);
            ulong Ese = BCe ^ ((~BCi) & BCo);
            ulong Esi = BCi ^ ((~BCo) & BCu);
            ulong Eso = BCo ^ ((~BCu) & BCa);
            ulong Esu = BCu ^ ((~BCa) & BCe);

            BCa = Eba^Ega^Eka^Ema^Esa;
            BCe = Ebe^Ege^Eke^Eme^Ese;
            BCi = Ebi^Egi^Eki^Emi^Esi;
            BCo = Ebo^Ego^Eko^Emo^Eso;
            BCu = Ebu^Egu^Eku^Emu^Esu;

            Da = BCu^ROL64(BCe, 1u);
            De = BCa^ROL64(BCi, 1u);
            Di = BCe^ROL64(BCo, 1u);
            Do = BCi^ROL64(BCu, 1u);
            Du = BCo^ROL64(BCa, 1u);

            Eba ^= Da; BCa = Eba;
            Ege ^= De; BCe = ROL64(Ege, 44u);
            Eki ^= Di; BCi = ROL64(Eki, 43u);
            Emo ^= Do; BCo = ROL64(Emo, 21u);
            Esu ^= Du; BCu = ROL64(Esu, 14u);
            Aba = BCa ^ ((~BCe) & BCi) ^ KECCAK_RC[r+1u];
            Abe = BCe ^ ((~BCi) & BCo);
            Abi = BCi ^ ((~BCo) & BCu);
            Abo = BCo ^ ((~BCu) & BCa);
            Abu = BCu ^ ((~BCa) & BCe);

            Ebo ^= Do; BCa = ROL64(Ebo, 28u);
            Egu ^= Du; BCe = ROL64(Egu, 20u);
            Eka ^= Da; BCi = ROL64(Eka, 3u);
            Eme ^= De; BCo = ROL64(Eme, 45u);
            Esi ^= Di; BCu = ROL64(Esi, 61u);
            Aga = BCa ^ ((~BCe) & BCi);
            Age = BCe ^ ((~BCi) & BCo);
            Agi = BCi ^ ((~BCo) & BCu);
            Ago = BCo ^ ((~BCu) & BCa);
            Agu = BCu ^ ((~BCa) & BCe);

            Ebe ^= De; BCa = ROL64(Ebe, 1u);
            Egi ^= Di; BCe = ROL64(Egi, 6u);
            Eko ^= Do; BCi = ROL64(Eko, 25u);
            Emu ^= Du; BCo = ROL64(Emu, 8u);
            Esa ^= Da; BCu = ROL64(Esa, 18u);
            Aka = BCa ^ ((~BCe) & BCi);
            Ake = BCe ^ ((~BCi) & BCo);
            Aki = BCi ^ ((~BCo) & BCu);
            Ako = BCo ^ ((~BCu) & BCa);
            Aku = BCu ^ ((~BCa) & BCe);

            Ebu ^= Du; BCa = ROL64(Ebu, 27u);
            Ega ^= Da; BCe = ROL64(Ega, 36u);
            Eke ^= De; BCi = ROL64(Eke, 10u);
            Emi ^= Di; BCo = ROL64(Emi, 15u);
            Eso ^= Do; BCu = ROL64(Eso, 56u);
            Ama = BCa ^ ((~BCe) & BCi);
            Ame = BCe ^ ((~BCi) & BCo);
            Ami = BCi ^ ((~BCo) & BCu);
            Amo = BCo ^ ((~BCu) & BCa);
            Amu = BCu ^ ((~BCa) & BCe);

            Ebi ^= Di; BCa = ROL64(Ebi, 62u);
            Ego ^= Do; BCe = ROL64(Ego, 55u);
            Eku ^= Du; BCi = ROL64(Eku, 39u);
            Ema ^= Da; BCo = ROL64(Ema, 41u);
            Ese ^= De; BCu = ROL64(Ese, 2u);
            Asa = BCa ^ ((~BCe) & BCi);
            Ase = BCe ^ ((~BCi) & BCo);
            Asi = BCi ^ ((~BCo) & BCu);
            Aso = BCo ^ ((~BCu) & BCa);
            Asu = BCu ^ ((~BCa) & BCe);
        }

        if (out_bytes == 32) {
            out_ptr[0] = Aba; out_ptr[1] = Abe; out_ptr[2] = Abi; out_ptr[3] = Abo;
            break;
        } else if (out_bytes == 256) {
            if (written == 0) {
                out_ptr[0] = Aba; out_ptr[1] = Abe; out_ptr[2] = Abi; out_ptr[3] = Abo; out_ptr[4] = Abu;
                out_ptr[5] = Aga; out_ptr[6] = Age; out_ptr[7] = Agi; out_ptr[8] = Ago; out_ptr[9] = Agu;
                out_ptr[10] = Aka; out_ptr[11] = Ake; out_ptr[12] = Aki; out_ptr[13] = Ako; out_ptr[14] = Aku;
                out_ptr[15] = Ama; out_ptr[16] = Ame; out_ptr[17] = Ami; out_ptr[18] = Amo; out_ptr[19] = Amu;
                out_ptr[20] = Asa;
                written = 21u;
                continue;
            } else {
                out_ptr[21] = Aba; out_ptr[22] = Abe; out_ptr[23] = Abi; out_ptr[24] = Abo; out_ptr[25] = Abu;
                out_ptr[26] = Aga; out_ptr[27] = Age; out_ptr[28] = Agi; out_ptr[29] = Ago; out_ptr[30] = Agu;
                out_ptr[31] = Aka;
                break;
            }
        } else {
            uint remaining = out_lanes - written;
            uint take = remaining < rate_lanes ? remaining : rate_lanes;
            
            if (take > 0) out_ptr[written + 0] = Aba;
            if (take > 1) out_ptr[written + 1] = Abe;
            if (take > 2) out_ptr[written + 2] = Abi;
            if (take > 3) out_ptr[written + 3] = Abo;
            if (take > 4) out_ptr[written + 4] = Abu;
            if (take > 5) out_ptr[written + 5] = Aga;
            if (take > 6) out_ptr[written + 6] = Age;
            if (take > 7) out_ptr[written + 7] = Agi;
            if (take > 8) out_ptr[written + 8] = Ago;
            if (take > 9) out_ptr[written + 9] = Agu;
            if (take > 10) out_ptr[written + 10] = Aka;
            if (take > 11) out_ptr[written + 11] = Ake;
            if (take > 12) out_ptr[written + 12] = Aki;
            if (take > 13) out_ptr[written + 13] = Ako;
            if (take > 14) out_ptr[written + 14] = Aku;
            if (take > 15) out_ptr[written + 15] = Ama;
            if (take > 16) out_ptr[written + 16] = Ame;
            if (take > 17) out_ptr[written + 17] = Ami;
            if (take > 18) out_ptr[written + 18] = Amo;
            if (take > 19) out_ptr[written + 19] = Amu;
            if (take > 20) out_ptr[written + 20] = Asa;
            
            written += take;
            if (written >= out_lanes) break;
        }
    }
}
```

Result of previous attempt:
     sha3_256_B16K: correct, 0.20 ms, 302.8 Gbitops/s (u64) (26.9% of 1125 Gops/s (u64 bitop, est))
    sha3_256_B256K: correct, 1.91 ms, 510.2 Gbitops/s (u64) (45.4% of 1125 Gops/s (u64 bitop, est))
      sha3_256_B4M: correct, 22.80 ms, 684.5 Gbitops/s (u64) (60.8% of 1125 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.4203

## History

- iter  0: compile=OK | correct=True | score=0.039192038870019597
- iter  1: compile=OK | correct=True | score=0.07657671873731396
- iter  2: compile=OK | correct=True | score=0.35800152655971385
- iter  3: compile=OK | correct=True | score=0.4203400031979236

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
