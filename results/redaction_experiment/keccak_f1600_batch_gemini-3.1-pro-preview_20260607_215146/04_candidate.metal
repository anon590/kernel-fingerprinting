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
        switch (msg_lanes) {
            case 0: Aba ^= dom; break;
            case 1: Abe ^= dom; break;
            case 2: Abi ^= dom; break;
            case 3: Abo ^= dom; break;
            case 4: Abu ^= dom; break;
            case 5: Aga ^= dom; break;
            case 6: Age ^= dom; break;
            case 7: Agi ^= dom; break;
            case 8: Ago ^= dom; break;
            case 9: Agu ^= dom; break;
            case 10: Aka ^= dom; break;
            case 11: Ake ^= dom; break;
            case 12: Aki ^= dom; break;
            case 13: Ako ^= dom; break;
            case 14: Aku ^= dom; break;
            case 15: Ama ^= dom; break;
            case 16: Ame ^= dom; break;
            case 17: Ami ^= dom; break;
            case 18: Amo ^= dom; break;
            case 19: Amu ^= dom; break;
            case 20: Asa ^= dom; break;
        }
    }

    ulong pad = 0x8000000000000000ul;
    uint r_idx = rate_lanes - 1u;
    switch (r_idx) {
        case 0: Aba ^= pad; break;
        case 1: Abe ^= pad; break;
        case 2: Abi ^= pad; break;
        case 3: Abo ^= pad; break;
        case 4: Abu ^= pad; break;
        case 5: Aga ^= pad; break;
        case 6: Age ^= pad; break;
        case 7: Agi ^= pad; break;
        case 8: Ago ^= pad; break;
        case 9: Agu ^= pad; break;
        case 10: Aka ^= pad; break;
        case 11: Ake ^= pad; break;
        case 12: Aki ^= pad; break;
        case 13: Ako ^= pad; break;
        case 14: Aku ^= pad; break;
        case 15: Ama ^= pad; break;
        case 16: Ame ^= pad; break;
        case 17: Ami ^= pad; break;
        case 18: Amo ^= pad; break;
        case 19: Amu ^= pad; break;
        case 20: Asa ^= pad; break;
    }

    uint written = 0u;

    for (;;) {
        #pragma unroll
        for (uint r = 0u; r < 24u; r++) {
            // Theta
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
            
            Aba ^= Da; Abe ^= De; Abi ^= Di; Abo ^= Do; Abu ^= Du;
            Aga ^= Da; Age ^= De; Agi ^= Di; Ago ^= Do; Agu ^= Du;
            Aka ^= Da; Ake ^= De; Aki ^= Di; Ako ^= Do; Aku ^= Du;
            Ama ^= Da; Ame ^= De; Ami ^= Di; Amo ^= Do; Amu ^= Du;
            Asa ^= Da; Ase ^= De; Asi ^= Di; Aso ^= Do; Asu ^= Du;

            // Rho and Pi (backward assignment chain)
            ulong T = Abe;
            Abe = ROL64(Age, 44u);
            Age = ROL64(Agu, 20u);
            Agu = ROL64(Asu, 14u);
            Asu = ROL64(Aso, 56u);
            Aso = ROL64(Amu, 8u);
            Amu = ROL64(Asa, 18u);
            Asa = ROL64(Abi, 62u);
            Abi = ROL64(Aki, 43u);
            Aki = ROL64(Ako, 25u);
            Ako = ROL64(Ama, 41u);
            Ama = ROL64(Abu, 27u);
            Abu = ROL64(Asi, 61u);
            Asi = ROL64(Aku, 39u);
            Aku = ROL64(Ase, 2u);
            Ase = ROL64(Ago, 55u);
            Ago = ROL64(Ame, 45u);
            Ame = ROL64(Aga, 36u);
            Aga = ROL64(Abo, 28u);
            Abo = ROL64(Amo, 21u);
            Amo = ROL64(Ami, 15u);
            Ami = ROL64(Ake, 10u);
            Ake = ROL64(Agi, 6u);
            Agi = ROL64(Aka, 3u);
            Aka = ROL64(T, 1u);

            // Chi and Iota
            BCa = Aba; BCe = Abe; BCi = Abi; BCo = Abo; BCu = Abu;
            Aba = BCa ^ ((~BCe) & BCi) ^ KECCAK_RC[r];
            Abe = BCe ^ ((~BCi) & BCo);
            Abi = BCi ^ ((~BCo) & BCu);
            Abo = BCo ^ ((~BCu) & BCa);
            Abu = BCu ^ ((~BCa) & BCe);

            BCa = Aga; BCe = Age; BCi = Agi; BCo = Ago; BCu = Agu;
            Aga = BCa ^ ((~BCe) & BCi);
            Age = BCe ^ ((~BCi) & BCo);
            Agi = BCi ^ ((~BCo) & BCu);
            Ago = BCo ^ ((~BCu) & BCa);
            Agu = BCu ^ ((~BCa) & BCe);

            BCa = Aka; BCe = Ake; BCi = Aki; BCo = Ako; BCu = Aku;
            Aka = BCa ^ ((~BCe) & BCi);
            Ake = BCe ^ ((~BCi) & BCo);
            Aki = BCi ^ ((~BCo) & BCu);
            Ako = BCo ^ ((~BCu) & BCa);
            Aku = BCu ^ ((~BCa) & BCe);

            BCa = Ama; BCe = Ame; BCi = Ami; BCo = Amo; BCu = Amu;
            Ama = BCa ^ ((~BCe) & BCi);
            Ame = BCe ^ ((~BCi) & BCo);
            Ami = BCi ^ ((~BCo) & BCu);
            Amo = BCo ^ ((~BCu) & BCa);
            Amu = BCu ^ ((~BCa) & BCe);

            BCa = Asa; BCe = Ase; BCi = Asi; BCo = Aso; BCu = Asu;
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