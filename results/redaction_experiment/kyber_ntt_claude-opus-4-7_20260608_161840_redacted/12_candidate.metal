#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_MAX = 256u;
constexpr constant uint Z_MAX = 256u;

inline uint mod_add(uint a, uint b, uint q) {
    uint t = a + b;
    return (t >= q) ? (t - q) : t;
}
inline uint mod_sub(uint a, uint b, uint q) {
    return (a >= b) ? (a - b) : (a + q - b);
}

inline uint compute_qinv(uint q) {
    uint x = q;
    x = x * (2u - q * x);
    x = x * (2u - q * x);
    x = x * (2u - q * x);
    x = x * (2u - q * x);
    x = x * (2u - q * x);
    return x;
}

inline uint mont_reduce(ulong T, uint q, uint qinv_neg) {
    uint T_lo = (uint)T;
    uint T_hi = (uint)(T >> 32);
    uint m = T_lo * qinv_neg;
    uint mq_hi = mulhi(m, q);
    uint mq_lo = m * q;
    uint sum_lo = T_lo + mq_lo;
    uint carry = (sum_lo < T_lo) ? 1u : 0u;
    uint t = T_hi + mq_hi + carry;
    if (t >= q) t -= q;
    return t;
}

kernel void kyber_ntt(
    device       uint *coeffs     [[buffer(0)]],
    device const uint *zetas      [[buffer(1)]],
    constant uint     &q           [[buffer(2)]],
    constant uint     &n           [[buffer(3)]],
    constant uint     &n_levels    [[buffer(4)]],
    constant uint     &batch       [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ltid [[thread_position_in_threadgroup]])
{
    if (tgid >= batch) return;

    threadgroup uint a[N_MAX];
    threadgroup uint zeta_mont[Z_MAX];

    uint qreg = q;
    uint nreg = n;
    uint nlv  = n_levels;
    uint half_n = nreg >> 1u;
    uint zcount = 1u << nlv;

    uint qinv = compute_qinv(qreg);
    uint qinv_neg = 0u - qinv;

    // R = 2^32 mod q (q odd, q < 2^31).
    uint Rmod;
    {
        ulong r = ((ulong)0xFFFFFFFFul) % (ulong)qreg;
        r += 1ul;
        if (r >= (ulong)qreg) r -= (ulong)qreg;
        Rmod = (uint)r;
    }

    // Cooperatively precompute Montgomery-form zetas.
    {
        ulong Q = (ulong)qreg;
        uint i1 = ltid;
        if (i1 < zcount) {
            zeta_mont[i1] = (uint)(((ulong)zetas[i1] * (ulong)Rmod) % Q);
        }
        uint i2 = ltid + half_n;
        if (i2 < zcount) {
            zeta_mont[i2] = (uint)(((ulong)zetas[i2] * (ulong)Rmod) % Q);
        }
    }

    device uint *poly = coeffs + (size_t)tgid * nreg;

    // Load into threadgroup memory.
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length  = half_n;
    uint k_start = 1u;

    // -------- Phase A: threadgroup-memory levels while length > 16 --------
    while (length > 16u) {
        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zm         = zeta_mont[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];
        uint t = mont_reduce((ulong)y * (ulong)zm, qreg, qinv_neg);
        uint xpt = mod_add(x, t, qreg);
        uint xmt = mod_sub(x, t, qreg);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        a[j]          = xpt;
        a[j + length] = xmt;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length  >>= 1u;
    }

    // -------- Phase B: simd-shuffle levels (length <= 16) --------
    // At this point each thread will own two coefficients living in registers.
    // We choose a layout that puts each butterfly-pair partner into the SAME
    // simdgroup of 32 lanes.
    //
    // After Phase A, length == 16 (assuming n=256, nlv=8; for smaller n we
    // just skip the simd phase). Currently a[0..n-1] is in threadgroup memory.
    //
    // Layout for register phase:
    //   thread ltid (0..half_n-1) owns coefficients
    //     reg_lo = a[ 2*ltid     ]   (even-indexed)
    //     reg_hi = a[ 2*ltid + 1 ]   (odd-indexed)
    // This means each simdgroup (32 lanes) covers indices [64*g .. 64*g+63].
    // A butterfly at length L pairs index j with j+L. For L <= 16, both
    // partners lie within the same 64-index block, so within the same simd.
    //
    // But Phase A leaves us with length still > 16 only if nlv hasn't finished.
    // If nlv finished inside Phase A, we just skip Phase B.
    bool do_simd_phase = (length > 0u) && (length <= 16u);

    if (do_simd_phase) {
        // Re-load coefficients in the register-friendly layout.
        // Use a barrier to make sure all phase-A writes are visible (already done).
        uint idx_lo = 2u * ltid;
        uint idx_hi = idx_lo + 1u;
        uint r_lo = a[idx_lo];
        uint r_hi = a[idx_hi];

        // Process remaining levels with simd_shuffle_xor.
        // At each level with stride 'length':
        //   For each lane, its butterfly partner index differs by 'length'.
        //   In our packed representation, two coeffs per lane:
        //     coeff_index 2*ltid   (r_lo)
        //     coeff_index 2*ltid+1 (r_hi)
        //   Partner of 2*ltid is 2*ltid + length, which lives in lane
        //     ltid_partner = ltid + length/2  if length is even (>=2)
        //     ... or the same lane's r_hi     if length == 1.
        //
        // length values in {16, 8, 4, 2, 1}.

        while (length > 0u) {
            uint group_idx  = ltid / length;
            uint zm         = zeta_mont[k_start + group_idx];

            if (length >= 2u) {
                // Partner lane for r_lo (coeff 2*ltid) lives in lane
                //   p = ltid XOR (length/2)
                // because 2*ltid XOR length = 2*(ltid XOR length/2).
                uint half_len = length >> 1u;
                uint partner = ltid ^ half_len;

                // Determine whether this lane holds the "low" (x) or "high" (y)
                // side of the butterfly. The butterfly j-th index is in [start,
                // start+length); start+length..start+2*length-1 is the y side.
                // group_idx = ltid / length; j_in_group = ltid - group_idx*length
                // is in [0, length). j = 2*group_idx*length + j_in_group means
                // each lane lies in the "low half" of its butterfly block in
                // coeff-index space. But we packed (r_lo, r_hi) = (2*ltid,
                // 2*ltid+1), so for length>=2 both r_lo and r_hi are on the
                // same side (both low or both high) iff length >= 2.
                //
                // Actually for length=2: indices 2*ltid and 2*ltid+1 differ by
                // 1 < length, so same side. For length>=2 same holds.
                //
                // Whether the lane is on low or high side depends on
                //   bit (log2 length) of (2*ltid) = bit (log2 length - 1) of ltid? No.
                //   coeff_index 2*ltid has bit log2(length) set iff lane is on high side.
                // Equivalent: ((ltid >> (log2(length)-1)) & 1) == 1 for length>=2.
                bool is_high = ((ltid & half_len) != 0u);

                // Exchange both r_lo and r_hi with partner.
                uint p_lo = simd_shuffle_xor(r_lo, half_len);
                uint p_hi = simd_shuffle_xor(r_hi, half_len);

                uint x_lo, y_lo, x_hi, y_hi;
                if (is_high) {
                    x_lo = p_lo; y_lo = r_lo;
                    x_hi = p_hi; y_hi = r_hi;
                } else {
                    x_lo = r_lo; y_lo = p_lo;
                    x_hi = r_hi; y_hi = p_hi;
                }

                uint t_lo = mont_reduce((ulong)y_lo * (ulong)zm, qreg, qinv_neg);
                uint t_hi = mont_reduce((ulong)y_hi * (ulong)zm, qreg, qinv_neg);

                uint nx_lo = mod_add(x_lo, t_lo, qreg);
                uint ny_lo = mod_sub(x_lo, t_lo, qreg);
                uint nx_hi = mod_add(x_hi, t_hi, qreg);
                uint ny_hi = mod_sub(x_hi, t_hi, qreg);

                if (is_high) {
                    r_lo = ny_lo;
                    r_hi = ny_hi;
                } else {
                    r_lo = nx_lo;
                    r_hi = nx_hi;
                }
            } else {
                // length == 1: butterfly between (2*ltid) and (2*ltid+1),
                // i.e. between r_lo and r_hi in the same lane.
                uint x = r_lo;
                uint y = r_hi;
                uint t = mont_reduce((ulong)y * (ulong)zm, qreg, qinv_neg);
                r_lo = mod_add(x, t, qreg);
                r_hi = mod_sub(x, t, qreg);
            }

            k_start <<= 1u;
            if (length == 1u) break;
            length >>= 1u;
        }

        // Write back in same layout.
        poly[idx_lo] = r_lo;
        poly[idx_hi] = r_hi;
    } else {
        poly[ltid]          = a[ltid];
        poly[ltid + half_n] = a[ltid + half_n];
    }
}