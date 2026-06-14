"""FRI folding round + Merkle commit -- Z5 task.

One FRI folding round on a polynomial committed via evaluations over a
Goldilocks coset, followed by a binary Poseidon2-t=3 Merkle commit of
the folded evaluations. Mirrors what a real STARK prover (plonky2 /
risc0 / winterfell) emits between consecutive FRI rounds.

Sizes (PLAN.md Section "Regime taxonomy"):
  * in-distribution: fold = 2, N in {2^16, 2^18, 2^20}
  * held-out:        fold = 4, N = 2^17

The held-out probe flips on candidates that:
  - hardcode fold = 2 in the gather / inner-sum loops;
  - bake the (j, j + N/2) pair stride into the kernel;
  - hardcode 1/2 instead of reading ``inv_fold`` from a buffer;
  - hardcode zeta = -1 (true only for fold = 2) and skip the
    ``zeta_inv_pow`` table;
  - assume the folded-evaluations length is N / 2.

The Merkle commit phase stays binary in both regimes (the held-out
probe varies only the FRI folding factor, not the commitment arity).

Roofline
========

Per-output modmul count in ``fri_fold`` (closed-form fold with an
inner geometric series):
  - 1 mul for ax = alpha * inv_x_base[j]
  - per m in [0, fold):
      1 mul for r_m = ax * zeta_inv_pow[m]
      fold - 1 muls for the power chain r_m, r_m^2, ..., r_m^{fold-1}
      1 mul for evals_in[src] * S_m
  - 1 mul for the final inv_fold scaling
  Total ~ 1 + fold * (fold + 1) + 1 = fold^2 + fold + 2 modmuls.
  At fold = 2: 8 muls/output -> 4 N muls/round.
  At fold = 4: 22 muls/output -> 5.5 N muls/round.

Per Poseidon2 t=3 commit perm (matches the merkle_build counting model):
  46 sboxes * 4 = 184 muls + 81 ext-mvm + 66 int-mvm = ~331 muls.
Total commit muls ~ 331 * (n_out - 1) ~ 331 * n_out for large n_out
(geometric sum of level node counts at arity = 2 is n_out - 1).

Per-round byte traffic (DRAM read + write, single-pass model):
  fri_fold:
    read  evals_in        : 8 * N
    read  inv_x_base      : 8 * n_out
    read  zeta_inv_pow    : 8 * fold    (negligible)
    write evals_out       : 8 * n_out
  Total                   : 8 * (N + 2 n_out) bytes
  commit:
    read  level lvl       : 8 * counts[lvl]   for lvl in [0, L-1)
    write level lvl+1     : 8 * counts[lvl+1]
  Total (geometric)       : ~ 8 * 2 * n_out bytes
  Round total             : 8 * (N + 4 n_out) bytes
  At fold = 2: 24 N bytes (16 N fold + 8 N commit-equivalent).
  At fold = 4: 16 N bytes.

The task reports the achieved fraction against the **binding**
roofline ``max(achieved_mul / peak_int64_mul, achieved_bw / peak_bw)``,
matching the goldilocks_ntt / merkle_build convention. At realistic
sizes the round is compute-bound (the Poseidon2 commit dominates the
modmul count); the BW number is surfaced in ``extra`` for sanity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import Metal

from ..reference.fri import (
    COSET_G, fri_round_cached, fri_round_constants,
)
from ..reference.goldilocks import P, random_field_elements
from ..reference.merkle import level_counts, level_offsets
from ..reference.poseidon2 import Poseidon2Goldilocks
from ..task import (
    Task, TaskSize, TaskSpec, SizeResult,
    gb_per_s, gops_per_s, register_task,
)


_SEED = Path(__file__).resolve().parent.parent.parent / "seeds" / "fri_round.metal"

# Modmul counting model for the Poseidon2-t=3 commit step (mirrors
# metal_zk/tasks/merkle_build.py::_modmuls_per_perm at t=3, r_f=8,
# r_p=22, alpha = 7 -> x^7 = 4 modmuls per sbox).
_POSEIDON2_T3_MULS_PER_PERM = (
    (8 * 3 + 22 * 1) * 4          # sboxes
    + (1 + 8) * (3 * 3)           # external matvec
    + 22 * 3                      # internal matvec
)


def _alpha_for_size(N: int, fold: int) -> int:
    """Deterministic transcript-derived FRI challenge for this (N, fold).

    A real prover derives alpha from a hash of the transcript; for the
    benchmark we just want a stable, size-keyed Goldilocks element that
    is unlikely to collide with a degenerate point of the fold (e.g.
    alpha = 0 makes E'[j] = E[j] independent of E[j + m n_out] for m>0,
    silently masking a bucket of held-out failures). Picking a large
    random-ish element via SplitMix-style mixing keeps both
    in-distribution and held-out probes honest.
    """
    z = (N * 0x9E3779B97F4A7C15 + fold * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    z ^= (z >> 33)
    z = (z * 0xFF51AFD7ED558CCD) & ((1 << 64) - 1)
    z ^= (z >> 33)
    z = (z * 0xC4CEB9FE1A85EC53) & ((1 << 64) - 1)
    z ^= (z >> 33)
    return int(z % P)


@register_task("fri_round")
class FriRoundTask(Task):
    spec = TaskSpec(
        name="fri_round",
        description=(
            "One FRI folding round on a polynomial committed via "
            "evaluations over a Goldilocks coset, followed by a "
            "binary Poseidon2-t=3 Merkle commit of the folded "
            "evaluations. The folding factor and N are bound at "
            "runtime through constant buffers; the kernel must use "
            "those runtime values rather than compile-time "
            "constants.\n\n"
            "Algebra of one round (closed-form FRI fold over a coset "
            "domain):\n"
            "  D  = { coset_g * omega_N^i  : i in [0, N)         }\n"
            "  D' = { coset_g^fold * omega_N^(j*fold) : j in [0, n_out) }\n"
            "  E'[j] = inv_fold * sum_{m=0..fold-1} S_m(j) * "
            "E[j + m * n_out]\n"
            "  r_m(j) = alpha / (coset_g * omega_N^{j + m * n_out})\n"
            "  S_m(j) = sum_{p=0..fold-1} r_m(j)^p\n"
            "with omega_N the primitive N-th root of unity in "
            "Goldilocks (p = 2^64 - 2^32 + 1, derived from the "
            "plonky2 / risc0 generator g_root_2^32 = "
            "1753635133440165772) and coset_g = 7. n_out = N / fold.\n\n"
            "Host-side precomputation (uploaded to device buffers):\n"
            "  inv_x_base[j]   = 1 / (coset_g * omega_N^j)   "
            "(length n_out)\n"
            "  zeta_inv_pow[m] = zeta^{-m}, zeta = omega_N^n_out  "
            "(length fold)\n"
            "  alpha           = round challenge (canonical [0, p))\n"
            "  inv_fold        = pow(fold, -1, p)\n"
            "so that r_m(j) = alpha * inv_x_base[j] * "
            "zeta_inv_pow[m].\n\n"
            "Two-kernel pipeline (host issues both in one compute "
            "command encoder; the serial encoder gives "
            "read-after-write ordering across dispatches with no "
            "explicit barrier):\n"
            "  1) fri_fold              : one dispatch over n_out "
            "threads; the host binds the tree buffer to fri_fold's "
            "evals_out slot, so the fold writes land in the level-0 "
            "(leaves) slice of the Merkle tree.\n"
            "  2) fri_commit_level (xL) : one dispatch per Merkle "
            "level over n_out folded leaves; produces the binary "
            "Poseidon2-t=3 Merkle root. The Merkle commit is binary "
            "Poseidon2-t=3 across every test size.\n\n"
            "Tree layout: a single contiguous ulong buffer holds "
            "ALL Merkle levels concatenated -- folded leaves first, "
            "then each parent level in order, finally the 1-element "
            "root. Total length = sum of binary level_counts(n_out). "
            "Per-level scalars (in_offset, out_offset, child_count) "
            "are bound at dispatch time via per-level uint offsets "
            "into a small constants buffer (mirrors the merkle_build "
            "task's host pattern).\n\n"
            "Correctness is bit-exact against the Python bigint "
            "reference:\n"
            "  * folded leaves slice (length n_out) must match the "
            "reference folded evaluations element-wise;\n"
            "  * the full tree (all levels, including every "
            "intermediate digest, not just the root) must match the "
            "reference Merkle commitment.\n"
            "Outputs MUST be canonical ([0, p)); a non-canonical "
            "value with the same residue class still counts as a "
            "mismatch. All test sizes satisfy fold <= 4 and t = 3; "
            "thread-private scratch arrays of size 4 are sufficient."
        ),
        kernel_signatures=(
            "kernel void fri_fold(\n"
            "    device const ulong *evals_in     [[buffer(0)]],\n"
            "    device       ulong *evals_out    [[buffer(1)]],\n"
            "    device const ulong *inv_x_base   [[buffer(2)]],\n"
            "    device const ulong *zeta_inv_pow [[buffer(3)]],\n"
            "    constant ulong     &alpha        [[buffer(4)]],\n"
            "    constant ulong     &inv_fold     [[buffer(5)]],\n"
            "    constant uint      &fold         [[buffer(6)]],\n"
            "    constant uint      &n_out        [[buffer(7)]],\n"
            "    uint j [[thread_position_in_grid]]);\n"
            "\n"
            "kernel void fri_commit_level(\n"
            "    device       ulong *tree         [[buffer(0)]],\n"
            "    device const ulong *rc_ext       [[buffer(1)]],\n"
            "    device const ulong *rc_int       [[buffer(2)]],\n"
            "    device const ulong *ext_mds      [[buffer(3)]],\n"
            "    device const ulong *int_diag     [[buffer(4)]],\n"
            "    constant uint      &in_offset    [[buffer(5)]],\n"
            "    constant uint      &out_offset   [[buffer(6)]],\n"
            "    constant uint      &child_count  [[buffer(7)]],\n"
            "    uint p [[thread_position_in_grid]]);\n"
            "\n"
            "Dispatch geometry (host-fixed):\n"
            "  fri_fold:\n"
            "    threadsPerGrid        = (n_out rounded up to TG width, 1, 1)\n"
            "    threadsPerThreadgroup = (min(n_out, 256), 1, 1)\n"
            "  fri_commit_level (one call per parent level):\n"
            "    threadsPerGrid        = (parent_count rounded up, 1, 1)\n"
            "    threadsPerThreadgroup = (min(parent_count, 64), 1, 1)\n"
            "\n"
            "fri_fold: each thread owns one output index j; guard "
            "against j >= n_out (the grid is rounded up to a "
            "multiple of the TG width). The same tree buffer is "
            "bound to evals_out (offset 0); the commit kernel reads "
            "from that buffer on the next dispatch.\n"
            "fri_commit_level: each thread owns one parent node at "
            "the current level; guard against p >= parent_count = "
            "ceil(child_count / 2). The host pre-binds rc_ext / "
            "rc_int / ext_mds / int_diag once and rebinds only the "
            "three per-level uint scalars per dispatch."
        ),
        kernel_names=["fri_fold", "fri_commit_level"],
        seed_path=_SEED,
        sizes=[
            TaskSize("f2_N64K", {"fold": 2, "n": 1 << 16}),
            TaskSize("f2_N256K", {"fold": 2, "n": 1 << 18}),
            TaskSize("f2_N1M",   {"fold": 2, "n": 1 << 20}),
        ],
        held_out_sizes=[
            TaskSize("f4_N128K", {"fold": 4, "n": 1 << 17}),
        ],
    )

    def evaluate_size(self, harness, pipelines, size, chip, n_warmup, n_measure):
        fold = int(size.params["fold"])
        N = int(size.params["n"])
        if N & (N - 1) != 0:
            raise ValueError(f"fri_round size N={N} must be a power of two")
        if fold not in (2, 4):
            raise ValueError(f"fri_round fold={fold} must be 2 or 4")
        n_out = N // fold

        cfg = Poseidon2Goldilocks(t=3)
        alpha = _alpha_for_size(N, fold)

        # Inputs + reference. Seed is salted with (N, fold) so the
        # held-out probe gets its own input set.
        evals = random_field_elements(
            N, seed=0xFA17_C0DE + N * 17 + fold * 1_000_003,
        )
        folded_ref, tree_ref = fri_round_cached(
            evals, fold=fold, alpha=alpha, coset_g=COSET_G, cfg=cfg,
        )

        # Per-level metadata (binary commit over the folded leaves).
        counts = level_counts(n_out, 2)             # ends with 1 (root)
        offsets = level_offsets(counts)             # len = len(counts) + 1
        n_hash_levels = len(counts) - 1
        total_nodes = offsets[-1]
        assert total_nodes == tree_ref.shape[0]

        # Host-side precomputation for fri_fold.
        consts = fri_round_constants(N, fold, alpha, coset_g=COSET_G)

        # Tree buffer: holds folded leaves + all parent levels. The
        # fri_fold dispatch writes the leaves slice; subsequent
        # fri_commit_level dispatches write the parent levels.
        b_tree = harness.buf_zeros(total_nodes * 8)

        # Input + precomputed constants for fri_fold.
        b_evals_in     = harness.buf_from_np(np.ascontiguousarray(evals,
                                                                 dtype=np.uint64))
        b_inv_x_base   = harness.buf_from_np(consts.inv_x_base)
        b_zeta_inv_pow = harness.buf_from_np(consts.zeta_inv_pow)
        b_alpha        = harness.buf_scalar(consts.alpha,    np.uint64)
        b_inv_fold     = harness.buf_scalar(consts.inv_fold, np.uint64)
        b_fold         = harness.buf_scalar(fold,            np.uint32)
        b_n_out        = harness.buf_scalar(n_out,           np.uint32)

        # Poseidon2 constants for fri_commit_level.
        b_rc_ext   = harness.buf_from_np(cfg.round_constants_external.ravel())
        b_rc_int   = harness.buf_from_np(cfg.round_constants_internal)
        b_ext_mds  = harness.buf_from_np(cfg.ext_mds.ravel())
        b_int_diag = harness.buf_from_np(cfg.int_diag)

        # Per-level scalar arrays bound via per-dispatch offset (same
        # pattern as goldilocks_ntt / merkle_build).
        in_offs_arr      = np.array(offsets[:n_hash_levels],      dtype=np.uint32)
        out_offs_arr     = np.array(offsets[1:n_hash_levels + 1], dtype=np.uint32)
        child_counts_arr = np.array(counts[:n_hash_levels],       dtype=np.uint32)
        b_in_offs      = harness.buf_from_np(in_offs_arr)
        b_out_offs     = harness.buf_from_np(out_offs_arr)
        b_child_counts = harness.buf_from_np(child_counts_arr)

        pso_fold   = pipelines["fri_fold"]
        pso_commit = pipelines["fri_commit_level"]

        # fri_fold geometry: one thread per output index.
        max_tg_f = int(pso_fold.maxTotalThreadsPerThreadgroup())
        tew_f    = int(pso_fold.threadExecutionWidth())
        tg_f     = min(max_tg_f, max(tew_f, 256))
        tg_f     = max(1, min(tg_f, n_out))
        grid_f   = ((n_out + tg_f - 1) // tg_f) * tg_f

        # fri_commit_level geometry: one thread per parent.
        max_tg_c = int(pso_commit.maxTotalThreadsPerThreadgroup())
        tew_c    = int(pso_commit.threadExecutionWidth())
        tg_c     = min(max_tg_c, max(tew_c, 64))

        view_tree = harness.np_view(b_tree, np.uint64, total_nodes)

        def reset():
            # Every dispatch fully overwrites the leaves slice (via
            # fri_fold) and each parent slice (via fri_commit_level);
            # we still wipe the whole tree so a buggy candidate that
            # leaves stale parents from a prior run doesn't get
            # credited for them on the correctness check.
            view_tree[:] = 0

        def dispatch(enc):
            # ---- Fold ----
            enc.setComputePipelineState_(pso_fold)
            enc.setBuffer_offset_atIndex_(b_evals_in,     0, 0)
            enc.setBuffer_offset_atIndex_(b_tree,         0, 1)  # leaves slice
            enc.setBuffer_offset_atIndex_(b_inv_x_base,   0, 2)
            enc.setBuffer_offset_atIndex_(b_zeta_inv_pow, 0, 3)
            enc.setBuffer_offset_atIndex_(b_alpha,        0, 4)
            enc.setBuffer_offset_atIndex_(b_inv_fold,     0, 5)
            enc.setBuffer_offset_atIndex_(b_fold,         0, 6)
            enc.setBuffer_offset_atIndex_(b_n_out,        0, 7)
            enc.dispatchThreads_threadsPerThreadgroup_(
                Metal.MTLSizeMake(grid_f, 1, 1),
                Metal.MTLSizeMake(tg_f, 1, 1),
            )

            # ---- Commit ----
            enc.setComputePipelineState_(pso_commit)
            enc.setBuffer_offset_atIndex_(b_tree,     0, 0)
            enc.setBuffer_offset_atIndex_(b_rc_ext,   0, 1)
            enc.setBuffer_offset_atIndex_(b_rc_int,   0, 2)
            enc.setBuffer_offset_atIndex_(b_ext_mds,  0, 3)
            enc.setBuffer_offset_atIndex_(b_int_diag, 0, 4)
            for lvl in range(n_hash_levels):
                parent_count = counts[lvl + 1]
                tg_lvl = max(1, min(tg_c, parent_count))
                grid_lvl = ((parent_count + tg_lvl - 1) // tg_lvl) * tg_lvl
                enc.setBuffer_offset_atIndex_(b_in_offs,      lvl * 4, 5)
                enc.setBuffer_offset_atIndex_(b_out_offs,     lvl * 4, 6)
                enc.setBuffer_offset_atIndex_(b_child_counts, lvl * 4, 7)
                enc.dispatchThreads_threadsPerThreadgroup_(
                    Metal.MTLSizeMake(grid_lvl, 1, 1),
                    Metal.MTLSizeMake(tg_lvl, 1, 1),
                )

        for _ in range(n_warmup):
            reset()
            harness.time_dispatch(dispatch)
        samples = []
        for _ in range(n_measure):
            reset()
            samples.append(harness.time_dispatch(dispatch))
        gpu_s = float(np.median(samples))

        # Final correctness pass.
        reset()
        harness.time_dispatch(dispatch)
        got = view_tree.copy()

        # Non-canonical: any tree slot >= p is a fault signal.
        non_canonical = int(np.sum(got >= np.uint64(P)))

        mismatches_fold  = int(np.sum(got[:n_out] != folded_ref))
        mismatches_tree  = int(np.sum(got != tree_ref))
        # ``mismatches_tree`` already covers the leaves slice; we keep
        # the split-out fold count so the LLM feedback signal points at
        # which kernel is at fault when only one of the two phases is
        # broken.
        mismatches = mismatches_tree
        correct = (mismatches == 0)

        # ---- Roofline ----
        # Modmul anchor.
        muls_per_output = fold * fold + fold + 2
        fold_muls   = muls_per_output * n_out
        n_perms     = int(sum(counts[1:]))               # all parent nodes
        commit_muls = _POSEIDON2_T3_MULS_PER_PERM * n_perms
        muls_total  = fold_muls + commit_muls
        achieved_mul = gops_per_s(muls_total, gpu_s)
        ceiling_mul  = float(chip.peak_int64_mul_gops)

        # BW anchor.
        bytes_fold = float(
            8 * N                       # evals_in
            + 8 * n_out                 # inv_x_base
            + 8 * n_out                 # evals_out (leaves slice)
            # zeta_inv_pow is fold elements (<= 32 B); ignore.
        )
        bytes_commit = float(
            8 * sum(counts[lvl] for lvl in range(n_hash_levels))            # reads
            + 8 * sum(counts[lvl + 1] for lvl in range(n_hash_levels))      # writes
        )
        bytes_total = bytes_fold + bytes_commit
        achieved_bw = gb_per_s(bytes_total, gpu_s)
        ceiling_bw  = float(chip.peak_bw_gb_s)

        frac_mul = achieved_mul / ceiling_mul if ceiling_mul > 0 else 0.0
        frac_bw  = achieved_bw  / ceiling_bw  if ceiling_bw  > 0 else 0.0
        if frac_mul >= frac_bw:
            achieved, achieved_unit = achieved_mul, "Gmodmul/s (int64)"
            ceiling,  ceiling_unit  = ceiling_mul,  "Gops/s (int64 mul, est)"
            fraction = frac_mul
            anchor = "int64_mul"
        else:
            achieved, achieved_unit = achieved_bw,  "GB/s"
            ceiling,  ceiling_unit  = ceiling_bw,   "GB/s"
            fraction = frac_bw
            anchor = "dram_bw"

        return SizeResult(
            size_label=size.label,
            correct=correct,
            error_value=mismatches,
            error_kind="bit_exact",
            gpu_seconds=gpu_s,
            achieved=achieved,
            achieved_unit=achieved_unit,
            ceiling=ceiling,
            ceiling_unit=ceiling_unit,
            fraction_of_ceiling=fraction,
            extra={
                "N": N, "fold": fold, "n_out": n_out,
                "alpha": int(consts.alpha),
                "n_levels": n_hash_levels,
                "total_tree_nodes": total_nodes,
                "n_perms_commit": n_perms,
                "fold_muls": fold_muls,
                "commit_muls": commit_muls,
                "muls_total": muls_total,
                "bytes_fold": bytes_fold,
                "bytes_commit": bytes_commit,
                "bytes_total": bytes_total,
                "achieved_mul_gops": achieved_mul,
                "achieved_bw_gb_s": achieved_bw,
                "mismatches_fold": mismatches_fold,
                "mismatches_tree": mismatches_tree,
                "non_canonical": non_canonical,
                "anchor": anchor,
            },
        )
