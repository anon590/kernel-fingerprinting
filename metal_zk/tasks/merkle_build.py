"""Merkle build — Z4 task.

Level-by-level Merkle tree build over the Goldilocks field with
Poseidon2 as the inner compression function. The same kernel is
dispatched once per level by the host; the buffer holding the tree
is a single contiguous ``ulong`` array with all levels concatenated.

Sizes (PLAN.md S Regime taxonomy):
  * in-distribution: arity = 2 (Poseidon2 t=3), N in {2^16, 2^18, 2^20}
  * held-out:        arity = 4 (Poseidon2 t=4), N = 2^19

The held-out probe flips on candidates that hardcode arity=2 (number
of siblings gathered per parent), the t=3 external MDS, the t=3 round
constants, or the boundary "child_count is a power of arity" assumption
(2^19 is not a power of 4: the top level has 2 children and must be
zero-padded to fill the t=4 Poseidon2 state).

Roofline:
  Per Poseidon2 permutation at t=3, R_F=8, R_P=22, alpha=7 S-box:
    sboxes  = R_F * t + R_P * 1 = 24 + 22 = 46;  4 modmuls each -> 184
    ext mvm = (1 + R_F) * t * t = 9 * 9 = 81 modmuls
    int mvm = R_P * t = 22 * 3 = 66 modmuls
                                                  total ~ 331 modmuls
  At t=4: sboxes -> 4*8 + 22 = 54 -> 216, ext mvm -> 9 * 16 = 144, int mvm
  -> 22 * 4 = 88; total ~ 448 modmuls (PLAN's estimate is 488 -- the
  difference is whether the int matvec is counted with the same scheme).

  Per build, the total work is
        n_perms_total = sum(level_counts[1:])  ~  N / (arity - 1)
  Bytes moved is dominated by the level-0 -> level-1 transition (each
  leaf is read once, each level-1 node is written once); the upper
  levels contribute a geometric tail of total weight
        bytes_total ~ 8 * N * arity / (arity - 1)

  The task reports the achieved fraction against the **binding**
  roofline ``max(achieved_mul / peak_int64_mul, achieved_bw / peak_bw)``.
  At realistic sizes Merkle-over-Poseidon2 is firmly compute-bound on
  Apple Silicon, so the int-mul anchor dominates; the BW number is
  surfaced in ``extra`` for sanity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import Metal

from ..harness import MetalHarness
from ..hardware import ChipSpec
from ..reference.goldilocks import random_field_elements
from ..reference.merkle import (
    build_tree_cached, level_counts, level_offsets,
)
from ..reference.poseidon2 import Poseidon2Goldilocks
from ..task import (
    Task, TaskSize, TaskSpec, SizeResult,
    gb_per_s, gops_per_s, register_task,
)


_SEED = Path(__file__).resolve().parent.parent.parent / "seeds" / "merkle_build.metal"


def _modmuls_per_perm(t: int, r_f: int, r_p: int) -> int:
    """Counting model (matches poseidon2_hash._modmuls_per_sponge)."""
    sbox = 4                                # x^7 = 4 modmuls
    n_sbox = r_f * t + r_p * 1
    matvec_ext = t * t
    n_matvec_ext = 1 + r_f
    matvec_int = t
    n_matvec_int = r_p
    return n_sbox * sbox + matvec_ext * n_matvec_ext + matvec_int * n_matvec_int


def _t_for_arity(arity: int) -> int:
    """Pick the smallest Poseidon2 width that hosts ``arity`` children."""
    if arity == 2:
        return 3                       # rate 2 + capacity 1
    if arity == 4:
        return 4                       # all four slots are children
    raise ValueError(f"unsupported arity: {arity}")


@register_task("merkle_build")
class MerkleBuildTask(Task):
    spec = TaskSpec(
        name="merkle_build",
        description=(
            "Level-by-level Merkle tree build over the Goldilocks field "
            "(p = 2^64 - 2^32 + 1) using Poseidon2 as the inner "
            "compression function. The tree has ``n_leaves`` input "
            "leaves at level 0; level k+1 has ``ceil(level_k / arity)`` "
            "nodes, computed by hashing groups of ``arity`` consecutive "
            "children. The build terminates at the 1-element root.\n\n"
            "Compression convention (1 Goldilocks element per digest):\n"
            "  state = [c0, c1, ..., c_{arity-1}, 0, ..., 0]   "
            "(zero-pad to width t)\n"
            "  state = Poseidon2_t(state)\n"
            "  parent_digest = state[0]\n\n"
            "The Poseidon2 permutation parameters (alpha=7 S-box, "
            "``r_f`` full rounds split half+half, ``r_p`` partial "
            "rounds, external MDS, internal-MDS diagonal ``int_diag`` "
            "with M_I = J + diag(int_diag) where J is the all-ones "
            "matrix) are all read at runtime from the bound device "
            "buffers, mirroring the layout of the Z3 ``poseidon2_hash`` "
            "task. The same kernel must therefore work at t=3 / "
            "arity=2 (in-distribution) and t=4 / arity=4 (held-out) "
            "without changes -- in particular, the kernel must use "
            "the runtime arity, the runtime t, and the runtime "
            "round-count parameters, not compile-time constants.\n\n"
            "Tree layout: a single contiguous ``ulong`` buffer holds "
            "**all levels** concatenated -- leaves first, then each "
            "parent level, finally the 1-element root. The total "
            "length is the sum of all level node counts. The host "
            "issues one kernel dispatch per parent level, binding the "
            "per-level scalars (``in_offset``, ``out_offset``, "
            "``child_count``); each dispatch reads from "
            "``tree[in_offset .. in_offset + child_count)`` and writes "
            "to ``tree[out_offset .. out_offset + parent_count)`` with "
            "``parent_count = ceil(child_count / arity)`` computed "
            "in-kernel. The serial compute encoder gives "
            "read-after-write ordering between consecutive level "
            "dispatches; the candidate need not insert any explicit "
            "barriers between levels.\n\n"
            "Boundary policy: at each level, if ``child_count`` is not "
            "a multiple of ``arity`` the last group is padded with "
            "**zero** field elements (i.e. the missing children read "
            "as zero into the Poseidon2 state). The CPU reference uses "
            "the same policy; any other padding scheme is a "
            "correctness failure. At arity=4 with N=2^19 leaves the "
            "padding kicks in only at the topmost level (2 children -> "
            "[c0, c1, 0, 0]).\n\n"
            "Correctness is bit-exact against the Python bigint "
            "reference applied to the entire tree (every intermediate "
            "digest is checked, not only the root). Outputs MUST be "
            "canonical (< p); a non-canonical value with the same "
            "residue class still counts as a mismatch. All test sizes "
            "satisfy t <= 4 and R_F <= 8, R_P <= 32; thread-private "
            "state arrays of size 4 are sufficient."
        ),
        kernel_signatures=(
            "kernel void merkle_build_level(\n"
            "    device       ulong *tree         [[buffer(0)]],\n"
            "    device const ulong *rc_ext       [[buffer(1)]],\n"
            "    device const ulong *rc_int       [[buffer(2)]],\n"
            "    device const ulong *ext_mds      [[buffer(3)]],\n"
            "    device const ulong *int_diag     [[buffer(4)]],\n"
            "    constant uint      &arity        [[buffer(5)]],\n"
            "    constant uint      &t            [[buffer(6)]],\n"
            "    constant uint      &r_f          [[buffer(7)]],\n"
            "    constant uint      &r_p          [[buffer(8)]],\n"
            "    constant uint      &in_offset    [[buffer(9)]],\n"
            "    constant uint      &out_offset   [[buffer(10)]],\n"
            "    constant uint      &child_count  [[buffer(11)]],\n"
            "    uint p [[thread_position_in_grid]]);\n"
            "\n"
            "Dispatch (host-fixed, one call per parent level):\n"
            "  threadsPerGrid        = (parent_count, 1, 1)   "
            "rounded up to the TG width\n"
            "  threadsPerThreadgroup = (min(parent_count, 64), 1, 1)\n"
            "Each thread owns ONE parent; guard against p >= "
            "parent_count (the grid is rounded up to a multiple of "
            "the TG width). Threadgroup-cooperative and simdgroup-"
            "cooperative implementations are valid so long as the "
            "external buffer layout above is preserved and the "
            "host's one-dispatch-per-level invocation pattern is "
            "honored (the kernel sees exactly one level's worth of "
            "parents per dispatch via ``child_count``, "
            "``in_offset``, ``out_offset``)."
        ),
        kernel_names=["merkle_build_level"],
        seed_path=_SEED,
        sizes=[
            TaskSize("a2_N64K",  {"arity": 2, "n_leaves": 1 << 16}),
            TaskSize("a2_N256K", {"arity": 2, "n_leaves": 1 << 18}),
            TaskSize("a2_N1M",   {"arity": 2, "n_leaves": 1 << 20}),
        ],
        held_out_sizes=[
            TaskSize("a4_N512K", {"arity": 4, "n_leaves": 1 << 19}),
        ],
    )

    def evaluate_size(self, harness, pipelines, size, chip, n_warmup, n_measure):
        arity = int(size.params["arity"])
        n_leaves = int(size.params["n_leaves"])
        t = _t_for_arity(arity)
        cfg = Poseidon2Goldilocks(t=t)

        # Random leaves -- seed depends on (arity, n_leaves) so the
        # held-out probe gets its own input set and doesn't accidentally
        # share a cache file with an in-dist size.
        leaves = random_field_elements(
            n_leaves, seed=0xBEEFCAFE + n_leaves * 13 + arity * 1009,
        )

        # CPU reference: full tree (leaves + all parent levels). Disk-cached.
        y_ref = build_tree_cached(leaves, arity, cfg)

        # Per-level metadata.
        counts = level_counts(n_leaves, arity)             # ends with 1 (root)
        offsets = level_offsets(counts)                    # len = len(counts) + 1
        n_hash_levels = len(counts) - 1
        total_nodes = offsets[-1]

        # Tree buffer: holds all levels concatenated. Initialised with the
        # leaves in the first ``n_leaves`` slots; everything above starts
        # as zero (the kernel overwrites those slots level by level).
        tree_init = np.zeros(total_nodes, dtype=np.uint64)
        tree_init[:n_leaves] = leaves
        b_tree = harness.buf_from_np(tree_init)

        # Poseidon2 constants.
        b_rc_ext   = harness.buf_from_np(cfg.round_constants_external.ravel())
        b_rc_int   = harness.buf_from_np(cfg.round_constants_internal)
        b_ext_mds  = harness.buf_from_np(cfg.ext_mds.ravel())
        b_int_diag = harness.buf_from_np(cfg.int_diag)
        b_arity    = harness.buf_scalar(arity, np.uint32)
        b_t        = harness.buf_scalar(t, np.uint32)
        b_rf       = harness.buf_scalar(cfg.r_f, np.uint32)
        b_rp       = harness.buf_scalar(cfg.r_p, np.uint32)

        # Per-level scalar arrays -- bound via per-dispatch offset to
        # avoid creating 3 * n_hash_levels small buffers (same pattern
        # the goldilocks_ntt task uses for ``stage_idx``).
        in_offs_arr      = np.array(offsets[:n_hash_levels],         dtype=np.uint32)
        out_offs_arr     = np.array(offsets[1:n_hash_levels + 1],    dtype=np.uint32)
        child_counts_arr = np.array(counts[:n_hash_levels],          dtype=np.uint32)
        b_in_offs       = harness.buf_from_np(in_offs_arr)
        b_out_offs      = harness.buf_from_np(out_offs_arr)
        b_child_counts  = harness.buf_from_np(child_counts_arr)

        pso = pipelines["merkle_build_level"]
        max_tg = int(pso.maxTotalThreadsPerThreadgroup())
        tew = int(pso.threadExecutionWidth())
        tg_w = min(max_tg, max(tew, 64))

        view_tree = harness.np_view(b_tree, np.uint64, total_nodes)

        def reset():
            # Re-zero the parent slots; the leaves slice stays valid
            # across iterations because we never overwrite it.
            view_tree[n_leaves:] = 0

        def dispatch(enc):
            enc.setComputePipelineState_(pso)
            enc.setBuffer_offset_atIndex_(b_tree,     0, 0)
            enc.setBuffer_offset_atIndex_(b_rc_ext,   0, 1)
            enc.setBuffer_offset_atIndex_(b_rc_int,   0, 2)
            enc.setBuffer_offset_atIndex_(b_ext_mds,  0, 3)
            enc.setBuffer_offset_atIndex_(b_int_diag, 0, 4)
            enc.setBuffer_offset_atIndex_(b_arity,    0, 5)
            enc.setBuffer_offset_atIndex_(b_t,        0, 6)
            enc.setBuffer_offset_atIndex_(b_rf,       0, 7)
            enc.setBuffer_offset_atIndex_(b_rp,       0, 8)
            for lvl in range(n_hash_levels):
                parent_count = counts[lvl + 1]
                grid_w = ((parent_count + tg_w - 1) // tg_w) * tg_w
                enc.setBuffer_offset_atIndex_(b_in_offs,      lvl * 4,  9)
                enc.setBuffer_offset_atIndex_(b_out_offs,     lvl * 4, 10)
                enc.setBuffer_offset_atIndex_(b_child_counts, lvl * 4, 11)
                enc.dispatchThreads_threadsPerThreadgroup_(
                    Metal.MTLSizeMake(grid_w, 1, 1),
                    Metal.MTLSizeMake(tg_w, 1, 1),
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

        mismatches = int(np.sum(got != y_ref))
        correct = (mismatches == 0)

        # Roofline: int-mul anchor and BW anchor; pick the binding one.
        muls_per_perm = _modmuls_per_perm(t, cfg.r_f, cfg.r_p)
        n_perms_total = int(sum(counts[1:]))            # all parent nodes
        muls_total = muls_per_perm * n_perms_total
        achieved_mul = gops_per_s(muls_total, gpu_s)
        ceiling_mul = float(chip.peak_int64_mul_gops)

        # BW model: each non-root node is read once by the next level's
        # kernel (level lvl reads counts[lvl] children); each parent node
        # is written once (level lvl writes counts[lvl + 1] parents).
        bytes_read  = sum(counts[lvl] for lvl in range(n_hash_levels)) * 8
        bytes_write = sum(counts[lvl] for lvl in range(1, n_hash_levels + 1)) * 8
        bytes_total = float(bytes_read + bytes_write)
        achieved_bw = gb_per_s(bytes_total, gpu_s)
        ceiling_bw = float(chip.peak_bw_gb_s)

        frac_mul = achieved_mul / ceiling_mul if ceiling_mul > 0 else 0.0
        frac_bw  = achieved_bw  / ceiling_bw  if ceiling_bw  > 0 else 0.0
        if frac_mul >= frac_bw:
            achieved, achieved_unit = achieved_mul, "Gmodmul/s (int64)"
            ceiling, ceiling_unit = ceiling_mul, "Gops/s (int64 mul, est)"
            fraction = frac_mul
            anchor = "int64_mul"
        else:
            achieved, achieved_unit = achieved_bw, "GB/s"
            ceiling, ceiling_unit = ceiling_bw, "GB/s"
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
                "arity": arity,
                "n_leaves": n_leaves,
                "t": t,
                "r_f": cfg.r_f,
                "r_p": cfg.r_p,
                "alpha": cfg.alpha,
                "n_levels": n_hash_levels,
                "total_nodes": total_nodes,
                "n_perms_total": n_perms_total,
                "muls_per_perm": muls_per_perm,
                "muls_total": muls_total,
                "bytes_total": bytes_total,
                "achieved_mul_gops": achieved_mul,
                "achieved_bw_gb_s": achieved_bw,
                "anchor": anchor,
            },
        )
