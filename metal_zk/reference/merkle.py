"""Merkle tree reference (CPU).

Level-by-level Merkle build over the Goldilocks field using the
Poseidon2 permutation from :mod:`metal_zk.reference.poseidon2` as the
inner compression function.

Compression convention (digest = 1 Goldilocks element):

  parent = Poseidon2_t(
      state = [c0, c1, ..., c_{arity-1}, 0, ..., 0]  # zero-pad to width t
  )[0]

with ``t = 3`` for ``arity = 2`` (rate 2 + capacity 1, two children +
one zero) and ``t = 4`` for ``arity = 4`` (all four slots filled). Both
parameter sets ship distinct round constants, external MDS, and
internal-MDS diagonal — see :mod:`metal_zk.reference.poseidon2` — so a
candidate that hardcodes either family silently fails the held-out
arity-4 probe.

Boundary policy: at each level, if ``child_count`` is not a multiple of
``arity`` the last group is padded with **zero** field elements. With
``arity = 4`` and ``N = 2^19`` leaves this happens only at the very top
of the tree (the level with 2 children produces a 1-node root via
``[c0, c1, 0, 0]``).

Tree layout: a single contiguous ``uint64`` array holding all levels
concatenated, leaves first, then each parent level in order, finally
the 1-element root. Total length ``sum(level_counts)``. This matches
the GPU buffer layout used by the Metal kernel (the kernel reads from
the level-l slice and writes to the level-(l+1) slice).

Disk + memory cache mirrors ``poseidon2.permute_batch_cached`` because
the pure-Python bigint Poseidon2 dominates the build for
``N = 2^20`` (one inner permutation ~ 100 us in Python, ~ 2^20
permutations total -> ~ 100 s).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from .goldilocks import P
from .poseidon2 import Poseidon2Goldilocks


def level_counts(n_leaves: int, arity: int) -> list[int]:
    """Per-level node counts ``[N, ceil(N/a), ceil(N/a^2), ..., 1]``.

    Length is the number of levels including leaves and root. For
    ``n_leaves <= 1`` returns ``[n_leaves]`` (no parent levels).
    """
    if arity < 2:
        raise ValueError(f"arity must be >= 2; got {arity}")
    if n_leaves < 1:
        raise ValueError(f"n_leaves must be >= 1; got {n_leaves}")
    counts = [int(n_leaves)]
    while counts[-1] > 1:
        counts.append((counts[-1] + arity - 1) // arity)
    return counts


def level_offsets(counts: list[int]) -> list[int]:
    """Prefix-sum offsets: ``offsets[i]`` is the start of level ``i`` in the
    flat tree buffer. ``offsets[-1]`` equals the total tree size."""
    offsets = [0]
    for c in counts:
        offsets.append(offsets[-1] + c)
    return offsets


def total_tree_nodes(n_leaves: int, arity: int) -> int:
    return sum(level_counts(n_leaves, arity))


def build_tree_reference(
    leaves: np.ndarray,
    arity: int,
    cfg: Poseidon2Goldilocks,
) -> np.ndarray:
    """Build the Merkle tree level-by-level. Returns a flat uint64 array
    ``[leaves | level_1 | level_2 | ... | root]`` of length
    ``total_tree_nodes(N, arity)``.

    All inputs are reduced mod p before hashing; outputs are canonical.
    """
    if leaves.ndim != 1:
        raise ValueError(f"leaves must be 1-D; got shape {leaves.shape}")
    if cfg.t < arity:
        raise ValueError(
            f"Poseidon2 t={cfg.t} cannot host arity={arity} children"
        )

    n_leaves = int(leaves.shape[0])
    counts = level_counts(n_leaves, arity)
    offsets = level_offsets(counts)
    tree = np.zeros(offsets[-1], dtype=np.uint64)
    tree[:n_leaves] = np.asarray(leaves, dtype=np.uint64) % np.uint64(P)

    t = cfg.t
    for lvl in range(len(counts) - 1):
        in_off = offsets[lvl]
        out_off = offsets[lvl + 1]
        child_count = counts[lvl]
        parent_count = counts[lvl + 1]
        for p in range(parent_count):
            state = np.zeros(t, dtype=np.uint64)
            base = p * arity
            for i in range(arity):
                src = base + i
                if src < child_count:
                    state[i] = tree[in_off + src]
            out = cfg.permute(state)
            tree[out_off + p] = out[0]

    return tree


# ----------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------

_CACHE_VERSION = "v1"
_MEM_CACHE: dict[str, np.ndarray] = {}


def _cache_dir() -> Path:
    root = Path(os.environ.get("METAL_ZK_CACHE",
                               "~/.cache/metal-zk")).expanduser()
    d = root / "merkle"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(arity: int, t: int, leaves: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(_CACHE_VERSION.encode())
    h.update(np.uint32(arity).tobytes())
    h.update(np.uint32(t).tobytes())
    h.update(np.uint64(leaves.shape[0]).tobytes())
    h.update(np.ascontiguousarray(leaves, dtype=np.uint64).tobytes())
    return h.hexdigest()[:16]


def build_tree_cached(
    leaves: np.ndarray,
    arity: int,
    cfg: Poseidon2Goldilocks,
) -> np.ndarray:
    """``build_tree_reference`` with in-memory + disk caching.

    Cache key is sha256(arity, t, N, input bytes). For the in-distribution
    sizes (up to 2^20 leaves at arity 2) the first-time build dominates;
    subsequent iterations of the evolve loop hit the cache for free.
    """
    n_leaves = int(leaves.shape[0])
    key = _cache_key(arity, cfg.t, leaves)
    mem_key = f"a{arity}_t{cfg.t}_N{n_leaves}_{key}"
    if mem_key in _MEM_CACHE:
        return _MEM_CACHE[mem_key]
    path = _cache_dir() / f"a{arity}_t{cfg.t}_N{n_leaves}_{key}.u64"
    expected_total = total_tree_nodes(n_leaves, arity)
    if path.exists():
        flat = np.fromfile(path, dtype=np.uint64)
        if flat.shape[0] == expected_total:
            _MEM_CACHE[mem_key] = flat
            return flat
    out = build_tree_reference(leaves, arity, cfg)
    tmp = path.with_suffix(path.suffix + ".tmp")
    out.tofile(tmp)
    os.replace(tmp, path)
    _MEM_CACHE[mem_key] = out
    return out


__all__ = [
    "level_counts",
    "level_offsets",
    "total_tree_nodes",
    "build_tree_reference",
    "build_tree_cached",
]
