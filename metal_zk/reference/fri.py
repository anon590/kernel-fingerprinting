"""FRI folding-round reference (CPU).

One round of FRI (Fast Reed-Solomon IOP of Proximity) folding over the
Goldilocks field, plus the binary-Poseidon2 Merkle commit of the folded
evaluations. Mirrors what every modern STARK prover (plonky2, risc0,
winterfell) emits between consecutive FRI rounds.

Convention
==========

We work over a coset of the multiplicative subgroup of order ``N``:

    D = { coset_g * omega_N^i : i in [0, N) }

with ``omega_N`` the primitive ``N``-th root of unity in Goldilocks
(derived from the plonky2 / risc0 generator
``g_root_2^32 = 1753635133440165772``) and ``coset_g = 7``, the
Goldilocks multiplicative generator (matches plonky2's
``Goldilocks::GENERATOR``).

The input is a length-``N`` evaluation table ``E`` of some polynomial
``f`` of degree ``< N`` on ``D``. The output of one folding round at
folding factor ``fold in {2, 4}`` is a length-``N/fold`` evaluation
table ``E'`` of the folded polynomial ``f'`` on the coset

    D' = { coset_g^fold * omega_N^(j * fold) : j in [0, N / fold) }.

The folded polynomial is

    f'(Y) = sum_{p=0}^{fold-1} alpha^p * f_p(Y)

where the ``f_p`` come from the polynomial decomposition

    f(X) = sum_{p=0}^{fold-1} X^p * f_p(X^fold).

Equivalently, gathering the ``fold`` coset-partners of each output
index ``j`` and applying the closed-form FRI fold:

    E'[j] = inv_fold * sum_{m=0}^{fold-1} S_m(j) * E[j + m * (N/fold)]

with ``S_m(j) = sum_{p=0}^{fold-1} r_m^p`` and
``r_m = alpha / (coset_g * omega_N^{j + m * (N/fold)})``. For
``fold = 2`` this collapses to the textbook ``(even + alpha * odd)``
fold; for ``fold = 4`` it is the iDFT-then-alpha-fold of the 4-point
preimage set.

After folding, we commit to ``E'`` with a binary Poseidon2-t=3 Merkle
tree (zero-pad to ``[E'[2i], E'[2i+1], 0]`` and take ``state[0]`` as the
parent digest -- same convention as the Z4 ``merkle_build`` task).

Disk + memory caching
=====================

Bigint Python evaluates one full round at ``N = 2^20`` in ~30 s. The
results are cached under ``~/.cache/metal-zk/fri/`` keyed by
``sha256(version, N, fold, alpha, coset_g, input bytes)``; subsequent
evolve-loop iterations on the same size + seed read the cache in
``O(disk)``.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .goldilocks import P, root_of_unity
from .merkle import build_tree_reference, level_counts, total_tree_nodes
from .poseidon2 import Poseidon2Goldilocks


# Plonky2 / Goldilocks multiplicative generator, used as the FRI coset
# shift. Any non-residue would do; 7 is the canonical choice for the
# field and matches plonky2's ``Goldilocks::GENERATOR``.
COSET_G: int = 7


# ----------------------------------------------------------------------
# Host-side precomputation for the Metal kernel
# ----------------------------------------------------------------------

@dataclass
class FriRoundConstants:
    """Constants the host uploads to the kernel for one FRI round."""
    inv_x_base: np.ndarray       # uint64[N // fold]; inv_x_base[j] = 1/(g * omega^j)
    zeta_inv_pow: np.ndarray     # uint64[fold]; zeta_inv_pow[m] = zeta^{-m}
    alpha: int                   # challenge, canonical
    inv_fold: int                # 1/fold mod p
    fold: int
    n_out: int                   # N // fold
    coset_g: int                 # g
    omega_N: int                 # primitive N-th root of unity


def fri_round_constants(
    N: int,
    fold: int,
    alpha: int,
    coset_g: int = COSET_G,
) -> FriRoundConstants:
    """Compute every kernel-side constant for one FRI round at length
    ``N``, fold factor ``fold``, challenge ``alpha`` and coset shift
    ``coset_g``. All outputs are canonical Goldilocks elements."""
    if N & (N - 1) != 0:
        raise ValueError(f"N={N} must be a power of two")
    if fold not in (2, 4):
        raise ValueError(f"fold={fold} not in (2, 4)")
    if N % fold != 0:
        raise ValueError(f"N={N} not divisible by fold={fold}")
    log_N = (N - 1).bit_length()
    omega_N = root_of_unity(log_N)
    n_out = N // fold

    # zeta = primitive fold-th root of unity = omega_N^(N/fold).
    zeta = pow(omega_N, n_out, P)
    zeta_inv = pow(zeta, P - 2, P)

    # zeta_inv_pow[m] = zeta^{-m}.
    zinv_pow = np.empty(fold, dtype=np.uint64)
    zinv_pow[0] = 1
    for m in range(1, fold):
        zinv_pow[m] = int(zinv_pow[m - 1]) * zeta_inv % P

    # inv_x_base[j] = 1 / (g * omega_N^j) for j in [0, n_out).
    g_canonical = int(coset_g) % P
    inv_g = pow(g_canonical, P - 2, P)
    omega_inv = pow(omega_N, P - 2, P)
    inv_x_base = np.empty(n_out, dtype=np.uint64)
    inv_x_base[0] = inv_g
    for j in range(1, n_out):
        inv_x_base[j] = int(inv_x_base[j - 1]) * omega_inv % P

    return FriRoundConstants(
        inv_x_base=inv_x_base,
        zeta_inv_pow=zinv_pow,
        alpha=int(alpha) % P,
        inv_fold=pow(fold, P - 2, P),
        fold=int(fold),
        n_out=n_out,
        coset_g=g_canonical,
        omega_N=omega_N,
    )


# ----------------------------------------------------------------------
# Reference fold + commit
# ----------------------------------------------------------------------

def fold_round_reference(
    evals: np.ndarray,
    fold: int,
    alpha: int,
    coset_g: int = COSET_G,
) -> np.ndarray:
    """One FRI folding round (no Merkle). Returns length-``N/fold``
    folded evaluations as canonical ``uint64``.

    Uses the closed-form fold (a single pass over ``fold`` gather
    indices per output, with an inner geometric series) so the kernel
    and the reference share an arithmetic structure; this makes
    bit-exact agreement testable without an iDFT-style alternative
    formulation drifting in rounding-free integer arithmetic (there is
    no drift over a prime field, but matching the kernel's algebra
    keeps debugging tractable when something fails).
    """
    if evals.ndim != 1:
        raise ValueError(f"evals must be 1D; got shape {evals.shape}")
    N = int(evals.shape[0])
    consts = fri_round_constants(N, fold, alpha, coset_g)
    n_out = consts.n_out
    inv_x = [int(v) for v in consts.inv_x_base.tolist()]
    zinv = [int(v) for v in consts.zeta_inv_pow.tolist()]
    a = consts.alpha
    inv_f = consts.inv_fold

    src = [int(v) % P for v in evals.tolist()]
    out = [0] * n_out
    for j in range(n_out):
        ax = a * inv_x[j] % P
        acc = 0
        for m in range(fold):
            rm = ax * zinv[m] % P
            sm = 0
            rpow = 1
            for _ in range(fold):
                sm = (sm + rpow) % P
                rpow = rpow * rm % P
            acc = (acc + sm * src[j + m * n_out]) % P
        out[j] = acc * inv_f % P
    return np.array(out, dtype=np.uint64)


def fri_round_reference(
    evals: np.ndarray,
    fold: int,
    alpha: int,
    coset_g: int = COSET_G,
    cfg: Poseidon2Goldilocks | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One full FRI round: fold + binary Poseidon2-t=3 Merkle commit.

    Returns ``(folded, tree)`` where:
      - ``folded`` is the length-``N/fold`` folded evaluation table;
      - ``tree``  is the flat ``uint64`` buffer
        ``[folded | level_1 | ... | root]`` of length
        ``total_tree_nodes(N/fold, 2)``.
    """
    if cfg is None:
        cfg = Poseidon2Goldilocks(t=3)
    if cfg.t != 3:
        raise ValueError("FRI commit uses binary Poseidon2-t=3")
    folded = fold_round_reference(evals, fold, alpha, coset_g)
    tree = build_tree_reference(folded, arity=2, cfg=cfg)
    return folded, tree


# ----------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------

_CACHE_VERSION = "v1"
_MEM_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _cache_dir() -> Path:
    root = Path(os.environ.get("METAL_ZK_CACHE",
                               "~/.cache/metal-zk")).expanduser()
    d = root / "fri"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(
    N: int, fold: int, alpha: int, coset_g: int, evals: np.ndarray,
) -> str:
    h = hashlib.sha256()
    h.update(_CACHE_VERSION.encode())
    h.update(np.uint64(N).tobytes())
    h.update(np.uint32(fold).tobytes())
    h.update(np.uint64(int(alpha) % P).tobytes())
    h.update(np.uint64(int(coset_g) % P).tobytes())
    h.update(np.ascontiguousarray(evals, dtype=np.uint64).tobytes())
    return h.hexdigest()[:16]


def fri_round_cached(
    evals: np.ndarray,
    fold: int,
    alpha: int,
    coset_g: int = COSET_G,
    cfg: Poseidon2Goldilocks | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``fri_round_reference`` with disk + memory caching.

    Cache key is sha256(version, N, fold, alpha, coset_g, input bytes).
    Stores ``(folded, tree)`` as two adjacent ``.u64`` files so the
    bigint Python fold + Merkle build runs once per (size, seed) per
    machine and is replayed thereafter in O(read).
    """
    if cfg is None:
        cfg = Poseidon2Goldilocks(t=3)
    N = int(evals.shape[0])
    n_out = N // fold
    tree_nodes = total_tree_nodes(n_out, 2)
    key = _cache_key(N, fold, alpha, coset_g, evals)
    mem_key = f"N{N}_f{fold}_{key}"
    if mem_key in _MEM_CACHE:
        return _MEM_CACHE[mem_key]
    base = _cache_dir() / f"N{N}_f{fold}_a{int(alpha) % P:016x}_{key}"
    folded_path = base.with_suffix(".folded.u64")
    tree_path   = base.with_suffix(".tree.u64")
    if folded_path.exists() and tree_path.exists():
        folded = np.fromfile(folded_path, dtype=np.uint64)
        tree   = np.fromfile(tree_path,   dtype=np.uint64)
        if folded.shape[0] == n_out and tree.shape[0] == tree_nodes:
            _MEM_CACHE[mem_key] = (folded, tree)
            return folded, tree
    folded, tree = fri_round_reference(evals, fold, alpha, coset_g, cfg)
    for arr, path in ((folded, folded_path), (tree, tree_path)):
        tmp = path.with_suffix(path.suffix + ".tmp")
        arr.tofile(tmp)
        os.replace(tmp, path)
    _MEM_CACHE[mem_key] = (folded, tree)
    return folded, tree


__all__ = [
    "COSET_G",
    "FriRoundConstants",
    "fri_round_constants",
    "fold_round_reference",
    "fri_round_reference",
    "fri_round_cached",
]
