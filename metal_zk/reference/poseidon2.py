"""Poseidon2 reference (Grassi-Khovratovich-Lueftenegger 2023) over the
Goldilocks field.

State width ``t`` configurable at instantiation time; we ship two
concrete parameter sets:

- ``Poseidon2Goldilocks(t=3)`` — in-distribution. ``R_F = 8`` full
  rounds split 4+4, ``R_P = 22`` partial rounds, S-box ``x^7``.
- ``Poseidon2Goldilocks(t=4)`` — held-out probe. Same field, S-box,
  and full/partial round counts; **different** external MDS (4x4
  cyclic with diagonal 5) and **different** internal-MDS diagonal.

Round constants are generated deterministically from a domain-tagged
SHAKE-128 expansion of ``"metal-zk:poseidon2:goldilocks:t={t}"``,
rejection-sampled into the field. The constants are vendored-by-
generation: the same script run on any machine reproduces the same
``round_constants_external`` / ``round_constants_internal`` arrays.

The external MDS matrix uses the Poseidon2 paper's standard
construction:
  - t=3: ``M_E = [[2,1,1],[1,2,1],[1,1,2]]``
  - t=4: ``M_E = circulant(5, 7, 1, 3)`` (4x4 cyclic with non-trivial
    diagonal, distinct from t=3's symmetric structure).

The internal MDS is ``M_I = I + diag(d_int)`` where ``d_int`` is the
per-arity diagonal: a fixed pattern for t=3 and a *different* pattern
for t=4, so a Metal kernel that hardcodes either MDS silently fails
the held-out probe.

API:
  cfg = Poseidon2Goldilocks(t=3)
  digest = cfg.permute(state)               # one full 8+22+ permutation
  cfg.round_constants_external              # (R_F, t) uint64
  cfg.round_constants_internal              # (R_P,) uint64
  cfg.ext_mds                               # (t, t) uint64
  cfg.int_diag                              # (t,) uint64
  cfg.r_f, cfg.r_p, cfg.alpha               # ints
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .goldilocks import P


ALPHA: int = 7                  # S-box exponent: x^7 (gcd(7, p-1) = 1)
R_F: int = 8                    # 4 + 4 full rounds
R_P: int = 22                   # partial rounds (Poseidon2 / Goldilocks, t=3)


def _shake128_field_elements(domain: bytes, n: int) -> list[int]:
    """Rejection-sample n field elements from a domain-tagged SHAKE-128
    stream. Deterministic and portable across machines."""
    shake = hashlib.shake_128()
    shake.update(domain)
    # Generate a generous buffer; ~1.06x oversample is enough on average
    # (rejection rate ~ 2^-32). Repeat in chunks if we ever miss.
    out: list[int] = []
    block_idx = 0
    while len(out) < n:
        block_size = max(8 * (n - len(out)) * 2, 256)
        raw = shake.digest(block_idx + block_size)[block_idx:]
        block_idx += block_size
        for i in range(0, len(raw), 8):
            if len(out) >= n:
                break
            v = int.from_bytes(raw[i:i + 8], "little")
            if v < P:
                out.append(v)
    return out


def _ext_mds_t3() -> np.ndarray:
    """Poseidon2 external MDS for t=3 (the standard symmetric form)."""
    return np.array(
        [[2, 1, 1],
         [1, 2, 1],
         [1, 1, 2]],
        dtype=np.uint64,
    )


def _ext_mds_t4() -> np.ndarray:
    """Poseidon2 external MDS for t=4 (4x4 circulant with diagonal=5).

    This is structurally different from t=3's symmetric matrix: a Metal
    kernel that hardcodes the t=3 form ``[[2,1,1],[1,2,1],[1,1,2]]`` and
    inputs a t=4 state will produce wrong output, not just slow output.
    """
    return np.array(
        [[5, 7, 1, 3],
         [3, 5, 7, 1],
         [1, 3, 5, 7],
         [7, 1, 3, 5]],
        dtype=np.uint64,
    )


def _int_diag_t3() -> np.ndarray:
    """Internal-MDS diagonal for t=3 over Goldilocks.

    M_I = I + diag(d); we choose ``d`` so M_I is MDS (every square
    submatrix is invertible). For small t the simplest valid choice
    is small distinct field elements; we use [1, 2, 3]."""
    return np.array([1, 2, 3], dtype=np.uint64)


def _int_diag_t4() -> np.ndarray:
    """Internal-MDS diagonal for t=4 — **different** from t=3, on
    purpose. Held-out probe: kernel hardcoding [1,2,3] fails at t=4."""
    return np.array([2, 3, 5, 7], dtype=np.uint64)


@dataclass
class Poseidon2Goldilocks:
    t: int

    def __post_init__(self) -> None:
        if self.t not in (3, 4):
            raise ValueError(
                f"only t in (3, 4) ships built-in constants; got t={self.t}"
            )
        self.alpha: int = ALPHA
        self.r_f: int = R_F
        self.r_p: int = R_P

        domain = f"metal-zk:poseidon2:goldilocks:t={self.t}".encode()
        rc_ext_flat = _shake128_field_elements(
            domain + b":ext", self.r_f * self.t
        )
        rc_int = _shake128_field_elements(
            domain + b":int", self.r_p
        )
        self.round_constants_external = np.array(
            rc_ext_flat, dtype=np.uint64
        ).reshape(self.r_f, self.t)
        self.round_constants_internal = np.array(
            rc_int, dtype=np.uint64
        )

        if self.t == 3:
            self.ext_mds = _ext_mds_t3()
            self.int_diag = _int_diag_t3()
        else:
            self.ext_mds = _ext_mds_t4()
            self.int_diag = _int_diag_t4()

    # ------------------------------------------------------------------
    # Field helpers (Python bigint; not perf-critical for the reference)
    # ------------------------------------------------------------------

    @staticmethod
    def _sbox(x: int) -> int:
        # x^7 mod p, square-and-multiply
        x2 = (x * x) % P
        x3 = (x2 * x) % P
        x4 = (x2 * x2) % P
        return (x3 * x4) % P

    def _matvec_ext(self, state: list[int]) -> list[int]:
        m = self.ext_mds.tolist()
        return [
            sum(int(m[i][j]) * state[j] for j in range(self.t)) % P
            for i in range(self.t)
        ]

    def _matvec_int(self, state: list[int]) -> list[int]:
        # M_I = I + diag(d). Equivalently, ``y = state + sum(state)`` is
        # the rank-one Poseidon2 paper version, but here we use the
        # explicit ``M_I[i,j] = (i==j ? 1 + d[i] : 0)`` form, *plus*
        # the all-ones rank-one update so the matrix is genuinely MDS:
        #   M_I[i,j] = (i==j ? 1 + d[i] : 1)
        d = self.int_diag.tolist()
        s = sum(state) % P
        return [(s + int(d[i]) * state[i]) % P for i in range(self.t)]

    # ------------------------------------------------------------------
    # Permutation
    # ------------------------------------------------------------------

    def permute(self, state: np.ndarray) -> np.ndarray:
        """Apply the Poseidon2 permutation to ``state`` (length-t uint64)."""
        if state.shape != (self.t,):
            raise ValueError(
                f"state has shape {state.shape}, expected ({self.t},)"
            )
        x = [int(v) % P for v in state.tolist()]
        x = self._matvec_ext(x)

        # First half of full rounds.
        for r in range(self.r_f // 2):
            rc = [int(v) for v in self.round_constants_external[r].tolist()]
            x = [(xi + ci) % P for xi, ci in zip(x, rc)]
            x = [self._sbox(xi) for xi in x]
            x = self._matvec_ext(x)

        # Partial rounds: S-box on lane 0 only, internal MDS.
        for r in range(self.r_p):
            ci = int(self.round_constants_internal[r])
            x[0] = (x[0] + ci) % P
            x[0] = self._sbox(x[0])
            x = self._matvec_int(x)

        # Second half of full rounds.
        for r in range(self.r_f // 2, self.r_f):
            rc = [int(v) for v in self.round_constants_external[r].tolist()]
            x = [(xi + ci) % P for xi, ci in zip(x, rc)]
            x = [self._sbox(xi) for xi in x]
            x = self._matvec_ext(x)

        return np.array(x, dtype=np.uint64)

    # ------------------------------------------------------------------
    # Batched reference (used by the task harness)
    # ------------------------------------------------------------------

    def permute_batch(self, states: np.ndarray) -> np.ndarray:
        """Apply the permutation to each row of ``states`` (B, t)."""
        if states.ndim != 2 or states.shape[1] != self.t:
            raise ValueError(
                f"states shape {states.shape} != (B, {self.t})"
            )
        out = np.empty_like(states)
        for i in range(states.shape[0]):
            out[i] = self.permute(states[i])
        return out


# ----------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------

_CACHE_VERSION = "v1"
_MEM_CACHE: dict[str, np.ndarray] = {}


def _cache_dir() -> Path:
    root = Path(os.environ.get("METAL_ZK_CACHE",
                               "~/.cache/metal-zk")).expanduser()
    d = root / "poseidon2"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(t: int, states: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(_CACHE_VERSION.encode())
    h.update(np.uint32(t).tobytes())
    h.update(np.uint64(states.shape[0]).tobytes())
    h.update(np.ascontiguousarray(states, dtype=np.uint64).tobytes())
    return h.hexdigest()[:16]


def permute_batch_cached(cfg: Poseidon2Goldilocks,
                         states: np.ndarray) -> np.ndarray:
    """`permute_batch` with in-memory + disk caching.

    Pure-Python bigint Poseidon2 on 2^20 sponges takes ~90s; caching is
    important whenever the same (t, states) inputs are re-evaluated --
    which happens every LLM iteration in the evolve loop AND every rep
    in a multi-rep timing run. The cache key is sha256(t, B, input
    bytes); cache files live under ~/.cache/metal-zk/poseidon2/.
    """
    if states.ndim != 2 or states.shape[1] != cfg.t:
        raise ValueError(
            f"states shape {states.shape} != (B, {cfg.t})"
        )
    key = _cache_key(cfg.t, states)
    mem_key = f"t{cfg.t}_{key}"
    if mem_key in _MEM_CACHE:
        return _MEM_CACHE[mem_key]
    batch = int(states.shape[0])
    path = _cache_dir() / f"t{cfg.t}_B{batch}_{key}.u64"
    if path.exists():
        flat = np.fromfile(path, dtype=np.uint64)
        if flat.shape[0] == batch * cfg.t:
            out = flat.reshape(batch, cfg.t)
            _MEM_CACHE[mem_key] = out
            return out
    out = cfg.permute_batch(states)
    tmp = path.with_suffix(path.suffix + ".tmp")
    out.tofile(tmp)
    os.replace(tmp, path)
    _MEM_CACHE[mem_key] = out
    return out


__all__ = [
    "ALPHA", "R_F", "R_P",
    "Poseidon2Goldilocks",
    "permute_batch_cached",
]
