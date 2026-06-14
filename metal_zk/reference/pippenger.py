"""Pippenger bucket-scatter reference (Z9).

Given ``N`` 256-bit scalars and ``N`` short-Weierstrass curve points
on BLS12-381 G1, compute the per-window bucket sums of Pippenger's
MSM. The window width is fixed at ``w = WINDOW_BITS = 16`` and we
process the bottom ``NUM_WINDOWS = 4`` windows of each scalar -- 64
bits of scalar input per pair. (PLAN.md Z9 specifies w=16 and 256-bit
scalars; we honor w=16 but truncate to 4 windows because a full
16-window output buffer at ``2^16-1`` buckets per window is ~150 MB.
The bucket-scatter lever is preserved end-to-end: the bucket index
space is the full 2^16-1, the bucket reduction is the full Jacobian
sum, only the per-pair window-multiplicity dimension is smaller.)

For each pair ``i`` and window ``k in [0, NUM_WINDOWS)``:

    b_k(s_i) = (s_i.bottom64 >> (k * w)) & (2^w - 1)

The output is the table ``S[k][b]`` of Jacobian sums (in affine
Montgomery form for the correctness check):

    S[k][b] = sum_{i : b_k(s_i) == b} P_i,  for b in [1, 2^w)

Bucket ``b = 0`` is excluded (it would contribute 0 to the MSM
aggregation step that follows this kernel; PLAN.md and arkworks /
icicle / plonky agree on this convention).

Held-out twist (PLAN.md Z9):
  - in-distribution: uniform scalars (each window value is uniformly
    distributed in ``[0, 2^w)``, so the bucket histogram is flat).
  - held-out: power-law scalars (each window value is sampled
    independently from a Zipf distribution truncated to ``[1, 2^w)``).
    Bucket 1 absorbs ~38% of all traffic under Zipf-1.5; the top-1%
    of buckets carry ~10^3x the median's traffic. A candidate that
    tuned its atomic-scatter strategy for the uniform contention
    pattern catastrophically serialises on the hot buckets.

Bit-exactness convention
------------------------
The seed (and any improved candidate) accumulates the bucket sums in
non-deterministic order -- thread scheduling decides which pair gets
added to bucket b first. EC addition is commutative so the final
affine point is invariant; the Jacobian representation is not. We
normalize *both* the GPU output and the reference to canonical affine
Montgomery form ``(X / Z^2, Y / Z^3) * R mod q`` and compare those
two ``(X_aff_mont, Y_aff_mont)`` tuples bit-exactly. Non-canonical
limbs (>= q) on the GPU side count as mismatches.

Cache
-----
Reference building is dominated by the ``N * NUM_WINDOWS`` Python
Jacobian additions (~70 us each in pure-Python bigint at 256-bit). At
``N = 2^16`` with 4 windows that's ~262 K adds = ~20 s; cached to
disk so subsequent runs / evaluations are instantaneous. The cache key
binds ``(curve, n_pairs, a, b, distribution, seed)`` so the held-out
probe never reads the in-distribution cache.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .msm import (
    BLS12_381_G1, CurveParams, JacMont, Montgomery, N_LIMBS,
    _seed_derived_ab, int_to_limbs, jac_add, jac_infinity,
    jac_to_affine_mont, limbs_to_int, precompute_points_cached,
)


# ----------------------------------------------------------------------
# Window / decomposition constants
# ----------------------------------------------------------------------

WINDOW_BITS: int = 16
WINDOW_VALUES: int = 1 << WINDOW_BITS          # number of possible window values (incl. 0)
WINDOW_BUCKETS: int = WINDOW_VALUES - 1        # number of non-zero buckets (b in [1, 2^w))
NUM_WINDOWS: int = 4                           # bottom 4 windows = bottom 64 bits used
SCALAR_BITS_USED: int = WINDOW_BITS * NUM_WINDOWS

# Approximate Jacobian-add cost (matches msm.MODMULS_PER_ADD).
MODMULS_PER_BUCKET_ADD: int = 16


# ----------------------------------------------------------------------
# Inputs container
# ----------------------------------------------------------------------

@dataclass
class PippengerInputs:
    """Ready-to-dispatch bucket-scatter test case.

    - ``scalars_u64`` shape ``(N, 4)`` uint64: little-endian limbs of s_i
      (256-bit, but only the bottom 64 bits are read by the kernel).
    - ``points_in_u64`` shape ``(N, 18)`` uint64: P_i in Jacobian Montgomery,
      six 64-bit limbs per coordinate, little-endian.
    - ``expected_buckets_aff_u64`` shape
      ``(NUM_WINDOWS, WINDOW_BUCKETS, 12)`` uint64: reference bucket sum in
      affine Montgomery form (X, Y per bucket).
    - ``expected_is_infinity`` shape ``(NUM_WINDOWS, WINDOW_BUCKETS)`` bool:
      True iff the bucket sum is the point at infinity.
    """
    curve: CurveParams
    distribution: str
    n_pairs: int
    scalars_u64: np.ndarray
    points_in_u64: np.ndarray
    expected_buckets_aff_u64: np.ndarray
    expected_is_infinity: np.ndarray
    a: int
    b: int
    seed: int


# ----------------------------------------------------------------------
# Scalar sampling
# ----------------------------------------------------------------------

def _sample_uniform_scalars(curve: CurveParams, n: int, seed: int) -> np.ndarray:
    """N scalars in ``[0, r)`` packed as (N, 4) little-endian uint64.

    Identical strategy to ``msm._random_scalars``: uniform draw out of
    ``[0, 2^256)`` with rejection of values >= r so the bottom-window
    histogram is flat.
    """
    rng = np.random.default_rng(seed)
    out = np.empty((n, 4), dtype=np.uint64)
    filled = 0
    r = curve.r
    while filled < n:
        chunk = max(1, (n - filled) * 2)
        raw = rng.integers(
            0, 1 << 64, size=(chunk, 4), dtype=np.uint64, endpoint=False,
        )
        for row in raw:
            if filled >= n:
                break
            v = ((int(row[3]) << 192) | (int(row[2]) << 128)
                 | (int(row[1]) << 64) | int(row[0]))
            if v < r:
                out[filled] = row
                filled += 1
    return out


def _sample_zipf_scalars(
    curve: CurveParams, n: int, seed: int, zipf_a: float,
) -> np.ndarray:
    """N scalars whose bottom 64 bits decompose into ``NUM_WINDOWS``
    16-bit window values, each independently sampled from a Zipf-``a``
    distribution truncated to ``[1, 2^w)``.

    The full 256-bit scalar's upper 192 bits are zero -- the kernel
    only inspects the bottom 64 bits, and zero-padding the rest keeps
    every drawn scalar strictly less than ``r`` for any reasonable
    curve.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros((n, 4), dtype=np.uint64)
    bottom = np.zeros(n, dtype=np.uint64)
    for k in range(NUM_WINDOWS):
        # numpy.random.zipf yields unbounded positive ints with mass
        # ~ 1/i^a; we map into [1, WINDOW_BUCKETS] via modular reduction
        # so the support matches the bucket index space without
        # rejection. The mapping preserves the heavy-tail shape (mass
        # at i=1 stays dominant) because the bulk of the distribution
        # is already in low ranks for any ``a > 1``.
        v = rng.zipf(zipf_a, size=n)
        v = ((v - 1) % WINDOW_BUCKETS) + 1     # [1, WINDOW_BUCKETS]
        bottom |= v.astype(np.uint64) << np.uint64(k * WINDOW_BITS)
    out[:, 0] = bottom
    return out


def _parse_distribution(name: str) -> tuple[str, float]:
    """``"uniform"`` -> ``("uniform", 0.0)``;
    ``"zipf"`` / ``"zipf-1.5"`` / ``"zipf-2.0"`` -> ``("zipf", a)``.
    """
    name = name.lower().strip()
    if name == "uniform":
        return "uniform", 0.0
    if name == "zipf":
        return "zipf", 1.5
    if name.startswith("zipf-"):
        return "zipf", float(name[len("zipf-"):])
    raise ValueError(f"unknown distribution: {name!r}")


# ----------------------------------------------------------------------
# Reference bucket computation
# ----------------------------------------------------------------------

def _batch_invert_mont(zs: list[int], m: Montgomery) -> list[int]:
    """Batched modular inversion via Montgomery's trick.

    Inputs are Montgomery-form non-zero residues; outputs are their
    Montgomery-form modular inverses. Cost: 3N modmuls + 1 inversion,
    versus N inversions for the naive path. At 256-bit ``q``, CPython
    ``pow(z, -1, q)`` is ~ 50 us and a modmul is ~ 5 us, so the
    batched path is ~ 3x faster at small N and ~ 10x faster at N >> 10.
    """
    n = len(zs)
    if n == 0:
        return []
    prefix = [zs[0]]
    for v in zs[1:]:
        prefix.append(m.mul(prefix[-1], v))
    inv_total = m.inv(prefix[-1])
    out: list[int] = [0] * n
    cur = inv_total
    for i in range(n - 1, 0, -1):
        out[i] = m.mul(cur, prefix[i - 1])
        cur = m.mul(cur, zs[i])
    out[0] = cur
    return out


def _bucket_sums_uncached(
    scalars_u64: np.ndarray,
    points_in_u64: np.ndarray,
    m: Montgomery,
) -> tuple[np.ndarray, np.ndarray]:
    """Naive iteration-by-pair reference. Returns
    ``(affine_buckets, is_infinity)`` where ``affine_buckets`` has
    shape ``(NUM_WINDOWS, WINDOW_BUCKETS, 12)`` uint64 storing
    ``(X_aff_mont, Y_aff_mont)`` per bucket and ``is_infinity`` has
    shape ``(NUM_WINDOWS, WINDOW_BUCKETS)`` bool.

    Empty buckets store ``(0, 0)`` X/Y limbs with ``is_infinity=True``.
    """
    n = int(scalars_u64.shape[0])
    # Bucket sums indexed by [k][b] with b in [0, WINDOW_VALUES). Slot
    # 0 is unused on output (we only emit b >= 1) but keeping it
    # simplifies the inner loop's index arithmetic.
    sums: list[list[JacMont]] = [
        [jac_infinity() for _ in range(WINDOW_VALUES)]
        for _ in range(NUM_WINDOWS)
    ]
    bottom_limb = scalars_u64[:, 0].astype(np.uint64)
    window_mask = np.uint64(WINDOW_BUCKETS)
    window_vals_per_k = [
        ((bottom_limb >> np.uint64(k * WINDOW_BITS)) & window_mask).astype(np.int64)
        for k in range(NUM_WINDOWS)
    ]
    # Per-pair: unpack to (X, Y, Z) Python ints once, then dispatch the
    # NUM_WINDOWS independent adds. The bigint conversion is the
    # bottleneck of the loop; doing it once per pair (rather than once
    # per (k, i) pair) gives a ~4x speedup at NUM_WINDOWS = 4.
    for i in range(n):
        p_x = limbs_to_int(points_in_u64[i, 0:N_LIMBS])
        p_y = limbs_to_int(points_in_u64[i, N_LIMBS:2 * N_LIMBS])
        p_z = limbs_to_int(points_in_u64[i, 2 * N_LIMBS:3 * N_LIMBS])
        pi = JacMont(X=p_x, Y=p_y, Z=p_z)
        for k in range(NUM_WINDOWS):
            b = int(window_vals_per_k[k][i])
            if b == 0:
                continue
            sums[k][b] = jac_add(sums[k][b], pi, m)

    # Normalize to affine via Montgomery's trick: collect all non-zero
    # Z values across (k, b), batch invert, then derive (X*Z^-2, Y*Z^-3)
    # per bucket.
    affine = np.zeros(
        (NUM_WINDOWS, WINDOW_BUCKETS, 2 * N_LIMBS), dtype=np.uint64,
    )
    is_inf = np.zeros((NUM_WINDOWS, WINDOW_BUCKETS), dtype=bool)
    non_inf_items: list[tuple[int, int, JacMont]] = []
    for k in range(NUM_WINDOWS):
        for b in range(1, WINDOW_VALUES):
            ji = sums[k][b]
            if ji.Z == 0:
                is_inf[k][b - 1] = True
            else:
                non_inf_items.append((k, b, ji))
    if non_inf_items:
        zs = [it[2].Z for it in non_inf_items]
        z_invs = _batch_invert_mont(zs, m)
        for (k, b, ji), z_inv in zip(non_inf_items, z_invs):
            z_inv_sq = m.mul(z_inv, z_inv)
            z_inv_cu = m.mul(z_inv_sq, z_inv)
            x_aff = m.mul(ji.X, z_inv_sq)
            y_aff = m.mul(ji.Y, z_inv_cu)
            affine[k][b - 1][0:N_LIMBS] = int_to_limbs(x_aff, N_LIMBS)
            affine[k][b - 1][N_LIMBS:2 * N_LIMBS] = int_to_limbs(y_aff, N_LIMBS)
    return affine, is_inf


# ----------------------------------------------------------------------
# Disk + memory cache
# ----------------------------------------------------------------------

_CACHE_VERSION = "v1"
_MEM_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _cache_dir() -> Path:
    root = Path(os.environ.get(
        "METAL_ZK_CACHE", "~/.cache/metal-zk",
    )).expanduser()
    d = root / "pippenger"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(
    curve: CurveParams, n: int, a: int, b: int, distribution: str,
) -> str:
    h = hashlib.sha256()
    h.update(_CACHE_VERSION.encode())
    h.update(curve.name.encode())
    h.update(distribution.encode())
    h.update(np.uint64(n).tobytes())
    h.update(np.uint32(NUM_WINDOWS).tobytes())
    h.update(np.uint32(WINDOW_BITS).tobytes())
    h.update(a.to_bytes(64, "little"))
    h.update(b.to_bytes(64, "little"))
    return h.hexdigest()[:16]


def _bucket_sums_cached(
    curve: CurveParams,
    n: int,
    a: int,
    b: int,
    distribution: str,
    scalars_u64: np.ndarray,
    points_in_u64: np.ndarray,
    m: Montgomery,
) -> tuple[np.ndarray, np.ndarray]:
    key = _cache_key(curve, n, a, b, distribution)
    mem_key = f"buckets_{key}"
    if mem_key in _MEM_CACHE:
        return _MEM_CACHE[mem_key]
    path = _cache_dir() / f"{curve.name}_{distribution}_N{n}_{key}.npz"
    if path.exists():
        npz = np.load(path)
        aff = np.asarray(npz["affine"], dtype=np.uint64)
        inf = np.asarray(npz["is_inf"], dtype=bool)
        expected_shape = (NUM_WINDOWS, WINDOW_BUCKETS, 2 * N_LIMBS)
        if aff.shape == expected_shape and inf.shape == expected_shape[:2]:
            _MEM_CACHE[mem_key] = (aff, inf)
            return aff, inf
    aff, inf = _bucket_sums_uncached(scalars_u64, points_in_u64, m)
    # ``np.savez_compressed`` silently appends ``.npz`` to string paths
    # that don't already end in it, so we hand it an open file handle
    # to write directly to a temp path with our chosen suffix and then
    # atomically rename into place.
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, affine=aff, is_inf=inf)
    os.replace(tmp, path)
    _MEM_CACHE[mem_key] = (aff, inf)
    return aff, inf


# ----------------------------------------------------------------------
# Top-level test input builder
# ----------------------------------------------------------------------

def gen_inputs(
    curve: CurveParams,
    n: int,
    seed: int,
    distribution: str = "uniform",
) -> PippengerInputs:
    """Build a ready-to-dispatch bucket-scatter test case.

    Deterministic on ``(curve, n, seed, distribution)``. The expensive
    parts (per-pair point precompute + per-bucket reference) are
    cached on disk.
    """
    dist_kind, zipf_a = _parse_distribution(distribution)
    m = Montgomery(curve.q, N_LIMBS)
    a, b = _seed_derived_ab(curve, seed)
    # Reuse the Z1 point-precomputation cache so we don't duplicate
    # 144 MB of P_i = (a + b*i) * G points on disk per curve.
    points_in_u64 = precompute_points_cached(curve, n, a, b, m)
    if dist_kind == "uniform":
        scalars_u64 = _sample_uniform_scalars(curve, n, seed)
    elif dist_kind == "zipf":
        scalars_u64 = _sample_zipf_scalars(curve, n, seed, zipf_a)
    else:
        raise AssertionError("unreachable")
    affine, is_inf = _bucket_sums_cached(
        curve, n, a, b, distribution, scalars_u64, points_in_u64, m,
    )
    return PippengerInputs(
        curve=curve, distribution=distribution, n_pairs=n,
        scalars_u64=scalars_u64, points_in_u64=points_in_u64,
        expected_buckets_aff_u64=affine, expected_is_infinity=is_inf,
        a=a, b=b, seed=seed,
    )


# ----------------------------------------------------------------------
# Output normalization (GPU side)
# ----------------------------------------------------------------------

def normalize_gpu_buckets(
    buckets_u64: np.ndarray,
    m: Montgomery,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Normalize the GPU-side bucket buffer to affine Montgomery form.

    Input shape: ``(NUM_WINDOWS, WINDOW_BUCKETS, 18)`` uint64 -- one
    Jacobian Montgomery point per bucket, ``(X, Y, Z)`` each six
    little-endian 64-bit limbs.

    Returns ``(affine, is_inf, non_canonical_count)``:
      - ``affine``: ``(NUM_WINDOWS, WINDOW_BUCKETS, 12)`` uint64
      - ``is_inf``: ``(NUM_WINDOWS, WINDOW_BUCKETS)`` bool
      - ``non_canonical_count``: number of (k, b) tuples where any of
        ``X``, ``Y``, ``Z`` had a non-canonical limb (value >= q). Per
        ``montgomery_msm``'s convention these count as mismatches.

    Non-canonical X/Y/Z are still normalized (reduced mod q) so the
    affine values are well-defined; the residue-class comparison
    against the reference is still meaningful but the bucket already
    counts as a mismatch.
    """
    q = m.q
    affine = np.zeros(
        (NUM_WINDOWS, WINDOW_BUCKETS, 2 * N_LIMBS), dtype=np.uint64,
    )
    is_inf = np.zeros((NUM_WINDOWS, WINDOW_BUCKETS), dtype=bool)
    non_canonical = 0
    non_inf_items: list[tuple[int, int, int, int, int]] = []
    for k in range(NUM_WINDOWS):
        for b1 in range(WINDOW_BUCKETS):
            base = buckets_u64[k, b1]
            x = limbs_to_int(base[0:N_LIMBS])
            y = limbs_to_int(base[N_LIMBS:2 * N_LIMBS])
            z = limbs_to_int(base[2 * N_LIMBS:3 * N_LIMBS])
            if x >= q or y >= q or z >= q:
                non_canonical += 1
                x %= q
                y %= q
                z %= q
            if z == 0:
                is_inf[k][b1] = True
            else:
                non_inf_items.append((k, b1, x, y, z))
    if non_inf_items:
        zs = [it[4] for it in non_inf_items]
        z_invs = _batch_invert_mont(zs, m)
        for (k, b1, x, y, _z), z_inv in zip(non_inf_items, z_invs):
            z_inv_sq = m.mul(z_inv, z_inv)
            z_inv_cu = m.mul(z_inv_sq, z_inv)
            x_aff = m.mul(x, z_inv_sq)
            y_aff = m.mul(y, z_inv_cu)
            affine[k][b1][0:N_LIMBS] = int_to_limbs(x_aff, N_LIMBS)
            affine[k][b1][N_LIMBS:2 * N_LIMBS] = int_to_limbs(y_aff, N_LIMBS)
    return affine, is_inf, non_canonical


__all__ = [
    "WINDOW_BITS",
    "WINDOW_VALUES",
    "WINDOW_BUCKETS",
    "NUM_WINDOWS",
    "SCALAR_BITS_USED",
    "MODMULS_PER_BUCKET_ADD",
    "PippengerInputs",
    "gen_inputs",
    "normalize_gpu_buckets",
]
