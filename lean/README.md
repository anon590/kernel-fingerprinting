# Lean 4 verification of Appendix B (`app:theory`)

**Paper:** `workshop_paper/main.tex`, Appendix B — "Evaluation reuse with
instance fingerprinting: a self-contained account".
**Project:** `AppBCheck/` (library `AppB`, 9 modules, ~1,900 lines).
**Toolchain:** Lean 4 `leanprover/lean4:v4.30.0-rc2` + Mathlib `5450b53e5ddc`.
**Status:** `lake build` succeeds; all **46** exported theorems pass the
axiom audit (`AxiomCheck.lean`): each depends only on
`propext, Classical.choice, Quot.sound` — no `sorry`, no extra axioms.

```bash
cd lean/AppBCheck
lake build                      # mathlib is cached in .lake/ (gitignored)
lake env lean AxiomCheck.lean   # axiom audit over all 46 theorems
```

Every quantitative step of Appendix B — moment identities, constants,
exponents, union-bound accounting, the attack construction, and the
enumerability converse (§B.7, `app:enum`) — is formalized and
machine-checked. The only external ingredients are two classical
inequalities, entering as hypotheses exactly where the paper cites them:
one-sided Hoeffding (per tree node, in Theorem B.3) and McDiarmid (in
Theorem B.6(c)). One standard bridge the paper uses tacitly in B.6(b) is
*eliminated*: `PairMoments.lean` proves the bound directly on the pool
space, with no probabilistic hypotheses at all. The enumerability layer
adds no new external ingredient: its weighted Cauchy–Schwarz step is proved
from first principles in `WeightedPool.lean`.

## Claim-by-claim map (Appendix B → Lean)

### Definition B.2 (`def:adaptive`) — leakage and node count

| Paper claim | Lean theorem | Status |
|---|---|---|
| `M = ∑_{t≤T} ∏_{i<t}\|𝔽_i\| ≤ 2^k` whenever every `\|𝔽_i\| ≥ 2`, with `k = ∑_{t≤T} log₂\|𝔽_t\|` | `AppB.node_count_le_total`, `AppB.node_count_le_two_pow_total` (`NodeCount.lean`) | **fully proved** |
| geometric-sum engine | `AppB.node_count_le` | **fully proved** (induction) |

### Theorem B.3 (`thm:upper`) — validity under bounded-leakage reuse

| Paper step | Lean theorem | Status |
|---|---|---|
| feedback tree: "including the initial incumbent, at most `M+1 ≤ 2^{k+1}` candidates" | `AppB.tree_count_le_total` (`NodeCount.lean`) | **fully proved** |
| per-node one-sided Hoeffding at radius `r = B√(((k+1)ln2+ln(1/β))/(2N))` lands exactly on level `β/2^{k+1}`: `exp(−2Nr²/B²) = β/2^{k+1}` | `AppB.hoeffding_level` (`TheoremUpper.lean`) | **fully proved** (exact equality) |
| union bound over the `≤ 2^{k+1}` nodes at fixed level `β/2^{k+1}` ⟹ realized failure `≤ β`, for every proposer and any data-dependent prefix map | `AppB.thm_upper_fixed_level`, `AppB.thm_upper`, `AppB.union_bound_uniform` | **fully proved** |
| "condition on the proposer's private randomness ω / integrate over ω" | `AppB.prob_le_of_condExp_le` (`UnionBound.lean`) | **fully proved** (tower property via `integral_condExp`) |
| Hoeffding's inequality itself (fixed `c_v`, i.i.d. `D`) | hypothesis `hEB` | cited classical primitive |
| Remark (bit budgets): full-precision feedback (`k = 64 × configs × rounds`) makes the radius exceed `B` — concrete instance `k = 1920`, `N = 3` | `AppB.remark_bit_budgets_vacuous` | **fully proved** (`r ≈ 14.9·B` at `β = 0.05`) |

### Assumption B.4 (`ass:richness`)

Not a claim; it is *modeled* in the Theorem B.6 files: the payoff
`Y(c_v, ξ) = μ₀ + σ·v(φ(ξ))` with `φ(ξ) ~ Unif[m]` appears as the fiber
decomposition in `AppB.empirical_mean_fiber` / `AppB.attack_population_mean`,
and the uniform pool measure on `[m]^N` is used directly in
`PairMoments.lean`.

### Lemma B.5 (`lem:mad`) — pair-difference anti-concentration

Fully proved with **no probabilistic primitives** (expectations are explicit
binomial sums), in `LemmaMad.lean`:

| Paper step | Lean theorem |
|---|---|
| `EΔ² = s/4` | `AppB.sum_choose_sq` |
| `EΔ⁴ = (s/4)(1+3(s−2)/4) ≤ 3(s/4)²` | `AppB.sum_choose_quart` |
| Cauchy–Schwarz `(EΔ²)² ≤ E\|Δ\|·E\|Δ\|³` | `AppB.cs_step` |
| Jensen (`x↦x^{3/4}` concave) `E\|Δ\|³ ≤ (EΔ⁴)^{3/4}` | `AppB.jensen_step` |
| combine, `√s/(2·3^{3/4}) ≥ √s/5` (i.e. `432 ≤ 625`) | `AppB.lemma_mad` |

(The moment identities are proved by induction via the Pascal-rule
reindexing `AppB.pascal_sum`.)

### Theorem B.6 (`thm:lowerapp`) — one-bit fingerprinting attack; tightness

(`Chernoff.lean`, `TheoremLower.lean`, `PairMoments.lean`)

| Paper step | Lean theorem | Status |
|---|---|---|
| (a) `∑_b v*(b) = 0` ⟹ `J(ĉ) = μ₀` exactly, for every bit realization | `AppB.attack_population_mean` | **fully proved** |
| `Ĵ_D(c_v) − Ĵ_D(c_0) = (σ/N)(n_{2t−1} − n_{2t})`; returned bit = sign | `AppB.empirical_mean_fiber` | **fully proved** |
| `Z = (σ/N) ∑_t \|n_{2t−1} − n_{2t}\|` (sign trick, ties covered) | `AppB.attack_inflation` | **fully proved** |
| Chernoff lower tail `P(s ≤ μ_s/2) ≤ e^{−μ_s/8}` | `AppB.binomial_lower_tail` | **fully proved from scratch** (tilting; in fact the stronger exponent `−(1−ln2)μ/2`) |
| `E[\|n₁−n₂\| ∣ s] = 2E\|Δ\| ≥ (2/5)√s` | `AppB.halfAbsMean_ge` | **fully proved** (via Lemma B.5) |
| (b) per pair: `≥ (2/5)√(N/m)(1−e^{−1}) ≥ ¼√(N/m)` using `μ_s = 2N/m ≥ 8` (`m ≤ N/4`) and `1−e^{−1} ≥ 5/8` | `AppB.pair_diff_lower` | **fully proved** (the paper's route) |
| (b) assembly: `EZ ≥ (σ/8)√(m/N)` | `AppB.attack_inflation_mean`, `AppB.thm_lower_b` | **fully proved** given the conditional decomposition `n_{2t−1} ∣ s ~ Bin(s,1/2)` (hypothesis `hEd`) |
| (b) **bridge-free**: same bound proved directly on the uniform pool space `[m]^N` via exact moments `ES² = 2N/m`, `ES⁴ = 12N(N−1)/m² + 2N/m` | `AppB.pool_S2`, `AppB.pool_S4`, `AppB.mad_lower`, `AppB.pair_diff_lower_uniform`, `AppB.thm_lower_b_uniform` | **fully proved end-to-end, no probabilistic hypotheses** |
| (c) one instance moves `Z` by `≤ 2σ/N` (incl. both-cells-in-one-pair) | `AppB.cnt_one_instance_diff`, `AppB.Z_bounded_diff` | **fully proved** |
| (c) McDiarmid exponent `2u²/(N(2σ/N)²) = m/512` at `u = (σ/16)√(m/N)` | `AppB.mcdiarmid_exponent` | **fully proved** (exact; the paper's `≤ e^{−m/512}` is equality) |
| (c) `u ≤ EZ/2` and `Z ≥ EZ − u` ⟹ `Z ≥ (σ/16)√(m/N)` | `AppB.deviation_conclusion` | **fully proved** |
| McDiarmid's inequality itself | — | cited classical primitive (Mathlib has Azuma–Hoeffding, not McDiarmid) |
| Certificates: `L̂ ≥ Ĵ_D(ĉ) − ε`, `ε ≤ (σ/32)√(m/N)` ⟹ `L̂ > μ₀ = J(ĉ)` | `AppB.thm_lower_disappointment` | **fully proved** |

The Remark (constants and simulation) — `EZ ≈ 0.56σ√(m/N)` at
`(N,m) ∈ {(48,12),(400,100),(1000,250)}` — is empirical;
`workshop_paper/sim_attack.py` reproduces it, not Lean.

### §B.7 (`app:enum`) — the converse: enumerability is exactly richness

Weighted-pool machinery (`WeightedPool.lean`) over an arbitrary probe law
`w : Ξ → ℝ` (`w ≥ 0`, `∑ w = 1`), then the dictionary (`Enumerability.lean`).
The load-bearing distinction (validated in the companion theory project's
`sim/` E1′): diffuseness alone does *not* defeat the attack — Assumption B.4
needs only `φ#w = Unif[m]`, which a *coarse* (quantile-lumping) fingerprint
realizes on any fine law; what grade A–C programs realize are *identity*
fingerprints that name `≤ L` atoms and cannot split the remainder.

| Paper claim | Lean theorem | Status |
|---|---|---|
| weighted pool expectation algebra; `E[n_b − N w_b] = 0`; **exact** `E[(n_b − N w_b)²] = N w_b(1−w_b)`; weighted Cauchy–Schwarz `E\|g\| ≤ √(E g²)` | `AppB.pexp_one`, `AppB.cmom1`, `AppB.cmom2`, `AppB.pexp_abs_le_sqrt` (`WeightedPool.lean`) | **fully proved from first principles** |
| product-law pushforward: a statistic of `φ∘ω` has the same `w`-pool expectation as on the cell pushforward law | `AppB.pexp_pushforward` | **fully proved** (`Fin.cons` induction) |
| **Thm B.8** (dictionary, forward): enumerable at resolution `m` ⟹ richness `R(m,σ)` ⟹ the B.6 attack achieves `EZ ≥ (σ/8)√(m/N)` on the `w`-pool | `AppB.enumerableId_richness`, `AppB.richness_attack_fires` | **fully proved** (reduces to `thm_lower_b_uniform`) |
| **Thm B.9** (no witness): `μ_max`-diffuse + identity class, `L·μ_max < ½` ⟹ no identity fingerprint has uniform pushforward on `m ≥ 2` cells (default cell carries `> ½`) | `AppB.diffuse_not_enumerableId` | **fully proved** (pigeonhole) |
| **Thm B.10** (starvation): any pool-adaptive identity-class selection has `E[Ĵ_D − J] ≤ 2σL√(μ_max/N)`; total centered mass `∑_b(n_b − N w_b) = 0` | `AppB.t1_diffuse_upper`, `AppB.cntC_total` | **fully proved** (constants drop out; `cmom2` + Cauchy–Schwarz per cell) |
| **Thm B.10** (two-sided): at `L = m = 2T`, `512·T·μ_max ≤ 1` ⟹ diffuse value `≤ (σ/8)√(2T/N)`, the enumerable guarantee — same `512` as B.6(c) | `AppB.diffuse_starves` | **fully proved** |

The lumping caveat is encoded in the definitions, not assumed: `RichnessB4`
is about the predicate *class* (`IdentityFingerprint`/`IdentityClass`), so
range/statistic predicates (grade D) are deliberately out of scope and
recover the full envelope, as the companion simulation shows.

## Notes

1. **No errors found.** Every constant in Appendix B — `σ/8`, `σ/16`,
   `σ/32`, `e^{−m/512}`, `√s/5`, `e^{−μ_s/8}`, the `m ≤ N/4` proviso, the
   `2σ/N` bounded difference — matches the verified statements exactly.
2. Theorem B.3's radius is slack-free: the one-sided Hoeffding tail at the
   displayed radius equals the per-node level `β/2^{k+1}` exactly
   (`hoeffding_level` is an equality, not a bound).
3. Two razor-thin but valid constants are load-bearing:
   `1 − e^{−1} ≥ 5/8` (i.e. `e ≥ 8/3`) in Theorem B.6(b), and
   `2·3^{3/4} ≤ 5` (i.e. `432 ≤ 625`) in Lemma B.5.

## Layout

```
lean/AppBCheck/
├── lakefile.toml, lean-toolchain, lake-manifest.json
├── AppB.lean             # library root, module map
├── AxiomCheck.lean       # #print axioms for all 46 exported theorems
└── AppB/
    ├── NodeCount.lean      # Def. B.2 node count M ≤ 2^k; tree count M+1 ≤ 2^{k+1}
    ├── UnionBound.lean     # union bound + tower-property step
    ├── TheoremUpper.lean   # Thm. B.3 assembly, radius arithmetic, Remark (bit budgets)
    ├── LemmaMad.lean       # Lemma B.5, fully proved
    ├── Chernoff.lean       # binomial lower tail, fully proved
    ├── TheoremLower.lean   # Thm. B.6 (a),(b),(c) + certificates
    ├── PairMoments.lean    # Thm. B.6(b) bridge-free on the pool space
    ├── WeightedPool.lean   # §B.7 weighted pool space: product laws, centered moments, Cauchy–Schwarz
    └── Enumerability.lean  # Thm. B.8–B.10: enumerable ⇒ richness ⇒ attack; diffuse ⇒ starvation
```
