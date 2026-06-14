/-
# L1: Enumerability, diffuseness, and the fingerprinting dictionary (T1)

This file makes the fingerprinting paper's design rule 2 ("probes retain
validity only on undisclosed *and* non-enumerable axes") a theorem-level
object, in the finitary style of the existing development.  The probe law
is a weight `w : Ξ → ℝ` on a finite configuration axis (`w ≥ 0`, `∑ w = 1`);
the pool measure is the product `wprod` of `AppB.WeightedPool`.

The load-bearing distinction (validated in simulation, `sim/` E1′): a
diffuse law does NOT by itself defeat the one-bit attack — Assumption B.4
only needs `φ#w = Unif[m]`, and a *coarse* fingerprint (equal-mass lumps)
realizes that on any fine law.  What grade A–C programs can realize are
*identity* fingerprints: equality/hash predicates that NAME at most `L`
atoms and cannot distinguish the unnamed remainder.  Hence:

* `RichnessB4 w T`      — some fingerprint pushes `w` to `Unif[2T]`
                          (Assumption B.4, as modeled in `TheoremLower`);
* `IdentityFingerprint φ S` — `φ` is constant off the named set `S`;
* `EnumerableId w T L`  — richness realizable by an identity fingerprint
                          with at most `L` names: the probe axis is
                          *enumerable* in the paper's sense;
* `Diffuse w μmax`      — every atom carries at most `μmax` mass;
* `IdentityClass v S`   — a candidate's payoff offset pattern: `|v| ≤ 1`,
                          constant off `S` (grades A–C of the taxonomy).

Theorems:

* `enumerableId_richness` — Enumerable ⇒ R(2T,σ): the dictionary's easy
  direction (the paper's design rule, read forward).
* `richness_attack_fires` — under `RichnessB4`, the B.6 attack value on
  the *w*-pool equals its value on the uniform cell pool and inherits
  `E Z ≥ (σ/8)√(m/N)` (`thm_lower_b_uniform`); with
  `enumerableId_richness`, enumerable axes are attackable: **the
  lower-bound witness exists**.
* `diffuse_not_enumerableId` — **T1, qualitative**: on a `μmax`-diffuse
  axis, an identity fingerprint with `L·μmax < 1/2` names cannot push `w`
  to uniform (pigeonhole on the default cell): the witness *cannot* exist.
* `t1_diffuse_upper` — **T1, quantitative**: for ANY pool-adaptive
  candidate values supported on `≤ L` named atoms,
  `E[Ĵ_D - J] ≤ 2σL√(μmax/N)` — the identity-attack channel's value
  vanishes with the max atom mass (matches the `sim/` E1 collapse law,
  constant ≈2.5× conservative).
* `diffuse_starves` — at the B.6 budget `L = 2T`: if `512·T·μmax ≤ 1`
  the diffuse value is below the enumerable guarantee `(σ/8)√(2T/N)` —
  the two-sided dictionary, with the same 512 as Theorem B.6(c).

NOT claimed (honest scoping, paper §4): nothing here bounds candidates
that condition on *statistics* of the instance (range/threshold
predicates — the grade-D channel); those can realize coarse lumps and
recover the full envelope, as the simulation shows.
-/
import AppB.WeightedPool
import AppB.TheoremLower
import AppB.PairMoments

open Finset

set_option linter.unusedSectionVars false

namespace AppB

variable {Ξ : Type*} [Fintype Ξ] [DecidableEq Ξ]

/-! ## Definitions -/

/-- Pushforward mass of the probe law under a fingerprint. -/
noncomputable def pushMass (w : Ξ → ℝ) {C : Type*} [DecidableEq C]
    (φ : Ξ → C) (c : C) : ℝ :=
  ∑ x ∈ Finset.univ.filter (fun x => φ x = c), w x

/-- Assumption B.4's richness, as modeled in `AppB.TheoremLower`: some
fingerprint pushes the probe law to the uniform law on `2T` cells (the
`±σ` candidate family is then automatic in the model). -/
def RichnessB4 (w : Ξ → ℝ) (T : ℕ) : Prop :=
  ∃ φ : Ξ → Fin T × Bool, ∀ c, pushMass w φ c = 1 / (2 * T)

/-- An identity (equality/hash-predicate) fingerprint: it can *name* the
atoms of `S` but cannot distinguish unnamed atoms from one another. -/
def IdentityFingerprint {C : Type*} (φ : Ξ → C) (S : Finset Ξ) : Prop :=
  ∀ x ∉ S, ∀ y ∉ S, φ x = φ y

/-- The probe axis is enumerable at pair-resolution `T` within naming
budget `L`: richness is realizable by an identity fingerprint. -/
def EnumerableId (w : Ξ → ℝ) (T L : ℕ) : Prop :=
  ∃ (φ : Ξ → Fin T × Bool) (S : Finset Ξ), IdentityFingerprint φ S
    ∧ S.card ≤ L ∧ ∀ c, pushMass w φ c = 1 / (2 * T)

/-- The probe law is `μmax`-diffuse: no single nameable identity carries
more than `μmax` mass. -/
def Diffuse (w : Ξ → ℝ) (μmax : ℝ) : Prop :=
  ∀ x, w x ≤ μmax

/-- Grade A–C candidate offset patterns: bounded payload, constant off the
named set `S` (an unnamed instance cannot be distinguished, so it takes
the default arm). -/
def IdentityClass (v : Ξ → ℝ) (S : Finset Ξ) : Prop :=
  (∀ b, |v b| ≤ 1) ∧ ∃ v₀, ∀ b ∉ S, v b = v₀

/-! ## The dictionary, forward: enumerable ⇒ richness ⇒ the attack fires -/

theorem enumerableId_richness (w : Ξ → ℝ) (T L : ℕ)
    (h : EnumerableId w T L) : RichnessB4 w T := by
  obtain ⟨φ, S, _, _, hu⟩ := h
  exact ⟨φ, hu⟩

/-- Pushforward of the product pool law: a statistic of the
fingerprinted pool `φ ∘ ω` has the same expectation under `w` as under
the pushforward law on cells. -/
theorem pexp_pushforward (w : Ξ → ℝ) {C : Type*} [Fintype C] [DecidableEq C]
    (φ : Ξ → C) (N : ℕ) (G : (Fin N → C) → ℝ) :
    pexp w N (fun ω => G (φ ∘ ω)) = pexp (pushMass w φ) N G := by
  induction N with
  | zero =>
    unfold pexp wprod
    simp only [Finset.univ_unique, Finset.sum_singleton, Fin.prod_univ_zero,
      one_mul]
    exact congrArg G (Subsingleton.elim _ _)
  | succ N ih =>
    rw [pexp_succ, pexp_succ]
    have hcons : ∀ (x : Ξ) (ω : Fin N → Ξ),
        φ ∘ Fin.cons x ω = Fin.cons (φ x) (φ ∘ ω) := by
      intro x ω
      funext i
      refine Fin.cases ?_ ?_ i <;> simp
    calc ∑ x : Ξ, w x * pexp w N (fun ω => G (φ ∘ Fin.cons x ω))
        = ∑ x : Ξ, w x * pexp w N (fun ω => G (Fin.cons (φ x) (φ ∘ ω))) := by
          refine Finset.sum_congr rfl fun x _ => ?_
          congr 1
          exact congrArg (pexp w N) (funext fun ω => by rw [hcons x ω])
      _ = ∑ x : Ξ, w x
            * pexp (pushMass w φ) N (fun ω' => G (Fin.cons (φ x) ω')) := by
          refine Finset.sum_congr rfl fun x _ => ?_
          rw [ih (fun ω' => G (Fin.cons (φ x) ω'))]
      _ = ∑ c : C, pushMass w φ c
            * pexp (pushMass w φ) N (fun ω' => G (Fin.cons c ω')) := by
          rw [← Finset.sum_fiberwise Finset.univ φ
            (fun x => w x
              * pexp (pushMass w φ) N (fun ω' => G (Fin.cons (φ x) ω')))]
          refine Finset.sum_congr rfl fun c _ => ?_
          unfold pushMass
          rw [Finset.sum_mul]
          refine Finset.sum_congr rfl fun x hx => ?_
          rw [Finset.mem_filter] at hx
          rw [hx.2]

/-- **Richness fires the attack** (dictionary, forward; Theorem B.6(b) on
the probe law): under `RichnessB4 w T`, the B.6 attack run through `φ` on
`w`-pools has `E Z ≥ (σ/8)√(m/N)`, `m = 2T`. -/
theorem richness_attack_fires (w : Ξ → ℝ) (T N m : ℕ) (σ : ℝ) (hσ : 0 ≤ σ)
    (hmT : m = 2 * T) (hT : 1 ≤ T) (hm : 2 ≤ m) (hNm : 4 * m ≤ N)
    (φ : Ξ → Fin T × Bool) (hu : ∀ c, pushMass w φ c = 1 / (2 * T)) :
    σ / 8 * Real.sqrt ((m : ℝ) / N)
      ≤ pexp w N (fun ω => σ / N
          * ∑ t : Fin T, |cnt (φ ∘ ω) (t, true) - cnt (φ ∘ ω) (t, false)|) := by
  have hm0 : (0 : ℝ) < m := by
    have : (0 : ℕ) < m := by omega
    exact_mod_cast this
  -- pushforward to the uniform cell pool
  rw [pexp_pushforward w φ N (fun ω' => σ / N
    * ∑ t : Fin T, |cnt ω' (t, true) - cnt ω' (t, false)|)]
  -- the pushforward law is the constant 1/m, so pexp = (∑ ·)/m^N
  have hpush : ∀ c, pushMass w φ c = 1 / (m : ℝ) := by
    intro c
    rw [hu c, hmT]
    push_cast
    ring_nf
  have hwprod : ∀ ω' : Fin N → Fin T × Bool,
      wprod (pushMass w φ) ω' = 1 / (m : ℝ) ^ N := by
    intro ω'
    unfold wprod
    calc ∏ i, pushMass w φ (ω' i) = ∏ _i : Fin N, 1 / (m : ℝ) :=
          Finset.prod_congr rfl fun i _ => hpush (ω' i)
      _ = (1 / (m : ℝ)) ^ N := by rw [Finset.prod_const, Finset.card_univ,
          Fintype.card_fin]
      _ = 1 / (m : ℝ) ^ N := by rw [one_div, one_div, inv_pow]
  have hexp : pexp (pushMass w φ) N (fun ω' => σ / N
      * ∑ t : Fin T, |cnt ω' (t, true) - cnt ω' (t, false)|)
      = (∑ ω' : Fin N → Fin T × Bool, σ / N
          * ∑ t : Fin T, |cnt ω' (t, true) - cnt ω' (t, false)|)
        / ((m : ℝ) ^ N) := by
    unfold pexp
    rw [Finset.sum_div]
    refine Finset.sum_congr rfl fun ω' _ => ?_
    rw [hwprod ω']
    ring
  rw [hexp]
  exact thm_lower_b_uniform T N m hmT hT hm hNm σ hσ

/-! ## The dictionary, backward: diffuse ⇒ no identity witness (T1) -/

/-- **T1, qualitative.**  On a `μmax`-diffuse axis, an identity
fingerprint with naming budget `L·μmax < 1/2` cannot realize Assumption
B.4: the unnamed mass `≥ 1/2` lands in a single default cell, which a
uniform pushforward (`≤ 1/2` per cell) cannot accommodate.  The
lower-bound witness of Theorem B.6 does not exist on this axis. -/
theorem diffuse_not_enumerableId (w : Ξ → ℝ) (hw : ∀ x, 0 ≤ w x)
    (hw1 : ∑ x : Ξ, w x = 1) (μmax : ℝ) (hdiff : Diffuse w μmax)
    (T L : ℕ) (hT : 1 ≤ T) (hL : (L : ℝ) * μmax < 1 / 2) :
    ¬ EnumerableId w T L := by
  rintro ⟨φ, S, hid, hcard, hu⟩
  have hμ0 : 0 ≤ μmax := by
    by_contra hneg
    rw [not_le] at hneg
    -- some atom exists since total mass is 1 > 0
    have : ∑ x : Ξ, w x ≤ 0 := by
      apply Finset.sum_nonpos
      intro x _
      linarith [hdiff x]
    linarith
  -- mass named is at most L μmax < 1/2
  have hSmass : ∑ x ∈ S, w x ≤ (L : ℝ) * μmax := by
    calc ∑ x ∈ S, w x ≤ ∑ _x ∈ S, μmax :=
          Finset.sum_le_sum fun x _ => hdiff x
      _ = (S.card : ℝ) * μmax := by rw [Finset.sum_const, nsmul_eq_mul]
      _ ≤ (L : ℝ) * μmax := by
          apply mul_le_mul_of_nonneg_right _ hμ0
          exact_mod_cast hcard
  -- the unnamed set is nonempty (else 1 ≤ L μmax < 1/2)
  have hcompl : ∑ x ∈ Sᶜ, w x = 1 - ∑ x ∈ S, w x := by
    have := Finset.sum_add_sum_compl S w
    linarith
  have hunnamed : (1 : ℝ) / 2 < ∑ x ∈ Sᶜ, w x := by
    rw [hcompl]
    linarith
  have hne : Sᶜ.Nonempty := by
    by_contra hempty
    rw [Finset.not_nonempty_iff_eq_empty] at hempty
    rw [hempty, Finset.sum_empty] at hunnamed
    linarith
  obtain ⟨x₀, hx₀⟩ := hne
  rw [Finset.mem_compl] at hx₀
  -- all unnamed mass lands in the default cell φ x₀
  have hsub : Sᶜ ⊆ Finset.univ.filter (fun x => φ x = φ x₀) := by
    intro x hx
    rw [Finset.mem_compl] at hx
    rw [Finset.mem_filter]
    exact ⟨Finset.mem_univ x, hid x hx x₀ hx₀⟩
  have hdefault : ∑ x ∈ Sᶜ, w x ≤ pushMass w φ (φ x₀) := by
    unfold pushMass
    exact Finset.sum_le_sum_of_subset_of_nonneg hsub fun x _ _ => hw x
  -- but the uniform pushforward gives the default cell mass ≤ 1/2
  have hT1 : (1 : ℝ) ≤ (T : ℝ) := by exact_mod_cast hT
  have hhalf : pushMass w φ (φ x₀) ≤ 1 / 2 := by
    rw [hu (φ x₀)]
    apply one_div_le_one_div_of_le (by norm_num : (0 : ℝ) < 2)
    linarith
  linarith

/-! ## T1, quantitative: the diffuse upper bound -/

/-- Total centered mass vanishes: `∑_b (n_b - N w_b) = 0`. -/
theorem cntC_total (w : Ξ → ℝ) (hw1 : ∑ x : Ξ, w x = 1) {N : ℕ}
    (ω : Fin N → Ξ) : ∑ b : Ξ, cntC w b ω = 0 := by
  unfold cntC
  rw [Finset.sum_sub_distrib]
  have hcnt : ∑ b : Ξ, cnt ω b = N := by
    unfold cnt
    rw [Finset.sum_comm]
    have hrow : ∀ i : Fin N, ∑ b : Ξ, (if ω i = b then (1 : ℝ) else 0) = 1 := by
      intro i
      rw [Finset.sum_ite_eq Finset.univ (ω i) (fun _ => (1 : ℝ))]
      simp
    rw [Finset.sum_congr rfl fun i _ => hrow i, Finset.sum_const,
      Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one]
  rw [hcnt, ← Finset.mul_sum, hw1, mul_one, sub_self]

/-- An `IdentityClass` pattern admits a default value of magnitude `≤ 1`
(when `S = univ` the off-`S` clause is vacuous, so `0` serves). -/
theorem identityClass_normalize (v : Ξ → ℝ) (S : Finset Ξ)
    (h : IdentityClass v S) :
    ∃ v₀, |v₀| ≤ 1 ∧ ∀ b ∉ S, v b = v₀ := by
  obtain ⟨hb1, v₀, hv₀⟩ := h
  by_cases hne : Sᶜ.Nonempty
  · obtain ⟨x, hx⟩ := hne
    rw [Finset.mem_compl] at hx
    exact ⟨v₀, by rw [← hv₀ x hx]; exact hb1 x, hv₀⟩
  · refine ⟨0, by norm_num, fun b hb => absurd ⟨b, Finset.mem_compl.mpr hb⟩ hne⟩

/-- **T1, quantitative (diffuse upper bound).**  For every pool-adaptive
choice of grade A–C candidate values on a fixed named support `S` of at
most `L` atoms — this includes every sign strategy decodable from
comparison bits, in particular the B.6 attack — the expected inflation
`E[Ĵ_D(ĉ) - J(ĉ)] = E[(σ/N) ∑_b vhat(b)(n_b - N w_b)]` satisfies

  `E ≤ 2 σ L √(μmax / N)`.

Constants drop out of the centered sum, each named cell fluctuates by at
most `√(N w_b(1-w_b)) ≤ √(N μmax)` (`cmom2` + Cauchy–Schwarz), and there
are at most `L` of them. -/
theorem t1_diffuse_upper (w : Ξ → ℝ) (hw : ∀ x, 0 ≤ w x)
    (hw1 : ∑ x : Ξ, w x = 1) (μmax : ℝ) (hdiff : Diffuse w μmax)
    (S : Finset Ξ) (L N : ℕ) (hS : S.card ≤ L) (hN : 1 ≤ N)
    (σ : ℝ) (hσ : 0 ≤ σ)
    (vhat : (Fin N → Ξ) → Ξ → ℝ) (hv : ∀ ω, IdentityClass (vhat ω) S) :
    pexp w N (fun ω => σ / N * ∑ b : Ξ, vhat ω b * cntC w b ω)
      ≤ 2 * σ * L * Real.sqrt (μmax / N) := by
  have hN0 : (0 : ℝ) < N := by exact_mod_cast hN
  have hμ0 : 0 ≤ μmax := by
    have hne : (Finset.univ : Finset Ξ).Nonempty := by
      by_contra hempty
      rw [Finset.not_nonempty_iff_eq_empty] at hempty
      rw [hempty, Finset.sum_empty] at hw1
      norm_num at hw1
    obtain ⟨x, _⟩ := hne
    exact le_trans (hw x) (hdiff x)
  -- pointwise: ∑_b vhat(b)·X_b = ∑_{b∈S} (vhat(b)-v₀)·X_b ≤ 2 ∑_{b∈S} |X_b|
  have hpt : ∀ ω : Fin N → Ξ,
      σ / N * ∑ b : Ξ, vhat ω b * cntC w b ω
        ≤ σ / N * ∑ b ∈ S, 2 * |cntC w b ω| := by
    intro ω
    obtain ⟨hb1, _, _⟩ := hv ω
    obtain ⟨v₀, hv₀le, hv₀⟩ := identityClass_normalize (vhat ω) S (hv ω)
    -- shift by the default value: the total centered mass is 0
    have hshift : ∑ b : Ξ, vhat ω b * cntC w b ω
        = ∑ b : Ξ, (vhat ω b - v₀) * cntC w b ω := by
      have hexpand : ∑ b : Ξ, (vhat ω b - v₀) * cntC w b ω
          = (∑ b : Ξ, vhat ω b * cntC w b ω)
            - v₀ * ∑ b : Ξ, cntC w b ω := by
        rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
        exact Finset.sum_congr rfl fun b _ => by ring
      rw [hexpand, cntC_total w hw1 ω, mul_zero, sub_zero]
    -- off-S terms vanish
    have hrestrict : ∑ b : Ξ, (vhat ω b - v₀) * cntC w b ω
        = ∑ b ∈ S, (vhat ω b - v₀) * cntC w b ω := by
      symm
      apply Finset.sum_subset (Finset.subset_univ S)
      intro b _ hbS
      rw [hv₀ b hbS, sub_self, zero_mul]
    -- per-term bound
    have hterm : ∀ b ∈ S, (vhat ω b - v₀) * cntC w b ω
        ≤ 2 * |cntC w b ω| := by
      intro b _
      have habs : |vhat ω b - v₀| ≤ 2 := by
        calc |vhat ω b - v₀| ≤ |vhat ω b| + |v₀| := abs_sub _ _
          _ ≤ 1 + 1 := add_le_add (hb1 b) hv₀le
          _ = 2 := by norm_num
      calc (vhat ω b - v₀) * cntC w b ω
          ≤ |(vhat ω b - v₀) * cntC w b ω| := le_abs_self _
        _ = |vhat ω b - v₀| * |cntC w b ω| := abs_mul _ _
        _ ≤ 2 * |cntC w b ω| :=
            mul_le_mul_of_nonneg_right habs (abs_nonneg _)
    rw [hshift, hrestrict]
    exact mul_le_mul_of_nonneg_left (Finset.sum_le_sum hterm) (by positivity)
  -- expectation chain
  have hcell : ∀ b ∈ S, pexp w N (fun ω => |cntC w b ω|)
      ≤ Real.sqrt ((N : ℝ) * μmax) := by
    intro b _
    calc pexp w N (fun ω => |cntC w b ω|)
        ≤ Real.sqrt (pexp w N (fun ω => cntC w b ω ^ 2)) :=
          pexp_abs_le_sqrt w hw hw1 N _
      _ = Real.sqrt ((N : ℝ) * (w b * (1 - w b))) := by rw [cmom2 w hw1]
      _ ≤ Real.sqrt ((N : ℝ) * μmax) := by
          apply Real.sqrt_le_sqrt
          have hwb1 : w b * (1 - w b) ≤ μmax := by
            have h1 : w b * (1 - w b) ≤ w b * 1 := by
              apply mul_le_mul_of_nonneg_left _ (hw b)
              linarith [hw b]
            calc w b * (1 - w b) ≤ w b * 1 := h1
              _ = w b := mul_one _
              _ ≤ μmax := hdiff b
          exact mul_le_mul_of_nonneg_left hwb1 hN0.le
  calc pexp w N (fun ω => σ / N * ∑ b : Ξ, vhat ω b * cntC w b ω)
      ≤ pexp w N (fun ω => σ / N * ∑ b ∈ S, 2 * |cntC w b ω|) :=
        pexp_mono w hw N _ _ hpt
    _ = σ / N * ∑ b ∈ S, 2 * pexp w N (fun ω => |cntC w b ω|) := by
        rw [pexp_smul, pexp_sum]
        congr 1
        exact Finset.sum_congr rfl fun b _ => pexp_smul w N 2 _
    _ ≤ σ / N * ∑ _b ∈ S, 2 * Real.sqrt ((N : ℝ) * μmax) := by
        apply mul_le_mul_of_nonneg_left _ (by positivity)
        exact Finset.sum_le_sum fun b hb =>
          mul_le_mul_of_nonneg_left (hcell b hb) (by norm_num)
    _ = σ / N * ((S.card : ℝ) * (2 * Real.sqrt ((N : ℝ) * μmax))) := by
        rw [Finset.sum_const, nsmul_eq_mul]
    _ ≤ σ / N * ((L : ℝ) * (2 * Real.sqrt ((N : ℝ) * μmax))) := by
        apply mul_le_mul_of_nonneg_left _ (by positivity)
        apply mul_le_mul_of_nonneg_right _ (by positivity)
        exact_mod_cast hS
    _ = 2 * σ * L * (Real.sqrt ((N : ℝ) * μmax) / N) := by ring
    _ = 2 * σ * L * Real.sqrt (μmax / N) := by
        congr 1
        rw [Real.sqrt_mul hN0.le, Real.sqrt_div hμ0 (N : ℝ)]
        rw [div_eq_div_iff hN0.ne' (Real.sqrt_pos.mpr hN0).ne']
        linear_combination Real.sqrt μmax * Real.mul_self_sqrt hN0.le

/-- **The dictionary, two-sided** (at the B.6 naming budget `L = 2T`):
on a `μmax`-diffuse axis with `512·T·μmax ≤ 1`, every identity-class
attack is worth at most the enumerable-case *guarantee* `(σ/8)√(2T/N)` —
the bound the attack is *assured to exceed* (in expectation) when the
axis is enumerable (`richness_attack_fires`).  Same `512` as Theorem
B.6(c). -/
theorem diffuse_starves (w : Ξ → ℝ) (hw : ∀ x, 0 ≤ w x)
    (hw1 : ∑ x : Ξ, w x = 1) (μmax : ℝ) (hμ0 : 0 ≤ μmax)
    (hdiff : Diffuse w μmax)
    (S : Finset Ξ) (T N : ℕ) (hS : S.card ≤ 2 * T) (hT : 1 ≤ T)
    (hN : 1 ≤ N) (σ : ℝ) (hσ : 0 ≤ σ)
    (hμT : 512 * (T : ℝ) * μmax ≤ 1)
    (vhat : (Fin N → Ξ) → Ξ → ℝ) (hv : ∀ ω, IdentityClass (vhat ω) S) :
    pexp w N (fun ω => σ / N * ∑ b : Ξ, vhat ω b * cntC w b ω)
      ≤ σ / 8 * Real.sqrt (2 * (T : ℝ) / N) := by
  have hN0 : (0 : ℝ) < N := by exact_mod_cast hN
  have hT0 : (0 : ℝ) < T := by exact_mod_cast hT
  have hmain := t1_diffuse_upper w hw hw1 μmax hdiff S (2 * T) N hS hN σ hσ vhat hv
  refine le_trans hmain ?_
  -- 2σ(2T)√(μmax/N) ≤ (σ/8)√(2T/N)  ⟺  16T²·μmax/N ≤ (1/64)·2T/N
  have hcast : ((2 * T : ℕ) : ℝ) = 2 * (T : ℝ) := by push_cast; ring
  rw [hcast]
  have hfactor : 2 * σ * (2 * (T : ℝ)) * Real.sqrt (μmax / N)
      = σ * (4 * (T : ℝ) * Real.sqrt (μmax / N)) := by ring
  have hrhs : σ / 8 * Real.sqrt (2 * (T : ℝ) / N)
      = σ * (Real.sqrt (2 * (T : ℝ) / N) / 8) := by ring
  rw [hfactor, hrhs]
  apply mul_le_mul_of_nonneg_left _ hσ
  -- both sides are √ of explicit radicands; compare radicands
  have hμN : 0 ≤ μmax / (N : ℝ) := by positivity
  have hl : 4 * (T : ℝ) * Real.sqrt (μmax / N)
      = Real.sqrt (16 * (T : ℝ) ^ 2 * (μmax / N)) := by
    rw [show 16 * (T : ℝ) ^ 2 * (μmax / N)
        = (4 * (T : ℝ)) ^ 2 * (μmax / N) by ring,
      Real.sqrt_mul (by positivity), Real.sqrt_sq (by positivity)]
  have h64 : Real.sqrt (64 : ℝ) = 8 := by
    rw [show (64 : ℝ) = 8 ^ 2 by norm_num,
      Real.sqrt_sq (by norm_num : (0 : ℝ) ≤ 8)]
  have hr : Real.sqrt (2 * (T : ℝ) / N) / 8
      = Real.sqrt (2 * (T : ℝ) / N / 64) := by
    rw [Real.sqrt_div (by positivity) 64, h64]
  rw [hl, hr]
  apply Real.sqrt_le_sqrt
  -- 16T²·(μmax/N) ≤ (2T/N)/64  ⟸  512·T·μmax ≤ 1
  rw [show 16 * (T : ℝ) ^ 2 * (μmax / N) = 16 * (T : ℝ) ^ 2 * μmax / N by ring,
    show 2 * (T : ℝ) / N / 64 = (T : ℝ) / 32 / N by ring,
    div_le_div_iff_of_pos_right hN0]
  nlinarith [mul_le_mul_of_nonneg_right hμT
    (by positivity : (0 : ℝ) ≤ (T : ℝ) / 32)]