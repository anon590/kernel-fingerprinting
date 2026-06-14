/-
# Appendix B, Theorem B.6 (`thm:lowerapp`): tightness via instance fingerprinting

Construction (paper): cells paired as `(2t-1, 2t)`, here indexed by
`Fin T × Bool` with `m = 2T`; query `t` compares `c_{v_t}` to `c_0` and
returns the sign `s_t` of `n_{(t,1)} - n_{(t,0)}`; the final candidate is
`c_{v*}` with `v*(t,1) = s_t`, `v*(t,0) = -s_t`.

Verified here:

* `attack_population_mean`  — **(a)**: `∑_b v*(b) = 0` for every sign
  realization, hence `J(ĉ) = μ₀` exactly;
* `empirical_mean_fiber`    — `Ĵ_D(c_v) = μ₀ + (σ/N) ∑_b v(b) n_b`;
* `attack_inflation`        — the sign trick `s_t(n₁-n₂) = |n₁-n₂|`
  (covering ties), giving `Z = (σ/N) ∑_t |n_{(t,1)} - n_{(t,0)}|`;
* `pair_diff_lower`         — **(b), per pair**: with `s ~ Bin(N, 2/m)` and
  Lemma B.5 applied conditionally,
  `E|n₁-n₂| = ∑_s binomW(s)·halfAbsMean(s) ≥ (1/4)√(N/m)`
  (uses the Chernoff tail `binomial_lower_tail`, `μ_s = 2N/m ≥ 8`, and
  `1 - e^{-1} ≥ 5/8`, i.e. `e ≥ 8/3`);
* `thm_lower_b`              — **(b), assembly**: `EZ ≥ (σ/8)√(m/N)`;
* `cnt_one_instance_diff`, `Z_bounded_diff` — **(c)**: one-instance changes move `Z`
  by at most `2σ/N` (the bounded-differences constant for McDiarmid);
* `mcdiarmid_exponent`      — **(c)**: the exponent `2t²/(N(2σ/N)²) = m/512`
  at `t = (σ/16)√(m/N)`;
* `deviation_conclusion`    — **(c)**: `t ≤ EZ/2` and `Z ≥ EZ - t` give
  `Z ≥ (σ/16)√(m/N)`;
* `thm_lower_disappointment` — the final step: any reported
  `L̂ ≥ Ĵ_D(ĉ) - ε` with `ε ≤ (σ/32)√(m/N)` satisfies `L̂ > μ₀ = J(ĉ)`.

Bridges (standard facts invoked by the paper but not re-proved here, each
marked as a hypothesis where used): the multinomial conditional decomposition
`E|n₁-n₂| = ∑_s P(Bin(N,2/m)=s)·E[|n₁-n₂| | s]` with
`n₁ | s ~ Bin(s,1/2)`, and McDiarmid's inequality itself.
-/
import Mathlib.Analysis.Complex.ExponentialBounds
import AppB.LemmaMad
import AppB.Chernoff

open Finset

namespace AppB

/-! ## The conditional pair-difference mean and its lower bound -/

/-- `E[|n₁ - n₂|]` for a fair binomial split of `s` items:
`∑_j C(s,j)|2j - s| / 2^s`  (note `|2j-s| = 2|j - s/2|`). -/
noncomputable def halfAbsMean (s : ℕ) : ℝ :=
  (∑ j ∈ range (s + 1), (s.choose j : ℝ) * |2 * (j : ℝ) - s|) / 2 ^ s

theorem halfAbsMean_nonneg (s : ℕ) : 0 ≤ halfAbsMean s :=
  div_nonneg (Finset.sum_nonneg fun j _ => by positivity) (by positivity)

/-- Lemma B.5 in the form used in (b): `E[|n₁-n₂| | s] = 2E|Δ| ≥ (2/5)√s`. -/
theorem halfAbsMean_ge (s : ℕ) (hs : 1 ≤ s) :
    2 / 5 * Real.sqrt s ≤ halfAbsMean s := by
  have h := lemma_mad s hs
  unfold halfAbsMean
  have hterm : ∀ j ∈ range (s + 1), (s.choose j : ℝ) * |2 * (j : ℝ) - s|
      = 2 * ((s.choose j : ℝ) * |(j : ℝ) - (s : ℝ) / 2|) := by
    intro j _
    have h2 : (2 : ℝ) * (j : ℝ) - s = 2 * ((j : ℝ) - (s : ℝ) / 2) := by ring
    rw [h2, abs_mul, abs_of_pos two_pos]
    ring
  rw [Finset.sum_congr rfl hterm, ← Finset.mul_sum, mul_div_assoc]
  linarith

/-- **Theorem B.6(b), per-pair bound.**  With pair-total `s ~ Bin(N, 2/m)`
(weights `binomW`) and Lemma B.5 applied conditionally on `s`,
`E|n₁ - n₂| = ∑_s binomW(s)·halfAbsMean(s) ≥ (1/4)√(N/m)`,
provided `m ≥ 2` and `m ≤ N/4` (so `μ_s = 2N/m ≥ 8`). -/
theorem pair_diff_lower (N m : ℕ) (hm : 2 ≤ m) (hNm : 4 * m ≤ N) :
    (1 / 4 : ℝ) * Real.sqrt ((N : ℝ) / m)
      ≤ ∑ s ∈ range (N + 1), binomW N (2 / m) s * halfAbsMean s := by
  have hm0 : (0 : ℝ) < m := by
    have : (0 : ℕ) < m := by omega
    exact_mod_cast this
  have hN0 : (0 : ℝ) < N := by
    have : (0 : ℕ) < N := by omega
    exact_mod_cast this
  have hNm' : (4 : ℝ) * m ≤ N := by exact_mod_cast hNm
  set p : ℝ := 2 / m with hpdef
  have hp0 : 0 ≤ p := by positivity
  have hp1 : p ≤ 1 := by
    rw [hpdef, div_le_one hm0]
    exact_mod_cast hm
  have hμhalf : (N : ℝ) * p / 2 = (N : ℝ) / m := by
    rw [hpdef]
    field_simp
  have hμ8 : 8 ≤ (N : ℝ) * p := by
    have hexp : (N : ℝ) * p = 2 * N / m := by rw [hpdef]; ring
    rw [hexp, le_div_iff₀ hm0]
    linarith
  have hNm4 : (4 : ℝ) ≤ (N : ℝ) / m := by
    rw [le_div_iff₀ hm0]
    linarith
  -- tail weight is at least 5/8
  have htail : (5 / 8 : ℝ)
      ≤ ∑ s ∈ (range (N + 1)).filter
          (fun s : ℕ => ¬((s : ℝ) ≤ (N : ℝ) * p / 2)), binomW N p s := by
    have hhead := binomial_lower_tail N p hp0 hp1
    have hsum := binomW_sum N p
    have hsplit := Finset.sum_filter_add_sum_filter_not (range (N + 1))
        (fun s : ℕ => (s : ℝ) ≤ (N : ℝ) * p / 2) (binomW N p)
    have hexp : Real.exp (-((N : ℝ) * p) / 8) ≤ 3 / 8 := by
      calc Real.exp (-((N : ℝ) * p) / 8)
          ≤ Real.exp (-1) := Real.exp_le_exp.mpr (by linarith)
        _ ≤ 3 / 8 := by
            rw [Real.exp_neg, show (3 / 8 : ℝ) = (8 / 3)⁻¹ by norm_num]
            apply inv_anti₀ (by norm_num)
            calc (8 / 3 : ℝ) ≤ 2.7182818283 := by norm_num
              _ ≤ Real.exp 1 := Real.exp_one_gt_d9.le
    linarith
  -- on the tail, `halfAbsMean s ≥ (2/5)√(N/m)`
  have hterm : ∀ s ∈ (range (N + 1)).filter
      (fun s : ℕ => ¬((s : ℝ) ≤ (N : ℝ) * p / 2)),
      2 / 5 * Real.sqrt ((N : ℝ) / m) * binomW N p s
        ≤ binomW N p s * halfAbsMean s := by
    intro s hs
    rw [Finset.mem_filter] at hs
    have hsgt : (N : ℝ) * p / 2 < s := not_le.mp hs.2
    have hsge : (N : ℝ) / m ≤ s := by
      rw [← hμhalf]
      linarith
    have hs1 : 1 ≤ s := by
      rcases Nat.eq_zero_or_pos s with rfl | h
      · norm_num at hsge
        linarith
      · exact h
    have hh := halfAbsMean_ge s hs1
    have hsq : Real.sqrt ((N : ℝ) / m) ≤ Real.sqrt s := Real.sqrt_le_sqrt hsge
    have hw := binomW_nonneg N p hp0 hp1 s
    calc 2 / 5 * Real.sqrt ((N : ℝ) / m) * binomW N p s
        ≤ 2 / 5 * Real.sqrt s * binomW N p s := by
          apply mul_le_mul_of_nonneg_right _ hw
          linarith
      _ ≤ halfAbsMean s * binomW N p s := mul_le_mul_of_nonneg_right hh hw
      _ = binomW N p s * halfAbsMean s := mul_comm _ _
  -- assemble
  calc (1 / 4 : ℝ) * Real.sqrt ((N : ℝ) / m)
      = 2 / 5 * Real.sqrt ((N : ℝ) / m) * (5 / 8) := by ring
    _ ≤ 2 / 5 * Real.sqrt ((N : ℝ) / m)
          * ∑ s ∈ (range (N + 1)).filter
              (fun s : ℕ => ¬((s : ℝ) ≤ (N : ℝ) * p / 2)), binomW N p s :=
        mul_le_mul_of_nonneg_left htail (by positivity)
    _ = ∑ s ∈ (range (N + 1)).filter
          (fun s : ℕ => ¬((s : ℝ) ≤ (N : ℝ) * p / 2)),
          2 / 5 * Real.sqrt ((N : ℝ) / m) * binomW N p s := Finset.mul_sum _ _ _
    _ ≤ ∑ s ∈ (range (N + 1)).filter
          (fun s : ℕ => ¬((s : ℝ) ≤ (N : ℝ) * p / 2)),
          binomW N p s * halfAbsMean s := Finset.sum_le_sum hterm
    _ ≤ ∑ s ∈ range (N + 1), binomW N p s * halfAbsMean s := by
        apply Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _)
        intro s _ _
        exact mul_nonneg (binomW_nonneg N p hp0 hp1 s) (halfAbsMean_nonneg s)

/-! ## The attack construction: candidates, counts, and identities -/

/-- The final attack candidate's offset pattern:
`v*(t, true) = s_t`, `v*(t, false) = -s_t`. -/
def vstar {T : ℕ} (sgn : Fin T → ℝ) : Fin T × Bool → ℝ :=
  fun b => cond b.2 (sgn b.1) (-sgn b.1)

/-- Cell counts of the pool, as sums of indicators. -/
noncomputable def cnt {B : Type*} [DecidableEq B] {N : ℕ}
    (ω : Fin N → B) (b : B) : ℝ :=
  ∑ i : Fin N, if ω i = b then 1 else 0

/-- **Theorem B.6(a).**  For *every* realization of the signs,
`∑_b v*(b) = 0`, hence the population mean of `Y(c_{v*}, ·) = μ₀ + σ v*(φ(·))`
under the uniform fingerprint is exactly `μ₀`. -/
theorem attack_population_mean (T : ℕ) (hT : 1 ≤ T) (sgn : Fin T → ℝ)
    (μ0 σ : ℝ) :
    (∑ b : Fin T × Bool, (μ0 + σ * vstar sgn b)) / (2 * T) = μ0 := by
  have hT0 : (0 : ℝ) < T := by exact_mod_cast hT
  have hsum0 : ∑ b : Fin T × Bool, vstar sgn b = 0 := by
    rw [Fintype.sum_prod_type]
    have hinner : ∀ t : Fin T, (∑ β : Bool, vstar sgn (t, β)) = 0 := by
      intro t
      rw [Fintype.sum_bool]
      simp only [vstar, cond_true, cond_false]
      ring
    rw [Finset.sum_congr rfl fun t _ => hinner t]
    simp
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, hsum0, mul_zero, add_zero,
    Finset.sum_const, Finset.card_univ]
  have hcard : Fintype.card (Fin T × Bool) = 2 * T := by
    simp [Fintype.card_prod, mul_comm]
  rw [hcard, nsmul_eq_mul]
  push_cast
  field_simp

/-- The empirical mean of the attack candidate decomposes over cell counts:
`Ĵ_D(c_v) = (1/N) ∑_i (μ₀ + σ v(ω_i)) = μ₀ + (σ/N) ∑_b v(b) n_b`. -/
theorem empirical_mean_fiber {B : Type*} [Fintype B] [DecidableEq B]
    (N : ℕ) (hN : 1 ≤ N) (ω : Fin N → B) (v : B → ℝ) (μ0 σ : ℝ) :
    (∑ i : Fin N, (μ0 + σ * v (ω i))) / N
      = μ0 + σ / N * ∑ b : B, cnt ω b * v b := by
  have hN0 : (0 : ℝ) < N := by exact_mod_cast hN
  have hfib : ∑ b : B, cnt ω b * v b = ∑ i : Fin N, v (ω i) := by
    unfold cnt
    calc ∑ b : B, (∑ i : Fin N, if ω i = b then (1 : ℝ) else 0) * v b
        = ∑ b : B, ∑ i : Fin N, (if ω i = b then (1 : ℝ) else 0) * v b := by
          refine Finset.sum_congr rfl fun b _ => ?_
          rw [Finset.sum_mul]
      _ = ∑ i : Fin N, ∑ b : B, (if ω i = b then (1 : ℝ) else 0) * v b :=
          Finset.sum_comm
      _ = ∑ i : Fin N, v (ω i) := by
          refine Finset.sum_congr rfl fun i _ => ?_
          simp [ite_mul]
  have hsum : ∑ i : Fin N, (μ0 + σ * v (ω i))
      = N * μ0 + σ * ∑ b : B, cnt ω b * v b := by
    rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ,
      Fintype.card_fin, ← Finset.mul_sum, hfib, nsmul_eq_mul]
  rw [hsum]
  field_simp

/-- **The inflation identity** (`Z = (σ/N)∑_t |n_{2t-1}-n_{2t}|`): with the
reported bits `s_t` consistent with the realized signs
(`s_t (n₁-n₂) = |n₁-n₂|`, which also covers ties, where the term vanishes),
the sign pattern `v*` turns the count contrast into absolute values. -/
theorem attack_inflation (T N : ℕ) (σ : ℝ) (n : Fin T × Bool → ℝ)
    (sgn : Fin T → ℝ)
    (hsgn : ∀ t, sgn t * (n (t, true) - n (t, false))
      = |n (t, true) - n (t, false)|) :
    σ / N * ∑ b : Fin T × Bool, vstar sgn b * n b
      = σ / N * ∑ t : Fin T, |n (t, true) - n (t, false)| := by
  congr 1
  rw [Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun t _ => ?_
  rw [Fintype.sum_bool, ← hsgn t]
  simp only [vstar, cond_true, cond_false]
  ring

/-- **Theorem B.6(b), assembly.**  With `m = 2T` cells and per-pair expectations
`Ed t ≥ (1/4)√(N/m)`, linearity gives `EZ = (σ/N)∑_t Ed t ≥ (σ/8)√(m/N)`. -/
theorem attack_inflation_mean (T N m : ℕ) (hmT : m = 2 * T) (hT : 1 ≤ T)
    (hN : 1 ≤ N) (σ : ℝ) (hσ : 0 ≤ σ) (Ed : Fin T → ℝ)
    (hEd : ∀ t, (1 / 4 : ℝ) * Real.sqrt ((N : ℝ) / m) ≤ Ed t) :
    σ / 8 * Real.sqrt ((m : ℝ) / N) ≤ σ / N * ∑ t : Fin T, Ed t := by
  have hT0 : (0 : ℝ) < T := by exact_mod_cast hT
  have hN0 : (0 : ℝ) < N := by exact_mod_cast hN
  have hm0 : (0 : ℝ) < m := by
    have : (0 : ℕ) < m := by omega
    exact_mod_cast this
  have hmT' : (m : ℝ) = 2 * T := by exact_mod_cast hmT
  -- √(m/N)·√(N/m) = 1 and (m/N)·√(N/m) = √(m/N)
  have huw : Real.sqrt ((m : ℝ) / N) * Real.sqrt ((N : ℝ) / m) = 1 := by
    rw [← Real.sqrt_mul (by positivity)]
    rw [show (m : ℝ) / N * ((N : ℝ) / m) = 1 by field_simp]
    exact Real.sqrt_one
  have hu2 : Real.sqrt ((m : ℝ) / N) ^ 2 = (m : ℝ) / N :=
    Real.sq_sqrt (by positivity)
  have hkey : (m : ℝ) / N * Real.sqrt ((N : ℝ) / m) = Real.sqrt ((m : ℝ) / N) := by
    calc (m : ℝ) / N * Real.sqrt ((N : ℝ) / m)
        = Real.sqrt ((m : ℝ) / N) ^ 2 * Real.sqrt ((N : ℝ) / m) := by rw [hu2]
      _ = Real.sqrt ((m : ℝ) / N)
            * (Real.sqrt ((m : ℝ) / N) * Real.sqrt ((N : ℝ) / m)) := by ring
      _ = Real.sqrt ((m : ℝ) / N) := by rw [huw, mul_one]
  have hsumge : (T : ℝ) * (1 / 4 * Real.sqrt ((N : ℝ) / m)) ≤ ∑ t : Fin T, Ed t := by
    have h := Finset.card_nsmul_le_sum Finset.univ Ed
        (1 / 4 * Real.sqrt ((N : ℝ) / m)) (fun t _ => hEd t)
    rwa [Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at h
  calc σ / 8 * Real.sqrt ((m : ℝ) / N)
      = σ / N * ((T : ℝ) * (1 / 4 * Real.sqrt ((N : ℝ) / m))) := by
        rw [← hkey, hmT']
        field_simp
        ring
    _ ≤ σ / N * ∑ t : Fin T, Ed t :=
        mul_le_mul_of_nonneg_left hsumge (by positivity)

/-- **Theorem B.6(b), full chain**: per-pair conditional decomposition
(hypothesis `hEd`, the law of total expectation for the multinomial pool —
the standard bridge the paper invokes as "conditionally on s,
n_{2t-1} ~ Bin(s,1/2)") plus `pair_diff_lower` give `EZ ≥ (σ/8)√(m/N)`. -/
theorem thm_lower_b (T N m : ℕ) (hmT : m = 2 * T) (hT : 1 ≤ T) (hm : 2 ≤ m)
    (hNm : 4 * m ≤ N) (σ : ℝ) (hσ : 0 ≤ σ) (Ed : Fin T → ℝ)
    (hEd : ∀ t, Ed t = ∑ s ∈ range (N + 1), binomW N (2 / m) s * halfAbsMean s) :
    σ / 8 * Real.sqrt ((m : ℝ) / N) ≤ σ / N * ∑ t : Fin T, Ed t := by
  have hN : 1 ≤ N := by omega
  apply attack_inflation_mean T N m hmT hT hN σ hσ Ed
  intro t
  rw [hEd t]
  exact pair_diff_lower N m hm hNm

/-! ## (c): bounded differences and the McDiarmid arithmetic -/

/-- Changing one instance moves the count vector by at most 2 in ℓ¹. -/
theorem cnt_one_instance_diff {B : Type*} [Fintype B] [DecidableEq B] {N : ℕ}
    (ω ω' : Fin N → B) (i₀ : Fin N) (hsame : ∀ i, i ≠ i₀ → ω i = ω' i) :
    ∑ b : B, |cnt ω b - cnt ω' b| ≤ 2 := by
  have hpt : ∀ b : B, cnt ω b - cnt ω' b
      = (if ω i₀ = b then (1 : ℝ) else 0) - (if ω' i₀ = b then (1 : ℝ) else 0) := by
    intro b
    unfold cnt
    rw [← Finset.sum_sub_distrib]
    rw [Finset.sum_eq_single i₀]
    · intro i _ hi
      rw [hsame i hi]
      ring
    · intro h
      exact absurd (Finset.mem_univ _) h
  calc ∑ b : B, |cnt ω b - cnt ω' b|
      = ∑ b : B, |(if ω i₀ = b then (1 : ℝ) else 0)
          - (if ω' i₀ = b then (1 : ℝ) else 0)| :=
        Finset.sum_congr rfl fun b _ => by rw [hpt b]
    _ ≤ ∑ b : B, ((if ω i₀ = b then (1 : ℝ) else 0)
          + (if ω' i₀ = b then (1 : ℝ) else 0)) := by
        apply Finset.sum_le_sum
        intro b _
        split_ifs <;> norm_num
    _ = (∑ b : B, if ω i₀ = b then (1 : ℝ) else 0)
          + ∑ b : B, if ω' i₀ = b then (1 : ℝ) else 0 :=
        Finset.sum_add_distrib
    _ = 2 := by
        rw [Finset.sum_ite_eq Finset.univ (ω i₀) (fun _ => (1 : ℝ)),
          Finset.sum_ite_eq Finset.univ (ω' i₀) (fun _ => (1 : ℝ))]
        simp
        norm_num

/-- **Theorem B.6(c), bounded differences**: a one-instance change moves
`Z = (σ/N) ∑_t |n_{(t,1)} - n_{(t,0)}|` by at most `2σ/N`. -/
theorem Z_bounded_diff (T N : ℕ) (σ : ℝ) (hσ : 0 ≤ σ)
    (ω ω' : Fin N → Fin T × Bool) (i₀ : Fin N)
    (hsame : ∀ i, i ≠ i₀ → ω i = ω' i) :
    abs (σ / N * (∑ t : Fin T, |cnt ω (t, true) - cnt ω (t, false)|)
      - σ / N * ∑ t : Fin T, |cnt ω' (t, true) - cnt ω' (t, false)|)
      ≤ 2 * σ / N := by
  have hsum : abs ((∑ t : Fin T, |cnt ω (t, true) - cnt ω (t, false)|)
      - ∑ t : Fin T, |cnt ω' (t, true) - cnt ω' (t, false)|) ≤ 2 := by
    calc abs ((∑ t : Fin T, |cnt ω (t, true) - cnt ω (t, false)|)
        - ∑ t : Fin T, |cnt ω' (t, true) - cnt ω' (t, false)|)
        = abs (∑ t : Fin T, (|cnt ω (t, true) - cnt ω (t, false)|
            - |cnt ω' (t, true) - cnt ω' (t, false)|)) := by
          rw [Finset.sum_sub_distrib]
      _ ≤ ∑ t : Fin T, abs (|cnt ω (t, true) - cnt ω (t, false)|
            - |cnt ω' (t, true) - cnt ω' (t, false)|) :=
          Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ t : Fin T, (|cnt ω (t, true) - cnt ω' (t, true)|
            + |cnt ω (t, false) - cnt ω' (t, false)|) := by
          apply Finset.sum_le_sum
          intro t _
          have h1 : abs (|cnt ω (t, true) - cnt ω (t, false)|
              - |cnt ω' (t, true) - cnt ω' (t, false)|)
              ≤ abs ((cnt ω (t, true) - cnt ω (t, false))
                - (cnt ω' (t, true) - cnt ω' (t, false))) :=
            abs_abs_sub_abs_le_abs_sub _ _
          have h2 : (cnt ω (t, true) - cnt ω (t, false))
              - (cnt ω' (t, true) - cnt ω' (t, false))
              = (cnt ω (t, true) - cnt ω' (t, true))
                - (cnt ω (t, false) - cnt ω' (t, false)) := by ring
          have h3 : abs ((cnt ω (t, true) - cnt ω' (t, true))
              - (cnt ω (t, false) - cnt ω' (t, false)))
              ≤ |cnt ω (t, true) - cnt ω' (t, true)|
                + |cnt ω (t, false) - cnt ω' (t, false)| := by
            have h := abs_add_le (cnt ω (t, true) - cnt ω' (t, true))
                (cnt ω' (t, false) - cnt ω (t, false))
            have heq : (cnt ω (t, true) - cnt ω' (t, true))
                + (cnt ω' (t, false) - cnt ω (t, false))
                = (cnt ω (t, true) - cnt ω' (t, true))
                  - (cnt ω (t, false) - cnt ω' (t, false)) := by ring
            have habs : |cnt ω' (t, false) - cnt ω (t, false)|
                = |cnt ω (t, false) - cnt ω' (t, false)| := abs_sub_comm _ _
            rw [heq, habs] at h
            exact h
          calc abs (|cnt ω (t, true) - cnt ω (t, false)|
              - |cnt ω' (t, true) - cnt ω' (t, false)|)
              ≤ abs ((cnt ω (t, true) - cnt ω (t, false))
                - (cnt ω' (t, true) - cnt ω' (t, false))) := h1
            _ = abs ((cnt ω (t, true) - cnt ω' (t, true))
                - (cnt ω (t, false) - cnt ω' (t, false))) := by rw [h2]
            _ ≤ _ := h3
      _ = ∑ b : Fin T × Bool, |cnt ω b - cnt ω' b| := by
          rw [Fintype.sum_prod_type]
          refine Finset.sum_congr rfl fun t _ => ?_
          rw [Fintype.sum_bool]
      _ ≤ 2 := cnt_one_instance_diff ω ω' i₀ hsame
  calc abs (σ / N * (∑ t : Fin T, |cnt ω (t, true) - cnt ω (t, false)|)
      - σ / N * ∑ t : Fin T, |cnt ω' (t, true) - cnt ω' (t, false)|)
      = σ / N * abs ((∑ t : Fin T, |cnt ω (t, true) - cnt ω (t, false)|)
          - ∑ t : Fin T, |cnt ω' (t, true) - cnt ω' (t, false)|) := by
        rw [← mul_sub, abs_mul, abs_of_nonneg (by positivity)]
    _ ≤ σ / N * 2 := mul_le_mul_of_nonneg_left hsum (by positivity)
    _ = 2 * σ / N := by ring

/-- **Theorem B.6(c), exponent arithmetic**: McDiarmid with bounded differences
`c_i = 2σ/N` at deviation `t = (σ/16)√(m/N)` has exponent
`2t² / (N(2σ/N)²) = m/512`. -/
theorem mcdiarmid_exponent (N m : ℕ) (σ : ℝ) (hσ : 0 < σ) (hN : 1 ≤ N) :
    2 * (σ / 16 * Real.sqrt ((m : ℝ) / N)) ^ 2 / ((N : ℝ) * (2 * σ / N) ^ 2)
      = (m : ℝ) / 512 := by
  have hN0 : (0 : ℝ) < N := by exact_mod_cast hN
  have hsq : Real.sqrt ((m : ℝ) / N) ^ 2 = (m : ℝ) / N :=
    Real.sq_sqrt (by positivity)
  rw [mul_pow, hsq]
  field_simp
  ring

/-- **Theorem B.6(c), deviation conclusion**: `EZ ≥ 2t` (from (b)) and
`Z ≥ EZ - t` (McDiarmid's event) give `Z ≥ t = (σ/16)√(m/N)`. -/
theorem deviation_conclusion (Z EZ σ mN : ℝ)
    (hEZ : σ / 8 * Real.sqrt mN ≤ EZ)
    (hdev : EZ - σ / 16 * Real.sqrt mN ≤ Z) :
    σ / 16 * Real.sqrt mN ≤ Z := by
  have h : σ / 8 * Real.sqrt mN = 2 * (σ / 16 * Real.sqrt mN) := by ring
  linarith [h ▸ hEZ]

/-- **Theorem B.6, disappointment.**  Any harness reporting
`L̂ ≥ Ĵ_D(ĉ) - ε` with `ε ≤ (σ/32)√(m/N)`, on the event
`Ĵ_D(ĉ) ≥ μ₀ + (σ/16)√(m/N)` of (c), satisfies `L̂ > μ₀ = J(ĉ)`:
the certificate *disappoints*. -/
theorem thm_lower_disappointment (μ0 σ mN Jemp Lhat ε : ℝ)
    (hσ : 0 < σ) (hmN : 0 < mN)
    (hJ : μ0 + σ / 16 * Real.sqrt mN ≤ Jemp)
    (hL : Jemp - ε ≤ Lhat) (hε : ε ≤ σ / 32 * Real.sqrt mN) :
    μ0 < Lhat := by
  have hs : 0 < Real.sqrt mN := Real.sqrt_pos.mpr hmN
  nlinarith

end AppB
