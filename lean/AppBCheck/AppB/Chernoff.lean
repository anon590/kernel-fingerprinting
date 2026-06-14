/-
# Multiplicative Chernoff lower tail for the binomial (used in Theorem B.6(b))

Paper step: "By the multiplicative Chernoff lower tail,
`P(s ≤ μ_s/2) ≤ e^{-μ_s/8}`" for `s ~ Bin(N, p)`, `μ_s = Np`.

Proved here *in full*, self-contained over the explicit binomial weights
`binomW N p k = C(N,k) p^k (1-p)^{N-k}` (the law of `s`): exponential tilting
at `t = ln 2`, the binomial theorem, `1 - p/2 ≤ e^{-p/2}`, and
`ln 2 ≤ 3/4`.  (The tilt at `ln 2` in fact yields the exponent
`-(1-ln2)μ/2 ≈ -0.153 μ`, stronger than the paper's `-μ/8`.)
-/
import Mathlib.Data.Nat.Choose.Sum
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.Complex.ExponentialBounds

open Finset

namespace AppB

/-- Binomial weight `P(Bin(N,p) = k) = C(N,k) p^k (1-p)^{N-k}`. -/
noncomputable def binomW (N : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (N.choose k : ℝ) * p ^ k * (1 - p) ^ (N - k)

theorem binomW_nonneg (N : ℕ) (p : ℝ) (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (k : ℕ) :
    0 ≤ binomW N p k := by
  have h1p : (0 : ℝ) ≤ 1 - p := by linarith
  exact mul_nonneg (mul_nonneg (Nat.cast_nonneg _) (pow_nonneg hp0 _))
    (pow_nonneg h1p _)

/-- The binomial weights sum to 1 (binomial theorem). -/
theorem binomW_sum (N : ℕ) (p : ℝ) :
    ∑ k ∈ range (N + 1), binomW N p k = 1 := by
  calc ∑ k ∈ range (N + 1), binomW N p k
      = ∑ k ∈ range (N + 1), p ^ k * (1 - p) ^ (N - k) * (N.choose k : ℝ) :=
        Finset.sum_congr rfl fun k _ => by unfold binomW; ring
    _ = (p + (1 - p)) ^ N := (add_pow p (1 - p) N).symm
    _ = 1 := by norm_num

/-- **Multiplicative Chernoff lower tail** for `Bin(N,p)`:
`P(s ≤ Np/2) ≤ exp(-Np/8)`. -/
theorem binomial_lower_tail (N : ℕ) (p : ℝ) (hp0 : 0 ≤ p) (hp1 : p ≤ 1) :
    ∑ k ∈ (range (N + 1)).filter (fun k : ℕ => (k : ℝ) ≤ (N : ℝ) * p / 2),
        binomW N p k
      ≤ Real.exp (-((N : ℝ) * p) / 8) := by
  set μ := (N : ℝ) * p with hμdef
  have hμnn : 0 ≤ μ := mul_nonneg (Nat.cast_nonneg N) hp0
  have hl2nn : (0 : ℝ) ≤ Real.log 2 := Real.log_nonneg one_le_two
  have hhalf : Real.exp (-Real.log 2) = 1 / 2 := by
    rw [Real.exp_neg, Real.exp_log two_pos]
    norm_num
  -- Step 1: tilt.  On `{k ≤ μ/2}` the factor `e^{(μ/2-k)ln2}` is ≥ 1.
  have step1 : ∑ k ∈ (range (N + 1)).filter (fun k : ℕ => (k : ℝ) ≤ μ / 2), binomW N p k
      ≤ ∑ k ∈ range (N + 1),
          binomW N p k * (Real.exp (μ / 2 * Real.log 2) * (1 / 2) ^ k) := by
    refine le_trans (Finset.sum_le_sum ?_)
      (Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _) ?_)
    · intro k hk
      rw [Finset.mem_filter] at hk
      have hkle : (k : ℝ) ≤ μ / 2 := hk.2
      have hpow : ((1 : ℝ) / 2) ^ k = Real.exp ((k : ℝ) * (-Real.log 2)) := by
        rw [Real.exp_nat_mul, hhalf]
      have htilt : (1 : ℝ) ≤ Real.exp (μ / 2 * Real.log 2) * (1 / 2) ^ k := by
        rw [hpow, ← Real.exp_add]
        apply Real.one_le_exp
        nlinarith
      exact le_mul_of_one_le_right (binomW_nonneg N p hp0 hp1 k) htilt
    · intro k _ _
      have h1 : (0 : ℝ) ≤ Real.exp (μ / 2 * Real.log 2) * (1 / 2) ^ k := by positivity
      exact mul_nonneg (binomW_nonneg N p hp0 hp1 k) h1
  -- Step 2: the tilted sum is the binomial theorem at `(p/2, 1-p)`.
  have step2 : ∑ k ∈ range (N + 1),
        binomW N p k * (Real.exp (μ / 2 * Real.log 2) * (1 / 2) ^ k)
      = Real.exp (μ / 2 * Real.log 2) * (1 - p / 2) ^ N := by
    have hterm : ∀ k ∈ range (N + 1),
        binomW N p k * (Real.exp (μ / 2 * Real.log 2) * (1 / 2) ^ k)
          = Real.exp (μ / 2 * Real.log 2)
              * ((p / 2) ^ k * (1 - p) ^ (N - k) * (N.choose k : ℝ)) := by
      intro k _
      unfold binomW
      rw [div_pow, div_pow]
      ring
    rw [Finset.sum_congr rfl hterm, ← Finset.mul_sum]
    congr 1
    have h := add_pow (p / 2) (1 - p) N
    rw [← h]
    congr 1
    ring
  -- Step 3: `(1 - p/2)^N ≤ e^{-μ/2}`.
  have step3 : (1 - p / 2) ^ N ≤ Real.exp (-(μ / 2)) := by
    have h1 : 1 - p / 2 ≤ Real.exp (-(p / 2)) := by
      have h := Real.add_one_le_exp (-(p / 2))
      linarith
    have h2 : (0 : ℝ) ≤ 1 - p / 2 := by linarith
    calc (1 - p / 2) ^ N ≤ (Real.exp (-(p / 2))) ^ N := pow_le_pow_left₀ h2 h1 N
      _ = Real.exp ((N : ℝ) * (-(p / 2))) := (Real.exp_nat_mul _ N).symm
      _ = Real.exp (-(μ / 2)) := by rw [hμdef]; ring_nf
  -- Combine and compare exponents:  μ ln2 / 2 - μ/2 ≤ -μ/8  ⟸  ln 2 ≤ 3/4.
  calc ∑ k ∈ (range (N + 1)).filter (fun k : ℕ => (k : ℝ) ≤ μ / 2), binomW N p k
      ≤ Real.exp (μ / 2 * Real.log 2) * (1 - p / 2) ^ N := step1.trans step2.le
    _ ≤ Real.exp (μ / 2 * Real.log 2) * Real.exp (-(μ / 2)) :=
        mul_le_mul_of_nonneg_left step3 (Real.exp_nonneg _)
    _ = Real.exp (μ / 2 * Real.log 2 + -(μ / 2)) := (Real.exp_add _ _).symm
    _ ≤ Real.exp (-μ / 8) := by
        apply Real.exp_le_exp.mpr
        have hl2 : Real.log 2 < 0.6931471808 := Real.log_two_lt_d9
        nlinarith

end AppB
