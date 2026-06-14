/-
# Appendix B, Lemma B.5 (`lem:mad`): pair-difference anti-concentration

Paper claim: for `X ~ Bin(s, 1/2)`, `s ≥ 1`, and `Δ = X - s/2`,
`E|Δ| ≥ √s / 5`.

Paper proof, formalized *in full* (no probabilistic primitives assumed —
expectations are the explicit binomial sums `∑_k C(s,k) (·) / 2^s`):

1. `EΔ² = s/4`                                        (`sum_choose_sq`)
2. `EΔ⁴ = (s/4)(1 + 3(s-2)/4) = s(3s-2)/16 ≤ 3(s/4)²` (`sum_choose_quart`)
3. Cauchy–Schwarz: `EΔ² = E[|Δ|^{1/2}|Δ|^{3/2}] ≤ (E|Δ|)^{1/2}(E|Δ|³)^{1/2}`
   — used in the equivalent squared form `(EΔ²)² ≤ E|Δ| · E|Δ|³` (`cs_step`)
4. Jensen (`x ↦ x^{3/4}` concave): `E|Δ|³ ≤ (EΔ⁴)^{3/4}`  (`jensen_step`)
5. Combine: `E|Δ| ≥ (EΔ²)²/(EΔ⁴)^{3/4} ≥ (s/4)^{1/2} 3^{-3/4} = √s/(2·3^{3/4})
   ≥ √s/5`  (since `(2·3^{3/4})⁴ = 432 ≤ 625 = 5⁴`)    (`lemma_mad`)

Moment identities 1–2 are proved by induction on `s` via the Pascal-rule
reindexing `pascal_sum` (the algebraic content of `Δ_{s+1} = Δ_s ± 1/2`).
-/
import Mathlib.Data.Nat.Choose.Sum
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Analysis.MeanInequalitiesPow
import Mathlib.Data.Real.Sqrt

open Finset

namespace AppB

/-- Pascal-rule reindexing:
`∑_{k ≤ s+1} C(s+1,k) f(k) = ∑_{k ≤ s} C(s,k) (f(k) + f(k+1))`. -/
theorem pascal_sum (s : ℕ) (f : ℕ → ℝ) :
    ∑ k ∈ range (s + 2), ((s + 1).choose k : ℝ) * f k
      = ∑ k ∈ range (s + 1), (s.choose k : ℝ) * (f k + f (k + 1)) := by
  have key : ∑ k ∈ range (s + 2), (s.choose k : ℝ) * f k
      = ∑ k ∈ range (s + 1), (s.choose k : ℝ) * f k := by
    rw [Finset.sum_range_succ, Nat.choose_succ_self]
    simp
  have h0 : (∑ k ∈ range (s + 1), (s.choose (k + 1) : ℝ) * f (k + 1)) + f 0
      = ∑ k ∈ range (s + 1), (s.choose k : ℝ) * f k := by
    have h := Finset.sum_range_succ' (fun k => (s.choose k : ℝ) * f k) (s + 1)
    simp only [Nat.choose_zero_right, Nat.cast_one, one_mul] at h
    rw [key] at h
    linarith
  calc ∑ k ∈ range (s + 2), ((s + 1).choose k : ℝ) * f k
      = (∑ k ∈ range (s + 1), ((s + 1).choose (k + 1) : ℝ) * f (k + 1))
          + ((s + 1).choose 0 : ℝ) * f 0 :=
        Finset.sum_range_succ' (fun k => ((s + 1).choose k : ℝ) * f k) (s + 1)
    _ = (∑ k ∈ range (s + 1),
          ((s.choose k : ℝ) * f (k + 1) + (s.choose (k + 1) : ℝ) * f (k + 1)))
          + f 0 := by
        congr 1
        · refine Finset.sum_congr rfl fun k _ => ?_
          rw [Nat.choose_succ_succ']
          push_cast
          ring
        · simp
    _ = (∑ k ∈ range (s + 1), (s.choose k : ℝ) * f (k + 1))
          + ((∑ k ∈ range (s + 1), (s.choose (k + 1) : ℝ) * f (k + 1)) + f 0) := by
        rw [Finset.sum_add_distrib]
        ring
    _ = (∑ k ∈ range (s + 1), (s.choose k : ℝ) * f (k + 1))
          + ∑ k ∈ range (s + 1), (s.choose k : ℝ) * f k := by rw [h0]
    _ = ∑ k ∈ range (s + 1), (s.choose k : ℝ) * (f k + f (k + 1)) := by
        rw [← Finset.sum_add_distrib]
        exact Finset.sum_congr rfl fun k _ => by ring

/-- `∑_k C(s,k) = 2^s`, cast to ℝ. -/
theorem sum_choose_real (s : ℕ) :
    ∑ k ∈ range (s + 1), (s.choose k : ℝ) = (2 : ℝ) ^ s := by
  exact_mod_cast Nat.sum_range_choose s

/-- **Second central moment** of Bin(s,1/2): `EΔ² = s/4`
(unnormalized: `∑_k C(s,k)(k - s/2)² = 2^s · s/4`). -/
theorem sum_choose_sq (s : ℕ) :
    ∑ k ∈ range (s + 1), (s.choose k : ℝ) * ((k : ℝ) - (s : ℝ) / 2) ^ 2
      = (2 : ℝ) ^ s * ((s : ℝ) / 4) := by
  induction s with
  | zero => norm_num
  | succ s ih =>
    have hpt : ∀ k ∈ range (s + 1),
        (s.choose k : ℝ) * (((k : ℝ) - ((s + 1 : ℕ) : ℝ) / 2) ^ 2
          + (((k + 1 : ℕ) : ℝ) - ((s + 1 : ℕ) : ℝ) / 2) ^ 2)
        = 2 * ((s.choose k : ℝ) * ((k : ℝ) - (s : ℝ) / 2) ^ 2)
          + (1 / 2) * (s.choose k : ℝ) := by
      intro k _
      push_cast
      ring
    calc ∑ k ∈ range (s + 1 + 1),
          ((s + 1).choose k : ℝ) * ((k : ℝ) - ((s + 1 : ℕ) : ℝ) / 2) ^ 2
        = ∑ k ∈ range (s + 1), (s.choose k : ℝ)
            * (((k : ℝ) - ((s + 1 : ℕ) : ℝ) / 2) ^ 2
              + (((k + 1 : ℕ) : ℝ) - ((s + 1 : ℕ) : ℝ) / 2) ^ 2) :=
          pascal_sum s (fun k => ((k : ℝ) - ((s + 1 : ℕ) : ℝ) / 2) ^ 2)
      _ = ∑ k ∈ range (s + 1),
            (2 * ((s.choose k : ℝ) * ((k : ℝ) - (s : ℝ) / 2) ^ 2)
              + (1 / 2) * (s.choose k : ℝ)) :=
          Finset.sum_congr rfl hpt
      _ = 2 * (∑ k ∈ range (s + 1), (s.choose k : ℝ) * ((k : ℝ) - (s : ℝ) / 2) ^ 2)
            + (1 / 2) * ∑ k ∈ range (s + 1), (s.choose k : ℝ) := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
      _ = 2 * ((2 : ℝ) ^ s * ((s : ℝ) / 4)) + (1 / 2) * (2 : ℝ) ^ s := by
          rw [ih, sum_choose_real]
      _ = (2 : ℝ) ^ (s + 1) * (((s + 1 : ℕ) : ℝ) / 4) := by
          push_cast
          ring

/-- **Fourth central moment** of Bin(s,1/2): `EΔ⁴ = s(3s-2)/16`
(unnormalized), which equals the paper's `(s/4)(1 + 3(s-2)/4)`. -/
theorem sum_choose_quart (s : ℕ) :
    ∑ k ∈ range (s + 1), (s.choose k : ℝ) * ((k : ℝ) - (s : ℝ) / 2) ^ 4
      = (2 : ℝ) ^ s * ((s : ℝ) * (3 * (s : ℝ) - 2) / 16) := by
  induction s with
  | zero => norm_num
  | succ s ih =>
    have hpt : ∀ k ∈ range (s + 1),
        (s.choose k : ℝ) * (((k : ℝ) - ((s + 1 : ℕ) : ℝ) / 2) ^ 4
          + (((k + 1 : ℕ) : ℝ) - ((s + 1 : ℕ) : ℝ) / 2) ^ 4)
        = 2 * ((s.choose k : ℝ) * ((k : ℝ) - (s : ℝ) / 2) ^ 4)
          + 3 * ((s.choose k : ℝ) * ((k : ℝ) - (s : ℝ) / 2) ^ 2)
          + (1 / 8) * (s.choose k : ℝ) := by
      intro k _
      push_cast
      ring
    calc ∑ k ∈ range (s + 1 + 1),
          ((s + 1).choose k : ℝ) * ((k : ℝ) - ((s + 1 : ℕ) : ℝ) / 2) ^ 4
        = ∑ k ∈ range (s + 1), (s.choose k : ℝ)
            * (((k : ℝ) - ((s + 1 : ℕ) : ℝ) / 2) ^ 4
              + (((k + 1 : ℕ) : ℝ) - ((s + 1 : ℕ) : ℝ) / 2) ^ 4) :=
          pascal_sum s (fun k => ((k : ℝ) - ((s + 1 : ℕ) : ℝ) / 2) ^ 4)
      _ = ∑ k ∈ range (s + 1),
            (2 * ((s.choose k : ℝ) * ((k : ℝ) - (s : ℝ) / 2) ^ 4)
              + 3 * ((s.choose k : ℝ) * ((k : ℝ) - (s : ℝ) / 2) ^ 2)
              + (1 / 8) * (s.choose k : ℝ)) :=
          Finset.sum_congr rfl hpt
      _ = 2 * (∑ k ∈ range (s + 1), (s.choose k : ℝ) * ((k : ℝ) - (s : ℝ) / 2) ^ 4)
            + 3 * (∑ k ∈ range (s + 1), (s.choose k : ℝ) * ((k : ℝ) - (s : ℝ) / 2) ^ 2)
            + (1 / 8) * ∑ k ∈ range (s + 1), (s.choose k : ℝ) := by
          rw [Finset.sum_add_distrib, Finset.sum_add_distrib,
            ← Finset.mul_sum, ← Finset.mul_sum, ← Finset.mul_sum]
      _ = 2 * ((2 : ℝ) ^ s * ((s : ℝ) * (3 * (s : ℝ) - 2) / 16))
            + 3 * ((2 : ℝ) ^ s * ((s : ℝ) / 4)) + (1 / 8) * (2 : ℝ) ^ s := by
          rw [ih, sum_choose_sq, sum_choose_real]
      _ = (2 : ℝ) ^ (s + 1)
            * (((s + 1 : ℕ) : ℝ) * (3 * ((s + 1 : ℕ) : ℝ) - 2) / 16) := by
          push_cast
          ring

/-- **Cauchy–Schwarz step** (paper step 3), squared form:
`(∑ C(s,k) (k-s/2)²)² ≤ (∑ C(s,k)|k-s/2|) · (∑ C(s,k)|k-s/2|³)`. -/
theorem cs_step (s : ℕ) :
    (∑ k ∈ range (s + 1), (s.choose k : ℝ) * ((k : ℝ) - (s : ℝ) / 2) ^ 2) ^ 2
      ≤ (∑ k ∈ range (s + 1), (s.choose k : ℝ) * |(k : ℝ) - (s : ℝ) / 2|)
        * ∑ k ∈ range (s + 1), (s.choose k : ℝ) * |(k : ℝ) - (s : ℝ) / 2| ^ 3 := by
  have h := Finset.sum_mul_sq_le_sq_mul_sq (range (s + 1))
      (fun k => Real.sqrt ((s.choose k : ℝ) * |(k : ℝ) - (s : ℝ) / 2|))
      (fun k => Real.sqrt ((s.choose k : ℝ) * |(k : ℝ) - (s : ℝ) / 2| ^ 3))
  have hFG : ∀ k ∈ range (s + 1),
      Real.sqrt ((s.choose k : ℝ) * |(k : ℝ) - (s : ℝ) / 2|)
        * Real.sqrt ((s.choose k : ℝ) * |(k : ℝ) - (s : ℝ) / 2| ^ 3)
      = (s.choose k : ℝ) * ((k : ℝ) - (s : ℝ) / 2) ^ 2 := by
    intro k _
    rw [← Real.sqrt_mul (by positivity)]
    have hsq : (s.choose k : ℝ) * |(k : ℝ) - (s : ℝ) / 2|
        * ((s.choose k : ℝ) * |(k : ℝ) - (s : ℝ) / 2| ^ 3)
        = ((s.choose k : ℝ) * ((k : ℝ) - (s : ℝ) / 2) ^ 2) ^ 2 := by
      calc (s.choose k : ℝ) * |(k : ℝ) - (s : ℝ) / 2|
            * ((s.choose k : ℝ) * |(k : ℝ) - (s : ℝ) / 2| ^ 3)
          = (s.choose k : ℝ) ^ 2 * (|(k : ℝ) - (s : ℝ) / 2| ^ 2) ^ 2 := by ring
        _ = (s.choose k : ℝ) ^ 2 * (((k : ℝ) - (s : ℝ) / 2) ^ 2) ^ 2 := by
            rw [sq_abs]
        _ = ((s.choose k : ℝ) * ((k : ℝ) - (s : ℝ) / 2) ^ 2) ^ 2 := by ring
    rw [hsq, Real.sqrt_sq (by positivity)]
  have hF2 : ∀ k ∈ range (s + 1),
      Real.sqrt ((s.choose k : ℝ) * |(k : ℝ) - (s : ℝ) / 2|) ^ 2
        = (s.choose k : ℝ) * |(k : ℝ) - (s : ℝ) / 2| :=
    fun k _ => Real.sq_sqrt (by positivity)
  have hG2 : ∀ k ∈ range (s + 1),
      Real.sqrt ((s.choose k : ℝ) * |(k : ℝ) - (s : ℝ) / 2| ^ 3) ^ 2
        = (s.choose k : ℝ) * |(k : ℝ) - (s : ℝ) / 2| ^ 3 :=
    fun k _ => Real.sq_sqrt (by positivity)
  rw [Finset.sum_congr rfl hFG, Finset.sum_congr rfl hF2,
    Finset.sum_congr rfl hG2] at h
  exact h

/-- **Jensen step** (paper step 4): `E|Δ|³ ≤ (EΔ⁴)^{3/4}` (normalized
binomial expectations), via concavity of `x ↦ x^{3/4}`
(`Real.arith_mean_le_rpow_mean` with `p = 4/3`). -/
theorem jensen_step (s : ℕ) :
    (∑ k ∈ range (s + 1), (s.choose k : ℝ) * |(k : ℝ) - (s : ℝ) / 2| ^ 3) / 2 ^ s
      ≤ ((∑ k ∈ range (s + 1), (s.choose k : ℝ) * ((k : ℝ) - (s : ℝ) / 2) ^ 4)
          / 2 ^ s) ^ (3 / 4 : ℝ) := by
  have hw' : ∑ k ∈ range (s + 1), (s.choose k : ℝ) / 2 ^ s = 1 := by
    rw [← Finset.sum_div, sum_choose_real]
    exact div_self (by positivity)
  have h := Real.arith_mean_le_rpow_mean (range (s + 1))
      (fun k => (s.choose k : ℝ) / 2 ^ s)
      (fun k => |(k : ℝ) - (s : ℝ) / 2| ^ 3)
      (fun k _ => by positivity) hw' (fun k _ => by positivity)
      (p := (4 / 3 : ℝ)) (by norm_num)
  have habs : ∀ k : ℕ, ((|(k : ℝ) - (s : ℝ) / 2| ^ 3 : ℝ)) ^ ((4 : ℝ) / 3)
      = ((k : ℝ) - (s : ℝ) / 2) ^ 4 := by
    intro k
    calc ((|(k : ℝ) - (s : ℝ) / 2| ^ 3 : ℝ)) ^ ((4 : ℝ) / 3)
        = (|(k : ℝ) - (s : ℝ) / 2| ^ ((3 : ℕ) : ℝ)) ^ ((4 : ℝ) / 3) := by
          rw [Real.rpow_natCast]
      _ = |(k : ℝ) - (s : ℝ) / 2| ^ (((3 : ℕ) : ℝ) * ((4 : ℝ) / 3)) :=
          (Real.rpow_mul (abs_nonneg _) _ _).symm
      _ = |(k : ℝ) - (s : ℝ) / 2| ^ (((4 : ℕ) : ℝ)) := by norm_num
      _ = |(k : ℝ) - (s : ℝ) / 2| ^ (4 : ℕ) := Real.rpow_natCast _ 4
      _ = ((k : ℝ) - (s : ℝ) / 2) ^ 4 := Even.pow_abs ⟨2, rfl⟩ _
  have hz : ∀ k ∈ range (s + 1),
      (s.choose k : ℝ) / 2 ^ s * (|(k : ℝ) - (s : ℝ) / 2| ^ 3) ^ ((4 : ℝ) / 3)
        = (s.choose k : ℝ) * ((k : ℝ) - (s : ℝ) / 2) ^ 4 / 2 ^ s := by
    intro k _
    rw [habs k]
    ring
  have hw : ∀ k ∈ range (s + 1),
      (s.choose k : ℝ) / 2 ^ s * (|(k : ℝ) - (s : ℝ) / 2| ^ 3)
        = (s.choose k : ℝ) * |(k : ℝ) - (s : ℝ) / 2| ^ 3 / 2 ^ s :=
    fun k _ => by ring
  rw [Finset.sum_congr rfl hw, Finset.sum_congr rfl hz,
    ← Finset.sum_div, ← Finset.sum_div] at h
  have hexp : (1 : ℝ) / (4 / 3) = 3 / 4 := by norm_num
  rwa [hexp] at h

/-- **Lemma B.5 (`lem:mad`), in full.**  For `X ~ Bin(s, 1/2)` with `s ≥ 1` and
`Δ = X - s/2`:  `E|Δ| ≥ √s / 5`, the expectation written as the explicit
binomial sum `E|Δ| = (∑_k C(s,k) |k - s/2|) / 2^s`. -/
theorem lemma_mad (s : ℕ) (hs : 1 ≤ s) :
    Real.sqrt s / 5
      ≤ (∑ k ∈ range (s + 1), (s.choose k : ℝ) * |(k : ℝ) - (s : ℝ) / 2|) / 2 ^ s := by
  have hspos : (0 : ℝ) < s := by exact_mod_cast hs
  have hs1 : (1 : ℝ) ≤ s := by exact_mod_cast hs
  have h2pos : (0 : ℝ) < (2 : ℝ) ^ s := by positivity
  set S1 := ∑ k ∈ range (s + 1), (s.choose k : ℝ) * |(k : ℝ) - (s : ℝ) / 2| with hS1def
  set S3 := ∑ k ∈ range (s + 1), (s.choose k : ℝ) * |(k : ℝ) - (s : ℝ) / 2| ^ 3
    with hS3def
  set A := (3 * ((s : ℝ) / 4) ^ 2) ^ ((3 : ℝ) / 4) with hAdef
  -- normalized second and fourth moments
  have hm2 : (∑ k ∈ range (s + 1), (s.choose k : ℝ) * ((k : ℝ) - (s : ℝ) / 2) ^ 2)
      / 2 ^ s = (s : ℝ) / 4 := by
    rw [sum_choose_sq, mul_div_cancel_left₀ _ (ne_of_gt h2pos)]
  have hm4 : (∑ k ∈ range (s + 1), (s.choose k : ℝ) * ((k : ℝ) - (s : ℝ) / 2) ^ 4)
      / 2 ^ s = (s : ℝ) * (3 * (s : ℝ) - 2) / 16 := by
    rw [sum_choose_quart, mul_div_cancel_left₀ _ (ne_of_gt h2pos)]
  have hm1nn : 0 ≤ S1 / 2 ^ s :=
    div_nonneg (Finset.sum_nonneg fun k _ => by positivity) h2pos.le
  -- Cauchy–Schwarz, normalized:  (s/4)² ≤ (E|Δ|)(E|Δ|³)
  have hCS : ((s : ℝ) / 4) ^ 2 ≤ (S1 / 2 ^ s) * (S3 / 2 ^ s) := by
    rw [← hm2, div_pow, div_mul_div_comm, ← pow_two]
    gcongr
    exact cs_step s
  -- Jensen + EΔ⁴ ≤ 3(s/4)²:  E|Δ|³ ≤ A
  have hJ : S3 / 2 ^ s ≤ ((s : ℝ) * (3 * (s : ℝ) - 2) / 16) ^ ((3 : ℝ) / 4) := by
    have h := jensen_step s
    rwa [hm4] at h
  have hm4le : (s : ℝ) * (3 * (s : ℝ) - 2) / 16 ≤ 3 * ((s : ℝ) / 4) ^ 2 := by
    nlinarith
  have hm4nn : (0 : ℝ) ≤ (s : ℝ) * (3 * (s : ℝ) - 2) / 16 := by nlinarith
  have hA' : ((s : ℝ) * (3 * (s : ℝ) - 2) / 16) ^ ((3 : ℝ) / 4) ≤ A :=
    Real.rpow_le_rpow hm4nn hm4le (by norm_num)
  have hApos : 0 < A := Real.rpow_pos_of_pos (by positivity) _
  -- chain:  (s/4)² ≤ (E|Δ|)·A
  have hchain : ((s : ℝ) / 4) ^ 2 ≤ (S1 / 2 ^ s) * A :=
    le_trans hCS (mul_le_mul_of_nonneg_left (le_trans hJ hA') hm1nn)
  -- constant step:  (√s/5)·A ≤ (s/4)²,  i.e.  2·3^{3/4} ≤ 5  (432 ≤ 625)
  have hx2 : Real.sqrt s ^ 2 = s := Real.sq_sqrt hspos.le
  have hA4 : A ^ (4 : ℕ) = (3 * ((s : ℝ) / 4) ^ 2) ^ (3 : ℕ) := by
    have hXnn : (0 : ℝ) ≤ 3 * ((s : ℝ) / 4) ^ 2 := by positivity
    calc A ^ (4 : ℕ)
        = ((3 * ((s : ℝ) / 4) ^ 2) ^ ((3 : ℝ) / 4)) ^ (((4 : ℕ) : ℝ)) := by
          rw [Real.rpow_natCast]
      _ = (3 * ((s : ℝ) / 4) ^ 2) ^ (((3 : ℝ) / 4) * ((4 : ℕ) : ℝ)) :=
          (Real.rpow_mul hXnn _ _).symm
      _ = (3 * ((s : ℝ) / 4) ^ 2) ^ (((3 : ℕ) : ℝ)) := by norm_num
      _ = (3 * ((s : ℝ) / 4) ^ 2) ^ (3 : ℕ) := Real.rpow_natCast _ 3
  have key : Real.sqrt s / 5 * A ≤ ((s : ℝ) / 4) ^ 2 := by
    have h2 : (0 : ℝ) ≤ ((s : ℝ) / 4) ^ 2 := by positivity
    have h4 : (Real.sqrt s / 5 * A) ^ (4 : ℕ) ≤ (((s : ℝ) / 4) ^ 2) ^ (4 : ℕ) := by
      have expand : (Real.sqrt s / 5 * A) ^ (4 : ℕ)
          = (Real.sqrt s ^ 2) ^ 2 * A ^ (4 : ℕ) / 625 := by ring
      rw [expand, hx2, hA4]
      have h8 : (0 : ℝ) ≤ (s : ℝ) ^ 8 := by positivity
      nlinarith [h8]
    exact le_of_pow_le_pow_left₀ (by norm_num) h2 h4
  -- conclude:  √s/5 ≤ E|Δ|
  have hfin : Real.sqrt s / 5 * A ≤ (S1 / 2 ^ s) * A := le_trans key hchain
  exact le_of_mul_le_mul_right hfin hApos

end AppB
