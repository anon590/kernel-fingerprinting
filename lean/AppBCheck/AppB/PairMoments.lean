/-
# Theorem B.6(b): per-pair anti-concentration, fully verified on the pool space

`pair_diff_lower` (in `AppB.TheoremLower`) verifies the paper's route to
`E|n_{2t-1} - n_{2t}| ≥ (1/4)√(N/m)` modulo the standard multinomial bridge
("conditionally on `s = n_{2t-1}+n_{2t}`, `n_{2t-1} ~ Bin(s,1/2)`").

This file *eliminates that bridge*: it proves the same inequality directly
for the uniform measure on the pool space `Fin N → B` (`card B = m`), via
exact moment computations on the product space:

  `E S = E S³ = 0`,  `E S² = 2N/m`,  `E S⁴ = 12N(N-1)/m² + 2N/m ≤ (25/8)(2N/m)²`

for `S = n_{b₁} - n_{b₂} = ∑_i ε(ω_i)`, followed by the same
Cauchy–Schwarz + Jensen chain as Lemma B.5 (`mad_lower` below):
`E|S| ≥ (ES²)²/(ES⁴)^{3/4} ≥ √(2N/m)/3 ≥ (1/4)√(N/m)`.

Everything here is fully formal; combined with `attack_inflation_mean` it
yields `thm_lower_b_uniform`: the (b) lower bound `EZ ≥ (σ/8)√(m/N)` as a
statement about the actual uniform pool measure, with no probabilistic
hypotheses left.
-/
import AppB.TheoremLower

open Finset

namespace AppB

/-! ## Generic fourth-moment anti-concentration chain (Lemma B.5's engine) -/

private theorem abs_cube_rpow (x : ℝ) : ((|x| ^ 3 : ℝ)) ^ ((4 : ℝ) / 3) = x ^ 4 := by
  calc ((|x| ^ 3 : ℝ)) ^ ((4 : ℝ) / 3)
      = (|x| ^ ((3 : ℕ) : ℝ)) ^ ((4 : ℝ) / 3) := by rw [Real.rpow_natCast]
    _ = |x| ^ (((3 : ℕ) : ℝ) * ((4 : ℝ) / 3)) := (Real.rpow_mul (abs_nonneg _) _ _).symm
    _ = |x| ^ (((4 : ℕ) : ℝ)) := by norm_num
    _ = |x| ^ (4 : ℕ) := Real.rpow_natCast _ 4
    _ = x ^ 4 := Even.pow_abs ⟨2, rfl⟩ _

/-- Cauchy–Schwarz: `(E S²)² ≤ E|S| · E|S|³` (unnormalized). -/
theorem cs_generic {ι : Type*} [Fintype ι] (g : ι → ℝ) :
    (∑ i : ι, g i ^ 2) ^ 2 ≤ (∑ i : ι, |g i|) * ∑ i : ι, |g i| ^ 3 := by
  have h := Finset.sum_mul_sq_le_sq_mul_sq Finset.univ
      (fun i => Real.sqrt |g i|) (fun i => Real.sqrt (|g i| ^ 3))
  have hFG : ∀ i ∈ Finset.univ (α := ι),
      Real.sqrt |g i| * Real.sqrt (|g i| ^ 3) = g i ^ 2 := by
    intro i _
    rw [← Real.sqrt_mul (abs_nonneg _)]
    have hsq : |g i| * |g i| ^ 3 = (g i ^ 2) ^ 2 := by
      calc |g i| * |g i| ^ 3 = (|g i| ^ 2) ^ 2 := by ring
        _ = (g i ^ 2) ^ 2 := by rw [sq_abs]
    rw [hsq, Real.sqrt_sq (by positivity)]
  have hF2 : ∀ i ∈ Finset.univ (α := ι), Real.sqrt |g i| ^ 2 = |g i| :=
    fun i _ => Real.sq_sqrt (abs_nonneg _)
  have hG2 : ∀ i ∈ Finset.univ (α := ι), Real.sqrt (|g i| ^ 3) ^ 2 = |g i| ^ 3 :=
    fun i _ => Real.sq_sqrt (by positivity)
  rw [Finset.sum_congr rfl hFG, Finset.sum_congr rfl hF2,
    Finset.sum_congr rfl hG2] at h
  exact h

/-- Jensen (`x ↦ x^{3/4}` concave): `E|S|³ ≤ (E S⁴)^{3/4}` (normalized). -/
theorem jensen_generic {ι : Type*} [Fintype ι] [Nonempty ι] (g : ι → ℝ) :
    (∑ i : ι, |g i| ^ 3) / (Fintype.card ι : ℝ)
      ≤ ((∑ i : ι, g i ^ 4) / (Fintype.card ι : ℝ)) ^ (3 / 4 : ℝ) := by
  have hW : (0 : ℝ) < (Fintype.card ι : ℝ) := by exact_mod_cast Fintype.card_pos
  have hw' : ∑ _i : ι, (1 : ℝ) / (Fintype.card ι : ℝ) = 1 := by
    rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
    field_simp
  have h := Real.arith_mean_le_rpow_mean Finset.univ
      (fun _ => (1 : ℝ) / (Fintype.card ι : ℝ)) (fun i => |g i| ^ 3)
      (fun i _ => by positivity) hw' (fun i _ => by positivity)
      (p := (4 / 3 : ℝ)) (by norm_num)
  have hz : ∀ i ∈ Finset.univ (α := ι),
      (1 : ℝ) / (Fintype.card ι : ℝ) * (|g i| ^ 3) ^ ((4 : ℝ) / 3)
        = g i ^ 4 / (Fintype.card ι : ℝ) := by
    intro i _
    rw [abs_cube_rpow]
    ring
  have hw : ∀ i ∈ Finset.univ (α := ι),
      (1 : ℝ) / (Fintype.card ι : ℝ) * (|g i| ^ 3)
        = |g i| ^ 3 / (Fintype.card ι : ℝ) :=
    fun i _ => by ring
  rw [Finset.sum_congr rfl hw, Finset.sum_congr rfl hz,
    ← Finset.sum_div, ← Finset.sum_div] at h
  have hexp : (1 : ℝ) / (4 / 3) = 3 / 4 := by norm_num
  rwa [hexp] at h

/-- **Fourth-moment anti-concentration** (the engine of Lemma B.5, generic):
if `E S² = v ≥ 8` and `E S⁴ ≤ (25/8)v²`, then `E|S| ≥ √v / 3`. -/
theorem mad_lower {ι : Type*} [Fintype ι] [Nonempty ι] (g : ι → ℝ) (v : ℝ)
    (hv : 8 ≤ v)
    (h2 : ∑ i : ι, g i ^ 2 = v * (Fintype.card ι : ℝ))
    (h4 : ∑ i : ι, g i ^ 4 ≤ 25 / 8 * v ^ 2 * (Fintype.card ι : ℝ)) :
    Real.sqrt v / 3 ≤ (∑ i : ι, |g i|) / (Fintype.card ι : ℝ) := by
  have hW : (0 : ℝ) < (Fintype.card ι : ℝ) := by exact_mod_cast Fintype.card_pos
  have hv0 : (0 : ℝ) < v := by linarith
  set A := (25 / 8 * v ^ 2) ^ ((3 : ℝ) / 4) with hAdef
  have h2' : (∑ i : ι, g i ^ 2) / (Fintype.card ι : ℝ) = v := by
    rw [h2]
    field_simp
  have hm1nn : 0 ≤ (∑ i : ι, |g i|) / (Fintype.card ι : ℝ) :=
    div_nonneg (Finset.sum_nonneg fun i _ => abs_nonneg _) hW.le
  -- Cauchy–Schwarz, normalized
  have hCS : v ^ 2 ≤ ((∑ i : ι, |g i|) / (Fintype.card ι : ℝ))
      * ((∑ i : ι, |g i| ^ 3) / (Fintype.card ι : ℝ)) := by
    rw [← h2', div_pow, div_mul_div_comm, ← pow_two]
    gcongr
    exact cs_generic g
  -- Jensen + the fourth-moment bound
  have hJ : (∑ i : ι, |g i| ^ 3) / (Fintype.card ι : ℝ) ≤ A := by
    refine le_trans (jensen_generic g) ?_
    rw [hAdef]
    apply Real.rpow_le_rpow (by positivity) ?_ (by norm_num)
    rw [div_le_iff₀ hW]
    linarith
  have hApos : 0 < A := Real.rpow_pos_of_pos (by positivity) _
  have hchain : v ^ 2 ≤ ((∑ i : ι, |g i|) / (Fintype.card ι : ℝ)) * A :=
    le_trans hCS (mul_le_mul_of_nonneg_left hJ hm1nn)
  -- constant step: (√v/3)·A ≤ v²  (4th powers: 15625/41472 ≤ 1)
  have hx2 : Real.sqrt v ^ 2 = v := Real.sq_sqrt hv0.le
  have hA4 : A ^ (4 : ℕ) = (25 / 8 * v ^ 2) ^ (3 : ℕ) := by
    have hXnn : (0 : ℝ) ≤ 25 / 8 * v ^ 2 := by positivity
    calc A ^ (4 : ℕ)
        = ((25 / 8 * v ^ 2) ^ ((3 : ℝ) / 4)) ^ (((4 : ℕ) : ℝ)) := by
          rw [Real.rpow_natCast]
      _ = (25 / 8 * v ^ 2) ^ (((3 : ℝ) / 4) * ((4 : ℕ) : ℝ)) :=
          (Real.rpow_mul hXnn _ _).symm
      _ = (25 / 8 * v ^ 2) ^ (((3 : ℕ) : ℝ)) := by norm_num
      _ = (25 / 8 * v ^ 2) ^ (3 : ℕ) := Real.rpow_natCast _ 3
  have key : Real.sqrt v / 3 * A ≤ v ^ 2 := by
    have h2nn : (0 : ℝ) ≤ v ^ 2 := by positivity
    have h4le : (Real.sqrt v / 3 * A) ^ (4 : ℕ) ≤ (v ^ 2) ^ (4 : ℕ) := by
      have expand : (Real.sqrt v / 3 * A) ^ (4 : ℕ)
          = (Real.sqrt v ^ 2) ^ 2 * A ^ (4 : ℕ) / 81 := by ring
      rw [expand, hx2, hA4]
      nlinarith [pow_nonneg hv0.le 8]
    exact le_of_pow_le_pow_left₀ (by norm_num) h2nn h4le
  have hfin : Real.sqrt v / 3 * A ≤ ((∑ i : ι, |g i|) / (Fintype.card ι : ℝ)) * A :=
    le_trans key hchain
  exact le_of_mul_le_mul_right hfin hApos

/-! ## The pair-difference statistic on the pool space -/

variable {B : Type*} [Fintype B] [DecidableEq B]

/-- Per-instance offset: `+1` on cell `b₁`, `-1` on cell `b₂`, `0` elsewhere. -/
noncomputable def eps (b₁ b₂ : B) (x : B) : ℝ :=
  (if x = b₁ then 1 else 0) - (if x = b₂ then 1 else 0)

/-- The pair-count difference `S(ω) = n_{b₁}(ω) - n_{b₂}(ω)`. -/
noncomputable def Spair (b₁ b₂ : B) {N : ℕ} (ω : Fin N → B) : ℝ :=
  ∑ i, eps b₁ b₂ (ω i)

theorem Spair_eq_cnt (b₁ b₂ : B) {N : ℕ} (ω : Fin N → B) :
    Spair b₁ b₂ ω = cnt ω b₁ - cnt ω b₂ := by
  unfold Spair eps cnt
  rw [← Finset.sum_sub_distrib]

/-! ### Per-instance moments -/

theorem eps_sum (b₁ b₂ : B) (hb : b₁ ≠ b₂) : ∑ x : B, eps b₁ b₂ x = 0 := by
  unfold eps
  rw [Finset.sum_sub_distrib]
  simp

theorem eps_sq_sum (b₁ b₂ : B) (hb : b₁ ≠ b₂) :
    ∑ x : B, eps b₁ b₂ x ^ 2 = 2 := by
  have hpt : ∀ x : B, eps b₁ b₂ x ^ 2
      = (if x = b₁ then (1 : ℝ) else 0) + (if x = b₂ then (1 : ℝ) else 0) := by
    intro x
    unfold eps
    split_ifs with h1 h2
    · exact absurd (h1 ▸ h2) (by simpa [h1] using hb)
    · norm_num
    · norm_num
    · norm_num
  rw [Finset.sum_congr rfl fun x _ => hpt x, Finset.sum_add_distrib]
  simp
  norm_num

theorem eps_cube_sum (b₁ b₂ : B) (hb : b₁ ≠ b₂) :
    ∑ x : B, eps b₁ b₂ x ^ 3 = 0 := by
  have hpt : ∀ x : B, eps b₁ b₂ x ^ 3 = eps b₁ b₂ x := by
    intro x
    unfold eps
    split_ifs <;> norm_num
  rw [Finset.sum_congr rfl fun x _ => hpt x]
  exact eps_sum b₁ b₂ hb

theorem eps_quart_sum (b₁ b₂ : B) (hb : b₁ ≠ b₂) :
    ∑ x : B, eps b₁ b₂ x ^ 4 = 2 := by
  have hpt : ∀ x : B, eps b₁ b₂ x ^ 4 = eps b₁ b₂ x ^ 2 := by
    intro x
    unfold eps
    split_ifs <;> norm_num
  rw [Finset.sum_congr rfl fun x _ => hpt x]
  exact eps_sq_sum b₁ b₂ hb

/-! ### Peeling one instance -/

theorem Spair_cons (b₁ b₂ : B) {N : ℕ} (x : B) (ω : Fin N → B) :
    Spair b₁ b₂ (Fin.cons x ω) = eps b₁ b₂ x + Spair b₁ b₂ ω := by
  unfold Spair
  rw [Fin.sum_univ_succ]
  simp [Fin.cons_zero, Fin.cons_succ]

theorem sum_pool_succ (b₁ b₂ : B) (N : ℕ) (f : ℝ → ℝ) :
    ∑ ω : Fin (N + 1) → B, f (Spair b₁ b₂ ω)
      = ∑ ω : Fin N → B, ∑ x : B, f (Spair b₁ b₂ ω + eps b₁ b₂ x) := by
  have h := Equiv.sum_comp (Fin.consEquiv (fun _ : Fin (N + 1) => B))
      (fun ω => f (Spair b₁ b₂ ω))
  rw [← h, Fintype.sum_prod_type, Finset.sum_comm]
  refine Finset.sum_congr rfl fun ω _ => Finset.sum_congr rfl fun x _ => ?_
  have hc : Spair b₁ b₂ ((Fin.consEquiv (fun _ : Fin (N + 1) => B)) (x, ω))
      = eps b₁ b₂ x + Spair b₁ b₂ ω := Spair_cons b₁ b₂ x ω
  rw [hc, add_comm]

/-! ### Pool moments (exact) -/

theorem pool_S1 (b₁ b₂ : B) (hb : b₁ ≠ b₂) (N : ℕ) :
    ∑ ω : Fin N → B, Spair b₁ b₂ ω = 0 := by
  induction N with
  | zero => simp [Spair]
  | succ N ih =>
    calc ∑ ω : Fin (N + 1) → B, Spair b₁ b₂ ω
        = ∑ ω : Fin N → B, ∑ x : B, (Spair b₁ b₂ ω + eps b₁ b₂ x) :=
          sum_pool_succ b₁ b₂ N (fun z => z)
      _ = ∑ ω : Fin N → B, ((Fintype.card B : ℝ) * Spair b₁ b₂ ω) := by
          refine Finset.sum_congr rfl fun ω _ => ?_
          rw [Finset.sum_add_distrib, Finset.sum_const, eps_sum b₁ b₂ hb,
            Finset.card_univ, nsmul_eq_mul, add_zero]
      _ = (Fintype.card B : ℝ) * ∑ ω : Fin N → B, Spair b₁ b₂ ω := by
          rw [← Finset.mul_sum]
      _ = 0 := by rw [ih, mul_zero]

theorem pool_S2 (b₁ b₂ : B) (hb : b₁ ≠ b₂) (N : ℕ) :
    (Fintype.card B : ℝ) * ∑ ω : Fin N → B, Spair b₁ b₂ ω ^ 2
      = 2 * N * (Fintype.card B : ℝ) ^ N := by
  induction N with
  | zero => simp [Spair]
  | succ N ih =>
    have hcardFun : (Fintype.card (Fin N → B) : ℝ) = (Fintype.card B : ℝ) ^ N := by
      rw [Fintype.card_fun, Fintype.card_fin]
      push_cast
      ring
    have hinner : ∀ ω : Fin N → B,
        ∑ x : B, (Spair b₁ b₂ ω + eps b₁ b₂ x) ^ 2
          = (Fintype.card B : ℝ) * Spair b₁ b₂ ω ^ 2 + 2 := by
      intro ω
      have hexp : ∀ x : B, (Spair b₁ b₂ ω + eps b₁ b₂ x) ^ 2
          = Spair b₁ b₂ ω ^ 2 + 2 * Spair b₁ b₂ ω * eps b₁ b₂ x
            + eps b₁ b₂ x ^ 2 := fun x => by ring
      rw [Finset.sum_congr rfl fun x _ => hexp x, Finset.sum_add_distrib,
        Finset.sum_add_distrib, Finset.sum_const, ← Finset.mul_sum,
        eps_sum b₁ b₂ hb, eps_sq_sum b₁ b₂ hb, Finset.card_univ, nsmul_eq_mul,
        mul_zero, add_zero]
    calc (Fintype.card B : ℝ) * ∑ ω : Fin (N + 1) → B, Spair b₁ b₂ ω ^ 2
        = (Fintype.card B : ℝ)
            * ∑ ω : Fin N → B, ∑ x : B, (Spair b₁ b₂ ω + eps b₁ b₂ x) ^ 2 := by
          rw [sum_pool_succ b₁ b₂ N (fun z => z ^ 2)]
      _ = (Fintype.card B : ℝ)
            * ∑ ω : Fin N → B, ((Fintype.card B : ℝ) * Spair b₁ b₂ ω ^ 2 + 2) := by
          rw [Finset.sum_congr rfl fun ω _ => hinner ω]
      _ = (Fintype.card B : ℝ)
            * ((Fintype.card B : ℝ) * ∑ ω : Fin N → B, Spair b₁ b₂ ω ^ 2
              + 2 * (Fintype.card B : ℝ) ^ N) := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const,
            Finset.card_univ, nsmul_eq_mul, hcardFun]
          ring
      _ = (Fintype.card B : ℝ)
            * (2 * N * (Fintype.card B : ℝ) ^ N + 2 * (Fintype.card B : ℝ) ^ N) := by
          rw [ih]
      _ = 2 * (N + 1 : ℕ) * (Fintype.card B : ℝ) ^ (N + 1) := by
          push_cast
          ring

theorem pool_S3 (b₁ b₂ : B) (hb : b₁ ≠ b₂) (N : ℕ) :
    ∑ ω : Fin N → B, Spair b₁ b₂ ω ^ 3 = 0 := by
  induction N with
  | zero => simp [Spair]
  | succ N ih =>
    have hinner : ∀ ω : Fin N → B,
        ∑ x : B, (Spair b₁ b₂ ω + eps b₁ b₂ x) ^ 3
          = (Fintype.card B : ℝ) * Spair b₁ b₂ ω ^ 3 + 6 * Spair b₁ b₂ ω := by
      intro ω
      have hexp : ∀ x : B, (Spair b₁ b₂ ω + eps b₁ b₂ x) ^ 3
          = Spair b₁ b₂ ω ^ 3 + 3 * Spair b₁ b₂ ω ^ 2 * eps b₁ b₂ x
            + 3 * Spair b₁ b₂ ω * eps b₁ b₂ x ^ 2 + eps b₁ b₂ x ^ 3 :=
        fun x => by ring
      rw [Finset.sum_congr rfl fun x _ => hexp x, Finset.sum_add_distrib,
        Finset.sum_add_distrib, Finset.sum_add_distrib, Finset.sum_const,
        ← Finset.mul_sum, ← Finset.mul_sum, eps_sum b₁ b₂ hb,
        eps_sq_sum b₁ b₂ hb, eps_cube_sum b₁ b₂ hb, Finset.card_univ,
        nsmul_eq_mul, mul_zero, add_zero, add_zero]
      ring
    calc ∑ ω : Fin (N + 1) → B, Spair b₁ b₂ ω ^ 3
        = ∑ ω : Fin N → B, ∑ x : B, (Spair b₁ b₂ ω + eps b₁ b₂ x) ^ 3 :=
          sum_pool_succ b₁ b₂ N (fun z => z ^ 3)
      _ = ∑ ω : Fin N → B,
            ((Fintype.card B : ℝ) * Spair b₁ b₂ ω ^ 3 + 6 * Spair b₁ b₂ ω) := by
          rw [Finset.sum_congr rfl fun ω _ => hinner ω]
      _ = (Fintype.card B : ℝ) * ∑ ω : Fin N → B, Spair b₁ b₂ ω ^ 3
            + 6 * ∑ ω : Fin N → B, Spair b₁ b₂ ω := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
      _ = 0 := by rw [ih, pool_S1 b₁ b₂ hb N, mul_zero, mul_zero, add_zero]

theorem pool_S4 (b₁ b₂ : B) (hb : b₁ ≠ b₂) (N : ℕ) :
    (Fintype.card B : ℝ) ^ 2 * ∑ ω : Fin N → B, Spair b₁ b₂ ω ^ 4
      = (12 * N * ((N : ℝ) - 1) + 2 * N * (Fintype.card B : ℝ))
          * (Fintype.card B : ℝ) ^ N := by
  induction N with
  | zero => simp [Spair]
  | succ N ih =>
    have hcardFun : (Fintype.card (Fin N → B) : ℝ) = (Fintype.card B : ℝ) ^ N := by
      rw [Fintype.card_fun, Fintype.card_fin]
      push_cast
      ring
    have hinner : ∀ ω : Fin N → B,
        ∑ x : B, (Spair b₁ b₂ ω + eps b₁ b₂ x) ^ 4
          = (Fintype.card B : ℝ) * Spair b₁ b₂ ω ^ 4
            + 12 * Spair b₁ b₂ ω ^ 2 + 2 := by
      intro ω
      have hexp : ∀ x : B, (Spair b₁ b₂ ω + eps b₁ b₂ x) ^ 4
          = Spair b₁ b₂ ω ^ 4 + 4 * Spair b₁ b₂ ω ^ 3 * eps b₁ b₂ x
            + 6 * Spair b₁ b₂ ω ^ 2 * eps b₁ b₂ x ^ 2
            + 4 * Spair b₁ b₂ ω * eps b₁ b₂ x ^ 3 + eps b₁ b₂ x ^ 4 :=
        fun x => by ring
      rw [Finset.sum_congr rfl fun x _ => hexp x, Finset.sum_add_distrib,
        Finset.sum_add_distrib, Finset.sum_add_distrib, Finset.sum_add_distrib,
        Finset.sum_const, ← Finset.mul_sum, ← Finset.mul_sum, ← Finset.mul_sum,
        eps_sum b₁ b₂ hb, eps_sq_sum b₁ b₂ hb, eps_cube_sum b₁ b₂ hb,
        eps_quart_sum b₁ b₂ hb, Finset.card_univ, nsmul_eq_mul,
        mul_zero, add_zero, mul_zero, add_zero]
      ring
    calc (Fintype.card B : ℝ) ^ 2 * ∑ ω : Fin (N + 1) → B, Spair b₁ b₂ ω ^ 4
        = (Fintype.card B : ℝ) ^ 2
            * ∑ ω : Fin N → B, ∑ x : B, (Spair b₁ b₂ ω + eps b₁ b₂ x) ^ 4 := by
          rw [sum_pool_succ b₁ b₂ N (fun z => z ^ 4)]
      _ = (Fintype.card B : ℝ) ^ 2
            * ∑ ω : Fin N → B, ((Fintype.card B : ℝ) * Spair b₁ b₂ ω ^ 4
              + 12 * Spair b₁ b₂ ω ^ 2 + 2) := by
          rw [Finset.sum_congr rfl fun ω _ => hinner ω]
      _ = (Fintype.card B : ℝ)
            * ((Fintype.card B : ℝ) ^ 2 * ∑ ω : Fin N → B, Spair b₁ b₂ ω ^ 4)
          + 12 * (Fintype.card B : ℝ)
            * ((Fintype.card B : ℝ) * ∑ ω : Fin N → B, Spair b₁ b₂ ω ^ 2)
          + 2 * (Fintype.card B : ℝ) ^ 2 * (Fintype.card B : ℝ) ^ N := by
          rw [Finset.sum_add_distrib, Finset.sum_add_distrib, Finset.sum_const,
            ← Finset.mul_sum, ← Finset.mul_sum, Finset.card_univ, nsmul_eq_mul,
            hcardFun]
          ring
      _ = (Fintype.card B : ℝ)
            * ((12 * N * ((N : ℝ) - 1) + 2 * N * (Fintype.card B : ℝ))
                * (Fintype.card B : ℝ) ^ N)
          + 12 * (Fintype.card B : ℝ) * (2 * N * (Fintype.card B : ℝ) ^ N)
          + 2 * (Fintype.card B : ℝ) ^ 2 * (Fintype.card B : ℝ) ^ N := by
          rw [ih, pool_S2 b₁ b₂ hb N]
      _ = (12 * (N + 1 : ℕ) * (((N + 1 : ℕ) : ℝ) - 1)
            + 2 * (N + 1 : ℕ) * (Fintype.card B : ℝ))
              * (Fintype.card B : ℝ) ^ (N + 1) := by
          push_cast
          ring

/-! ### The fully verified per-pair lower bound -/

/-- **Theorem B.6(b), per-pair bound, fully verified on the pool space.**
For the uniform measure on `Fin N → B` with `card B = m`, `2 ≤ m`,
`4m ≤ N`, and distinct cells `b₁ ≠ b₂`:
`E|n_{b₁} - n_{b₂}| ≥ (1/4)√(N/m)`. -/
theorem pair_diff_lower_uniform (b₁ b₂ : B) (hb : b₁ ≠ b₂) (N m : ℕ)
    (hcard : Fintype.card B = m) (hm : 2 ≤ m) (hNm : 4 * m ≤ N) :
    (1 / 4 : ℝ) * Real.sqrt ((N : ℝ) / m)
      ≤ (∑ ω : Fin N → B, |Spair b₁ b₂ ω|) / ((m : ℝ) ^ N) := by
  have hm0 : (0 : ℝ) < m := by
    have : (0 : ℕ) < m := by omega
    exact_mod_cast this
  have hN0 : (0 : ℝ) < N := by
    have : (0 : ℕ) < N := by omega
    exact_mod_cast this
  have hNm' : (4 : ℝ) * m ≤ N := by exact_mod_cast hNm
  haveI : Nonempty B := Fintype.card_pos_iff.mp (by omega)
  set v : ℝ := 2 * N / m with hvdef
  have hv8 : 8 ≤ v := by
    rw [hvdef, le_div_iff₀ hm0]
    linarith
  have hv0 : 0 < v := by linarith
  have hcardι : (Fintype.card (Fin N → B) : ℝ) = (m : ℝ) ^ N := by
    rw [Fintype.card_fun, Fintype.card_fin, hcard]
    push_cast
    ring
  -- second moment: E S² = v
  have h2 : ∑ ω : Fin N → B, Spair b₁ b₂ ω ^ 2
      = v * (Fintype.card (Fin N → B) : ℝ) := by
    have h := pool_S2 b₁ b₂ hb N
    rw [hcard] at h
    rw [hcardι, hvdef]
    have hmne : (m : ℝ) ≠ 0 := ne_of_gt hm0
    field_simp
    linarith [h]
  -- fourth moment: E S⁴ ≤ (25/8)v²
  have h4 : ∑ ω : Fin N → B, Spair b₁ b₂ ω ^ 4
      ≤ 25 / 8 * v ^ 2 * (Fintype.card (Fin N → B) : ℝ) := by
    have h := pool_S4 b₁ b₂ hb N
    rw [hcard] at h
    rw [hcardι, hvdef]
    -- m² E S⁴ = (12N(N-1) + 2Nm) m^N  and  (25/8)v²m² = (25/2)N²
    have hmne : (m : ℝ) ≠ 0 := ne_of_gt hm0
    have hgoal : (m : ℝ) ^ 2 * ∑ ω : Fin N → B, Spair b₁ b₂ ω ^ 4
        ≤ (m : ℝ) ^ 2 * (25 / 8 * (2 * N / m) ^ 2 * (m : ℝ) ^ N) := by
      rw [h]
      have hrhs : (m : ℝ) ^ 2 * (25 / 8 * (2 * N / m) ^ 2 * (m : ℝ) ^ N)
          = 25 / 2 * N ^ 2 * (m : ℝ) ^ N := by
        field_simp
        ring
      rw [hrhs]
      have hfac : (12 * N * ((N : ℝ) - 1) + 2 * N * m) ≤ 25 / 2 * (N : ℝ) ^ 2 := by
        nlinarith [hN0, hm0]
      have hpow : (0 : ℝ) ≤ (m : ℝ) ^ N := by positivity
      nlinarith [mul_le_mul_of_nonneg_right hfac hpow]
    have hsq : (0 : ℝ) < (m : ℝ) ^ 2 := by positivity
    exact le_of_mul_le_mul_left hgoal hsq
  -- the generic chain
  have hmain := mad_lower (Spair b₁ b₂ (N := N)) v hv8 h2 h4
  rw [hcardι] at hmain
  -- √v/3 = √(2N/m)/3 ≥ (1/4)√(N/m)
  refine le_trans ?_ hmain
  have hsplit : Real.sqrt v = Real.sqrt 2 * Real.sqrt ((N : ℝ) / m) := by
    rw [hvdef, show (2 : ℝ) * N / m = 2 * ((N : ℝ) / m) by ring]
    exact Real.sqrt_mul (by norm_num) _
  rw [hsplit]
  have hs2 : (1 : ℝ) ≤ Real.sqrt 2 := by
    nlinarith [Real.sq_sqrt (show (0 : ℝ) ≤ 2 by norm_num),
      Real.sqrt_nonneg (2 : ℝ)]
  have hsnn : 0 ≤ Real.sqrt ((N : ℝ) / m) := Real.sqrt_nonneg _
  nlinarith [hs2, hsnn]

/-- **Theorem B.6(b), fully verified end-to-end**: for the uniform pool measure
on `Fin N → (Fin T × Bool)` (`m = 2T` cells, `2 ≤ m`, `4m ≤ N`),
`E Z = E[(σ/N) ∑_t |n_{(t,1)} - n_{(t,0)}|] ≥ (σ/8)√(m/N)` — no probabilistic
hypotheses remain. -/
theorem thm_lower_b_uniform (T N m : ℕ) (hmT : m = 2 * T) (hT : 1 ≤ T)
    (hm : 2 ≤ m) (hNm : 4 * m ≤ N) (σ : ℝ) (hσ : 0 ≤ σ) :
    σ / 8 * Real.sqrt ((m : ℝ) / N)
      ≤ (∑ ω : Fin N → Fin T × Bool,
          σ / N * ∑ t : Fin T, |cnt ω (t, true) - cnt ω (t, false)|)
        / ((m : ℝ) ^ N) := by
  have hN : 1 ≤ N := by omega
  have hcard : Fintype.card (Fin T × Bool) = m := by
    simp [Fintype.card_prod, hmT, mul_comm]
  -- E Z = (σ/N) ∑_t E|d_t|
  have hswap : (∑ ω : Fin N → Fin T × Bool,
        σ / N * ∑ t : Fin T, |cnt ω (t, true) - cnt ω (t, false)|)
        / ((m : ℝ) ^ N)
      = σ / N * ∑ t : Fin T,
          (∑ ω : Fin N → Fin T × Bool, |Spair (t, true) (t, false) ω|)
            / ((m : ℝ) ^ N) := by
    rw [← Finset.mul_sum, Finset.sum_comm]
    rw [mul_div_assoc, Finset.sum_div]
    congr 1
    refine Finset.sum_congr rfl fun t _ => ?_
    congr 1
    refine Finset.sum_congr rfl fun ω _ => ?_
    rw [Spair_eq_cnt]
  rw [hswap]
  apply attack_inflation_mean T N m hmT hT hN σ hσ
  intro t
  exact pair_diff_lower_uniform (t, true) (t, false) (by simp) N m hcard hm hNm

end AppB
