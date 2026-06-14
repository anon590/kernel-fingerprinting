/-
# Weighted pool space: product laws, centered cell counts, Cauchy–Schwarz

`AppB.PairMoments` verifies pool-space expectations for the *uniform* law on
`Fin N → B`.  The enumerability dictionary (L1, `AppB.Enumerability`) needs
the same machinery for an arbitrary probe law `w : Ξ → ℝ` (`w ≥ 0`,
`∑ w = 1`): the pool measure is the product weight
`wprod w ω = ∏ i, w (ω i)`, and expectations are
`pexp w N f = ∑_ω wprod w ω · f ω`.

Proved here, all by the same `Fin.cons`-peeling induction as
`AppB.PairMoments` (fully formal, no probabilistic primitives):

* `pexp_const`, `pexp_add`, `pexp_smul`, `pexp_mono` — expectation algebra;
* `pexp_one` — total mass `(∑ w)^N = 1`;
* `cmom1`    — `E[n_b - N w_b] = 0`;
* `cmom2`    — `E[(n_b - N w_b)²] = N · w_b (1 - w_b)`  (exact);
* `pexp_abs_le_sqrt` — `E|g| ≤ √(E g²)` (weighted Cauchy–Schwarz).
-/
import AppB.PairMoments

open Finset

set_option linter.unusedSectionVars false

namespace AppB

variable {Ξ : Type*} [Fintype Ξ] [DecidableEq Ξ]

/-- Product weight of a pool under the probe law `w`. -/
noncomputable def wprod (w : Ξ → ℝ) {N : ℕ} (ω : Fin N → Ξ) : ℝ :=
  ∏ i, w (ω i)

/-- Pool expectation under the product law. -/
noncomputable def pexp (w : Ξ → ℝ) (N : ℕ) (f : (Fin N → Ξ) → ℝ) : ℝ :=
  ∑ ω : Fin N → Ξ, wprod w ω * f ω

theorem wprod_nonneg (w : Ξ → ℝ) (hw : ∀ x, 0 ≤ w x) {N : ℕ}
    (ω : Fin N → Ξ) : 0 ≤ wprod w ω :=
  Finset.prod_nonneg fun i _ => hw (ω i)

theorem wprod_cons (w : Ξ → ℝ) {N : ℕ} (x : Ξ) (ω : Fin N → Ξ) :
    wprod w (Fin.cons x ω) = w x * wprod w ω := by
  unfold wprod
  rw [Fin.prod_univ_succ]
  simp [Fin.cons_zero, Fin.cons_succ]

/-- Peeling one instance off the pool expectation. -/
theorem pexp_succ (w : Ξ → ℝ) (N : ℕ) (f : (Fin (N + 1) → Ξ) → ℝ) :
    pexp w (N + 1) f
      = ∑ x : Ξ, w x * pexp w N (fun ω => f (Fin.cons x ω)) := by
  unfold pexp
  have h := Equiv.sum_comp (Fin.consEquiv (fun _ : Fin (N + 1) => Ξ))
      (fun ω => wprod w ω * f ω)
  rw [← h, Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun ω _ => ?_
  have hc : (Fin.consEquiv (fun _ : Fin (N + 1) => Ξ)) (x, ω) = Fin.cons x ω :=
    rfl
  rw [hc, wprod_cons]
  ring

theorem pexp_one (w : Ξ → ℝ) (hw1 : ∑ x : Ξ, w x = 1) (N : ℕ) :
    pexp w N (fun _ => 1) = 1 := by
  induction N with
  | zero =>
    unfold pexp wprod
    simp
  | succ N ih =>
    rw [pexp_succ]
    calc ∑ x : Ξ, w x * pexp w N (fun _ => 1)
        = ∑ x : Ξ, w x := by
          refine Finset.sum_congr rfl fun x _ => ?_
          rw [ih, mul_one]
      _ = 1 := hw1

theorem pexp_add (w : Ξ → ℝ) (N : ℕ) (f g : (Fin N → Ξ) → ℝ) :
    pexp w N (fun ω => f ω + g ω) = pexp w N f + pexp w N g := by
  unfold pexp
  rw [← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun ω _ => by ring

theorem pexp_smul (w : Ξ → ℝ) (N : ℕ) (c : ℝ) (f : (Fin N → Ξ) → ℝ) :
    pexp w N (fun ω => c * f ω) = c * pexp w N f := by
  unfold pexp
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun ω _ => by ring

theorem pexp_mono (w : Ξ → ℝ) (hw : ∀ x, 0 ≤ w x) (N : ℕ)
    (f g : (Fin N → Ξ) → ℝ) (hfg : ∀ ω, f ω ≤ g ω) :
    pexp w N f ≤ pexp w N g := by
  unfold pexp
  exact Finset.sum_le_sum fun ω _ =>
    mul_le_mul_of_nonneg_left (hfg ω) (wprod_nonneg w hw ω)

theorem pexp_nonneg (w : Ξ → ℝ) (hw : ∀ x, 0 ≤ w x) (N : ℕ)
    (f : (Fin N → Ξ) → ℝ) (hf : ∀ ω, 0 ≤ f ω) : 0 ≤ pexp w N f := by
  unfold pexp
  exact Finset.sum_nonneg fun ω _ =>
    mul_nonneg (wprod_nonneg w hw ω) (hf ω)

/-- Expectation of a finite sum of statistics. -/
theorem pexp_sum (w : Ξ → ℝ) (N : ℕ) {α : Type*} (s : Finset α)
    (f : α → (Fin N → Ξ) → ℝ) :
    pexp w N (fun ω => ∑ a ∈ s, f a ω) = ∑ a ∈ s, pexp w N (f a) := by
  unfold pexp
  calc ∑ ω : Fin N → Ξ, wprod w ω * ∑ a ∈ s, f a ω
      = ∑ ω : Fin N → Ξ, ∑ a ∈ s, wprod w ω * f a ω :=
        Finset.sum_congr rfl fun ω _ => Finset.mul_sum _ _ _
    _ = ∑ a ∈ s, ∑ ω : Fin N → Ξ, wprod w ω * f a ω := Finset.sum_comm

/-! ## Centered cell-count moments -/

/-- The centered count of cell `b`: `n_b - N w_b` as a sum of centered
indicators. -/
noncomputable def cntC (w : Ξ → ℝ) (b : Ξ) {N : ℕ} (ω : Fin N → Ξ) : ℝ :=
  cnt ω b - N * w b

theorem cntC_cons (w : Ξ → ℝ) (b : Ξ) {N : ℕ} (x : Ξ) (ω : Fin N → Ξ) :
    cntC w b (Fin.cons x ω)
      = ((if x = b then 1 else 0) - w b) + cntC w b ω := by
  unfold cntC cnt
  rw [Fin.sum_univ_succ]
  simp only [Fin.cons_zero, Fin.cons_succ]
  push_cast
  ring

/-- Per-instance first moment: `∑_x w x ((1{x=b}) - w b) = w b (1 - ∑ w)= 0`. -/
theorem eps_center_mean (w : Ξ → ℝ) (hw1 : ∑ x : Ξ, w x = 1) (b : Ξ) :
    ∑ x : Ξ, w x * ((if x = b then 1 else 0) - w b) = 0 := by
  have hexp : ∀ x : Ξ, w x * ((if x = b then 1 else 0) - w b)
      = (if x = b then w x else 0) - w x * w b := by
    intro x
    split_ifs <;> ring
  rw [Finset.sum_congr rfl fun x _ => hexp x, Finset.sum_sub_distrib,
    Finset.sum_ite_eq' Finset.univ b w, ← Finset.sum_mul, hw1]
  simp

/-- Per-instance second moment:
`∑_x w x ((1{x=b}) - w b)² = w b (1 - w b) (2∑w - ... ) = w b (1 - w b)`
when `∑ w = 1`. -/
theorem eps_center_sq (w : Ξ → ℝ) (hw1 : ∑ x : Ξ, w x = 1) (b : Ξ) :
    ∑ x : Ξ, w x * ((if x = b then 1 else 0) - w b) ^ 2
      = w b * (1 - w b) := by
  have hexp : ∀ x : Ξ, w x * ((if x = b then 1 else 0) - w b) ^ 2
      = (if x = b then w x else 0) * (1 - 2 * w b)
        + w x * (w b) ^ 2 := by
    intro x
    split_ifs <;> ring
  rw [Finset.sum_congr rfl fun x _ => hexp x, Finset.sum_add_distrib,
    ← Finset.sum_mul, ← Finset.sum_mul,
    Finset.sum_ite_eq' Finset.univ b w, hw1]
  simp
  ring

/-- `E[n_b - N w_b] = 0`. -/
theorem cmom1 (w : Ξ → ℝ) (hw1 : ∑ x : Ξ, w x = 1) (b : Ξ) (N : ℕ) :
    pexp w N (fun ω => cntC w b ω) = 0 := by
  induction N with
  | zero =>
    unfold pexp wprod cntC cnt
    simp
  | succ N ih =>
    rw [pexp_succ]
    have hstep : ∀ x : Ξ, pexp w N (fun ω => cntC w b (Fin.cons x ω))
        = ((if x = b then 1 else 0) - w b) * pexp w N (fun _ => 1)
          + pexp w N (fun ω => cntC w b ω) := by
      intro x
      have hfun : (fun ω : Fin N → Ξ => cntC w b (Fin.cons x ω))
          = fun ω => ((if x = b then 1 else 0) - w b) * 1 + cntC w b ω :=
        funext fun ω => by rw [cntC_cons]; ring
      rw [hfun, pexp_add, pexp_smul]
    calc ∑ x : Ξ, w x * pexp w N (fun ω => cntC w b (Fin.cons x ω))
        = ∑ x : Ξ, w x * (((if x = b then 1 else 0) - w b) * 1 + 0) := by
          refine Finset.sum_congr rfl fun x _ => ?_
          rw [hstep x, pexp_one w hw1, ih]
      _ = ∑ x : Ξ, w x * ((if x = b then 1 else 0) - w b) := by
          refine Finset.sum_congr rfl fun x _ => ?_
          ring
      _ = 0 := eps_center_mean w hw1 b

/-- **Exact centered second moment**: `E[(n_b - N w_b)²] = N w_b (1 - w_b)`. -/
theorem cmom2 (w : Ξ → ℝ) (hw1 : ∑ x : Ξ, w x = 1) (b : Ξ) (N : ℕ) :
    pexp w N (fun ω => cntC w b ω ^ 2) = N * (w b * (1 - w b)) := by
  induction N with
  | zero =>
    unfold pexp wprod cntC cnt
    simp
  | succ N ih =>
    rw [pexp_succ]
    have hstep : ∀ x : Ξ, pexp w N (fun ω => cntC w b (Fin.cons x ω) ^ 2)
        = ((if x = b then 1 else 0) - w b) ^ 2 * pexp w N (fun _ => 1)
          + 2 * ((if x = b then 1 else 0) - w b)
            * pexp w N (fun ω => cntC w b ω)
          + pexp w N (fun ω => cntC w b ω ^ 2) := by
      intro x
      have hfun : (fun ω : Fin N → Ξ => cntC w b (Fin.cons x ω) ^ 2)
          = fun ω => (((if x = b then 1 else 0) - w b) ^ 2 * 1
              + 2 * ((if x = b then 1 else 0) - w b) * cntC w b ω)
              + cntC w b ω ^ 2 :=
        funext fun ω => by rw [cntC_cons]; ring
      rw [hfun, pexp_add, pexp_add, pexp_smul, pexp_smul]
    calc ∑ x : Ξ, w x * pexp w N (fun ω => cntC w b (Fin.cons x ω) ^ 2)
        = ∑ x : Ξ, w x * (((if x = b then 1 else 0) - w b) ^ 2 * 1
            + 2 * ((if x = b then 1 else 0) - w b) * 0
            + N * (w b * (1 - w b))) := by
          refine Finset.sum_congr rfl fun x _ => ?_
          rw [hstep x, pexp_one w hw1, cmom1 w hw1, ih]
      _ = (∑ x : Ξ, w x * ((if x = b then 1 else 0) - w b) ^ 2)
            + (∑ x : Ξ, w x) * (N * (w b * (1 - w b))) := by
          rw [Finset.sum_mul, ← Finset.sum_add_distrib]
          refine Finset.sum_congr rfl fun x _ => ?_
          ring
      _ = (N + 1 : ℕ) * (w b * (1 - w b)) := by
          rw [eps_center_sq w hw1, hw1]
          push_cast
          ring

/-! ## Weighted Cauchy–Schwarz -/

/-- `E|g| ≤ √(E g²)` for the (probability) pool weights. -/
theorem pexp_abs_le_sqrt (w : Ξ → ℝ) (hw : ∀ x, 0 ≤ w x)
    (hw1 : ∑ x : Ξ, w x = 1) (N : ℕ) (g : (Fin N → Ξ) → ℝ) :
    pexp w N (fun ω => |g ω|) ≤ Real.sqrt (pexp w N (fun ω => g ω ^ 2)) := by
  have hsq : (pexp w N (fun ω => |g ω|)) ^ 2
      ≤ pexp w N (fun ω => g ω ^ 2) := by
    have h := Finset.sum_mul_sq_le_sq_mul_sq Finset.univ
        (fun ω : Fin N → Ξ => Real.sqrt (wprod w ω))
        (fun ω : Fin N → Ξ => Real.sqrt (wprod w ω) * |g ω|)
    have hFG : ∀ ω ∈ Finset.univ (α := Fin N → Ξ),
        Real.sqrt (wprod w ω) * (Real.sqrt (wprod w ω) * |g ω|)
          = wprod w ω * |g ω| := by
      intro ω _
      rw [← mul_assoc, Real.mul_self_sqrt (wprod_nonneg w hw ω)]
    have hF2 : ∀ ω ∈ Finset.univ (α := Fin N → Ξ),
        Real.sqrt (wprod w ω) ^ 2 = wprod w ω :=
      fun ω _ => Real.sq_sqrt (wprod_nonneg w hw ω)
    have hG2 : ∀ ω ∈ Finset.univ (α := Fin N → Ξ),
        (Real.sqrt (wprod w ω) * |g ω|) ^ 2 = wprod w ω * g ω ^ 2 := by
      intro ω _
      rw [mul_pow, Real.sq_sqrt (wprod_nonneg w hw ω), sq_abs]
    rw [Finset.sum_congr rfl hFG, Finset.sum_congr rfl hF2,
      Finset.sum_congr rfl hG2] at h
    have hwsum : ∑ ω : Fin N → Ξ, wprod w ω = 1 := by
      have := pexp_one w hw1 N
      unfold pexp at this
      simpa using this
    rw [hwsum, one_mul] at h
    exact h
  have habs : 0 ≤ pexp w N (fun ω => |g ω|) :=
    pexp_nonneg w hw N _ fun ω => abs_nonneg _
  calc pexp w N (fun ω => |g ω|)
      = Real.sqrt ((pexp w N (fun ω => |g ω|)) ^ 2) :=
        (Real.sqrt_sq habs).symm
    _ ≤ Real.sqrt (pexp w N (fun ω => g ω ^ 2)) := Real.sqrt_le_sqrt hsq

end AppB
