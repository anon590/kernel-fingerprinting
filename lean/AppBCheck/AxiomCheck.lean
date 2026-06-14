import AppB

/- Axiom audit: every key theorem should use only the three standard Lean
axioms (`propext`, `Classical.choice`, `Quot.sound`) — no `sorryAx`. -/

#print axioms AppB.node_count_le
#print axioms AppB.node_count_le_total
#print axioms AppB.node_count_le_two_pow_total
#print axioms AppB.tree_count_le_total
#print axioms AppB.union_bound_uniform
#print axioms AppB.prob_le_of_condExp_le
#print axioms AppB.thm_upper
#print axioms AppB.thm_upper_fixed_level
#print axioms AppB.hoeffding_level
#print axioms AppB.remark_bit_budgets_vacuous
#print axioms AppB.pascal_sum
#print axioms AppB.sum_choose_sq
#print axioms AppB.sum_choose_quart
#print axioms AppB.cs_step
#print axioms AppB.jensen_step
#print axioms AppB.lemma_mad
#print axioms AppB.binomW_sum
#print axioms AppB.binomial_lower_tail
#print axioms AppB.halfAbsMean_ge
#print axioms AppB.pair_diff_lower
#print axioms AppB.attack_population_mean
#print axioms AppB.empirical_mean_fiber
#print axioms AppB.attack_inflation
#print axioms AppB.attack_inflation_mean
#print axioms AppB.thm_lower_b
#print axioms AppB.cnt_one_instance_diff
#print axioms AppB.Z_bounded_diff
#print axioms AppB.mcdiarmid_exponent
#print axioms AppB.deviation_conclusion
#print axioms AppB.thm_lower_disappointment
#print axioms AppB.mad_lower
#print axioms AppB.pool_S2
#print axioms AppB.pool_S4
#print axioms AppB.pair_diff_lower_uniform
#print axioms AppB.thm_lower_b_uniform

/- §B.7 (`app:enum`): weighted pool space + enumerability dictionary -/
#print axioms AppB.pexp_one
#print axioms AppB.cmom1
#print axioms AppB.cmom2
#print axioms AppB.pexp_abs_le_sqrt
#print axioms AppB.pexp_pushforward
#print axioms AppB.enumerableId_richness
#print axioms AppB.richness_attack_fires
#print axioms AppB.diffuse_not_enumerableId
#print axioms AppB.cntC_total
#print axioms AppB.t1_diffuse_upper
#print axioms AppB.diffuse_starves
