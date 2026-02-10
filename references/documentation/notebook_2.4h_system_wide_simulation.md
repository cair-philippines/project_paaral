# Notebooks 2.4h & 2.4i — System-Wide ESC Simulation

> Created: 2026-02-09 | Updated: 2026-02-10
> **2.4h** (exploratory): `notebooks/2.4h-placeholder-something.ipynb`
> **2.4i** (compiled): `notebooks/2.4i.-system-wide-esc-simulation.ipynb`
> Model: `0209_clusteredSE` (Negative Binomial Regression with clustered SE)

**2.4i** is the refactored, compiled version of **2.4h**. Both share this documentation.
Use 2.4i for reproducible runs; use 2.4h for exploratory work.

---

## 1. How 2.4h/2.4i Differ from Previous Notebooks

### 1.1. Comparison Table

| Dimension | 2.4f (Constrained Path) | 2.4g (Slot Reallocation) | 2.4h/2.4i (System-Wide) |
|---|---|---|---|
| **Scope** | Congestion-scoped only | Congestion-scoped only | All BoSY system learners |
| **Candidate pool** | 400,936 (from origins feeding congested public JHS) | 400,936 (same as 2.4f) | 562,958 (all observed G7 flows within Mega Manila) |
| **Slot pool column** | `available_slots` (contracted minus billed) | `available_slots` (same as 2.4f) | `slots_total` (full contracted capacity) |
| **Slot pool size** | 26,491 (only unfilled slots) | 26,608 (before over-enrollment clamping) | 106,654 (total system capacity) |
| **NBR model file** | `0207_model4` | `0207_model4` | `0209_clusteredSE` |
| **ESC schools** | 1,088 (with available > 0) | ~1,088 | ~1,370 (all with predicted demand) |
| **Primary question** | How many students can the remaining slots absorb? | Can moving slots from surplus to deficit improve outcomes? | What is the full decongestion role of the ESC system? |
| **Decongestion metric** | System-level diversions + per-school CIs | Diversions under reallocation strategies | Marginal + total predicted, per ESC school |

### 1.2. Key Conceptual Differences

**2.4f asks:** "Given the *remaining* ESC capacity (unfilled slots) and only students from congested-feeding origins, how many diversions occur?"
- This is a *marginal capacity* question scoped to congestion
- Result: ~25,936 diversions — nearly all 26,491 available slots consumed (97.9%)

**2.4g asks:** "If we *move* slots from surplus schools to deficit schools (budget-neutral), does decongestion improve?"
- This is a *reallocation efficiency* question
- Result: Marginal gains (+75–90 diversions) because demand vastly exceeds supply everywhere

**2.4h/2.4i ask:** "Across the *entire* ESC system with *all* contracted slots, what is the predicted enrollment and how much of it contributes to decongestion — both in total and beyond what's already enrolled?"
- This is a *full system assessment* question
- Uses `slots_total` (not just unfilled) because it re-simulates the full enrollment process
- Uses all BoSY learners (not just congestion-scoped) because it models the complete system

### 1.3. Why `slots_total` Instead of `available_slots`

In 2.4f/2.4g, the question was "what can the *remaining* capacity absorb?" — so `available_slots` (= `slots_total` - `slots_billed`) was correct.

In 2.4h/2.4i, the question is "what does the model *predict* the full system would do?" — so `slots_total` is correct. We then compare predicted enrollment against `slots_billed` to derive marginals. Using `available_slots` here would double-count the subtraction of billed students.

### 1.4. Why the Broader Candidate Pool

2.4f filtered candidates to only those from origins feeding congested public JHS (400,936). This made sense for measuring decongestion potential of remaining capacity.

2.4h/2.4i use all 562,958 BoSY system learners because they model the full enrollment competition. Students from non-congested origins still compete for ESC slots and affect how many slots are available for congestion-relevant students. Excluding them would overestimate per-school predicted enrollment.

---

## 2. Standardized Labels for Observed ESC Quantities

Multiple measures of "observed ESC enrollment" appear throughout 2.4i. They are **not** interchangeable:

| Standardized Label | Value | What It Counts | Data Source |
|---|---|---|---|
| **All students at ESC schools** | ~88,663 | Everyone attending ESC schools (beneficiaries + non-beneficiaries/self-paying) | `df_flow_tagged` where `is_esc_destination`, column `total_students` |
| **ESC beneficiaries (flow)** | ~76,438 | Only students tagged as ESC beneficiaries in student flow data, aggregated by destination | `system_slots['count_esc_beneficiary']` (from `student_flow` joined to `esc_slots`) |
| **ESC beneficiaries (flow, CBP origins)** | ~73,624 | Same as above but aggregated by origin, filtered to origins in the CBP | `flow_by_origin['count_esc_beneficiary']` |
| **Slots billed (admin)** | ~82,071 | Administrative billing records from ESC contracting data | `system_slots['slots_billed']` (from `processed_esc_slots.parquet`) |

**Why they differ:**
- **88,663 vs 76,438**: `total_students` includes non-beneficiaries (self-paying students at ESC schools); `count_esc_beneficiary` does not
- **76,438 vs 73,624**: destination-side aggregation captures all beneficiaries; origin-side filtered to CBP origins drops a small number
- **82,071 vs 76,438**: different data sources — admin billing records vs observed student flow data

**Usage in results table:** The cross-scenario comparison table uses `obs_by_esc['obs_esc_beneficiaries']` (~74,232) and `obs_by_esc['obs_benef_from_congested']` (~55,188) for the observed row, labeled `observed (benef)`. These come from 1,369 ESC schools with observed student flow (4 fewer than the 1,373 in `system_slots`, which has ~76,438 beneficiaries). The minor difference is because a few system schools have no student flow records.

---

## 3. Observed Baseline (2.4i Section 2)

Before running simulations, 2.4i examines where students *actually* go. This provides ground truth for validating model predictions later (Sections 6.4–6.5).

### 3.1. Data: `df_flow_tagged`

The full observed student flow table (`grade_7_student_flow_table_sy2324.parquet`) is tagged with two boolean columns:

| Column | Source | Description |
|---|---|---|
| `is_congested_destination` | `flow_to_congested` destination IDs | Is the destination a congested public JHS? |
| `is_esc_destination` | CBP destination IDs (`cbp_dest_ids`) | Is the destination an ESC school in our system? |
| `total_students` | `count_non_beneficiary + count_esc_beneficiary` | All students in this origin-destination pair |

### 3.2. Origin-Level Analysis (`obs_by_origin`)

For each origin school, computes what fraction of its students flow to congested public JHS vs ESC schools.

| Column | Description |
|---|---|
| `total_students` | All G7 students from this origin |
| `students_to_congested` | Students going to congested public JHS |
| `students_to_esc` | Students going to ESC schools |
| `pct_to_congested` | Percent to congested |
| `pct_to_esc` | Percent to ESC |
| `is_congested_feeding` | Whether this origin is in `congested_feeding_origins` |

**Purpose:** Validates the "congested-feeding" origin tag — origins tagged as congested-feeding should have meaningfully higher `pct_to_congested` than non-tagged origins.

### 3.3. ESC-School-Level Analysis (`obs_by_esc`)

For each ESC school, computes observed enrollment and the fraction of enrollees from congested-feeding origins. This is the empirical analogue of the model's `congested_frac`.

| Column | Description |
|---|---|
| `obs_total_students` | All students at this ESC school (beneficiaries + non-beneficiaries) |
| `obs_esc_beneficiaries` | ESC beneficiaries (flow) at this school |
| `obs_from_congested_origins` | All students from congested-feeding origins |
| `obs_benef_from_congested` | ESC beneficiaries from congested-feeding origins |
| `obs_congested_frac` | `obs_from_congested_origins / obs_total_students` |

**Purpose:** Provides observed ground truth for comparing against the model's `congested_frac` in Section 6.5. The beneficiary columns (`obs_esc_beneficiaries`, `obs_benef_from_congested`) are used for the observed row in the cross-scenario comparison table.

---

## 4. Decongestion Analysis

### 4.1. Congested-Feeding Fraction (`congested_frac`)

For each ESC school, we ask: "What share of its NBR-predicted demand comes from origins that also feed congested public JHS?"

**Computation:**
1. Load `flow_to_congested.parquet` — observed student flows from elementary schools to congested public JHS
2. Extract the set of `congested_feeding_origins` (8,015 origin schools)
3. For each ESC destination, compute:
   - `mu_all` = sum of `mu_current_subsidy` across all origin-destination pairs
   - `mu_congested` = sum of `mu_current_subsidy` only from pairs where origin is in `congested_feeding_origins`
   - `congested_frac` = `mu_congested / mu_all`

**Result:** ~1,370/1,373 ESC schools have congested-feeding demand. Mean fraction = 94.4%.

**Note on approximation:** `congested_frac` is computed from raw (unconstrained) mu values, not from simulation results. This is a proxy — the actual fraction of accepted students from congested-feeding origins may differ due to coupled depletion and shuffle order. 2.4i validates this at the system level by comparing the raw mu fraction (0.988) against the actual simulation-based fraction (0.957, from `per_origin` data). The 3.1% relative difference is within acceptable range for system-level analysis but means per-school fractions are proxies, not exact values. See Section 7 for full validation results.

### 4.2. Two Analytical Approaches

#### Question A — Marginal Decongestion

> "Beyond what's already enrolled, how many *additional* students does the model predict each ESC school would absorb from congested-feeding origins?"

```
marginal = max(0, mean_predicted - slots_billed)
marginal_decongestion = marginal * congested_frac
```

- Answers: "What is the *untapped* decongestion potential?"
- Only counts students above current enrollment as new diversions
- 1,057 ESC schools have positive marginal (predicted > billed)
- 284 schools have negative marginal — two distinct causes:
  - **Over-enrolled** (185 schools, billed > slots_total): simulation caps at slots_total, mechanically producing predicted < billed
  - **Model under-prediction** (99 schools): NBR genuinely predicts less demand than observed
- System total: ~25,293 marginal diversions, of which ~98.1% from congested-feeding origins (~24,820)

#### Question B — Total Predicted Decongestion Role

> "Of all predicted ESC flow, how much comes from congested-feeding origins?"

```
predicted_congested = mean_predicted * congested_frac
```

- Answers: "What is the ESC system's *full structural role* in decongestion?"
- No subtraction of billed — this is the total predicted decongestion role
- System total: ~95,695 predicted congested-feeding flow (95.7% of total predicted)

#### When to Use Which

- **Question A** for policy: "Where should we focus expansion to get *additional* decongestion?"
- **Question B** for understanding: "How important is the ESC system *overall* for decongestion?"
- **Negative raw_marginal** for diagnostics: use `is_over_enrolled` flag (2.4i) to distinguish structural cap from model failure

### 4.3. Existing vs Hypothetical Path Split

Both approaches decompose into:
- **Existing paths**: origin-destination pairs observed in actual student flow data
- **Hypothetical paths**: origin-destination pairs predicted by NBR but not yet observed

For marginal decongestion, the split uses proportional attribution:
```
marginal_existing = marginal * (mean_existing / mean_predicted)
marginal_hypothetical = marginal * (mean_hypothetical / mean_predicted)
```

System-wide: hypothetical paths account for ~53.9% of marginal decongestion.

---

## 5. Subsidy Scenario Sweep

2.4h (Sections 3.3–3.5) and 2.4i (Section 5) run the full analysis under multiple subsidy scenarios. The cross-scenario comparison table includes an `observed (benef)` row showing actual ESC beneficiaries (flow) for direct comparison:

| Scenario | Net Cost Reduction | Mean Predicted Flow | Predicted Congested | Marginal Congested |
|---|---|---|---|---|
| `observed (benef)` | — | ~74,232 | ~55,188 | — |
| `-1k_net_cost` | -1,000 | ~99,992 | ~95,695 | ~24,820 |
| `-5k_net_cost` | -5,000 | ~100,442 | ~96,126 | ~24,988 |
| `-10k_net_cost` | -10,000 | ~100,944 | ~96,603 | ~25,183 |
| `-15k_net_cost` | -15,000 | ~101,428 | ~97,063 | ~25,274 |
| `-20k_net_cost` | -20,000 | ~101,818 | ~97,417 | ~25,381 |

**Note:** `current_subsidy` was removed from the scenario sweep after team discussion. The five scenarios above represent incremental net cost reductions from the current subsidy level. Percent changes from observed are computed in the notebook.

**Key finding:** Even reducing net cost by 20k only increases predicted flow by ~1.8% over the lowest-subsidy scenario. The system is **slot-constrained, not price-constrained**.

**Note:** The model predicts ~99,992 flow vs ~74,232 observed beneficiaries. The gap reflects that the simulation uses `slots_total` (full contracted capacity), not just currently billed slots — it answers "what *would* the system absorb?" not "what *has* the system absorbed?"

---

## 6. Per-School DataFrame (`df_school`) Column Reference

| Column | Type | Description |
|---|---|---|
| `destination_school_id` | str | ESC school identifier |
| `school_name` | str | School name (from `sch_info`) |
| `division` | str | DepEd division |
| `slots_total` | int | Total contracted ESC slots |
| `slots_billed` | int | Slots billed (admin) |
| `is_over_enrolled` | bool | `billed > slots_total` (2.4i only) |
| `mean_predicted` | float | MC mean of total accepted students across iterations |
| `mean_existing` | float | MC mean from existing (observed) paths |
| `mean_hypothetical` | float | MC mean from hypothetical (unobserved) paths |
| `pct_hypothetical` | float | `mean_hypothetical / mean_predicted * 100` |
| `raw_marginal` | float | `mean_predicted - slots_billed` (unclamped, can be negative) |
| `congested_frac` | float | Share of predicted demand from congested-feeding origins (0–1) |
| `predicted_congested` | float | **Question B**: `mean_predicted * congested_frac` |
| `marginal_decongestion` | float | **Question A**: `max(0, raw_marginal) * congested_frac` |
| `marginal_existing` | float | Question A, existing paths only |
| `marginal_hypothetical` | float | Question A, hypothetical paths only |

---

## 7. Observed vs Predicted Comparisons (2.4i Sections 6.4–6.5)

### 7.1. `congested_frac` Approximation Validation (Section 6.1)

Compares the raw-mu-based congested fraction (pre-computed proxy) against the actual simulation-based fraction (from `per_origin` tracking):

| Metric | Value |
|---|---|
| Raw mu (pre-computed approximation) | 0.9879 |
| Simulation (per_origin, MC mean) | 0.9569 |
| Relative difference | 3.14% |

The ~3% gap arises because coupled depletion and shuffle order alter which paths get accepted vs. the unconstrained mu ratios. This is acceptable for system-level analysis (the approximation preserves the right order of magnitude and directional finding that ~96% of flow comes from congested-feeding origins). Per-school `congested_frac` values are proxies, not exact.

### 7.2. Origin-Level Comparison (Section 6.4)

For each origin school, compares:
- **Observed:** `obs_esc_students` — all students at ESC schools from this origin (from `df_flow_tagged`)
- **Predicted:** `model_predicted_esc` — sum of `mu_current_subsidy` across all ESC destinations for that origin

**Results:**
- Congested-feeding tag validated: tagged origins (6,554) have mean 53.0% of students going to congested public JHS; non-tagged origins (1,068) have 0.0%
- Correlation between observed and predicted ESC enrollment per origin: **0.325** (moderate)

### 7.3. ESC-School-Level Comparison (Section 6.5)

For each ESC school, directly compares:
- **Observed (all-student):** `obs_congested_frac` — share of all students from congested-feeding origins
- **Observed (beneficiary-based):** `obs_benef_congested_frac` — share of ESC beneficiaries from congested-feeding origins (apples-to-apples with model)
- **Model:** `congested_frac` — share of predicted demand from congested-feeding origins (from raw mu)

**System-level results:**

| Metric | Value |
|---|---|
| Observed (all students) | 0.7501 |
| Observed (beneficiaries only) | 0.7435 |
| Model (from mu ratios, enrollment-weighted) | 0.9361 |
| Diff (model - obs beneficiaries) | +0.1926 |

**Per-school results:**
- Correlation (model vs observed all-student): 0.345
- Correlation (model vs observed beneficiary): 0.344
- Mean per-school overestimate (beneficiary basis): +0.221

**Interpretation:** The model systematically overestimates the congested fraction per school (mean +0.22). This is because the raw mu ratios reflect unconstrained predicted demand — heavily weighted toward congested-feeding origins due to proximity — while observed flows reflect realized enrollment which is more diffuse. The moderate correlation (0.34) means the model captures the rank ordering of schools' congested-feeding exposure reasonably but not precisely. For the aggregate decongestion analysis, the overestimate means the ~95.7% system-level finding is an upper bound; the observed beneficiary-based system fraction is ~74%.

### 7.4 Interpretation

**The capacity gap reveals untapped decongestion potential.** The ESC system currently enrolls ~74,232 beneficiaries, but the model predicts ~100,000 students would fill slots if all 106,654 contracted slots were fully utilized. The ~25,000 marginal students represent the additional absorption capacity beyond current enrollment, and 98% of them would come from origins that also feed congested public JHS. Virtually all of the ESC system's unused capacity is geographically positioned to relieve congested schools. However, 1,072 ESC schools currently have 26,331 unutilized slots — the barrier is not a lack of demand (demand exceeds supply 5.3x) but that students have not yet been matched to these available slots.

**Subsidy reduction is not the lever; the slot ceiling is.** Reducing net cost by 20,000 pesos — effectively making ESC free — only adds ~1,826 students (+1.8%) to the predicted flow. Distance is the dominant factor in the NBR model (approximately 4x more important than cost). When 563,000 candidates compete for 107,000 slots, cheaper tuition shuffles which students fill slots but barely changes the total. The policy implication is that expanding the number of contracted ESC slots, or reallocating existing slots from surplus to deficit areas (see notebook 2.4g), would be far more effective than increasing the subsidy.

**Over half the untapped decongestion potential lies in currently unobserved paths.** Of the ~25,000 marginal students, 53.9% (13,621) would flow through origin-destination pairs where no student currently enrolls. These hypothetical paths are geographically feasible (predicted by the NBR model based on distance, cost, and school characteristics) but have not yet been activated. This suggests that information barriers or inertia — not just capacity — may limit the ESC system's decongestion reach. Targeted outreach to students from congested-feeding origins about nearby ESC schools they have not yet considered could activate these latent pathways.

**The ESC system's decongestion role is structurally significant but bounded.** Currently, 416,458 students flow to the 1,254 congested public JHS (23.3% of all G7 students). The model's predicted congested-feeding flow of ~95,695 represents about 23% of that congestion volume — a substantial structural counterweight. The marginal component (~24,820 additional students) represents about 6% of the congested flow. Even full utilization of existing ESC capacity would not eliminate congestion, but it would meaningfully reduce pressure on the most crowded public schools.

**The model's congested fraction is directionally correct but overstated at the school level.** The model assigns 94% of predicted demand to congested-feeding origins, whereas the observed beneficiary-based fraction is 74%. This ~20 percentage point gap arises because the NBR's distance-dominated predictions concentrate predicted flow on nearby origins — which tend to be congested-feeding — while actual enrollment is more geographically diffuse. At the system level, this means the 95.7% "congested share" of predicted flow is an upper bound; the true decongestion-relevant share is likely closer to the observed 74%. The core finding — that the large majority of ESC flow is decongestion-relevant — holds under either estimate.

---

## 8. Data Dependencies

| File | Used For |
|---|---|
| `output/full_candidate_beneficiary_pool_without_probdist_0209_clusteredSE.parquet` | CBP with mu values (325,395 origin-ESC pairs) |
| `output/processed_esc_slots.parquet` | ESC slots: `slots_total`, `slots_billed` (admin), `slots_unutilized` |
| `output/processed_project_bukas_school_information.parquet` | School names, divisions, metadata |
| `output/grade_7_student_flow_table_sy2324.parquet` | Observed student flows (for existing/hypothetical classification, candidate pool, `count_esc_beneficiary`) |
| `output/analysis_payload/flow_to_congested.parquet` | Origins feeding congested public JHS (for `congested_frac`) |

---

## 9. Simulation Implementation Notes

### Coupled Depletion
Same as 2.4f: each iteration shuffles all 325,395 paths, iterates in random order, and accepts `min(origin_capacity, slot_capacity, mu_value)`. Both origin candidate pool and destination slot pool are depleted.

### Early Termination
Iteration stops when either total remaining candidates or total remaining slots reaches zero.

### Monte Carlo
100 iterations (seeds 0–99). System-level std is small (~10–15 on ~100K total), indicating stable convergence. Per-school variance is higher due to shuffle-order effects.

### 2.4i Refactoring Improvements over 2.4h
- Single `run_simulation_detailed()` function (2.4h had both detailed and non-detailed)
- `compute_decongestion_stats()` — extracted into reusable function (2.4h copy-pasted 4 times)
- `build_school_dataframe()` — extracted into reusable function
- Scenario loop runs all scenarios in one cell (2.4h had separate sections)
- Results table built programmatically from `all_decongestion_stats` (2.4h manually transcribed)
- `congested_frac` validated against `per_origin` simulation data
- `is_over_enrolled` flag distinguishes structural negative marginals from model under-predictions
- ESC slots filtered by CBP destinations only (2.4h used an OR filter that included origin IDs)
- Consistent billed total from a single `system_slots` source
- Baseline-relative percent changes added to results table
- Observed baseline section (`df_flow_tagged`, `obs_by_origin`, `obs_by_esc`) for ground-truth comparison
- Origin-level and ESC-school-level observed vs predicted comparison cells (Sections 6.4–6.5)
- Cross-scenario table includes `observed (benef)` row for direct comparison with simulation results
- Standardized labels across all cells to disambiguate the four observed ESC quantities
- `obs_by_esc` tracks both all-student and beneficiary-only columns for congested-origin flows
- `system_slots` includes `count_esc_beneficiary` from student flow (joined from `student_flow`)
- `current_subsidy` scenario removed from sweep; five net cost reduction scenarios remain (-1k through -20k)

---

## 10. KDD 2026 Paper Writing Progress

**Target:** SIGKDD 2026 AI for Science Track

### Paper Structure (Methodology & Results)

| Section | Label | Status |
|---|---|---|
| 3.1 | Data Sources | Written (by team) |
| 3.2 | Empirical Model of Student Flows | Written (by team) |
| 3.3 | Candidate Beneficiary Pool and Policy Simulation | Written (by team) |
| **3.4** | **Counterfactual Decongestion via Choice-Based Policy Simulation** | **Drafted** |
| 4.1 | Student Flow Modeling Results | Written (by team) |
| **4.2** | **Decongestion Simulation Results** | **Drafted** |
| 4.3 | Student Redistribution | Likely deprecated — subsumed by 4.2 |
| 5 | Limitations and Ethical Considerations | Written (by team) |

### Section 3.4 — Methodology Draft

**File:** `references/paper_drafts/section_3.4_methodology.tex`

Covers:
- Doubly constrained stochastic allocation (coupled depletion of $E_i$ and $K_j$)
- Acceptance rule: $a_{ij} = \min(e_i, k_j, \hat{y}_{ij}^{(s)})$
- Decongestion attribution via congested fraction $\phi_j$ (Eq. \ref{eq:congested-frac})
- Two measures: total predicted role ($D_j^{\text{total}}$) and marginal potential ($D_j^{\text{marg}}$)
- Existing vs hypothetical path decomposition
- Five counterfactual scenarios (-1K to -20K net cost reduction)

**Notation aligned to 3.2–3.3:** $\hat{y}_{ij}^{(s)}$ (not $\mu$), $E_i$ (candidate pool), $c_{ij}$ (net cost), $F_{ij}$ (observed flow), $Y_j^{(s,r)}$ (simulated enrollment).

### Section 4.2 — Results Draft

Covers:
- Table 2: Cross-scenario simulation results (predicted flow, Δ from observed, Δ from -1K)
- Slot ceiling dominates price sensitivity (+1.8% across full subsidy range)
- Decongestion geographically concentrated (95.7% from congested-feeding origins)
- 53.9% of marginal from hypothetical (unobserved) pathways
- Decongestion meaningful but bounded (~6% of total congested flow)

### Introduction (Last Two Paragraphs) — Revised

Reframed three contributions for AI for Science track:
1. Data harmonization → first national-scale OD flow graph
2. Gravity model → scientific finding (distance dominates cost 4x, challenges subsidy-first assumption)
3. Doubly constrained simulation → structural discovery (capacity-bound, latent pathways)

Generalizability reframed from "other governments" to "spatial allocation problems where behavioral models must be reconciled with capacity constraints."

### Open Issues
- **Candidate pool discrepancy:** Paper 3.3 defines $E_i = \text{Enrollment}_i - \sum_j \text{Beneficiaries}_{ij}$ (non-beneficiaries only), but 2.4i code uses all students (non-bene + bene = 562,958). Needs reconciliation.
- **4.3 deprecation:** Confirm with team that Section 4.3 "Student Redistribution" is fully subsumed by 4.2.
