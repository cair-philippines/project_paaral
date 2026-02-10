# From Redistribution to DCM-Informed Policy Simulation

**Document Created:** 2026-02-05
**Status:** Conceptual Design Phase

---

## 1. Background: Why We Transitioned

### The Original Redistribution Approach (Notebooks 2.4x)

The original algorithm assumed **authoritative control** — that DepEd could direct students to specific schools. This manifested as:
- Deterministic assignment of students to private ESC schools
- Greedy/LP-based allocation treating students as units to be moved
- Optimization objective: maximize redistribution subject to capacity constraints

### The Flaw

This assumption is **false**. DepEd does not have the mandate to exercise authoritative control over where students enroll. Students and families make choices. The original design conflated:
- What the system *wants* (congestion relief)
- What the system *can do* (influence choices via subsidies)

### The Paradigm Shift

| Aspect | Redistribution Model | DCM-Informed Simulation |
|--------|---------------------|------------------------|
| Student agency | Ignored | Central |
| Mechanism | Direct assignment | Subsidy-induced choice |
| Output | Deterministic allocation | Probability distributions |
| Question answered | "Where should students go?" | "Where would students likely go under policy X?" |

---

## 2. The Choice Modeling Foundation

### Model: Negative Binomial Regression (NBR)

Developed in notebook `2.2-pjm-dcm-v3`, the model predicts expected enrollment (`count_of_students`) for origin-destination pairs. Estimated on 29,224 observations with cluster-robust standard errors. Pseudo R² = 0.085. All coefficients significant (p < 0.02).

**Coefficients (Model 4, 08 Feb 2026):**

| Variable | Coef | Std Err | z | p | Interpretation |
|---|---|---|---|---|---|
| Intercept | +3.036 | 0.115 | 26.4 | 0.000 | Baseline expected log-count |
| `log_distance` | **-0.4496** | 0.010 | -46.3 | 0.000 | Strongest predictor. 1% ↑ distance → ~0.45% ↓ enrollment |
| `log_net_cost_k` | **-0.1104** | 0.013 | -8.4 | 0.000 | 1% ↑ net cost → ~0.11% ↓ enrollment. Cost matters, but distance matters ~4x more |
| `esc_rating` | -0.0191 | 0.008 | -2.3 | 0.019 | Higher ESC rating → slightly fewer students (counterintuitive; possibly confounded with selectivity or price) |
| `log_origin_lgu_income` | **-0.0524** | 0.005 | -10.9 | 0.000 | Wealthier LGUs → fewer ESC enrollees. ESC most effective in lower-income areas |
| `origin_region` (Central Luzon) | -0.0783 | 0.028 | -2.8 | 0.005 | Fewer students flow from Central Luzon origins (vs reference region) |
| `origin_region` (CALABARZON) | -0.0908 | 0.021 | -4.2 | 0.000 | Fewer from CALABARZON origins |
| `destination_region` (Central Luzon) | +0.0781 | 0.025 | 3.1 | 0.002 | ESC schools in Central Luzon attract more students |
| `destination_region` (CALABARZON) | +0.0969 | 0.018 | 5.3 | 0.000 | ESC schools in CALABARZON attract even more |
| alpha (overdispersion) | 0.3948 | 0.011 | 35.8 | 0.000 | Confirms negative binomial is appropriate |

**Output:**
- `mu`: Expected number of students for each origin-destination pair under a given subsidy scenario
- 22 subsidy scenarios: baseline (0 subsidy), current, minus 1k through minus 20k net cost

### Key Insights from NBR

1. **Distance dominates.** At -0.45, it is ~4x the magnitude of net cost (-0.11). If ESC schools aren't near congested origins, students won't enroll regardless of subsidy amount.

2. **Net cost is significant but secondary.** Subsidy increases do matter, but the effect is modest. Combined with the declining uptake reality, even modest cost sensitivity can drive students out of the system when net cost rises over time.

3. **LGU income reveals targeting opportunity.** The -0.0524 coefficient on `log_origin_lgu_income` means ESC is most effective in lower-income areas. Targeted subsidies should focus there.

4. **ESC rating is counterintuitively negative.** Higher-rated schools attract slightly fewer students — possibly because they charge more, are more selective, or families don't use ratings in decisions. Effect is small (-0.019).

5. **Regional asymmetry.** CALABARZON ESC schools attract more students (+0.097) while CALABARZON origins send fewer (-0.091). Suggests strong ESC supply in CALABARZON but lower demand from its own public schools.

---

## 3. Stakeholder Requirements

Despite transitioning to choice-based modeling, certain normative constraints must be retained:

### Requirement 1: Geographic Accessibility
> "We will not recommend private ESC schools outside the reach of origin schools."

Even if some students historically chose distant schools, policy recommendations should not encourage this pattern for the general population.

### Requirement 2: Prioritization Criteria

Three factors should guide which options are recommended:
1. **Distance**: Prefer nearer ESC schools
2. **Tuition/Net Cost**: Prefer more affordable options
3. **School Size**: Prefer larger schools (proxy: JHS enrollment)

These align with NBR coefficients (distance and net cost are already negative), but stakeholder requires **enforcement**, not just observation.

---

## 4. Proposed Design: Two-Layer Architecture

### Layer 1: Feasibility Filter (Rule-Based)

**Purpose:** Define the universe of acceptable recommendations.

**Constraints:**
- Maximum distance threshold (e.g., ≤ 5km or ≤ 10km)
- Optional: Maximum tuition ceiling
- Optional: Minimum school size

**Effect:** Creates a *feasible candidate pool* from the full candidate beneficiary pool.

### Layer 2: Demand Estimation (Deterministic)

**Purpose:** Estimate predicted enrollment using mu values.

**Method:**
1. For each origin-destination pair in feasible set:
   - Retrieve `mu` (expected enrollment) under chosen subsidy scenario
   - Apply feasibility weight: `weighted_mu = mu × feasibility_weight`
2. No capacity enforcement — accept mu values as-is

### Layer 3: Demand vs Supply Tracking

**Purpose:** Identify where predicted demand exceeds available slots.

**Method:**
1. Sum `weighted_mu` per destination ESC school
2. Compare against `available_slots` (unutilized slots)
3. Flag schools where demand > supply
4. Compute slot gap: `max(0, total_demand - available_slots)`

**Policy Value:**
- Identifies ESC schools needing additional slot allocations
- Quantifies unmet demand under each subsidy scenario

### Workflow Diagram

```
Full Candidate Pool (311K pairs)
        │
        ▼
┌─────────────────────────────────┐
│  LAYER 1: Feasibility Filter   │
│  - Compute feasibility_weight  │
│  - weighted_mu = mu × weight   │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  DEMAND vs SUPPLY TRACKING     │
│  - Sum weighted_mu per ESC     │
│  - Compare against slots       │
│  - Flag: demand > supply       │
│  - Compute slot gap            │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  AGGREGATION                   │
│  - Sum by congested school     │
│  - Compare against aisle       │
│    learners                    │
└─────────────────────────────────┘
        │
        ▼
Policy Simulation Output
  ├── Predicted diversions (unconstrained)
  ├── ESC schools with excess demand
  └── Slot gap per ESC school
```

---

## 5. Literature Validation

The proposed design aligns with established research in school choice, mechanism design, and policy simulation.

### 5.1 School Choice with Capacity Constraints

**Barbara, Rey, Rashidi, & Nair (2024)** - [School choice modeling and network optimization in an urban environment](https://link.springer.com/article/10.1007/s00168-023-01230-5)
- Models school choice where preferred schools may be unavailable due to capacity
- Students forced to attend schools offering less utility when first choice is full
- Validates our approach of combining choice prediction with capacity constraints

**Harris & Larsen (2018)** - [You can't always get what you want: Capacity constraints in a choice-based school system](https://www.sciencedirect.com/science/article/abs/pii/S0272775717302571)
- Studies gap between preferred and actual school quality in New Orleans
- Demonstrates that choice systems must account for capacity limitations
- Supports our two-layer design separating feasibility from choice

### 5.2 Mechanism Design with Constraints

**Abdulkadiroglu et al.** - [School Choice: A Mechanism Design Approach](https://www.jstor.org/stable/3132114)
- Foundational work on deferred acceptance algorithm
- Establishes that school choice mechanisms must balance efficiency and fairness
- Informs our prioritization criteria design

**Ehlers et al.** - [School Choice with Controlled Choice Constraints: Hard Bounds Versus Soft Bounds](https://www.researchgate.net/publication/228193083_School_Choice_with_Controlled_Choice_Constraints_Hard_Bounds_Versus_Soft_Bounds)
- Distinguishes hard constraints (must be satisfied) from soft constraints (preferred)
- Our distance threshold = hard bound; prioritization score = soft preference
- Validates the two-layer approach

### 5.3 Voucher/Subsidy Simulation

**Ferreyra (2007)** - [Estimating the Effects of Private School Vouchers in Multidistrict Economies](https://www.aeaweb.org/articles?id=10.1257/aer.97.3.789)
- General equilibrium model of voucher effects on enrollment
- Simulates how subsidies affect private school choice
- Validates using choice models for policy counterfactuals

**Dynarski et al.** - [Where Does Voucher Funding Go?](https://www.nber.org/papers/w21687)
- Estimates price elasticity of demand for private schooling
- Shows subsidy effects on enrollment vs. pricing
- Supports our use of NBR to predict enrollment response to subsidy changes

### 5.4 Multiple Constraints in Choice Models

**Bhat (2012)** - [Accommodating multiple constraints in the MDCEV choice model](https://ideas.repec.org/a/eee/transb/v46y2012i6p729-743.html)
- Extends choice models to handle multiple resource constraints
- Ignoring multiple constraints leads to poor fit and inconsistent estimation
- Supports our inclusion of distance, cost, and capacity constraints

### 5.5 Constrained Priority Mechanisms

**Combining Outcome-Based and Preference-Based Matching** - [Cambridge Political Analysis](https://www.cambridge.org/core/journals/political-analysis/article/combining-outcomebased-and-preferencebased-matching-a-constrained-priority-mechanism/7B159412B53FF81769F32BF55E2FCB94)
- Describes mechanisms that balance predicted outcomes with stated preferences
- Relevant to our design where choice model predicts outcomes, but stakeholder preferences constrain recommendations

---

## 6. Design Decisions (Finalized)

### 6.1 Distance Threshold ✅
- **Decision:** Graduated penalty from 3km to 5km
- **Rationale:** Soft constraint that penalizes distance progressively rather than hard cutoff
- **Implementation:** Options within 3km get no penalty; 3-5km get graduated penalty; beyond 5km excluded or heavily penalized

### 6.2 Probability Distribution Usage ✅ (REVISED)
- **Original Decision:** Sample from the full PMF for stochastic simulation
- **Revised Decision:** Use `mu` values directly (expected enrollment)
- **Rationale:** Simpler, faster computation; point estimates sufficient for policy comparison
- **Implementation:** Use `mu_*` columns directly instead of sampling from `prob_dist_*` columns

### 6.3 Origins with No Feasible Options ✅
- **Decision:** Allow 0 flow — this reflects real-world constraints
- **Rationale:** Some origin schools may genuinely have no nearby ESC options; forcing allocation would be unrealistic
- **Implementation:** Origins with no feasible destinations contribute 0 to decongestion

### 6.4 Unit of Simulation ✅
- **Decision:** Per congested public school (aggregate inflows)
- **Rationale:** Main goal is decongestion; measuring success at the congested school level aligns with policy objective
- **Implementation:** Aggregate predicted enrollment shifts across all origin-destination pairs feeding into each congested public JHS

### 6.5 Subsidy Scenarios ✅
- **Decision:** Counterfactual scenarios (+5k, +10k, etc.) with current subsidy as baseline
- **Rationale:** Allows policy comparison — "what if we increased subsidy by X?"
- **Implementation:** Use `mu` values for different net cost scenarios already computed in candidate pool

### 6.6 Capacity Constraint Handling ✅ (REVISED)
- **Original Approach:** Enforce capacity constraints (scale down when demand > supply)
- **Revised Decision:** Track demand vs supply gaps instead of enforcing constraints
- **Rationale:** Policy value in identifying where demand exceeds supply; informs slot allocation recommendations
- **Implementation:** Flag ESC schools where predicted demand > available slots; compute slot gap

### 6.7 Prioritization Scoring (To Be Determined)
- **Open question:** How to weight distance, tuition, and size in priority score?
- **Consideration:** Should weights be uniform or derived from NBR coefficients?

---

## 7. Terminology

| Old Term | New Term | Rationale |
|----------|----------|-----------|
| Redistribution | Policy simulation | We simulate outcomes, not assign students |
| Redirection | Subsidy-induced enrollment shift | Describes mechanism accurately |
| Redistribution algorithm | Allocation heuristic | Still uses rules, but informed by choice model |
| Aisle learners to be redistributed | Potential beneficiaries | They may choose ESC if subsidized |

---

## 8. Design Evolution (2026-02-06)

### 8.1 Recognizing Residual "Control" Thinking

Upon reflection, several design elements from the initial approach were symptoms of the authoritative control mindset we intended to abandon:

| Element | Why It's "Control" Thinking |
|---------|----------------------------|
| Distance threshold (5km cutoff) | Imposing our preference on student choices; NBR already captures distance sensitivity |
| Congestion-based prioritization | Deciding which students "should" divert first, rather than predicting who "would" |
| Feasibility filter | Excluding options we deem inappropriate, rather than letting the model predict behavior |

**Key insight:** The NBR model already encodes distance and cost preferences through its coefficients. Adding rule-based constraints on top conflates behavioral prediction with normative policy.

### 8.2 Understanding the Data: Existing vs Hypothetical Paths

The `full_candidate_beneficiary_pool.parquet` contains **two types of origin-destination pairs**:

| Type | Description | Source |
|------|-------------|--------|
| **Existing paths** | Pairs where students actually flowed (observed in historical data) | Matched in `grade_7_student_flow_table_sy2324.parquet` |
| **Hypothetical paths** | Pairs generated as potential options (no observed flow) | Generated based on proximity/feasibility, not observed |

This distinction is critical because the policy question shifts:

> **Old question:** "How do we allocate students to ESC schools?"
>
> **New question:** "What is the marginal contribution of hypothetical paths to decongestion?"

### 8.3 The Refined Policy Question

Colleague's framing:
> "DepEd could investigate the feasibility of those hypothetical paths. Ideally, we have some value to qualify the marginal contribution of those paths to decongestion. That is, adding those new paths contributes an additional X to decongestion."

**Policy lever:** Subsidy amount (net cost reduction) — already captured in mu columns for different scenarios.

**Simulation goal:** Quantify how much decongestion each path (existing or hypothetical) contributes under different subsidy scenarios.

---

## 9. Revised Design: Path-Level Marginal Analysis

### 9.1 Overview

Instead of filtering and allocating, we now:
1. Flag each path as existing or hypothetical
2. Link each path to the congested school(s) it diverts from
3. Compute decongestion contribution per path under each subsidy scenario
4. Compare existing vs hypothetical contributions

### 9.2 Implementation Steps

**Step 1: Create `is_hypothetical` Flag**
- Join candidate beneficiary pool with observed student flow data
- If (origin, destination) pair exists in flow data → `is_hypothetical = FALSE`
- Otherwise → `is_hypothetical = TRUE`

**Step 2: Compute Proportional Weights for Congested Schools**
- An origin may feed multiple congested public schools
- Compute weight = flow to congested school C / total flow to all congested schools
- This distributes a path's contribution proportionally

**Step 3: Compute Path-Level Metrics for All Scenarios**

| Scenario | mu Column | Marginal (vs Current) |
|----------|-----------|----------------------|
| Baseline (0 subsidy) | `mu_baseline_0_subsidy` | — |
| Current subsidy | `mu_with_current_subsidy` | (baseline) |
| -1k net cost | `mu_with_minus_1k_net_cost` | `mu_minus_1k - mu_current` |
| -10k net cost | `mu_with_minus_10k_net_cost` | `mu_minus_10k - mu_current` |
| -15k net cost | `mu_with_minus_15k_net_cost` | `mu_minus_15k - mu_current` |
| -20k net cost | `mu_with_minus_20k_net_cost` | `mu_minus_20k - mu_current` |

**Step 4: Explode by Congested School and Compute Decongestion**
- Each path becomes multiple rows (one per congested school it diverts from)
- `decongestion_contribution = mu × proportional_weight`
- `marginal_decongestion = marginal_mu × proportional_weight`

**Step 5: Handle Edge Cases**
- Origins not feeding any congested school: Keep in dataset, set decongestion = 0
- Paths with no observed flow: Mark as hypothetical

### 9.3 Output Schema

**Primary table (path-level, exploded by congested school):**

| Column | Description |
|--------|-------------|
| `origin_school_id` | Origin school |
| `destination_school_id` | ESC destination |
| `congested_school_id` | Congested public school this diverts from |
| `is_hypothetical` | TRUE if no observed flow |
| `proportional_weight` | Share of origin's flow to this congested school |
| `mu_baseline` | Expected enrollment (0 subsidy) |
| `mu_current` | Expected enrollment (current subsidy) |
| `mu_minus_1k` | Expected enrollment (-1k net cost) |
| `mu_minus_10k` | Expected enrollment (-10k net cost) |
| `mu_minus_15k` | Expected enrollment (-15k net cost) |
| `mu_minus_20k` | Expected enrollment (-20k net cost) |
| `decongestion_current` | `mu_current × weight` |
| `decongestion_minus_10k` | `mu_minus_10k × weight` |
| `marginal_decong_10k` | `(mu_minus_10k - mu_current) × weight` |
| ... | (similar for all scenarios) |

### 9.4 Aggregations

| Aggregation | Policy Question |
|-------------|-----------------|
| By congested school | "How much total decongestion does school C receive?" |
| By hypothetical flag | "What % of decongestion comes from hypothetical paths?" |
| By ESC destination | "Which ESC schools contribute most to decongestion?" |
| By scenario | "How does increasing subsidy affect decongestion?" |
| Hypothetical × scenario | "At what subsidy level do hypothetical paths become significant?" |

### 9.5 Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Existing path definition | Any observed flow (ESC or non-beneficiary) | Broader definition captures known travel patterns |
| Multiple congested schools | Proportional distribution by flow volume | Fair attribution without double-counting |
| Origins without congested linkage | Keep with decongestion = 0 | Complete picture; may still have ESC value |
| Output scope | All paths (existing + hypothetical) | Enables comparison and full analysis |
| Distance constraint | **Removed** | NBR already captures distance preference; no need for additional filter |

---

## 10. Key Findings (Notebook 2.4e)

### 10.1 Path Classification Results

Of the 311,074 origin-destination pairs in the candidate beneficiary pool:

| Path Type | Count | Percentage |
|-----------|-------|------------|
| **Existing** | 76,102 | 24% |
| **Hypothetical** | 234,972 | 76% |

**Implication:** Three-quarters of the paths in our candidate pool are routes that students have not yet used. These represent untapped opportunities for decongestion.

### 10.2 Decongestion Contribution by Path Type

At the -10k net cost scenario:

> **Hypothetical paths contribute 61.3% of total predicted decongestion.**

| Path Type | Share of Decongestion |
|-----------|----------------------|
| Existing | ~39% |
| Hypothetical | ~61% |

**Implication:** The majority of decongestion potential comes from paths that students haven't used yet. Simply increasing subsidies on existing paths will not maximize decongestion impact.

### 10.3 Congested Schools Most Dependent on Hypothetical Paths

Some congested public schools have decongestion potential that depends almost entirely on hypothetical paths:

| Congested School | Existing | Hypothetical | Total | % Hypothetical |
|------------------|----------|--------------|-------|----------------|
| 301190 | 763.5 | 1,800.4 | 2,563.9 | 70.2% |
| 301196 | 785.3 | 1,737.9 | 2,523.2 | 68.9% |
| 305330 | 245.9 | 2,255.8 | 2,501.7 | **90.2%** |
| 301186 | 276.4 | 2,220.8 | 2,497.2 | **88.9%** |
| 301192 | 1,060.7 | 1,393.3 | 2,454.0 | 56.8% |

**Implication:** Schools like 305330 and 301186 (with ~90% hypothetical contribution) will see minimal decongestion benefit from subsidy increases alone. Targeted interventions to enable hypothetical paths are required.

### 10.4 Policy Implications

1. **Subsidy increases alone are insufficient.** 61% of decongestion potential lies in paths students don't currently use.

2. **Hypothetical paths must be "activated."** This could involve:
   - Information campaigns to make families aware of ESC options
   - Partnerships with ESC schools not currently drawing from certain areas
   - Strategic slot allocation to schools serving underutilized routes

3. **Some congested schools need targeted intervention.** Schools with 80-90% hypothetical dependency cannot be decongested through general subsidy policy — they need specific actions to enable new paths.

4. **Existing paths still matter.** 39% of decongestion comes from paths already in use. Subsidy increases will strengthen these flows.

---

## 11. Key Findings (Notebook 2.4f — Constrained Simulation)

### 11.1 Simulation Design

Notebook 2.4e revealed that unconstrained mu values produce unrealistic totals. Notebook 2.4f addresses this with a **constrained Monte Carlo simulation**:

- **Coupled depletion**: A diversion only occurs when both the origin has candidates AND the destination has slots
- **Raw mu values**: No flooring — fractional predictions preserved and accumulated
- **Candidate pool scoped**: Only non-beneficiaries flowing to congested schools (400,936 students)
- **ESC slot constraint**: 26,491 available slots across 1,088 schools
- **Monte Carlo**: 50-100 iterations per scenario, shuffling path order each time

### 11.2 System Constraint Discovery

| Resource | Total | Consumed | Remaining | % Consumed |
|----------|-------|----------|-----------|------------|
| Candidate pool | 400,936 | ~25,936 | ~375,000 | **6.5%** |
| ESC slots | 26,491 | ~25,936 | ~555 | **97.9%** |

> **ESC slots are the binding constraint.** Demand overwhelms supply — slots are nearly fully depleted while the candidate pool is barely touched.

### 11.3 Constrained Existing vs Hypothetical Split

Monte Carlo results (100 iterations, minus_1k scenario):

| Metric | Mean | 95% CI |
|--------|------|--------|
| Total diverted | 25,936 | ~stable |
| Existing paths | 11,737 (45.3%) | |
| Hypothetical paths | 14,198 (**54.7%**) | [54.1%, 55.3%] |

Compared to the unconstrained estimate (61.3% hypothetical in 2.4e), the constrained simulation yields a lower but still majority hypothetical contribution at **54.7%**.

### 11.4 Subsidy Increases Have Negligible Impact

| Scenario | Total Diverted | Existing | Hypothetical | % Hypothetical | Remaining Slots |
|----------|---------------|----------|--------------|----------------|-----------------|
| Current subsidy | 25,931 | 11,730 | 14,202 | 54.8% | 560 |
| -1k net cost | 25,936 | 11,733 | 14,203 | 54.8% | 555 |
| -3k net cost | 25,945 | 11,741 | 14,204 | 54.7% | 546 |
| -5k net cost | 25,951 | 11,748 | 14,203 | 54.7% | 540 |
| -10k net cost | 25,962 | 11,756 | 14,206 | 54.7% | 529 |
| -15k net cost | 25,972 | 11,760 | 14,212 | 54.7% | 519 |

**A 15,000-peso subsidy increase produces only ~41 additional students diverted** (0.16% marginal gain). The system is slot-constrained: demand already exceeds supply at current subsidy levels.

### 11.5 Policy Implications (Revised)

1. **The bottleneck is slot capacity, not subsidy amount.** Increasing ESC subsidies yields negligible additional decongestion because slots fill up regardless.

2. **The most impactful policy lever is expanding ESC slot capacity** — through new school partnerships, expanding existing school capacity, or reallocating underutilized slots across regions.

3. **Hypothetical paths remain the majority contributor (54.7%)** even under constraints, confirming that enabling new paths is essential for maximizing decongestion.

4. **The existing/hypothetical split is robust** — the 95% CI of [54.1%, 55.3%] across 100 Monte Carlo iterations shows this is not an artifact of processing order.

### 11.6 School-Level Disaggregation (Sections 8.1, 8.2, 8.3 of 2.4f)

While system-level totals are stable (~25,936 ± 10.5), disaggregating to individual schools reveals the heterogeneity hidden by the aggregate.

**Section 8.1 — Per ESC School CI:**
- For each ESC school, the mean and 95% CI of accepted student flow across 100 Monte Carlo iterations, split by existing, hypothetical, and both
- Reveals which ESC schools show high variance (sensitive to shuffle order / competition for slots) vs stable absorption

**Section 8.2 — Per Congested Public JHS CI:**
- Diversions attributed back to congested public JHS using **proportional weights**: when origin X diverts students, the reduction is distributed across congested schools that X feeds, proportional to observed non-beneficiary flow volume
- For each congested school, the mean and 95% CI of decongestion across 100 iterations, split by existing, hypothetical, and both
- Identifies which congested schools benefit most from the ESC program and how dependent their relief is on hypothetical paths

**Section 8.3 — Multi-Scenario Comparison (Detailed):**
- Runs 100 iterations per scenario (current through -15k), retaining per-iteration results
- Reports mean, std for existing, hypothetical, and both per scenario
- Reports raw and percentage reduction to system congestion (denominator: total_candidates = 400,936) with 95% CI
- Confirms subsidy negligibility at the scenario level with proper uncertainty quantification

**Exports:**
- `esc_school_ci.csv` — per ESC school CI from Section 8.1
- `congested_school_ci.csv` — per congested JHS CI from Section 8.2

### 11.7 ESC School-Level Unmet Demand Analysis (Section 10 of 2.4f)

Since the system is slot-constrained, we identified which ESC schools would contribute most to decongestion if given additional slots.

**Methodology:**
- Filtered CBP to origins feeding congested schools (via `cand_pool`)
- Summed predicted demand (mu) per ESC destination school
- Compared against aggregated available slots per school
- Computed `unmet_demand = max(0, total_demand - available_slots)`

**Enriched metrics per ESC school:**

| Metric | Description |
|--------|-------------|
| `total_predicted_demand` | Sum of mu across all congested-feeding origins |
| `available_slots` | Aggregated ESC slots at this school |
| `unmet_demand` | Demand exceeding available slots |
| `demand_to_slot_ratio` | How many times over slots are demanded |
| `n_congested_origins` | Distinct feeder origin schools |
| `n_congested_destinations` | Distinct congested public JHS this ESC school could help decongest |
| `demand_existing` / `demand_hypothetical` | Demand split by path type |

**Key distinction:**
- `n_congested_origins` = feeder schools whose non-beneficiaries flow to congested JHS and could divert to this ESC school
- `n_congested_destinations` = congested public JHS that this ESC school could help decongest (traced via shared origins)

### 11.8 Distance-Based Unmet Demand (Section 10.1 of 2.4f)

**Question:** Which ESC schools have high unmet demand from *nearby* origin schools feeding students to congested JHS?

**Methodology:**
- Binned origin-ESC distances into **≤3 km**, **3–5 km**, and **>5 km**
- Summed predicted demand per ESC school per distance band
- Defined "nearby demand" as ≤5 km (sum of ≤3 km and 3–5 km bands)
- Computed `nearby_unmet_demand = max(0, nearby_demand - available_slots)`
- Split nearby demand into existing vs hypothetical paths

**Policy relevance:** ESC schools with high nearby unmet demand are the most actionable expansion targets — their potential beneficiaries are geographically proximate, making diversion realistic and practical. The existing/hypothetical split within the nearby band indicates whether expansion would reinforce known flows or require activating new paths.

**Exports:**
- `esc_unmet_demand_ranking.csv` — full ranking with enriched metrics
- `esc_nearby_unmet_demand_ranking.csv` — distance-based ranking with nearby demand breakdown

---

## 12. Choice-Respecting Mechanisms Leveraging DCM Results

The earlier framing positioned redistribution as purely "authoritative control." In reality, established mechanisms exist that use demand model outputs to design policy while preserving family agency. The NBR serves as a **prediction engine for policy optimization** — the planner uses predicted behavior to design the choice environment (subsidies, capacity, information, menus), while families retain full choice.

| Authoritative Control | Choice-Respecting Design |
|---|---|
| Assigns students to schools | Designs the menu/subsidies/information families choose from |
| Uses optimization to maximize system objective | Uses prediction to anticipate behavioral response |
| NBR output = allocation target | NBR output = demand forecast for policy calibration |

### 12.1 Capacity Design (Strategic Slot Allocation)

**Sources:** Afacan & Van der Linden, "Capacity Design in School Choice," *Games and Economic Behavior* 146, 2024. Hammond & Xu, "Designing School Choice Mechanisms," *Economic Inquiry*, 2024.

The planner decides how many ESC slots to contract at each school before families choose. The DCM tells you where demand is and where marginal seats yield the largest welfare gains. Families still choose freely — only the supply side is adjusted.

**ESC fit:** Directly maps to the unmet demand analysis (Sections 10/10.1 of 2.4f). NBR-predicted demand per ESC school, compared to available slots, informs where DepEd should expand capacity.

### 12.2 Personalized Subsidy Rules (Marginal Treatment Effects)

**Sources:** Chen & Xie, "Personalized Subsidy Rules," arXiv:2202.13545, 2022. Javaudin, de Palma & Araldo, "Large-Scale Allocation of Personalized Incentives," IEEE ITSC 2022.

Instead of uniform subsidies, set different voucher amounts for different student types based on who is at the margin of switching. Students with high predicted enrollment even without subsidy get lower vouchers (infra-marginal); students near the switching threshold get higher vouchers (highest marginal return per peso spent). Key theoretical result: subsidy rules weakly dominate treatment rules (direct assignment) because they implicitly target through unobserved heterogeneity.

**ESC fit:** The NBR estimates heterogeneous price sensitivity via `log_net_cost`. Students whose mu jumps significantly between subsidy scenarios are the marginal cases worth targeting. Polynomial-time algorithm demonstrated at scale (200K individuals).

### 12.3 Information Provision / Smart Matching Platforms

**Sources:** Arteaga, Kapor, Neilson & Zimmerman, "Smart Matching Platforms and Heterogeneous Beliefs in Centralized School Choice," *QJE* 137(3): 1791-1848, 2022.

A platform uses back-end demand predictions to give families personalized information about admission chances and school fit. Deployed nationally in Chile — 22% of families changed applications upon receiving warnings, reducing non-placement risk by 58%.

**ESC fit:** Directly addresses the hypothetical paths finding — 54.7% of decongestion potential comes from paths families haven't used, possibly due to information gaps. NBR predictions could power recommendations: "Given where you live, here are ESC schools where you'd be a strong candidate."

### 12.4 Econometric Market Design (Simulation-Based Policy Evaluation)

**Sources:** Pathak & Shi, "How Well Do Structural Demand Models Work?," *Journal of Econometrics* 222(1): 161-195, 2021. Pathak & Shi, "Simulating Alternative School Choice Options in Boston," MIT Blueprint Labs, 2013.

The DCM becomes a simulation engine: for each candidate policy design, simulate family choices and compute outcome metrics. Used in Boston to redesign school choice affecting 9,500 children annually. Structural demand models capture stable preference distributions across policy regimes.

**ESC fit:** The constrained Monte Carlo simulation (notebook 2.4f) already does this at a basic level. The extension: simulate joint policy changes (subsidy + slot expansion + information) and evaluate combined impact.

### 12.5 Incentive-Compatible Menu Design

**Sources:** Dizon-Ross & Zucker, "Mechanism Design for Personalized Policy: A Field Experiment Incentivizing Exercise," NBER WP 33624, 2025.

Design a menu of voucher contracts that families self-select into. Example: (a) high-value voucher valid only at schools in underserved areas, (b) medium voucher at any school, (c) small universal voucher. The menu is engineered so truthful self-selection leads to the planner's desired allocation. Almost doubled treatment effect vs one-size-fits-all, with no cost increase.

**ESC fit:** NBR's preference heterogeneity (distance-sensitive vs price-sensitive families) identifies latent types that sort into different contracts. The WTP distribution determines menu spacing.

### 12.6 Targeted Vouchers with Supply-Side Competition

**Sources:** Neilson, "Targeted Vouchers, Competition Among Schools, and the Academic Achievement of Poor Students," submitted to *Econometrica*, revised April 2025.

Chile's SEP program raised transfers by 50% for disadvantaged students. Structural model shows this changed the marginal revenue from quality improvement, inducing schools to compete on quality. Quality markdowns are larger in poorer areas.

**ESC fit:** If DepEd targets higher ESC subsidies at disadvantaged students, the NBR predicts enrollment response. The policy question shifts from "will students move?" to "will schools respond by improving quality or capturing rents?"

### 12.7 Reserve Design (Priority-Based Allocation)

**Sources:** Dur, Kominers, Pathak & Sonmez, "Reserve Design: Unintended Consequences and the Demise of Boston's Walk Zones," *JPE* 126(6): 2457-2479, 2018. Phan, Tierney & Zhou, "Crowding in School Choice," *AER* 114(8): 2526-2552, 2024.

Schools have multiple seat categories (e.g., 50% for nearby students, 50% open). Families rank freely; the reserve structure determines priority. DCM predicts how many students of each type apply to each school, enabling reserve sizing that achieves demographic/geographic targets without mandating assignments.

**ESC fit:** ESC slots could be partially reserved for students from specific congested schools, with the remainder open. The NBR predicts uptake under different reserve structures.

### 12.8 Experimental Validation of Voucher Welfare

**Sources:** Arcidiacono, Muralidharan & Singleton, "Experimentally Validating Welfare Evaluation of School Vouchers," NBER WP 32968, 2024.

Uses experimental variation to validate structural choice models and compute the marginal value of public funds (MVPF) for different voucher designs. Targeted vouchers to households with limited assets yielded MVPF > 3 (each peso generates 3+ pesos of welfare). Critical finding: all estimated models underpredicted take-up because the voucher induced household search and supply-side response.

**ESC fit:** Caution for our work — static NBR predictions may understate impact because the voucher itself changes information sets and school behavior. Suggests augmenting with a search or dynamic updating component.

### 12.9 Most Promising for KDD Paper

1. **Capacity Design (12.1)** — already supported by Sections 10/10.1 of 2.4f
2. **Personalized Subsidies (12.2)** — leverages heterogeneous price sensitivity already in the NBR
3. **Information Provision (12.3)** — directly addresses the hypothetical paths finding
4. **Econometric Market Design (12.4)** — extends the existing simulation framework

---

## 13. Next Steps

1. ~~Finalize distance threshold based on data exploration~~ (Removed — not needed)
2. ~~Implement Layer 1 (feasibility filter)~~ (Removed — no longer filtering)
3. ~~Create `is_hypothetical` flag by joining with observed flow data~~ ✅
4. ~~Compute proportional weights for origins feeding multiple congested schools~~ ✅
5. ~~Implement path-level marginal analysis for all subsidy scenarios~~ ✅
6. ~~Generate aggregated summaries for policy insights~~ ✅
7. ~~Constrained simulation with Monte Carlo~~ ✅
8. ~~Cross-scenario comparison~~ ✅
9. ~~Simulate impact of expanding ESC slot capacity (the binding constraint)~~ ✅ (Addressed via unmet demand analysis)
10. ~~Identify which ESC schools/regions would benefit most from additional slots~~ ✅ (Section 10 & 10.1 of 2.4f)
11. Validate constrained simulation results against observed ESC uptake patterns
12. Aggregate unmet demand findings by region/division for policy-level recommendations
13. Develop narrative connecting slot-constrained finding → unmet demand ranking → distance-based prioritization for paper
14. Select and implement choice-respecting mechanism(s) from Section 12 for paper contribution

---

## 14. Related Files

- **Choice Model Notebook:** `references/2.2-pjm-dcm-v3`
- **Candidate Pool (updated):** `output/full_candidate_beneficiary_pool_without_probdist_0207_model4.parquet`
- **Observed Flow Data:** `output/grade_7_student_flow_table_sy2324.parquet`
- **Flow to Congested:** `output/analysis_payload/flow_to_congested.parquet`
- **ESC Slot Availability:** `output/analysis_payload/esc_available.parquet`
- **Inspection Notebook:** `notebooks/1.11.-inspect-dcm-results.ipynb`
- **Original Redistribution:** `notebooks/2.4c.-relieve-congestion-probability-distribution.ipynb`
- **Policy Simulation Notebook (deprecated):** `notebooks/2.4d.-dcm-policy-simulation.ipynb`
- **Path-Level Marginal Analysis:** `notebooks/2.4e.-path-level-marginal-analysis.ipynb`
- **Constrained Simulation:** `notebooks/2.4f.-constrained-path-simulation.ipynb`
- **Analysis Output:** `output/path_marginal_analysis/`
- **Simulation Output:** `output/constrained_simulation/`

---

## Revision Log

| Date | Changes |
|------|---------|
| 2026-02-05 | Initial document created; conceptual design outlined; literature review completed |
| 2026-02-05 | Finalized key design decisions: graduated penalty (3-5km), stochastic PMF sampling, per-congested-school unit, counterfactual scenarios |
| 2026-02-06 | Revised: Use mu values directly instead of PMF sampling (simpler, deterministic) |
| 2026-02-06 | Revised: Track demand vs supply gaps instead of enforcing capacity constraints; added Layer 3 for gap analysis |
| 2026-02-06 | Recognized residual "control" thinking in distance constraints and prioritization logic |
| 2026-02-06 | Identified existing vs hypothetical paths distinction in candidate beneficiary pool |
| 2026-02-06 | Redesigned simulation: Path-level marginal analysis focusing on decongestion contribution |
| 2026-02-06 | Removed distance filtering; NBR already captures distance preference behaviorally |
| 2026-02-06 | Added proportional weight logic for origins feeding multiple congested schools |
| 2026-02-06 | Implemented notebook 2.4e; documented key findings: 76% hypothetical paths, 61% decongestion from hypothetical |
| 2026-02-06 | Identified congested schools with high hypothetical dependency (up to 90%) requiring targeted intervention |
| 2026-02-07 | Updated candidate pool to model 4 (325,395 pairs, 22 scenarios, no prob distributions) |
| 2026-02-07 | Implemented constrained Monte Carlo simulation (notebook 2.4f) with coupled depletion |
| 2026-02-07 | Key finding: System is slot-constrained (97.9% slots consumed vs 6.5% candidates consumed) |
| 2026-02-07 | Key finding: Subsidy increases yield negligible marginal gain (~41 students / 0.16% from 15k increase) |
| 2026-02-07 | Key finding: Hypothetical contribution stable at 54.7% under constraints (down from 61.3% unconstrained) |
| 2026-02-07 | Added Section 10 to 2.4f: ESC school-level unmet demand analysis with enriched metrics |
| 2026-02-07 | Fixed duplicate school IDs from esc_available multi-row join; aggregated slots per school before merge |
| 2026-02-07 | Added `n_congested_destinations` metric (congested JHS an ESC school could help decongest, traced via shared origins) |
| 2026-02-07 | Added Section 10.1 to 2.4f: Distance-based unmet demand analysis (≤3km, 3-5km, >5km bands) with nearby unmet demand ranking |
| 2026-02-08 | Added `run_simulation_detailed()` variant tracking per-destination and per-origin accepted amounts |
| 2026-02-08 | Added Section 8.1 to 2.4f: Per ESC school CI of accepted student flow (existing/hypothetical/both) |
| 2026-02-08 | Added Section 8.2 to 2.4f: Per congested JHS CI of decongestion using proportional attribution from observed flows |
| 2026-02-08 | Added Section 8.3 to 2.4f: Multi-scenario comparison with std and raw/pct congestion reduction with CI |
| 2026-02-08 | Added Section 12 to .md: Choice-respecting mechanisms leveraging DCM results (8 mechanisms with sources and ESC fit) |
| 2026-02-08 | Updated Section 2 with full NBR Model 4 coefficients table and key insights (from screenshot_nbr_regression_results_table.png) |
