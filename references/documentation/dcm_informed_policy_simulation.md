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

Developed in notebook `2.2-pjm-dcm-v3`, the model predicts expected enrollment for origin-destination pairs based on:

**Features:**
- `log_distance`: Road network distance (negative coefficient — students prefer nearer schools)
- `log_net_cost`: Tuition minus subsidy (negative coefficient — students prefer lower out-of-pocket cost)
- `esc_amount_k`: Subsidy amount (interaction with tuition)
- `origin_region`: Regional fixed effects

**Output:**
- `mu`: Expected number of students for each origin-destination pair under a given subsidy scenario
- Probability mass function (PMF) from negative binomial distribution: P(X = k) for k = 0, 1, 2, ...

### Key Insight

The model captures **behavioral tendencies** observed in actual ESC beneficiary flows. The negative coefficients on distance and net cost confirm that students *tend* to prefer nearer, cheaper options — but outliers exist in the data.

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

## 11. Next Steps

1. ~~Finalize distance threshold based on data exploration~~ (Removed — not needed)
2. ~~Implement Layer 1 (feasibility filter)~~ (Removed — no longer filtering)
3. ~~Create `is_hypothetical` flag by joining with observed flow data~~ ✅
4. ~~Compute proportional weights for origins feeding multiple congested schools~~ ✅
5. ~~Implement path-level marginal analysis for all subsidy scenarios~~ ✅
6. ~~Generate aggregated summaries for policy insights~~ ✅
7. Validate results against observed patterns
8. Identify specific interventions for high-hypothetical-dependency schools
9. Develop recommendations for enabling hypothetical paths

---

## 12. Related Files

- **Choice Model Notebook:** `references/2.2-pjm-dcm-v3`
- **Candidate Pool:** `output/full_candidate_beneficiary_pool.parquet`
- **Observed Flow Data:** `output/grade_7_student_flow_table_sy2324.parquet`
- **Flow to Congested:** `output/analysis_payload/flow_to_congested.parquet`
- **Inspection Notebook:** `notebooks/1.11.-inspect-dcm-results.ipynb`
- **Original Redistribution:** `notebooks/2.4c.-relieve-congestion-probability-distribution.ipynb`
- **Policy Simulation Notebook (deprecated):** `notebooks/2.4d.-dcm-policy-simulation.ipynb`
- **Path-Level Marginal Analysis:** `notebooks/2.4e.-path-level-marginal-analysis.ipynb`
- **Analysis Output:** `output/path_marginal_analysis/`

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
