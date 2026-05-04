# PAARAL DepEd Planning View — Project Context

## ⚠️ CURRENT PHASE: MOCKUP

**This is a UI/UX mockup for stakeholder demonstrations, NOT production implementation.**

**What's Real:**
- UI components and interaction patterns
- Visual layout and design principles
- User flow and decision-making workflow

**What's Mocked:**
- Optimization engine (simple heuristics, not actual ILP solver)
- Data (synthetic student/school records, not real DepEd data)
- Calculations (approximate formulas, not research-validated models)

**Purpose:** Demonstrate what the DepEd Planning View would look like and how it would work. Enable stakeholder feedback on UI/UX before building production backend.

**Production Phase (Later):**
- Real optimization solvers (PuLP, OR-Tools, Gurobi)
- BigQuery integration for actual student flow data
- Validated gravity model parameters
- Backend API for heavy computation

---

## Geographic Scope (Mockup)

**This mockup covers two regions only: NCR and Region IV-A (CALABARZON).**

This scope was chosen because NCR is the primary congestion hotspot and Region IV-A is the adjacent overflow region with the most ESC activity. Together they represent the highest-priority policy context for stakeholder demonstration.

**Subsidy tiers within scope** (keyed by `city_type` field on school records):
| `city_type` | Areas | ESC Subsidy |
|---|---|---|
| `ncr` | All schools in NCR | ₱13,000 |
| `huc` | Lucena City only (sole Region IV-A HUC per 2025 PSA list) | ₱11,000 |
| `other` | All other Region IV-A municipalities | ₱9,000 |

Reference: `public/reference/huc_list.png` (2025 PSA List of Highly Urbanized Cities)

**Production phase** will expand to all regions of the Philippines.

---

## Project Overview

PAARAL (Platform for Analyzing Access and Resource Allocation in Learning) is a three-tier computational policy engine for the Philippine Department of Education's Educational Service Contracting (ESC) program. ESC provides geographically-differentiated subsidies for students to attend private schools to reduce public school overcrowding.

**Core Intelligence Claim:** "The law targets households. Our data sees schools. PAARAL closes that gap."

PAARAL operationalizes Senate Bill No. 1981 provisions around congestion thresholds and school isolation by transforming individual learner-level transitions into system-level decision intelligence.

## DepEd Planning View: Current Mockup (Descriptive Dashboard)

The **DepEd Planning View** is being built in two stages:

**Stage 1 (current mockup):** A flat, descriptive dashboard that shows the existing state of congestion across JHS schools in NCR and Region IV-A. No simulation, no flow maps, no optimization. The goal is to confirm that DepEd leadership can read and interpret the congestion picture before layering on prescriptive analytics.

**Stage 2 (next):** A prescriptive analytics module — a policy simulation sandbox that enables DepEd leadership to run constrained optimization simulations on ESC slot allocation, answering the four questions below.

### What the Current Mockup Shows

1. **Summary strip** — Total JHS in scope, schools with congestion (by type), congested public JHS, ESC schools oversubscribed (Rank-1 demand > slots available)
2. **Filter bar** — School type (Public JHS / Private No-ESC / Private ESC); congestion definition (enrollees > seats OR classroom-to-learner ratio > threshold); region → province → city drill-down
3. **Geographic breakdown** — Collapsible region → province → municipality rows with congestion counts
4. **School table** — Sortable by any field; inline ESC columns (ESC Slots, Rank-1 Demand, Overflow) on ESC school rows; paginated

**Congestion is defined two ways** (user-toggleable):
- *By seats:* `enrollment > seats` (seats = num_classrooms × 40)
- *By ratio:* `classroom-to-learner ratio > threshold` (threshold is a slider, 25–60)

**ESC overflow** = Rank-1 ESC applicants (from `students.json`) minus `slots_total` for that school. Positive values mean the school is oversubscribed at first preference.

### The Four Questions (Stage 2 Vision)

**Note:** Each question below describes what the *production system* would do. They are not yet implemented in the mockup.

#### 1. How do we reduce congestion without disrupting thousands of families?

**The Problem:** Simply moving students to less-congested schools ignores their preferences and creates cascading disruption.

**What the View Does:**
- Finds the assignment that achieves target decongestion while respecting ranked choice preferences
- Optimization objective: `min Σ(rank_perturbation) - α·Σ(decongestion)`
- Shows which students must move and which can stay at their Rank 1 choice
- Quantifies tradeoff: "Moving 200 students to Rank 2 achieves 35:1 threshold with minimal disruption"

**Output Metrics:**
- Students affected (count + %)
- Rank distribution (% at Rank 1/2/3)
- Congestion relief (classrooms saved)
- Average distance increase

#### 2. What happens if we change the subsidy amount?

**The Problem:** ESC subsidies have been fixed at ₱13k (NCR), ₱11k (urban), ₱9k (other) since 2017. What if we adjust them?

**What the View Does:**
- Counterfactual simulation: increase NCR subsidy from ₱13k to ₱15k
- Model predicts enrollment shifts based on distance friction (4× stronger than cost per gravity model)
- Shows slot utilization changes and residual congestion
- Calculates budget impact: "₱15k NCR subsidy → 47% more ESC enrollment → ₱2.3M additional cost"

**Output Metrics:**
- Enrollment shift by region
- Slot utilization change (before/after %)
- Residual congestion (schools still >40:1)
- Total budget requirement

#### 3. Where should we allocate new ESC slots?

**The Problem:** DepEd gets 5,000 new ESC slots. Which schools should receive them for maximum decongestion impact?

**What the View Does:**
- Optimization identifies school pairs with largest marginal decongestion potential
- Prioritizes school pairs based on: (a) origin congestion severity, (b) destination capacity, (c) distance friction
- Shows "bang for buck": slots allocated → congestion reduction per slot
- Avoids spray-and-pray: concentrates slots where they'll actually get used

**Output Metrics:**
- Recommended slot allocation by school
- Marginal decongestion per slot (students diverted per slot allocated)
- Expected utilization rate (% of slots likely to be filled)
- Geographic equity distribution

#### 4. If we shift students to their 2nd or 3rd choice, how many families are affected?

**The Problem:** Hitting aggressive congestion targets (e.g., 30:1) may require moving students to lower-ranked choices. What's the impact?

**What the View Does:**
- Rank calibration simulation: allows X% of students to be assigned to Rank 2 or Rank 3
- Shows minimum perturbation needed to hit threshold: "35:1 requires 8% at Rank 2, 30:1 requires 22% at Rank 2"
- Transparent tradeoff visualization: "Moving 200 students to Rank 2 avoids 500 students in 50:1 classrooms"
- Identifies which students are most constrained (few good options) vs. flexible (many good options)

**Output Metrics:**
- Students moved by rank (Rank 1→2, Rank 1→3)
- Congestion distribution histogram (before/after)
- Distance impact (avg km increase for moved students)
- Geographic clustering (which regions most affected)

## Optimization Framework

**Note:** This section describes the *target* optimization model for production. The mockup uses simplified heuristics to simulate results for UI demonstration.

### Objective Function (Production Target)

```
minimize: Σ(rank_perturbation_i) - α·Σ(decongestion_j)

where:
  rank_perturbation_i = 0 if student i at Rank 1, 1 if at Rank 2, 2 if at Rank 3
  decongestion_j = reduction in students at overcrowded school j
  α = congestion penalty weight (user-adjustable)
```

### Mockup Implementation

The mockup uses **simple heuristics** to approximate optimization results:

```javascript
// Mock calculation (not actual optimization)
const studentsAffected = (40 - threshold) * 50 + rankTolerance * 10;
const congestionRelief = studentsAffected / 40;  // 40 students per classroom
const budgetUtilization = Math.min(100, (studentsAffected / 5000) * 100);
```

This creates realistic-looking results for UI testing. Production will replace this with actual ILP solvers.

### Hard Constraints

1. **ESC slot capacity:** `Σ(assignments_to_school_j) ≤ slots_available_j`
2. **Physical capacity:** `enrollment_j ≤ capacity_j`
3. **Feasibility:** Every student assigned exactly once
4. **Public fallback:** Every student has ≥1 public JHS in their ranked list

### Soft Constraints (Preferences)

1. Respect student rank preferences when feasible
2. Minimize geographic disruption (distance friction coefficient: 4×)
3. Maximize slot utilization (avoid unused ESC slots)

## Key Data Assets

### School Data (Real identities, synthetic numerics)

**Source:** `github.com/cair-philippines/project-school-coordinates`

Real fields: `school_id` (DepEd LIS / BEIS), `name`, `latitude`, `longitude`, `region`, `province`, `municipality` (as `city`), `barangay`

| Type | Count | Notes |
|---|---|---|
| `public_es` | 3,160 | Origin schools for student preference generation |
| `public_jhs` | 1,086 | Primary congestion targets |
| `private_jhs` | 561 | Private, no ESC participation |
| `private_jhs_esc` | 952 | ESC partner schools; have `slots_total`, `slots_available` |

Synthetic fields (generated by `scripts/generate_data.js`):
- `num_classrooms`, `seats` (= num_classrooms × 40), `enrollment`, `congestion_ratio`
- `tuition_annual` (private schools only)
- `slots_total`, `slots_available` (ESC schools only; `slots_total ≤ seats` enforced)

### Student Preference Data (Fully synthetic)

- 2,000 synthetic students from 300 sampled public ES origins
- Each student: origin ES → Rank 1/2/3 ESC school preferences (scored by road distance + 5km cross-region penalty + noise)
- Road distance via circuity factor: NCR ×1.3, near-NCR IVA ×1.4, provincial IVA ×1.6

**ESC Subsidy Structure (Fixed since 2017):**
- NCR: ₱13,000
- Highly urbanized cities (Region IV-A: Lucena City only per 2025 PSA list): ₱11,000
- Other areas (all other Region IV-A municipalities): ₱9,000

### Output Data

**Optimized Assignment Matrix:**
- Student × School assignments
- Rank distribution (actual assignment vs. preference)
- Distance traveled
- Cost (tuition - subsidy)

**System Metrics:**
- Congestion relief (classrooms saved)
- Budget utilization (% of ESC slots used)
- Preference respect (% at Rank 1)
- Geographic equity (decongestion by region)

## Mockup Assumptions & Defaults

All synthetic numeric values are generated by `scripts/generate_data.js`. They are deliberate approximations — not arbitrary — and should be the first things revisited when moving to production data.

### Synthetic Enrollment Ranges

| Type | Enrollment | Rationale |
|---|---|---|
| Public JHS | ratio = rand(40–55 NCR, 30–50 IVA) → enrollment = ratio × num_classrooms | NCR consistently overcrowded; IVA moderate |
| Private (No ESC) | rand(65–110%) of seats | Mostly below capacity; ~23% above |
| Private (ESC) | base = rand(65–100%) of seats + ESC slots used | Base can be high; ~13% congested overall |

### Synthetic Slot Ranges (ESC schools)

| Field | Range | Constraint |
|---|---|---|
| `slots_total` | min(rand(20–100), seats) | `slots_total ≤ seats` always enforced |
| `slots_available` | rand(20–65%) of slots_total | Some slots filled, some still open |

### Synthetic Data Parameters (`scripts/generate_data.js`)

**School counts:**
| Type | Total | NCR | IVA | Rationale |
|---|---|---|---|---|
| `public_es` | 20 | 12 | 8 | Origin schools; NCR-heavy reflects urban density |
| `public_jhs` | 30 | 18 | 12 | Congestion sources; NCR more overcrowded |
| `private_jhs` | 15 | 9 | 6 | No-ESC private schools; present but not primary ESC target |
| `private_jhs_esc` | 35 | 21 | 14 | ESC destinations; more abundant in NCR |

**Congestion ranges (students per classroom):**
| Region | Min | Max | Rationale |
|---|---|---|---|
| NCR | 40 | 55 | Severe overcrowding; NCR public JHS consistently above standard |
| Region IV-A | 30 | 50 | Moderate; some schools near threshold, others below |

**Tuition ranges (annual, PHP):**
| `city_type` | Min | Max | Rationale |
|---|---|---|---|
| `ncr` | ₱50,000 | ₱150,000 | Metro Manila private school market rates |
| `huc` | ₱35,000 | ₱100,000 | Lucena City mid-tier private schools |
| `other` | ₱20,000 | ₱60,000 | Provincial private schools; lower cost base |

**Road distance circuity factors:**
| Area | Factor | Rationale |
|---|---|---|
| NCR | ×1.3 | Dense urban grid; many indirect routes but short distances |
| Cavite / Laguna / Rizal | ×1.4 | Partial expressway access (CAVITEX, SLEX, C6) but congested |
| Batangas / Quezon | ×1.6 | More indirect provincial roads; fewer expressway options |

**Standard classroom size:** 40 students — standard DepEd classroom capacity used for congestion ratio computation (`congestion_ratio = enrollment / num_classrooms`).

**Student preference model:** Road distance + 5 km cross-region penalty, with ±1.5 km random noise. Reflects gravity model finding that distance friction is the dominant factor in school choice.

## Research Foundation

From stochastic gravity model (KDD '26 paper, 29k school-pair flows):

1. **Distance friction dominates:** Geographic proximity constrains choice 4× more than tuition costs
2. **Capacity-bound system:** Slot capacity (not subsidy price) is the binding constraint
3. **Unobserved pairs:** Majority of marginal decongestion potential flows through school pairs not historically observed

**Implication for DepEd Planning View:**
- Distance threshold is critical in slot allocation (private schools some distance threshold rarely absorb students)
- Increasing subsidy yields marginal gains; increasing slot allocation yields massive gains
- Optimization must consider school pairs never historically connected

## Analytical Scope

**What This View Does:**
- Prescriptive optimization (what should happen under optimal allocation)
- Counterfactual simulation (what if we change X?)
- Scenario comparison (Policy A vs. Policy B vs. Baseline)

**What This View Does NOT Do:**
- Causal identification (cannot isolate subsidy treatment effect with fixed amounts)
- Predictive forecasting (not estimating future behavior without intervention)
- Dictate implementation (suggests allocations, DepEd coordinates with schools)

**Methodological Constraint:**
- This is a prediction/simulation framework for policy counterfactuals, not causal inference
- "Structural" refers to the model's capability for policy simulation, not causal identifications

## Success Criteria

**For the Current Descriptive Dashboard (Stage 1):**

1. **Visual clarity:** Non-technical DepEd leadership can read the congestion table and geographic breakdown without training
2. **Data trust:** Real school names and locations make the picture credible
3. **Correct filtering:** Toggling between school types and geographies gives expected results
4. **ESC legibility:** Oversubscribed ESC schools are immediately visible; overflow count is interpretable
5. **Stakeholder confirmation:** DepEd leadership agrees this view reflects the actual congestion picture they recognize

**For Stage 2 (Prescriptive Analytics, Next):**

1. **Answer "What if Rank-1 preferences are honored?"** — show which public JHS schools decongestion when ESC applicants get their first choice, subject to slot availability
2. **Scenario toggle** — Baseline vs. "Rank-1 honored" side-by-side
3. **Production eventually:** Real optimization solvers on actual student data; validated gravity model; BigQuery integration

---

**Remember:**

**Stage 1 (current):** A descriptive dashboard showing the existing congestion state. Real school names and locations; synthetic enrollment and slot numbers. No simulation engine.

**Stage 2 (next):** Add a Rank-1 feasibility simulation — honor ESC first preferences subject to slot caps, show which public schools decongestion as a result.

**What we're building:** UI components, interaction patterns, visual layouts, user flows.

**What we're NOT building yet:** Optimization solvers, production data pipelines, validated statistical models.

**Goal:** Confirm that DepEd leadership can read and trust the descriptive view, then layer on the prescriptive simulation.

---

## Additional resources
- For specific stack and skills to be used, see @SKILL.md
- For mockup specifications, see @SPEC.md