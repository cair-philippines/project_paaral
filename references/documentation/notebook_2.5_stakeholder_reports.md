# Notebook 2.5: Scenario 1 Stakeholder Reports

**File:** `notebooks/2.5.-scenario1-stakeholder-reports.ipynb`
**Last Updated:** 2026-01-28
**Status:** Complete (Refactored to payload-based architecture)

---

## Purpose

Generate CSV reports to communicate Scenario 1 optimization results to stakeholders. The reports help answer:
1. Which congested public schools need the most attention?
2. Which private ESC schools are most helpful to each congested public school?
3. Which schools were excluded from the analysis and why?
4. How do student flows change after optimization?
5. If we have additional funding, which private schools should receive more ESC slots?

---

## Architecture

**Payload-Based Approach:** This notebook loads pre-computed data from notebook 2.4 instead of duplicating the entire optimization workflow. This ensures:
- Single source of truth (optimization runs once in 2.4)
- Faster report generation (no re-running optimization)
- Consistency between analysis and reports

---

## Data Dependencies

### Input Files (Payload from Notebook 2.4)

| File | Description |
|------|-------------|
| `output/scenario1_payload/congestion.parquet` | Congestion metrics table (~1,060 public JHS) |
| `output/scenario1_payload/redirections_greedy.parquet` | Greedy algorithm redirection log |
| `output/scenario1_payload/redirections_lp.parquet` | LP optimization redirection log |
| `output/scenario1_payload/redirection_options.parquet` | All possible redirection options with priority scores |
| `output/scenario1_payload/esc_available.parquet` | ESC schools with available slots |
| `output/scenario1_payload/flow_to_congested.parquet` | Student flows to congested public JHS |
| `output/scenario1_payload/school_info.parquet` | School metadata (slim version) |
| `output/scenario1_payload/student_flow.parquet` | Complete student flow data |
| `output/scenario1_payload/blocked_demand_analysis.parquet` | Pre-computed blocked demand analysis for Report 5 |
| `output/scenario1_payload/metadata.json` | Parameters and statistics from optimization |
| `output/all_schools_coordinates_ncr_region4a.parquet` | School coordinates for JHS-to-private distance calculation |

### Output Files

| File | Description |
|------|-------------|
| `output/reports/scenario1_public_jhs_congestion_summary.csv` | Report 1: Public JHS congestion with top 3 partners (origin-to-private distance) |
| `output/reports/scenario1_public_jhs_congestion_summary_jhs_dist.csv` | Report 1B: Public JHS congestion with top 3 partners (JHS-to-private distance) |
| `output/reports/scenario1_public_private_pairs.csv` | Report 2: All public-private redirection pairs |
| `output/reports/scenario1_excluded_public_schools.csv` | Report 3A: Excluded public schools |
| `output/reports/scenario1_excluded_private_schools.csv` | Report 3B: Excluded private ESC schools |
| `output/reports/scenario1_student_flow_changes_greedy.csv` | Report 4A: Student flow changes (Greedy) |
| `output/reports/scenario1_student_flow_changes_lp.csv` | Report 4B: Student flow changes (LP) |
| `output/reports/scenario1_slot_expansion_priorities.csv` | Report 5: Slot expansion priorities |

---

## Workflow Sections

### Section 0.0 - Setup
Standard imports and directory configuration. Creates `output/reports/` directory.

### Section 1.0 - Load Payload from Notebook 2.4
Loads all pre-computed data from the payload directory:
- `df_cong` - Congestion metrics table
- `redirections_greedy` - Greedy algorithm results
- `redirections_lp` - LP optimization results
- `redirection_options` - All options with priority scores
- `esc_available` - ESC schools with slot availability
- `flow_to_congested` - Flows to congested destinations
- `sch_info` - School metadata
- `flow` - Complete student flow data
- `metadata` - Parameters (MAX_DISTANCE_KM, TUITION_WEIGHT, SIZE_WEIGHT, etc.)
- `coordinates` - School coordinates for JHS-to-private distance calculation (Report 1B)

### Section 2.0 - Generate Reports

#### Section 2.1 - Report 1: Public JHS Congestion Summary
**Purpose:** One row per public JHS showing baseline congestion, intervention results, and top 3 private partners.

**Columns (28 total):**

| Category | Columns |
|----------|---------|
| School Info | `school_id`, `school_name`, `division`, `region` |
| Capacity | `enrollment_jhs`, `seat_count`, `aisle_learners_baseline` |
| Greedy Results | `redirected_greedy`, `aisle_learners_greedy`, `pct_reduction_greedy` |
| LP Results | `redirected_lp`, `aisle_learners_lp`, `pct_reduction_lp`, `is_fully_relieved` |
| Top Partner 1 | `top_partner_1_id`, `top_partner_1_name`, `top_partner_1_redirected`, `top_partner_1_distance_km`, `top_partner_1_tuition` |
| Top Partner 2 | `top_partner_2_id`, `top_partner_2_name`, `top_partner_2_redirected`, `top_partner_2_distance_km`, `top_partner_2_tuition` |
| Top Partner 3 | `top_partner_3_id`, `top_partner_3_name`, `top_partner_3_redirected`, `top_partner_3_distance_km`, `top_partner_3_tuition` |

**Notes:**
- Top 3 partners derived from **Greedy algorithm** (prioritizes nearest + cheapest schools via `priority_score`)
- Distance is **average road distance from origin ES schools to private ESC** (in km)
- Tuition is JHS tuition fee at the private ESC school (in PHP)
- LP results included for comparison
- Schools with fewer than 3 partners have NaN in unused columns
- Sorted by `aisle_learners_baseline` descending (most congested first)

---

#### Section 2.1B - Report 1B: Public JHS Congestion Summary (JHS-to-Private Distance)
**Purpose:** Same as Report 1, but with distance measured from the **congested public JHS to the private ESC partner** (instead of origin ES to private ESC).

**Key Difference from Report 1:**

| Report | Distance Column | Distance Meaning |
|--------|-----------------|------------------|
| Report 1 | `top_partner_X_distance_km` | Origin ES → Private ESC (averaged across all origins) |
| Report 1B | `top_partner_X_dist_from_jhs_km` | Public JHS → Private ESC (straight-line haversine) |

**Columns (29 total):**

| Category | Columns |
|----------|---------|
| School Info | `school_id`, `school_name`, `division`, `region` |
| Capacity | `enrollment_jhs`, `seat_count`, `aisle_learners_baseline` |
| Greedy Results | `redirected_greedy`, `aisle_learners_greedy`, `pct_reduction_greedy` |
| LP Results | `redirected_lp`, `aisle_learners_lp`, `pct_reduction_lp`, `is_fully_relieved` |
| Top Partner 1 | `top_partner_1_id`, `top_partner_1_name`, `top_partner_1_redirected`, `top_partner_1_dist_from_jhs_km`, `top_partner_1_tuition` |
| Top Partner 2 | `top_partner_2_id`, `top_partner_2_name`, `top_partner_2_redirected`, `top_partner_2_dist_from_jhs_km`, `top_partner_2_tuition` |
| Top Partner 3 | `top_partner_3_id`, `top_partner_3_name`, `top_partner_3_redirected`, `top_partner_3_dist_from_jhs_km`, `top_partner_3_tuition` |

**Notes:**
- Uses **haversine (straight-line) distance** from congested public JHS to private ESC
- Useful for understanding geographic proximity between public and private partners
- Same top 3 partners as Report 1 (ranking based on students redirected, not distance)
- Sorted by `aisle_learners_baseline` descending (most congested first)

**When to Use Report 1 vs 1B:**
- **Report 1**: Shows how far students travel from their origin ES to the private school
- **Report 1B**: Shows how close the private partner is to the congested public JHS being relieved

---

#### Section 2.2 - Report 2: Public-Private Redirection Pairs
**Purpose:** Detailed view of all (public JHS, private ESC) partnerships from LP optimization.

**Columns (13 total):**

| Column | Description |
|--------|-------------|
| `public_jhs_id` | Congested public JHS school ID |
| `public_jhs_name` | School name |
| `public_jhs_division` | Division |
| `public_jhs_aisle_learners` | Baseline aisle learners |
| `private_esc_id` | Partner private ESC school ID |
| `private_esc_name` | School name |
| `private_esc_division` | Division |
| `total_count_slots` | Total ESC slots at private school |
| `available_slots_before` | Available slots before optimization |
| `count_redirected` | Students redirected to this private school |
| `avg_distance_km` | Average distance from origin schools |
| `pct_of_public_relief` | % of public school's relief from this private |
| `num_origin_schools` | Number of origin ES feeding this pair |

**Notes:**
- Uses LP optimization results
- Sorted by `count_redirected` descending
- One row per unique (public, private) pair

---

#### Section 2.3 - Report 3A: Excluded Public Schools
**Purpose:** Document which public JHS were NOT included in optimization.

**Columns (5 total):**
- `school_id`, `school_name`, `division`, `region`, `exclusion_reason`

**Exclusion Reasons:**
| Reason | Description |
|--------|-------------|
| Not congested (no aisle learners) | School has seats >= enrollment |
| No JHS enrollment data | Missing from enrollment dataset |
| Not in distance matrix (missing coordinates) | No valid coordinates for routing |

---

#### Section 2.4 - Report 3B: Excluded Private ESC Schools
**Purpose:** Document which private ESC schools were NOT used in optimization.

**Columns (8 total):**
- `school_id`, `school_name`, `division`, `region`
- `total_slots`, `current_beneficiaries`, `available_slots`
- `exclusion_reason`

**Exclusion Reasons:**
| Reason | Description |
|--------|-------------|
| Outside NCR/Region IV-A | School not in target regions |
| Not in distance matrix (missing coordinates) | No valid coordinates for routing |
| No available slots (fully occupied) | Current beneficiaries >= total slots |
| No nearby origin schools within 15km | No origin ES within distance threshold |
| Not selected by LP (slots used elsewhere) | LP chose other schools |

---

#### Section 2.5 - Report 4A: Student Flow Changes (Greedy)
**Purpose:** Show how student flows from origin ES to destination public JHS change after Greedy optimization.

**Columns (17 total):**

| Category | Columns |
|----------|---------|
| Origin Info | `school_id_origin`, `school_name_origin`, `division_origin` |
| Destination Info | `school_id_destination`, `school_name_destination`, `division_destination` |
| Original Flow | `original_total`, `original_beneficiaries`, `original_non_beneficiaries` |
| After Optimization | `students_redirected`, `remaining_non_beneficiaries`, `pct_redirected` |
| Redirect Details | `num_private_partners`, `top_private_esc_id`, `top_private_esc_name`, `top_private_count`, `distance_origin_to_top_private_km` |

**Notes:**
- One row per (origin ES, destination public JHS) pair
- Includes flows with zero redirections (shows untouched flows)
- `distance_origin_to_top_private_km` is distance from origin ES to the top private ESC
- Sorted by `students_redirected` descending

---

#### Section 2.6 - Report 4B: Student Flow Changes (LP)
**Purpose:** Show how student flows change after LP optimization.

**Columns:** Same structure as Report 4A (17 columns) but with LP optimization results.

---

#### Section 2.7 - Report 5: Slot Expansion Priorities
**Purpose:** Identify which private ESC schools should receive additional slots to maximize congestion reduction.

**Logic:** For each flow with remaining demand after LP optimization, assign it to the "best" capacity-constrained private school using priority score. Each flow is assigned to exactly ONE school to avoid double-counting students.

**Assignment Rule:**
```
priority_score = norm_distance + TUITION_WEIGHT × norm_tuition
Assign flow → private school with LOWEST priority_score (nearest + cheapest)
```

**Columns (13 total):**

| Category | Columns |
|----------|---------|
| School Info | `private_esc_id`, `private_esc_name`, `division`, `region` |
| Current Capacity | `total_slots`, `available_slots_before`, `slots_used_lp`, `utilization_rate` |
| Expansion Potential | `unique_blocked_demand`, `num_assigned_flows`, `num_congested_jhs_affected` |
| Cost/Distance | `tuition_jhs`, `avg_priority_score` |

**Key Metric:** `unique_blocked_demand` — students assigned exclusively to this school (no double-counting across schools)

**Calculation:**
1. Identify flows with remaining demand after LP optimization
2. For each flow, find all capacity-constrained private schools (≥95% utilized) within 5km of origin
3. Calculate priority score for each option (distance + tuition)
4. Assign the flow to the school with the LOWEST priority score
5. Aggregate unique_blocked_demand per school

**Notes:**
- Each student is counted exactly once (assigned to best school option)
- Total unique_blocked_demand across all schools = actual remaining students
- `avg_priority_score` shows how "natural" a fit this school is (lower = better)
- Sorted by `unique_blocked_demand` descending, then `avg_priority_score` ascending

**Interpreting the Output:**

| Metric | Example Value | Interpretation |
|--------|---------------|----------------|
| Flows with remaining demand | 18,666 | Total flows that still have students to redirect after LP |
| Total remaining students | 265,526 | All students who could potentially be redirected (much larger than LP's ~26,500 because LP was slot-constrained) |
| Flows assigned to constrained schools | 14,924 | Flows that have at least one capacity-constrained school within 5km |
| Total unique blocked demand | 231,118 | Students assigned to constrained schools (no double-counting) |
| Schools in report | 779 | Constrained schools that are the "best" option for at least one flow |

**Understanding the Gap:**
- **Unassigned flows:** 18,666 - 14,924 = 3,742 flows (34,408 students) could NOT be assigned to any constrained school
- These represent **underserved areas** where either:
  - No capacity-constrained private school exists within 5km, OR
  - Nearby private schools still have unused capacity (non-constrained)
- This gap is useful for identifying areas needing NEW ESC partnerships, not just slot expansion

**Why Some Schools Have Low `slots_used_lp` but High `unique_blocked_demand`:**
- A school may have had very few available slots (e.g., 2 slots) and hit 95% utilization quickly
- But because it's **cheap and well-located**, it's the preferred option for many nearby flows
- These schools are **severely under-allocated relative to demand** — high-priority targets for slot expansion

---

### Section 3.0 - Export Reports
Exports all 4 CSV files to `output/reports/` directory.

### Section 4.0 - Quick Stats for Stakeholder Presentation
Prints key statistics summary:
- Baseline: Total schools, congested schools, total aisle learners
- Intervention: Students redirected, remaining aisle learners, reduction %
- Private partners: Schools used, avg students per school
- Constraints: Max distance, slot utilization

---

## Report Usage Guide

### For Policy Makers
**Use Report 1** to:
- Identify most congested public JHS (sort by `aisle_learners_baseline`)
- See potential reduction with ESC intervention (`pct_reduction_greedy`)
- Find recommended private partners (`top_partner_1/2/3_name`)
- Understand student travel distances (origin ES to private ESC)

**Use Report 1B** to:
- See how close private partners are to the congested public JHS
- Identify partnerships where the private school is geographically near the public school
- Useful for stakeholder discussions about "nearby" private schools serving a public JHS

### For Program Managers
**Use Report 2** to:
- Plan partnerships with specific private schools
- Estimate slot requirements per private school
- Identify geographic clusters (by division)

### For Data Quality
**Use Reports 3A/3B** to:
- Identify data gaps (missing coordinates, enrollment)
- Understand analysis limitations
- Plan data collection efforts

### For Flow Analysis
**Use Reports 4A/4B** to:
- See before/after comparison of student flows
- Identify which origin-destination pairs were most affected
- Understand which private schools absorbed students from each origin
- Compare Greedy vs LP flow redistribution patterns

### For Budget Planning
**Use Report 5** to:
- Identify which private schools should receive additional ESC slots
- Prioritize investment by impact potential (`unique_blocked_demand`)
- Consider cost-effectiveness (`tuition_jhs`) and fit (`avg_priority_score`)
- Answer: "If we have additional funding, where should we add slots?"
- Note: Total `unique_blocked_demand` across all schools = actual remaining students (no double-counting)
- Schools with low `slots_used_lp` but high `unique_blocked_demand` are severely under-allocated — high priority targets
- The gap between "flows with demand" and "flows assigned" reveals underserved areas needing NEW ESC partnerships

---

## Technical Notes

### Why Greedy for Top 3 Partners?
- Greedy uses **priority score** = normalized_distance + (TUITION_WEIGHT × normalized_tuition)
- Balances geographic proximity AND tuition cost
- More practical recommendations for stakeholders (nearby + affordable)
- LP optimizes globally but may spread students across more distant schools

### Tuition Cost Integration
- Tuition data from `processed_gastpe_data.parquet` (GASTPE program)
- `TUITION_WEIGHT = 0.5` — balances distance vs tuition preference
- Missing tuition values imputed with median
- High-cost schools deprioritized (appear lower in partner ranking)

### Distance Types in Reports
- **Report 1** (`top_partner_X_distance_km`): Road network distance from **origin ES to private ESC**, averaged across all origins redirecting to that partner. This represents actual student travel distance.
- **Report 1B** (`top_partner_X_dist_from_jhs_km`): Haversine (straight-line) distance from **congested public JHS to private ESC**. This shows geographic proximity between the partnered schools.

### Distance Threshold
- `MAX_DISTANCE_KM = 5` for this analysis
- Only private schools within 5km of origin ES are considered
- This threshold was set based on stakeholder feedback for practical commute distances

### Slot Availability
- `available_slots = total_count_slots - current_beneficiaries`
- Schools with 0 available slots are excluded

### Flow Constraint in Greedy Algorithm
- Greedy tracks `remaining_non_benef` per (origin, destination) flow
- Prevents redirecting more students than exist in each flow
- When multiple private ESC options serve the same flow, remaining count decrements
- Ensures `remaining_non_beneficiaries` in Report 4A is never negative

---

## Sample Output Statistics

*(Values will vary based on actual data)*

| Metric | Value |
|--------|-------|
| Public JHS analyzed | ~1,060 |
| Congested public JHS | ~780 |
| Total aisle learners (baseline) | ~689,000 |
| Students redirected (Greedy) | ~27,000 |
| Reduction % | ~3.9% |
| Private ESC schools used | ~950 |
| Public-private pairs | ~5,000 |
| Excluded public schools | ~318 |
| Excluded private schools | ~3,600 |

---

## Related Files

- **Notebook 2.4:** Congestion analysis and optimization (source of payload data)
- **Payload Directory:** `output/scenario1_payload/` (9 parquet files + metadata.json)
- **Notebook 2.2:** Distance matrix generation
- **Notebook 1.8:** ESC slots data processing
- **Documentation:** `notebook_2.4_congestion_analysis.md`, `stakeholder_presentation_guide.md`

---

## Changelog

| Date | Changes |
|------|---------|
| 2026-01-28 | **Report 1B - JHS-to-Private Distance:** Added Report 1B showing distance from congested public JHS to private ESC partners (instead of origin ES to private); Uses haversine straight-line distance; Loads coordinates from `all_schools_coordinates_ncr_region4a.parquet`; Helps understand geographic proximity between public and private partners |
| 2026-01-27 | **Report 5 - Load from Payload:** Updated Report 5 to load pre-computed `blocked_demand_analysis.parquet` from payload instead of recalculating; Fixed column name mismatches (`private_esc_id` instead of `school_id_esc`, `distance_to_private_km` instead of `distance_km`); Updated payload directory path to `output/scenario1_payload/` |
| 2026-01-27 | **Major Refactoring - Payload-Based Architecture:** Completely rewrote notebook to load pre-computed data from `output/scenario1_payload/` instead of duplicating optimization workflow from notebook 2.4; Reduced from ~130 cells to ~20 cells; Added metadata.json loading for parameters; Ensures single source of truth for optimization results |
| 2026-01-27 | **Report 5 - Unique Blocked Demand:** Replaced blocked_demand calculation with unique assignment approach; Each flow assigned to exactly ONE best private school option using priority_score; Eliminates double-counting; Added `avg_priority_score` column |
| 2026-01-27 | **Bug Fix - Flow Over-Redirection:** Added `remaining_non_benef` tracking to Greedy algorithm; Fixed issue where `remaining_non_beneficiaries` in Report 4A could be negative; Added Reports 4A/4B documentation |
| 2026-01-26 | **Tuition Integration:** Added GASTPE tuition data loading; Updated Greedy to use `priority_score` (distance + tuition); Added `top_partner_X_tuition` columns to Report 1 (now 28 columns) |
| 2026-01-26 | Initial creation; Report 1 with top 3 partners (Greedy), Reports 2-3B; Complete documentation |
