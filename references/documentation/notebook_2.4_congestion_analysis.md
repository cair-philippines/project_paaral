# Notebook 2.4: Public JHS Congestion Analysis

**File:** `notebooks/2.4.-analyze-public-jhs-congestion.ipynb`
**Last Updated:** 2026-01-28
**Status:** Clean refactored version with Greedy + LP optimization

---

## Purpose

Analyze seat congestion at public junior high schools (JHS) in NCR and Region IV-A, then optimize student redistribution to minimize "aisle learners" (students without seats) through ESC policy levers.

This notebook is the **dedicated analysis notebook** for congestion optimization. Report generation is handled separately in notebook 2.5.

---

## Problem Statement

Public JHS schools in NCR and Region IV-A face overcrowding, measured by:
- **Seat utilization** = JHS enrollment / seat count (>1.0 means overcrowded)
- **Aisle learners** = max(0, enrollment - seats) — students without seats

**Goal:** Minimize total aisle learners by optimizing ESC policy levers.

---

## Data Dependencies

### Input Files

| File | Source | Description |
|------|--------|-------------|
| `output/processed_project_bukas_school_information.parquet` | Module 1 | School metadata (61,442 schools) |
| `output/school_distance_matrix_osrm.npy` | Notebook 2.2 | Distance matrix (8,331 x 8,331 → trimmed to 8,307) |
| `output/school_distance_matrix_index.json` | Notebook 2.2 | School ID mappings |
| `output/grade_7_student_flow_table_sy2324.parquet` | Notebook 2.1 | Student flow data (SY 2023-24) |
| `output/processed_public_seats.parquet` | Module 4 | Public school seat counts |
| `output/processed_project_bukas_school_enrollment.parquet` | Module 1 | Enrollment data (multi-year) |
| `output/all_schools_coordinates_ncr_region4a.parquet` | Notebook 2.2 | School coordinates |
| `output/processed_esc_slots.parquet` | Notebook 1.8 | ESC slots data |
| `output/processed_gastpe_data.parquet` | Module 6 | School tuition fees |

### Output Variables (In-Memory)

| Variable | Description |
|----------|-------------|
| `df_cong` | Congestion metrics table (~1,060 public JHS) |
| `redirections` | Greedy algorithm redirection log |
| `redirections_lp` | LP optimization redirection log |
| `scenario1_congestion` | Greedy before/after comparison |
| `scenario1_lp_congestion` | LP before/after comparison |

### Output Files (Payload for Notebook 2.5)

| File | Description |
|------|-------------|
| `output/analysis_payload/congestion.parquet` | Congestion metrics table |
| `output/analysis_payload/redirections_greedy.parquet` | Greedy redirection log |
| `output/analysis_payload/redirections_lp.parquet` | LP redirection log |
| `output/analysis_payload/redirection_options.parquet` | All redirection options |
| `output/analysis_payload/esc_available.parquet` | ESC schools with slots |
| `output/analysis_payload/flow_to_congested.parquet` | Flows to congested JHS |
| `output/analysis_payload/school_info.parquet` | School metadata |
| `output/analysis_payload/student_flow.parquet` | Complete flow data |
| `output/analysis_payload/blocked_demand_analysis.parquet` | Slot expansion priorities |
| `output/analysis_payload/metadata.json` | Parameters and statistics |

---

## Notebook Structure (~72 cells)

### Section 0.0 - Setup
Standard imports: pandas, numpy, sklearn, json, re, gc.

### Section 1.0 - Load Data

| Section | Description | Key Details |
|---------|-------------|-------------|
| 1.1 | School Information | Loads `sch_info` with essential columns |
| 1.2 | Distance Matrix | Handles duplicate indices, creates `re_schids_to_idx` mapping |
| 1.3 | Student Flow | Grade 7 flow data (SY 2023-24) |
| 1.4 | Public School Seats | Seat counts by education level |
| 1.5 | JHS Enrollment | **Filters:** `school_year == '2023_24'`, `education_level == 'junior_high_school'` |
| 1.6 | School Coordinates | NCR/Region IV-A schools |
| 1.7 | ESC Slots | Private school ESC slot allocations |
| 1.8 | School Fees (GASTPE) | Tuition data for ESC-delivering schools |
| 1.9 | Private School Size | JHS enrollment for private schools, calculates `size_bonus` using percentile rank |

#### Section 1.2 - Distance Matrix Handling

The distance matrix contains 24 integrated schools with duplicate indices (separate ES/JHS coordinates). The notebook:
1. Identifies duplicates where `school_id_to_idx` returns a list of 2 indices
2. Drops the higher index for each duplicate
3. Creates `old_to_new_idx` mapping to account for shifted indices
4. Creates `re_schids_to_idx` mapping (school_id → new index)

### Section 2.0 - Build Congestion Table

| Section | Description | Key Details |
|---------|-------------|-------------|
| 2.1 | Filter Schools | `sector == 'Public'`, `offers_jhs == True`, `old_region in ['NCR', 'Region IV-A']` |
| 2.2 | Join Enrollment | Maps `enrollment_jhs` from aggregated JHS enrollment |
| 2.3 | Join Seat Data | Filters seats to `education_level == 'Junior High School'` |
| 2.4 | Impute Missing Seats | KNN with k=5, features: `[old_region, division, enrollment_jhs]` |
| 2.5 | Calculate Congestion | Computes `seat_utilization_jhs` and `count_aisle_learner_jhs` |

### Section 3.0 - Prepare Optimization Data

| Section | Description | Key Details |
|---------|-------------|-------------|
| 3.1 | Filter ESC Schools | ESC schools present in distance matrix |
| 3.2 | Filter Student Flow | Both origin and destination in `re_schids` |
| 3.3 | Build Distance Lookup | Origin ES → Private ESC pairs within `MAX_DISTANCE_KM` |
| 3.4 | Link Flow to Congested | Join flow with congestion data, filter to congested destinations |
| 3.5 | Build Redirection Options | Merge flow with origin-to-private distance lookup |
| 3.6 | Calculate Available Slots | `available_slots = total_slots - current_beneficiaries` |
| 3.7 | Add Tuition and School Size | Join tuition + size data, calculate `priority_score` with size bonus |

### Section 4.0 - Greedy Optimization

| Section | Description |
|---------|-------------|
| 4.1 | Initialize Tracking Variables |
| 4.2 | Run Greedy Algorithm |
| 4.3 | Calculate Greedy Results |

**Algorithm:**
1. Sort redirection options by: most congested destination first, then lowest `priority_score`
2. For each option, check remaining capacity (slots, aisle learners, non-beneficiaries)
3. Redirect minimum of available capacities
4. Update tracking dictionaries

**Flow Constraint:** Tracks `remaining_non_benef` per (origin, destination) pair to prevent over-redirecting from a single flow.

### Section 5.0 - LP Optimization

| Section | Description |
|---------|-------------|
| 5.1 | Setup LP Problem |
| 5.2 | Define Decision Variables |
| 5.3 | Add Constraints |
| 5.4 | Define Objective Function |
| 5.5 | Solve LP |
| 5.6 | Extract LP Results |

**Constraints:**
1. **Flow Constraint:** `sum(x) <= non_beneficiaries` per (origin, destination)
2. **Slot Constraint:** `sum(x) <= available_slots` per private ESC
3. **Aisle Constraint:** `sum(x) <= aisle_learners` per public JHS

**Objective:** Maximize weighted redirections with distance/cost penalties.

### Section 6.0 - Compare Results
Side-by-side comparison table of Greedy vs LP optimization.

### Section 6.1 - Blocked Demand Analysis

Analyze remaining demand after LP optimization to identify slot expansion priorities.

| Step | Description |
|------|-------------|
| 1 | Calculate slots used by LP for each private ESC |
| 2 | Identify capacity-constrained schools (≥95% utilized) |
| 3 | Calculate remaining demand per flow after LP |
| 4 | Assign each flow to best constrained school (lowest priority_score) |
| 5 | Aggregate unique blocked demand per school |

**Priority Score Formula** (same as main optimization):
```
priority_score = norm_distance + TUITION_WEIGHT × norm_tuition - SIZE_WEIGHT × size_bonus
```
- Lower = better (nearer + cheaper + larger schools prioritized)
- `size_bonus` = percentile rank [0.0, 1.0] based on private JHS enrollment

**Key Output:** `blocked_demand_analysis` DataFrame with columns (15 total):
- School info: `private_esc_id`, `private_esc_name`, `division`, `region`
- Capacity: `total_slots`, `available_slots_before`, `slots_used_lp`, `utilization_rate`
- Demand: `unique_blocked_demand`, `num_assigned_flows`, `num_congested_jhs_affected`
- Cost/fit: `tuition_jhs`, `avg_priority_score`
- School size: `private_jhs_enrollment`, `size_bonus`

**Key Metric:** `unique_blocked_demand` = students exclusively assigned to this school (no double-counting across schools)

### Section 7.0 - Export Payload for Notebook 2.5

Exports all key dataframes and metadata for report generation in notebook 2.5.

**Output Directory:** `output/analysis_payload/`

**Exported Files:**

| File | Description |
|------|-------------|
| `congestion.parquet` | Congestion metrics table (`df_cong`) |
| `redirections_greedy.parquet` | Greedy algorithm redirection log |
| `redirections_lp.parquet` | LP optimization redirection log |
| `redirection_options.parquet` | All redirection options with priority scores |
| `esc_available.parquet` | ESC schools with available slots |
| `flow_to_congested.parquet` | Student flows to congested destinations |
| `school_info.parquet` | School metadata (slim version) |
| `student_flow.parquet` | Complete student flow data |
| `blocked_demand_analysis.parquet` | Slot expansion priorities (blocked demand per school) |
| `metadata.json` | Parameters and summary statistics |

**Metadata Contents:**
- Parameters: `max_distance_km`, `tuition_weight`, `size_weight`, `cost_penalty`, `distance_penalty`, `size_weight_lp`
- Aisle learners: `baseline_aisle_learners`, `greedy_aisle_learners`, `lp_aisle_learners`
- Redirections: `greedy_students_redirected`, `lp_students_redirected`
- Schools: `greedy_schools_used`, `lp_schools_used`
- Blocked demand: `constrained_schools`, `total_blocked_demand`

---

## Key Parameters

| Parameter | Value | Location | Description |
|-----------|-------|----------|-------------|
| `MAX_DISTANCE_KM` | 5 | Section 3.3 | Maximum distance from origin ES to private ESC |
| `TUITION_WEIGHT` | 0.5 | Section 3.7 | Weight for tuition in priority score |
| `SIZE_WEIGHT` | 0.3 | Section 3.7 | Weight for size bonus in priority score; `size_bonus` is percentile rank [0.0, 1.0] of private JHS enrollment |
| `COST_PENALTY` | 0.3 | Section 5.4 | LP penalty for expensive schools |
| `DISTANCE_PENALTY` | 0.3 | Section 5.4 | LP penalty for distant schools |
| `SIZE_WEIGHT_LP` | 0.3 | Section 5.4 | LP weight for size bonus; `size_bonus` is percentile rank [0.0, 1.0] of private JHS enrollment |
| `n_neighbors` | 5 | Section 2.4 | KNN imputation neighbors |

---

## Key Formulas

### Priority Score (Greedy)
```
priority_score = norm_distance + (TUITION_WEIGHT × norm_tuition) - (SIZE_WEIGHT × size_bonus)
```
- Normalized per destination group
- `size_bonus` = percentile rank of school's JHS enrollment among all private schools nationwide
- `size_bonus` ranges from 0.0 (smallest school) to 1.0 (largest school)
- Lower = better (nearer + cheaper + larger schools prioritized)

### LP Objective Coefficient
```
coef = 1.0 - (DISTANCE_PENALTY × norm_distance) - (COST_PENALTY × norm_tuition) + (SIZE_WEIGHT_LP × size_bonus)
```
- `size_bonus` = percentile rank (0.0 to 1.0) based on private JHS enrollment
- Higher coefficient = more weight in objective function
- Larger schools receive proportionally higher bonus (largest gets +0.3, median gets +0.15)

### Congestion Metrics
```
seat_utilization_jhs = enrollment_jhs / seat_count
count_aisle_learner_jhs = max(0, enrollment_jhs - seat_count)
```

### Available Slots
```
available_slots = total_count_slots - current_beneficiaries
```
Where `current_beneficiaries` = sum of `count_esc_beneficiary` going to each ESC school.

---

## Data Filters Summary

| Data | Filter Criteria |
|------|-----------------|
| Schools | `sector == 'Public'`, `offers_jhs == True`, `old_region in ['NCR', 'Region IV-A']` |
| Enrollment | `school_year == '2023_24'`, `education_level == 'junior_high_school'` |
| Seats | `education_level == 'Junior High School'` |
| Flow | Both origin and destination in distance matrix |
| ESC Schools | Present in distance matrix |
| Redirection Options | `available_slots > 0`, distance ≤ 5km |
| School Size | `size_bonus` = percentile rank of private JHS enrollment nationwide (0.0 to 1.0) |

---

## Expected Results

*(Values from latest run with school size feature and 5km distance threshold)*

| Metric | Baseline | After Greedy | After LP |
|--------|----------|--------------|----------|
| Total aisle learners | ~577,700 | ~551,900 | ~551,800 |
| Students redirected | — | ~25,800 | ~25,900 |
| Reduction | — | ~4.5% | ~4.5% |
| Private ESC schools used | — | ~947 | ~947 |
| Avg redirect distance | — | ~2.5 km | ~1.0 km |

---

## Related Files

- **Notebook 2.5:** Report generation (loads payload from this notebook)
- **Payload Directory:** `output/analysis_payload/` (exported by Section 7.0)
- **Notebook 2.2:** Distance matrix generation
- **Documentation:** `stakeholder_presentation_guide.md`, `notebook_2.5_stakeholder_reports.md`

---

## Changelog

| Date | Changes |
|------|---------|
| 2026-01-28 | **Documentation fix:** Corrected output directory path (`analysis_payload` not `scenario1_payload`); Updated metadata key names to match actual export; Added `private_jhs_enrollment` and `size_bonus` to blocked demand column documentation; Updated expected results to match actual output (~577K baseline, ~4.5% reduction); Fixed changelog order (newest first) |
| 2026-01-27 | **Section 6.1 - Fix:** Added `size_bonus` to blocked demand priority score calculation for consistency with main optimization formula |
| 2026-01-27 | **Section 6.1 - Blocked Demand Analysis:** Added analysis of remaining demand after LP; Identifies capacity-constrained schools (≥95% utilized); Calculates `unique_blocked_demand` per school (no double-counting); Exports `blocked_demand_analysis.parquet` for Report 5 |
| 2026-01-27 | **Section 7.0 - Export Payload:** Added export section to save all key dataframes and metadata to `output/analysis_payload/` for use by notebook 2.5 |
| 2026-01-27 | **School Size Feature (Percentile Rank):** Added Section 1.9 with `size_bonus` = percentile rank of private JHS enrollment; Updated Greedy and LP with `SIZE_WEIGHT` × `size_bonus`; Continuous scoring (0.0 to 1.0) without arbitrary binning |
| 2026-01-27 | Fixed Section 1.5 enrollment filter; Fixed Section 1.2 distance matrix index mapping |
| 2026-01-27 | **Major refactoring:** Cleaned notebook from 129 cells to 68 cells; Added flow constraint tracking (`remaining_non_benef`) |
| 2026-01-26 | Added tuition cost integration with priority scoring |
| 2026-01-26 | Initial LP optimization implementation |
