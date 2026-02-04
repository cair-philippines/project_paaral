# Notebook 2.4c: Public JHS Congestion Analysis (Extended)

**File:** `notebooks/2.4c.-relieve-congestion-probability-distribution.ipynb`
**Last Updated:** 2026-02-04
**Status:** Extended version with Region III and official ESC slots data

---

## Purpose

Extended version of notebook 2.4 that expands geographic coverage to include **Region III** alongside NCR and Region IV-A, and incorporates updated ESC data sources including official unutilized slots from GASS and ESC certification ratings.

This notebook analyzes seat congestion at public junior high schools (JHS) and optimizes student redistribution to minimize "aisle learners" through ESC policy levers.

---

## Key Differences from Notebook 2.4

| Aspect | Notebook 2.4 | Notebook 2.4c |
|--------|--------------|---------------|
| Geographic scope | NCR + Region IV-A | NCR + Region IV-A + **Region III** |
| Distance matrix | `school_distance_matrix_osrm.npy` (8,307 schools) | Expanded matrix (~13,106 schools) |
| Coordinates file | `all_schools_coordinates_ncr_region4a.parquet` | `all_schools_coordinates_ncr_region4a_region3.parquet` |
| Available slots source | Calculated: `total_slots - current_beneficiaries` | **Official data:** `slots_unutilized` from GASS |
| ESC certification | Not included | Loaded from `processed_esc_certification_rating.parquet` |

---

## Problem Statement

Public JHS schools in NCR, Region IV-A, and Region III face overcrowding, measured by:
- **Seat utilization** = JHS enrollment / seat count (>1.0 means overcrowded)
- **Aisle learners** = max(0, enrollment - seats) — students without seats

**Goal:** Minimize total aisle learners by optimizing ESC policy levers.

---

## Data Dependencies

### Input Files

| File | Source | Description |
|------|--------|-------------|
| `output/processed_project_bukas_school_information.parquet` | Module 1 | School metadata (61,442 schools) |
| `output/processed_school_distance_matrix_osrm.npy` | Notebook 2.2 | Distance matrix (~13,106 x 13,106) |
| `output/processed_school_distance_matrix_index.json` | Notebook 2.2 | School ID mappings |
| `output/grade_7_student_flow_table_sy2324.parquet` | Notebook 2.1 | Student flow data (SY 2023-24) |
| `output/processed_public_seats.parquet` | Module 4 | Public school seat counts |
| `output/processed_project_bukas_school_enrollment.parquet` | Module 1 | Enrollment data (multi-year) |
| `output/all_schools_coordinates_ncr_region4a_region3.parquet` | Notebook 2.2 | School coordinates (includes Region III) |
| `output/processed_esc_slots.parquet` | Notebook 1.8 | ESC slots with official `slots_unutilized` |
| `output/processed_gastpe_data.parquet` | Module 6 | School tuition fees |
| `output/processed_esc_certification_rating.parquet` | Notebook 1.10 | ESC certification ratings |

### New Data Sources

#### ESC Slots (Notebook 1.8, Section 3)

**Source:** `SY 23-24 ESC Slots as of 020426.xlsx` from GASS (Government Assistance to Students and Teachers in Private Education)

| Column | Description |
|--------|-------------|
| `school_id` | DepEd school ID |
| `esc_school_id` | ESC school ID |
| `school_name` | School name |
| `slots_total` | Total ESC slots allocated |
| `slots_unutilized` | **Official unutilized slots** (slots_total - slots_billed) |
| `has_deped_school_id` | Whether school has matching DepEd ID |

**Key Change:** The `slots_unutilized` column contains official data from GASS, eliminating the need to calculate available slots from flow data.

#### ESC Certification Ratings (Notebook 1.10)

**Source:** `GASTPE Yearend Report SY 24-25 (Certification Annexes).pdf`

| Column | Description |
|--------|-------------|
| `school_id` | DepEd school ID (if matched) |
| `esc_school_id` | ESC school ID |
| `annex` | Source annex (D, E, F, or G) |
| `annex_label` | Description of certification category |
| `has_deped_school_id` | Whether school has matching DepEd ID |
| `rating_rank` | Numerical rating (1 = best, 8 = worst) |

**Rating Hierarchy:**

| Rank | Rating | Description |
|------|--------|-------------|
| 1 | Accredited | Highest certification level |
| 2 | Level 3 - Certified | Full certification |
| 3 | Level 2 - Substantial Compliance | Near-full compliance |
| 4 | Level 2 - Partial Compliance | Partial compliance |
| 5 | Level 1 - Limited Compliance | Limited compliance |
| 6 | Failed Certification | Did not pass |
| 7 | Failure of Activity | Activity-related failure |
| 8 | For Termination | Pending termination |

**Future Use:** Certification ratings can be incorporated into the priority scoring to favor higher-quality ESC schools in redistribution optimization.

### Output Files (Payload for Notebook 2.5)

| File | Description |
|------|-------------|
| `output/analysis_payload/congestion.parquet` | Congestion metrics table |
| `output/analysis_payload/redirections_greedy.parquet` | Greedy redirection log |
| `output/analysis_payload/redirections_lp.parquet` | LP redirection log |
| `output/analysis_payload/redirection_options.parquet` | All redirection options |
| `output/analysis_payload/esc_available.parquet` | ESC schools with available slots |
| `output/analysis_payload/flow_to_congested.parquet` | Flows to congested JHS |
| `output/analysis_payload/school_info.parquet` | School metadata |
| `output/analysis_payload/student_flow.parquet` | Complete flow data |
| `output/analysis_payload/blocked_demand_analysis.parquet` | Slot expansion priorities |
| `output/analysis_payload/metadata.json` | Parameters and statistics |

---

## Notebook Structure

### Section 0.0 - Setup
Standard imports: pandas, numpy, sklearn, json, re, gc.

### Section 1.0 - Load Data

| Section | Description | Key Details |
|---------|-------------|-------------|
| 1.1 | School Information | Loads `sch_info` with essential columns |
| 1.2 | Distance Matrix | Loads expanded matrix for NCR/Region IV-A/Region III |
| 1.3 | Student Flow | Grade 7 flow data (SY 2023-24) |
| 1.4 | Public School Seats | Seat counts by education level |
| 1.5 | JHS Enrollment | **Filters:** `school_year == '2023_24'`, `education_level == 'junior_high_school'` |
| 1.6 | School Coordinates | NCR/Region IV-A/Region III schools |
| 1.7 | ESC Slots | **Official `slots_unutilized` from GASS** |
| 1.8 | School Fees (GASTPE) | Tuition data for ESC-delivering schools |
| 1.9 | Private School Size | JHS enrollment for private schools, calculates `size_bonus` using percentile rank |
| 1.10 | ESC Certification Ratings | Certification levels for ESC schools (for future use) |

### Section 2.0 - Build Congestion Table

| Section | Description | Key Details |
|---------|-------------|-------------|
| 2.1 | Filter Schools | `sector == 'Public'`, `offers_jhs == True`, `old_region in ['NCR', 'Region IV-A', 'Region III']` |
| 2.2 | Join Enrollment | Maps `enrollment_jhs` from aggregated JHS enrollment |
| 2.3 | Join Seat Data | Filters seats to `education_level == 'Junior High School'` |
| 2.4 | Impute Missing Seats | KNN with k=5, features: `[old_region, division, enrollment_jhs]` |
| 2.5 | Calculate Congestion | Computes `seat_utilization_jhs` and `count_aisle_learner_jhs` |

### Section 3.0 - Prepare Optimization Data

| Section | Description | Key Details |
|---------|-------------|-------------|
| 3.1 | Filter ESC Schools | ESC schools present in distance matrix; **must aggregate duplicate `school_id`** |
| 3.2 | Filter Student Flow | Both origin and destination in `re_schids` |
| 3.3 | Build Distance Lookup | Origin ES → Private ESC pairs within `MAX_DISTANCE_KM` |
| 3.4 | Link Flow to Congested | Join flow with congestion data, filter to congested destinations |
| 3.5 | Build Redirection Options | Merge flow with origin-to-private distance lookup |
| 3.6 | Available ESC Slots | **Uses `slots_unutilized` directly from official GASS data** |
| 3.7 | Add Tuition and School Size | Join tuition + size data, calculate `priority_score` with size bonus |

#### Section 3.1 - Important: Duplicate School ID Handling

The `processed_esc_slots.parquet` file has one row per `esc_school_id`, but multiple ESC school IDs can map to the same `school_id` (DepEd ID). When filtering to schools in the distance matrix, **aggregate duplicates** to avoid row explosion in downstream merges:

```python
# Aggregate duplicates: sum slots for schools with multiple ESC grants
esc_in_loc = esc_in_loc.groupby("school_id", as_index=False).agg({
    "slots_total": "sum",
    "slots_unutilized": "sum"
})
```

#### Section 3.6 - Available Slots (Changed)

**Previous approach (Notebook 2.4):**
```python
available_slots = total_slots - current_beneficiaries
```
Where `current_beneficiaries` was calculated from flow data.

**Current approach (Notebook 2.4c):**
```python
# Use official slots_unutilized data directly (no recalculation needed)
redirection_options = redirection_options.rename(
    columns={"slots_unutilized": "available_slots"}
)
```

This change uses official stakeholder data from GASS, which is more accurate than deriving available slots from flow data.

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

### Section 6.0 - Compare Results
Side-by-side comparison table of Greedy vs LP optimization.

### Section 6.1 - Blocked Demand Analysis
Analyze remaining demand after LP optimization to identify slot expansion priorities.

### Section 7.0 - Export Payload
Exports all key dataframes and metadata for report generation.

---

## Key Parameters

| Parameter | Value | Location | Description |
|-----------|-------|----------|-------------|
| `MAX_DISTANCE_KM` | 5 | Section 3.3 | Maximum distance from origin ES to private ESC |
| `TUITION_WEIGHT` | 0.5 | Section 3.7 | Weight for tuition in priority score |
| `SIZE_WEIGHT` | 0.3 | Section 3.7 | Weight for size bonus in priority score |
| `COST_PENALTY` | 0.3 | Section 5.4 | LP penalty for expensive schools |
| `DISTANCE_PENALTY` | 0.3 | Section 5.4 | LP penalty for distant schools |
| `SIZE_WEIGHT_LP` | 0.3 | Section 5.4 | LP weight for size bonus |
| `n_neighbors` | 5 | Section 2.4 | KNN imputation neighbors |

---

## Key Formulas

### Priority Score (Greedy)
```
priority_score = norm_distance + (TUITION_WEIGHT × norm_tuition) - (SIZE_WEIGHT × size_bonus)
```
- Normalized per destination group
- `size_bonus` = percentile rank of school's JHS enrollment among all private schools
- Lower = better (nearer + cheaper + larger schools prioritized)

### LP Objective Coefficient
```
coef = 1.0 - (DISTANCE_PENALTY × norm_distance) - (COST_PENALTY × norm_tuition) + (SIZE_WEIGHT_LP × size_bonus)
```

### Congestion Metrics
```
seat_utilization_jhs = enrollment_jhs / seat_count
count_aisle_learner_jhs = max(0, enrollment_jhs - seat_count)
```

### Available Slots
```
available_slots = slots_unutilized  # Official GASS data
```

---

## Data Filters Summary

| Data | Filter Criteria |
|------|-----------------|
| Schools | `sector == 'Public'`, `offers_jhs == True`, `old_region in ['NCR', 'Region IV-A', 'Region III']` |
| Enrollment | `school_year == '2023_24'`, `education_level == 'junior_high_school'` |
| Seats | `education_level == 'Junior High School'` |
| Flow | Both origin and destination in distance matrix |
| ESC Schools | Present in distance matrix, aggregated by `school_id` |
| Redirection Options | `available_slots > 0`, distance ≤ 5km |

---

## Future Enhancements

### ESC Certification Integration

The ESC certification ratings loaded in Section 1.10 can be incorporated into the priority scoring to favor higher-quality schools:

```python
# Potential enhancement to priority score
CERTIFICATION_WEIGHT = 0.2
priority_score = (
    norm_distance
    + TUITION_WEIGHT × norm_tuition
    - SIZE_WEIGHT × size_bonus
    + CERTIFICATION_WEIGHT × norm_rating_rank  # Lower rank = better
)
```

This would prioritize:
1. Nearer schools (lower distance)
2. Cheaper schools (lower tuition)
3. Larger schools (higher size_bonus)
4. Better-certified schools (lower rating_rank)

---

## Related Files

- **Notebook 2.4:** Original analysis (NCR + Region IV-A only)
- **Notebook 2.5:** Report generation (loads payload from this notebook)
- **Notebook 1.8:** ESC slots processing (source of `slots_unutilized`)
- **Notebook 1.10:** ESC certification ratings processing
- **Notebook 2.2:** Distance matrix generation

---

## Changelog

| Date | Changes |
|------|---------|
| 2026-02-04 | **Initial version:** Extended notebook 2.4 to include Region III; Added `processed_esc_certification_rating.parquet` from notebook 1.10; Changed available slots calculation to use official `slots_unutilized` from GASS data; Added Section 3.1 duplicate `school_id` aggregation requirement; Updated coordinates file to `all_schools_coordinates_ncr_region4a_region3.parquet` |
