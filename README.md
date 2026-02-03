# Project PAARAL

School congestion — where enrollment exceeds physical capacity — is a persistent challenge in low- and middle-income countries that deepens educational inequity. This project introduces a data-driven framework for modeling student flow patterns and simulating policy scenarios under the Philippine Educational Service Contracting (ESC) program, one of the world's largest education subsidy programs. By synthesizing heterogeneous government data across nearly 3,000 institutions, we demonstrate how adjustments in inter-school distances, subsidy amounts, and slot availability can serve as actionable levers for relieving system-wide congestion in Philippine basic education.

---

## Directory Structure

```
project_paaral/
├── data/                    # Raw and processed input data (not tracked in git)
│   ├── public/              # Publicly sourced DepEd datasets
│   ├── private/             # Private/restricted datasets (ESC, tuition, furniture)
│   └── processed/           # Intermediate processed files
├── modules/                 # Reusable Python modules for data processing
├── notebooks/               # Jupyter notebooks (processing → analysis → reports)
│   ├── 1.x                  # Stage 1: Data processing
│   ├── 2.x                  # Stage 2: Analysis and optimization
│   └── 2.xb                 # Explainer/verification variants
├── output/                  # Generated outputs (not tracked in git)
│   ├── analysis_payload/    # Parquet files passed from 2.4 → 2.5
│   └── reports/             # Stakeholder report CSVs
├── references/              # Literature and documentation
│   └── documentation/       # Notebook-level .md documentation
└── results/                 # Final results for publication
```

## Notebook Pipeline

Notebooks are numbered to reflect their execution order. **Stage 1** processes raw data into clean parquet/CSV outputs. **Stage 2** consumes those outputs for analysis and reporting.

### Stage 1: Data Processing

Each notebook reads from `data/` and writes processed files to `output/`.

| Notebook | Purpose | Key Output |
|----------|---------|------------|
| **1.1** Process Enrollment | Consolidate multi-year enrollment records | `processed_project_bukas_school_enrollment.parquet`, `processed_project_bukas_school_information.parquet` |
| **1.3** Process G7 Enrollment | Parse Grade 7 origin-destination enrollment | `processed_grade_7_enrollment.parquet` |
| **1.4** Process ESC Beneficiaries | Parse ESC beneficiary lists by region/year | `processed_esc_beneficiaries.parquet` |
| **1.5** Process Coordinates | Geocode public and private schools | `all_schools_coordinates_ncr_region4a.parquet` |
| **1.6** Process Seats | Extract seat counts for public and private schools | `processed_public_seats.parquet` |
| **1.7** Process GASTPE | Extract tuition fee data for ESC schools | `processed_gastpe_data.parquet` |
| **1.8** Process ESC Slots | Consolidate ESC slot allocations | `processed_esc_slots.parquet` |

### Stage 2: Analysis and Reporting

Each notebook reads from `output/` (files produced by Stage 1 or earlier Stage 2 notebooks).

| Notebook | Purpose | Depends On | Key Output |
|----------|---------|------------|------------|
| **2.1** Build Student Flow Table | Build origin→destination flow matrix from G7 data | 1.1, 1.3, 1.4 | `grade_7_student_flow_table_sy2324.parquet` |
| **2.2** Build Distance Matrix | Compute road-network distances via OSRM ([setup guide](references/osrm_setup_guide.md)) | 1.5 | `school_distance_matrix_osrm.npy`, `school_distance_matrix_index.json` |
| **2.4** Congestion Analysis | Optimize student redistribution (Greedy + LP) | 1.1, 1.5, 1.6, 1.7, 1.8, 2.1, 2.2 | `output/analysis_payload/` (10 files) |
| **2.4b** Congestion Explainer | Same analysis as 2.4 with enhanced documentation | *(same as 2.4)* | *(same as 2.4)* |
| **2.5** Stakeholder Reports | Generate CSV reports from analysis payload | 2.4 | `output/reports/` |

### Dependency Graph

```
data/
 │
 ├─ 1.1 ──┬──────────────────────────── 2.1 ──┐
 ├─ 1.3 ──┘                                    │
 ├─ 1.4 ──┘                                    │
 ├─ 1.5 ──────────────────────────── 2.2 ──┐   │
 ├─ 1.6 ──────────────────────────────────┐ │   │
 ├─ 1.7 ────────────────────────────────┐ │ │   │
 └─ 1.8 ──────────────────────────────┐ │ │ │   │
                                       ▼ ▼ ▼ ▼   ▼
                                      2.4 / 2.4b
                                           │
                                           ▼
                                          2.5
```

---

## Required Data Files

The `data/` directory is not tracked in version control. To run the notebooks, populate the following files:

### `data/public/`

| Filename | Used In | Description |
|----------|---------|-------------|
| `SY 2022-2023 Gr 7 Enrollees.xlsx` | 1.3 | Grade 7 enrollment by origin-destination school pair |
| `SY 2023-2024 Gr 7 Enrollees.xlsx` | 1.3 | Grade 7 enrollment by origin-destination school pair |
| `SY 2024-2025 Gr 7 Enrollees.xlsx` | 1.3 | Grade 7 enrollment by origin-destination school pair |
| `SY 2023-2024 SEAT-LEARNER RATIO.xlsx` | 1.6 | Public school seat counts by education level |
| `SY 2024-2025 School Level Database WITH PSGC.xlsx` | 1.5 | School registry with geolocation and PSGC codes |

### `data/private/`

| Filename | Used In | Description |
|----------|---------|-------------|
| `Alphalist-Schools-Slots-addon_slots.csv` | 1.8 | ESC add-on slot allocations |
| `Alphalist-Schools-Slots-fixed_slots.csv` | 1.8 | ESC fixed slot allocations |
| `Alphalist-Schools-Slots-incentive_slots.csv` | 1.8 | ESC incentive slot allocations |
| `ESC and SHSVP Tuition.xlsx` | 1.7 | Tuition fee data for ESC/SHSVP schools |
| `priv_classroom_furniture.xlsx` | 1.6 | Private school classroom and furniture data |

### `data/private/ESC/`

ESC beneficiary lists organized by school year and region. Each school year directory contains 17 regional `.xlsx` files:

```
data/private/ESC/
├── SY 2022-2023/
│   ├── BARMM.xlsx
│   ├── CAR.xlsx
│   ├── NCR.xlsx
│   ├── R01.xlsx
│   ├── R02.xlsx
│   ├── R03.xlsx
│   ├── R04-A.xlsx
│   ├── R04-B.xlsx
│   ├── R05.xlsx
│   ├── R06.xlsx
│   ├── R07.xlsx
│   ├── R08.xlsx
│   ├── R09.xlsx
│   ├── R10.xlsx
│   ├── R11.xlsx
│   ├── R12.xlsx
│   └── R13.xlsx
├── SY 2023-2024/
│   └── (same 17 files)
└── SY 2024-2025/
    └── (same 17 files)
```

### `data/processed/`

| Filename | Used In | Description |
|----------|---------|-------------|
| `project_bukas_enrollment_2022-23.csv` | 1.1 | Pre-processed enrollment data |
| `project_bukas_enrollment_2023-24.csv` | 1.1 | Pre-processed enrollment data |
| `project_bukas_enrollment_2024-25.csv` | 1.1 | Pre-processed enrollment data |
