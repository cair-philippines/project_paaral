# Claude Code Session Log - Education Data Processing

## Modules Created (SY 2023-2024)

### 1. Enrollment Data (`modules/enrollment_preprocessor.py`)
- **Source**: `data/public/Copy of SY 2023-2024 SCHOOL LEVEL DATA ON ENROLLMENT.csv`
- **Output**: Long format with `school_id`, `grade_level`, `gender`, `academic_track`, `student_type`, `enrollment_count`
- **Key Data**: 27M+ enrollments, grade levels K→G12, resolved double counting & Special Needs data (164K students)

### 2. Public School Coordinates (`modules/school_coordinates_preprocessor.py`)
- **Source**: `data/public/SY 2023-2024 LIST OF SCHOOLS WITH LONGITUDE AND LATITUDE.xlsx`
- **Output**: ~47K schools with coordinates + quality flags (valid/missing/out_of_bounds/potentially_switched)
- **Validation**: Philippine bounds (116°-127°E, 4°-21°N), lat/lon reversal detection

### 3. Private School Coordinates (`modules/private_schools_processor.py`)
- **Source**: `data/private/raw_validation_sheets/` (16 regional Excel files)
- **Output**: ~11,837 schools with coordinates, region/division tracking
- **Features**: Dynamic "Region" detection (regex), DMS/DMM formats, optimized Excel reading (10x faster)

### 4. Seat-Learner Ratio (`modules/seat_learner_preprocessor.py`)
- **Source**: `data/public/SY 2023-2024 SEAT-LEARNER RATIO.xlsx`
- **Output**: Long format with `school_id`, `education_level` (Elementary/JHS/SHS), `seat_count`

### 5. Private Furniture (`modules/private_furniture_preprocessor.py`)
- **Source**: `data/private/priv_classroom_furniture.xlsx`
- **Output**: Long format with `school_id`, `grade_level`, `furniture_type`, `furniture_count`
- **Features**: DepEd EMISD furniture multipliers (Desks: 2x), grade level standardization

### 6. Subsidy Tuition (`modules/subsidy_tuition_processor.py`)
- **Source**: `data/private/ESC and SHSVP Tuition.xlsx`
- **Tab 1 (ESC)**: Wide→Long transformation for G7-G10 tuition/fees
- **Tab 2 (SHSVP)**: Long format SHS tuition by Track/Strand
- **Output**: Two DataFrames with `school_id`, grade/track/strand info, fee types, amounts
- **Features**: Automatic strand expansion (splits concatenated NC I/II/III programs into separate rows)

### 7. PSGC Geographic Data Consolidator (`modules/psgc_consolidator.py`)
- **Source**: `data/philippines-psgc-shapefiles/dist/` (4 CSV files + 1 shapefile)
  - PH_Adm1_Regions.csv (17 regions)
  - PH_Adm2_ProvDists.csv (88 provinces/districts)
  - PH_Adm3_MuniCities.csv (1,642 municipalities/cities)
  - PH_Adm4_BgySubMuns.csv (42,017 barangays)
  - PH_Adm4_BgySubMuns.shp.zip (45,597 geometries, 366 MB)
- **Output**: GeoDataFrame with complete hierarchical geography + geometries
- **Consolidation Process**:
  1. Hierarchical left joins starting from Adm4 (barangay level)
  2. Join Adm4 ← Adm3 on `[adm1_psgc, adm2_psgc, adm3_psgc]`
  3. Join Result ← Adm2 on `[adm1_psgc, adm2_psgc]`
  4. Join Result ← Adm1 on `[adm1_psgc]`
  5. Merge consolidated CSV with shapefile on `psgc_code`
- **Features**:
  - Complete Philippine administrative hierarchy (Region→Province→Municipality→Barangay)
  - CRS: EPSG:4326 (WGS84)
  - Includes geographic measurements (area, length) at all levels
  - Filter methods by region/province
  - Export to GeoJSON, Shapefile, GeoPackage, CSV

## Common Features (All Modules)
- **Verbose Logging**: `verbose` parameter (default: True) controls INFO vs WARNING level logging
- **Whitespace Trimming**: `_trim_whitespaces()` method for string columns
- **Data Type Optimization**: Categorical columns with proper ordering
- **Validation**: Bounds checking, null handling, data type conversion
- **Integration Ready**: Standardized School IDs (string type) for cross-dataset joining

## Data Integration
- **Primary Keys**:
  - School IDs (string) across education modules 1-6
  - PSGC codes (integer) for geographic hierarchy in module 7
- **Coverage**:
  - Education: Public (~47K) + Private (~11K) schools with coordinates, enrollment, seats, furniture, tuition data
  - Geography: Complete PH administrative hierarchy with 42K+ barangays and geometries
- **Grade Level Harmonization**: Multiple categorical systems aligned for analysis
- **Quality Flags**: Coordinate validation, furniture multipliers, strand expansion
- **Spatial Integration**: School coordinates can be joined with PSGC geographic boundaries for spatial analysis

## Key Patterns
1. **Variable Header Detection**: CSV skip 5 rows, Excel rows 6-10
2. **Wide→Long Transformation**: Consistent long format output
3. **Categorical Ordering**: Custom education progressions for analysis
4. **Quality Validation**: Geographic bounds, positive counts, data consistency

## Session History

### 2025-09-30 (Earlier)
- **Excel Performance**: Optimized `private_schools_processor.py` with multi-engine selection (calamine→fastexcel→openpyxl) for 10x speedup
- Validated all modules against source data totals

### 2025-09-30 (Earlier Session)
- **Subsidy Tuition Module**: Created `subsidy_tuition_processor.py` for ESC/SHSVP tuition data
  - Wide→Long transformation for ESC (G7-G10)
  - Strand expansion: splits concatenated NC I/II/III programs (regex: `r'\(NC\s*I+\)|NC\s*I+'`)
- **Verbose Logging**: Added `verbose` parameter to all 6 preprocessor modules
- **Agent Update**: Updated `.claude/agents/data-extractor.md` to always include verbose option in future modules

### 2025-09-30 (Current Session)
- **PSGC Geographic Consolidator Module**: Created `psgc_consolidator.py` for Philippine geographic hierarchy
  - **Data Source Investigation**: Examined `philippines-psgc-shapefiles` repository structure
    - Found 4 administrative levels (Adm0-4) with CSV + shapefile pairs
    - Source: https://github.com/altcoder/philippines-psgc-shapefiles
  - **Consolidation Strategy**: Designed hierarchical merge approach
    - Start with Adm4 (barangay) as base: 42,017 records
    - Sequential left joins: Adm4→Adm3→Adm2→Adm1
    - PSGC code matching: multi-column joins for proper hierarchy
  - **Git LFS Challenge**: Repository shapefiles were Git LFS pointers (134 bytes each)
    - LFS bandwidth quota exceeded on GitHub
    - Resolution: Downloaded 366 MB `PH_Adm4_BgySubMuns.shp.zip` directly from OCHA source
  - **Module Implementation**:
    - Initial attempt: Manual zip extraction with `zipfile` + `tempfile` → Failed
    - Fix: Geopandas native zip reading with `zip://` protocol
    - Column mapping issue: Shapefile uses `psgc_code`, CSV uses `adm4_psgc`
    - Final merge: `left_on='psgc_code'`, `right_on='adm4_psgc'`
  - **Output**: GeoDataFrame with 42K barangays, complete hierarchy, EPSG:4326 CRS
  - **Notebook**: Created `notebooks/consolidate_psgc_data.ipynb` for interactive exploration

## Architecture
- **Pattern**: All processors follow consistent architecture (load→process→validate→export)
- **Logging**: `verbose=True` (INFO level) or `verbose=False` (WARNING only)
- **Integration**: Standardized School IDs enable cross-dataset merging
- **Extensible**: Easy to add new datasets following established patterns