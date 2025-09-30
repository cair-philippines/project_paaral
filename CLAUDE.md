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

### 8. PSGC Reconciler (`modules/psgc_reconciler.py`)
- **Purpose**: Resolve PSGC code discrepancies between shapefile and CSV data
- **Input**: Shapefile GeoDataFrame + CSV DataFrame from module 7
- **Problem**: ~3,500 barangays have mismatched PSGC codes between OCHA shapefiles and PSA CSV data
- **Multi-Strategy Matching Pipeline** (sequential execution):
  1. **corr_code matching**: Direct mapping using shapefile's correction code field
  2. **Spatial join**: Geographic containment (barangay within municipality) + name matching
  3. **Hierarchical matching**: Admin hierarchy (region/province/municipality) + normalized names
  4. **Fuzzy matching**: String similarity at 90%, 85%, 80% thresholds (requires `rapidfuzz`)
- **Output**: Two GeoDataFrames - matched records and unmatched (for manual review)
- **Statistics Tracking**: Records matched/unmatched at each strategy level
- **Export Formats**: CSV, GeoJSON, Shapefile, GeoPackage

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

### 2025-09-30 (Earlier in Session)
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

### 2025-09-30 (Session 2)
- **PSGC Reconciler Module**: Created `psgc_reconciler.py` to resolve shapefile-CSV PSGC code mismatches
  - **Problem Identified**: ~3,500 barangays (out of 45,597 in shapefile) don't match CSV PSGC codes
    - Same barangay names, different PSGC codes between OCHA shapefile and PSA CSV data
    - Example: "Agapito del Rosario" has code `0305401001` in shapefile vs `0305403013` in CSV
  - **Multi-Strategy Matching Approach** (in priority order):
    1. **corr_code field**: Use shapefile's correction code field for direct mapping
    2. **Spatial join**: Match by geography (which municipality contains barangay) + name
    3. **Hierarchical matching**: Match using region→province→municipality→barangay hierarchy + normalized names
    4. **Fuzzy string matching**: Handle name variations with 90%, 85%, 80% similarity thresholds
  - **Implementation Features**:
    - Verbose logging tracks progress through each matching strategy
    - Statistics tracking for each strategy's success rate
    - Name normalization (lowercase, whitespace, special characters)
    - PSGC code standardization (leading zeros)
    - Requires `rapidfuzz` library for fuzzy matching (optional dependency)
  - **Module Methods**:
    - `reconcile()`: Main pipeline running all strategies
    - `print_summary()`: Display match statistics
    - `export_results()`: Save matched/unmatched to CSV, GeoJSON, GeoPackage
  - **Deliverables**:
    - `modules/psgc_reconciler.py`: Core reconciliation module
    - `test_psgc_reconciler.py`: Command-line test script
    - `notebooks/0.3-psgc-reconciliation.ipynb`: Interactive demonstration notebook

### 2025-09-30 (Current Session)
- **Repository Analysis**: Examined `philippines-psgc-shapefiles` repository structure
  - **script.py Investigation**: Analyzed the repository's data processing pipeline
    - Uses Q4 2023 PSGC codes (not 2024) to update shapefiles
    - Implements 5-stage matching: exact match → manual corrections → corr_code → municipality name → admin hierarchy
    - ~70 manual municipality PSGC updates (highly urbanized cities, NCR, Maguindanao)
    - ~3,500 barangays failed all matching stages - these are the discrepancies we found
  - **dist/ Files Clarification**: Confirmed processed files in `dist/` are script.py outputs
    - Not raw data, but partially processed with known limitations
    - Repository README claims "cleaned and matched with most recent PSGC codes" but matching incomplete
    - Our reconciler module complements/completes the repository's work
  - **OSM Reverse Geocoding Discussion**: Proposed 5th matching strategy
    - Use barangay polygon centroids to query OpenStreetMap Nominatim API
    - Extract municipality/province names from OSM
    - Match OSM admin names with CSV data
    - Pros: Fresh data, handles renamed areas. Cons: API rate limits (~1 req/sec), 60 min for 3,500 queries

- **PSGC Consolidator Revision**: Rewrote `modules/psgc_consolidator.py` based on user's preferred approach
  - **User's Requirements** (from `notebooks/0.2-map-resources.ipynb` section 1.1.3):
    1. Fix City of Manila missing data (897 NCR barangays with NaN municipality)
    2. Standardize PSGC codes to 10-digit string format (add leading zeros)
    3. Reorder columns: [PSGC codes] + [names reversed] + [other data]
    4. Shapefile→CSV left join (preserve all valid geometries, not CSV-first)
    5. Filter null geometries before merge (42,048 features vs 45,597)
    6. Select only relevant shapefile columns (5 columns: psgc_code, corr_code, name, adm4_en, geometry)
    7. String dtype for all PSGC codes
  - **New Helper Methods**:
    - `_add_leading_zeros()`: Converts 9-digit codes to 10-digit strings
    - `_fix_city_of_manila()`: Fills missing NCR municipality names
    - `_prepare_shapefile_for_merge()`: Standardizes codes, filters geometry, selects columns
  - **Enhanced consolidate_hierarchy()**: Now includes 7-step process (was 4 steps)
  - **Revised merge_with_geometry()**: Shapefile-first left join strategy (was right join)
  - **Output Changes**: 42,048 features (was 42,017), 0 missing City of Manila (was 897), cleaner columns
  - **Testing**: Created `test_revised_psgc_consolidator.py` validation script

- **Configuration System**: Created portable config system for environment-agnostic notebook execution
  - **config/config.json**: Centralized path configuration
    - All relative paths (data, modules, notebooks, output, psgc_shapefiles, etc.)
    - PSGC shapefile names (adm0-adm4)
    - Logging and Jupyter settings
  - **config/config.py**: Path resolution module
    - `Config` class with automatic project root detection (searches for modules/, data/, notebooks/)
    - `setup_notebook()`: One-line notebook setup (changes dir, adds to sys.path)
    - `get_path()`, `get_psgc_path()`, `get_data_path()`, `get_output_path()`: Path utilities
    - `PROJECT_ROOT` constant for backward compatibility
  - **config/__init__.py**: Package initialization with exports
  - **config/README.md**: Complete documentation (API reference, migration guide, troubleshooting)
  - **config/USAGE_EXAMPLES.md**: 10 real-world usage examples
  - **config/notebook_setup.py**: Bootstrap helper for notebooks
  - **config/NOTEBOOK_TEMPLATE.md**: Copy-paste templates for notebook setup
  - **Bootstrap Solution**: Fixed `ModuleNotFoundError` when importing config
    - Problem: Notebooks run from `notebooks/`, Python can't find `config` without project root in sys.path
    - Solution: 3-line bootstrap adds project root to path before importing config
    ```python
    import sys
    from pathlib import Path
    project_root = Path.cwd().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from config import setup_notebook
    setup_notebook()
    ```
  - **Testing**: Created `test_config.py` comprehensive test suite
  - **Benefits**: Environment-agnostic, no hardcoded paths, single import replaces manual setup

## Architecture
- **Pattern**: All processors follow consistent architecture (load→process→validate→export)
- **Logging**: `verbose=True` (INFO level) or `verbose=False` (WARNING only)
- **Integration**: Standardized School IDs enable cross-dataset merging
- **Configuration**: Portable config system (`config/`) for environment-agnostic execution
  - Auto-detects project root from any directory
  - Centralized path management via `config.json`
  - Bootstrap solution for notebook imports
  - No hardcoded absolute paths
- **Extensible**: Easy to add new datasets following established patterns