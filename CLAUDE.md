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
- **Problem**: ~3,500 barangays have mismatched PSGC codes between OCHA shapefiles and PSA CSV data
- **Multi-Strategy Matching Pipeline**:
  1. corr_code matching → 2. Spatial join → 3. Hierarchical matching → 4. Fuzzy matching (90%, 85%, 80%)
- **Output**: Matched/unmatched GeoDataFrames for manual review
- **Export Formats**: CSV, GeoJSON, Shapefile, GeoPackage

### 9. Regional Road Network Extractor (`modules/regional_road_network_extractor.py`)
- **Purpose**: Extract OSM drive networks for Philippine regions using OSMNx with province-level querying for archipelagic reliability
- **Input**: GeoDataFrame from psgc_consolidator (module 7)
- **PSGC Code Structure**: First 2 digits = region, first 4 digits = province, digits 5-7 = municipality, digits 8-10 = barangay
- **Query Methods**:
  - **Province Breakdown** (default, recommended): Queries each province separately then merges → complete coverage
  - **Direct Query**: Queries entire region shapefile → faster but may miss islands in multi-polygon regions
- **Key Features**:
  - MultiPolygon decomposition: Splits islands for individual querying when `decompose_islands=True`
  - Automatic caching: Repeated queries return instantly
  - Edge deduplication: Merges graphs by osmid to remove border duplicates
  - Region/province filtering by 2-digit/4-digit codes or names
- **Visualization Methods**:
  - `plot_graph()`: OSMnx native plotting (no igraph dependency, ARM-compatible)
  - `plot_graph_with_boundary()`: Network overlaid on region/province shapefile boundaries
  - Customizable styling: colors, linewidths, transparency, DPI
- **Output**: NetworkX MultiDiGraph + metadata (nodes, edges, query method, statistics)
- **Export Options**: GeoDataFrame (shapefile/GeoJSON), GraphML
- **Use Case**: Extract complete road networks for spatial analysis, network metrics, accessibility studies

## Common Features (All Modules)
- **Verbose Logging**: `verbose` parameter (default: True) controls INFO vs WARNING level logging
- **Whitespace Trimming**: `_trim_whitespaces()` method for string columns
- **Data Type Optimization**: Categorical columns with proper ordering
- **Validation**: Bounds checking, null handling, data type conversion
- **Integration Ready**: Standardized School IDs (string type) for cross-dataset joining

## Data Integration
- **Primary Keys**:
  - School IDs (string) across education modules 1-6
  - PSGC codes (10-digit string): First 2 digits = region, first 4 = province, digits 5-7 = municipality, 8-10 = barangay
- **Coverage**:
  - Education: Public (~47K) + Private (~11K) schools with coordinates, enrollment, seats, furniture, tuition
  - Geography: Complete PH admin hierarchy (42K+ barangays) with geometries
  - Infrastructure: OSM road networks by region/province via module 9
- **Spatial Integration**:
  - School coordinates → PSGC boundaries → Road networks
  - Enable accessibility analysis, catchment areas, network metrics

## Key Patterns
1. **Variable Header Detection**: CSV skip 5 rows, Excel rows 6-10
2. **Wide→Long Transformation**: Consistent long format output
3. **Categorical Ordering**: Custom education progressions for analysis
4. **Quality Validation**: Geographic bounds, positive counts, data consistency

## Session History

### 2025-09-30 Sessions (Summary)
- **Modules 1-6**: Created education data preprocessors (enrollment, coordinates, seats, furniture, tuition)
- **Module 7**: PSGC Consolidator - hierarchical merge of 4 admin levels + 366MB shapefile, 42,048 features
  - Fixed City of Manila missing data, 10-digit PSGC standardization, shapefile-first left join
- **Module 8**: PSGC Reconciler - 4-strategy pipeline to resolve ~3,500 mismatched barangay codes
- **Configuration System**: Created `config/` package for environment-agnostic notebook execution
  - Auto-detects project root, centralized paths, 3-line bootstrap solution

### 2025-10-01 (Current Session)
- **Module 9: Regional Road Network Extractor** (`modules/regional_road_network_extractor.py`)
  - **Problem**: Archipelagic regions (MIMAROPA, Central Visayas) return incomplete OSMNx queries
  - **Solution**: Province-level querying with automatic island decomposition
  - **PSGC Digit Structure Implementation**:
    - Updated all methods to use first 2 digits for region codes (e.g., '07' = Central Visayas)
    - First 4 digits for province codes (e.g., '0722' = Cebu)
    - Helper methods: `_extract_region_code()`, `_extract_province_code()`
  - **Query Options**:
    - `use_province_breakdown=True` (default): Queries each province → merge → complete coverage
    - `use_province_breakdown=False`: Direct region query → faster but may miss islands
  - **Visualization Methods** (OSMnx native, no igraph dependency):
    - `plot_graph()`: Simple network plot
    - `plot_graph_with_boundary()`: Network overlaid on region/province shapefiles
    - Both support custom styling (colors, linewidths, alpha, DPI)
  - **Features**: Caching, edge deduplication by osmid, MultiPolygon decomposition
  - **Added comprehensive docstring examples**: 13 usage patterns covering all methods

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