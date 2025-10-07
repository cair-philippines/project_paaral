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

### 3. Private School Coordinates (`modules/private_coordinates_processor.py`)
- **Source**: `data/private/raw_validation_sheets/` (16 regional Excel files)
- **Output**: ~11,837 schools with coordinates, region/division tracking
- **Features**:
  - Dynamic "Region" detection (regex), optimized Excel reading (10x faster)
  - **Coordinate Cleaning** (2025-10-02): Automatic preprocessing improves validity by 80-90%
    - Strips trailing commas (`, ` and `,`)
    - Removes cardinal direction suffixes (N/S/E/W with/without `°`)
    - Extracts first value before " or " text
    - Reconstructs split coordinates across columns
  - **Coordinate Validation**: Creates `coordinates_valid` (bool) and `coordinates_invalid_reason` (string) columns
  - Expected valid coordinates: ~95%+ (up from ~86%)

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

### 8. Regional Road Network Extractor (`modules/regional_road_network_extractor.py`) [DEPRECATED]
- **Status**: Superseded by Module 9 (Provincial Road Extractor) for better performance
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
- **Limitations**: Slow (hours for all provinces), memory-intensive, API-dependent

### 9. Provincial Road Network Extractor (`modules/provincial_road_extractor.py`)
- **Purpose**: Extract provincial road networks from OSM PBF files using memory-efficient PyOsmium streaming
- **Source**: `data/networks/philippines-251002.osm.pbf` (581MB from GeoFabrik)
- **Input**: Consolidated geodata from Module 7 (uses `adm2_pcode` for reliable province identification)
- **Output**: One `.geojsonl` file per province (88 files total)
  - Filename format: `{adm2_pcode}_{province_name}.geojsonl`
  - Example: `PH03014_bulacan.geojsonl`, `PH04021_cavite.geojsonl`
- **Key Features**:
  - **Streaming architecture**: Processes PBF file once, writes to all provinces simultaneously
  - **LRU file handle cache**: Manages 88 output files with max 16 open at once
  - **Spatial indexing**: STRtree for fast province intersection queries
  - **Highway filtering**: Extracts driveable roads only (motorway, trunk, primary, etc.)
  - **Metadata**: Includes `osm_id`, `highway`, `name`, `oneway`, `maxspeed`
- **Performance**:
  - Processes entire Philippines in **~2.8 minutes** (vs hours with OSMNx)
  - Constant low memory usage via streaming
  - Offline operation (no API dependencies)
- **Methods**:
  - `extract_all_provinces()`: Extract all 88 provinces
  - `extract_provinces(whitelist)`: Extract specific provinces by adm2_pcode
  - `get_province_list()`: List provinces with pcodes, names, filenames
- **Parameters**: `verbose`, `do_clip`, `max_open_files`
- **Advantages**: 20-30x faster than Module 8, memory-efficient, reliable, consistent data

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
  - Infrastructure: OSM road networks by region/province via module 8
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
- **Configuration System**: Created `config/` package for environment-agnostic notebook execution
  - Auto-detects project root, centralized paths, 3-line bootstrap solution

### 2025-10-01
- **Module 8: Regional Road Network Extractor** (`modules/regional_road_network_extractor.py`)
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

### 2025-10-02 (Sessions 1-3)

**Session 3 Summary**: Enhanced Module 3 (Private School Coordinates) with automatic coordinate cleaning and validation

- **Module 8 Investigation: Provincial Breakdown vs Direct Query Limitations**
  - **Problem**: Provincial breakdown method showed disjointed edges at boundaries
    - Separate provincial queries generate duplicate nodes with different IDs at same coordinates
    - `truncate_by_edge=True` was cutting roads at exact boundary, creating disconnected segments
    - Roads crossing boundaries appeared disconnected in merged graph

  - **Attempted Solutions**:
    1. **Spatial proximity-based node deduplication** (`_merge_boundary_nodes()`)
       - Uses scipy KDTree for efficient spatial indexing
       - Union-find algorithm to merge duplicate node clusters
       - Tested tolerances: 5m, 20m
    2. **Edge preservation** - Changed `truncate_by_edge=False` in all `graph_from_polygon()` calls
    3. **Increased buffer** - Tested up to 1000m

  - **Findings**: Provincial breakdown has **fundamental limitations**
    - OSM Overpass API returns **different/incomplete data** for small provincial queries vs large regional queries
    - Provincial breakdown shows significantly **lower road network density** in central areas
    - Node merging and edge preservation cannot fix incomplete source data
    - Visual comparison (Region III): Direct query shows dense connected network, provincial breakdown shows sparse disconnected segments

  - **Recommendations**:
    - **Contiguous regions** (Region III, NCR, etc.): Use **direct query only**
      - Better data quality and density
      - Natural connectivity preservation
    - **Archipelagic regions** (MIMAROPA, Central Visayas): Use **provincial breakdown**
      - Accepts some data loss for geographic completeness
      - Direct query may miss entire islands
    - Provincial breakdown is a **coverage vs quality tradeoff**, not a superior method

  - **Current Investigation**: Buffer behavior
    - Buffer parameter (e.g., `buffer_meters=1000`) applied to query polygon
    - Expected: Roads extend beyond region boundary (shows cross-boundary connections)
    - Observed: Roads still contained within original boundary even with 1km buffer
    - Investigating if OSMNx simplification or boundary recognition is trimming results post-query

- **Module 3 Enhancement: Coordinate Cleaning** (`modules/private_coordinates_processor.py`)
  - **Problem**: ~1,625 invalid coordinates due to minor formatting issues
    - Trailing commas: `"16.422706348227834, "` (hundreds of cases)
    - Cardinal direction suffixes: `"17.4665 N"`, `"121.4622 E"`
    - Alternative formats: `"16.3931668 or 16°23′34″N"`
    - Split coordinates: `"16.388404775016976, 1"` (lat) + `"20.60320161"` (lon)

  - **Solution**: New `clean_coordinates()` method with preprocessing steps
    1. Strip trailing commas (`, ` and `,`)
    2. Remove cardinal direction suffixes (N/S/E/W with/without `°` symbols)
    3. Extract first value before " or " text
    4. Reconstruct split coordinates across columns
    5. Strip whitespace

  - **New Methods**:
    - `clean_coordinates()`: Main cleaning method with statistics tracking
    - `_clean_single_coordinate(value)`: Clean individual coordinate values
    - `_reconstruct_split_coordinates(df, lat_col, lon_col)`: Fix coordinates split by commas
    - `validate_coordinates_with_reasons(clean_first=True)`: Validate with automatic cleaning

  - **Integration**:
    - `validate_coordinates_with_reasons()` now calls `clean_coordinates()` by default
    - Creates `coordinates_valid` (bool) and `coordinates_invalid_reason` (string) columns
    - Expected improvement: 80-90% reduction in invalid coordinates

  - **Bug Fixes**:
    - Fixed `read_only` parameter error in `pd.read_excel()` - now passes via `engine_kwargs`
    - Fixed `get_summary()` AttributeError - changed from `.keys()` to direct list copy

  - **Usage Example**:
    ```python
    processor = pcp.PrivateSchoolsProcessor(directory_path='../data/private/raw_validation_sheets')
    processed_data = processor.process()

    # Automatic cleaning + validation (recommended)
    validated_data = processor.validate_coordinates_with_reasons(clean_first=True)

    # View invalid coordinates with reasons
    invalid = validated_data[~validated_data['coordinates_valid']]
    print(invalid[['school_name', 'latitude', 'longitude', 'coordinates_invalid_reason']])
    ```

### 2025-10-05

**Summary**: Enhanced Module 7 with spatial matching for unmatched barangays, shifted road network extraction to PyOsmium architecture, configuration cleanup

### 2025-10-06

**Session 1 Summary**: Debugged and fixed spatial matching bugs - reference boundaries now populated from authoritative CSV sources, and mask recreation bug preventing column updates resolved

- **Spatial Matching Bug Fix** (`modules/psgc_consolidator.py` - `_build_reference_boundaries()`)
  - **Problem Identified**: Spatial matching was still producing significant NaN values in region/province names
    - Root cause: Reference boundaries were inheriting NaN values from matched barangays
    - Even "matched" barangays (where adm1_psgc is not null) had many NaN values in name columns:
      - `adm2_en`: 3,097 NaN values (province names)
      - `adm1_en`: 14 NaN values (region names)
      - `adm3_en`: 16 NaN values (municipality names)
    - When spatial matching copied from reference boundaries, it was copying these NaN values

  - **Solution**: Populate names from authoritative CSV sources
    - Changed `_build_reference_boundaries()` to merge with admin-level CSV data after dissolving
    - After dissolving matched barangays to municipality level, now:
      1. Keeps only PSGC codes and geometry initially
      2. Merges with `adm3_data` to get municipality names
      3. Merges with `adm2_data` to get province names
      4. Merges with `adm1_data` to get region names
      5. Ensures PSGC codes have leading zeros for proper matching
    - This guarantees reference boundaries have complete name information from source CSV files
    - Added logging to report name completeness statistics

  - **Technical Details**:
    ```python
    # Before (buggy): Kept name columns from dissolved matched barangays
    municipalities = municipalities[
        ['adm1_psgc', 'adm2_psgc', 'adm3_psgc',
         'adm1_en', 'adm2_en', 'adm3_en', 'geometry']  # These had NaN values!
    ]

    # After (fixed): Merge with authoritative sources
    municipalities = municipalities[['adm1_psgc', 'adm2_psgc', 'adm3_psgc', 'geometry']]
    # Merge with adm3_data, adm2_data, adm1_data to populate names
    municipalities = municipalities.merge(adm3_names, ...).merge(adm2_names, ...).merge(adm1_names, ...)
    ```

  - **Impact**: Spatial matching now produces complete admin codes AND names for all ~3,580 unmatched barangays

- **Spatial Matching Critical Bug Fix** (`modules/psgc_consolidator.py` - `apply_spatial_matching()`)
  - **Problem**: Spatially matched barangays still had NaN values in all columns except adm1_psgc
    - Only the first column (adm1_psgc) was being updated
    - All other columns (adm2_psgc, adm3_psgc, adm1_en, adm2_en, adm3_en) remained NaN
    - Reference boundaries had complete data, but updates weren't being applied

  - **Root Cause**: Mask was being recreated inside the for loop
    - Loop iteration 1 (adm1_psgc): Mask finds 3,580 rows with NaN, updates adm1_psgc ✓
    - Loop iteration 2 (adm2_psgc): Mask recreated - finds 0 rows (adm1_psgc now filled!), updates nothing ✗
    - Subsequent iterations update nothing ✗

  - **The Bug**:
    ```python
    # BUGGY CODE (line 648)
    for col in ['adm1_psgc', 'adm2_psgc', 'adm3_psgc', 'adm1_en', 'adm2_en', 'adm3_en']:
        mapping = dict(zip(matched_codes['psgc_code'], matched_codes[col]))
        mask = self.consolidated_geodata['adm1_psgc'].isna()  # BUG: Recreated in loop!
        self.consolidated_geodata.loc[mask, col] = ...
    ```

  - **The Fix**:
    ```python
    # FIXED CODE
    # Create mask ONCE before loop
    mask = self.consolidated_geodata['adm1_psgc'].isna()

    for col in ['adm1_psgc', 'adm2_psgc', 'adm3_psgc', 'adm1_en', 'adm2_en', 'adm3_en']:
        mapping = dict(zip(matched_codes['psgc_code'], matched_codes[col]))
        self.consolidated_geodata.loc[mask, col] = ...  # Uses same mask for all columns
    ```

  - **Impact**: All 6 columns now properly updated for spatially matched barangays

  - **Debugging Process**:
    - User ran section 2.1 and reported NaN values still present after spatial matching
    - Added diagnostic cells to notebook to check:
      1. Reference boundaries structure and name completeness
      2. PSGC code formats and data types across all sources
    - Diagnostic results showed:
      - ✅ Reference boundaries properly populated (1,582 municipalities, only 1 NaN in adm3_en)
      - ✅ PSGC codes correctly formatted (string type with leading zeros)
      - ✅ Merge with admin CSV data working correctly
    - Concluded issue was NOT in `_build_reference_boundaries()` but in `apply_spatial_matching()`
    - Found mask recreation bug by reviewing update logic at line 648
    - Fix verified by reloading module and re-running section 2.1

  - **Files Modified**:
    - `modules/psgc_consolidator.py`: Fixed mask recreation bug in `apply_spatial_matching()`
    - `notebooks/0.2-map-resources.ipynb`: Added diagnostic cells in section 2.1
    - Updated module reload cell to use `importlib.reload(psgc_consolidator)` for testing

- **Session Summary**:
  - **Total bugs fixed**: 2 critical bugs in spatial matching
  - **Bug 1**: Reference boundaries inheriting NaN values from matched barangays
    - Fixed by merging with authoritative CSV sources after dissolving
  - **Bug 2**: Mask recreation inside for loop preventing column updates
    - Fixed by creating mask once before loop
  - **Result**: Complete spatial matching functionality
    - All ~3,580 unmatched barangays now have complete PSGC codes AND names
    - `is_spatially_matched` column properly identifies spatially matched rows
  - **Debugging methodology**: Added diagnostic cells to isolate issue location
  - **Testing approach**: Module reload and iterative testing in notebook

**Session 2 Summary**: Fixed NCR district mapping bug - all 4 NCR districts now properly tagged in consolidated data

- **NCR District Mapping Bug** (`modules/psgc_consolidator.py` - hierarchical joins)
  - **Problem Identified**: Only 1st District remaining in consolidated_data after joins
    - NCR has 4 districts (Adm2 level): 1st, 2nd, 3rd, 4th covering 17 cities
    - After consolidation, only 1st District data was retained
    - Other 3 districts (2nd, 3rd, 4th) were being lost during joins

  - **Root Cause**: Mismatched PSGC codes between Adm3 (cities) and Adm2 (districts)
    - **Adm3 CSV (cities)**: Each city uses its own city code as `adm2_psgc`
      - Manila: `adm2_psgc = 1380600000` (same as `adm3_psgc`)
      - Quezon City: `adm2_psgc = 1381300000`
      - Makati: `adm2_psgc = 1380300000`
      - Pattern: All 17 NCR cities have `adm2_psgc = adm3_psgc` (self-referential)
    - **Adm2 CSV (districts)**: Districts have different codes
      - 1st District (Capital): `adm2_psgc = 1303900000`
      - 2nd District (Eastern Manila): `adm2_psgc = 1307400000`
      - 3rd District (Camanava): `adm2_psgc = 1307500000`
      - 4th District (Southern): `adm2_psgc = 1307600000`
    - **When joining on `['adm1_psgc', 'adm2_psgc']`**: No matches because codes don't align

  - **Solution**: Create NCR city-to-district mapping applied before Adm2 join
    - New method: `_fix_ncr_district_codes()`
    - Maps all 17 NCR cities from self-referential codes to correct district codes
    - Applied in `consolidate_hierarchy()` after Adm3 join, before Adm2 join
    - Mapping structure:
      ```python
      ncr_city_to_district = {
          # 1st District - Capital District (1 city)
          '1380600000': '1303900000',  # Manila

          # 2nd District - Eastern Manila District (5 cities)
          '1380500000': '1307400000',  # Mandaluyong
          '1380700000': '1307400000',  # Marikina
          '1381200000': '1307400000',  # Pasig
          '1381300000': '1307400000',  # Quezon City
          '1381400000': '1307400000',  # San Juan

          # 3rd District - Camanava (4 cities)
          '1380100000': '1307500000',  # Caloocan
          '1380400000': '1307500000',  # Malabon
          '1380900000': '1307500000',  # Navotas
          '1381600000': '1307500000',  # Valenzuela

          # 4th District - Southern Manila District (7 cities/municipality)
          '1380200000': '1307600000',  # Las Piñas
          '1380300000': '1307600000',  # Makati
          '1380800000': '1307600000',  # Muntinlupa
          '1381000000': '1307600000',  # Parañaque
          '1381100000': '1307600000',  # Pasay
          '1381700000': '1307600000',  # Pateros
          '1381500000': '1307600000',  # Taguig
      }
      ```

  - **Implementation Details**:
    - Method detects NCR rows using `adm1_psgc == 1300000000`
    - Replaces city codes with district codes via dictionary mapping
    - Logs number of rows fixed and district distribution
    - Integrated into consolidation pipeline at line 317-318

  - **Updated Process Flow** (consolidate_hierarchy):
    1. Start with Adm4 (barangays) as base
    2. Join with Adm3 (municipalities/cities) on `[adm1_psgc, adm2_psgc, adm3_psgc]`
    3. **Fix NCR district codes** ← NEW STEP
    4. Join with Adm2 (provinces/districts) on `[adm1_psgc, adm2_psgc]`
    5. Join with Adm1 (regions) on `[adm1_psgc]`
    6. Fix City of Manila missing data
    7. Add leading zeros to PSGC codes
    8. Reorder columns

  - **Impact**: All 4 NCR districts now properly represented in consolidated_geodata
    - Complete district-level (Adm2) information for NCR
    - Enables proper analysis of NCR's administrative structure
    - All 17 cities correctly linked to their respective districts

  - **Files Modified**:
    - `modules/psgc_consolidator.py`: Added `_fix_ncr_district_codes()` method and integration
    - Module docstring updated to document NCR district mapping feature
    - `consolidate_hierarchy()` docstring updated with new step

  - **Follow-up Fix 1**: Corrected dtype handling in `_fix_ncr_district_codes()`
    - **Issue**: Initial implementation used string keys in mapping dict, causing dtype conflicts
      - FutureWarning: "Setting an item of incompatible dtype is deprecated"
      - Converting adm2_psgc to string early broke subsequent joins with Adm2 data (still int64)
    - **Solution**: Changed mapping to use integer keys and values
      - Mapping now works with native int64 dtype from CSV data
      - No premature type conversions - PSGC codes converted to string later in pipeline
      - Preserves compatibility with existing join operations
    - Result: Clean execution without warnings, proper district assignment

  - **Follow-up Fix 2**: Updated `_fix_city_of_manila()` to work with NCR district fix
    - **Issue**: "City of Manila" missing from `adm3_en` unique values in NCR
      - Original method required both `adm3_en` AND `adm2_en` to be NaN
      - After NCR district fix, `adm2_en` is now populated for Manila barangays
      - Condition `(df['adm2_en'].isna())` was False, preventing Manila fix from triggering
      - Result: ~897 Manila barangays had NaN in `adm3_en` column
    - **Solution**: Removed `adm2_en` check from mask
      - Now only checks if `adm3_en` is NaN for NCR barangays
      - Works correctly whether `adm2_en` is populated or not
      - Added clarifying comment about interaction with NCR district fix
    - Result: All 17 NCR cities now appear in `adm3_en` unique values, including City of Manila

  - **Follow-up Fix 3**: Fixed reference boundaries merge to populate NCR city names in spatial matching
    - **Issue**: NCR cities missing from `adm3_en` in spatially matched barangays
      - User reported that spatially matched NCR barangays had NaN in `adm3_en` column
      - Problem in `_build_reference_boundaries()` line 534
      - Merge with `adm3_data` used `['adm1_psgc', 'adm2_psgc', 'adm3_psgc']` as join keys
      - **Mismatch**: Reference boundaries have district codes (e.g., '1303900000') while `adm3_data` still has city codes (e.g., '1380600000')
      - NCR cities failed to match, leaving `adm3_en` as NaN in reference boundaries
    - **Root Cause**: `adm3_data` was never updated with district codes
      - Our NCR district fix only updated `consolidated_data` during hierarchical joins
      - Original `self.adm3_data` from CSV still has city codes in `adm2_psgc`
      - When building reference boundaries, merge on `adm2_psgc` fails for NCR
    - **Solution**: Changed merge to use only `['adm1_psgc', 'adm3_psgc']`
      - Removed `adm2_psgc` from join keys in adm3_names merge
      - `adm3_psgc` is already unique within a region, so adm2_psgc is redundant
      - Works for all regions, not just NCR
      - Added clarifying comment about NCR adm2_psgc mismatch
    - **Before (buggy)**:
      ```python
      municipalities = municipalities.merge(
          adm3_names,
          on=['adm1_psgc', 'adm2_psgc', 'adm3_psgc'],  # NCR fails here!
          how='left'
      )
      ```
    - **After (fixed)**:
      ```python
      municipalities = municipalities.merge(
          adm3_names,
          on=['adm1_psgc', 'adm3_psgc'],  # Works for all regions including NCR
          how='left'
      )
      ```
    - **Impact**:
      - Reference boundaries now correctly populated with NCR city names
      - Spatially matched NCR barangays get complete admin information
      - All 17 NCR cities appear in `adm3_en` for spatial matching results

  - **Follow-up Fix 4**: Implemented fuzzy matching for sub-municipality codes in Adm3 join
    - **Issue**: 899 NCR barangays missing city names in `consolidated_data`
      - User reported only 2 NCR barangays matched after Adm3 join
      - Comprehensive diagnostic revealed: 899 barangays with NO city name after Adm3 join
      - 15 unique `adm3_psgc` values NOT found in Adm3 CSV
      - Problem in `consolidate_hierarchy()` line 306 - Adm3 join
    - **Root Cause**: Sub-municipality codes don't exist in Adm3 CSV
      - Many barangays have `adm3_psgc` like `1303901000`, `1380601000` (sub-municipality codes)
      - Adm3 CSV only contains parent city codes like `1380600000` (City of Manila)
      - Exact match on `adm3_psgc` fails for these sub-municipality codes
      - **Example**: Manila districts
        - Barangay code: `1303901000` (Manila sub-municipality)
        - Adm3 CSV: `1380600000` (City of Manila parent)
        - First 6 digits: `130390` vs `138060` - no match!
    - **Solution**: Implemented two-stage Adm3 join with fuzzy matching fallback
      1. **Exact Match**: First try exact match on `['adm1_psgc', 'adm3_psgc']`
         - Matches 813 NCR barangays with standard city codes
      2. **Fuzzy Match**: For unmatched rows, match on first 6 digits of `adm3_psgc`
         - Create lookup: first 6 digits of `adm3_psgc` → city name
         - Extract first 6 digits from unmatched barangay `adm3_psgc`
         - Map to parent city using prefix lookup
         - Catches remaining 899 NCR barangays with sub-municipality codes
    - **Implementation** (lines 306-348):
      ```python
      # Join with Adm3 (Municipalities/Cities)
      # First, try exact match on [adm1_psgc, adm3_psgc]
      consolidated = consolidated.merge(
          self.adm3_data[['adm1_psgc', 'adm3_psgc', 'adm3_en']],
          on=['adm1_psgc', 'adm3_psgc'],
          how='left',
          suffixes=('', '_adm3')
      )

      # For unmatched rows, try fuzzy match on first 6 digits of adm3_psgc
      unmatched_mask = consolidated['adm3_en'].isna()
      if unmatched_mask.sum() > 0:
          # Create lookup: first 6 digits of adm3_psgc → city name
          adm3_lookup = self.adm3_data.copy()
          adm3_lookup['adm3_psgc_str'] = adm3_lookup['adm3_psgc'].astype(str).str.zfill(10)
          adm3_lookup['adm3_prefix'] = adm3_lookup['adm3_psgc_str'].str[:6]
          city_lookup = dict(zip(adm3_lookup['adm3_prefix'], adm3_lookup['adm3_en']))

          # Apply fuzzy match
          consolidated.loc[unmatched_mask, 'adm3_prefix'] = (
              consolidated.loc[unmatched_mask, 'adm3_psgc'].astype(str).str.zfill(10).str[:6]
          )
          consolidated.loc[unmatched_mask, 'adm3_en'] = (
              consolidated.loc[unmatched_mask, 'adm3_prefix'].map(city_lookup)
          )

          # Clean up temporary column
          consolidated.drop(columns=['adm3_prefix'], inplace=True, errors='ignore')
      ```
    - **Impact**:
      - All 1,712 NCR barangays now get city names (813 exact + 899 fuzzy)
      - All 17 NCR cities properly represented in `consolidated_data`
      - Complete NCR shape coverage in spatial matching results
      - Matched_gdf now shows entire NCR matching raw shapefile coverage

  - **Follow-up Fix 5**: Fixed overly broad City of Manila assignment in `_fix_city_of_manila()`
    - **Issue**: 1,316 barangays assigned to "City of Manila" (Manila only has 897)
      - User reported all NCR barangays with missing `adm3_en` were assigned to Manila
      - Original method assigned Manila to ANY NCR barangay with missing city name
    - **Root Cause**: Missing specificity check
      - Method only checked if region is NCR and `adm3_en` is NaN
      - Didn't verify if barangay is actually in Manila
      - After fuzzy match implementation, this fix became redundant but still needed correction
    - **Solution**: Changed to use district code identification (more reliable)
      - After NCR district fix, all Manila barangays have `adm2_psgc = 1303900000` (1st District)
      - Check district code instead of prefix matching on `adm3_psgc`
    - **Before (buggy)**:
      ```python
      mask = (
          (df['adm1_en'].astype('string').str.contains(r'capital', flags=2, na=False))
          & (df['adm3_en'].isna())
      )
      ```
    - **After (fixed)**:
      ```python
      mask = (
          (df['adm2_psgc'] == 1303900000)  # 1st District = Manila
          & (df['adm3_en'].isna())
      )
      ```
    - **Impact**:
      - Only actual Manila barangays get Manila assignment (899 barangays)
      - Precise city distribution across all 17 NCR cities
      - Fixed 899 Manila records (vs only 2 before)

  - **Follow-up Fix 6**: Discovered data source mismatch between CSV and shapefile
    - **Critical Discovery**: NCR CSV and shapefile have completely different PSGC codes
      - **CSV codes**: `1303901906`, `1303901907`, `1380100001`, `1380100002`, etc. (1,712 barangays)
      - **Shapefile codes**: `1303901001`, `1303901002`, `1303901003`, `1303901004`, etc. (1,712 geometries)
      - **Overlap**: Only 2 codes match (`1303901906`, `1303901907` - both Manila)
      - Different barangays or different PSGC versions between data sources
    - **Impact on Spatial Matching**:
      - Only 2 NCR barangays matched between CSV and shapefile
      - Reference boundaries built from 2 matched barangays only
      - Created single NCR municipality boundary: "City of Manila"
      - All 1,710 unmatched NCR geometries spatially assigned to Manila
      - Result: 1,316 barangays incorrectly labeled as Manila
    - **Root Cause**: Data source incompatibility (not a code bug)
    - **Diagnostic Output**:
      ```
      NCR in consolidated_data: 1712 (CSV)
      NCR in shapefile: 1712 (geometries)
      NCR matched (has adm1_psgc): 2
      Overlap: 2/1712 codes
      ```
    - **Solution Options**:
      1. Obtain matching versions of CSV and shapefile
      2. Use shapefile-only for NCR (ignore CSV hierarchical data)
      3. Accept incomplete NCR coverage with generic assignment

  - **Follow-up Fix 7**: Implemented Metro Manila generic assignment for NCR
    - **User Decision**: Accept generic "Metro Manila" assignment for all NCR barangays
      - Given CSV-shapefile mismatch is unfixable in code
      - User satisfied with region-level aggregation for NCR
    - **Solution**: Post-processing step in `apply_spatial_matching()`
      - After spatial matching completes
      - Identify all NCR barangays by `psgc_code` starting with `'13'`
      - Assign uniform values:
        - `adm2_en = 'National Capital Region (NCR)'`
        - `adm3_en = 'Metro Manila'`
    - **Implementation** (lines 822-832):
      ```python
      # Post-processing: Fix NCR barangays with generic Metro Manila assignment
      ncr_mask = self.consolidated_geodata['psgc_code'].str.startswith('13', na=False)
      ncr_count = ncr_mask.sum()

      if ncr_count > 0:
          logger.info(f"Post-processing: Assigning {ncr_count} NCR barangays to Metro Manila...")
          self.consolidated_geodata.loc[ncr_mask, 'adm2_en'] = 'National Capital Region (NCR)'
          self.consolidated_geodata.loc[ncr_mask, 'adm3_en'] = 'Metro Manila'
      ```
    - **Impact**:
      - All 1,712 NCR barangays now have consistent assignment
      - No misleading individual city names that aren't supported by data
      - Enables region-level analysis for NCR
      - User satisfied with this approach

  - **Follow-up Fix 8**: Retained shapefile pcode columns in merged output
    - **User Request**: Keep administrative boundary codes from shapefile
      - Columns: `adm1_pcode`, `adm2_pcode`, `adm3_pcode`, `adm4_pcode`
      - Provide alternative administrative coding system from shapefile source
    - **Solution**: Updated `_prepare_shapefile_for_merge()` to include pcode columns
    - **Implementation** (lines 461-466):
      ```python
      relevant_columns = ['psgc_code', 'corr_code', 'name', 'adm4_en',
                         'adm1_pcode', 'adm2_pcode', 'adm3_pcode', 'adm4_pcode',
                         'geometry']
      # Filter to only existing columns (in case some don't exist in shapefile)
      existing_columns = [col for col in relevant_columns if col in shapefile.columns]
      shapefile = shapefile[existing_columns]
      ```
    - **Impact**:
      - All pcode columns now available in `matched_gdf` output
      - Provides dual coding system: PSGC codes (from CSV) + pcode (from shapefile)
      - Useful for cross-referencing with other datasets using different coding systems

### 2025-10-07

**Session Summary**: Created Module 9 (Provincial Road Extractor) - lightweight PyOsmium-based solution for extracting provincial road networks from OSM PBF files

- **Module 9: Provincial Road Network Extractor** (`modules/provincial_road_extractor.py`)
  - **Purpose**: Extract provincial road networks from OpenStreetMap PBF files using memory-efficient streaming
  - **Problem Context**: Previous OSMnx approach was slow and memory-intensive for province-level extraction
  - **Key Innovation**: Uses `adm2_pcode` from consolidated geodata instead of unreliable PSGC codes
  - **Input**:
    - Consolidated geodata (.gpkg) from Module 7
    - OSM PBF file (581MB Philippines extract from GeoFabrik)
  - **Output**: One `.geojsonl` file per province
    - Filename format: `{adm2_pcode}_{province_name}.geojsonl`
    - Example: `PH03014_bulacan.geojsonl`, `PH04021_cavite.geojsonl`

  - **Architecture Components**:
    1. **ProvincialRoadExtractor (Main Class)**:
       - `extract_all_provinces()` - extracts all 88 provinces
       - `extract_provinces(whitelist)` - extracts specific provinces by adm2_pcode
       - `get_province_list()` - returns province metadata (pcode, name, filename)

    2. **LRUWriters (File Handle Cache)**:
       - Solves "too many open files" error when writing to 88 provinces simultaneously
       - Keeps max 16 files open, auto-closes least recently used
       - Prevents OS resource exhaustion

    3. **DriveHandler (PyOsmium Streaming Handler)**:
       - Processes OSM ways one at a time without loading entire file into memory
       - Spatial indexing with Shapely STRtree for fast intersection queries
       - Handles Shapely 1.x vs 2.x API differences (`query_items`, `query_bulk`, `query`)
       - Filters to driveable roads only (motorway, trunk, primary, secondary, etc.)

    4. **load_provinces() Function**:
       - Aggregates 42,048 barangays to 88 provinces using `adm2_pcode`
       - Extracts most common `adm2_en` for each province
       - Builds spatial index (STRtree) for fast intersection queries
       - Generates consistent filenames

  - **Performance**:
    - Processes entire Philippines (581MB PBF) in **~2.8 minutes**
    - Memory-efficient streaming (uses `sparse_mmap_array` index)
    - Outputs 88 separate `.geojsonl` files in one pass

  - **Features**:
    - `verbose` parameter: Controls logging (INFO vs WARNING only)
    - `do_clip` option: Clip roads at province boundaries (slower) vs intersect-only (faster)
    - `whitelist` parameter: Extract specific provinces for testing
    - Road metadata: Includes `osm_id`, `highway`, `name`, `oneway`, `maxspeed`

  - **Why adm2_pcode instead of PSGC?**
    - **Problem**: PSGC codes showed inconsistencies between CSV and shapefile (e.g., NCR had only 2 matching codes out of 1,712)
    - **Solution**: Use `adm2_pcode` from shapefile which exists consistently across all geometries
    - **Benefit**: Reliable province identification without post-processing rename steps

  - **Usage Example**:
    ```python
    from modules.provincial_road_extractor import ProvincialRoadExtractor

    # Initialize
    extractor = ProvincialRoadExtractor(
        consolidated_geodata_path="output/consolidated_geodata_matched.gpkg",
        pbf_path="data/networks/philippines-251002.osm.pbf",
        output_dir="output/province_road_networks",
        verbose=True
    )

    # Extract all provinces (~2.8 minutes)
    counts = extractor.extract_all_provinces()

    # Or extract specific provinces for testing
    counts = extractor.extract_provinces(whitelist={"PH03014", "PH04021"})

    # View available provinces
    provinces = extractor.get_province_list()
    ```

  - **Advantages over OSMnx approach**:
    - **Speed**: 20-30x faster (minutes vs hours for all provinces)
    - **Memory**: Constant low memory usage vs loading entire graphs
    - **Reliability**: No API timeouts or rate limits (works offline with PBF file)
    - **Consistency**: Same data source for all provinces (not dependent on OSM API state)
    - **Flexibility**: Easy to re-run with different filters or parameters

  - **Integration with Project**:
    - Uses consolidated geodata from Module 7 (PSGC Consolidator)
    - Complements Module 8 (Regional Road Network Extractor - deprecated in favor of this approach)
    - Province-level granularity matches school location analysis needs
    - GeoJSONL format easy to read into geopandas for further analysis

  - **Files Created**:
    - `modules/provincial_road_extractor.py`: Main extraction module (540 lines)
    - Notebook `0.4-get-road-networks-v2.ipynb`: Documents development and testing

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