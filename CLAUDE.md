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

### 8. Regional Road Network Extractor (`modules/regional_road_network_extractor.py`)
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

### 2025-10-05 (Current Session)

**Summary**: Enhanced Module 7 with spatial matching for unmatched barangays, shifted road network extraction to PyOsmium architecture, configuration cleanup

- **Module 7 Enhancement: Spatial Matching** (`modules/psgc_consolidator.py`)
  - **Problem**: ~3,580 unmatched barangays in consolidated GeoDataFrame
    - Shapefile contains 42,048 features (PH_Adm4_BgySubMuns.shp.zip)
    - CSV contains 42,017 barangays (PH_Adm4_BgySubMuns.csv)
    - PSGC code mismatches cause NaN values in admin columns (adm1_psgc, adm2_psgc, adm3_psgc, region/province/municipality names)
    - Root causes: Renamed barangays, merged/split administrative units, data vintage differences

  - **Solution**: Point-in-polygon spatial matching using STRtree
    - Dissolves matched barangays to municipality-level reference boundaries
    - Uses centroid-based containment testing for unmatched barangays
    - Falls back to nearest neighbor for boundary cases
    - Performance: ~1-2 minutes for 3,580 unmatched barangays
    - Results in complete dataset with no NaN admin codes

  - **New Class Attributes**:
    - `consolidated_geodata_original`: Stores pre-matching data (with NaN rows)
    - `reference_boundaries`: Dissolved municipality boundaries for spatial queries

  - **New Methods**:
    - `_build_reference_boundaries()`: Dissolves matched barangays by [adm1_psgc, adm2_psgc, adm3_psgc] to create municipality polygons
    - `_spatial_match_unmatched(unmatched_gdf, reference_gdf)`: STRtree-based spatial matching
      - Builds spatial index (STRtree) from reference municipality geometries
      - Prepares geometries for optimized containment testing
      - For each unmatched barangay:
        1. Query spatial index with barangay centroid
        2. Test centroid containment in candidate municipalities
        3. If no match, use nearest neighbor fallback
      - Returns DataFrame with matched admin codes
    - `apply_spatial_matching(save_original=True)`: Main public method
      - Saves original GeoDataFrame before matching (if requested)
      - Builds reference boundaries from matched barangays
      - Matches unmatched barangays to admin units
      - Updates admin code columns (adm1_psgc, adm2_psgc, adm3_psgc, adm1_en, adm2_en, adm3_en)
      - Adds `is_spatially_matched` boolean column for transparency
      - Returns updated consolidated GeoDataFrame
    - `export_original(output_path)`: Export pre-matching data with NaN rows
    - `export_matched(output_path)`: Export post-matching data with is_spatially_matched column

  - **Updated Methods**:
    - `process(auto_spatial_match=False)`: Added optional automatic spatial matching
      - If `auto_spatial_match=True`, calls `apply_spatial_matching()` after consolidation
      - Defaults to `False` to preserve original behavior

  - **Shapely 2.x Compatibility Fix**:
    - **Problem**: KeyError when using `geom_to_idx` mapping in spatial matching
    - **Root Cause**: Shapely 2.x STRtree.query() returns indices directly (not geometry objects)
    - **Fix**: Removed geom_to_idx mapping, use returned indices directly
    - **Before**: `cand_idx = geom_to_idx[id(cand_geom)]` (incorrect for Shapely 2.x)
    - **After**: `candidates_idx = tree.query(centroid)` (indices already returned)

  - **Updated Module Docstring**:
    - Added spatial matching step (step 8) to consolidation process documentation
    - Added "Spatial Matching" section explaining the approach
    - Updated example usage with two patterns:
      1. Manual control: Call `apply_spatial_matching()` explicitly
      2. Automatic: Use `process(auto_spatial_match=True)`

  - **Usage Example**:
    ```python
    # Pattern 1: Manual control
    consolidator = PSGCConsolidator(base_dir='data/philippines-psgc-shapefiles/dist')
    consolidated = consolidator.process()

    # Apply spatial matching
    matched = consolidator.apply_spatial_matching(save_original=True)

    # Export both versions
    consolidator.export_original('output/original_with_nans.gpkg')
    consolidator.export_matched('output/matched_complete.gpkg')

    # Pattern 2: Automatic
    consolidator = PSGCConsolidator(base_dir='data/philippines-psgc-shapefiles/dist')
    consolidated = consolidator.process(auto_spatial_match=True)

    # Check spatially matched rows
    matched_rows = consolidated[consolidated['is_spatially_matched'] == True]
    print(f"Spatially matched: {len(matched_rows)} barangays")
    ```

- **Road Network Extraction: Architecture Shift to PyOsmium**
  - **Problem**: OSMNx provincial breakdown showed fundamental limitations
    - Overpass API returns incomplete/different data for small provincial queries
    - Provincial breakdown shows lower road network density in central areas
    - Node merging and edge preservation cannot fix incomplete source data
    - Direct regional queries miss islands in archipelagic regions

  - **New Approach**: Local PBF processing with PyOsmium (`0.4-get-road-networks-v2.ipynb`)
    - Downloads Philippines PBF from GeoFabrik (~581MB)
    - Single-pass streaming handler processes entire PBF file
    - Performance: 2.79 minutes for all 88 provinces (vs hours with OSMNx provincial breakdown)
    - Output: Provincial .geojsonl files with complete road network coverage

  - **Implementation Details**:
    - Uses PyOsmium's osmium.SimpleHandler for memory-efficient streaming
    - Builds STRtree spatial index from province boundaries
    - For each OSM way:
      1. Checks if highway tag is drivable (excludes footpaths, etc.)
      2. Queries STRtree to find intersecting province(s)
      3. Adds way to province's road network
    - Outputs GeoJSONL format (line-delimited GeoJSON) for efficient processing

  - **File Naming Convention**: PSGC format `RR-PPP.geojsonl`
    - RR = 2-digit region code (e.g., '03' = Region III)
    - PPP = 3-digit province code (e.g., '014' = Bulacan)
    - Example: `03-014.geojsonl` (Bulacan, Region III)
    - Enables hierarchical organization and filtering by region/province

  - **Renaming Script**: Created `rename_to_psgc()` function in notebook
    - Maps slug-based filenames to PSGC format
    - Example: `region_iii_central_luzon_bulacan.geojsonl` → `03-014.geojsonl`
    - Dry-run mode to preview changes before execution
    - See `rename_networks_cells.txt` for implementation

  - **Advantages over OSMNx**:
    - Complete coverage: Single PBF contains all Philippine road data
    - Consistent quality: No API limitations or query size restrictions
    - Performance: 20-40x faster than provincial breakdown with OSMNx
    - Reliability: No network calls, fully offline processing
    - Flexibility: Full control over filtering and processing logic

- **Configuration Cleanup**
  - **Removed**: 'reconciled' directory paths and references
    - Removed `"reconciled": "data/reconciled"` from `config/config.json`
    - Removed 'reconciled' from auto-created directories list in `config/config.py`
    - Decision: Not proceeding with PSGC reconciler approach (Module 8 from Oct 1 session)

- **File Reversion and Re-application**
  - **Problem**: Accidentally reverted psgc_consolidator.py and CLAUDE.md to earlier versions
    - All spatial matching code disappeared from psgc_consolidator.py
    - Today's documentation (2025-10-05) missing from CLAUDE.md
  - **Solution**: Re-applied all changes systematically
    - Restored complete spatial matching implementation to psgc_consolidator.py
    - Updated CLAUDE.md with 2025-10-05 session documentation

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