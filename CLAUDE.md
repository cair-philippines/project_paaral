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
  - **Region Standardization**: Cleans and standardizes region names
  - **Curricular Offering Mapping** (2025-10-29): Maps modified_coc values to standard categories
    - 'Purely ES', 'Purely JHS', 'Purely SHS'
    - 'ES and JHS', 'JHS with SHS'
    - 'All Offering' (K-12 complete)
    - Handles 40+ variations including misspellings
  - Expected valid coordinates: ~95%+ (up from ~86%)
- **Streamlined Structure** (2025-10-29): Removed 11 redundant methods, kept only essential functionality

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

### 10. Facilities Data Preprocessor (`modules/facilities_preprocessor.py`)
- **Source**: `data/public/facilities_2023-24.csv`
- **Output**: Columns 1-12 only (school metadata + classroom counts)
  - `school_id`, `sector`, `school_management` (string dtype)
  - `offers_es`, `offers_jhs`, `offers_shs` (boolean)
  - Classroom counts by level and type (float, nullable): `es/jhs/shs_classrooms_instructional`, `es/jhs/shs_classrooms_non_instructional`
- **Data Coverage**: 60,167 schools (Public and Private sectors)
- **Key Features**:
  - **Blank value handling**: All blank/empty classroom values explicitly converted to NaN
  - **String dtype conversion**: Uses pandas `'string'` dtype (not `object`) for proper string representation
  - **Validation**: Checks for duplicate school IDs, negative classroom counts, consistency between offered levels and classroom data
  - **Whitespace trimming**: Strips whitespace from string columns
- **Data Characteristics**:
  - Private schools: All classroom columns are NaN (no data reported)
  - Public schools: NaN in classroom columns when school doesn't offer that education level
- **Usage**:
  ```python
  from modules.facilities_preprocessor import FacilitiesPreprocessor

  processor = FacilitiesPreprocessor(file_path='data/public/facilities_2023-24.csv')
  facilities_data = processor.process()

  # Get summary statistics
  summary = processor.get_summary()

  # Export
  processor.export_csv('output/facilities_classrooms.csv')
  ```

### 11. Node Table Builder (`modules/node_table_builder.py`)
- **Purpose**: Consolidate school data from multiple sources into comprehensive GeoDataFrame node tables ready for graph network analysis
- **Created**: 2025-11-12 (Refactored from notebook 0.6)
- **Output**: GeoDataFrames with Point geometries, admin boundaries, and validation
  - `public_node_table` - Public schools (~47K)
  - `private_node_table` - Private schools (~11.8K)
  - `combined_node_table` - All schools with sector column
- **Key Features**:
  - **Spatial Integration**:
    - Point geometries from lat/lon (EPSG:4326)
    - Spatial join with PSGC consolidated geodata (Module 7)
    - Administrative boundary assignment: region, province, municipality
    - `adm2_pcode` column for direct matching with provincial road networks (Module 9)
  - **Data Consolidation**:
    - Public workflow: Coordinates → Enrollment → Facilities → Seats
    - Private workflow: Coordinates → GASTPE → Furniture → Enrollment
    - Sequential merging with validation flag creation (`has_*_data`)
  - **Tiered Validation**:
    - Level 1 (required): `school_id`, `coordinates_valid`, `geometry`, `admin_assignment_valid`
    - Level 2 (core): Level 1 + (enrollment OR facilities OR GASTPE data)
    - Level 3 (complete): Level 2 + all data sources present
    - Configurable validation level via `validation_level` parameter
  - **Computed Metrics** (for graph node weights):
    - `total_enrollment` - Sum of ES + JHS + SHS enrollment
    - `total_seats` - Sum of ES + JHS + SHS seats
    - `capacity_utilization` - Enrollment/seats ratio
  - **Quality Reporting**:
    - `get_summary()` - Comprehensive statistics for both sectors
    - `get_validation_report()` - Detailed list of validation failures
    - Completeness percentages by data source
    - Spatial coverage metrics
  - **Multiple Export Formats**:
    - GeoPackage (`.gpkg`) - Primary format, preserves geometry + attributes
    - CSV (`.csv`) - Non-spatial format
    - Parquet (`.parquet`) - Memory-efficient format
    - Quality report (`.csv`) - Validation and completeness metrics
  - **Graph-Ready Output**:
    - All required attributes for network analysis in single file
    - Provincial filtering via `adm2_pcode` for subgraph generation
    - Sector column enables public/private/mixed network analysis
- **Integration Points**:
  - **Input**: Uses all preprocessor modules (1-6, 10) + PSGC geodata (Module 7)
  - **Output for Notebook 1.0**: Node tables with geometry, attributes, and validation
  - **Provincial Road Networks**: `adm2_pcode` matches filenames from Module 9 (e.g., `PH03014_bulacan.geojsonl`)
- **Usage**:
  ```python
  from modules.node_table_builder import NodeTableBuilder

  builder = NodeTableBuilder(
      verbose=True,
      psgc_geodata_path='output/consolidated_geodata_matched.gpkg',
      validation_level='complete'  # 'required', 'core', or 'complete'
  )

  # Build node tables (returns GeoDataFrame)
  public_gdf = builder.build_public_node_table()
  private_gdf = builder.build_private_node_table()
  all_schools_gdf = builder.build_combined_node_table()

  # Get summaries
  summary = builder.get_summary()
  validation_report = builder.get_validation_report()

  # Export for graph generation
  builder.export_geopackage('output/all_nodes.gpkg', sector='both')
  builder.export_geopackage('output/all_nodes_valid.gpkg', sector='both', valid_only=True)
  builder.export_quality_report('output/data_quality_report.csv')
  ```
- **Key Methods**:
  - `build_public_node_table()` - Build public school nodes
  - `build_private_node_table()` - Build private school nodes
  - `build_combined_node_table()` - Build combined public + private
  - `get_summary()` / `get_public_summary()` / `get_private_summary()` - Statistics
  - `get_validation_report()` - Detailed validation issues
  - `export_geopackage()` / `export_csv()` / `export_parquet()` - Export methods
  - `export_quality_report()` - Export quality metrics
- **Performance**: ~800-1000 lines, comprehensive logging, caching of intermediate results

### 12. Provincial Network Builder (`modules/provincial_network_builder.py`)
- **Purpose**: Build road network graphs and distance matrices for a single province using ARM-compatible libraries
- **Created**: 2025-11-13
- **ARM Compatibility**: Uses NetworkX (pure Python) instead of igraph for ARM Windows support
- **Output**: Distance graph + Beneficiary graph + Distance matrix + Summary statistics
- **Performance**: Provincial scale (500-1000 schools) takes ~30 seconds - 2 minutes for distance computation
- **Note**: Module 12.1 (scipy-optimized) provides ~10x speedup - recommended for production use

### 12.1. Provincial Network Builder (SciPy-Optimized) (`modules/provincial_network_builder_scipy.py`)
- **Purpose**: Speed up distance matrix computation by ~10x using scipy.sparse.csgraph
- **Created**: 2025-11-13 (same day as Module 12)
- **Performance**: ~10x faster than Module 12 (NetworkX) - provincial scale takes **30-60 seconds** vs 5-10 minutes
- **Output**: Same format as Module 12 (drop-in replacement)
- **Key Innovation**: Uses scipy.sparse.csgraph.dijkstra() instead of NetworkX loops
  - Cython-optimized C code (not pure Python)
  - Vectorized operations (computes all distances at once)
  - Sparse matrix representation (LIL for construction, CSR for computation)
  - No multiprocessing needed (scipy already optimized)
- **API Changes from Module 12**:
  - Class name: `ProvincialNetworkBuilderSciPy` (vs `ProvincialNetworkBuilder`)
  - Removed `n_processes` parameter (not needed)
  - Same initialization and output format
- **When to Use**: Production use (almost always) - only use Module 12 for debugging or path reconstruction needs
- **Limitations**:
  - No path reconstruction (distances only, unless `return_predecessors=True`)
  - Memory spike during NetworkX → scipy conversion (not issue at provincial scale)
  - Less flexible than NetworkX (numeric edge weights only)
- **Key Features**:
  - **NetworkX-Only Architecture**:
    - Uses NetworkX for graph operations (no igraph dependency)
    - scipy.spatial.cKDTree for spatial indexing (not rtree)
    - Pure Python libraries compatible with ARM architecture
    - Parallel processing with multiprocessing (standard library)
  - **Dual Graph Output**:
    - Distance Graph: Nodes = schools, Edges = road distances (meters)
    - Beneficiary Graph: Nodes = schools, Edges = student flow counts
    - Both graphs share same vertices for interchangeable analysis
  - **Dual CRS Strategy**:
    - EPSG:3123 (PRS92 Philippines) for distance calculations
    - EPSG:4326 (WGS84) for visualization and storage
  - **Provincial Scope**:
    - Single province at a time for manageable computation
    - Filters schools and beneficiary flows to province
    - Includes cross-provincial flows (external origins/destinations)
  - **Regional Merging Design**:
    - Node coordinates stored with 5 decimal precision (~1 meter)
    - Province code tagged on all nodes/edges
    - Boundary nodes identified for cross-provincial connections
    - GraphML export preserves all attributes for merging
  - **Parallel Distance Computation**:
    - Multiprocessing for school-to-school distance calculations
    - Buffer-based spatial search (default: 5km radius)
    - Maximum distance cutoff (default: 15km)
    - NetworkX single-source Dijkstra for shortest paths
- **Workflow**:
  1. Load provincial road network from GeoJSONL (Module 9 output)
  2. Snap schools to nearest road nodes using KDTree
  3. Build spatial index for fast proximity queries
  4. Compute distance matrix (parallel processing)
  5. Build distance and beneficiary NetworkX graphs
  6. Identify boundary nodes for regional merging
- **Input Requirements**:
  - Province-filtered public/private node tables (from Module 11)
  - Beneficiary edges (from BeneficiaryProcessor)
  - Provincial road network GeoJSONL (from Module 9)
  - Consolidated geodata (optional, for boundary identification)
- **Outputs**:
  - Distance matrix CSV: Origin × Destination road distances (meters)
  - Distance graph GraphML: NetworkX graph with road distance edges
  - Beneficiary graph GraphML: NetworkX graph with student flow edges
  - Summary JSON: Comprehensive statistics
- **Performance**:
  - Provincial scale (500-1000 schools): ~30 seconds - 2 minutes for distance computation
  - NetworkX 2-5x slower than igraph but acceptable for provincial scope
  - Memory efficient with streaming spatial queries
- **Usage**:
  ```python
  from modules.provincial_network_builder import ProvincialNetworkBuilder

  # Filter data to province
  province_code = 'PH03014'  # Bulacan
  public_province = public_nodes[public_nodes['adm2_pcode'] == province_code]
  private_province = private_nodes[private_nodes['adm2_pcode'] == province_code]

  # Initialize builder
  builder = ProvincialNetworkBuilder(
      province_code='PH03014',
      province_name='bulacan',
      public_nodes_gdf=public_province,
      private_nodes_gdf=private_province,
      beneficiary_edges_df=beneficiary_edges,
      road_network_path='output/province_road_networks/PH03014_bulacan.geojsonl',
      consolidated_geodata_path='output/consolidated_geodata_matched.gpkg'
  )

  # Build complete network
  results = builder.build_complete_network(
      buffer_distance_m=5000,
      max_distance_km=15,
      n_processes=4
  )

  # Access results
  distance_matrix = results['distance_matrix']
  distance_graph = results['distance_graph']
  beneficiary_graph = results['beneficiary_graph']

  # Export all
  builder.export_all('output/provincial_networks')
  ```
- **Key Methods**:
  - `build_complete_network()` - Execute complete workflow
  - `_load_road_network()` - GeoJSONL → NetworkX conversion
  - `_snap_schools_to_network()` - KDTree-based school snapping
  - `_build_spatial_index()` - KDTree for proximity queries
  - `_compute_distance_matrix()` - Parallel distance computation
  - `_build_distance_graph()` - Create distance NetworkX graph
  - `_build_beneficiary_graph()` - Create beneficiary NetworkX graph
  - `_identify_boundary_nodes()` - Find nodes near province boundary
  - `export_all()` - Export all results to directory
- **Integration**:
  - Input: Uses Module 11 (node tables) + BeneficiaryProcessor outputs
  - Input: Uses Module 9 (provincial road networks)
  - Output: GraphML graphs ready for regional merging (Module 13, future)
  - Output: Distance matrices ready for discrete choice modeling
- **Next Steps**: Module 13 will merge provincial networks into regional graphs

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

### 2025-10-14

**Session Summary**: Created Module 10 (Facilities Data Preprocessor) - extracts classroom counts and school metadata from comprehensive facilities dataset

- **Module 10: Facilities Data Preprocessor** (`modules/facilities_preprocessor.py`)
  - **Data Source**: `data/public/facilities_2023-24.csv` (60,167 schools)
  - **Scope**: Columns 1-12 only (school metadata + classroom infrastructure)
  - **Objective**: Extract classroom capacity data for integration with enrollment and coordinate datasets

  - **Data Structure**:
    - **Metadata columns** (3): `school_id`, `sector`, `school_management`
    - **Education level flags** (3): `offers_es`, `offers_jhs`, `offers_shs` (boolean)
    - **Classroom counts** (6): Instructional and non-instructional classrooms by education level (ES/JHS/SHS)

  - **Key Features**:
    1. **String dtype handling**: Explicitly converts string columns to pandas `'string'` dtype (not `object`)
       - Ensures `.dtypes` displays `string` instead of generic `object`
       - Uses `astype('string')` for proper pandas nullable string representation
    2. **NaN value handling**: All blank classroom values explicitly converted to NaN
       - Private schools: All classroom columns are NaN (no data reported)
       - Public schools: NaN when school doesn't offer that education level
    3. **Data validation**:
       - Checks for duplicate school IDs
       - Validates non-negative classroom counts
       - Checks consistency (schools not offering a level shouldn't have classroom data)
    4. **Whitespace trimming**: Strips whitespace from string columns for clean data
    5. **Removed sector validation**: Accepts all sector values without warnings (not just "Public"/"Private")

  - **Default file path behavior**: Constructor accepts optional `file_path` parameter
    - If None: defaults to `'data/public/facilities_2023-24.csv'`
    - Enables flexible usage across notebooks and scripts

  - **Processing pipeline**:
    1. Load CSV with `low_memory=False` for proper dtype handling
    2. Select columns 1-12
    3. Convert data types (string → `'string'`, boolean, numeric)
    4. Handle blank values as NaN
    5. Validate data quality
    6. Trim whitespaces

  - **Integration potential**:
    - Combines with Module 1 (Enrollment) for capacity vs demand analysis
    - Links with Module 2 (Public Coordinates) via `school_id`
    - Enables classroom shortage/surplus calculations by education level

  - **Files created**:
    - `modules/facilities_preprocessor.py`: Main processor (~380 lines)
    - Follows established module patterns (load→process→validate→export)

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

### 2025-10-29

**Session Summary**: Enhanced Module 3 (Private School Coordinates) with curricular offering mapping and module streamlining

- **Module 3 Enhancement: Curricular Offering Mapping** (`modules/private_coordinates_processor.py`)
  - **Requirement**: User provided comprehensive mapping for `modified_coc` column
    - Maps 40+ variations to 6 standardized categories
    - Handles misspellings (e.g., "KINDEGARTEN" → "Purely ES")
    - Handles formatting variations (trailing spaces, commas, etc.)

  - **New Method**: `map_curricular_offerings()`
    - **Standard Categories**:
      - `'Purely ES'` - Elementary School only (includes Kindergarten/Preschool)
      - `'Purely JHS'` - Junior High School only
      - `'Purely SHS'` - Senior High School only
      - `'ES and JHS'` - Elementary and Junior High School
      - `'JHS with SHS'` - Junior High School and Senior High School
      - `'All Offering'` - Complete K-12 (ES, JHS, and SHS)
    - **Mapping Examples**:
      - "Kindergarten", "K TO G6", "KINDERGARTEN", "KINDEGARTEN" → "Purely ES"
      - "K TO JHS", "Elementary and JHS", "ES & JHS" → "ES and JHS"
      - "K TO SHS", "ES,JHS and SHS", "K, ES, JHS, SHS" → "All Offering"
      - "JHS and SHS", "JHS, SHS" → "JHS with SHS"
    - **Logging**: Reports original vs standardized unique values and distribution
    - **NaN handling**: NaN values explicitly mapped to None (missing data)

  - **Integration**: Method added to standard processing pipeline
    - Typically called after coordinate validation and region standardization
    - Column check: Verifies `modified_coc` exists before mapping

- **Module 3 Streamlining: Code Cleanup**
  - **Problem**: Module had 25+ methods, many redundant or rarely used
    - Too many simple getter methods (just attribute access)
    - Overly detailed summary methods
    - Redundant validation method (`validate_coordinates()` when `validate_coordinates_with_reasons()` was superior)

  - **Methods Removed** (11 total):
    1. `get_raw_data()` - Redundant (direct attribute access: `processor.raw_data`)
    2. `get_processed_data()` - Redundant (direct attribute access: `processor.processed_data`)
    3. `get_file_summary()` - Overly detailed, rarely used
    4. `get_sheet_data()` - Too granular, rarely used
    5. `list_files()` - Simple attribute access
    6. `list_sheets()` - Simple attribute access
    7. `get_failed_sheets()` - Simple attribute access (`processor.failed_sheets`)
    8. `get_successful_sheets()` - Simple attribute access (`processor.successful_sheets`)
    9. `get_data_quality_summary()` - Overly detailed, rarely used
    10. `validate_coordinates()` - Redundant (superseded by `validate_coordinates_with_reasons()`)
    11. Duplicate `validate_coordinates()` definition - Bug fix

  - **Methods Kept** (14 essential):
    - **Core**: `process()`, `export_processed()`, `get_summary()`
    - **Data Cleaning**: `clean_coordinates()`, `replace_unclean_region_values()`, `map_curricular_offerings()`
    - **Validation**: `validate_coordinates_with_reasons()`
    - **Excel Reading**: `read_all_files()`, `_select_optimal_engine()`, `_is_engine_available()`
    - **Engine-Specific**: `_get_sheet_names_optimized()`, `_read_excel_optimized()`, `_read_with_calamine()`, `_read_with_fastexcel()`, `_read_with_openpyxl()`
    - **Validation Helpers** (private): `_find_coordinate_columns()`, `_clean_single_coordinate()`, `_reconstruct_split_coordinates()`, `_validate_single_coordinate()`, `_validate_coordinate_column()`, `_validate_column_as_coordinate()`

  - **get_summary() Simplified**:
    - Removed detailed lists (`successful_sheet_details`, `failed_sheet_details`)
    - Kept only essential statistics: file counts, success rate, dataset dimensions
    - Users can access raw lists via attributes if needed: `processor.successful_sheets`, `processor.failed_sheets`

  - **Result**: Module reduced from ~1,350 lines to ~1,260 lines (~90 lines removed)
    - More focused and maintainable
    - Clearer API with only essential methods
    - Better performance (fewer method calls)

- **Documentation Updates**:
  - **Module Docstring**: Updated to reflect streamlined structure
    - Clear list of main methods
    - Removed mention of deprecated methods
    - Updated "Updated" date to 2025-10-29
  - **Example Usage**: Enhanced to demonstrate all key methods
    - Added region standardization step
    - Added curricular offering mapping step
    - Shows complete processing pipeline
  - **CLAUDE.md**: Updated Module 3 description with new features

- **Convenience Method Added**: `process_and_clean_all()`
  - **Purpose**: One-call complete pipeline from raw Excel to clean CSV
  - **Steps executed**:
    1. Read and process Excel files
    2. Clean coordinate values
    3. Validate coordinates with detailed error reasons
    4. Standardize region values
    5. Map curricular offerings
    6. Optionally export to CSV
  - **Parameters**:
    - `export_path` (optional): Output CSV path
    - `engine` (optional): Excel reading engine
    - `use_read_only` (optional): Use read_only mode (default: True)

- **Usage Pattern** (Updated):
  ```python
  from modules.private_coordinates_processor import PrivateSchoolsProcessor

  # OPTION 1: One-call pipeline (Recommended)
  processor = PrivateSchoolsProcessor('data/private/raw_validation_sheets')
  data = processor.process_and_clean_all('output/private_schools_clean.csv')

  # Get summary
  summary = processor.get_summary()
  print(f"Processed {summary['total_files_processed']} files")
  print(f"Success rate: {summary['success_rate']:.1f}%")

  # OPTION 2: Manual step-by-step (for fine-grained control)
  processor = PrivateSchoolsProcessor('data/private/raw_validation_sheets')
  data = processor.process()
  validated_data = processor.validate_coordinates_with_reasons(clean_first=True)
  processor.replace_unclean_region_values()
  processor.map_curricular_offerings()
  processor.export_processed('output/private_schools_clean.csv')
  ```

- **Files Modified**:
  - `modules/private_coordinates_processor.py`: Added `map_curricular_offerings()`, `process_and_clean_all()`, removed 11 methods, updated docs
  - `CLAUDE.md`: Updated Module 3 description and session history

- **Summary of Changes**:
  - ✅ Added curricular offering mapping with 40+ variations handled
  - ✅ Removed 11 redundant methods (90 lines of code)
  - ✅ Added convenience method `process_and_clean_all()` for one-call pipeline
  - ✅ Updated all documentation and examples
  - 📉 Module size: ~1,350 lines → ~1,310 lines (more focused and maintainable)

### 2025-11-12

**Session Summary**: Refactored notebook 0.6 node table creation code into Module 11 (NodeTableBuilder) with enhanced spatial integration for graph generation

- **Module 11: Node Table Builder** (`modules/node_table_builder.py`)
  - **Objective**: Consolidate school data from multiple sources into graph-ready GeoDataFrames
  - **Refactoring from**: Notebook 0.6 (scattered cell-based logic → cohesive module)

  - **Key Enhancements Over Original Notebook**:
    1. **Spatial Integration** (NEW):
       - GeoDataFrame output with Point geometries (EPSG:4326)
       - Spatial join with PSGC consolidated geodata (Module 7)
       - Administrative boundary assignment: `region`, `province`, `municipality`
       - `adm2_pcode` column for direct matching with provincial road networks (Module 9)
       - Enables provincial filtering for subgraph generation

    2. **Tiered Validation System** (ENHANCED):
       - Level 1 (required): `school_id`, `coordinates_valid`, `geometry`, `admin_assignment_valid`
       - Level 2 (core): Level 1 + (enrollment OR facilities OR GASTPE)
       - Level 3 (complete): Level 2 + all data sources present
       - Configurable via `validation_level` parameter
       - Replaces simple `all_valid` boolean with detailed validation breakdown

    3. **Computed Metrics for Graph Weights** (NEW):
       - `total_enrollment` - Sum of ES + JHS + SHS
       - `total_seats` - Sum of ES + JHS + SHS
       - `capacity_utilization` - Enrollment/seats ratio
       - Ready for use as node attributes in NetworkX graphs

    4. **Enhanced Reporting** (NEW):
       - `get_summary()` - Comprehensive statistics (validation breakdown, completeness by source, spatial coverage)
       - `get_validation_report()` - Detailed list of validation failures with reasons
       - `export_quality_report()` - CSV export of quality metrics

    5. **Multiple Export Formats** (NEW):
       - GeoPackage (`.gpkg`) - Primary format, preserves geometry + CRS
       - CSV (`.csv`) - Non-spatial format
       - Parquet (`.parquet`) - Memory-efficient format
       - Quality report - Validation and completeness metrics

    6. **Code Organization** (IMPROVED):
       - Reusable methods: `_merge_with_validation()`, `_pivot_by_education_level()`, `_create_geometry_column()`
       - Lazy loading with caching (preprocessor results cached to avoid redundant reads)
       - Clear separation: public workflow, private workflow, combined workflow, spatial utilities
       - Comprehensive docstrings with usage examples

  - **Architecture**:
    ```
    NodeTableBuilder
    ├── Public Workflow
    │   └── build_public_node_table()
    │       ├── Load coordinates (Module 2)
    │       ├── Load enrollment (Module 1)
    │       ├── Load facilities (Module 10)
    │       ├── Load seats (Module 4)
    │       ├── Create geometry
    │       ├── Assign admin boundaries (Module 7)
    │       └── Validate
    ├── Private Workflow
    │   └── build_private_node_table()
    │       ├── Load coordinates (Module 3)
    │       ├── Load GASTPE (Module 6)
    │       ├── Load furniture (Module 5)
    │       ├── Load enrollment (Module 1)
    │       ├── Create geometry
    │       ├── Assign admin boundaries (Module 7)
    │       └── Validate
    ├── Combined Workflow
    │   └── build_combined_node_table()
    │       └── Merge public + private with 'sector' column
    └── Export Methods
        ├── export_geopackage()
        ├── export_csv()
        ├── export_parquet()
        └── export_quality_report()
    ```

  - **Integration for Graph Generation (Notebook 1.0)**:
    - **Problem Solved**: Original notebook 0.6 created DataFrames without geometry or admin boundaries
    - **Solution**: Module 11 creates graph-ready GeoDataFrames with all spatial attributes in place
    - **Benefits**:
      - No spatial joins needed in notebook 1.0 (already done)
      - Direct provincial filtering via `adm2_pcode` (matches Module 9 road network filenames)
      - Node attributes ready for graph algorithms (enrollment, capacity, utilization)
      - Clean separation: data preparation (Module 11) vs graph analysis (Notebook 1.0)

  - **Usage Example** (from Notebook 0.7):
    ```python
    from modules.node_table_builder import NodeTableBuilder

    builder = NodeTableBuilder(
        verbose=True,
        psgc_geodata_path='output/consolidated_geodata_matched.gpkg',
        validation_level='complete'
    )

    # Build all node tables
    public_gdf = builder.build_public_node_table()
    private_gdf = builder.build_private_node_table()
    all_schools_gdf = builder.build_combined_node_table()

    # Export for graph generation
    builder.export_geopackage('output/all_nodes_valid.gpkg', sector='both', valid_only=True)

    # Example: Filter to Bulacan province for provincial graph
    bulacan_schools = all_schools_gdf[all_schools_gdf['adm2_pcode'] == 'PH03014']
    # Load corresponding road network: PH03014_bulacan.geojsonl
    ```

- **Notebook 0.7: Node Tables - Refined Module Approach** (`notebooks/0.7-node-tables-refined.ipynb`)
  - **Purpose**: Demonstrate comprehensive usage of NodeTableBuilder module
  - **Structure**:
    - Section 1: Build node tables (public, private, combined)
    - Section 2: Data quality review (summaries, validation reports, visualizations)
    - Section 3: Export to multiple formats
    - Section 4: Provincial filtering example (Bulacan)
  - **Visualizations**:
    - Validation levels breakdown (bar charts)
    - Data completeness by source (horizontal bar charts)
    - Spatial coverage maps (coordinate distribution)
    - Capacity utilization histograms
  - **Outputs Generated**:
    - `output/public_nodes.gpkg` - Public school nodes
    - `output/private_nodes.gpkg` - Private school nodes
    - `output/all_nodes.gpkg` - Combined nodes (all schools)
    - `output/all_nodes_valid.gpkg` - Valid schools only (for graph generation)
    - `output/data_quality_report.csv` - Quality metrics
  - **Replaces**: Notebook 0.6 (old cell-based approach)
  - **Next Step**: Notebook 1.0 will import `all_nodes_valid.gpkg` for graph generation

- **Key Design Decisions**:
  1. **GeoDataFrame over DataFrame**: Enables spatial operations in downstream notebooks
  2. **PSGC Integration**: Spatial join adds admin boundaries for provincial graph filtering
  3. **adm2_pcode Column**: Direct matching with Module 9 road network filenames
  4. **Tiered Validation**: Flexible quality standards (required/core/complete)
  5. **Computed Totals**: Enrollment and capacity aggregates ready for graph node weights
  6. **Multiple Export Formats**: GeoPackage (primary), CSV, Parquet for different use cases
  7. **Notebook 0.7 over 0.6.1**: Clearer sequence (0.7 follows 0.6 naturally)

- **Benefits of Refactoring**:
  - ✅ **Reusability**: Single module usable across multiple notebooks/scripts
  - ✅ **Maintainability**: Centralized node table logic (1 file vs 100+ cells)
  - ✅ **Testability**: Methods can be unit tested
  - ✅ **Reproducibility**: Consistent results across runs
  - ✅ **Graph-Ready**: All spatial attributes computed upfront
  - ✅ **Provincial Analysis**: Direct filtering via `adm2_pcode`
  - ✅ **Quality Assurance**: Enhanced validation and reporting
  - ✅ **Performance**: Caching reduces redundant preprocessing

- **Files Created**:
  - `modules/node_table_builder.py` (~950 lines)
  - `notebooks/0.7-node-tables-refined.ipynb` (comprehensive usage examples)
  - Updated `CLAUDE.md` (Module 11 documentation + session history)

- **Module Size**: ~950 lines
  - 350 lines: Public workflow
  - 250 lines: Private workflow
  - 100 lines: Combined workflow + spatial utilities
  - 150 lines: Validation logic
  - 100 lines: Reporting methods
  - 100 lines: Export methods

### 2025-11-13

**Session 1 Summary**: Created Module 12 (ProvincialNetworkBuilder) - ARM-compatible NetworkX-based provincial graph network builder with regional merging design

- **Module 12: Provincial Network Builder** (`modules/provincial_network_builder.py`)
  - **Objective**: Build road network graphs and distance matrices for single province using ARM-compatible libraries
  - **Context**: User has Microsoft Surface Pro 11 with ARM architecture - igraph not reliably available

  - **Architecture Redesign for ARM Compatibility**:
    - **Replaced igraph → NetworkX**: Pure Python graph library (no C dependencies)
    - **Replaced rtree → scipy.spatial.cKDTree**: NumPy/SciPy spatial indexing (ARM-compatible)
    - **Kept multiprocessing**: Standard library, works on all architectures
    - **Performance tradeoff**: NetworkX 2-5x slower than igraph, but acceptable for provincial scope

  - **Key Features**:
    1. **Dual Graph Architecture**:
       - Distance Graph: Nodes = schools, Edges = road distances (meters)
       - Beneficiary Graph: Nodes = schools, Edges = student flow counts (from ESC data)
       - Both graphs share same vertices for interchangeable analysis

    2. **Dual CRS Strategy** (from reference implementation):
       - EPSG:3123 (PRS92 Philippines projected) for accurate distance calculations
       - EPSG:4326 (WGS84) for visualization and storage

    3. **Provincial Scope**:
       - Single province at a time (manageable computation)
       - Filters schools by `adm2_pcode` (e.g., 'PH03014' = Bulacan)
       - Includes cross-provincial flows (external origins/destinations)

    4. **Regional Merging Design** (built-in from start):
       - Node coordinates stored with 5 decimal precision (~1 meter accuracy)
       - Province code tagged on all nodes/edges for tracking origin
       - Boundary nodes identified (within 100m of province boundary)
       - GraphML export preserves all attributes for merging
       - Coordinate-based node deduplication strategy for Module 13

  - **Workflow Steps**:
    1. Load provincial road network from GeoJSONL (Module 9 output)
    2. Convert GeoJSONL → NetworkX graph (custom function)
    3. Project to EPSG:3123 for distance calculations
    4. Snap schools to nearest road nodes using KDTree
    5. Build KDTree spatial index for fast proximity queries
    6. Compute distance matrix (parallel processing with multiprocessing)
    7. Build distance and beneficiary NetworkX graphs
    8. Identify boundary nodes for regional merging

  - **Road Network Conversion** (`_geojsonl_to_networkx()`):
    - Reads GeoJSONL LineString features
    - Creates nodes for each coordinate (rounded to 5 decimals)
    - Creates edges between consecutive coordinates
    - Stores highway type, name, OSM ID as edge attributes
    - Tags all nodes/edges with province code

  - **School Snapping** (`_snap_schools_to_network()`):
    - Uses scipy.spatial.cKDTree for fast nearest neighbor search
    - Projects schools to EPSG:3123 for accurate distance measurement
    - Maps school_id → network_node (coordinate tuple)
    - Attaches school metadata to network nodes
    - Warns if snap distance >500m

  - **Distance Computation** (`_compute_distance_matrix()`):
    - Worker function for multiprocessing: `_compute_distances_for_school()`
    - For each origin school:
      - Find nearby schools within buffer radius (KDTree query_ball_point)
      - Run NetworkX single_source_dijkstra_path_length()
      - Map network nodes back to school IDs
    - Consolidates results into pandas DataFrame (sparse matrix)
    - Parameters: buffer_distance_m=5000, max_distance_km=15, n_processes=4

  - **Graph Building**:
    - Distance graph: Edges from distance matrix with distance_m attribute
    - Beneficiary graph: Edges from validated beneficiary flows with beneficiary_count attribute
    - Both graphs include node attributes: school_id, sector, coordinates, enrollment, seats, province
    - External schools (outside province) tagged with sector='external'

  - **Boundary Node Identification** (`_identify_boundary_nodes()`):
    - Loads province boundary from consolidated geodata
    - Creates 100m buffer around boundary
    - Tags nodes within buffer as boundary nodes
    - Critical for regional merging (cross-provincial road connections)

  - **Export Formats**:
    - Distance matrix: CSV (sparse, origin × destination)
    - Distance graph: GraphML (NetworkX native, preserves all attributes)
    - Beneficiary graph: GraphML
    - Summary statistics: JSON
    - `export_all()` method exports everything to directory

  - **Design Decisions for Regional Merging**:
    - **Coordinate precision**: 5 decimal places (~1m) for node deduplication
    - **Node ID strategy**: Use (x_round, y_round) tuples as node IDs
    - **Province tagging**: All nodes/edges tagged with province code
    - **Boundary identification**: Enables cross-provincial edge detection
    - **GraphML format**: Preserves all node/edge attributes for merging

  - **Regional Merging Strategy (documented for Module 13)**:
    ```python
    # Pseudocode for Module 13: RegionalNetworkBuilder

    # 1. Node deduplication
    merged_G = nx.MultiDiGraph()
    node_mapping = {}  # Maps (x, y) → canonical_node_id

    for province_code, G_province in provincial_graphs.items():
        for node, data in G_province.nodes(data=True):
            coord_key = (round(data['x'], 5), round(data['y'], 5))
            if coord_key not in node_mapping:
                merged_G.add_node(node, **data)  # New node
                node_mapping[coord_key] = node
            else:
                canonical_node = node_mapping[coord_key]  # Duplicate at boundary
                # Merge school metadata if needed

    # 2. Edge addition with canonical nodes
    for u, v, data in G_province.edges(data=True):
        u_canonical = node_mapping[(round(u[0], 5), round(u[1], 5))]
        v_canonical = node_mapping[(round(v[0], 5), round(v[1], 5))]
        merged_G.add_edge(u_canonical, v_canonical, **data)

    # 3. Cross-provincial distance computation
    # Fill NaN values in merged distance matrix using merged road network
    ```

  - **Performance Expectations**:
    - Provincial scale (500-1000 schools): ~30 seconds - 2 minutes for distance computation
    - NetworkX slower than igraph (2-5x) but acceptable for provincial scope
    - Memory efficient with streaming spatial queries
    - Parallelization speeds up computation (4-8 processes recommended)

- **Notebook 0.9: Provincial Network Builder** (`notebooks/0.9-provincial-network-builder.ipynb`)
  - **Purpose**: Demonstrate comprehensive usage of ProvincialNetworkBuilder module
  - **Example Province**: Bulacan (PH03014)

  - **Structure**:
    - Section 0: Setup and imports
    - Section 1: Load data (public nodes, private nodes, beneficiary edges)
    - Section 2: Select province and filter data
    - Section 3: Initialize network builder
    - Section 4: Build complete network (6-step workflow)
    - Section 5: Analyze results (distance matrix, graph statistics)
    - Section 6: Visualizations (distance distribution, school locations, beneficiary flows)
    - Section 7: Export results (GraphML, CSV, JSON)
    - Section 8: Graph analysis examples (shortest paths, centrality, top flows)

  - **Visualizations**:
    - Distance distribution histogram + box plot
    - School locations map (public vs private)
    - Beneficiary flow distribution (linear + log scale)
    - Network statistics summary

  - **Analysis Examples**:
    - Shortest path between schools with distance
    - Degree centrality in beneficiary graph (in-degree/out-degree)
    - Top 5 destination schools (by incoming beneficiary count)
    - Top 5 origin schools (by outgoing beneficiary count)
    - Total beneficiaries by school (sent vs received)

  - **Outputs Generated**:
    - `output/provincial_networks/PH03014_bulacan_distance_matrix.csv`
    - `output/provincial_networks/PH03014_bulacan_distance_graph.graphml`
    - `output/provincial_networks/PH03014_bulacan_beneficiary_graph.graphml`
    - `output/provincial_networks/PH03014_bulacan_summary.json`

- **Key Design Patterns**:
  1. **ARM Compatibility First**: Chose libraries based on ARM support (NetworkX, scipy)
  2. **Regional Merging Built-In**: Coordinate precision, province tagging, boundary nodes from start
  3. **Separation of Concerns**: Provincial builder (Module 12) vs Regional merger (Module 13, future)
  4. **Reusable Components**: KDTree spatial indexing, NetworkX conversion, parallel workers
  5. **Export Flexibility**: Multiple formats (GraphML for graphs, CSV for matrices, JSON for metadata)

- **Integration Points**:
  - **Input**: Module 11 (node tables), BeneficiaryProcessor (validated edges), Module 9 (road networks)
  - **Output**: GraphML graphs ready for regional merging (Module 13)
  - **Output**: Distance matrices ready for discrete choice modeling
  - **Next Module**: Module 13 (RegionalNetworkBuilder) will merge provincial networks

- **Files Created**:
  - `modules/provincial_network_builder.py` (~800 lines)
  - `notebooks/0.9-provincial-network-builder.ipynb` (comprehensive usage demonstration)
  - Updated `CLAUDE.md` (Module 12 documentation + session history)

- **Module Size**: ~800 lines
  - 150 lines: Initialization and setup
  - 200 lines: Road network loading and conversion
  - 100 lines: School snapping and spatial indexing
  - 150 lines: Distance computation (parallel)
  - 100 lines: Graph building
  - 50 lines: Boundary node identification
  - 50 lines: Export methods

**Session 2 Summary**: Created Module 12.1 (scipy optimization) + Notebook 0.9.1 with comprehensive network verification visualizations

- **Performance Problem Identified**: Module 12 distance matrix computation too slow (10+ minutes for 1000 schools)
  - **Bottleneck**: NetworkX `single_source_dijkstra_path_length()` called in loop for each school
  - **Issue**: NetworkX pure Python implementation not optimized for large-scale shortest path computation
  - **User Request**: ~10x speedup needed for provincial scale analysis

- **Module 12.1: Provincial Network Builder (SciPy-Optimized)** (`modules/provincial_network_builder_scipy.py`)
  - **Objective**: Speed up distance matrix computation by ~10x using scipy.sparse.csgraph
  - **Created**: 2025-11-13 (same day as Module 12)

  - **Key Differences from Module 12**:
    | Aspect | Module 12 (NetworkX) | Module 12.1 (SciPy) |
    |--------|---------------------|---------------------|
    | **Class Name** | `ProvincialNetworkBuilder` | `ProvincialNetworkBuilderSciPy` |
    | **Distance Algorithm** | `nx.single_source_dijkstra_path_length()` (loop) | `scipy.sparse.csgraph.dijkstra()` (vectorized) |
    | **Multiprocessing** | Required (`n_processes` parameter) | Not needed (scipy is optimized) |
    | **Graph Representation** | NetworkX MultiDiGraph only | scipy sparse CSR matrix + NetworkX |
    | **Speed** | Baseline (slow) | **~10x faster** |
    | **Memory** | Higher (dense operations) | Lower (sparse matrix) |

  - **scipy.sparse.csgraph Optimization** (`_compute_distance_matrix_scipy()`):
    ```python
    # Step 1: Convert NetworkX graph to scipy sparse adjacency matrix
    adj_matrix = lil_matrix((n_nodes, n_nodes), dtype=np.float32)
    for u, v, data in G.edges(data=True):
        u_idx = node_index_map[u]
        v_idx = node_index_map[v]
        length = data.get('length', 1.0)
        adj_matrix[u_idx, v_idx] = length

    adj_matrix = adj_matrix.tocsr()  # Convert to CSR for fast computation

    # Step 2: Compute ALL distances at once using scipy (FAST!)
    dist_matrix = dijkstra(
        csgraph=adj_matrix,
        directed=True,
        indices=school_node_indices,  # Only compute from school nodes
        limit=max_distance_km * 1000,
        return_predecessors=False
    )

    # Step 3: Extract school-to-school distances and convert to DataFrame
    ```

  - **Why scipy.sparse.csgraph is ~10x Faster**:
    1. **Cython-optimized C code** (not pure Python like NetworkX)
    2. **Vectorized operations** (computes all distances at once, not in loop)
    3. **Sparse matrix representation** (only stores edges, not full n×n matrix)
    4. **Efficient priority queue** (C implementation, not Python heap)
    5. **No Python overhead** (stays in C for inner loops)

  - **Matrix Formats Used**:
    - **LIL (List of Lists)**: Efficient for construction (adding edges)
    - **CSR (Compressed Sparse Row)**: Efficient for computation (scipy algorithms)
    - **Conversion**: `lil_matrix → tocsr() → CSR format`

  - **Research Findings** (Web search 2024-2025):
    - scipy ~10x faster than NetworkX for shortest-path problems
    - NetworKit also ~10x faster but requires additional installation
    - graph-tool 40-250x faster but complex installation and poor ARM support
    - **Decision**: Use scipy for immediate ~10x speedup with zero installation overhead

  - **Performance Expectations**:
    | Province Size | NetworkX (Module 12) | SciPy (Module 12.1) | Speedup |
    |---------------|---------------------|---------------------|---------|
    | **Bulacan (885 schools)** | 5-10 minutes (10 processes) | **30-60 seconds** | **~10x faster** |
    | **Cebu (2000+ schools)** | 20-40 minutes | **2-4 minutes** | **~10x faster** |

  - **API Changes**:
    - Removed `n_processes` parameter (not needed with scipy optimization)
    - Same output format as Module 12 (drop-in replacement)
    - Same initialization and export methods

  - **Limitations**:
    1. **No path reconstruction**: scipy.dijkstra returns distances only, not actual paths
       - If paths needed, use NetworkX version or add `return_predecessors=True`
    2. **Memory spike during conversion**: NetworkX → scipy requires full graph in memory
       - Not an issue for provincial scale (<500k nodes)
       - May be issue for regional scale (>2M nodes) - use chunking
    3. **Less flexible**: scipy.sparse.csgraph has fewer features than NetworkX
       - No arbitrary node attributes during computation
       - No custom weight functions (must be numeric edge weights)

  - **When to Use Each Version**:
    - **Use Module 12 (NetworkX)** if:
      - Debugging algorithm (NetworkX code easier to read)
      - Very small graphs (<100 schools)
      - Need intermediate path information (not just distances)
      - Already familiar with NetworkX API
    - **Use Module 12.1 (SciPy)** if:
      - **Need speed** (almost always!)
      - Processing multiple provinces
      - Large provinces (>500 schools)
      - Only need distance matrices (not full path details)

  - **Recommendation**: **Use Module 12.1 (SciPy) for production**, keep Module 12 for reference

- **Notebook 0.9.1: Provincial Network Builder (SciPy-Optimized)** (`notebooks/0.9.1-provincial-network-builder-scipy.ipynb`)
  - **Purpose**: Demonstrate scipy-optimized network builder with comprehensive network verification visualizations
  - **Example Province**: Bulacan (PH03014, 885 schools)

  - **Structure**:
    - Sections 0-5: Same as notebook 0.9 (setup, load data, filter province, build network, analyze)
    - **Section 6: Network Verification Visualizations** (NEW):
      - Visualization 1: Province boundary + road network + schools overlay
      - Visualization 2: Same as Viz 1 + highlighted school connections <3km apart
      - Visualization 3: Beneficiary flow network with varying line widths
      - Visualization 4: Interactive Plotly visualization with hover tooltips
    - Section 7-8: Same as notebook 0.9 (export, graph analysis)

  - **Visualization 1: Basic Network Overlay** (matplotlib)
    - Province boundary (black outline)
    - Road network (grey lines, low opacity)
    - Public schools (blue circles)
    - Private schools (orange circles)
    - Purpose: Verify spatial coverage and data alignment

  - **Visualization 2: School Connections <3km** (matplotlib)
    - Same as Viz 1 base layers
    - PLUS: Red lines connecting schools within 3km road distance
    - Line width proportional to 1/distance (closer = thicker)
    - Purpose: Verify distance matrix accuracy and identify dense clusters

  - **Visualization 3: Beneficiary Flow Network** (matplotlib with LineCollection optimization)
    - **Initial Implementation**: Slow (30-60 seconds for 4,016 flow lines)
      ```python
      # SLOW - 4,016 iterations
      for idx, row in flows_gdf.iterrows():
          gpd.GeoSeries([row.geometry], crs='EPSG:4326').plot(...)
      ```

    - **Optimized Implementation**: Fast (1-2 seconds)
      ```python
      # FAST - Single batch operation
      from matplotlib.collections import LineCollection

      segments = [list(geom.coords) for geom in flows_gdf.geometry]
      widths = flows_gdf['line_width'].values

      lc = LineCollection(
          segments,
          linewidths=widths,
          colors='purple',
          alpha=0.5,
          zorder=3,
          label='Beneficiary Flows (width ∝ students)'
      )
      ax.add_collection(lc)
      ```

    - **Performance Improvement**: 100-1000x speedup (single matplotlib call vs 4,016 individual plot() calls)
    - **Visual Elements**:
      - Base layers: boundary, roads, schools
      - Purple lines: Beneficiary flows (origin → destination JHS)
      - Line width: Proportional to beneficiary count (scaled 0.5-3.5)
      - Purpose: Visualize student flow patterns and identify major destinations

  - **Visualization 4: Interactive Plotly Network** (plotly.graph_objects)
    - **User Requirements**:
      1. Hover tooltips showing school name, ID, in/out edges, total beneficiary counts
      2. Diamond markers for destination JHS schools
      3. Circle markers for origin schools
      4. Greyed out schools with no beneficiary edges
      5. Granular control via configuration dictionary

    - **Implementation**: 3 cells added to notebook
      - **Cell 40: Configuration Dictionary** (`viz_config`)
        ```python
        viz_config = {
            'figure': {
                'width': 1400,
                'height': 1000,
                'title': f'Interactive Beneficiary Flow Network - {PROVINCE_NAME.title()}',
                'title_font_size': 18
            },
            'schools': {
                'dest_jhs': {
                    'size': 14, 'color': 'crimson', 'symbol': 'diamond',
                    'opacity': 0.85, 'name': 'Destination JHS', ...
                },
                'origin_only': {
                    'size': 10, 'color': 'dodgerblue', 'symbol': 'circle',
                    'opacity': 0.75, 'name': 'Origin Schools', ...
                },
                'both': {
                    'size': 12, 'color': 'forestgreen', 'symbol': 'square',
                    'opacity': 0.8, 'name': 'Both In/Out', ...
                },
                'no_edges': {
                    'size': 6, 'color': 'lightgray', 'symbol': 'circle',
                    'opacity': 0.3, 'name': 'No Flows (Greyed Out)', ...
                }
            },
            'road_network': {'width': 0.4, 'color': 'gray', 'opacity': 0.12, 'sample_rate': 0.3},
            'flows': {'color': 'purple', 'opacity': 0.35, 'width_min': 0.5, 'width_max': 3.5},
            'boundary': {'width': 2.5, 'color': 'black', 'opacity': 0.9}
        }
        ```

      - **Cell 41: School Classification** (vectorized pandas operations)
        ```python
        # Calculate in/out totals from beneficiary graph
        in_flow_totals = flows_gdf.groupby('dest_id')['beneficiary_count'].sum().to_dict()
        out_flow_totals = flows_gdf.groupby('origin_id')['beneficiary_count'].sum().to_dict()
        in_counts = dict(beneficiary_graph.in_degree())
        out_counts = dict(beneficiary_graph.out_degree())

        # Classify into 4 mutually exclusive categories
        dest_jhs_mask = (all_schools['in_count'] > 0) & (all_schools['offers_jhs'] == True)
        origin_only_mask = (all_schools['out_count'] > 0) & (all_schools['in_count'] == 0)
        both_mask = (all_schools['in_count'] > 0) & (all_schools['out_count'] > 0) & (~dest_jhs_mask)
        no_edges_mask = (all_schools['in_count'] == 0) & (all_schools['out_count'] == 0)

        # Split into 4 DataFrames
        dest_jhs = all_schools[dest_jhs_mask].copy()
        origin_only = all_schools[origin_only_mask].copy()
        both_in_out = all_schools[both_mask].copy()
        no_edges_schools = all_schools[no_edges_mask].copy()
        ```

      - **Cell 42: Plotly Visualization** (with MultiPolygon error fix)
        - Custom hover templates with school details
        - Separate traces for each school category
        - Road network (sampled for performance)
        - Beneficiary flow lines (purple arrows)
        - Province boundary handling (Polygon vs MultiPolygon)

    - **MultiPolygon Error Fix**:
      - **Problem**: `AttributeError: 'MultiPolygon' object has no attribute 'exterior'`
        - Province boundary was MultiPolygon (multiple islands/disconnected areas)
        - Original code only handled Polygon case
      - **Solution**: Added isinstance check for both geometry types
        ```python
        from shapely.geometry import Polygon, MultiPolygon

        if isinstance(geom, MultiPolygon):
            # MultiPolygon - plot each polygon's exterior
            for poly in geom.geoms:
                boundary_coords = list(poly.exterior.coords)
                boundary_lons = [coord[0] for coord in boundary_coords]
                boundary_lats = [coord[1] for coord in boundary_coords]
                # Add trace for this polygon
        elif isinstance(geom, Polygon):
            # Single Polygon
            boundary_coords = list(geom.exterior.coords)
            # Add single trace
        ```
      - **Result**: Visualization works for both contiguous provinces and archipelagic provinces

    - **Interactive Features**:
      - Hover over schools: Name, ID, in/out edges, beneficiary counts
      - Click legend: Toggle school categories, flows, roads, boundary
      - Zoom and pan: Explore network details
      - Export to HTML: `fig.write_html('output/viz4_interactive_network.html')`

  - **Outputs Generated**:
    - Same as notebook 0.9 (distance matrix, graphs, summary)
    - PLUS: 4 visualization figures in notebook cells
    - Optional: Interactive HTML export for Viz 4

  - **Performance Summary**:
    - Distance matrix computation: **~45 seconds** (vs 8 minutes with NetworkX)
    - Visualization 3 plotting: **~1-2 seconds** (vs 30-60 seconds before LineCollection)
    - Overall notebook execution: **~2-3 minutes total** (vs 10+ minutes with Module 12)

- **Key Technical Concepts**:
  - **scipy.sparse.csgraph.dijkstra()**: Cython-optimized C shortest path algorithm (~10x faster than NetworkX)
  - **matplotlib.collections.LineCollection**: Batch plotting for multiple lines (100-1000x faster than individual plot() calls)
  - **Plotly interactive visualization**: Browser-based interactive plots with hover tooltips
  - **Vectorized pandas operations**: Efficient school classification using boolean masking
  - **MultiPolygon vs Polygon geometry**: Handling provinces with multiple disconnected areas vs single contiguous area
  - **Sparse matrix formats**: LIL for construction, CSR for computation

- **Error Fixes**:
  1. **Viz 3 Performance Issue**:
     - Problem: 4,016 individual plot() calls taking 30-60 seconds
     - Solution: LineCollection batch plotting (single matplotlib call)
     - Result: 100-1000x speedup (1-2 seconds)

  2. **Viz 4 MultiPolygon Error**:
     - Problem: `AttributeError: 'MultiPolygon' object has no attribute 'exterior'`
     - Root Cause: Province boundary was MultiPolygon, code only handled Polygon
     - Solution: isinstance check to handle both geometry types
     - Result: Works for all province geometries

- **Files Created/Modified**:
  - `modules/provincial_network_builder_scipy.py` (~850 lines)
  - `notebooks/0.9.1-provincial-network-builder-scipy.ipynb` (comprehensive usage + 4 visualizations)
  - `SCIPY_OPTIMIZATION_SUMMARY.md` (detailed optimization documentation)
  - Updated `CLAUDE.md` (this file)

- **Documentation**:
  - Created `SCIPY_OPTIMIZATION_SUMMARY.md`: Comprehensive guide to scipy optimization
    - Performance problem analysis
    - Solution architecture (scipy.sparse.csgraph)
    - Research findings and benchmarks
    - Technical details (matrix formats, algorithm parameters)
    - Verification steps and usage recommendations
    - Limitations and future optimizations

- **Integration Points**:
  - **Input**: Same as Module 12 (Module 11 node tables, BeneficiaryProcessor edges, Module 9 road networks)
  - **Output**: Same format as Module 12 (drop-in replacement)
  - **Usage**: Replace `ProvincialNetworkBuilder` → `ProvincialNetworkBuilderSciPy` in imports
  - **Next Steps**: Use Module 12.1 for all future provincial network generation

- **Next Optimizations** (if scipy still too slow):
  1. **NetworKit**: Same ~10x speedup, specialized for graph algorithms
  2. **Chunked Processing**: Process provinces in batches for regional scale
  3. **GPU Acceleration**: cuGraph (NVIDIA RAPIDS) for 100x+ speedup on massive graphs

### 2025-11-17

**Session Summary**: Planned aggregation approach for unified Grade 7 dataset to prepare for graph creation

- **Unified Grade 7 Dataset Aggregation Requirement**
  - **Context**: Module `unified_gr7_flow_builder.py` creates student-level (one row per LRN) flow data
  - **Need**: Aggregate by origin-destination school pairs for graph network creation
  - **Comparison**: Original beneficiary-exclusive dataset aggregates by school year (creates columns `beneficiaries_sy_2021`, `beneficiaries_sy_2022`, etc.)
  - **Key Difference**: Unified dataset uses single school year but has TWO student types (beneficiaries + non-beneficiaries)

- **Aggregation Approach Design**
  - **Grouping dimension**: `(school_id_origin, school_id_destination)` pairs
  - **Parsing dimension**: Flow type (beneficiary vs non-beneficiary) instead of school year
  - **Method**: Similar to BeneficiaryProcessor's `_pivot_by_school_year()` but pivoting on `is_beneficiary` status
  - **Rationale**: Single school year means no need for multiple year columns; instead parse by student beneficiary status

- **Proposed Aggregated Column Structure** (13 columns total):

  1. **School Pair Identifiers** (2 columns):
     - `school_id_origin` (string) - Origin school
     - `school_id_destination` (string) - Destination school

  2. **Student Counts by Flow Type** (2 columns):
     - `beneficiary_count` (int) - Count of ESC beneficiary students in this flow
     - `non_beneficiary_count` (int) - Count of non-beneficiary students in this flow

  3. **Total Count** (1 column):
     - `total_student_count` (int) - Sum of beneficiary + non-beneficiary

  4. **School Year** (1 column):
     - `sy_grade6` (string) - Year value (e.g., "2023") - single year since unified dataset uses one year

  5. **School Attributes - Origin** (3 columns):
     - `sector_origin` (string) - "Public" or "Private"
     - `latitude_origin` (float)
     - `longitude_origin` (float)

  6. **School Attributes - Destination** (3 columns):
     - `sector_destination` (string) - "Public" or "Private"
     - `latitude_destination` (float)
     - `longitude_destination` (float)

  7. **Distance Metrics** (1 column):
     - `distance_straightline_km` (float) - Average or most common distance for this pair

- **Sample Aggregated Data Structure**:
  ```
  school_id_origin | school_id_destination | beneficiary_count | non_beneficiary_count | total_student_count | sy_grade6 | sector_origin | sector_destination | distance_straightline_km
  -----------------|----------------------|-------------------|----------------------|---------------------|-----------|---------------|--------------------|-----------------------
  100001           | 200001               | 15                | 0                    | 15                  | 2023      | Public        | Private            | 3.2
  100001           | 200002               | 0                 | 45                   | 45                  | 2023      | Public        | Public             | 1.5
  100002           | 200001               | 8                 | 0                    | 8                   | 2023      | Public        | Private            | 5.7
  100002           | 200003               | 3                 | 12                   | 15                  | 2023      | Public        | Public             | 2.1
  ```

- **Key Differences: Unified vs Original Beneficiary Dataset**:

  | Aspect | Original Beneficiary Dataset | Unified Grade 7 Dataset |
  |--------|------------------------------|------------------------|
  | **Parsing dimension** | School year (2021, 2022, 2023, ...) | Flow type (beneficiary vs non-beneficiary) |
  | **Count columns** | `beneficiaries_sy_2021`, `beneficiaries_sy_2022`, `beneficiaries_sy_2023`, ... | `beneficiary_count`, `non_beneficiary_count` |
  | **Total column** | `total_beneficiaries` | `total_student_count` |
  | **Year representation** | Multiple year columns (wide format) | Single `sy_grade6` column (dataset uses one year) |
  | **Student types** | Beneficiaries only | Both beneficiaries AND non-beneficiaries |
  | **Use case** | Multi-year trend analysis | Comprehensive single-year flow analysis (all Grade 7 students) |

- **Implementation Plan** (3 steps):

  1. **Aggregation by school pair + beneficiary status**:
     ```python
     grouped = unified_data.groupby([
         'school_id_origin',
         'school_id_destination',
         'is_beneficiary'
     ]).agg({
         'lrn': 'count',  # Count students
         'sector_origin': 'first',
         'sector_destination': 'first',
         'latitude_origin': 'first',
         'longitude_origin': 'first',
         'latitude_destination': 'first',
         'longitude_destination': 'first',
         'distance_straightline_km': 'mean',
         'sy_grade6': 'first'
     }).reset_index()
     ```

  2. **Pivot on is_beneficiary to create separate count columns**:
     ```python
     pivoted = grouped.pivot_table(
         index=['school_id_origin', 'school_id_destination'],
         columns='is_beneficiary',
         values='lrn',  # Student counts
         fill_value=0,
         aggfunc='sum'
     )
     # Result: beneficiary_count (True), non_beneficiary_count (False)
     ```

  3. **Calculate totals and merge attributes**:
     ```python
     pivoted['total_student_count'] = (
         pivoted['beneficiary_count'] +
         pivoted['non_beneficiary_count']
     )
     # Merge back school attributes and distance
     ```

- **Integration with Graph Generation**:
  - Output format compatible with ProvincialNetworkBuilder (Module 12.1)
  - Each row becomes an edge in beneficiary graph
  - `beneficiary_count` becomes edge weight for beneficiary flows
  - `non_beneficiary_count` provides context for total flow analysis
  - `total_student_count` enables comprehensive network analysis (all Grade 7 transitions)

- **Next Steps**:
  - [ ] Implement aggregation method in `unified_gr7_flow_builder.py`
  - [ ] Add export method for aggregated dataset
  - [ ] Test with sample province data
  - [ ] Update test scripts to verify aggregated structure
  - [ ] Document usage in module docstring

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