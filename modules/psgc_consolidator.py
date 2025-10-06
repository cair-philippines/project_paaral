"""
PSGC (Philippine Standard Geographic Code) Data Consolidator

This module consolidates Philippines geographic administrative hierarchy data from
multiple CSV files and merges with shapefile geometries. It creates a comprehensive
GeoDataFrame with complete hierarchical information from regions to barangays.

The consolidation process:
1. Loads administrative data at 4 levels: Regions (Adm1), Provinces/Districts (Adm2),
   Municipalities/Cities (Adm3), and Barangays/Sub-Municipalities (Adm4)
2. Performs hierarchical joins using PSGC codes
3. Fixes City of Manila missing data (897 NCR barangays)
4. Adds leading zeros to PSGC codes (ensures 10-digit format)
5. Reorders columns for better organization (PSGC codes, names, other data)
6. Prepares shapefile: standardizes codes, filters null geometries, selects relevant columns
7. Merges: shapefile (left) → consolidated CSV (right) for complete geographic coverage
8. OPTIONAL: Spatial matching to fill ~3,580 unmatched barangays using point-in-polygon

Key features:
- Automatic City of Manila detection and filling for NCR barangays
- PSGC code standardization with leading zeros (string type)
- Cleaner column organization (codes → names → data)
- Better merge strategy (shapefile-first to preserve all valid geometries)
- Selective shapefile columns (psgc_code, corr_code, name, adm4_en, geometry)
- Spatial matching using STRtree for efficient point-in-polygon queries
- Preserves original data (with NaN) before spatial matching
- Tags spatially-matched barangays with 'is_spatially_matched' column

Spatial Matching:
- Dissolves matched barangays to municipality boundaries (reference)
- Uses centroid-based point-in-polygon for unmatched barangays
- Falls back to nearest neighbor for boundary cases
- ~1-2 minutes to match 3,580 unmatched barangays
- Results in complete dataset with no NaN admin codes

Author: Data Processing System
"""

import pandas as pd
import geopandas as gpd
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PSGCConsolidator:
    """
    Consolidates Philippines PSGC geographic administrative hierarchy data
    and merges with shapefile geometries.

    The processor reads CSV files for all 4 administrative levels (Region, Province,
    Municipality, Barangay), performs hierarchical joins on PSGC codes, and merges
    with shapefile data to produce a complete GeoDataFrame with geographic boundaries.
    """

    def __init__(self, base_path: Optional[str] = None, verbose: bool = True):
        """
        Initialize the PSGC consolidator.

        Args:
            base_path: Path to directory containing PSGC files. If None, uses default path.
            verbose: If True, logs at INFO level. If False, logs at WARNING level only.
        """
        if base_path is None:
            base_path = r"C:\Users\elibu\Documents\Work\innovation-projects\project_paaral\data\philippines-psgc-shapefiles\dist"

        self.base_path = Path(base_path)
        self.verbose = verbose

        # Data storage
        self.adm1_data = None  # Regions
        self.adm2_data = None  # Provinces/Districts
        self.adm3_data = None  # Municipalities/Cities
        self.adm4_data = None  # Barangays/Sub-Municipalities
        self.adm4_geodata = None  # Shapefile data
        self.consolidated_data = None  # Final consolidated DataFrame
        self.consolidated_geodata = None  # Final GeoDataFrame with geometry
        self.consolidated_geodata_original = None  # Original geodata before spatial matching
        self.reference_boundaries = None  # Dissolved municipality boundaries for spatial matching

        # Set logging level based on verbose flag
        if not verbose:
            logger.setLevel(logging.WARNING)

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all PSGC CSV files and the shapefile.

        Returns:
            Dictionary containing all loaded DataFrames
        """
        try:
            logger.info(f"Loading PSGC data from {self.base_path}")

            # Load Adm1 - Regions
            logger.info("Loading Adm1 (Regions) data...")
            adm1_path = self.base_path / "PH_Adm1_Regions.csv"
            self.adm1_data = pd.read_csv(adm1_path)
            self.adm1_data.columns = self.adm1_data.columns.str.strip()
            self.adm1_data = self._trim_whitespaces(self.adm1_data)
            logger.info(f"Loaded {len(self.adm1_data)} regions")

            # Load Adm2 - Provinces/Districts
            logger.info("Loading Adm2 (Provinces/Districts) data...")
            adm2_path = self.base_path / "PH_Adm2_ProvDists.csv"
            self.adm2_data = pd.read_csv(adm2_path)
            self.adm2_data.columns = self.adm2_data.columns.str.strip()
            self.adm2_data = self._trim_whitespaces(self.adm2_data)
            logger.info(f"Loaded {len(self.adm2_data)} provinces/districts")

            # Load Adm3 - Municipalities/Cities
            logger.info("Loading Adm3 (Municipalities/Cities) data...")
            adm3_path = self.base_path / "PH_Adm3_MuniCities.csv"
            self.adm3_data = pd.read_csv(adm3_path)
            self.adm3_data.columns = self.adm3_data.columns.str.strip()
            self.adm3_data = self._trim_whitespaces(self.adm3_data)
            logger.info(f"Loaded {len(self.adm3_data)} municipalities/cities")

            # Load Adm4 - Barangays/Sub-Municipalities
            logger.info("Loading Adm4 (Barangays/Sub-Municipalities) data...")
            adm4_path = self.base_path / "PH_Adm4_BgySubMuns.csv"
            self.adm4_data = pd.read_csv(adm4_path)
            self.adm4_data.columns = self.adm4_data.columns.str.strip()
            self.adm4_data = self._trim_whitespaces(self.adm4_data)
            logger.info(f"Loaded {len(self.adm4_data)} barangays/sub-municipalities")

            # Load Adm4 Shapefile
            logger.info("Loading Adm4 shapefile data...")
            shapefile_path = self.base_path / "PH_Adm4_BgySubMuns.shp.zip"

            # Geopandas can read .shp.zip files directly
            self.adm4_geodata = gpd.read_file(f"zip://{shapefile_path}")
            self.adm4_geodata.columns = self.adm4_geodata.columns.str.strip()
            logger.info(f"Loaded shapefile with {len(self.adm4_geodata)} geometries")
            logger.info(f"Shapefile CRS: {self.adm4_geodata.crs}")

            return {
                'adm1': self.adm1_data,
                'adm2': self.adm2_data,
                'adm3': self.adm3_data,
                'adm4': self.adm4_data,
                'adm4_geo': self.adm4_geodata
            }

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def _trim_whitespaces(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trim leading and trailing whitespaces from string columns.

        Args:
            df: DataFrame to process

        Returns:
            DataFrame with trimmed string columns
        """
        df_trimmed = df.copy()
        string_columns = df_trimmed.select_dtypes(include=['object', 'string']).columns

        for col in string_columns:
            mask = df_trimmed[col].notna()
            df_trimmed.loc[mask, col] = df_trimmed.loc[mask, col].astype(str).str.strip()

        logger.info(f"Trimmed whitespaces from {len(string_columns)} string columns")
        return df_trimmed

    def _add_leading_zeros(self, psgc_code) -> str:
        """
        Add leading zeros to PSGC codes to ensure 10-digit format.

        Args:
            psgc_code: PSGC code (int or str)

        Returns:
            10-digit PSGC code as string
        """
        code_str = str(psgc_code)
        if len(code_str) == 9:
            return '0' + code_str
        else:
            return code_str

    def _fix_city_of_manila(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing City of Manila data for NCR barangays.

        Args:
            df: DataFrame with hierarchical data

        Returns:
            DataFrame with City of Manila filled in
        """
        # Identify NCR barangays without city/municipality name
        mask = (
            (df['adm1_en'].astype('string').str.contains(r'capital', flags=2, na=False))
            & (df['adm3_en'].isna())
            & (df['adm2_en'].isna())
        )

        rows_fixed = mask.sum()
        if rows_fixed > 0:
            df.loc[mask, 'adm3_en'] = 'City of Manila'
            logger.info(f"Fixed {rows_fixed} City of Manila records")

        return df

    def consolidate_hierarchy(self) -> pd.DataFrame:
        """
        Perform hierarchical joins on PSGC codes to consolidate all administrative levels.

        The join strategy:
        1. Start with Adm4 (Barangays) as base
        2. Join with Adm3 (Municipalities) on [adm1_psgc, adm2_psgc, adm3_psgc]
        3. Join with Adm2 (Provinces) on [adm1_psgc, adm2_psgc]
        4. Join with Adm1 (Regions) on [adm1_psgc]
        5. Fix City of Manila missing data
        6. Add leading zeros to PSGC codes
        7. Reorder columns for better organization

        Returns:
            Consolidated DataFrame with all hierarchical information
        """
        if any(x is None for x in [self.adm1_data, self.adm2_data, self.adm3_data, self.adm4_data]):
            raise ValueError("Data not loaded. Call load_data() first.")

        logger.info("Starting hierarchical consolidation...")

        # Start with Adm4 as base (most detailed level)
        consolidated = self.adm4_data.copy()
        initial_rows = len(consolidated)
        logger.info(f"Base data (Adm4): {initial_rows} rows")

        # Join with Adm3 (Municipalities/Cities)
        logger.info("Joining with Adm3 (Municipalities/Cities)...")
        merge_keys_adm3 = ['adm1_psgc', 'adm2_psgc', 'adm3_psgc']
        consolidated = consolidated.merge(
            self.adm3_data,
            on=merge_keys_adm3,
            how='left',
            suffixes=('', '_adm3')
        )
        logger.info(f"After Adm3 join: {len(consolidated)} rows")
        self._validate_merge(consolidated, initial_rows, "Adm3")

        # Join with Adm2 (Provinces/Districts)
        logger.info("Joining with Adm2 (Provinces/Districts)...")
        merge_keys_adm2 = ['adm1_psgc', 'adm2_psgc']
        consolidated = consolidated.merge(
            self.adm2_data,
            on=merge_keys_adm2,
            how='left',
            suffixes=('', '_adm2')
        )
        logger.info(f"After Adm2 join: {len(consolidated)} rows")
        self._validate_merge(consolidated, initial_rows, "Adm2")

        # Join with Adm1 (Regions)
        logger.info("Joining with Adm1 (Regions)...")
        merge_keys_adm1 = ['adm1_psgc']
        consolidated = consolidated.merge(
            self.adm1_data,
            on=merge_keys_adm1,
            how='left',
            suffixes=('', '_adm1')
        )
        logger.info(f"After Adm1 join: {len(consolidated)} rows")
        self._validate_merge(consolidated, initial_rows, "Adm1")

        # Clean up duplicate columns (keep the most specific level's values)
        # Priority: Adm4 > Adm3 > Adm2 > Adm1
        duplicate_cols = ['geo_level', 'len_crs', 'area_crs', 'len_km', 'area_km2']
        for col in duplicate_cols:
            # Keep the base column (from Adm4) and drop suffixed versions
            cols_to_drop = [c for c in consolidated.columns if c.startswith(f"{col}_")]
            if cols_to_drop:
                consolidated = consolidated.drop(columns=cols_to_drop)
                logger.info(f"Cleaned duplicate column: {col}")

        # Fix City of Manila missing data
        logger.info("Fixing City of Manila missing data...")
        consolidated = self._fix_city_of_manila(consolidated)

        # Add leading zeros to PSGC codes and convert to string
        logger.info("Adding leading zeros to PSGC codes...")
        psgc_columns = [col for col in consolidated.columns if '_psgc' in col]
        for col in psgc_columns:
            consolidated[col] = consolidated[col].apply(self._add_leading_zeros)
            consolidated[col] = consolidated[col].astype('string')
        logger.info(f"Standardized {len(psgc_columns)} PSGC code columns to 10-digit string format")

        # Reorder columns: [psgc codes] + [names (reversed)] + [other columns]
        logger.info("Reordering columns for better organization...")
        columns_psgc = [col for col in consolidated.columns if '_psgc' in col]
        columns_en = [col for col in consolidated.columns if '_en' in col]
        columns_other = [col for col in consolidated.columns if col not in columns_psgc + columns_en]

        # Reverse the _en columns so most specific (adm4) comes last
        consolidated = consolidated[columns_psgc + columns_en[::-1] + columns_other]
        logger.info("Columns reordered: PSGC codes first, names (reversed), then other data")

        self.consolidated_data = consolidated
        logger.info(f"Hierarchical consolidation complete: {len(self.consolidated_data)} rows, {len(self.consolidated_data.columns)} columns")

        return self.consolidated_data

    def _validate_merge(self, df: pd.DataFrame, expected_rows: int, level_name: str):
        """
        Validate that merge didn't create unexpected duplicates or lose rows.

        Args:
            df: DataFrame after merge
            expected_rows: Expected number of rows
            level_name: Name of the administrative level for logging
        """
        if len(df) != expected_rows:
            logger.warning(f"Row count changed after {level_name} merge: {expected_rows} -> {len(df)}")
            if len(df) > expected_rows:
                logger.warning(f"Unexpected duplicates created in {level_name} merge")
            else:
                logger.warning(f"Rows lost in {level_name} merge")
        else:
            logger.info(f"Merge validation passed for {level_name}")

    def _prepare_shapefile_for_merge(self) -> gpd.GeoDataFrame:
        """
        Prepare shapefile data for merging by filtering and selecting relevant columns.

        Returns:
            Prepared GeoDataFrame with only valid geometries and relevant columns
        """
        shapefile = self.adm4_geodata.copy()

        # Add leading zeros to psgc_code
        logger.info("Standardizing shapefile PSGC codes...")
        shapefile['psgc_code'] = shapefile['psgc_code'].apply(self._add_leading_zeros)
        shapefile['psgc_code'] = shapefile['psgc_code'].astype('string')

        # Select only relevant columns
        relevant_columns = ['psgc_code', 'corr_code', 'name', 'adm4_en', 'geometry']
        shapefile = shapefile[relevant_columns]

        # Filter out rows without geometry
        initial_count = len(shapefile)
        mask_valid_geometry = shapefile['geometry'].notna()
        shapefile = shapefile.loc[mask_valid_geometry].copy()
        removed_count = initial_count - len(shapefile)

        if removed_count > 0:
            logger.info(f"Filtered out {removed_count} shapefile rows without valid geometry")

        logger.info(f"Shapefile prepared: {len(shapefile)} features with valid geometry")

        return shapefile

    def merge_with_geometry(self) -> gpd.GeoDataFrame:
        """
        Merge consolidated CSV data with shapefile geometries using preferred approach.

        Preferred merge strategy:
        1. Prepare shapefile: add leading zeros, filter out null geometries, select relevant columns
        2. Left join: shapefile (left) → consolidated CSV (right) on psgc_code = adm4_psgc
        3. Result includes all valid geometries, with matched CSV data where available

        Returns:
            GeoDataFrame with all hierarchical data and geometries
        """
        if self.consolidated_data is None:
            raise ValueError("Data not consolidated. Call consolidate_hierarchy() first.")

        if self.adm4_geodata is None:
            raise ValueError("Shapefile not loaded. Call load_data() first.")

        logger.info("Merging consolidated data with geometries using preferred approach...")

        # Prepare shapefile for merge
        shapefile_prepared = self._prepare_shapefile_for_merge()

        # The shapefile uses 'psgc_code' while CSV uses 'adm4_psgc'
        # Merge: shapefile (left) → consolidated CSV (right)
        if 'adm4_psgc' not in self.consolidated_data.columns:
            raise ValueError(f"Could not find adm4_psgc column in consolidated data. Available columns: {list(self.consolidated_data.columns)}")

        logger.info("Merging: shapefile (left) → consolidated CSV (right) on psgc_code = adm4_psgc")

        # Left join: shapefile → consolidated data
        consolidated_geo = shapefile_prepared.merge(
            self.consolidated_data,
            left_on='psgc_code',
            right_on='adm4_psgc',
            how='left',
            suffixes=('_psgc', '_shapes')
        )

        # Convert to GeoDataFrame
        self.consolidated_geodata = gpd.GeoDataFrame(
            consolidated_geo,
            geometry='geometry',
            crs=self.adm4_geodata.crs
        )

        logger.info(f"Geometry merge complete: {len(self.consolidated_geodata)} features")
        logger.info(f"Features with valid geometry: {self.consolidated_geodata.geometry.notna().sum()}")
        logger.info(f"Features with null geometry: {self.consolidated_geodata.geometry.isna().sum()}")

        # Log match statistics
        matched_count = self.consolidated_geodata['adm1_psgc'].notna().sum()
        unmatched_count = self.consolidated_geodata['adm1_psgc'].isna().sum()
        logger.info(f"Matched with CSV data: {matched_count} features")
        logger.info(f"No CSV match (shapefile only): {unmatched_count} features")

        return self.consolidated_geodata

    def _build_reference_boundaries(self) -> gpd.GeoDataFrame:
        """
        Build reference municipality boundaries from matched barangays for spatial matching.

        Dissolves matched barangay geometries to municipality level to create reference
        boundaries for spatially matching unmatched barangays.

        Returns:
            GeoDataFrame with dissolved municipality boundaries and admin codes
        """
        if self.consolidated_geodata is None:
            raise ValueError("Data not processed. Call merge_with_geometry() first.")

        logger.info("Building reference municipality boundaries for spatial matching...")

        # Get only matched barangays (those with valid admin codes)
        matched = self.consolidated_geodata[
            self.consolidated_geodata['adm1_psgc'].notna()
        ].copy()

        logger.info(f"Using {len(matched)} matched barangays to build reference boundaries")

        # Dissolve to municipality level
        municipalities = matched.dissolve(
            by=['adm1_psgc', 'adm2_psgc', 'adm3_psgc'],
            as_index=False
        ).reset_index(drop=True)

        # Keep only PSGC codes and geometry initially
        municipalities = municipalities[
            ['adm1_psgc', 'adm2_psgc', 'adm3_psgc', 'geometry']
        ]

        logger.info(f"Created {len(municipalities)} municipality reference boundaries")

        # Merge with admin-level data to get authoritative names
        # This ensures we have complete name information even if matched barangays had NaN values
        logger.info("Populating admin names from authoritative sources...")

        # Merge with Adm3 (Municipality names)
        adm3_names = self.adm3_data[['adm1_psgc', 'adm2_psgc', 'adm3_psgc', 'adm3_en']].copy()
        # Ensure PSGC codes have leading zeros
        for col in ['adm1_psgc', 'adm2_psgc', 'adm3_psgc']:
            adm3_names[col] = adm3_names[col].apply(self._add_leading_zeros).astype('string')
        municipalities = municipalities.merge(
            adm3_names,
            on=['adm1_psgc', 'adm2_psgc', 'adm3_psgc'],
            how='left'
        )

        # Merge with Adm2 (Province names)
        adm2_names = self.adm2_data[['adm1_psgc', 'adm2_psgc', 'adm2_en']].copy()
        # Ensure PSGC codes have leading zeros
        for col in ['adm1_psgc', 'adm2_psgc']:
            adm2_names[col] = adm2_names[col].apply(self._add_leading_zeros).astype('string')
        municipalities = municipalities.merge(
            adm2_names,
            on=['adm1_psgc', 'adm2_psgc'],
            how='left'
        )

        # Merge with Adm1 (Region names)
        adm1_names = self.adm1_data[['adm1_psgc', 'adm1_en']].copy()
        # Ensure PSGC codes have leading zeros
        adm1_names['adm1_psgc'] = adm1_names['adm1_psgc'].apply(self._add_leading_zeros).astype('string')
        municipalities = municipalities.merge(
            adm1_names,
            on=['adm1_psgc'],
            how='left'
        )

        # Reorder columns for consistency
        municipalities = municipalities[
            ['adm1_psgc', 'adm2_psgc', 'adm3_psgc',
             'adm1_en', 'adm2_en', 'adm3_en', 'geometry']
        ]

        # Report on name completeness
        nan_counts = municipalities[['adm1_en', 'adm2_en', 'adm3_en']].isna().sum()
        logger.info(f"Reference boundaries name completeness: "
                   f"adm1_en: {len(municipalities) - nan_counts['adm1_en']}/{len(municipalities)}, "
                   f"adm2_en: {len(municipalities) - nan_counts['adm2_en']}/{len(municipalities)}, "
                   f"adm3_en: {len(municipalities) - nan_counts['adm3_en']}/{len(municipalities)}")

        self.reference_boundaries = municipalities
        return municipalities

    def _spatial_match_unmatched(self, unmatched_gdf: gpd.GeoDataFrame,
                                  reference_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Match unmatched barangays to admin units using spatial containment.

        Uses STRtree spatial indexing for efficient point-in-polygon queries.
        Falls back to nearest neighbor for barangays on boundaries.

        Parameters:
            unmatched_gdf : GeoDataFrame
                Barangays without admin info
            reference_gdf : GeoDataFrame
                Dissolved municipality boundaries with admin codes

        Returns:
            DataFrame with matched admin codes for each unmatched barangay
        """
        from shapely.strtree import STRtree
        from shapely.prepared import prep

        logger.info(f"Starting spatial matching for {len(unmatched_gdf)} unmatched barangays...")

        # Build spatial index from reference boundaries
        ref_geoms = list(reference_gdf.geometry.values)
        tree = STRtree(ref_geoms)

        # Prepare geometries for faster containment tests
        prepared = [prep(g) for g in ref_geoms]

        # Results storage
        matches = []
        matched_count = 0
        fallback_count = 0

        for idx, row in unmatched_gdf.iterrows():
            # Use centroid for point-in-polygon (much faster than full polygon)
            centroid = row.geometry.centroid

            # Query spatial index - returns INDICES in Shapely 2.x
            candidates_idx = tree.query(centroid)

            # Handle both return types (list/array)
            if hasattr(candidates_idx, 'tolist'):
                candidates_idx = candidates_idx.tolist()
            else:
                candidates_idx = list(candidates_idx) if not isinstance(candidates_idx, list) else candidates_idx

            # Test actual containment
            matched = False
            for cand_idx in candidates_idx:
                # Fast containment test using prepared geometry
                if prepared[cand_idx].contains(centroid):
                    # Found match - extract admin codes
                    matches.append({
                        'psgc_code': row['psgc_code'],
                        'adm1_psgc': reference_gdf.iloc[cand_idx]['adm1_psgc'],
                        'adm2_psgc': reference_gdf.iloc[cand_idx]['adm2_psgc'],
                        'adm3_psgc': reference_gdf.iloc[cand_idx]['adm3_psgc'],
                        'adm1_en': reference_gdf.iloc[cand_idx]['adm1_en'],
                        'adm2_en': reference_gdf.iloc[cand_idx]['adm2_en'],
                        'adm3_en': reference_gdf.iloc[cand_idx]['adm3_en']
                    })
                    matched = True
                    matched_count += 1
                    break

            if not matched:
                # Fallback: nearest neighbor (for barangays on boundaries)
                nearest_idx = tree.nearest(centroid)
                # Handle single index return
                if hasattr(nearest_idx, '__iter__') and not isinstance(nearest_idx, str):
                    nearest_idx = list(nearest_idx)[0] if len(list(nearest_idx)) > 0 else 0

                matches.append({
                    'psgc_code': row['psgc_code'],
                    'adm1_psgc': reference_gdf.iloc[nearest_idx]['adm1_psgc'],
                    'adm2_psgc': reference_gdf.iloc[nearest_idx]['adm2_psgc'],
                    'adm3_psgc': reference_gdf.iloc[nearest_idx]['adm3_psgc'],
                    'adm1_en': reference_gdf.iloc[nearest_idx]['adm1_en'],
                    'adm2_en': reference_gdf.iloc[nearest_idx]['adm2_en'],
                    'adm3_en': reference_gdf.iloc[nearest_idx]['adm3_en']
                })
                fallback_count += 1

        logger.info(f"Spatial matching complete: {matched_count} direct matches, {fallback_count} nearest neighbor fallbacks")

        return pd.DataFrame(matches)

    def apply_spatial_matching(self, save_original: bool = True) -> gpd.GeoDataFrame:
        """
        Apply spatial matching to fill unmatched barangay admin codes.

        This method uses spatial containment to match barangays that don't have
        corresponding CSV data. It dissolves matched barangays to municipality level,
        then uses point-in-polygon queries to assign admin codes to unmatched barangays
        based on their geographic location.

        Parameters:
            save_original : bool, default True
                If True, keeps original data with NaN values in self.consolidated_geodata_original

        Returns:
            GeoDataFrame with spatial matching applied and 'is_spatially_matched' column
        """
        if self.consolidated_geodata is None:
            raise ValueError("Data not merged. Call merge_with_geometry() first.")

        logger.info("Starting spatial matching process...")

        # Save original if requested
        if save_original:
            self.consolidated_geodata_original = self.consolidated_geodata.copy()
            logger.info("Original geodata (with NaN rows) saved to self.consolidated_geodata_original")

        # Build reference boundaries from matched data
        reference = self._build_reference_boundaries()

        # Get unmatched rows
        unmatched = self.consolidated_geodata[
            self.consolidated_geodata['adm1_psgc'].isna()
        ].copy()

        unmatched_count = len(unmatched)
        logger.info(f"Found {unmatched_count} unmatched barangays to process")

        if unmatched_count == 0:
            logger.info("No unmatched barangays found. Adding is_spatially_matched column (all False)")
            self.consolidated_geodata['is_spatially_matched'] = False
            return self.consolidated_geodata

        # Run spatial matching
        matched_codes = self._spatial_match_unmatched(unmatched, reference)

        # Add is_spatially_matched column (initialize all as False)
        self.consolidated_geodata['is_spatially_matched'] = False

        # Create mask ONCE before updating (critical: don't recreate inside loop!)
        # This identifies rows that were originally unmatched
        mask = self.consolidated_geodata['adm1_psgc'].isna()
        logger.info(f"Updating {mask.sum()} rows with spatially matched admin codes")

        # Update admin codes for unmatched rows
        for col in ['adm1_psgc', 'adm2_psgc', 'adm3_psgc', 'adm1_en', 'adm2_en', 'adm3_en']:
            # Create a mapping from psgc_code to matched value
            mapping = dict(zip(matched_codes['psgc_code'], matched_codes[col]))

            # Update only unmatched rows (using the mask created before the loop)
            self.consolidated_geodata.loc[mask, col] = (
                self.consolidated_geodata.loc[mask, 'psgc_code'].map(mapping)
            )

        # Mark spatially matched rows
        matched_psgc_codes = set(matched_codes['psgc_code'])
        self.consolidated_geodata.loc[
            self.consolidated_geodata['psgc_code'].isin(matched_psgc_codes),
            'is_spatially_matched'
        ] = True

        # Log final statistics
        still_unmatched = self.consolidated_geodata['adm1_psgc'].isna().sum()
        spatially_matched = self.consolidated_geodata['is_spatially_matched'].sum()

        logger.info(f"Spatial matching results:")
        logger.info(f"  - Spatially matched: {spatially_matched} barangays")
        logger.info(f"  - Still unmatched: {still_unmatched} barangays")
        logger.info(f"  - Total features: {len(self.consolidated_geodata)}")

        return self.consolidated_geodata

    def process(self, auto_spatial_match: bool = False) -> gpd.GeoDataFrame:
        """
        Main processing pipeline: load, consolidate, and merge with geometries.

        Parameters:
            auto_spatial_match : bool, default False
                If True, automatically applies spatial matching to fill unmatched barangays

        Returns:
            GeoDataFrame with complete hierarchical geographic data
        """
        logger.info("Starting PSGC consolidation pipeline")

        # Load all data
        self.load_data()

        # Consolidate hierarchical data
        self.consolidate_hierarchy()

        # Merge with geometries
        self.merge_with_geometry()

        # Apply spatial matching if requested
        if auto_spatial_match:
            logger.info("Auto-applying spatial matching...")
            self.apply_spatial_matching(save_original=True)

        logger.info("PSGC consolidation pipeline completed successfully")
        return self.consolidated_geodata

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the consolidated data.

        Returns:
            Dictionary with summary statistics
        """
        if self.consolidated_geodata is None:
            raise ValueError("Data not processed. Call process() first.")

        summary = {
            'total_features': len(self.consolidated_geodata),
            'features_with_geometry': self.consolidated_geodata.geometry.notna().sum(),
            'features_without_geometry': self.consolidated_geodata.geometry.isna().sum(),
            'unique_regions': self.consolidated_geodata['adm1_psgc'].nunique(),
            'unique_provinces': self.consolidated_geodata['adm2_psgc'].nunique(),
            'unique_municipalities': self.consolidated_geodata['adm3_psgc'].nunique(),
            'unique_barangays': self.consolidated_geodata['adm4_psgc'].nunique(),
            'crs': str(self.consolidated_geodata.crs),
            'columns': self.consolidated_geodata.columns.tolist(),
            'total_area_km2': self.consolidated_geodata['area_km2'].sum() if 'area_km2' in self.consolidated_geodata.columns else None
        }

        # Add geographic level breakdown
        if 'geo_level' in self.consolidated_geodata.columns:
            summary['geo_level_counts'] = self.consolidated_geodata['geo_level'].value_counts().to_dict()

        return summary

    def filter_by_region(self, region_psgc: int) -> gpd.GeoDataFrame:
        """
        Filter data by region PSGC code.

        Args:
            region_psgc: Region PSGC code (e.g., 100000000 for Region I)

        Returns:
            Filtered GeoDataFrame
        """
        if self.consolidated_geodata is None:
            raise ValueError("Data not processed. Call process() first.")

        filtered = self.consolidated_geodata[
            self.consolidated_geodata['adm1_psgc'] == region_psgc
        ].copy()

        logger.info(f"Filtered to {len(filtered)} features for region {region_psgc}")
        return filtered

    def filter_by_province(self, province_psgc: int) -> gpd.GeoDataFrame:
        """
        Filter data by province PSGC code.

        Args:
            province_psgc: Province PSGC code (e.g., 102800000 for Ilocos Norte)

        Returns:
            Filtered GeoDataFrame
        """
        if self.consolidated_geodata is None:
            raise ValueError("Data not processed. Call process() first.")

        filtered = self.consolidated_geodata[
            self.consolidated_geodata['adm2_psgc'] == province_psgc
        ].copy()

        logger.info(f"Filtered to {len(filtered)} features for province {province_psgc}")
        return filtered

    def export_processed(self, output_path: str) -> None:
        """
        Export processed GeoDataFrame to file.

        Supports multiple formats based on file extension:
        - .geojson: GeoJSON format
        - .shp: Shapefile format
        - .gpkg: GeoPackage format
        - .csv: CSV format (without geometry)

        Args:
            output_path: Path for the output file
        """
        if self.consolidated_geodata is None:
            raise ValueError("Data not processed. Call process() first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        suffix = output_path.suffix.lower()

        if suffix == '.csv':
            # Export as CSV (drop geometry column)
            csv_data = self.consolidated_geodata.drop(columns=['geometry'])
            csv_data.to_csv(output_path, index=False)
            logger.info(f"Exported {len(csv_data)} records to CSV: {output_path}")
        elif suffix in ['.geojson', '.json']:
            # Export as GeoJSON
            self.consolidated_geodata.to_file(output_path, driver='GeoJSON')
            logger.info(f"Exported {len(self.consolidated_geodata)} features to GeoJSON: {output_path}")
        elif suffix == '.shp':
            # Export as Shapefile
            self.consolidated_geodata.to_file(output_path, driver='ESRI Shapefile')
            logger.info(f"Exported {len(self.consolidated_geodata)} features to Shapefile: {output_path}")
        elif suffix == '.gpkg':
            # Export as GeoPackage
            self.consolidated_geodata.to_file(output_path, driver='GPKG')
            logger.info(f"Exported {len(self.consolidated_geodata)} features to GeoPackage: {output_path}")
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Use .csv, .geojson, .shp, or .gpkg")

    def export_original(self, output_path: str) -> None:
        """
        Export original GeoDataFrame (before spatial matching) to file.

        This exports the data with unmatched barangays still containing NaN values
        in admin code columns, exactly as it was before spatial matching was applied.

        Supports multiple formats based on file extension:
        - .geojson: GeoJSON format
        - .shp: Shapefile format
        - .gpkg: GeoPackage format
        - .csv: CSV format (without geometry)

        Args:
            output_path: Path for the output file

        Raises:
            ValueError: If original data not available (spatial matching not applied)
        """
        if self.consolidated_geodata_original is None:
            raise ValueError(
                "Original data not available. Either spatial matching was not applied, "
                "or save_original=False was used in apply_spatial_matching()"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        suffix = output_path.suffix.lower()

        if suffix == '.csv':
            csv_data = self.consolidated_geodata_original.drop(columns=['geometry'])
            csv_data.to_csv(output_path, index=False)
            logger.info(f"Exported {len(csv_data)} original records to CSV: {output_path}")
        elif suffix in ['.geojson', '.json']:
            self.consolidated_geodata_original.to_file(output_path, driver='GeoJSON')
            logger.info(f"Exported {len(self.consolidated_geodata_original)} original features to GeoJSON: {output_path}")
        elif suffix == '.shp':
            self.consolidated_geodata_original.to_file(output_path, driver='ESRI Shapefile')
            logger.info(f"Exported {len(self.consolidated_geodata_original)} original features to Shapefile: {output_path}")
        elif suffix == '.gpkg':
            self.consolidated_geodata_original.to_file(output_path, driver='GPKG')
            logger.info(f"Exported {len(self.consolidated_geodata_original)} original features to GeoPackage: {output_path}")
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Use .csv, .geojson, .shp, or .gpkg")

    def export_matched(self, output_path: str) -> None:
        """
        Export spatially-matched GeoDataFrame to file.

        This exports the data with spatial matching applied. The 'is_spatially_matched'
        column indicates which barangays had their admin codes filled via spatial matching.

        Supports multiple formats based on file extension:
        - .geojson: GeoJSON format
        - .shp: Shapefile format
        - .gpkg: GeoPackage format
        - .csv: CSV format (without geometry)

        Args:
            output_path: Path for the output file

        Raises:
            ValueError: If data not processed or spatial matching not applied
        """
        if self.consolidated_geodata is None:
            raise ValueError("Data not processed. Call process() first.")

        if 'is_spatially_matched' not in self.consolidated_geodata.columns:
            raise ValueError(
                "Spatial matching not applied. Call apply_spatial_matching() first "
                "or use process(auto_spatial_match=True)"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        suffix = output_path.suffix.lower()

        if suffix == '.csv':
            csv_data = self.consolidated_geodata.drop(columns=['geometry'])
            csv_data.to_csv(output_path, index=False)
            logger.info(f"Exported {len(csv_data)} matched records to CSV: {output_path}")
        elif suffix in ['.geojson', '.json']:
            self.consolidated_geodata.to_file(output_path, driver='GeoJSON')
            logger.info(f"Exported {len(self.consolidated_geodata)} matched features to GeoJSON: {output_path}")
        elif suffix == '.shp':
            self.consolidated_geodata.to_file(output_path, driver='ESRI Shapefile')
            logger.info(f"Exported {len(self.consolidated_geodata)} matched features to Shapefile: {output_path}")
        elif suffix == '.gpkg':
            self.consolidated_geodata.to_file(output_path, driver='GPKG')
            logger.info(f"Exported {len(self.consolidated_geodata)} matched features to GeoPackage: {output_path}")
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Use .csv, .geojson, .shp, or .gpkg")

    def get_processed_data(self) -> gpd.GeoDataFrame:
        """
        Get the processed GeoDataFrame with all hierarchical data and geometries.

        Returns:
            Processed GeoDataFrame
        """
        if self.consolidated_geodata is None:
            raise ValueError("Data not processed. Call process() first.")

        return self.consolidated_geodata.copy()


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("PSGC Consolidator - Example Usage")
    print("=" * 80)

    # === Option 1: Manual control (process then optionally apply spatial matching) ===
    print("\n--- Option 1: Manual Control ---")

    # Initialize consolidator
    consolidator = PSGCConsolidator()

    # Process data (default: no spatial matching)
    geodata = consolidator.process()

    # Get summary
    summary = consolidator.get_summary()
    print("\nPSGC Consolidated Data Summary (before spatial matching):")
    print(f"Total features: {summary['total_features']:,}")
    print(f"Features with geometry: {summary['features_with_geometry']:,}")
    print(f"Unique regions: {summary['unique_regions']}")
    print(f"Unique provinces: {summary['unique_provinces']}")
    print(f"Unique municipalities: {summary['unique_municipalities']}")
    print(f"Unique barangays: {summary['unique_barangays']}")
    print(f"CRS: {summary['crs']}")

    # Export original (with NaN rows)
    consolidator.export_processed("output/psgc_consolidated_original.gpkg")
    print("\nOriginal data exported to: output/psgc_consolidated_original.gpkg")

    # Apply spatial matching
    print("\nApplying spatial matching...")
    geodata_matched = consolidator.apply_spatial_matching(save_original=True)

    # Export matched version
    consolidator.export_matched("output/psgc_consolidated_matched.gpkg")
    print("Matched data exported to: output/psgc_consolidated_matched.gpkg")

    # Also export original separately (shows NaN rows before matching)
    consolidator.export_original("output/psgc_consolidated_before_matching.gpkg")
    print("Pre-matching data exported to: output/psgc_consolidated_before_matching.gpkg")

    print("\n" + "=" * 80)
    print("\n--- Option 2: Auto-match in pipeline ---")

    # Initialize new consolidator
    consolidator2 = PSGCConsolidator()

    # Process with auto spatial matching
    geodata_auto = consolidator2.process(auto_spatial_match=True)

    print("\nData processed with automatic spatial matching")
    print(f"Spatially matched barangays: {geodata_auto['is_spatially_matched'].sum():,}")

    # Export final data
    consolidator2.export_matched("output/psgc_consolidated_auto_matched.gpkg")
    print("Auto-matched data exported to: output/psgc_consolidated_auto_matched.gpkg")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)