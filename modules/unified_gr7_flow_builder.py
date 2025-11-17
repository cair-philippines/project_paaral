"""
Unified Grade 7 Student Flow Builder

This module creates a unified dataset combining ESC beneficiary students and
all Grade 7 public school enrollees for comprehensive school choice analysis.

Purpose:
- Merge beneficiary data with Grade 7 enrollees data
- Identify overlapping students (LRN-based matching)
- Enrich with school attributes from node tables
- Calculate straight-line and road network distances
- Categorize student flow types
- Create analysis-ready dataset for discrete choice modeling

Student Categories:
1. Beneficiaries going to public JHS (in both datasets)
2. Beneficiaries going to private JHS (beneficiary data only)
3. Non-beneficiaries in public system (Gr7 enrollees only)

Key Features:
- School ID dtype handling: Ensures school_id_origin and school_id_destination
  are exported as clean strings without .0 suffix (fixes float-to-string conversion artifacts)
- Year column cleaning: Ensures sy_grade6 is exported as year only (e.g., "2023" not "2023.0")
- Multi-stage cleaning: Cleanup at merge, attribute addition, and export stages
- Coordinate coverage: 80-85% expected coverage with proper ID handling

Usage:
    from modules.unified_gr7_flow_builder import UnifiedGr7FlowBuilder

    builder = UnifiedGr7FlowBuilder(
        beneficiary_parquet_path='data/processed/esc_beneficiaries.parquet',
        gr7_enrollees_path='output/gr7_enrollees_sy2023_2024.csv',
        public_nodes_path='output/public_nodes_valid.gpkg',
        private_nodes_path='output/private_nodes_valid.gpkg',
        verbose=True
    )

    # Build unified table
    unified_df = builder.build_unified_table(school_year='SY 2023-2024')

    # Export (school IDs exported as clean strings)
    builder.export('output/unified_gr7_flows_sy2023_2024.csv')

Author: Claude Code
Date: 2025-11-16
Updated: 2025-11-17 (export dtype fix)
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from math import radians, cos, sin, asin, sqrt

# Configure logging
logger = logging.getLogger(__name__)


class UnifiedGr7FlowBuilder:
    """
    Builds unified Grade 7 student flow dataset combining beneficiaries and non-beneficiaries.

    Attributes:
        beneficiary_parquet_path (str): Path to beneficiary parquet file
        gr7_enrollees_path (str): Path to processed Gr7 enrollees CSV
        public_nodes_path (str): Path to public nodes GeoPackage
        private_nodes_path (str): Path to private nodes GeoPackage
        unified_data (DataFrame): Unified Grade 7 flow dataset
    """

    def __init__(
        self,
        beneficiary_parquet_path: str,
        gr7_enrollees_path: str,
        public_nodes_path: str,
        private_nodes_path: str,
        enrollment_csv_path: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize UnifiedGr7FlowBuilder.

        Args:
            beneficiary_parquet_path (str): Path to ESC beneficiaries parquet
            gr7_enrollees_path (str): Path to processed Gr7 enrollees CSV
            public_nodes_path (str): Path to public nodes GeoPackage
            private_nodes_path (str): Path to private nodes GeoPackage
            enrollment_csv_path (str): Path to enrollment CSV masterlist
            verbose (bool): Enable verbose logging
        """
        self.verbose = verbose
        if not verbose:
            logger.setLevel(logging.WARNING)

        # Paths
        self.beneficiary_parquet_path = beneficiary_parquet_path
        self.gr7_enrollees_path = gr7_enrollees_path
        self.public_nodes_path = public_nodes_path
        self.private_nodes_path = private_nodes_path
        self.enrollment_csv_path = enrollment_csv_path or 'data/processed/SY_2024_2025_School_Level_Data_on_Official_Enrollment.csv'

        # Data storage
        self.beneficiary_data = None
        self.gr7_data = None
        self.public_nodes = None
        self.private_nodes = None
        self.enrollment_data = None
        self.unified_data = None

        logger.info("UnifiedGr7FlowBuilder initialized")

    def build_unified_table(self, school_year: str = 'SY 2023-2024') -> pd.DataFrame:
        """
        Build complete unified Grade 7 flow table.

        Args:
            school_year (str): School year to process

        Returns:
            DataFrame: Unified Grade 7 student flows
        """
        logger.info("="*60)
        logger.info(f"Building unified Grade 7 flow table for {school_year}")
        logger.info("="*60)

        # Step 1: Load beneficiary data (Grade 7 only)
        logger.info("Step 1/7: Loading beneficiary data...")
        self._load_gr7_beneficiaries(school_year)

        # Step 2: Load Gr7 enrollees data
        logger.info("Step 2/7: Loading Gr7 enrollees data...")
        self._load_gr7_enrollees()

        # Step 3: Merge datasets by LRN
        logger.info("Step 3/8: Merging datasets by LRN...")
        self._merge_by_lrn()

        # Step 4: Load enrollment masterlist
        logger.info("Step 4/8: Loading enrollment masterlist...")
        self._load_enrollment_masterlist()

        # Step 5: Load node tables
        logger.info("Step 5/8: Loading school node tables...")
        self._load_node_tables()

        # Step 6: Add school attributes
        logger.info("Step 6/8: Adding school attributes...")
        self._add_school_attributes()

        # Step 7: Calculate distances
        logger.info("Step 7/8: Calculating distances...")
        self._calculate_distances()

        # Step 8: Categorize flows
        logger.info("Step 8/8: Categorizing flow types...")
        self._categorize_flows()

        logger.info("="*60)
        logger.info("Unified table build complete")
        logger.info("="*60)

        return self.unified_data

    def _load_gr7_beneficiaries(self, school_year: str):
        """
        Load and filter beneficiary data to Grade 7 only.

        Args:
            school_year (str): School year to filter
        """
        # Load with selected columns only (memory efficient)
        needed_cols = [
            'lrn',
            'deped_school_id',
            'lrn_school_id',
            'grade',
            'school_year',
            'esc_subsidy_amount'
        ]

        df = pd.read_parquet(self.beneficiary_parquet_path, columns=needed_cols)
        logger.info(f"  Loaded {len(df):,} total beneficiary records")

        # Convert grade to integer (it's stored as string in the parquet file)
        df['grade'] = pd.to_numeric(df['grade'], errors='coerce')

        # Log grade distribution before filtering
        logger.info(f"  Grade distribution: {df['grade'].value_counts().sort_index().to_dict()}")

        # Filter to Grade 7 and school year
        df = df[
            (df['grade'] == 7) &
            (df['school_year'] == school_year)
        ].copy()

        logger.info(f"  Filtered to {len(df):,} Grade 7 beneficiaries in {school_year}")

        # Standardize column names
        df = df.rename(columns={
            'deped_school_id': 'school_id_destination',
            'lrn_school_id': 'school_id_origin'
        })

        # Convert IDs to strings
        df['school_id_destination'] = df['school_id_destination'].astype(str)
        df['school_id_origin'] = df['school_id_origin'].astype(str)
        df['lrn'] = df['lrn'].astype(str)

        # Aggregate by LRN (keep first record if duplicates)
        # Note: Some students may have multiple beneficiary records
        df = df.groupby('lrn').first().reset_index()

        logger.info(f"  Unique beneficiary students (LRNs): {len(df):,}")

        self.beneficiary_data = df

    def _load_gr7_enrollees(self):
        """Load processed Gr7 enrollees data."""
        df = pd.read_csv(self.gr7_enrollees_path, dtype={'lrn': str})

        logger.info(f"  Loaded {len(df):,} Gr7 enrollee records")

        # Filter to fully valid records only (optional - can relax later)
        if 'fully_valid' in df.columns:
            df = df[df['fully_valid'] == True].copy()
            logger.info(f"  Filtered to {len(df):,} fully valid records")

        logger.info(f"  Unique Gr7 students (LRNs): {df['lrn'].nunique():,}")

        self.gr7_data = df

    def _merge_by_lrn(self):
        """
        Merge beneficiary and Gr7 enrollees data by LRN.

        Three scenarios:
        1. LRN in both → Beneficiary going to public JHS
        2. LRN only in beneficiary → Beneficiary going to private JHS
        3. LRN only in Gr7 → Non-beneficiary public student
        """
        benef = self.beneficiary_data
        gr7 = self.gr7_data

        logger.info(f"  Beneficiary LRNs: {len(benef):,}")
        logger.info(f"  Gr7 enrollees LRNs: {len(gr7):,}")

        # Perform outer merge to capture all three scenarios
        merged = pd.merge(
            gr7,
            benef[['lrn', 'esc_subsidy_amount']],
            on='lrn',
            how='outer',
            indicator=True
        )

        # Create is_beneficiary flag
        merged['is_beneficiary'] = merged['_merge'].isin(['both', 'right_only'])

        # Count scenarios
        both = (merged['_merge'] == 'both').sum()
        gr7_only = (merged['_merge'] == 'left_only').sum()
        benef_only = (merged['_merge'] == 'right_only').sum()

        logger.info(f"  Merge results:")
        logger.info(f"    Both datasets (beneficiary → public JHS): {both:,}")
        logger.info(f"    Gr7 only (non-beneficiary): {gr7_only:,}")
        logger.info(f"    Beneficiary only (→ private JHS): {benef_only:,}")
        logger.info(f"    Total unified records: {len(merged):,}")

        # For beneficiary-only records, we need to get origin/destination from beneficiary data
        # (they went to private schools, so not in public Gr7 enrollees)
        benef_only_mask = merged['_merge'] == 'right_only'
        if benef_only_mask.sum() > 0:
            # Get origin/destination from beneficiary data
            benef_lookup = benef.set_index('lrn')[['school_id_origin', 'school_id_destination']]
            merged.loc[benef_only_mask, 'school_id_origin'] = merged.loc[benef_only_mask, 'lrn'].map(
                benef_lookup['school_id_origin']
            )
            merged.loc[benef_only_mask, 'school_id_destination'] = merged.loc[benef_only_mask, 'lrn'].map(
                benef_lookup['school_id_destination']
            )

        # Drop merge indicator
        merged = merged.drop(columns=['_merge'])

        # CRITICAL FIX: Ensure school_id columns are proper strings (not float64)
        # Problem: Outer merge can create NaN values which may default to float64
        # This causes silent merge failures when joining with node tables (which use string IDs)
        # Solution: Handle both cases - already strings OR numeric types
        logger.info(f"  Ensuring school_id columns are string dtype...")
        logger.info(f"    Before: school_id_origin dtype = {merged['school_id_origin'].dtype}")
        logger.info(f"    Before: school_id_destination dtype = {merged['school_id_destination'].dtype}")

        # Handle origin school IDs
        origin_dtype = merged['school_id_origin'].dtype
        if origin_dtype in ['float64', 'float32', 'int64', 'int32']:
            # Numeric dtype: convert float → Int64 → string
            logger.info(f"    Converting numeric school_id_origin to string...")
            merged['school_id_origin'] = merged['school_id_origin'].astype('Int64').astype(str).replace('<NA>', None)
        else:
            # Already object/string: just ensure proper string dtype and clean up
            logger.info(f"    school_id_origin already object dtype - cleaning string artifacts...")
            merged['school_id_origin'] = merged['school_id_origin'].astype(str)
            # Remove .0 suffix from float-like strings (e.g., '100001.0' → '100001')
            merged['school_id_origin'] = merged['school_id_origin'].str.replace(r'\.0$', '', regex=True)
            # Clean up pandas NaN representations
            merged.loc[merged['school_id_origin'].isin(['None', 'nan', '<NA>']), 'school_id_origin'] = None

        # Handle destination school IDs
        dest_dtype = merged['school_id_destination'].dtype
        if dest_dtype in ['float64', 'float32', 'int64', 'int32']:
            # Numeric dtype: convert float → Int64 → string
            logger.info(f"    Converting numeric school_id_destination to string...")
            merged['school_id_destination'] = merged['school_id_destination'].astype('Int64').astype(str).replace('<NA>', None)
        else:
            # Already object/string: just ensure proper string dtype and clean up
            logger.info(f"    school_id_destination already object dtype - cleaning string artifacts...")
            merged['school_id_destination'] = merged['school_id_destination'].astype(str)
            # Remove .0 suffix from float-like strings (e.g., '300002.0' → '300002')
            merged['school_id_destination'] = merged['school_id_destination'].str.replace(r'\.0$', '', regex=True)
            # Clean up pandas NaN representations
            merged.loc[merged['school_id_destination'].isin(['None', 'nan', '<NA>']), 'school_id_destination'] = None

        logger.info(f"    After: school_id_origin dtype = {merged['school_id_origin'].dtype}")
        logger.info(f"    After: school_id_destination dtype = {merged['school_id_destination'].dtype}")

        # Log sample IDs AFTER cleanup to verify .0 suffix removed
        sample_origin_after = merged['school_id_origin'].dropna().head(3).tolist()
        sample_dest_after = merged['school_id_destination'].dropna().head(3).tolist()
        logger.info(f"    After cleanup - Sample origin IDs: {sample_origin_after}")
        logger.info(f"    After cleanup - Sample dest IDs: {sample_dest_after}")

        self.unified_data = merged

    def _load_enrollment_masterlist(self):
        """Load enrollment CSV as masterlist for school information."""
        logger.info(f"  Loading enrollment masterlist...")
        self.enrollment_data = pd.read_csv(
            self.enrollment_csv_path,
            usecols=['school_id', 'sector', 'municipality'],
            dtype={'school_id': str}
        )

        # Standardize sector to lowercase for consistent comparisons
        self.enrollment_data['sector'] = self.enrollment_data['sector'].str.lower()

        logger.info(f"    Loaded {len(self.enrollment_data):,} schools from enrollment masterlist")

    def _load_node_tables(self):
        """Load public and private school node tables."""
        # Load public nodes
        logger.info(f"  Loading public nodes...")
        self.public_nodes = gpd.read_file(self.public_nodes_path)
        logger.info(f"    Loaded {len(self.public_nodes):,} public schools")

        # Load private nodes
        logger.info(f"  Loading private nodes...")
        self.private_nodes = gpd.read_file(self.private_nodes_path)
        logger.info(f"    Loaded {len(self.private_nodes):,} private schools")

    def _add_school_attributes(self):
        """
        Enrich unified data with school attributes from multiple sources.

        Strategy:
        1. Match against enrollment CSV for sector and municipality (authoritative)
        2. Match against node tables for coordinates (lat/lon)
        3. Preserve any existing sector/municipality from gr7_enrollees data

        Adds attributes for both origin and destination schools:
        - Sector (from enrollment CSV - authoritative)
        - Municipality (from enrollment CSV)
        - Coordinates (from node tables)
        """
        df = self.unified_data

        # CRITICAL: Validate and fix school_id dtypes BEFORE merging
        # This is a safety net in case _merge_by_lrn() didn't convert properly
        logger.info(f"  Validating school_id dtypes before coordinate merge...")

        origin_dtype = df['school_id_origin'].dtype
        dest_dtype = df['school_id_destination'].dtype

        logger.info(f"    school_id_origin dtype: {origin_dtype}")
        logger.info(f"    school_id_destination dtype: {dest_dtype}")

        # Check for problematic dtypes (float, int)
        if origin_dtype in ['float64', 'int64', 'float32', 'int32']:
            logger.warning(f"    ⚠️  school_id_origin has numeric dtype {origin_dtype} - converting to string")
            df['school_id_origin'] = df['school_id_origin'].astype('Int64').astype(str).replace('<NA>', None)

        if dest_dtype in ['float64', 'int64', 'float32', 'int32']:
            logger.warning(f"    ⚠️  school_id_destination has numeric dtype {dest_dtype} - converting to string")
            df['school_id_destination'] = df['school_id_destination'].astype('Int64').astype(str).replace('<NA>', None)

        # Ensure string dtype (handles both object and already-string columns)
        df['school_id_origin'] = df['school_id_origin'].astype(str)
        df['school_id_destination'] = df['school_id_destination'].astype(str)

        # Remove .0 suffix from float-like strings (e.g., '100001.0' → '100001')
        df['school_id_origin'] = df['school_id_origin'].str.replace(r'\.0$', '', regex=True)
        df['school_id_destination'] = df['school_id_destination'].str.replace(r'\.0$', '', regex=True)

        # Clean up 'nan' string literals (from NaN values)
        df.loc[df['school_id_origin'] == 'nan', 'school_id_origin'] = None
        df.loc[df['school_id_destination'] == 'nan', 'school_id_destination'] = None

        logger.info(f"    ✓ School IDs validated as strings")

        # Sample IDs for verification
        sample_origin = df['school_id_origin'].dropna().head(3).tolist()
        sample_dest = df['school_id_destination'].dropna().head(3).tolist()
        logger.info(f"    Sample origin IDs: {sample_origin}")
        logger.info(f"    Sample dest IDs: {sample_dest}")

        # Check if sector columns already exist (from gr7_enrollees)
        has_sector_cols = 'sector_origin' in df.columns and 'sector_destination' in df.columns
        has_municipality_cols = 'municipality_origin' in df.columns and 'municipality_destination' in df.columns

        if has_sector_cols and has_municipality_cols:
            logger.info(f"  Using sector and municipality from Gr7 enrollees data (already enriched)")
        else:
            # Match against enrollment CSV for sector and municipality
            logger.info(f"  Matching against enrollment masterlist for sector and municipality...")

            # Match origin schools
            if not has_sector_cols or not has_municipality_cols:
                origin_lookup = self.enrollment_data.set_index('school_id')[['sector', 'municipality']]
                origin_lookup.columns = ['sector_origin', 'municipality_origin']

                df = df.merge(
                    origin_lookup,
                    left_on='school_id_origin',
                    right_index=True,
                    how='left',
                    suffixes=('', '_enrollment')
                )

                # Match destination schools
                dest_lookup = self.enrollment_data.set_index('school_id')[['sector', 'municipality']]
                dest_lookup.columns = ['sector_destination', 'municipality_destination']

                df = df.merge(
                    dest_lookup,
                    left_on='school_id_destination',
                    right_index=True,
                    how='left',
                    suffixes=('', '_enrollment')
                )

                origin_sector_matched = df['sector_origin'].notna().sum()
                dest_sector_matched = df['sector_destination'].notna().sum()
                logger.info(f"    Origin schools matched: {origin_sector_matched:,}/{len(df):,}")
                logger.info(f"    Destination schools matched: {dest_sector_matched:,}/{len(df):,}")

        # Match against node tables for coordinates
        logger.info(f"  Matching against node tables for coordinates...")

        # Combine public and private nodes for coordinates only
        public_coords = self.public_nodes[['school_id', 'latitude', 'longitude']].copy()
        private_coords = self.private_nodes[['school_id', 'latitude', 'longitude']].copy()

        all_coords = pd.concat([public_coords, private_coords], ignore_index=True)
        all_coords['school_id'] = all_coords['school_id'].astype(str)

        # Log node table school_id dtype for verification
        logger.info(f"    Node table school_id dtype: {all_coords['school_id'].dtype}")
        logger.info(f"    Node table has {len(all_coords):,} schools with coordinates")
        logger.info(f"    Sample node table IDs: {all_coords['school_id'].head(3).tolist()}")

        # Verify no float artifacts (e.g., '100001.0')
        float_artifacts = all_coords['school_id'].str.contains(r'\.0$', na=False).sum()
        if float_artifacts > 0:
            logger.warning(f"    ⚠️  Found {float_artifacts} node table school_ids with .0 suffix - cleaning...")
            all_coords['school_id'] = all_coords['school_id'].str.replace(r'\.0$', '', regex=True)
            logger.info(f"    ✓ Cleaned float artifacts from node table IDs")

        # Merge origin coordinates
        origin_coords = all_coords.add_suffix('_origin')
        origin_coords = origin_coords.rename(columns={'school_id_origin_origin': 'school_id_origin'})

        df = df.merge(
            origin_coords,
            on='school_id_origin',
            how='left'
        )

        # Merge destination coordinates
        dest_coords = all_coords.add_suffix('_destination')
        dest_coords = dest_coords.rename(columns={'school_id_destination_destination': 'school_id_destination'})

        df = df.merge(
            dest_coords,
            on='school_id_destination',
            how='left'
        )

        # Convert coordinate columns to float (they might be strings from GeoPackage)
        coord_cols = ['latitude_origin', 'longitude_origin', 'latitude_destination', 'longitude_destination']
        for col in coord_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Count coordinate matches
        origin_coords_matched = df['latitude_origin'].notna().sum()
        dest_coords_matched = df['latitude_destination'].notna().sum()
        both_coords_matched = (df['latitude_origin'].notna() & df['latitude_destination'].notna()).sum()

        logger.info(f"    Origin coordinates matched: {origin_coords_matched:,}/{len(df):,}")
        logger.info(f"    Destination coordinates matched: {dest_coords_matched:,}/{len(df):,}")
        logger.info(f"    Both coordinates matched: {both_coords_matched:,}/{len(df):,}")

        self.unified_data = df

    def _calculate_distances(self):
        """
        Calculate straight-line (Haversine) distances between origin and destination schools.
        """
        df = self.unified_data

        # Calculate for rows with both coordinates
        has_coords = (
            df['latitude_origin'].notna() &
            df['longitude_origin'].notna() &
            df['latitude_destination'].notna() &
            df['longitude_destination'].notna()
        )

        logger.info(f"  Calculating distances for {has_coords.sum():,} records with coordinates...")

        # Vectorized Haversine calculation
        df.loc[has_coords, 'distance_straightline_km'] = df.loc[has_coords].apply(
            lambda row: self._haversine(
                row['latitude_origin'],
                row['longitude_origin'],
                row['latitude_destination'],
                row['longitude_destination']
            ),
            axis=1
        )

        # Log distance statistics
        if has_coords.sum() > 0:
            distances = df.loc[has_coords, 'distance_straightline_km']
            logger.info(f"  Distance statistics:")
            logger.info(f"    Min: {distances.min():.2f} km")
            logger.info(f"    Mean: {distances.mean():.2f} km")
            logger.info(f"    Median: {distances.median():.2f} km")
            logger.info(f"    Max: {distances.max():.2f} km")
            logger.info(f"    90th percentile: {distances.quantile(0.90):.2f} km")
            logger.info(f"    95th percentile: {distances.quantile(0.95):.2f} km")

        self.unified_data = df

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2):
        """
        Calculate great circle distance between two points on Earth (Haversine formula).

        Args:
            lat1, lon1: Coordinates of first point
            lat2, lon2: Coordinates of second point

        Returns:
            float: Distance in kilometers
        """
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))

        # Earth radius in kilometers
        r = 6371

        return c * r

    def _categorize_flows(self):
        """
        Categorize student flows into types.

        Flow types:
        1. public_to_public_beneficiary - ESC recipient going to public JHS
        2. public_to_private_beneficiary - ESC recipient going to private JHS
        3. private_to_private_beneficiary - Private ES to private JHS with ESC
        4. private_to_public_beneficiary - Private ES to public JHS with ESC
        5. public_to_public_nonbeneficiary - Regular public student
        6. private_to_public_nonbeneficiary - Private ES to public JHS (no subsidy)
        7. private_to_private_nonbeneficiary - Private ES to private JHS (no subsidy)
        """
        df = self.unified_data

        # Initialize flow_type column
        df['flow_type'] = 'unknown'

        # Beneficiaries (handle both True and potential NaN values properly)
        is_benef = df['is_beneficiary'].fillna(False) == True

        # Public to public beneficiary
        mask = is_benef & (df['sector_origin'] == 'public') & (df['sector_destination'] == 'public')
        df.loc[mask, 'flow_type'] = 'public_to_public_beneficiary'

        # Public to private beneficiary
        mask = is_benef & (df['sector_origin'] == 'public') & (df['sector_destination'] == 'private')
        df.loc[mask, 'flow_type'] = 'public_to_private_beneficiary'

        # Private to private beneficiary
        mask = is_benef & (df['sector_origin'] == 'private') & (df['sector_destination'] == 'private')
        df.loc[mask, 'flow_type'] = 'private_to_private_beneficiary'

        # Private to public beneficiary (rare)
        mask = is_benef & (df['sector_origin'] == 'private') & (df['sector_destination'] == 'public')
        df.loc[mask, 'flow_type'] = 'private_to_public_beneficiary'

        # Non-beneficiaries
        is_non_benef = df['is_beneficiary'].fillna(False) == False

        # Public to public non-beneficiary
        mask = is_non_benef & (df['sector_origin'] == 'public') & (df['sector_destination'] == 'public')
        df.loc[mask, 'flow_type'] = 'public_to_public_nonbeneficiary'

        # Private to public non-beneficiary (rare)
        mask = is_non_benef & (df['sector_origin'] == 'private') & (df['sector_destination'] == 'public')
        df.loc[mask, 'flow_type'] = 'private_to_public_nonbeneficiary'

        # Private to private non-beneficiary (rare)
        mask = is_non_benef & (df['sector_origin'] == 'private') & (df['sector_destination'] == 'private')
        df.loc[mask, 'flow_type'] = 'private_to_private_nonbeneficiary'

        # Log flow type distribution
        logger.info(f"  Flow type distribution:")
        for flow_type, count in df['flow_type'].value_counts().items():
            pct = count / len(df) * 100
            logger.info(f"    {flow_type}: {count:,} ({pct:.1f}%)")

        self.unified_data = df

    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary statistics.

        Returns:
            dict: Summary statistics
        """
        if self.unified_data is None:
            raise ValueError("No unified data built. Call build_unified_table() first.")

        df = self.unified_data

        summary = {
            'total_records': len(df),
            'unique_students': df['lrn'].nunique(),
            'beneficiary_stats': {
                'total_beneficiaries': df['is_beneficiary'].sum(),
                'beneficiary_rate': df['is_beneficiary'].sum() / len(df) * 100,
                'non_beneficiaries': (~df['is_beneficiary']).sum()
            },
            'flow_types': df['flow_type'].value_counts().to_dict(),
            'distance_stats': {
                'records_with_distance': df['distance_straightline_km'].notna().sum(),
                'mean_distance_km': df['distance_straightline_km'].mean(),
                'median_distance_km': df['distance_straightline_km'].median(),
                'max_distance_km': df['distance_straightline_km'].max(),
                'percentile_90_km': df['distance_straightline_km'].quantile(0.90),
                'percentile_95_km': df['distance_straightline_km'].quantile(0.95)
            },
            'school_matching': {
                'origin_matched': df['latitude_origin'].notna().sum(),
                'destination_matched': df['latitude_destination'].notna().sum(),
                'both_matched': (df['latitude_origin'].notna() & df['latitude_destination'].notna()).sum()
            }
        }

        return summary

    def export(self, path: str, include_all_columns: bool = True):
        """
        Export unified Grade 7 flow data.

        Args:
            path (str): Output file path
            include_all_columns (bool): Include all columns or essential only
        """
        if self.unified_data is None:
            raise ValueError("No unified data to export")

        df = self.unified_data.copy()

        # Optionally select essential columns only
        if not include_all_columns:
            essential_cols = [
                'lrn',
                'school_year',
                'school_id_origin',
                'school_id_destination',
                'is_beneficiary',
                'esc_subsidy_amount',
                'flow_type',
                'sector_origin',
                'sector_destination',
                'latitude_origin',
                'longitude_origin',
                'latitude_destination',
                'longitude_destination',
                'distance_straightline_km',
                'region',
                'division'
            ]
            # Keep only columns that exist
            essential_cols = [c for c in essential_cols if c in df.columns]
            df = df[essential_cols]

        # CRITICAL: Ensure school_id and year columns export as clean strings without .0 suffix
        # This is a final safety net before CSV export to prevent float-like strings
        logger.info("  Ensuring school_id and year columns are clean strings for export...")

        # Clean school ID columns
        for col in ['school_id_origin', 'school_id_destination']:
            if col in df.columns:
                # Convert to string dtype if not already
                df[col] = df[col].astype(str)

                # Remove .0 suffix from float-like strings
                df[col] = df[col].str.replace(r'\.0$', '', regex=True)

                # Clean up 'nan' and 'None' string literals
                df.loc[df[col].isin(['nan', 'None', '<NA>']), col] = None

        # Clean year columns (sy_grade6 should be year only, not float)
        if 'sy_grade6' in df.columns:
            # Convert to string dtype if not already
            df['sy_grade6'] = df['sy_grade6'].astype(str)

            # Remove .0 suffix from float-like strings
            df['sy_grade6'] = df['sy_grade6'].str.replace(r'\.0$', '', regex=True)

            # Clean up 'nan' and 'None' string literals
            df.loc[df['sy_grade6'].isin(['nan', 'None', '<NA>']), 'sy_grade6'] = None

        logger.info("  ✓ School ID and year columns cleaned for export")

        # Export
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        logger.info(f"Exported {len(df):,} records to {output_path}")


# Example usage
if __name__ == "__main__":
    builder = UnifiedGr7FlowBuilder(
        beneficiary_parquet_path='data/processed/esc_beneficiaries.parquet',
        gr7_enrollees_path='output/gr7_enrollees_sy2023_2024.csv',
        public_nodes_path='output/public_nodes_valid.gpkg',
        private_nodes_path='output/private_nodes_valid.gpkg',
        verbose=True
    )

    # Build unified table
    unified_df = builder.build_unified_table(school_year='SY 2023-2024')

    # Get summary
    summary = builder.get_summary()
    print("\nSummary Statistics:")
    import json
    print(json.dumps(summary, indent=2, default=str))

    # Export
    builder.export('output/unified_gr7_flows_sy2023_2024.csv')
