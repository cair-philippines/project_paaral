"""
Grade 7 Enrollees Data Processor

This module processes Grade 7 enrollment transition data tracking student flows
from Grade 6 (elementary) to Grade 7 (junior high school).

Purpose:
- Load Grade 7 enrollees data from Excel files (by school year)
- Standardize school IDs and column names
- Validate against BOTH public and private school node tables
- Create student-level flow records (origin → destination)
- Support unified Grade 7 analysis combining beneficiary and non-beneficiary students

Student Flow Direction: ORIGIN (Grade 6 School) → DESTINATION (Grade 7 School)

Important Notes:
1. **Dataset Scope:** This dataset contains ALL Grade 7 enrollments, including:
   - Public to Public transitions (majority)
   - Public to Private transitions
   - Private to Private transitions
   - Private to Public transitions (rare)

2. **Column Name Typo:** The original Excel files have a typo in the Grade 7 column name:
   "School Namein Grade 7" (no space between "Name" and "in"). The Grade 6 column is
   correct: "School Name in Grade 6". Both are standardized during processing.

Key Features:
- Memory-efficient Excel loading (selected columns only)
- School ID validation (6-digit format check)
- Integration with BOTH public and private node tables
- Multi-year support (SY 2022-2023, 2023-2024, 2024-2025)
- Validation flags consistent with other processors
- Sector identification (public/private) for origin and destination schools

Usage:
    from modules.gr7_enrollees_processor import Gr7EnrolleesProcessor

    processor = Gr7EnrolleesProcessor(
        public_nodes_path='output/public_nodes_valid.gpkg',
        verbose=True
    )

    # Process single year
    df = processor.load_school_year('SY 2023-2024')

    # Process all years
    all_years_df = processor.load_all_years()

    # Validate and export
    validated = processor.validate_school_ids()
    processor.export('output/gr7_enrollees_processed.csv')

Author: Claude Code
Date: 2025-11-16
"""

import pandas as pd
import geopandas as gpd
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import re

# Configure logging
logger = logging.getLogger(__name__)


class Gr7EnrolleesProcessor:
    """
    Processes Grade 7 enrollment transition data.

    Tracks student flows from Grade 6 (origin) to Grade 7 (destination).
    Includes ALL transitions: public-to-public, public-to-private, private-to-private,
    and private-to-public.

    Attributes:
        public_nodes_path (str): Path to public nodes GeoPackage
        private_nodes_path (str): Path to private nodes GeoPackage
        data_dir (str): Directory containing Grade 7 enrollees Excel files
        gr7_data (DataFrame): Processed Grade 7 enrollees data
        public_nodes (GeoDataFrame): Public school nodes for validation
        private_nodes (GeoDataFrame): Private school nodes for validation
        valid_public_ids (set): Set of valid public school IDs
        valid_private_ids (set): Set of valid private school IDs
    """

    def __init__(
        self,
        public_nodes_path: Optional[str] = None,
        private_nodes_path: Optional[str] = None,
        enrollment_csv_path: Optional[str] = None,
        data_dir: str = 'data/public',
        verbose: bool = True
    ):
        """
        Initialize Gr7EnrolleesProcessor.

        Args:
            public_nodes_path (str): Path to public nodes GeoPackage
            private_nodes_path (str): Path to private nodes GeoPackage
            enrollment_csv_path (str): Path to enrollment CSV masterlist
            data_dir (str): Directory containing Gr 7 Enrollees Excel files
            verbose (bool): Enable verbose logging
        """
        self.verbose = verbose
        if not verbose:
            logger.setLevel(logging.WARNING)

        # Paths
        self.public_nodes_path = public_nodes_path or 'output/public_nodes_valid.gpkg'
        self.private_nodes_path = private_nodes_path or 'output/private_nodes_valid.gpkg'
        self.enrollment_csv_path = enrollment_csv_path or 'data/processed/SY_2024_2025_School_Level_Data_on_Official_Enrollment.csv'
        self.data_dir = Path(data_dir)

        # Data storage
        self.gr7_data = None
        self.public_nodes = None
        self.private_nodes = None
        self.enrollment_data = None
        self.valid_public_ids = None
        self.valid_private_ids = None

        logger.info("Gr7EnrolleesProcessor initialized")

    def load_school_year(
        self,
        school_year: str,
        data_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load Grade 7 enrollees data for a specific school year.

        Args:
            school_year (str): School year (e.g., 'SY 2023-2024')
            data_dir (str): Optional override for data directory

        Returns:
            DataFrame: Processed Grade 7 enrollees data
        """
        logger.info(f"Loading Grade 7 enrollees data for {school_year}...")

        # Construct file path
        data_path = Path(data_dir) if data_dir else self.data_dir
        file_path = data_path / f"{school_year} Gr 7 Enrollees.xlsx"

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load Excel with selected columns only (memory efficient)
        logger.info(f"  Reading {file_path.name}...")
        df = pd.read_excel(
            file_path,
            sheet_name=0,
            usecols=[
                'Region',
                'Division',
                'LRN',
                'BEIS School ID in Grade 7',
                'School Namein Grade 7',  # Note: Typo in source data - no space between "Name" and "in"
                'SY enrolled in Grade 6',
                'BEIS School ID in Grade 6',
                'School Name in Grade 6'  # Note: Correct spacing in source data
            ],
            dtype={
                'LRN': str,  # Keep as string to preserve leading zeros
                'BEIS School ID in Grade 7': str,
                'BEIS School ID in Grade 6': str,
            }
        )

        logger.info(f"  Loaded {len(df):,} student records")

        # Standardize column names
        df = df.rename(columns={
            'LRN': 'lrn',
            'BEIS School ID in Grade 7': 'school_id_destination',
            'School Namein Grade 7': 'school_name_destination',  # Note: Source has typo (no space)
            'BEIS School ID in Grade 6': 'school_id_origin',
            'School Name in Grade 6': 'school_name_origin',  # Note: Source has correct spacing
            'SY enrolled in Grade 6': 'sy_grade6',
            'Region': 'region',
            'Division': 'division'
        })

        # Add school_year column
        df['school_year'] = school_year

        # Clean and validate school IDs
        df = self._standardize_school_ids(df)

        # Store in instance variable
        self.gr7_data = df

        return df

    def load_all_years(
        self,
        years: Optional[List[str]] = None,
        data_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load and concatenate all Grade 7 enrollees data across school years.

        Args:
            years (list): List of school years to load (default: all available)
            data_dir (str): Optional override for data directory

        Returns:
            DataFrame: Combined Grade 7 enrollees data across all years
        """
        logger.info("Loading all Grade 7 enrollees data...")

        # Default years
        if years is None:
            years = ['SY 2022-2023', 'SY 2023-2024', 'SY 2024-2025']

        # Load each year
        dfs = []
        for year in years:
            try:
                df_year = self.load_school_year(year, data_dir)
                dfs.append(df_year)
                logger.info(f"  Loaded {year}: {len(df_year):,} records")
            except FileNotFoundError:
                logger.warning(f"  File not found for {year}, skipping")

        # Concatenate
        if not dfs:
            raise ValueError("No data files found")

        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined data: {len(combined):,} total records across {len(dfs)} years")

        self.gr7_data = combined
        return combined

    def _standardize_school_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize and validate school IDs.

        Valid school IDs:
        - Exactly 6 digits long
        - Not equal to '0' or '000000'
        - Numeric only

        Args:
            df (DataFrame): DataFrame with school_id columns

        Returns:
            DataFrame: DataFrame with standardized IDs and validation flags
        """
        logger.info("  Standardizing school IDs...")

        # Process origin school IDs
        df['school_id_origin_valid'] = df['school_id_origin'].apply(self._is_valid_school_id)

        # Process destination school IDs
        df['school_id_destination_valid'] = df['school_id_destination'].apply(self._is_valid_school_id)

        # Both IDs valid flag
        df['both_school_ids_valid'] = (
            df['school_id_origin_valid'] & df['school_id_destination_valid']
        )

        # Log validation results
        total = len(df)
        origin_valid = df['school_id_origin_valid'].sum()
        dest_valid = df['school_id_destination_valid'].sum()
        both_valid = df['both_school_ids_valid'].sum()

        logger.info(f"  School ID validation:")
        logger.info(f"    Origin IDs valid: {origin_valid:,}/{total:,} ({origin_valid/total*100:.1f}%)")
        logger.info(f"    Destination IDs valid: {dest_valid:,}/{total:,} ({dest_valid/total*100:.1f}%)")
        logger.info(f"    Both IDs valid: {both_valid:,}/{total:,} ({both_valid/total*100:.1f}%)")

        return df

    @staticmethod
    def _is_valid_school_id(school_id: str) -> bool:
        """
        Check if school ID is valid (6 digits, not '0').

        Args:
            school_id (str): School ID to validate

        Returns:
            bool: True if valid, False otherwise
        """
        if pd.isna(school_id):
            return False

        school_id_str = str(school_id).strip()

        # Check if numeric
        if not school_id_str.isdigit():
            return False

        # Check length
        if len(school_id_str) != 6:
            return False

        # Check not all zeros or single zero
        if school_id_str in ('0', '000000'):
            return False

        return True

    def load_enrollment_masterlist(self):
        """Load enrollment CSV as masterlist for school information."""
        if self.enrollment_data is None:
            logger.info(f"Loading enrollment masterlist from {Path(self.enrollment_csv_path).name}...")

            if not Path(self.enrollment_csv_path).exists():
                logger.warning(f"  Enrollment CSV not found: {self.enrollment_csv_path}")
                self.enrollment_data = pd.DataFrame()
            else:
                self.enrollment_data = pd.read_csv(
                    self.enrollment_csv_path,
                    usecols=['school_id', 'sector', 'municipality'],
                    dtype={'school_id': str}
                )

                # Standardize sector to lowercase for consistent comparisons
                self.enrollment_data['sector'] = self.enrollment_data['sector'].str.lower()

                logger.info(f"  Loaded {len(self.enrollment_data):,} schools from enrollment masterlist")
        else:
            logger.debug("  Using cached enrollment masterlist")

    def load_node_tables(self):
        """Load both public and private school nodes for validation."""
        # Load public nodes
        if self.public_nodes is None:
            logger.info(f"Loading public nodes from {Path(self.public_nodes_path).name}...")

            if not Path(self.public_nodes_path).exists():
                logger.warning(f"  Public nodes file not found: {self.public_nodes_path}")
                self.valid_public_ids = set()
            else:
                self.public_nodes = gpd.read_file(self.public_nodes_path)
                self.valid_public_ids = set(self.public_nodes['school_id'].astype(str))
                logger.info(f"  Loaded {len(self.public_nodes):,} public schools")
        else:
            logger.debug("  Using cached public nodes")

        # Load private nodes
        if self.private_nodes is None:
            logger.info(f"Loading private nodes from {Path(self.private_nodes_path).name}...")

            if not Path(self.private_nodes_path).exists():
                logger.warning(f"  Private nodes file not found: {self.private_nodes_path}")
                self.valid_private_ids = set()
            else:
                self.private_nodes = gpd.read_file(self.private_nodes_path)
                self.valid_private_ids = set(self.private_nodes['school_id'].astype(str))
                logger.info(f"  Loaded {len(self.private_nodes):,} private schools")
        else:
            logger.debug("  Using cached private nodes")

    def validate_against_nodes(self) -> pd.DataFrame:
        """
        Validate school IDs and enrich with school information.

        Steps:
        1. Load enrollment CSV masterlist for sector and municipality
        2. Match school IDs against enrollment CSV (authoritative source)
        3. Validate against node tables for coordinate availability

        Creates:
        - Enrollment match columns: sector_origin, sector_destination, municipality_origin, municipality_destination
        - Node validation flags: origin_in_node_tables, destination_in_node_tables
        - Combined validation: fully_valid

        Returns:
            DataFrame: Data with validation flags and school information
        """
        if self.gr7_data is None:
            raise ValueError("No Grade 7 data loaded. Call load_school_year() or load_all_years() first.")

        # Load enrollment masterlist if not already loaded
        if self.enrollment_data is None:
            self.load_enrollment_masterlist()

        # Load node tables if not already loaded
        if self.valid_public_ids is None or self.valid_private_ids is None:
            self.load_node_tables()

        logger.info("Matching school IDs against enrollment masterlist...")

        df = self.gr7_data.copy()

        # Match origin schools with enrollment data
        origin_lookup = self.enrollment_data.set_index('school_id')[['sector', 'municipality']]
        origin_lookup.columns = ['sector_origin', 'municipality_origin']

        df = df.merge(
            origin_lookup,
            left_on='school_id_origin',
            right_index=True,
            how='left'
        )

        # Match destination schools with enrollment data
        dest_lookup = self.enrollment_data.set_index('school_id')[['sector', 'municipality']]
        dest_lookup.columns = ['sector_destination', 'municipality_destination']

        df = df.merge(
            dest_lookup,
            left_on='school_id_destination',
            right_index=True,
            how='left'
        )

        # Count enrollment matches
        origin_in_enrollment = df['sector_origin'].notna().sum()
        dest_in_enrollment = df['sector_destination'].notna().sum()
        both_in_enrollment = (df['sector_origin'].notna() & df['sector_destination'].notna()).sum()

        total = len(df)
        logger.info(f"Enrollment masterlist matching:")
        logger.info(f"  Origin schools matched: {origin_in_enrollment:,} ({origin_in_enrollment/total*100:.1f}%)")
        logger.info(f"  Destination schools matched: {dest_in_enrollment:,} ({dest_in_enrollment/total*100:.1f}%)")
        logger.info(f"  Both matched: {both_in_enrollment:,} ({both_in_enrollment/total*100:.1f}%)")

        # Validate against node tables (for coordinate availability)
        logger.info("Validating coordinate availability in node tables...")

        df['origin_in_public_nodes'] = df['school_id_origin'].isin(self.valid_public_ids)
        df['origin_in_private_nodes'] = df['school_id_origin'].isin(self.valid_private_ids)
        df['origin_in_node_tables'] = df['origin_in_public_nodes'] | df['origin_in_private_nodes']

        df['destination_in_public_nodes'] = df['school_id_destination'].isin(self.valid_public_ids)
        df['destination_in_private_nodes'] = df['school_id_destination'].isin(self.valid_private_ids)
        df['destination_in_node_tables'] = df['destination_in_public_nodes'] | df['destination_in_private_nodes']

        df['both_in_node_tables'] = (
            df['origin_in_node_tables'] & df['destination_in_node_tables']
        )

        # Combined validation: valid IDs AND in enrollment masterlist AND in node tables
        df['both_in_enrollment'] = df['sector_origin'].notna() & df['sector_destination'].notna()
        df['fully_valid'] = (
            df['both_school_ids_valid'] &
            df['both_in_enrollment'] &
            df['both_in_node_tables']
        )

        # Log node table validation
        origin_coords = df['origin_in_node_tables'].sum()
        dest_coords = df['destination_in_node_tables'].sum()
        both_coords = df['both_in_node_tables'].sum()

        logger.info(f"Coordinate availability (node tables):")
        logger.info(f"  Origin schools: {origin_coords:,} ({origin_coords/total*100:.1f}%)")
        logger.info(f"  Destination schools: {dest_coords:,} ({dest_coords/total*100:.1f}%)")
        logger.info(f"  Both schools: {both_coords:,} ({both_coords/total*100:.1f}%)")

        # Log overall validation
        fully_valid = df['fully_valid'].sum()
        logger.info(f"  Fully valid records: {fully_valid:,} ({fully_valid/total*100:.1f}%)")

        # Log flow type distribution (by sector from enrollment)
        if 'sector_origin' in df.columns and 'sector_destination' in df.columns:
            logger.info(f"Flow type distribution (by sector from enrollment):")
            flow_types = df.groupby(['sector_origin', 'sector_destination'], dropna=False).size()
            for (origin, dest), count in flow_types.items():
                origin_str = str(origin) if pd.notna(origin) else 'unknown'
                dest_str = str(dest) if pd.notna(dest) else 'unknown'
                logger.info(f"    {origin_str} → {dest_str}: {count:,} ({count/total*100:.1f}%)")

        self.gr7_data = df
        return df

    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary statistics.

        Returns:
            dict: Summary statistics
        """
        if self.gr7_data is None:
            raise ValueError("No data loaded")

        df = self.gr7_data

        summary = {
            'total_records': len(df),
            'unique_students': df['lrn'].nunique(),
            'school_years': df['school_year'].unique().tolist(),
            'school_id_validation': {
                'origin_valid': df['school_id_origin_valid'].sum(),
                'destination_valid': df['school_id_destination_valid'].sum(),
                'both_valid': df['both_school_ids_valid'].sum(),
                'validation_rate': df['both_school_ids_valid'].sum() / len(df) * 100
            },
            'schools': {
                'unique_origins': df['school_id_origin'].nunique(),
                'unique_destinations': df['school_id_destination'].nunique(),
            },
            'geographic': {
                'regions': df['region'].nunique(),
                'divisions': df['division'].nunique()
            }
        }

        # Add node validation if available
        if 'fully_valid' in df.columns:
            summary['node_validation'] = {
                'origin_in_nodes': df['origin_in_node_tables'].sum(),
                'destination_in_nodes': df['destination_in_node_tables'].sum(),
                'both_in_nodes': df['both_in_node_tables'].sum(),
                'fully_valid': df['fully_valid'].sum(),
                'full_validation_rate': df['fully_valid'].sum() / len(df) * 100
            }

            # Add sector breakdown if available
            if 'sector_origin' in df.columns:
                summary['sector_breakdown'] = {
                    'origin_sectors': df['sector_origin'].value_counts().to_dict(),
                    'destination_sectors': df['sector_destination'].value_counts().to_dict()
                }

                # Add flow type breakdown
                if 'sector_destination' in df.columns:
                    flow_types = (
                        df['sector_origin'].astype(str) + ' → ' +
                        df['sector_destination'].astype(str)
                    ).value_counts().to_dict()
                    summary['flow_types'] = flow_types

        return summary

    def export(self, path: str, valid_only: bool = False):
        """
        Export processed Grade 7 enrollees data.

        Args:
            path (str): Output file path
            valid_only (bool): Export only fully valid records
        """
        if self.gr7_data is None:
            raise ValueError("No data to export")

        # Select data
        if valid_only and 'fully_valid' in self.gr7_data.columns:
            export_df = self.gr7_data[self.gr7_data['fully_valid']].copy()
            logger.info(f"Exporting {len(export_df):,} fully valid records")
        else:
            export_df = self.gr7_data.copy()
            logger.info(f"Exporting {len(export_df):,} total records")

        # Export
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_df.to_csv(output_path, index=False)

        logger.info(f"Exported to {output_path}")

    def process(
        self,
        school_year: str = 'SY 2023-2024',
        export_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Complete processing pipeline: load → validate → return.

        Args:
            school_year (str): School year to process
            export_path (str): Optional export path

        Returns:
            DataFrame: Processed and validated data
        """
        logger.info("="*60)
        logger.info(f"Processing Grade 7 enrollees for {school_year}")
        logger.info("="*60)

        # Load data
        self.load_school_year(school_year)

        # Validate against node tables
        self.validate_against_nodes()

        # Export if requested
        if export_path:
            self.export(export_path, valid_only=False)

        logger.info("="*60)
        logger.info("Processing complete")
        logger.info("="*60)

        return self.gr7_data


# Example usage
if __name__ == "__main__":
    processor = Gr7EnrolleesProcessor(
        public_nodes_path='output/public_nodes_valid.gpkg',
        verbose=True
    )

    # Process SY 2023-2024
    df = processor.process(
        school_year='SY 2023-2024',
        export_path='output/gr7_enrollees_sy2023_2024.csv'
    )

    # Get summary
    summary = processor.get_summary()
    print("\nSummary Statistics:")
    import json
    print(json.dumps(summary, indent=2))
