"""
Node Table Builder Module

Purpose:
    Consolidate and validate school-level data from multiple sources into comprehensive
    "node tables" ready for graph network analysis. Creates GeoDataFrames with spatial
    attributes, administrative boundary assignments, and multi-tier validation.

Features:
    - Builds separate node tables for public and private schools
    - Creates combined public+private node table for complete network
    - Spatial integration: GeoDataFrame output with Point geometries (EPSG:4326)
    - Administrative boundary assignment via spatial join with PSGC data
    - Provincial road network matching (adm2_pcode column)
    - Tiered validation (required/core/complete levels)
    - Enhanced spatial validation (boundary containment, admin assignment)
    - Computed totals for graph node weights (total_enrollment, total_seats)
    - Multiple export formats (CSV, Parquet, GeoPackage)
    - Comprehensive quality reporting

Usage:
    from modules.node_table_builder import NodeTableBuilder

    # Initialize builder
    builder = NodeTableBuilder(
        verbose=True,
        psgc_geodata_path='output/consolidated_geodata_matched.gpkg',
        validation_level='complete'
    )

    # Build node tables (returns GeoDataFrame)
    public_gdf = builder.build_public_node_table()
    private_gdf = builder.build_private_node_table()
    all_schools_gdf = builder.build_combined_node_table()

    # Get summaries
    summary = builder.get_summary()
    public_summary = builder.get_public_summary()

    # Export for graph generation
    builder.export_geopackage('output/all_nodes.gpkg', sector='both')
    builder.export_quality_report('output/data_quality_report.csv')

Author: Claude Code
Created: 2025-11-12
Updated: 2025-11-12
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import logging
import json

# Import existing processors
from modules.enrollment_processor import EnrollmentDataProcessor
from modules.public_coordinates_processor import PublicSchoolsProcessor
from modules.facilities_processor import FacilitiesProcessor
from modules.public_furnitures_processor import PublicFurnituresProcessor
from modules.private_coordinates_processor import PrivateSchoolsProcessor
from modules.private_furniture_processor import PrivateFurnitureProcessor
from modules.subsidy_tuition_processor import SubsidyTuitionProcessor

# Configure logging
logger = logging.getLogger(__name__)


class NodeTableBuilder:
    """
    Builds comprehensive node tables for school network analysis.

    Consolidates data from multiple sources (coordinates, enrollment, facilities,
    seats, GASTPE programs) into validated GeoDataFrames with spatial attributes
    and administrative boundary assignments.

    Attributes:
        verbose (bool): Enable INFO level logging
        validation_level (str): 'required', 'core', or 'complete'
        psgc_geodata_path (str): Path to PSGC consolidated geodata
        public_node_table (GeoDataFrame): Public school nodes
        private_node_table (GeoDataFrame): Private school nodes
        combined_node_table (GeoDataFrame): All schools combined
    """

    def __init__(self, verbose=True, config_path=None, validation_level='complete',
                 psgc_geodata_path=None):
        """
        Initialize NodeTableBuilder.

        Args:
            verbose (bool): Enable verbose logging (default: True)
            config_path (str): Path to config file (optional, uses default if None)
            validation_level (str): Validation tier - 'required', 'core', or 'complete'
            psgc_geodata_path (str): Path to PSGC geodata GeoPackage
        """
        # Configure logging
        self.verbose = verbose
        log_level = logging.INFO if verbose else logging.WARNING
        logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')

        # Validation level
        valid_levels = ['required', 'core', 'complete']
        if validation_level not in valid_levels:
            raise ValueError(f"validation_level must be one of {valid_levels}")
        self.validation_level = validation_level

        # Detect project root
        self.project_root = self._detect_project_root()

        # Paths
        self.config_path = config_path
        self.psgc_geodata_path = psgc_geodata_path or 'output/consolidated_geodata_matched.gpkg'
        # Make psgc_geodata_path absolute relative to project root
        if not Path(self.psgc_geodata_path).is_absolute():
            self.psgc_geodata_path = str(self.project_root / self.psgc_geodata_path)

        # Data storage
        self.psgc_geodata = None
        self.psgc_provinces = None  # Province-level dissolved boundaries
        self.public_node_table = None
        self.private_node_table = None
        self.combined_node_table = None

        # Intermediate data cache
        self._public_coordinates = None
        self._public_enrollment = None
        self._public_facilities = None
        self._public_seats = None
        self._private_coordinates = None
        self._private_furniture = None
        self._gastpe_data = None
        self._private_enrollment = None

        logger.info(f"NodeTableBuilder initialized (validation_level={validation_level})")

    def _detect_project_root(self):
        """
        Detect the project root directory by looking for the 'modules' folder.

        Returns:
            Path: Absolute path to project root
        """
        # Start from this file's location
        current_path = Path(__file__).resolve().parent

        # Go up until we find the directory containing 'modules'
        while current_path.parent != current_path:  # Not at filesystem root
            if (current_path / 'modules').is_dir():
                return current_path
            current_path = current_path.parent

        # If not found, use current working directory
        logger.warning("Could not detect project root, using current working directory")
        return Path.cwd()

    def _resolve_path(self, path):
        """
        Resolve a path relative to project root if it's not absolute.

        Args:
            path (str or Path): Path to resolve

        Returns:
            Path: Absolute path
        """
        path = Path(path)
        if path.is_absolute():
            return path
        else:
            return self.project_root / path

    # ==================== PUBLIC WORKFLOW ====================

    def build_public_node_table(self, include_geometry=True, assign_boundaries=True,
                                 compute_totals=True):
        """
        Build comprehensive public school node table.

        Workflow:
            1. Load public coordinates as base
            2. Load and merge enrollment data
            3. Load and merge facilities data
            4. Load and merge seats data
            5. Create geometry column (if requested)
            6. Assign administrative boundaries (if requested)
            7. Compute total metrics (if requested)
            8. Validate with tiered validation

        Args:
            include_geometry (bool): Convert to GeoDataFrame with Point geometry
            assign_boundaries (bool): Spatial join to assign province/municipality
            compute_totals (bool): Compute total_enrollment, total_seats, etc.

        Returns:
            GeoDataFrame: Public school node table with validation columns
        """
        logger.info("=" * 60)
        logger.info("Building PUBLIC node table")
        logger.info("=" * 60)

        # 1. Load base coordinates
        logger.info("Step 1/7: Loading public coordinates...")
        coords_df = self._load_public_coordinates()
        base_df = coords_df.copy()
        logger.info(f"  Loaded {len(base_df):,} public schools")

        # 2. Load and merge enrollment
        logger.info("Step 2/7: Loading and merging enrollment data...")
        enrollment_df = self._load_public_enrollment()
        base_df = self._merge_with_validation(
            base_df, enrollment_df,
            on='school_id',
            how='left',
            data_source_name='enrollment'
        )

        # 3. Load and merge facilities
        logger.info("Step 3/7: Loading and merging facilities data...")
        facilities_df = self._load_public_facilities()
        base_df = self._merge_with_validation(
            base_df, facilities_df,
            on='school_id',
            how='left',
            data_source_name='facilities'
        )

        # 4. Load and merge seats
        logger.info("Step 4/7: Loading and merging seats data...")
        seats_df = self._load_public_seats()
        base_df = self._merge_with_validation(
            base_df, seats_df,
            on='school_id',
            how='left',
            data_source_name='seats'
        )

        # 4.1. Clean seats data based on offers_* flags
        logger.info("  Cleaning seats data based on education levels offered...")
        base_df = self._clean_seats_by_offers(base_df)

        # 5. Create geometry column
        if include_geometry:
            logger.info("Step 5/7: Creating geometry column...")
            base_df = self._create_geometry_column(base_df)
        else:
            logger.info("Step 5/7: Skipping geometry creation (include_geometry=False)")

        # 6. Assign administrative boundaries
        if assign_boundaries and include_geometry:
            logger.info("Step 6/7: Assigning administrative boundaries...")
            base_df = self._assign_admin_boundaries(base_df)
        else:
            if not include_geometry:
                logger.info("Step 6/7: Skipping boundary assignment (requires geometry)")
            else:
                logger.info("Step 6/7: Skipping boundary assignment (assign_boundaries=False)")

        # 7. Compute totals
        if compute_totals:
            logger.info("Step 7/7: Computing total metrics...")
            base_df = self._compute_totals(base_df, sector='public')
        else:
            logger.info("Step 7/7: Skipping total computation (compute_totals=False)")

        # 8. Validate
        logger.info("Validating public node table...")
        base_df = self._validate_node_table(base_df, sector='public')

        # Store result
        self.public_node_table = base_df

        # Summary
        valid_count = base_df['all_valid'].sum() if 'all_valid' in base_df.columns else 0
        logger.info(f"PUBLIC node table complete: {len(base_df):,} schools, {valid_count:,} valid")
        logger.info("=" * 60)

        return base_df

    def _load_public_coordinates(self):
        """Load public school coordinates using SchoolCoordinatesProcessor."""
        if self._public_coordinates is not None:
            logger.debug("  Using cached public coordinates")
            return self._public_coordinates

        processor = PublicSchoolsProcessor(verbose=self.verbose)
        coords_df = processor.process()

        # Keep only necessary columns
        keep_cols = ['school_id_processed', 'latitude', 'longitude', 'coord_valid']
        coords_df = (
            coords_df[keep_cols]
            .rename(
                columns={
                    'school_id_processed':'school_id',
                    'coord_valid':'coordinates_valid'
                }
            )
        )

        # Ensure school_id is string type for consistent merging
        coords_df['school_id'] = coords_df['school_id'].astype(str)

        self._public_coordinates = coords_df
        return coords_df

    def _load_public_enrollment(self):
        """Load and pivot public enrollment data by education level."""
        if self._public_enrollment is not None:
            logger.debug("  Using cached public enrollment")
            return self._public_enrollment

        processor = EnrollmentDataProcessor(verbose=self.verbose)
        enrollment_long = processor.process()

        # Filter to public schools only
        enrollment_long = enrollment_long[enrollment_long['sector'] == 'Public'].copy()

        # Convert school_id to string for consistent merging
        enrollment_long['school_id'] = enrollment_long['school_id'].astype(str)

        # Pivot by education level
        enrollment_pivot = self._pivot_by_education_level(
            enrollment_long,
            index_col='school_id',
            columns_col='grade_level',
            value_col='enrollment_count',
            prefix='enrollment'
        )

        # Ensure school_id is string type after pivot
        enrollment_pivot['school_id'] = enrollment_pivot['school_id'].astype(str)

        self._public_enrollment = enrollment_pivot
        return enrollment_pivot

    def _load_public_facilities(self):
        """Load public facilities data."""
        if self._public_facilities is not None:
            logger.debug("  Using cached public facilities")
            return self._public_facilities

        processor = FacilitiesProcessor(verbose=self.verbose)
        facilities_df = processor.process()

        # Filter to public schools
        facilities_df = facilities_df[facilities_df['sector'] == 'Public'].copy()

        # Convert school_id to string for consistent merging
        facilities_df['school_id'] = facilities_df['school_id'].astype(str)

        # Keep relevant columns
        keep_cols = ['school_id', 'offers_es', 'offers_jhs', 'offers_shs',
                     'es_classrooms_instructional', 'es_classrooms_non_instructional',
                     'jhs_classrooms_instructional', 'jhs_classrooms_non_instructional',
                     'shs_classrooms_instructional', 'shs_classrooms_non_instructional']
        existing_cols = [col for col in keep_cols if col in facilities_df.columns]
        facilities_df = facilities_df[existing_cols]

        self._public_facilities = facilities_df
        return facilities_df

    def _load_public_seats(self):
        """Load and pivot public seats data by education level."""
        if self._public_seats is not None:
            logger.debug("  Using cached public seats")
            return self._public_seats

        processor = PublicFurnituresProcessor(verbose=self.verbose)
        seats_long = processor.process()

        # Convert school_id to string for consistent merging
        seats_long['school_id'] = seats_long['school_id'].astype(str)

        # Pivot by education level
        seats_pivot = self._pivot_by_education_level(
            seats_long,
            index_col='school_id',
            columns_col='education_level',
            value_col='seat_count',
            prefix='seats'
        )

        # Ensure school_id is string type after pivot
        seats_pivot['school_id'] = seats_pivot['school_id'].astype(str)

        self._public_seats = seats_pivot
        return seats_pivot

    def _clean_seats_by_offers(self, df):
        """
        Clean seats data based on education levels offered.

        Sets seats_es/jhs/shs to NaN if the school doesn't offer that level
        (based on offers_es/jhs/shs flags from facilities data).

        Args:
            df (DataFrame): Node table with offers_* and seats_* columns

        Returns:
            DataFrame: Node table with cleaned seats data
        """
        df = df.copy()

        # Track cleaning statistics
        cleaned_counts = {'es': 0, 'jhs': 0, 'shs': 0}

        # Clean ES seats
        if 'offers_es' in df.columns and 'seats_es' in df.columns:
            # Set seats_es to NaN where offers_es is False
            mask = (df['offers_es'] == False) & df['seats_es'].notna()
            cleaned_counts['es'] = mask.sum()
            df.loc[mask, 'seats_es'] = np.nan

        # Clean JHS seats
        if 'offers_jhs' in df.columns and 'seats_jhs' in df.columns:
            # Set seats_jhs to NaN where offers_jhs is False
            mask = (df['offers_jhs'] == False) & df['seats_jhs'].notna()
            cleaned_counts['jhs'] = mask.sum()
            df.loc[mask, 'seats_jhs'] = np.nan

        # Clean SHS seats
        if 'offers_shs' in df.columns and 'seats_shs' in df.columns:
            # Set seats_shs to NaN where offers_shs is False
            mask = (df['offers_shs'] == False) & df['seats_shs'].notna()
            cleaned_counts['shs'] = mask.sum()
            df.loc[mask, 'seats_shs'] = np.nan

        # Log cleaning results
        total_cleaned = sum(cleaned_counts.values())
        if total_cleaned > 0:
            logger.info(f"    Set {cleaned_counts['es']:,} ES seats to NaN (school doesn't offer ES)")
            logger.info(f"    Set {cleaned_counts['jhs']:,} JHS seats to NaN (school doesn't offer JHS)")
            logger.info(f"    Set {cleaned_counts['shs']:,} SHS seats to NaN (school doesn't offer SHS)")
        else:
            logger.info(f"    No seats data to clean (all consistent with offers_* flags)")

        return df

    # ==================== PRIVATE WORKFLOW ====================

    def build_private_node_table(self, include_geometry=True, assign_boundaries=True,
                                  compute_totals=True):
        """
        Build comprehensive private school node table.

        Workflow:
            1. Load private coordinates as base
            2. Load and merge GASTPE data (ESC + SHSVP)
            3. Load and merge furniture/seats data
            4. Load and merge enrollment data
            5. Create geometry column (if requested)
            6. Assign administrative boundaries (if requested)
            7. Compute total metrics (if requested)
            8. Validate with tiered validation

        Args:
            include_geometry (bool): Convert to GeoDataFrame with Point geometry
            assign_boundaries (bool): Spatial join to assign province/municipality
            compute_totals (bool): Compute total_enrollment, total_seats, etc.

        Returns:
            GeoDataFrame: Private school node table with validation columns
        """
        logger.info("=" * 60)
        logger.info("Building PRIVATE node table")
        logger.info("=" * 60)

        # 1. Load base coordinates
        logger.info("Step 1/7: Loading private coordinates...")
        coords_df = self._load_private_coordinates()
        base_df = coords_df.copy()
        logger.info(f"  Loaded {len(base_df):,} private schools")

        # 2. Load and merge GASTPE data
        logger.info("Step 2/7: Loading and merging GASTPE data...")
        gastpe_df = self._load_gastpe_data()
        base_df = self._merge_with_validation(
            base_df, gastpe_df,
            on='school_id',
            how='left',
            data_source_name='gastpe'
        )

        # 3. Load and merge furniture/seats
        logger.info("Step 3/7: Loading and merging furniture/seats data...")
        furniture_df = self._load_private_furniture()
        base_df = self._merge_with_validation(
            base_df, furniture_df,
            on='school_id',
            how='left',
            data_source_name='furniture'
        )

        # 4. Load and merge enrollment
        logger.info("Step 4/7: Loading and merging enrollment data...")
        enrollment_df = self._load_private_enrollment()
        base_df = self._merge_with_validation(
            base_df, enrollment_df,
            on='school_id',
            how='left',
            data_source_name='enrollment'
        )

        # 5. Create geometry column
        if include_geometry:
            logger.info("Step 5/7: Creating geometry column...")
            base_df = self._create_geometry_column(base_df)
        else:
            logger.info("Step 5/7: Skipping geometry creation (include_geometry=False)")

        # 6. Assign administrative boundaries
        if assign_boundaries and include_geometry:
            logger.info("Step 6/7: Assigning administrative boundaries...")
            base_df = self._assign_admin_boundaries(base_df)
        else:
            if not include_geometry:
                logger.info("Step 6/7: Skipping boundary assignment (requires geometry)")
            else:
                logger.info("Step 6/7: Skipping boundary assignment (assign_boundaries=False)")

        # 7. Compute totals
        if compute_totals:
            logger.info("Step 7/7: Computing total metrics...")
            base_df = self._compute_totals(base_df, sector='private')
        else:
            logger.info("Step 7/7: Skipping total computation (compute_totals=False)")

        # 8. Validate
        logger.info("Validating private node table...")
        base_df = self._validate_node_table(base_df, sector='private')

        # Store result
        self.private_node_table = base_df

        # Summary
        valid_count = base_df['all_valid'].sum() if 'all_valid' in base_df.columns else 0
        logger.info(f"PRIVATE node table complete: {len(base_df):,} schools, {valid_count:,} valid")
        logger.info("=" * 60)

        return base_df

    def _load_private_coordinates(self):
        """Load private school coordinates using PrivateSchoolsProcessor."""
        if self._private_coordinates is not None:
            logger.debug("  Using cached private coordinates")
            return self._private_coordinates

        processor = PrivateSchoolsProcessor(
            directory_path='data/private/raw_validation_sheets',
            verbose=self.verbose
        )
        coords_df = processor.process()

        # Clean and validate coordinates
        coords_df = processor.validate_coordinates_with_reasons(clean_first=True)

        # Standardize regions
        processor.replace_unclean_region_values()

        # Map curricular offerings
        processor.map_curricular_offerings()

        # Get processed data
        coords_df = processor.processed_data.copy()

        # Convert school_id to string for consistent merging
        coords_df = coords_df.rename(
            columns={
                'beis_school_id':'school_id'
            }
        )
        # Handle numeric IDs properly (324731.0 → '324731')
        coords_df['school_id'] = pd.to_numeric(coords_df['school_id'], errors='coerce')
        coords_df['school_id'] = coords_df['school_id'].astype('Int64').astype(str).str.strip()

        # Keep only necessary columns
        keep_cols = ['school_id', 'school_name', 'latitude', 'longitude',
                     'coordinates_valid', 'region', 'modified_coc']
        existing_cols = [col for col in keep_cols if col in coords_df.columns]
        coords_df = coords_df[existing_cols]

        self._private_coordinates = coords_df
        return coords_df

    def _load_private_furniture(self):
        """Load and pivot private furniture data by education level."""
        if self._private_furniture is not None:
            logger.debug("  Using cached private furniture")
            return self._private_furniture

        processor = PrivateFurnitureProcessor(verbose=self.verbose)
        furniture_long = processor.process()

        # Convert school_id to string, handling floats properly (324731.0 → '324731')
        # First convert to numeric to handle any string representations of numbers
        furniture_long['school_id'] = pd.to_numeric(furniture_long['school_id'], errors='coerce')
        # Convert to int (removes decimals), then to string
        furniture_long['school_id'] = furniture_long['school_id'].astype('Int64').astype(str)
        # Strip whitespace
        furniture_long['school_id'] = furniture_long['school_id'].str.strip()

        # Pivot by education level
        furniture_pivot = self._pivot_by_education_level(
            furniture_long,
            index_col='school_id',
            columns_col='grade_level',
            value_col='furniture_count',
            prefix='seats'  # Use 'seats' prefix (furniture counts approximate seats)
        )

        # Ensure school_id is string type after pivot
        furniture_pivot['school_id'] = furniture_pivot['school_id'].astype(str).str.strip()

        logger.info(f"  Furniture data: {len(furniture_pivot):,} schools with furniture counts")

        self._private_furniture = furniture_pivot
        return furniture_pivot

    def _load_gastpe_data(self):
        """
        Load GASTPE subsidy program data (ESC + SHSVP).

        Aggregates to school level with average fees across grades/tracks.
        One row per school.
        """
        if self._gastpe_data is not None:
            logger.debug("  Using cached GASTPE data")
            return self._gastpe_data

        processor = SubsidyTuitionProcessor(verbose=self.verbose)
        esc_df, shsvp_df = processor.process()

        # Convert school_id to string for consistent merging
        esc_df['school_id'] = esc_df['school_id'].astype(str)
        shsvp_df['school_id'] = shsvp_df['school_id'].astype(str)

        # Aggregate ESC to school level (average fees across grades)
        # ESC has columns: school_id, grade_level, fee_type, amount
        # Pivot to get separate columns for each fee_type, then average
        esc_pivot = esc_df.pivot_table(
            index='school_id',
            columns='fee_type',
            values='amount',
            aggfunc='mean'
        ).reset_index()

        # Rename columns to standardized format
        column_mapping = {}
        for col in esc_pivot.columns:
            if col == 'school_id':
                continue
            elif 'tuition' in col.lower():
                column_mapping[col] = 'esc_average_tuition_fees'
            elif 'miscellaneous' in col.lower() or 'misc' in col.lower():
                column_mapping[col] = 'esc_average_misc_fees'
            elif 'other' in col.lower():
                column_mapping[col] = 'esc_average_other_fees'

        esc_agg = esc_pivot.rename(columns=column_mapping)
        esc_agg['esc_delivering'] = True

        # Aggregate SHSVP to school level (average fees across tracks/strands)
        # SHSVP has columns: school_id, Track, Strand, Tuition, Other, Miscellaneous
        shsvp_agg = shsvp_df.groupby('school_id', as_index=False).agg({
            'Tuition': 'mean',
            'Other': 'mean',
            'Miscellaneous': 'mean'
        }).rename(columns={
            'Tuition': 'shsvp_average_tuition_fees',
            'Other': 'shsvp_average_other_fees',
            'Miscellaneous': 'shsvp_average_misc_fees'
        })
        shsvp_agg['shsvp_delivering'] = True

        # Merge ESC and SHSVP (outer join to capture all schools)
        gastpe_df = pd.merge(
            esc_agg,
            shsvp_agg,
            on='school_id',
            how='outer'
        )

        # Fill NaN flags with False
        gastpe_df['esc_delivering'] = gastpe_df['esc_delivering'].fillna(False)
        gastpe_df['shsvp_delivering'] = gastpe_df['shsvp_delivering'].fillna(False)

        logger.info(f"  Aggregated GASTPE data: {len(gastpe_df):,} schools")
        logger.info(f"    ESC delivering: {gastpe_df['esc_delivering'].sum():,}")
        logger.info(f"    SHSVP delivering: {gastpe_df['shsvp_delivering'].sum():,}")

        self._gastpe_data = gastpe_df
        return gastpe_df

    def _load_private_enrollment(self):
        """Load private school enrollment data (filtered from public enrollment file)."""
        if self._private_enrollment is not None:
            logger.debug("  Using cached private enrollment")
            return self._private_enrollment

        processor = EnrollmentDataProcessor(verbose=self.verbose)
        enrollment_long = processor.process()

        # Filter to private schools only
        enrollment_long = enrollment_long[enrollment_long['sector'] == 'Private'].copy()

        # Convert school_id to string for consistent merging
        enrollment_long['school_id'] = enrollment_long['school_id'].astype(str)

        # Pivot by education level
        enrollment_pivot = self._pivot_by_education_level(
            enrollment_long,
            index_col='school_id',
            columns_col='grade_level',
            value_col='enrollment_count',
            prefix='enrollment'
        )

        # Ensure school_id is string type after pivot
        enrollment_pivot['school_id'] = enrollment_pivot['school_id'].astype(str)

        self._private_enrollment = enrollment_pivot
        return enrollment_pivot

    # ==================== COMBINED WORKFLOW ====================

    def build_combined_node_table(self, include_geometry=True, assign_boundaries=True,
                                   compute_totals=True):
        """
        Build combined public + private node table.

        Creates a unified GeoDataFrame with all schools, adding a 'sector' column
        to distinguish public from private schools.

        Args:
            include_geometry (bool): Include Point geometry
            assign_boundaries (bool): Assign administrative boundaries
            compute_totals (bool): Compute total metrics

        Returns:
            GeoDataFrame: Combined node table with 'sector' column
        """
        logger.info("=" * 60)
        logger.info("Building COMBINED node table")
        logger.info("=" * 60)

        # Build public and private tables if not already built
        if self.public_node_table is None:
            logger.info("Building public table first...")
            self.build_public_node_table(include_geometry, assign_boundaries, compute_totals)

        if self.private_node_table is None:
            logger.info("Building private table first...")
            self.build_private_node_table(include_geometry, assign_boundaries, compute_totals)

        # Add sector column
        public_df = self.public_node_table.copy()
        private_df = self.private_node_table.copy()

        public_df['sector'] = 'public'
        private_df['sector'] = 'private'

        # Align columns (use intersection of columns)
        common_cols = list(set(public_df.columns) & set(private_df.columns))

        # Ensure 'geometry' is included if present
        if 'geometry' in public_df.columns or 'geometry' in private_df.columns:
            if 'geometry' not in common_cols:
                common_cols.append('geometry')

        logger.info(f"Combining tables with {len(common_cols)} common columns...")

        # Select common columns
        public_subset = public_df[common_cols]
        private_subset = private_df[common_cols]

        # Concatenate
        if isinstance(public_subset, gpd.GeoDataFrame) or isinstance(private_subset, gpd.GeoDataFrame):
            combined_df = gpd.GeoDataFrame(
                pd.concat([public_subset, private_subset], ignore_index=True),
                crs='EPSG:4326' if include_geometry else None
            )
        else:
            combined_df = pd.concat([public_subset, private_subset], ignore_index=True)

        # Store result
        self.combined_node_table = combined_df

        logger.info(f"COMBINED node table complete: {len(combined_df):,} schools")
        logger.info(f"  Public: {len(public_df):,}, Private: {len(private_df):,}")
        logger.info("=" * 60)

        return combined_df

    # ==================== SPATIAL UTILITIES ====================

    def _load_psgc_geodata(self):
        """Load PSGC consolidated geodata for boundary assignment."""
        if self.psgc_geodata is not None:
            logger.debug("  Using cached PSGC geodata")
            return self.psgc_geodata

        psgc_path = Path(self.psgc_geodata_path)
        if not psgc_path.exists():
            logger.warning(f"  PSGC geodata not found at {psgc_path}")
            logger.warning("  Skipping boundary assignment")
            return None

        logger.info(f"  Loading PSGC geodata from {psgc_path}...")
        psgc_gdf = gpd.read_file(psgc_path)

        # Dissolve to municipality (adm3) level for faster spatial joins
        logger.info("  Dissolving to municipality level...")
        psgc_municipalities = psgc_gdf.dissolve(
            by=['adm1_psgc', 'adm2_psgc', 'adm3_psgc', 'adm2_pcode'],
            as_index=False
        )

        # Keep only necessary columns
        keep_cols = ['adm1_psgc', 'adm2_psgc', 'adm3_psgc', 'adm2_pcode',
                     'adm1_en', 'adm2_en', 'adm3_en', 'geometry']
        existing_cols = [col for col in keep_cols if col in psgc_municipalities.columns]
        psgc_municipalities = psgc_municipalities[existing_cols]

        # Rename columns for clarity
        rename_map = {
            'adm1_en': 'region',
            'adm2_en': 'province',
            'adm3_en': 'municipality'
        }
        psgc_municipalities.rename(columns=rename_map, inplace=True)

        self.psgc_geodata = psgc_municipalities
        logger.info(f"  Loaded {len(psgc_municipalities):,} municipalities")

        return psgc_municipalities

    def _load_psgc_provinces(self):
        """
        Load PSGC geodata dissolved to province level for robust spatial matching.

        Province-level boundaries are more reliable for matching than municipality-level
        because they have simpler geometries and fewer boundary edge cases.

        Returns:
            GeoDataFrame: Provinces with adm2_pcode, province names, and geometries
        """
        if self.psgc_provinces is not None:
            logger.debug("  Using cached PSGC provinces")
            return self.psgc_provinces

        psgc_path = Path(self.psgc_geodata_path)
        if not psgc_path.exists():
            logger.warning(f"  PSGC geodata not found at {psgc_path}")
            return None

        logger.info("  Loading PSGC geodata for province boundaries...")
        psgc_gdf = gpd.read_file(psgc_path)

        # Dissolve to province (adm2) level
        logger.info("  Dissolving to province level...")
        provinces = psgc_gdf.dissolve(by='adm2_pcode').reset_index()

        # Keep only necessary columns
        keep_cols = ['adm2_pcode', 'adm2_en', 'adm1_pcode', 'adm1_en', 'geometry']
        existing_cols = [col for col in keep_cols if col in provinces.columns]
        provinces = provinces[existing_cols]

        # Rename for clarity
        rename_map = {
            'adm1_en': 'region',
            'adm2_en': 'province'
        }
        provinces.rename(columns=rename_map, inplace=True)

        self.psgc_provinces = provinces
        logger.info(f"  Loaded {len(provinces):,} provinces")

        return provinces

    def _create_geometry_column(self, df):
        """
        Convert latitude/longitude to Point geometry, return GeoDataFrame.

        Only creates geometries for rows with coordinates_valid=True.
        Invalid coordinates are set to None geometry.

        Args:
            df (DataFrame): DataFrame with 'latitude' and 'longitude' columns

        Returns:
            GeoDataFrame: GeoDataFrame with 'geometry' column (EPSG:4326)
        """
        if isinstance(df, gpd.GeoDataFrame):
            logger.debug("  Already a GeoDataFrame")
            return df

        # Check if coordinates_valid column exists
        has_valid_column = 'coordinates_valid' in df.columns

        # Create Point geometries (only for valid coordinates)
        geometry = []
        for idx, row in df.iterrows():
            lat = row.get('latitude')
            lon = row.get('longitude')
            is_valid = row.get('coordinates_valid', True) if has_valid_column else True

            # Only create geometry if coordinates are valid
            if is_valid and pd.notna(lat) and pd.notna(lon):
                try:
                    # Ensure coordinates are numeric
                    lat_float = float(lat)
                    lon_float = float(lon)
                    geometry.append(Point(lon_float, lat_float))
                except (ValueError, TypeError):
                    # If conversion fails, set to None
                    geometry.append(None)
            else:
                geometry.append(None)

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

        valid_geom_count = gdf['geometry'].notna().sum()
        logger.info(f"  Created geometry for {valid_geom_count:,}/{len(gdf):,} schools")

        if has_valid_column:
            invalid_coords = (~df['coordinates_valid']).sum()
            logger.info(f"  Skipped {invalid_coords:,} schools with invalid coordinates")

        return gdf

    def _assign_admin_boundaries(self, gdf):
        """
        Spatial join to assign province/municipality to each school using two-stage matching.

        Strategy:
            1. Province-level matching (robust, broad boundaries)
            2. Municipality-level matching for finer geographic detail
            3. Track and log unmatched schools with reasons

        Args:
            gdf (GeoDataFrame): Schools with Point geometry

        Returns:
            GeoDataFrame: Schools with admin boundary columns + admin_assignment_valid flag
        """
        if not isinstance(gdf, gpd.GeoDataFrame):
            logger.warning("  Input is not a GeoDataFrame, skipping boundary assignment")
            # Add admin_assignment_valid = False for all
            gdf['admin_assignment_valid'] = False
            return gdf

        # Separate schools with/without valid geometry
        valid_geom_mask = gdf['geometry'].notna()
        schools_with_geom = gdf[valid_geom_mask].copy()
        schools_without_geom = gdf[~valid_geom_mask].copy()

        logger.info(f"  Schools with geometry: {len(schools_with_geom):,}")
        logger.info(f"  Schools without geometry: {len(schools_without_geom):,}")

        # === STAGE 1: Province-level matching (robust) ===
        logger.info("  Stage 1: Province-level spatial join...")
        provinces = self._load_psgc_provinces()
        if provinces is None:
            gdf['admin_assignment_valid'] = False
            return gdf

        # Ensure CRS match
        if schools_with_geom.crs != provinces.crs:
            logger.info(f"  Reprojecting provinces from {provinces.crs} to {schools_with_geom.crs}...")
            provinces = provinces.to_crs(schools_with_geom.crs)

        # Province-level spatial join
        province_join = gpd.sjoin(
            schools_with_geom,
            provinces[['adm2_pcode', 'adm1_pcode', 'region', 'province', 'geometry']],
            how='left',
            predicate='within'
        )

        # Remove index_right column
        if 'index_right' in province_join.columns:
            province_join = province_join.drop(columns=['index_right'])

        province_matched = province_join['adm2_pcode'].notna().sum()
        logger.info(f"  Province matches: {province_matched:,} / {len(schools_with_geom):,} "
                   f"({province_matched/len(schools_with_geom)*100:.1f}%)")

        # === STAGE 2: Municipality-level matching for finer detail ===
        logger.info("  Stage 2: Municipality-level spatial join...")
        municipalities = self._load_psgc_geodata()
        if municipalities is not None:
            # Ensure CRS match
            if province_join.crs != municipalities.crs:
                municipalities = municipalities.to_crs(province_join.crs)

            # Join to get municipality names
            munic_join = gpd.sjoin(
                province_join,
                municipalities[['adm3_psgc', 'municipality', 'geometry']],
                how='left',
                predicate='within'
            )

            # Remove index_right column
            if 'index_right' in munic_join.columns:
                munic_join = munic_join.drop(columns=['index_right'])

            result = munic_join
        else:
            result = province_join
            result['municipality'] = None

        # === STAGE 3: Handle unmatched schools ===
        # Mark schools with successful admin assignment
        result['admin_assignment_valid'] = result['adm2_pcode'].notna()

        # Combine with schools without geometry
        if len(schools_without_geom) > 0:
            # Add missing columns with NaN
            for col in ['adm1_pcode', 'adm2_pcode', 'adm3_psgc', 'region', 'province', 'municipality']:
                if col not in schools_without_geom.columns:
                    schools_without_geom[col] = None

            schools_without_geom['admin_assignment_valid'] = False

            result = pd.concat([result, schools_without_geom], ignore_index=True)
            result = gpd.GeoDataFrame(result, geometry='geometry', crs=gdf.crs)

        # === STAGE 4: Report statistics ===
        total_schools = len(result)
        matched_province = result['adm2_pcode'].notna().sum()
        matched_municipality = result['municipality'].notna().sum()
        valid_admin = result['admin_assignment_valid'].sum()

        logger.info(f"  Final results:")
        logger.info(f"    Province assigned: {matched_province:,} / {total_schools:,} "
                   f"({matched_province/total_schools*100:.1f}%)")
        logger.info(f"    Municipality assigned: {matched_municipality:,} / {total_schools:,} "
                   f"({matched_municipality/total_schools*100:.1f}%)")
        logger.info(f"    Admin assignment valid: {valid_admin:,} / {total_schools:,} "
                   f"({valid_admin/total_schools*100:.1f}%)")

        # Log unmatched schools
        unmatched_count = total_schools - valid_admin
        if unmatched_count > 0:
            logger.warning(f"  ⚠ {unmatched_count:,} schools could not be assigned to provinces")
            logger.warning(f"    These schools may have:")
            logger.warning(f"      - Invalid coordinates (outside Philippines)")
            logger.warning(f"      - Coordinates in unmatched areas")
            logger.warning(f"      - Missing geometry")

            # Store unmatched school IDs for validation report
            if not hasattr(self, 'unmatched_schools'):
                self.unmatched_schools = {}

            unmatched_ids = result[~result['admin_assignment_valid']]['school_id'].tolist()
            self.unmatched_schools['admin_boundary'] = unmatched_ids

        return result

    def _compute_totals(self, df, sector='public'):
        """
        Compute total enrollment, seats, and capacity utilization.

        Args:
            df (DataFrame): Node table
            sector (str): 'public' or 'private'

        Returns:
            DataFrame: Node table with computed columns
        """
        # Total enrollment
        enrollment_cols = ['enrollment_es', 'enrollment_jhs', 'enrollment_shs']
        existing_enrollment_cols = [col for col in enrollment_cols if col in df.columns]

        if existing_enrollment_cols:
            df['total_enrollment'] = df[existing_enrollment_cols].sum(axis=1)
        else:
            df['total_enrollment'] = np.nan

        # Total seats
        seats_cols = ['seats_es', 'seats_jhs', 'seats_shs']
        existing_seats_cols = [col for col in seats_cols if col in df.columns]

        if existing_seats_cols:
            df['total_seats'] = df[existing_seats_cols].sum(axis=1)
        else:
            df['total_seats'] = np.nan

        # Capacity utilization
        if 'total_enrollment' in df.columns and 'total_seats' in df.columns:
            df['capacity_utilization'] = df['total_enrollment'] / df['total_seats']
            # Replace inf with NaN
            df['capacity_utilization'] = df['capacity_utilization'].replace([np.inf, -np.inf], np.nan)

        # Count computed
        has_enrollment = df['total_enrollment'].notna().sum()
        has_seats = df['total_seats'].notna().sum()
        has_utilization = df['capacity_utilization'].notna().sum() if 'capacity_utilization' in df.columns else 0

        logger.info(f"  Computed totals: {has_enrollment:,} enrollment, {has_seats:,} seats, {has_utilization:,} utilization")

        return df

    # ==================== VALIDATION ====================

    def _validate_node_table(self, df, sector='public'):
        """
        Apply tiered validation to node table.

        Validation Tiers:
            Level 1 (required): school_id, coordinates_valid, geometry (if GeoDataFrame)
            Level 2 (core): Level 1 + (enrollment OR facilities)
            Level 3 (complete): Level 2 + enrollment + seats + facilities (public) or GASTPE (private)

        Args:
            df (DataFrame/GeoDataFrame): Node table
            sector (str): 'public' or 'private'

        Returns:
            DataFrame: Node table with validation columns
        """
        # Level 1: Required
        df['validation_level_1'] = (
            df['school_id'].notna() &
            (df['coordinates_valid'] == True)
        )

        # Add geometry check if GeoDataFrame
        if isinstance(df, gpd.GeoDataFrame):
            df['validation_level_1'] = df['validation_level_1'] & df['geometry'].notna()

        # Add spatial boundary check if admin columns exist
        # Use admin_assignment_valid if already set by _assign_admin_boundaries
        if 'admin_assignment_valid' not in df.columns:
            # Compute it if not already set
            if 'adm2_pcode' in df.columns:
                df['admin_assignment_valid'] = df['adm2_pcode'].notna()
            else:
                df['admin_assignment_valid'] = True  # No boundary data available

        # Include admin_assignment_valid in Level 1 validation
        df['validation_level_1'] = df['validation_level_1'] & df['admin_assignment_valid']

        # Level 2: Core (required + some data)
        has_enrollment = df['has_enrollment_data'] if 'has_enrollment_data' in df.columns else False
        has_facilities = df['has_facilities_data'] if 'has_facilities_data' in df.columns else False
        has_gastpe = (df['has_gastpe_data'] if 'has_gastpe_data' in df.columns else False) if sector == 'private' else False

        df['validation_level_2'] = (
            df['validation_level_1'] &
            (has_enrollment | has_facilities | has_gastpe)
        )

        # Level 3: Complete (all data sources)
        if sector == 'public':
            df['validation_level_3'] = (
                df['validation_level_2'] &
                (df['has_enrollment_data'] if 'has_enrollment_data' in df.columns else False) &
                (df['has_seats_data'] if 'has_seats_data' in df.columns else False) &
                (df['has_facilities_data'] if 'has_facilities_data' in df.columns else False)
            )
        else:  # private
            df['validation_level_3'] = (
                df['validation_level_2'] &
                (df['has_enrollment_data'] if 'has_enrollment_data' in df.columns else False) &
                (df['has_furniture_data'] if 'has_furniture_data' in df.columns else False)
            )

        # Set all_valid based on requested validation level
        if self.validation_level == 'required':
            df['all_valid'] = df['validation_level_1']
        elif self.validation_level == 'core':
            df['all_valid'] = df['validation_level_2']
        else:  # complete
            df['all_valid'] = df['validation_level_3']

        # Count results
        level_1_count = df['validation_level_1'].sum()
        level_2_count = df['validation_level_2'].sum()
        level_3_count = df['validation_level_3'].sum()

        logger.info(f"  Validation results:")
        logger.info(f"    Level 1 (required): {level_1_count:,}/{len(df):,} ({100*level_1_count/len(df):.1f}%)")
        logger.info(f"    Level 2 (core): {level_2_count:,}/{len(df):,} ({100*level_2_count/len(df):.1f}%)")
        logger.info(f"    Level 3 (complete): {level_3_count:,}/{len(df):,} ({100*level_3_count/len(df):.1f}%)")
        logger.info(f"  Using validation level: {self.validation_level}")

        return df

    # ==================== UTILITIES ====================

    def _merge_with_validation(self, base_df, other_df, on, how, data_source_name):
        """
        Merge two DataFrames and create validation flag.

        Args:
            base_df (DataFrame): Base DataFrame
            other_df (DataFrame): DataFrame to merge
            on (str): Column to merge on
            how (str): Merge type ('left', 'outer', etc.)
            data_source_name (str): Name for validation flag (e.g., 'enrollment')

        Returns:
            DataFrame: Merged DataFrame with has_{data_source_name}_data column
        """
        # Track which schools have data in other_df
        schools_with_data = set(other_df[on].dropna().unique())

        # Debug logging (temporarily using INFO to diagnose issue)
        base_schools = set(base_df[on].dropna().unique())
        overlap = base_schools.intersection(schools_with_data)
        if len(overlap) == 0:
            # Only log if there's a problem
            logger.warning(f"  {data_source_name} merge: NO OVERLAP!")
            logger.warning(f"    Base schools: {len(base_schools):,}")
            logger.warning(f"    Other schools: {len(schools_with_data):,}")
            # Show sample IDs to diagnose
            logger.warning(f"    Sample base IDs: {list(base_schools)[:5]}")
            logger.warning(f"    Sample other IDs: {list(schools_with_data)[:5]}")

        # Merge
        merged = pd.merge(base_df, other_df, on=on, how=how, suffixes=('', f'_{data_source_name}'))

        # Create validation flag
        validation_col = f'has_{data_source_name}_data'
        merged[validation_col] = merged[on].isin(schools_with_data)

        matched_count = merged[validation_col].sum()
        logger.info(f"  Merged {data_source_name}: {matched_count:,}/{len(merged):,} schools have data")

        return merged

    def _pivot_by_education_level(self, df, index_col, columns_col, value_col, prefix):
        """
        Pivot long-format data to wide format by education level.

        Maps education levels to ES/JHS/SHS:
            ES: Kindergarten, G1-G6
            JHS: G7-G10
            SHS: G11-G12

        Args:
            df (DataFrame): Long-format data
            index_col (str): Index column (usually 'school_id')
            columns_col (str): Column to pivot (usually 'grade_level' or 'education_level')
            value_col (str): Value column to aggregate
            prefix (str): Prefix for output columns (e.g., 'enrollment', 'seats')

        Returns:
            DataFrame: Pivoted data with columns {prefix}_es, {prefix}_jhs, {prefix}_shs
        """
        # Create education level mapping if pivoting by grade_level
        if columns_col == 'grade_level':
            df = df.copy()

            # Map grade levels to education levels
            es_grades = ['Kindergarten', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'Elementary', 'elementary']
            jhs_grades = ['G7', 'G8', 'G9', 'G10', 'JHS', 'Junior High School', 'junior high school']
            shs_grades = ['G11', 'G12', 'SHS', 'Senior High School', 'senior high school']

            def map_to_education_level(grade):
                if grade in es_grades:
                    return 'ES'
                elif grade in jhs_grades:
                    return 'JHS'
                elif grade in shs_grades:
                    return 'SHS'
                else:
                    return None

            df['education_level'] = df[columns_col].apply(map_to_education_level)

            # Group by school and education level
            df_grouped = df.groupby([index_col, 'education_level'])[value_col].sum().reset_index()
            columns_col = 'education_level'
            df = df_grouped

        # Pivot table
        pivot = df.pivot_table(
            index=index_col,
            columns=columns_col,
            values=value_col,
            aggfunc='sum'
        ).reset_index()

        # Rename columns (handle all education level variations)
        rename_map = {}
        for col in pivot.columns:
            if col == index_col:
                continue
            # Elementary variations
            elif col in ['Elementary', 'ES', 'elementary']:
                rename_map[col] = f'{prefix}_es'
            # Junior High School variations
            elif col in ['JHS', 'Junior High School', 'junior high school', 'Junior HS']:
                rename_map[col] = f'{prefix}_jhs'
            # Senior High School variations
            elif col in ['SHS', 'Senior High School', 'senior high school', 'Senior HS']:
                rename_map[col] = f'{prefix}_shs'

        pivot.rename(columns=rename_map, inplace=True)

        return pivot

    # ==================== REPORTING ====================

    def get_summary(self):
        """
        Get comprehensive summary of both node tables.

        Returns:
            dict: Summary statistics for public and private tables
        """
        summary = {}

        if self.public_node_table is not None:
            summary['public'] = self._get_table_summary(self.public_node_table, 'public')

        if self.private_node_table is not None:
            summary['private'] = self._get_table_summary(self.private_node_table, 'private')

        if self.combined_node_table is not None:
            summary['combined'] = {
                'total_schools': len(self.combined_node_table),
                'public_count': len(self.combined_node_table[self.combined_node_table['sector'] == 'public']),
                'private_count': len(self.combined_node_table[self.combined_node_table['sector'] == 'private'])
            }

        return summary

    def get_public_summary(self):
        """Get summary for public node table only."""
        if self.public_node_table is None:
            logger.warning("Public node table not built yet")
            return None
        return self._get_table_summary(self.public_node_table, 'public')

    def get_private_summary(self):
        """Get summary for private node table only."""
        if self.private_node_table is None:
            logger.warning("Private node table not built yet")
            return None
        return self._get_table_summary(self.private_node_table, 'private')

    def _get_table_summary(self, df, sector):
        """
        Calculate summary statistics for a node table.

        Args:
            df (DataFrame): Node table
            sector (str): 'public' or 'private'

        Returns:
            dict: Summary statistics
        """
        summary = {
            'total_schools': len(df),
            'validation_breakdown': {},
            'completeness_by_source': {},
            'spatial_coverage': {},
            'computed_metrics': {}
        }

        # Validation breakdown
        if 'validation_level_1' in df.columns:
            summary['validation_breakdown']['level_1_required'] = int(df['validation_level_1'].sum())
        if 'validation_level_2' in df.columns:
            summary['validation_breakdown']['level_2_core'] = int(df['validation_level_2'].sum())
        if 'validation_level_3' in df.columns:
            summary['validation_breakdown']['level_3_complete'] = int(df['validation_level_3'].sum())
        if 'all_valid' in df.columns:
            summary['validation_breakdown']['all_valid'] = int(df['all_valid'].sum())

        # Completeness by source
        data_source_cols = [col for col in df.columns if col.startswith('has_') and col.endswith('_data')]
        for col in data_source_cols:
            source_name = col.replace('has_', '').replace('_data', '')
            count = int(df[col].sum())
            pct = 100 * count / len(df)
            summary['completeness_by_source'][source_name] = {
                'count': count,
                'percentage': round(pct, 1)
            }

        # Spatial coverage
        if 'coordinates_valid' in df.columns:
            valid_coords = int(df['coordinates_valid'].sum())
            summary['spatial_coverage']['valid_coordinates'] = valid_coords
            summary['spatial_coverage']['invalid_coordinates'] = len(df) - valid_coords

        if 'adm2_pcode' in df.columns:
            matched = int(df['adm2_pcode'].notna().sum())
            summary['spatial_coverage']['admin_boundary_matched'] = matched
            summary['spatial_coverage']['admin_boundary_unmatched'] = len(df) - matched

        # Computed metrics
        if 'total_enrollment' in df.columns:
            summary['computed_metrics']['schools_with_enrollment'] = int(df['total_enrollment'].notna().sum())
            summary['computed_metrics']['total_enrollment_sum'] = int(df['total_enrollment'].sum())

        if 'total_seats' in df.columns:
            summary['computed_metrics']['schools_with_seats'] = int(df['total_seats'].notna().sum())
            summary['computed_metrics']['total_seats_sum'] = int(df['total_seats'].sum())

        if 'capacity_utilization' in df.columns:
            valid_util = df['capacity_utilization'].notna()
            summary['computed_metrics']['schools_with_utilization'] = int(valid_util.sum())
            if valid_util.sum() > 0:
                summary['computed_metrics']['avg_capacity_utilization'] = round(
                    df.loc[valid_util, 'capacity_utilization'].mean(), 2
                )

        return summary

    def get_validation_report(self):
        """
        Get detailed validation report showing failed validations.

        Returns:
            DataFrame: Report with validation issues
        """
        reports = []

        if self.public_node_table is not None:
            public_report = self._get_validation_issues(self.public_node_table, 'public')
            reports.append(public_report)

        if self.private_node_table is not None:
            private_report = self._get_validation_issues(self.private_node_table, 'private')
            reports.append(private_report)

        if reports:
            return pd.concat(reports, ignore_index=True)
        else:
            return pd.DataFrame()

    def _get_validation_issues(self, df, sector):
        """Extract schools with validation issues."""
        invalid_mask = ~df['all_valid'] if 'all_valid' in df.columns else pd.Series([False] * len(df))
        invalid_schools = df[invalid_mask].copy()

        if len(invalid_schools) == 0:
            return pd.DataFrame()

        # Create issue descriptions
        issues = []
        for idx, row in invalid_schools.iterrows():
            issue_list = []

            if 'coordinates_valid' in row and not row['coordinates_valid']:
                issue_list.append('invalid_coordinates')
            if 'admin_assignment_valid' in row and not row['admin_assignment_valid']:
                issue_list.append('no_admin_boundary')
            if 'has_enrollment_data' in row and not row['has_enrollment_data']:
                issue_list.append('no_enrollment')
            if 'has_seats_data' in row and not row['has_seats_data']:
                issue_list.append('no_seats')
            if 'has_facilities_data' in row and not row['has_facilities_data']:
                issue_list.append('no_facilities')

            issues.append(', '.join(issue_list) if issue_list else 'unknown')

        invalid_schools['validation_issues'] = issues
        invalid_schools['sector'] = sector

        # Select relevant columns
        keep_cols = ['school_id', 'sector', 'validation_issues']
        if 'school_name' in invalid_schools.columns:
            keep_cols.insert(1, 'school_name')

        existing_cols = [col for col in keep_cols if col in invalid_schools.columns]

        return invalid_schools[existing_cols]

    # ==================== EXPORT ====================

    def export_geopackage(self, path, sector='both', valid_only=False):
        """
        Export node table(s) to GeoPackage format.

        Args:
            path (str): Output path (.gpkg), relative to project root or absolute
            sector (str): 'public', 'private', or 'both'
            valid_only (bool): Export only valid schools
        """
        path = self._resolve_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if sector == 'both':
            if self.combined_node_table is None:
                self.build_combined_node_table()
            df_to_export = self.combined_node_table
            layer_name = 'schools'
        elif sector == 'public':
            if self.public_node_table is None:
                self.build_public_node_table()
            df_to_export = self.public_node_table
            layer_name = 'public_schools'
        elif sector == 'private':
            if self.private_node_table is None:
                self.build_private_node_table()
            df_to_export = self.private_node_table
            layer_name = 'private_schools'
        else:
            raise ValueError("sector must be 'public', 'private', or 'both'")

        # Filter to valid only if requested
        if valid_only and 'all_valid' in df_to_export.columns:
            df_to_export = df_to_export[df_to_export['all_valid']].copy()
            logger.info(f"Filtered to {len(df_to_export):,} valid schools")

        # Export
        if isinstance(df_to_export, gpd.GeoDataFrame):
            df_to_export.to_file(path, layer=layer_name, driver='GPKG')
            logger.info(f"Exported {len(df_to_export):,} schools to {path}")
        else:
            logger.warning("Cannot export to GeoPackage: not a GeoDataFrame")

    def export_csv(self, path, sector='both', valid_only=False):
        """
        Export node table(s) to CSV format.

        Args:
            path (str): Output path (.csv), relative to project root or absolute
            sector (str): 'public', 'private', or 'both'
            valid_only (bool): Export only valid schools
        """
        path = self._resolve_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if sector == 'both':
            if self.combined_node_table is None:
                self.build_combined_node_table()
            df_to_export = self.combined_node_table
        elif sector == 'public':
            if self.public_node_table is None:
                self.build_public_node_table()
            df_to_export = self.public_node_table
        elif sector == 'private':
            if self.private_node_table is None:
                self.build_private_node_table()
            df_to_export = self.private_node_table
        else:
            raise ValueError("sector must be 'public', 'private', or 'both'")

        # Filter to valid only if requested
        if valid_only and 'all_valid' in df_to_export.columns:
            df_to_export = df_to_export[df_to_export['all_valid']].copy()
            logger.info(f"Filtered to {len(df_to_export):,} valid schools")

        # Drop geometry column if present
        if isinstance(df_to_export, gpd.GeoDataFrame):
            df_to_export = pd.DataFrame(df_to_export.drop(columns=['geometry']))

        # Export
        df_to_export.to_csv(path, index=False)
        logger.info(f"Exported {len(df_to_export):,} schools to {path}")

    def export_parquet(self, path, sector='both', valid_only=False):
        """
        Export node table(s) to Parquet format.

        Args:
            path (str): Output path (.parquet), relative to project root or absolute
            sector (str): 'public', 'private', or 'both'
            valid_only (bool): Export only valid schools
        """
        path = self._resolve_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if sector == 'both':
            if self.combined_node_table is None:
                self.build_combined_node_table()
            df_to_export = self.combined_node_table
        elif sector == 'public':
            if self.public_node_table is None:
                self.build_public_node_table()
            df_to_export = self.public_node_table
        elif sector == 'private':
            if self.private_node_table is None:
                self.build_private_node_table()
            df_to_export = self.private_node_table
        else:
            raise ValueError("sector must be 'public', 'private', or 'both'")

        # Filter to valid only if requested
        if valid_only and 'all_valid' in df_to_export.columns:
            df_to_export = df_to_export[df_to_export['all_valid']].copy()
            logger.info(f"Filtered to {len(df_to_export):,} valid schools")

        # Export (Parquet supports GeoDataFrame)
        df_to_export.to_parquet(path, index=False)
        logger.info(f"Exported {len(df_to_export):,} schools to {path}")

    def export_quality_report(self, path):
        """
        Export comprehensive quality report to CSV.

        Args:
            path (str): Output path (.csv), relative to project root or absolute
        """
        path = self._resolve_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        summary = self.get_summary()

        # Flatten summary to DataFrame
        rows = []
        for sector, data in summary.items():
            if sector == 'combined':
                continue

            # Overall
            rows.append({
                'sector': sector,
                'metric': 'total_schools',
                'value': data.get('total_schools', 0)
            })

            # Validation breakdown
            for level, count in data.get('validation_breakdown', {}).items():
                rows.append({
                    'sector': sector,
                    'metric': f'validation_{level}',
                    'value': count
                })

            # Completeness
            for source, stats in data.get('completeness_by_source', {}).items():
                rows.append({
                    'sector': sector,
                    'metric': f'completeness_{source}_count',
                    'value': stats['count']
                })
                rows.append({
                    'sector': sector,
                    'metric': f'completeness_{source}_pct',
                    'value': stats['percentage']
                })

            # Spatial coverage
            for metric, value in data.get('spatial_coverage', {}).items():
                rows.append({
                    'sector': sector,
                    'metric': f'spatial_{metric}',
                    'value': value
                })

            # Computed metrics
            for metric, value in data.get('computed_metrics', {}).items():
                rows.append({
                    'sector': sector,
                    'metric': f'computed_{metric}',
                    'value': value
                })

        report_df = pd.DataFrame(rows)
        report_df.to_csv(path, index=False)
        logger.info(f"Exported quality report to {path}")
