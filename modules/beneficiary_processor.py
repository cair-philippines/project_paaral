"""
Beneficiary Data Processor for School Network Analysis

This module processes ESC/SHSVP beneficiary flow data and validates it against
school node tables to create clean edge lists for graph network construction.

STUDENT FLOW DIRECTION: ORIGIN → DESTINATION
- ORIGIN schools (lrn_school_id): Where students came FROM (public or private schools)
- DESTINATION schools (deped_school_id/school_id_esc): Where students GO TO (ESC recipient private schools)

Key Features:
- Loads and aggregates beneficiary data from parquet files
- Validates school IDs against public and private node tables
- Creates validated edge lists (origin → destination student flows)
- Supports provincial filtering via adm2_pcode
- Provides comprehensive quality reporting

Usage:
    from modules.beneficiary_processor import BeneficiaryProcessor

    processor = BeneficiaryProcessor(
        public_nodes_path='output/public_nodes_valid.gpkg',
        private_nodes_path='output/private_nodes_valid.gpkg'
    )

    # Process beneficiary data
    processor.load_beneficiary_data('data/processed/esc_beneficiaries.parquet')

    # Get validated edges
    valid_edges = processor.get_valid_edges()

    # Filter to specific province
    bulacan_edges = processor.filter_by_province('PH03014')

    # Export for graph building
    processor.export_edge_list('output/beneficiary_edges.csv')

Author: Claude Code
Date: 2025-11-13
"""

import pandas as pd
import geopandas as gpd
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Set

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BeneficiaryProcessor:
    """
    Processes and validates beneficiary flow data for school network analysis.

    Attributes:
        public_nodes (GeoDataFrame): Valid public school nodes
        private_nodes (GeoDataFrame): Valid private school nodes
        beneficiary_data (DataFrame): Raw beneficiary data
        validated_edges (DataFrame): Validated edge list with flags
    """

    def __init__(
        self,
        public_nodes_path: Optional[str] = None,
        private_nodes_path: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize BeneficiaryProcessor.

        Args:
            public_nodes_path (str): Path to public nodes GeoPackage
            private_nodes_path (str): Path to private nodes GeoPackage
            verbose (bool): Enable verbose logging
        """
        self.verbose = verbose
        if not verbose:
            logger.setLevel(logging.WARNING)

        # Default paths
        self.public_nodes_path = public_nodes_path or 'output/public_nodes_valid.gpkg'
        self.private_nodes_path = private_nodes_path or 'output/private_nodes_valid.gpkg'

        # Data storage
        self.public_nodes = None
        self.private_nodes = None
        self.beneficiary_data = None
        self.validated_edges = None

        # Valid ID sets for fast lookup
        self.valid_public_ids = None
        self.valid_private_ids = None

        logger.info("BeneficiaryProcessor initialized")

    def load_node_tables(self):
        """
        Load valid public and private node tables.

        Creates sets of valid school IDs for validation.
        """
        logger.info("Loading node tables...")

        # Load public nodes
        if Path(self.public_nodes_path).exists():
            self.public_nodes = gpd.read_file(self.public_nodes_path)
            self.valid_public_ids = set(self.public_nodes['school_id'].astype(str))
            logger.info(f"  Loaded {len(self.public_nodes):,} valid public schools")
        else:
            logger.warning(f"  Public nodes not found at {self.public_nodes_path}")
            self.valid_public_ids = set()

        # Load private nodes
        if Path(self.private_nodes_path).exists():
            self.private_nodes = gpd.read_file(self.private_nodes_path)
            self.valid_private_ids = set(self.private_nodes['school_id'].astype(str))
            logger.info(f"  Loaded {len(self.private_nodes):,} valid private schools")
        else:
            logger.warning(f"  Private nodes not found at {self.private_nodes_path}")
            self.valid_private_ids = set()

    def load_beneficiary_data(
        self,
        beneficiary_path: str,
        origin_col: str = 'lrn_school_id',
        destination_col: str = 'deped_school_id',
        count_col: str = 'lrn',
        additional_cols: Optional[list] = None
    ):
        """
        Load and aggregate beneficiary data from parquet file.

        Student flow direction: ORIGIN → DESTINATION
        - Origin schools (lrn_school_id): Where students came FROM (public or private)
        - Destination schools (deped_school_id/school_id_esc): Where students GO TO (ESC recipient private schools)

        Args:
            beneficiary_path (str): Path to beneficiary parquet file
            origin_col (str): Column name for origin school ID (default: 'lrn_school_id')
            destination_col (str): Column name for destination school ID (default: 'deped_school_id')
            count_col (str): Column to count (e.g., 'lrn' for unique learners)
            additional_cols (list): Additional columns to include in groupby
        """
        logger.info(f"Loading beneficiary data from {beneficiary_path}...")

        # Load data
        df = pd.read_parquet(beneficiary_path)
        logger.info(f"  Loaded {len(df):,} raw beneficiary records")

        # Define columns for aggregation
        groupby_cols = [destination_col, origin_col]
        if additional_cols:
            groupby_cols.extend(additional_cols)

        # Aggregate by origin-destination pairs
        logger.info("  Aggregating beneficiary flows...")
        agg = (
            df.groupby(groupby_cols)
            .agg({count_col: 'nunique'})
            .reset_index()
            .rename(columns={
                destination_col: 'school_id_destination',
                origin_col: 'school_id_origin',
                count_col: 'beneficiary_count'
            })
        )

        # Ensure IDs are strings
        agg['school_id_destination'] = agg['school_id_destination'].astype(str)
        agg['school_id_origin'] = agg['school_id_origin'].astype(str)

        self.beneficiary_data = agg
        logger.info(f"  Aggregated to {len(agg):,} unique origin-destination pairs")
        logger.info(f"  Unique origins: {agg['school_id_origin'].nunique():,}")
        logger.info(f"  Unique destinations: {agg['school_id_destination'].nunique():,}")

    def validate_edges(self):
        """
        Validate beneficiary edges against node tables.

        Student flow direction: ORIGIN → DESTINATION
        - ORIGIN schools: Where students came FROM (lrn_school_id → school_id_origin)
        - DESTINATION schools: Where students GO TO (deped_school_id/school_id_esc → school_id_destination)

        Creates validation flags:
        - destination_in_private_nodes: Destination school (ESC recipient) exists in private nodes
        - origin_in_public_nodes: Origin school exists in public nodes
        - origin_in_private_nodes: Origin school exists in private nodes
        - origin_valid: Origin school exists in either node table
        - both_schools_valid: Complete edge (both origin and destination valid)
        """
        if self.beneficiary_data is None:
            raise ValueError("No beneficiary data loaded. Call load_beneficiary_data() first.")

        if self.valid_public_ids is None or self.valid_private_ids is None:
            self.load_node_tables()

        logger.info("Validating edges against node tables...")
        logger.info("  Flow direction: ORIGIN (where students came from) → DESTINATION (ESC recipient schools)")

        df = self.beneficiary_data.copy()

        # Validate DESTINATION schools (ESC recipients - should be private schools receiving subsidies)
        df['destination_in_private_nodes'] = df['school_id_destination'].isin(self.valid_private_ids)

        # Validate ORIGIN schools (where students came from - could be either public or private)
        df['origin_in_public_nodes'] = df['school_id_origin'].isin(self.valid_public_ids)
        df['origin_in_private_nodes'] = df['school_id_origin'].isin(self.valid_private_ids)

        # Origin is valid if found in either node table
        df['origin_valid'] = df['origin_in_public_nodes'] | df['origin_in_private_nodes']

        # Both schools valid (complete edge: origin → destination)
        df['both_schools_valid'] = df['destination_in_private_nodes'] & df['origin_valid']

        self.validated_edges = df

        # Log validation results
        total = len(df)
        dest_valid = df['destination_in_private_nodes'].sum()
        origin_pub = df['origin_in_public_nodes'].sum()
        origin_priv = df['origin_in_private_nodes'].sum()
        origin_valid = df['origin_valid'].sum()
        both_valid = df['both_schools_valid'].sum()

        logger.info(f"Validation complete:")
        logger.info(f"  Total edges: {total:,}")
        logger.info(f"  DESTINATION validation (ESC recipient schools → private nodes): {dest_valid:,} ({dest_valid/total*100:.1f}%)")
        logger.info(f"  ORIGIN validation (where students came from):")
        logger.info(f"    In public nodes: {origin_pub:,} ({origin_pub/total*100:.1f}%)")
        logger.info(f"    In private nodes: {origin_priv:,} ({origin_priv/total*100:.1f}%)")
        logger.info(f"    Valid (either): {origin_valid:,} ({origin_valid/total*100:.1f}%)")
        logger.info(f"  Complete edges (ORIGIN → DESTINATION both valid): {both_valid:,} ({both_valid/total*100:.1f}%)")

        # Log unique school statistics
        logger.info(f"Unique school analysis:")
        logger.info(f"  Unique destinations:")
        logger.info(f"    Total: {df['school_id_destination'].nunique():,}")
        logger.info(f"    Matched: {df[df['destination_in_private_nodes']]['school_id_destination'].nunique():,}")
        logger.info(f"  Unique origins:")
        logger.info(f"    Total: {df['school_id_origin'].nunique():,}")
        logger.info(f"    In public nodes: {df[df['origin_in_public_nodes']]['school_id_origin'].nunique():,}")
        logger.info(f"    In private nodes: {df[df['origin_in_private_nodes']]['school_id_origin'].nunique():,}")
        logger.info(f"    Matched (either): {df[df['origin_valid']]['school_id_origin'].nunique():,}")

    def get_valid_edges(self) -> pd.DataFrame:
        """
        Get only validated edges where both schools exist in node tables.

        Returns:
            DataFrame: Filtered edges with both_schools_valid=True
        """
        if self.validated_edges is None:
            raise ValueError("No validated edges. Call validate_edges() first.")

        valid = self.validated_edges[self.validated_edges['both_schools_valid']].copy()
        logger.info(f"Returning {len(valid):,} valid edges")
        return valid

    def get_all_edges(self) -> pd.DataFrame:
        """
        Get all edges with validation flags.

        Returns:
            DataFrame: All edges with validation columns
        """
        if self.validated_edges is None:
            raise ValueError("No validated edges. Call validate_edges() first.")

        return self.validated_edges.copy()

    def filter_by_province(
        self,
        adm2_pcode: str,
        valid_only: bool = True
    ) -> pd.DataFrame:
        """
        Filter edges to schools within a specific province.

        Args:
            adm2_pcode (str): Province code (e.g., 'PH03014' for Bulacan)
            valid_only (bool): Return only valid edges

        Returns:
            DataFrame: Filtered edges
        """
        if self.validated_edges is None:
            raise ValueError("No validated edges. Call validate_edges() first.")

        if self.public_nodes is None or self.private_nodes is None:
            self.load_node_tables()

        logger.info(f"Filtering edges for province {adm2_pcode}...")

        # Get school IDs in this province
        province_public_ids = set()
        province_private_ids = set()

        if 'adm2_pcode' in self.public_nodes.columns:
            province_public = self.public_nodes[
                self.public_nodes['adm2_pcode'] == adm2_pcode
            ]
            province_public_ids = set(province_public['school_id'].astype(str))

        if 'adm2_pcode' in self.private_nodes.columns:
            province_private = self.private_nodes[
                self.private_nodes['adm2_pcode'] == adm2_pcode
            ]
            province_private_ids = set(province_private['school_id'].astype(str))

        province_school_ids = province_public_ids | province_private_ids

        # Filter edges where either origin or destination is in province
        df = self.validated_edges if not valid_only else self.get_valid_edges()

        filtered = df[
            df['school_id_origin'].isin(province_school_ids) |
            df['school_id_destination'].isin(province_school_ids)
        ].copy()

        logger.info(f"  Found {len(filtered):,} edges involving province schools")
        logger.info(f"    Origins in province: {filtered['school_id_origin'].isin(province_school_ids).sum():,}")
        logger.info(f"    Destinations in province: {filtered['school_id_destination'].isin(province_school_ids).sum():,}")

        return filtered

    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary statistics.

        Returns:
            dict: Summary statistics
        """
        if self.validated_edges is None:
            raise ValueError("No validated edges. Call validate_edges() first.")

        df = self.validated_edges

        # Origin school type breakdown
        origin_public_only = (df['origin_in_public_nodes'] & ~df['origin_in_private_nodes']).sum()
        origin_private_only = (~df['origin_in_public_nodes'] & df['origin_in_private_nodes']).sum()
        origin_both = (df['origin_in_public_nodes'] & df['origin_in_private_nodes']).sum()
        origin_neither = (~df['origin_valid']).sum()

        summary = {
            'total_edges': len(df),
            'valid_edges': df['both_schools_valid'].sum(),
            'validation_rate': df['both_schools_valid'].sum() / len(df) * 100,
            'destination_validation': {
                'in_private_nodes': df['destination_in_private_nodes'].sum(),
                'percentage': df['destination_in_private_nodes'].sum() / len(df) * 100
            },
            'origin_validation': {
                'in_public_nodes': df['origin_in_public_nodes'].sum(),
                'in_private_nodes': df['origin_in_private_nodes'].sum(),
                'valid_either': df['origin_valid'].sum(),
                'percentage': df['origin_valid'].sum() / len(df) * 100
            },
            'origin_type_breakdown': {
                'public_only': origin_public_only,
                'private_only': origin_private_only,
                'both_datasets': origin_both,
                'neither': origin_neither
            },
            'unique_schools': {
                'total_origins': df['school_id_origin'].nunique(),
                'total_destinations': df['school_id_destination'].nunique(),
                'origins_matched': df[df['origin_valid']]['school_id_origin'].nunique(),
                'destinations_matched': df[df['destination_in_private_nodes']]['school_id_destination'].nunique()
            },
            'beneficiary_stats': {
                'total_beneficiaries': df['beneficiary_count'].sum(),
                'valid_beneficiaries': df[df['both_schools_valid']]['beneficiary_count'].sum(),
                'mean_per_edge': df['beneficiary_count'].mean(),
                'median_per_edge': df['beneficiary_count'].median()
            }
        }

        return summary

    def export_edge_list(
        self,
        path: str,
        valid_only: bool = True,
        include_validation_flags: bool = False
    ):
        """
        Export edge list to CSV.

        Args:
            path (str): Output file path
            valid_only (bool): Export only valid edges
            include_validation_flags (bool): Include validation columns
        """
        if self.validated_edges is None:
            raise ValueError("No validated edges. Call validate_edges() first.")

        # Select data
        df = self.get_valid_edges() if valid_only else self.validated_edges.copy()

        # Select columns
        if include_validation_flags:
            export_df = df
        else:
            # Keep only essential columns
            export_cols = [
                'school_id_origin',
                'school_id_destination',
                'beneficiary_count'
            ]
            # Add any additional columns from original data
            other_cols = [c for c in df.columns if c not in export_cols and not c.endswith('_nodes') and c != 'origin_valid' and c != 'both_schools_valid']
            export_cols.extend(other_cols)
            export_df = df[export_cols]

        # Export
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        export_df.to_csv(path, index=False)

        logger.info(f"Exported {len(export_df):,} edges to {path}")

    def process(
        self,
        beneficiary_path: str,
        export_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Complete processing pipeline: load → validate → return.

        Args:
            beneficiary_path (str): Path to beneficiary parquet file
            export_path (str, optional): Export valid edges to this path

        Returns:
            DataFrame: Validated edges
        """
        logger.info("="*60)
        logger.info("Starting beneficiary processing pipeline")
        logger.info("="*60)

        # Load node tables
        self.load_node_tables()

        # Load and aggregate beneficiary data
        self.load_beneficiary_data(beneficiary_path)

        # Validate
        self.validate_edges()

        # Export if requested
        if export_path:
            self.export_edge_list(export_path, valid_only=True)

        logger.info("="*60)
        logger.info("Processing complete")
        logger.info("="*60)

        return self.get_valid_edges()


# Example usage
if __name__ == "__main__":
    processor = BeneficiaryProcessor()

    # Process beneficiary data
    valid_edges = processor.process(
        beneficiary_path='data/processed/esc_beneficiaries.parquet',
        export_path='output/beneficiary_edges_valid.csv'
    )

    # Get summary
    summary = processor.get_summary()
    print("\nSummary Statistics:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Filter to specific province
    bulacan_edges = processor.filter_by_province('PH03014')
    print(f"\nBulacan edges: {len(bulacan_edges)}")
