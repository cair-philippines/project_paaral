"""
PSGC (Philippine Standard Geographic Code) Data Consolidator

This module consolidates Philippines geographic administrative hierarchy data from
multiple CSV files and merges with shapefile geometries. It creates a comprehensive
GeoDataFrame with complete hierarchical information from regions to barangays.

The consolidation process:
1. Loads administrative data at 4 levels: Regions (Adm1), Provinces/Districts (Adm2),
   Municipalities/Cities (Adm3), and Barangays/Sub-Municipalities (Adm4)
2. Performs hierarchical joins using PSGC codes
3. Merges with shapefile geometries to create a complete GeoDataFrame

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

    def consolidate_hierarchy(self) -> pd.DataFrame:
        """
        Perform hierarchical joins on PSGC codes to consolidate all administrative levels.

        The join strategy:
        1. Start with Adm4 (Barangays) as base
        2. Join with Adm3 (Municipalities) on [adm1_psgc, adm2_psgc, adm3_psgc]
        3. Join with Adm2 (Provinces) on [adm1_psgc, adm2_psgc]
        4. Join with Adm1 (Regions) on [adm1_psgc]

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

    def merge_with_geometry(self) -> gpd.GeoDataFrame:
        """
        Merge consolidated CSV data with shapefile geometries.

        Returns:
            GeoDataFrame with all hierarchical data and geometries
        """
        if self.consolidated_data is None:
            raise ValueError("Data not consolidated. Call consolidate_hierarchy() first.")

        if self.adm4_geodata is None:
            raise ValueError("Shapefile not loaded. Call load_data() first.")

        logger.info("Merging consolidated data with geometries...")

        # Identify PSGC column in shapefile
        logger.info(f"Shapefile columns: {list(self.adm4_geodata.columns)}")

        # The shapefile uses 'psgc_code' while CSV uses 'adm4_psgc'
        # Merge on these keys
        if 'psgc_code' not in self.adm4_geodata.columns:
            raise ValueError(f"Could not find psgc_code column in shapefile. Available columns: {list(self.adm4_geodata.columns)}")

        if 'adm4_psgc' not in self.consolidated_data.columns:
            raise ValueError(f"Could not find adm4_psgc column in consolidated data. Available columns: {list(self.consolidated_data.columns)}")

        logger.info("Merging on shapefile 'psgc_code' = consolidated 'adm4_psgc'")

        # Merge on the PSGC code columns
        consolidated_geo = self.adm4_geodata.merge(
            self.consolidated_data,
            left_on='psgc_code',
            right_on='adm4_psgc',
            how='right',
            suffixes=('_shp', '_csv')
        )

        # Convert to GeoDataFrame
        self.consolidated_geodata = gpd.GeoDataFrame(
            consolidated_geo,
            geometry='geometry',
            crs=self.adm4_geodata.crs
        )

        # Clean up duplicate columns from merge
        # Keep non-geometry columns from consolidated_data
        duplicate_cols = [col for col in self.consolidated_geodata.columns if col.endswith('_geo')]
        if duplicate_cols:
            # For each duplicate, check if we should keep it
            for col in duplicate_cols:
                base_col = col.replace('_geo', '')
                if base_col in self.consolidated_geodata.columns and base_col != 'geometry':
                    # Drop the _geo version if base column exists
                    self.consolidated_geodata = self.consolidated_geodata.drop(columns=[col])
                    logger.info(f"Dropped duplicate geometry column: {col}")

        logger.info(f"Geometry merge complete: {len(self.consolidated_geodata)} features")
        logger.info(f"Features with valid geometry: {self.consolidated_geodata.geometry.notna().sum()}")
        logger.info(f"Features with null geometry: {self.consolidated_geodata.geometry.isna().sum()}")

        return self.consolidated_geodata

    def process(self) -> gpd.GeoDataFrame:
        """
        Main processing pipeline: load, consolidate, and merge with geometries.

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
    # Initialize consolidator
    consolidator = PSGCConsolidator()

    # Process data
    geodata = consolidator.process()

    # Get summary
    summary = consolidator.get_summary()
    print("\nPSGC Consolidated Data Summary:")
    print(f"Total features: {summary['total_features']:,}")
    print(f"Features with geometry: {summary['features_with_geometry']:,}")
    print(f"Unique regions: {summary['unique_regions']}")
    print(f"Unique provinces: {summary['unique_provinces']}")
    print(f"Unique municipalities: {summary['unique_municipalities']}")
    print(f"Unique barangays: {summary['unique_barangays']}")
    print(f"CRS: {summary['crs']}")
    if summary['total_area_km2']:
        print(f"Total area: {summary['total_area_km2']:,.2f} km²")

    # Show sample data
    print("\nSample data:")
    print(geodata.head(5))

    # Export processed data
    consolidator.export_processed("output/psgc_consolidated.geojson")
    consolidator.export_processed("output/psgc_consolidated.csv")