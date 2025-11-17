"""
School coordinates data preprocessor module.
Handles preprocessing of school location data with coordinate validation and data quality checks.
Author: Data Processing System
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PublicSchoolsProcessor:
    """Processor for school coordinates data with validation and quality checks."""

    # Philippine coordinate boundaries
    PH_LONGITUDE_MIN = 116.0
    PH_LONGITUDE_MAX = 127.0
    PH_LATITUDE_MIN = 4.0
    PH_LATITUDE_MAX = 21.0

    def __init__(self, file_path: Optional[str] = None, verbose: bool = True):
        """Initialize processor with optional file path and verbose logging control."""
        self.excel_path = file_path or "data/public/SY 2023-2024 LIST OF SCHOOLS WITH LONGITUDE AND LATITUDE.xlsx"
        self.verbose = verbose
        self.raw_data = None
        self.processed_data = None
        self.quality_report = {}

        # Configure logging based on verbose setting
        if not verbose:
            logging.getLogger(__name__).setLevel(logging.WARNING)

    def _log(self, message: str, level: str = 'info') -> None:
        """Conditional logging based on verbose setting."""
        if self.verbose or level in ['warning', 'error']:
            if level == 'info':
                logger.info(message)
            elif level == 'warning':
                logger.warning(message)
            elif level == 'error':
                logger.error(message)
            elif level == 'debug':
                logger.debug(message)

    def load_data(self) -> pd.DataFrame:
        """Load and perform initial cleaning of coordinate data."""
        try:
            # Try to read Excel file with different sheet possibilities
            excel_file = pd.ExcelFile(self.excel_path)
            self._log(f"Available sheets: {excel_file.sheet_names}")

            # Use the first sheet or look for a specific sheet name
            if len(excel_file.sheet_names) == 1:
                sheet_name = excel_file.sheet_names[0]
            else:
                # Look for common sheet names
                common_names = ['Sheet1', 'Schools', 'Data', 'Coordinates']
                sheet_name = None
                for name in common_names:
                    if name in excel_file.sheet_names:
                        sheet_name = name
                        break
                if not sheet_name:
                    sheet_name = excel_file.sheet_names[0]

            self._log(f"Using sheet: {sheet_name}")

            # Read Excel file with correct structure:
            # - Skip first 5 rows (metadata/title information)
            # - Use row 6 (0-indexed row 5) as header row
            # - Data starts from row 7 onwards
            self._log("Reading Excel file with header at row 6 (0-indexed row 5), skipping first 5 rows")
            self.raw_data = pd.read_excel(
                self.excel_path,
                sheet_name=sheet_name,
                header=5,  # Use row 6 (0-indexed row 5) as header
                skiprows=None  # Don't skip additional rows since header parameter handles it
            )

            # Clean column names - remove extra spaces and standardize
            self.raw_data.columns = self.raw_data.columns.str.strip()
            self._log(f"Detected columns from header row: {list(self.raw_data.columns)}")

            # Remove any completely empty rows
            original_length = len(self.raw_data)
            self.raw_data = self.raw_data.dropna(how='all')
            self._log(f"Removed {original_length - len(self.raw_data)} completely empty rows")

            # Log column detection for coordinates
            coordinate_columns_found = []
            for col in self.raw_data.columns:
                col_lower = col.lower().strip()
                if any(term in col_lower for term in ['longitude', 'long', 'lng']):
                    coordinate_columns_found.append(f"Longitude-like: '{col}'")
                elif any(term in col_lower for term in ['latitude', 'lat']):
                    coordinate_columns_found.append(f"Latitude-like: '{col}'")

            if coordinate_columns_found:
                self._log(f"Found coordinate columns: {coordinate_columns_found}")
            else:
                self._log("No coordinate columns detected in header row", 'warning')

            self._log(f"Successfully loaded {len(self.raw_data)} records with {len(self.raw_data.columns)} columns")
            if self.verbose:
                self._log(f"All columns: {list(self.raw_data.columns)}")

            return self.raw_data

        except Exception as e:
            self._log(f"Error loading data: {e}", 'error')
            self._log(f"Make sure the Excel file has the expected structure:", 'error')
            self._log(f"- Rows 1-5: Metadata/title information", 'error')
            self._log(f"- Row 6: Column headers (including Latitude, Longitude)", 'error')
            self._log(f"- Row 7+: Data rows", 'error')
            raise

    def _find_column_case_insensitive(self, target_column: str) -> Optional[str]:
        """Find column with case-insensitive exact match."""
        target_lower = target_column.lower().strip()
        for col in self.raw_data.columns:
            if col.lower().strip() == target_lower:
                return col
        return None

    def standardize_columns(self) -> pd.DataFrame:
        """Standardize column names using only exact case-insensitive matches."""
        if self.raw_data is None:
            self.load_data()

        self._log("Starting simplified column standardization process")
        if self.verbose:
            self._log(f"Original columns: {list(self.raw_data.columns)}")

        # Create mapping for coordinate columns only - exact matches
        column_mapping = {}

        # Find LONGITUDE column (exact case-insensitive match)
        longitude_col = self._find_column_case_insensitive('LONGITUDE')
        if longitude_col:
            column_mapping[longitude_col] = 'Longitude'
            self._log(f"Exact match: '{longitude_col}' -> 'Longitude'")

        # Find LATITUDE column (exact case-insensitive match)
        latitude_col = self._find_column_case_insensitive('LATITUDE')
        if latitude_col:
            column_mapping[latitude_col] = 'Latitude'
            self._log(f"Exact match: '{latitude_col}' -> 'Latitude'")

        # Apply only coordinate column mapping - leave all other columns unchanged
        if column_mapping:
            self.raw_data = self.raw_data.rename(columns=column_mapping)
            self._log(f"Applied {len(column_mapping)} coordinate column mappings")
        else:
            self._log("No coordinate columns found for standardization", 'warning')

        # Log what coordinate columns we have
        coord_columns_found = []
        if 'Longitude' in self.raw_data.columns:
            coord_columns_found.append('Longitude')
        if 'Latitude' in self.raw_data.columns:
            coord_columns_found.append('Latitude')

        self._log(f"Coordinate columns available: {coord_columns_found}")
        if self.verbose:
            self._log(f"Total columns (unchanged except coordinates): {len(self.raw_data.columns)}")

        # Lowercase all headers and replace whitespaces with underscores
        all_columns = self.raw_data.columns
        lowered_columns = ['_'.join(col.strip().lower().split(' ')) for col in all_columns]
        self.raw_data.columns = lowered_columns

        return self.raw_data

    def clean_coordinates(self) -> pd.DataFrame:
        """Clean and validate coordinate data."""
        if self.raw_data is None:
            self.standardize_columns()

        self._log("Starting coordinate column cleaning and detection")

        # First, check for and handle duplicate column names
        self._handle_duplicate_columns()

        # Ensure we have coordinate columns
        coord_cols = ['longitude', 'latitude']
        missing_coords = [col for col in coord_cols if col not in self.raw_data.columns]

        if missing_coords:
            self._log(f"Missing standardized coordinate columns: {missing_coords}", 'warning')
            if self.verbose:
                self._log(f"Available columns: {list(self.raw_data.columns)}")

            # Try to find alternative column names with more detailed matching
            longitude_candidates = []
            latitude_candidates = []

            for col in self.raw_data.columns:
                col_lower = col.lower().strip()
                if any(term in col_lower for term in ['longitude', 'long', 'lng']):
                    longitude_candidates.append(col)
                elif any(term in col_lower for term in ['latitude', 'lat']):
                    latitude_candidates.append(col)

            if self.verbose:
                self._log(f"Found longitude candidates: {longitude_candidates}")
                self._log(f"Found latitude candidates: {latitude_candidates}")

            # Use the first candidate found for each coordinate type
            if longitude_candidates and 'longitude' not in self.raw_data.columns:
                selected_lon_col = longitude_candidates[0]
                self.raw_data['longitude'] = self._safe_column_select(selected_lon_col)
                self._log(f"Using '{selected_lon_col}' as Longitude column")

            if latitude_candidates and 'latitude' not in self.raw_data.columns:
                selected_lat_col = latitude_candidates[0]
                self.raw_data['latitude'] = self._safe_column_select(selected_lat_col)
                self._log(f"Using '{selected_lat_col}' as Latitude column")

        # Final check for coordinate columns
        final_missing_coords = [col for col in coord_cols if col not in self.raw_data.columns]
        if final_missing_coords:
            self._log(f"Could not find coordinate columns: {final_missing_coords}", 'error')
            self._log(f"Available columns after search: {list(self.raw_data.columns)}", 'error')
            self._log("Please verify the Excel file structure and column names", 'error')
        else:
            self._log("Successfully identified coordinate columns: Longitude, Latitude")

        # Convert coordinates to numeric, coercing errors to NaN
        for coord in ['longitude', 'latitude']:
            if coord in self.raw_data.columns:
                try:
                    # Use safe column selection to handle potential duplicates
                    coord_series = self._safe_column_select(coord)

                    # Log column selection details
                    if self.verbose:
                        self._log(f"Processing {coord} column: type={type(coord_series)}, shape={getattr(coord_series, 'shape', 'N/A')}")

                    if coord_series is None:
                        self._log(f"Failed to select {coord} column safely", 'error')
                        continue

                    original_dtype = coord_series.dtype
                    non_null_before = coord_series.notna().sum()

                    # Convert to numeric
                    numeric_series = pd.to_numeric(coord_series, errors='coerce')
                    self.raw_data[coord] = numeric_series

                    non_null_after = numeric_series.notna().sum()
                    conversion_errors = non_null_before - non_null_after

                    self._log(f"{coord}: converted from {original_dtype} to numeric")
                    if conversion_errors > 0:
                        self._log(f"{coord}: {conversion_errors} values could not be converted to numeric (set to NaN)", 'warning')
                    if self.verbose:
                        self._log(f"{coord}: {non_null_after} valid numeric values out of {len(self.raw_data)} total records")

                except Exception as e:
                    self._log(f"Error processing {coord} column: {e}", 'error')
                    if self.verbose:
                        self._log(f"Column details: {coord} in columns: {coord in self.raw_data.columns}", 'error')
                    continue

        return self.raw_data

    def _handle_duplicate_columns(self) -> None:
        """Handle duplicate column names by keeping only the first occurrence."""
        if self.raw_data is None:
            return

        original_columns = list(self.raw_data.columns)
        duplicate_columns = []

        # Find duplicate column names
        seen_columns = set()
        for col in original_columns:
            if col in seen_columns:
                duplicate_columns.append(col)
            else:
                seen_columns.add(col)

        if duplicate_columns:
            self._log(f"Found duplicate column names: {duplicate_columns}", 'warning')

            # Remove duplicate columns by keeping only the first occurrence
            # This is done by selecting columns by position
            unique_columns = []
            seen_names = set()

            for i, col in enumerate(original_columns):
                if col not in seen_names:
                    unique_columns.append(i)
                    seen_names.add(col)
                else:
                    if self.verbose:
                        self._log(f"Removing duplicate column '{col}' at position {i}")

            # Select only unique columns
            self.raw_data = self.raw_data.iloc[:, unique_columns]
            if self.verbose:
                self._log(f"After removing duplicates: {len(self.raw_data.columns)} columns remaining")
        else:
            if self.verbose:
                self._log("No duplicate column names found")

    def _safe_column_select(self, column_name: str) -> Optional[pd.Series]:
        """Safely select a column, handling cases where it might return DataFrame."""
        if self.raw_data is None:
            self._log("No data available for column selection", 'error')
            return None

        if column_name not in self.raw_data.columns:
            self._log(f"Column '{column_name}' not found in data", 'error')
            return None

        try:
            # Get the column
            column_data = self.raw_data[column_name]

            # Log what we got (debug level)
            if self.verbose:
                self._log(f"Column '{column_name}' selection: type={type(column_data)}", 'debug')

            # If it's a DataFrame (due to duplicate columns), take the first column
            if isinstance(column_data, pd.DataFrame):
                self._log(f"Column '{column_name}' returned DataFrame (likely duplicates), taking first occurrence", 'warning')
                if column_data.shape[1] > 0:
                    return column_data.iloc[:, 0]  # Take first column
                else:
                    self._log(f"Empty DataFrame returned for column '{column_name}'", 'error')
                    return None

            # If it's already a Series, return it
            elif isinstance(column_data, pd.Series):
                return column_data

            else:
                self._log(f"Unexpected type returned for column '{column_name}': {type(column_data)}", 'error')
                return None

        except Exception as e:
            self._log(f"Error selecting column '{column_name}': {e}", 'error')
            return None

    def validate_coordinates(self) -> Dict[str, Any]:
        """Validate coordinate values and detect common issues."""
        if self.raw_data is None:
            self.clean_coordinates()

        validation_results = {
            'total_records': len(self.raw_data),
            'missing_longitude': 0,
            'missing_latitude': 0,
            'missing_both': 0,
            'out_of_bounds': 0,
            'potentially_switched': 0,
            'valid_coordinates': 0,
            'issues': []
        }

        # Initialize quality flag columns (always create them)
        self.raw_data['coord_valid'] = False
        self.raw_data['coord_missing'] = True  # Default to True, will be updated below
        self.raw_data['coord_out_of_bounds'] = False
        self.raw_data['coord_potentially_switched'] = False

        if 'longitude' not in self.raw_data.columns or 'latitude' not in self.raw_data.columns:
            validation_results['issues'].append("Missing coordinate columns")
            # Keep all records marked as having missing coordinates
            return validation_results

        # Count missing values
        missing_lon = self.raw_data['longitude'].isna()
        missing_lat = self.raw_data['latitude'].isna()

        validation_results['missing_longitude'] = missing_lon.sum()
        validation_results['missing_latitude'] = missing_lat.sum()
        validation_results['missing_both'] = (missing_lon & missing_lat).sum()

        # Update missing coordinate flags
        self.raw_data['coord_missing'] = missing_lon | missing_lat

        # Check coordinate bounds for non-missing values
        valid_coords_mask = ~missing_lon & ~missing_lat
        valid_coords = self.raw_data[valid_coords_mask]

        if len(valid_coords) > 0:
            # Check Philippine bounds
            lon_in_bounds = (valid_coords['longitude'] >= self.PH_LONGITUDE_MIN) & \
                           (valid_coords['longitude'] <= self.PH_LONGITUDE_MAX)
            lat_in_bounds = (valid_coords['latitude'] >= self.PH_LATITUDE_MIN) & \
                           (valid_coords['latitude'] <= self.PH_LATITUDE_MAX)

            both_in_bounds = lon_in_bounds & lat_in_bounds
            validation_results['valid_coordinates'] = both_in_bounds.sum()
            validation_results['out_of_bounds'] = len(valid_coords) - both_in_bounds.sum()

            # Check for potentially switched coordinates
            # If longitude is in latitude range and latitude is in longitude range
            potential_switch = (
                (valid_coords['longitude'] >= self.PH_LATITUDE_MIN) &
                (valid_coords['longitude'] <= self.PH_LATITUDE_MAX) &
                (valid_coords['latitude'] >= self.PH_LONGITUDE_MIN) &
                (valid_coords['latitude'] <= self.PH_LONGITUDE_MAX) &
                ~both_in_bounds
            )
            validation_results['potentially_switched'] = potential_switch.sum()

            # Set flags for valid coordinates
            self.raw_data.loc[valid_coords_mask, 'coord_valid'] = both_in_bounds.values
            self.raw_data.loc[valid_coords_mask, 'coord_out_of_bounds'] = ~both_in_bounds.values
            self.raw_data.loc[valid_coords_mask, 'coord_potentially_switched'] = potential_switch.values

        self.quality_report = validation_results
        if self.verbose:
            self._log(f"Coordinate validation completed: {validation_results}")
        else:
            # Show only summary when not verbose
            self._log(f"Coordinate validation: {validation_results.get('valid_coordinates', 0)} valid out of {validation_results.get('total_records', 0)} records")

        return validation_results

    def fix_switched_coordinates(self, auto_fix: bool = False) -> pd.DataFrame:
        """Detect and optionally fix switched longitude/latitude coordinates."""
        # Ensure validation has been run and quality columns exist
        if (self.raw_data is None or
            'coord_potentially_switched' not in self.raw_data.columns or
            self.raw_data.empty):
            self.validate_coordinates()

        # Safety check: ensure the column exists after validation
        if 'coord_potentially_switched' not in self.raw_data.columns:
            self._log("coord_potentially_switched column not found even after validation. Skipping switched coordinate fix.", 'warning')
            return self.raw_data

        switched_mask = self.raw_data['coord_potentially_switched'] == True
        switched_count = switched_mask.sum()

        if switched_count > 0:
            self._log(f"Found {switched_count} potentially switched coordinate pairs")

            if auto_fix:
                # Verify we have the necessary coordinate columns
                if 'longitude' not in self.raw_data.columns or 'latitude' not in self.raw_data.columns:
                    self._log("Cannot fix switched coordinates: Longitude or Latitude column missing", 'error')
                    return self.raw_data

                # Swap longitude and latitude for switched coordinates
                self._log(f"Auto-fixing {switched_count} switched coordinate pairs")

                # Store original values
                self.raw_data['original_longitude'] = self.raw_data['longitude'].copy()
                self.raw_data['original_latitude'] = self.raw_data['latitude'].copy()

                # Swap coordinates
                temp_lon = self.raw_data.loc[switched_mask, 'longitude'].copy()
                self.raw_data.loc[switched_mask, 'longitude'] = self.raw_data.loc[switched_mask, 'latitude']
                self.raw_data.loc[switched_mask, 'latitude'] = temp_lon

                # Mark as fixed
                self.raw_data['coord_fixed'] = switched_mask

                # Re-validate coordinates
                self.validate_coordinates()

                if self.verbose:
                    self._log(f"Fixed coordinates. New validation results: {self.quality_report}")
                else:
                    self._log(f"Fixed {switched_count} switched coordinate pairs")
            else:
                # Just flag them for manual review
                self._log("Switched coordinates detected but not auto-fixed. Set auto_fix=True to fix automatically.")
        else:
            if self.verbose:
                self._log("No potentially switched coordinates found.")

        return self.raw_data

    def standardize_school_ids(self) -> pd.DataFrame:
        """Standardize school ID format using LIS SCHOOL ID column."""
        if self.raw_data is None:
            self.fix_switched_coordinates()

        # Find LIS SCHOOL ID column (case-insensitive exact match)
        lis_school_id_col = self._find_column_case_insensitive('lis_school_id')

        if lis_school_id_col:
            self._log(f"Found LIS SCHOOL ID column: '{lis_school_id_col}'")

            # Convert to string and create processed version
            original_series = self.raw_data[lis_school_id_col].astype(str)

            # Create new processed column while keeping original unchanged
            self.raw_data['school_id_processed'] = original_series.str.strip()

            # Count valid IDs
            valid_ids = self.raw_data['school_id_processed'].notna().sum()

            self._log(f"Created school_id_processed column from '{lis_school_id_col}'")
            if self.verbose:
                self._log(f"Processed {valid_ids} school IDs")
        else:
            self._log("No 'lis_school_id' column found", 'warning')

        return self.raw_data

    def process(self, auto_fix_coordinates: bool = True, verbose: Optional[bool] = None) -> pd.DataFrame:
        """Main processing pipeline with optional verbose override."""
        # Override verbose setting if specified
        original_verbose = self.verbose
        if verbose is not None:
            self.verbose = verbose
            # Update logging level if needed
            if not verbose:
                logging.getLogger(__name__).setLevel(logging.WARNING)
            else:
                logging.getLogger(__name__).setLevel(logging.INFO)

        try:
            self._log("="*60)
            self._log("STARTING SCHOOL COORDINATES PREPROCESSING")
            self._log("="*60)
            self._log(f"Excel file path: {self.excel_path}")
            if self.verbose:
                self._log("Expected Excel structure:")
                self._log("  - Rows 1-5: Metadata/title information (will be skipped)")
                self._log("  - Row 6: Column headers (Latitude, Longitude, etc.)")
                self._log("  - Row 7+: Actual school data")

            # Step 1: Load and standardize
            self._log("\n--- STEP 1: Loading data with correct header structure ---")
            self.load_data()

            self._log("\n--- STEP 2: Standardizing column names ---")
            self.standardize_columns()

            # Step 2: Clean coordinates
            self._log("\n--- STEP 3: Cleaning and detecting coordinate columns ---")
            self.clean_coordinates()

            # Step 3: Validate coordinates
            self._log("\n--- STEP 4: Validating coordinate data quality ---")
            self.validate_coordinates()

            # Step 4: Fix switched coordinates if requested
            self._log(f"\n--- STEP 5: Checking for switched coordinates (auto_fix={auto_fix_coordinates}) ---")
            self.fix_switched_coordinates(auto_fix=auto_fix_coordinates)

            # Step 5: Standardize school IDs
            self._log("\n--- STEP 6: Standardizing school IDs ---")
            self.standardize_school_ids()

            # Create processed data copy
            self.processed_data = self.raw_data.copy()

            self._log("\n" + "="*60)
            self._log("SCHOOL COORDINATES PREPROCESSING COMPLETED")
            self._log("="*60)
            self._log(f"Total records processed: {len(self.processed_data)}")
            if self.verbose:
                self._log(f"Columns in final dataset: {len(self.processed_data.columns)}")

            return self.processed_data

        finally:
            # Restore original verbose setting
            self.verbose = original_verbose
            if not original_verbose:
                logging.getLogger(__name__).setLevel(logging.WARNING)
            else:
                logging.getLogger(__name__).setLevel(logging.INFO)

    def get_quality_report(self) -> Dict[str, Any]:
        """Get comprehensive data quality report."""
        if not self.quality_report:
            self.validate_coordinates()

        # Add additional statistics
        report = self.quality_report.copy()

        if self.processed_data is not None:
            # Ensure quality flag columns exist before using them
            required_columns = ['coord_valid', 'coord_missing']
            missing_columns = [col for col in required_columns if col not in self.processed_data.columns]

            if missing_columns:
                self._log(f"Quality flag columns missing from processed data: {missing_columns}. Running validation first.", 'warning')
                if self.raw_data is not None:
                    self.validate_coordinates()
                    self.processed_data = self.raw_data.copy()

            # Only add statistics if we have the necessary columns
            if all(col in self.processed_data.columns for col in required_columns):
                try:
                    report.update({
                        'total_schools': len(self.processed_data),
                        'schools_with_valid_coords': (self.processed_data['coord_valid'] == True).sum(),
                        'schools_with_missing_coords': (self.processed_data['coord_missing'] == True).sum(),
                        'coordinate_completeness_rate': (self.processed_data['coord_valid'] == True).sum() / len(self.processed_data) * 100,
                    })
                except KeyError as e:
                    self._log(f"KeyError when calculating quality statistics: {e}", 'error')

            # Regional distribution
            if 'region' in self.processed_data.columns:
                report['regional_distribution'] = self.processed_data['region'].value_counts().to_dict()

            # Division distribution
            if 'division' in self.processed_data.columns:
                report['division_distribution'] = self.processed_data['division'].value_counts().head(10).to_dict()

        return report

    def get_problematic_records(self, issue_type: str = 'all') -> pd.DataFrame:
        """Get records with specific coordinate issues."""
        if self.processed_data is None:
            self._log("No processed data available. Run process() first.", 'warning')
            return pd.DataFrame()

        # Ensure quality flag columns exist
        required_columns = ['coord_missing', 'coord_out_of_bounds', 'coord_potentially_switched', 'coord_valid']
        missing_columns = [col for col in required_columns if col not in self.processed_data.columns]

        if missing_columns:
            self._log(f"Quality flag columns missing: {missing_columns}. Running validation first.", 'warning')
            # Re-run validation to ensure columns exist
            if self.raw_data is not None:
                self.validate_coordinates()
                self.processed_data = self.raw_data.copy()
            else:
                self._log("Cannot create quality flags: no raw data available", 'error')
                return pd.DataFrame()

        # Double-check that columns exist after validation
        missing_columns = [col for col in required_columns if col not in self.processed_data.columns]
        if missing_columns:
            self._log(f"Quality flag columns still missing after validation: {missing_columns}", 'error')
            return pd.DataFrame()

        try:
            if issue_type == 'missing':
                mask = self.processed_data['coord_missing'] == True
            elif issue_type == 'out_of_bounds':
                mask = self.processed_data['coord_out_of_bounds'] == True
            elif issue_type == 'switched':
                mask = self.processed_data['coord_potentially_switched'] == True
            elif issue_type == 'invalid':
                mask = (self.processed_data['coord_valid'] != True) | (self.processed_data['coord_missing'] == True)
            else:  # all issues
                mask = (self.processed_data['coord_valid'] != True) | (self.processed_data['coord_missing'] == True)

            return self.processed_data[mask]
        except KeyError as e:
            self._log(f"KeyError when accessing quality flag columns: {e}", 'error')
            return pd.DataFrame()

    def get_valid_coordinates(self) -> pd.DataFrame:
        """Get only records with valid coordinates."""
        if self.processed_data is None:
            self._log("No processed data available. Run process() first.", 'warning')
            return pd.DataFrame()

        # Ensure quality flag columns exist
        if 'coord_valid' not in self.processed_data.columns:
            self._log("coord_valid column missing. Running validation first.", 'warning')
            # Re-run validation to ensure columns exist
            if self.raw_data is not None:
                self.validate_coordinates()
                self.processed_data = self.raw_data.copy()
            else:
                self._log("Cannot create quality flags: no raw data available", 'error')
                return pd.DataFrame()

        # Double-check that column exists after validation
        if 'coord_valid' not in self.processed_data.columns:
            self._log("coord_valid column still missing after validation", 'error')
            return pd.DataFrame()

        try:
            return self.processed_data[self.processed_data['coord_valid'] == True]
        except KeyError as e:
            self._log(f"KeyError when accessing coord_valid column: {e}", 'error')
            return pd.DataFrame()

    def export_processed(self, output_path: str = 'output/public_school_coordinates_processed.csv'):
        """Export processed public school coordinate data to CSV."""
        if self.processed_data is None:
            self._log("No processed data available. Run process() first.", 'warning')
            return

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Export with quality flags
        self.processed_data.to_csv(output_path, index=False)
        self._log(f"Exported processed coordinate data to {output_path}")

    def export_quality_report(self, output_path: str = 'output/public_school_coordinate_quality_report.json'):
        """Export quality report to JSON."""
        if not self.quality_report:
            self.get_quality_report()

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        import json
        with open(output_path, 'w') as f:
            json.dump(self.quality_report, f, indent=2, default=str)
        self._log(f"Exported quality report to {output_path}")

    def export_problematic_records(self, output_path: str = 'output/coordinate_issues.csv'):
        """Export records with coordinate issues for manual review."""
        problematic = self.get_problematic_records()

        if len(problematic) == 0:
            self._log("No problematic records found.")
            return

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        problematic.to_csv(output_path, index=False)
        self._log(f"Exported {len(problematic)} problematic records to {output_path}")

    def merge_ready_data(self) -> pd.DataFrame:
        """Prepare data for merging with enrollment data."""
        if self.processed_data is None:
            self._log("No processed data available. Run process() first.", 'warning')
            return pd.DataFrame()

        # Ensure quality flag columns exist before using them
        quality_cols = ['coord_valid', 'coord_missing', 'coord_out_of_bounds', 'coord_potentially_switched']
        missing_quality_cols = [col for col in quality_cols if col not in self.processed_data.columns]

        if missing_quality_cols:
            self._log(f"Quality flag columns missing: {missing_quality_cols}. Running validation first.", 'warning')
            if self.raw_data is not None:
                self.validate_coordinates()
                self.processed_data = self.raw_data.copy()
            else:
                self._log("Cannot create quality flags: no raw data available", 'error')

        # Select key columns for merging
        merge_columns = []

        # Always include these if available
        essential_cols = ['lis_school_id', 'school_id_processed', 'school_name', 'longitude', 'latitude']
        for col in essential_cols:
            if col in self.processed_data.columns:
                merge_columns.append(col)

        # Include location information if available
        location_cols = ['region', 'division', 'district', 'province', 'municipality', 'barangay']
        for col in location_cols:
            if col in self.processed_data.columns:
                merge_columns.append(col)

        # Include quality flags (only those that actually exist)
        for col in quality_cols:
            if col in self.processed_data.columns:
                merge_columns.append(col)

        if not merge_columns:
            self._log("No valid columns found for merge-ready dataset", 'error')
            return pd.DataFrame()

        try:
            merge_ready = self.processed_data[merge_columns].copy()
            merge_ready.set_index('school_id_processed', inplace=True)
            
            self._log(f"Prepared merge-ready dataset with {len(merge_ready)} records and {len(merge_columns)} columns")
            return merge_ready
        except KeyError as e:
            self._log(f"KeyError when creating merge-ready dataset: {e}", 'error')
            if self.verbose:
                self._log(f"Available columns: {list(self.processed_data.columns)}", 'error')
                self._log(f"Requested columns: {merge_columns}", 'error')
            return pd.DataFrame()


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = PublicSchoolsProcessor()

    # Process data with automatic coordinate fixing
    processed_data = processor.process(auto_fix_coordinates=True)

    # Get quality report
    quality_report = processor.get_quality_report()
    print("Data Quality Report:")
    for key, value in quality_report.items():
        if isinstance(value, dict):
            print(f"  {key}: {len(value)} items")
        else:
            print(f"  {key}: {value}")

    # Export processed data
    processor.export_processed()
    processor.export_quality_report()
    processor.export_problematic_records()

    # Show sample of valid coordinates
    valid_coords = processor.get_valid_coordinates()
    print(f"\nSample of valid coordinates ({len(valid_coords)} total):")
    if len(valid_coords) > 0:
        print(valid_coords[['school_name', 'longitude', 'latitude', 'region']].head())

    # Prepare merge-ready data
    merge_data = processor.merge_ready_data()
    print(f"\nMerge-ready data prepared with {len(merge_data)} records")