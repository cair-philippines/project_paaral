"""
Private Schools Processor Module

A fast, combined module that integrates reading and processing capabilities
for private school data Excel files. Optimized for speed while maintaining
the user's proven preprocessing logic.

Features:
- Reads all Excel files from directory
- Processes data using user's efficient approach from notebook section 1.3.1
- Minimal overhead and logging when verbose=False
- Vectorized operations for maximum performance
- Memory-efficient processing

Author: Data Preprocessing Specialist
Created: 2025-09-26
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import logging
import warnings
import re

# Suppress warnings for cleaner output when verbose=False
warnings.filterwarnings('ignore')


class PrivateSchoolsProcessor:
    """
    Fast, combined processor for private school data with optimized performance.

    Combines reading and processing capabilities in a single efficient module.
    Uses the user's proven preprocessing approach with minimal overhead.
    """

    def __init__(self, directory_path: Optional[str] = None, verbose: bool = True):
        """
        Initialize the PrivateSchoolsProcessor.

        Args:
            directory_path (str): Path to directory containing Excel files
            verbose (bool): Whether to enable logging and progress information
        """
        self.directory_path = Path(directory_path) if directory_path else Path('data/private/raw_validation_sheets')
        self.verbose = verbose
        self.raw_data = {}
        self.processed_data = None
        self.failed_sheets = []
        self.successful_sheets = []

        # Configure logging for performance
        if verbose:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            logging.basicConfig(level=logging.CRITICAL)
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.CRITICAL)

        # Validate directory exists
        if not self.directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        if verbose:
            self.logger.info(f"Initialized PrivateSchoolsProcessor for directory: {directory_path}")

    def _log(self, message: str) -> None:
        """Minimal logging for performance."""
        if self.verbose:
            self.logger.info(message)

    def read_all_files(self) -> Dict:
        """
        Read all Excel files from directory with minimal overhead.

        Returns:
            Dict: Raw data organized by filename and sheet name
        """
        self._log("Reading Excel files...")

        # Find Excel files efficiently
        excel_files = [*self.directory_path.glob("*.xlsx"), *self.directory_path.glob("*.xls")]

        if not excel_files:
            if self.verbose:
                self.logger.warning("No Excel files found")
            return {}

        # Process files with minimal logging
        for file_path in excel_files:
            filename = file_path.stem

            try:
                excel_file = pd.ExcelFile(file_path)
                self.raw_data[filename] = {}

                # Read all sheets efficiently
                for sheet_name in excel_file.sheet_names:
                    try:
                        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
                        self.raw_data[filename][sheet_name] = df
                    except Exception as e:
                        self.failed_sheets.append(f"{filename} - {sheet_name}: {str(e)}")

                excel_file.close()

            except Exception as e:
                if self.verbose:
                    self.logger.error(f"Failed to read {filename}: {e}")

        self._log(f"Successfully read {len(self.raw_data)} files")
        return self.raw_data

    def get_raw_data(self) -> Dict:
        """Return raw data dictionary."""
        return self.raw_data

    def process(self) -> pd.DataFrame:
        """
        Main processing pipeline using user's proven efficient approach.

        This method implements the exact logic from the user's notebook section 1.3.1
        with minimal modifications for maximum speed.

        Returns:
            pd.DataFrame: Processed and concatenated DataFrame
        """
        self._log("Starting data processing...")

        # Step 1: Load raw data if not already loaded
        if not self.raw_data:
            self.read_all_files()

        if not self.raw_data:
            return pd.DataFrame()

        # Step 2: Process all sheets using user's efficient approach
        all_processed_dfs = []

        for filename, divisions_dict in self.raw_data.items():
            file_processed_dfs = []

            for division_name, df_div in divisions_dict.items():
                try:
                    # USER'S EXACT APPROACH - Find "Region" header in first 3 columns
                    max_search_cols = min(3, df_div.shape[1])
                    if max_search_cols == 0:
                        continue

                    # Find region header row efficiently
                    region_row_mask = (
                        df_div.iloc[:, :max_search_cols]
                        .astype(str)
                        .apply(lambda s: s.str.strip().str.fullmatch(r'(?i)region'))
                        .any(axis=1)
                    )

                    matches = region_row_mask[region_row_mask].index.tolist()
                    if not matches:
                        self.failed_sheets.append(f"{filename} - {division_name}: No Region header")
                        continue

                    region_index = matches[0]

                    # USER'S EXACT HEADER PROCESSING
                    headers = df_div.iloc[region_index, :].values
                    headers_processed = ['_'.join(str(col).strip().lower().split(' ')) for col in headers]

                    # Create processed DataFrame efficiently
                    df_processed = df_div.copy()
                    df_processed.columns = headers_processed

                    # Remove duplicate columns (retain first)
                    df_processed = df_processed.loc[:, ~df_processed.columns.duplicated()]

                    # Drop rows before region header
                    df_processed = df_processed.iloc[region_index+1:, :].copy()

                    # Handle beis_school_id if exists
                    if 'beis_school_id' in df_processed.columns:
                        df_processed['beis_school_id'] = df_processed['beis_school_id'].astype('string')

                    # Add metadata
                    df_processed['excel_filename'] = filename
                    df_processed['sheet_name'] = division_name

                    # Remove completely empty rows
                    df_processed = df_processed.dropna(how='all')

                    if len(df_processed) > 0:
                        file_processed_dfs.append(df_processed)
                        self.successful_sheets.append(f"{filename} - {division_name}")

                except Exception as e:
                    self.failed_sheets.append(f"{filename} - {division_name}: {str(e)}")
                    continue

            # Concatenate processed sheets from this file
            if file_processed_dfs:
                file_combined = pd.concat(file_processed_dfs, ignore_index=True)
                all_processed_dfs.append(file_combined)

        # Step 3: Create final dataset using user's approach
        if not all_processed_dfs:
            self._log("No data to process")
            return pd.DataFrame()

        # USER'S EXACT CONCATENATION APPROACH
        final_df = pd.concat(all_processed_dfs, ignore_index=True)

        # Apply column limit as in original code (up to "sheet_name")
        if 'sheet_name' in final_df.columns:
            sheet_name_idx = final_df.columns.get_loc('sheet_name')
            final_df = final_df.iloc[:, :sheet_name_idx+1]

        self._log(f"Processing complete: {len(final_df)} rows, {len(final_df.columns)} columns")

        self.processed_data = final_df

        return self.processed_data

    def replace_unclean_region_values(self) -> pd.DataFrame:
        tmp_df = self.processed_data.copy()
        tmp_df = tmp_df[tmp_df['region'].notna()]
        
        mask = tmp_df['region'] == 'hud'
        tmp_df.loc[mask, 'region'] = 'NCR'
        
        mask = tmp_df['region'] == 'Corrected'
        tmp_df = tmp_df.loc[~mask]
        
        tmp_df['region'] = tmp_df['region'].replace(
            {
                'REGION 10 - Misamis Occidental':'Region X',
                'REGION 10 - Ozamis City':'Region X',
                ' ':'Region IV-A',
                'REGION 4A':'Region IV-A',
                'REGION 4A ':'Region IV-A',
                'REGION 4A - BACOOR CITY':'Region IV-A',
                'IV-A':'Region IV-A',
                'REGION 4A - BINAN CITY':'Region IV-A',
                'Region IV-A ':'Region IV-A',
                8.23803:'Region IX'
            }
        )
        tmp_df['region'] = tmp_df['region'].astype('string')

        self.processed_data = tmp_df

        return self.processed_data
    
    def get_processed_data(self) -> pd.DataFrame:
        """Return processed DataFrame."""
        if self.processed_data is None:
            if self.verbose:
                self.logger.warning("No processed data. Run process() first.")
            return pd.DataFrame()
        return self.processed_data

    def export_processed(self, output_path: str = 'output/private_schools_processed.csv') -> None:
        """
        Export processed data to CSV.

        Args:
            output_path (str): Path for output CSV file
        """
        if self.processed_data is None:
            if self.verbose:
                self.logger.warning("No processed data to export")
            return

        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Export efficiently
        self.processed_data.to_csv(output_path, index=False)
        self._log(f"Exported to {output_path}")

    def get_summary(self) -> Dict[str, Any]:
        """Get processing summary."""
        if not hasattr(self, 'raw_data'):
            return {}

        total_files = len(self.raw_data)
        total_sheets = sum(len(sheets) for sheets in self.raw_data.values())
        successful_count = len(self.successful_sheets)
        failed_count = len(self.failed_sheets)

        return {
            'total_files_processed': total_files,
            'total_sheets_found': total_sheets,
            'successful_sheets': successful_count,
            'failed_sheets': failed_count,
            'success_rate': (successful_count / total_sheets * 100) if total_sheets > 0 else 0,
            'final_dataset_rows': len(self.processed_data) if self.processed_data is not None else 0,
            'final_dataset_columns': len(self.processed_data.columns) if self.processed_data is not None else 0,
            'successful_sheet_details': self.successful_sheets.copy(),
            'failed_sheet_details': self.failed_sheets.copy()
        }

    # Additional methods for compatibility and future expansion
    def get_file_summary(self) -> Dict:
        """Get summary of loaded files and sheets."""
        if not self.raw_data:
            return {'total_files': 0, 'file_details': {}}

        summary = {
            'total_files': len(self.raw_data),
            'file_details': {}
        }

        for filename, sheets in self.raw_data.items():
            summary['file_details'][filename] = {
                'sheet_count': len(sheets),
                'sheet_names': list(sheets.keys()),
                'sheet_shapes': {name: df.shape for name, df in sheets.items()}
            }

        return summary

    def get_sheet_data(self, filename: str, sheet_name: str) -> Optional[pd.DataFrame]:
        """Get raw data from specific sheet."""
        if filename in self.raw_data and sheet_name in self.raw_data[filename]:
            return self.raw_data[filename][sheet_name]
        return None

    def list_files(self) -> List[str]:
        """Get list of loaded filenames."""
        return list(self.raw_data.keys())

    def list_sheets(self, filename: str) -> List[str]:
        """Get list of sheets for specific file."""
        return list(self.raw_data.get(filename, {}).keys())

    def get_failed_sheets(self) -> List[str]:
        """Get list of failed sheet descriptions."""
        return self.failed_sheets.copy()

    def get_successful_sheets(self) -> List[str]:
        """Get list of successful sheet descriptions."""
        return self.successful_sheets.copy()

    def get_data_quality_summary(self) -> Dict[str, Any]:
        """Get basic data quality metrics."""
        if self.processed_data is None:
            return {}

        df = self.processed_data
        quality_summary = {
            'total_records': len(df),
            'duplicate_records': df.duplicated().sum(),
            'completely_empty_rows': df.isnull().all(axis=1).sum(),
        }

        # Check key columns
        if 'beis_school_id' in df.columns:
            quality_summary['beis_school_id_completeness'] = (
                df['beis_school_id'].notnull().sum() / len(df) * 100
            )

        if 'excel_filename' in df.columns:
            quality_summary['files_represented'] = df['excel_filename'].nunique()

        if 'sheet_name' in df.columns:
            quality_summary['sheets_represented'] = df['sheet_name'].nunique()

        return quality_summary

    def validate_coordinates(self) -> Dict[str, Any]:
        """
        Validate latitude and longitude coordinates in the processed data.

        Validates coordinates using Philippine geographic bounds and decimal degrees format.
        Creates a new 'coordinates_valid' column indicating valid coordinate pairs.

        Philippine Bounds:
        - Longitude: 116° to 127° East
        - Latitude: 4° to 21° North

        Returns:
            Dict[str, Any]: Validation summary statistics
        """
        if self.processed_data is None:
            self._log("No processed data available. Run process() first.")
            return {}

        df = self.processed_data
        self._log("Starting coordinate validation...")

        # Find latitude and longitude columns
        lat_col, lon_col = self._find_coordinate_columns(df)

        if not lat_col or not lon_col:
            self._log("No latitude/longitude columns found in data")
            df['coordinates_valid'] = False
            return {
                'latitude_column': lat_col,
                'longitude_column': lon_col,
                'total_records': len(df),
                'valid_coordinates': 0,
                'invalid_coordinates': len(df),
                'validation_rate': 0.0,
                'issues_found': ['No coordinate columns found']
            }

        self._log(f"Found coordinate columns: {lat_col}, {lon_col}")

        # Philippine coordinate bounds
        PHILIPPINES_LAT_MIN = 4.0
        PHILIPPINES_LAT_MAX = 21.0
        PHILIPPINES_LON_MIN = 116.0
        PHILIPPINES_LON_MAX = 127.0

        # Initialize validation arrays
        total_records = len(df)
        lat_valid = np.zeros(total_records, dtype=bool)
        lon_valid = np.zeros(total_records, dtype=bool)

        # Get coordinate series
        lat_series = df[lat_col]
        lon_series = df[lon_col]

        # Validate latitude coordinates
        lat_valid, lat_issues = self._validate_coordinate_column(
            lat_series, PHILIPPINES_LAT_MIN, PHILIPPINES_LAT_MAX, 'latitude'
        )

        # Validate longitude coordinates
        lon_valid, lon_issues = self._validate_coordinate_column(
            lon_series, PHILIPPINES_LON_MIN, PHILIPPINES_LON_MAX, 'longitude'
        )

        # Create coordinates_valid column (both lat and lon must be valid)
        coordinates_valid = lat_valid & lon_valid
        df['coordinates_valid'] = coordinates_valid

        # Calculate statistics
        valid_count = coordinates_valid.sum()
        invalid_count = total_records - valid_count
        validation_rate = (valid_count / total_records * 100) if total_records > 0 else 0

        # Combine issues
        all_issues = lat_issues + lon_issues

        # Create detailed summary
        validation_summary = {
            'latitude_column': lat_col,
            'longitude_column': lon_col,
            'total_records': total_records,
            'valid_coordinates': int(valid_count),
            'invalid_coordinates': int(invalid_count),
            'validation_rate': round(validation_rate, 2),
            'philippine_bounds_used': {
                'latitude': f"{PHILIPPINES_LAT_MIN}° to {PHILIPPINES_LAT_MAX}° North",
                'longitude': f"{PHILIPPINES_LON_MIN}° to {PHILIPPINES_LON_MAX}° East"
            },
            'issues_found': list(set(all_issues)) if all_issues else ['No major issues found'],
            'latitude_validation': {
                'valid_count': int(lat_valid.sum()),
                'invalid_count': int((~lat_valid).sum()),
                'validation_rate': round(lat_valid.sum() / total_records * 100, 2)
            },
            'longitude_validation': {
                'valid_count': int(lon_valid.sum()),
                'invalid_count': int((~lon_valid).sum()),
                'validation_rate': round(lon_valid.sum() / total_records * 100, 2)
            }
        }

        # Log results
        self._log(f"Coordinate validation complete:")
        self._log(f"  Valid coordinates: {valid_count:,} ({validation_rate:.1f}%)")
        self._log(f"  Invalid coordinates: {invalid_count:,}")

        if all_issues:
            self._log(f"  Issues identified: {', '.join(set(all_issues))}")

        return validation_summary

    def _find_coordinate_columns(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        """
        Find latitude and longitude columns in the DataFrame.

        Uses precise pattern matching to avoid false positives like 'legislative_district'
        being matched as a latitude column.

        Args:
            df (pd.DataFrame): DataFrame to search

        Returns:
            Tuple[Optional[str], Optional[str]]: (latitude_column, longitude_column)
        """
        self._log(f"Searching for coordinate columns among: {list(df.columns)}")

        columns_lower = [col.lower() for col in df.columns]

        # Exact matches first (highest priority)
        exact_lat_matches = ['latitude', 'lat', 'y', 'lat_decimal', 'latitude_decimal']
        exact_lon_matches = ['longitude', 'lon', 'lng', 'long', 'x', 'lon_decimal', 'longitude_decimal']

        # Check for exact matches first
        lat_col = None
        for exact_match in exact_lat_matches:
            if exact_match in columns_lower:
                original_idx = columns_lower.index(exact_match)
                lat_col = df.columns[original_idx]
                self._log(f"Found latitude column by exact match: '{lat_col}'")
                break

        lon_col = None
        for exact_match in exact_lon_matches:
            if exact_match in columns_lower:
                original_idx = columns_lower.index(exact_match)
                lon_col = df.columns[original_idx]
                self._log(f"Found longitude column by exact match: '{lon_col}'")
                break

        # If no exact matches, use more precise patterns
        if not lat_col:
            # More precise latitude patterns - avoid partial matches
            lat_patterns = [
                r'^latitude$',           # Exact match
                r'^lat$',               # Exact match
                r'^.*_lat$',            # Ends with _lat
                r'^lat_.*$',            # Starts with lat_
                r'^.*_latitude$',       # Ends with _latitude
                r'^latitude_.*$',       # Starts with latitude_
                r'^.*decimal.*lat.*$',  # Contains decimal and lat
                r'^.*dekimal.*lat.*$',  # Contains dekimal and lat (Filipino)
                r'^y_coordinate$',      # Y coordinate
                r'^coord_lat$',         # Coordinate latitude
                r'^lat_coord$'          # Latitude coordinate
            ]

            for pattern in lat_patterns:
                for i, col in enumerate(columns_lower):
                    if re.match(pattern, col, re.IGNORECASE):
                        # Additional validation: avoid columns that clearly aren't coordinates
                        excluded_words = ['district', 'legislative', 'electoral', 'political', 'admin', 'region_name']
                        if not any(excluded in col for excluded in excluded_words):
                            lat_col = df.columns[i]
                            self._log(f"Found latitude column by pattern '{pattern}': '{lat_col}'")
                            break
                if lat_col:
                    break

        if not lon_col:
            # More precise longitude patterns
            lon_patterns = [
                r'^longitude$',         # Exact match
                r'^lon$',              # Exact match
                r'^lng$',              # Exact match
                r'^long$',             # Exact match
                r'^.*_lon$',           # Ends with _lon
                r'^lon_.*$',           # Starts with lon_
                r'^.*_lng$',           # Ends with _lng
                r'^lng_.*$',           # Starts with lng_
                r'^.*_longitude$',     # Ends with _longitude
                r'^longitude_.*$',     # Starts with longitude_
                r'^.*decimal.*lon.*$', # Contains decimal and lon
                r'^.*dekimal.*lon.*$', # Contains dekimal and lon (Filipino)
                r'^x_coordinate$',     # X coordinate
                r'^coord_lon$',        # Coordinate longitude
                r'^lon_coord$'         # Longitude coordinate
            ]

            for pattern in lon_patterns:
                for i, col in enumerate(columns_lower):
                    if re.match(pattern, col, re.IGNORECASE):
                        # Additional validation: avoid columns that clearly aren't coordinates
                        excluded_words = ['district', 'legislative', 'electoral', 'political', 'admin', 'region_name']
                        if not any(excluded in col for excluded in excluded_words):
                            lon_col = df.columns[i]
                            self._log(f"Found longitude column by pattern '{pattern}': '{lon_col}'")
                            break
                if lon_col:
                    break

        # Final validation: check if the detected columns actually contain coordinate-like data
        if lat_col and lon_col:
            lat_validation = self._validate_column_as_coordinate(df[lat_col], 'latitude')
            lon_validation = self._validate_column_as_coordinate(df[lon_col], 'longitude')

            if not lat_validation['likely_coordinate']:
                self._log(f"Warning: Column '{lat_col}' doesn't appear to contain coordinate data")
                if not lat_validation['has_numeric_values']:
                    lat_col = None

            if not lon_validation['likely_coordinate']:
                self._log(f"Warning: Column '{lon_col}' doesn't appear to contain coordinate data")
                if not lon_validation['has_numeric_values']:
                    lon_col = None

        if lat_col:
            self._log(f"Final latitude column: '{lat_col}'")
        else:
            self._log("No suitable latitude column found")

        if lon_col:
            self._log(f"Final longitude column: '{lon_col}'")
        else:
            self._log("No suitable longitude column found")

        return lat_col, lon_col

    def _validate_column_as_coordinate(self, series: pd.Series, coord_type: str) -> Dict[str, Any]:
        """
        Validate that a column likely contains coordinate data.

        Args:
            series (pd.Series): Column to validate
            coord_type (str): Type of coordinate ('latitude' or 'longitude')

        Returns:
            Dict[str, Any]: Validation results
        """
        # Check if series has any non-null values
        non_null_values = series.dropna()
        if len(non_null_values) == 0:
            return {
                'likely_coordinate': False,
                'has_numeric_values': False,
                'reason': 'All values are null'
            }

        # Convert to string and check for coordinate-like patterns
        str_values = non_null_values.astype(str).str.strip()

        # Check for numeric values that could be coordinates
        numeric_values = pd.to_numeric(str_values, errors='coerce')
        numeric_count = numeric_values.notnull().sum()
        total_count = len(str_values)

        # Check for coordinate formats (DMS, DMM)
        dms_pattern = r'.*°.*[\'"].*[NSEW]?'
        dmm_pattern = r'.*°.*\.\d+.*[\'"].*[NSEW]?'
        coordinate_format_count = (
            str_values.str.match(dms_pattern, na=False).sum() +
            str_values.str.match(dmm_pattern, na=False).sum()
        )

        # Calculate percentage of values that look like coordinates
        coordinate_like_count = numeric_count + coordinate_format_count
        coordinate_percentage = coordinate_like_count / total_count if total_count > 0 else 0

        # Determine if this is likely a coordinate column
        has_numeric_values = numeric_count > 0
        likely_coordinate = coordinate_percentage >= 0.5  # At least 50% coordinate-like values

        # Additional validation for numeric values
        if has_numeric_values:
            valid_numeric = numeric_values.dropna()
            if len(valid_numeric) > 0:
                # Check if values are in reasonable coordinate ranges
                if coord_type == 'latitude':
                    # Philippines latitude range: 4° to 21° N
                    reasonable_range = (valid_numeric >= -90) & (valid_numeric <= 90)
                    in_philippines = (valid_numeric >= 4) & (valid_numeric <= 21)
                else:  # longitude
                    # Philippines longitude range: 116° to 127° E
                    reasonable_range = (valid_numeric >= -180) & (valid_numeric <= 180)
                    in_philippines = (valid_numeric >= 116) & (valid_numeric <= 127)

                reasonable_percentage = reasonable_range.mean()
                philippines_percentage = in_philippines.mean()

                # Adjust likelihood based on reasonable values
                if reasonable_percentage < 0.5:
                    likely_coordinate = False

        return {
            'likely_coordinate': likely_coordinate,
            'has_numeric_values': has_numeric_values,
            'coordinate_percentage': round(coordinate_percentage * 100, 1),
            'numeric_count': int(numeric_count),
            'total_count': int(total_count),
            'coordinate_format_count': int(coordinate_format_count)
        }

    def _validate_coordinate_column(self, series: pd.Series, min_bound: float,
                                  max_bound: float, coord_type: str) -> Tuple[np.ndarray, List[str]]:
        """
        Validate a coordinate column (latitude or longitude).

        Args:
            series (pd.Series): Coordinate series to validate
            min_bound (float): Minimum valid coordinate value
            max_bound (float): Maximum valid coordinate value
            coord_type (str): Type of coordinate ('latitude' or 'longitude')

        Returns:
            Tuple[np.ndarray, List[str]]: (validation_mask, list_of_issues)
        """
        valid_mask = np.zeros(len(series), dtype=bool)
        issues = []

        # Check for missing values
        missing_mask = series.isnull()
        missing_count = missing_mask.sum()
        if missing_count > 0:
            issues.append(f"{missing_count} missing {coord_type} values")

        # Process non-missing values
        non_missing = series[~missing_mask]
        non_missing_indices = series[~missing_mask].index

        if len(non_missing) == 0:
            return valid_mask, issues

        # Convert to string for pattern checking
        str_values = non_missing.astype(str).str.strip()

        # Check for DMS format (e.g., "14°35'23"N", "121°05'45"E")
        dms_pattern = r'.*°.*[\'"].*[NSEW]?'
        dms_mask = str_values.str.match(dms_pattern, na=False)
        dms_count = dms_mask.sum()
        if dms_count > 0:
            issues.append(f"{dms_count} {coord_type} values in DMS format (not decimal degrees)")

        # Check for DMM format (e.g., "14°35.23'N", "121°05.45'E")
        dmm_pattern = r'.*°.*\.\d+.*[\'"].*[NSEW]?'
        dmm_mask = str_values.str.match(dmm_pattern, na=False)
        dmm_count = dmm_mask.sum()
        if dmm_count > 0:
            issues.append(f"{dmm_count} {coord_type} values in DMM format (not decimal degrees)")

        # Identify non-DMS/DMM values for decimal validation
        coordinate_format_mask = ~(dms_mask | dmm_mask)
        potential_decimal = str_values[coordinate_format_mask]
        potential_decimal_indices = non_missing_indices[coordinate_format_mask]

        if len(potential_decimal) == 0:
            return valid_mask, issues

        # Try to convert to numeric (decimal degrees)
        numeric_values = pd.to_numeric(potential_decimal, errors='coerce')

        # Check for conversion failures (non-numeric text)
        conversion_failures = numeric_values.isnull()
        failure_count = conversion_failures.sum()
        if failure_count > 0:
            issues.append(f"{failure_count} {coord_type} values are non-numeric text")

        # Validate numeric values within bounds
        valid_numeric = numeric_values[~conversion_failures]
        valid_numeric_indices = potential_decimal_indices[~conversion_failures]

        if len(valid_numeric) > 0:
            # Check bounds
            within_bounds = (valid_numeric >= min_bound) & (valid_numeric <= max_bound)

            # Update valid mask for coordinates within bounds
            bounds_valid_indices = valid_numeric_indices[within_bounds]
            valid_mask[bounds_valid_indices] = True

            # Count out of bounds
            out_of_bounds_count = (~within_bounds).sum()
            if out_of_bounds_count > 0:
                issues.append(f"{out_of_bounds_count} {coord_type} values outside Philippine bounds")

        return valid_mask, issues


# Example usage demonstrating fast processing with coordinate validation
if __name__ == "__main__":
    # Initialize processor
    directory_path = r"C:\Users\elibu\Documents\Work\education\project_gastpe\data\private\raw_validation_sheets"

    # Fast processing with minimal logging
    processor = PrivateSchoolsProcessor(directory_path, verbose=False)

    # Process data efficiently
    processed_data = processor.process()

    # Get summary
    summary = processor.get_summary()
    print(f"Processed {summary['total_files_processed']} files")
    print(f"Success rate: {summary['success_rate']:.1f}%")
    print(f"Final dataset: {summary['final_dataset_rows']} rows × {summary['final_dataset_columns']} columns")

    # Validate coordinates if data was processed
    if len(processed_data) > 0:
        # Perform coordinate validation
        coord_summary = processor.validate_coordinates()

        if coord_summary:
            print(f"\nCoordinate Validation Results:")
            print(f"  Valid coordinates: {coord_summary['valid_coordinates']:,} ({coord_summary['validation_rate']:.1f}%)")
            print(f"  Invalid coordinates: {coord_summary['invalid_coordinates']:,}")

            if coord_summary['latitude_column'] and coord_summary['longitude_column']:
                print(f"  Columns used: {coord_summary['latitude_column']}, {coord_summary['longitude_column']}")

            if coord_summary['issues_found'] and coord_summary['issues_found'] != ['No major issues found']:
                print(f"  Issues found: {'; '.join(coord_summary['issues_found'])}")

        # Export processed data with coordinate validation
        processor.export_processed()
        print("Data exported successfully with coordinate validation")