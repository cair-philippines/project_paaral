"""
Facilities Data Preprocessor for Philippine Schools (SY 2023-2024)

This module processes school facilities data from DepEd, extracting classroom counts
and basic school information.

Key Features:
- Extracts columns 1-12 (school metadata and classroom counts)
- Handles blank values in classroom columns as NaN
- Validates data types and school identifiers
- Supports both Public and Private schools

Input:
    data/public/facilities_2023-24.csv

Output:
    DataFrame with columns:
    - school_id: School identifier (string)
    - sector: Public or Private (categorical)
    - school_management: Management type (categorical)
    - offers_es: Offers elementary school (boolean)
    - offers_jhs: Offers junior high school (boolean)
    - offers_shs: Offers senior high school (boolean)
    - es_classrooms_instructional: ES instructional classroom count (float, nullable)
    - jhs_classrooms_instructional: JHS instructional classroom count (float, nullable)
    - shs_classrooms_instructional: SHS instructional classroom count (float, nullable)
    - es_classrooms_non_instructional: ES non-instructional classroom count (float, nullable)
    - jhs_classrooms_non_instructional: JHS non-instructional classroom count (float, nullable)
    - shs_classrooms_non_instructional: SHS non-instructional classroom count (float, nullable)

Usage:
    from modules.facilities_preprocessor import FacilitiesProcessor

    processor = FacilitiesProcessor(file_path='data/public/facilities_2023-24.csv')
    facilities_data = processor.process()

    # View summary
    summary = processor.get_summary()
    print(summary)

    # Export to CSV
    processor.export_csv('output/facilities_classrooms.csv')

Author: Claude Code
Date: 2025-10-14
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Configure logging
logger = logging.getLogger(__name__)


class FacilitiesProcessor:
    """
    Preprocessor for school facilities data (SY 2023-2024).

    Extracts classroom counts and basic school information from the facilities dataset.
    """

    def __init__(self, file_path: str = None, verbose: bool = True):
        """
        Initialize the FacilitiesProcessor.

        Parameters:
            file_path (str): Path to the facilities CSV file
            verbose (bool): If True, set logging to INFO level; if False, WARNING level
        """
        self.file_path = Path('data/public/facilities_2023-24.csv') if file_path == None else Path(file_path)
        self.verbose = verbose
        self.processed_data: Optional[pd.DataFrame] = None

        # Configure logging level
        if self.verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)

        # Add console handler if not already present
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        logger.info(f"Initialized FacilitiesProcessor with file: {self.file_path}")

    def process(self) -> pd.DataFrame:
        """
        Main processing method to load and transform facilities data.

        Returns:
            pd.DataFrame: Processed facilities data with columns 1-12
        """
        logger.info("Starting facilities data processing...")

        # Load data
        self._load_data()

        # Select columns 1-12
        self._select_columns()

        # Convert data types
        self._convert_data_types()

        # Handle blank values as NaN
        self._handle_blank_values()

        # Validate data
        self._validate_data()

        # Trim whitespaces
        self._trim_whitespaces()

        logger.info(f"Processing complete. Final dataset shape: {self.processed_data.shape}")

        return self.processed_data

    def _load_data(self) -> None:
        """Load the facilities CSV file."""
        logger.info(f"Loading data from {self.file_path}...")

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        self.processed_data = pd.read_csv(self.file_path, dtype={'school_id': str}, low_memory=False)

        logger.info(f"Loaded {len(self.processed_data):,} rows")

    def _select_columns(self) -> None:
        """Select only columns 1-12 (indices 0-11)."""
        logger.info("Selecting columns 1-12...")

        columns_to_keep = [
            'school_id',
            'sector',
            'school_management',
            'offers_es',
            'offers_jhs',
            'offers_shs',
            'es_classrooms_instructional',
            'jhs_classrooms_instructional',
            'shs_classrooms_instructional',
            'es_classrooms_non_instructional',
            'jhs_classrooms_non_instructional',
            'shs_classrooms_non_instructional'
        ]

        self.processed_data = self.processed_data[columns_to_keep].copy()

        logger.info(f"Selected {len(columns_to_keep)} columns")

    def _convert_data_types(self) -> None:
        """Convert columns to appropriate data types."""
        logger.info("Converting data types...")
        
        # Boolean columns
        boolean_columns = ['offers_es', 'offers_jhs', 'offers_shs']
        for col in boolean_columns:
            self.processed_data[col] = self.processed_data[col].astype(bool)

        # Classroom count columns - will be converted to nullable float after handling blanks
        classroom_columns = [
            'es_classrooms_instructional',
            'jhs_classrooms_instructional',
            'shs_classrooms_instructional',
            'es_classrooms_non_instructional',
            'jhs_classrooms_non_instructional',
            'shs_classrooms_non_instructional'
        ]

        # First, replace empty strings with NaN
        for col in classroom_columns:
            self.processed_data[col] = self.processed_data[col].replace('', np.nan)
            self.processed_data[col] = pd.to_numeric(self.processed_data[col], errors='coerce')

        logger.info("Data type conversion complete")

    def _handle_blank_values(self) -> None:
        """
        Explicitly handle blank values in classroom columns as NaN.

        Blank values occur when:
        - School is Private (no data in columns 7-12)
        - School is Public but doesn't offer that education level
        """
        logger.info("Handling blank values in classroom columns...")

        classroom_columns = [
            'es_classrooms_instructional',
            'jhs_classrooms_instructional',
            'shs_classrooms_instructional',
            'es_classrooms_non_instructional',
            'jhs_classrooms_non_instructional',
            'shs_classrooms_non_instructional'
        ]

        # Count NaN values before (for reporting)
        nan_counts_before = self.processed_data[classroom_columns].isna().sum()

        # Blanks are already converted to NaN in _convert_data_types
        # But let's ensure any remaining empty strings or whitespace are converted
        for col in classroom_columns:
            # Replace whitespace-only strings with NaN
            mask = self.processed_data[col].astype(str).str.strip() == ''
            self.processed_data.loc[mask, col] = np.nan

        # Count NaN values after
        nan_counts_after = self.processed_data[classroom_columns].isna().sum()

        # Report statistics
        total_private = (self.processed_data['sector'] == 'Private').sum()
        total_public = (self.processed_data['sector'] == 'Public').sum()

        logger.info(f"Total schools: {len(self.processed_data):,}")
        logger.info(f"  - Private: {total_private:,}")
        logger.info(f"  - Public: {total_public:,}")
        logger.info(f"NaN values in classroom columns (per column):")
        for col in classroom_columns:
            logger.info(f"  - {col}: {nan_counts_after[col]:,}")

    def _validate_data(self) -> None:
        """Validate the processed data."""
        logger.info("Validating data...")

        # Check for duplicate school IDs
        duplicates = self.processed_data['school_id'].duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate school IDs")

        # Validate classroom counts are non-negative
        classroom_columns = [
            'es_classrooms_instructional',
            'jhs_classrooms_instructional',
            'shs_classrooms_instructional',
            'es_classrooms_non_instructional',
            'jhs_classrooms_non_instructional',
            'shs_classrooms_non_instructional'
        ]

        for col in classroom_columns:
            negative_count = (self.processed_data[col] < 0).sum()
            if negative_count > 0:
                logger.warning(f"Found {negative_count} negative values in {col}")

        # Check consistency: schools not offering a level should have NaN in classroom counts
        level_mappings = [
            ('offers_es', ['es_classrooms_instructional', 'es_classrooms_non_instructional']),
            ('offers_jhs', ['jhs_classrooms_instructional', 'jhs_classrooms_non_instructional']),
            ('offers_shs', ['shs_classrooms_instructional', 'shs_classrooms_non_instructional'])
        ]

        for offers_col, classroom_cols in level_mappings:
            not_offered = ~self.processed_data[offers_col]
            for classroom_col in classroom_cols:
                # Count schools that don't offer level but have classroom data
                inconsistent = not_offered & self.processed_data[classroom_col].notna()
                if inconsistent.sum() > 0:
                    logger.warning(
                        f"Found {inconsistent.sum()} schools with {offers_col}=False "
                        f"but have data in {classroom_col}"
                    )

        logger.info("Validation complete")

    def _trim_whitespaces(self) -> None:
        """Trim whitespaces from string columns."""
        logger.info("Trimming whitespaces from string columns...")

        string_columns = ['school_id', 'sector', 'school_management']

        for col in string_columns:
            if col in self.processed_data.columns:
                self.processed_data[col] = self.processed_data[col].astype(str).str.strip()

        logger.info("Whitespace trimming complete")

    def get_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the processed data.

        Returns:
            dict: Summary statistics
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Run process() first.")

        summary = {
            'total_schools': len(self.processed_data),
            'schools_by_sector': self.processed_data['sector'].value_counts().to_dict(),
            'schools_by_management': self.processed_data['school_management'].value_counts().to_dict(),
            'schools_offering_es': self.processed_data['offers_es'].sum(),
            'schools_offering_jhs': self.processed_data['offers_jhs'].sum(),
            'schools_offering_shs': self.processed_data['offers_shs'].sum(),
            'total_es_instructional_classrooms': self.processed_data['es_classrooms_instructional'].sum(),
            'total_jhs_instructional_classrooms': self.processed_data['jhs_classrooms_instructional'].sum(),
            'total_shs_instructional_classrooms': self.processed_data['shs_classrooms_instructional'].sum(),
            'nan_counts': self.processed_data.isna().sum().to_dict()
        }

        return summary

    def export_csv(self, output_path: str, index: bool = False) -> None:
        """
        Export processed data to CSV.

        Parameters:
            output_path (str): Path for output CSV file
            index (bool): Whether to include index in output
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Run process() first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.processed_data.to_csv(output_path, index=index)
        logger.info(f"Exported data to {output_path}")

    def export_excel(self, output_path: str, sheet_name: str = 'Facilities', index: bool = False) -> None:
        """
        Export processed data to Excel.

        Parameters:
            output_path (str): Path for output Excel file
            sheet_name (str): Name of the Excel sheet
            index (bool): Whether to include index in output
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Run process() first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.processed_data.to_excel(output_path, sheet_name=sheet_name, index=index)
        logger.info(f"Exported data to {output_path}")


# Convenience function for quick usage
def load_facilities_data(file_path: str, verbose: bool = True) -> pd.DataFrame:
    """
    Convenience function to quickly load and process facilities data.

    Parameters:
        file_path (str): Path to the facilities CSV file
        verbose (bool): Enable verbose logging

    Returns:
        pd.DataFrame: Processed facilities data

    Example:
        >>> facilities_df = load_facilities_data('data/public/facilities_2023-24.csv')
    """
    processor = FacilitiesProcessor(file_path=file_path, verbose=verbose)
    return processor.process()


if __name__ == "__main__":
    # Example usage
    processor = FacilitiesProcessor(
        file_path='../data/public/facilities_2023-24.csv',
        verbose=True
    )

    # Process the data
    facilities_data = processor.process()

    # Display summary
    print("\n" + "="*50)
    print("FACILITIES DATA SUMMARY")
    print("="*50)
    summary = processor.get_summary()
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v:,}" if isinstance(v, (int, float)) else f"  {k}: {v}")
        else:
            print(f"{key}: {value:,}" if isinstance(value, (int, float)) else f"{key}: {value}")

    # Display first few rows
    print("\n" + "="*50)
    print("SAMPLE DATA (first 10 rows)")
    print("="*50)
    print(facilities_data.head(10))

    # Display data types
    print("\n" + "="*50)
    print("DATA TYPES")
    print("="*50)
    print(facilities_data.dtypes)
