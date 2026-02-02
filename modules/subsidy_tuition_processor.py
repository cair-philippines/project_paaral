"""
Subsidy Tuition Data Preprocessor

This module processes the ESC and SHSVP Tuition.xlsx file to extract tuition and fees
data for private schools participating in ESC (Educational Service Contracting) and
SHSVP (Senior High School Voucher Program).

The Excel file contains two sheets:
1. ESC Tuition from PEAC - Wide format tuition data for Grade 7-10 (Junior High School)
2. SHSVP Tuition from PEAC - Long format tuition data for Senior High School by Track/Strand

Author: Data Processing System
"""

import pandas as pd
import numpy as np
import logging
import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SubsidyTuitionProcessor:
    """
    Processes subsidy tuition data from Excel to long format DataFrames.

    The processor reads both ESC and SHSVP tabs from the Excel file, transforms
    ESC data from wide to long format, and returns standardized DataFrames for
    both datasets suitable for analysis and merging with other datasets.
    """

    def __init__(self, file_path: Optional[str] = None, verbose: bool = True):
        """
        Initialize the processor.

        Args:
            file_path: Path to the Excel file. If None, uses default path.
            verbose: If True, logs at INFO level. If False, logs at WARNING level only.
        """
        if file_path is None:
            file_path = "data/private/ESC and SHSVP Tuition.xlsx"

        self.file_path = Path(file_path)
        self.verbose = verbose
        self.raw_esc_data = None
        self.raw_shsvp_data = None
        self.processed_esc_data = None
        self.processed_shsvp_data = None

        # Set logging level based on verbose flag
        if not verbose:
            logger.setLevel(logging.WARNING)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the ESC and SHSVP tuition data from Excel file.

        Returns:
            Tuple of (ESC DataFrame, SHSVP DataFrame) with raw data
        """
        try:
            logger.info(f"Loading data from {self.file_path}")

            # Load ESC Tuition tab (Tab 1)
            logger.info("Loading ESC Tuition from PEAC sheet...")
            self.raw_esc_data = pd.read_excel(
                self.file_path,
                sheet_name='ESC Tuition from PEAC'
            )

            # Clean column names
            self.raw_esc_data.columns = self.raw_esc_data.columns.str.strip()

            # Trim whitespaces from string columns
            self.raw_esc_data = self._trim_whitespaces(self.raw_esc_data)

            logger.info(f"Loaded ESC data: {len(self.raw_esc_data)} records with {len(self.raw_esc_data.columns)} columns")

            # Load SHSVP Tuition tab (Tab 2)
            logger.info("Loading SHSVP Tuition from PEAC sheet...")
            self.raw_shsvp_data = pd.read_excel(
                self.file_path,
                sheet_name='SHSVP Tuition from PEAC'
            )

            # Clean column names
            self.raw_shsvp_data.columns = self.raw_shsvp_data.columns.str.strip()

            # Trim whitespaces from string columns
            self.raw_shsvp_data = self._trim_whitespaces(self.raw_shsvp_data)

            logger.info(f"Loaded SHSVP data: {len(self.raw_shsvp_data)} records with {len(self.raw_shsvp_data.columns)} columns")

            return self.raw_esc_data, self.raw_shsvp_data

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

    def _identify_esc_columns(self, df: pd.DataFrame) -> Dict[str, list]:
        """
        Identify school info columns and tuition/fees columns in ESC data.

        Based on user specification:
        - Columns A-H (indices 0-7): School information
        - Columns I-T (indices 8-19): Tuition and fees for G7-G10

        Args:
            df: ESC DataFrame

        Returns:
            Dictionary with 'school_info' and 'tuition_fees' column lists
        """
        all_columns = df.columns.tolist()

        # School info columns: A-H (first 8 columns)
        school_info_cols = all_columns[0:8]

        # Tuition and fees columns: I-T (columns 8-19)
        tuition_fees_cols = all_columns[8:20]

        logger.info(f"Identified {len(school_info_cols)} school info columns")
        logger.info(f"Identified {len(tuition_fees_cols)} tuition/fees columns")

        return {
            'school_info': school_info_cols,
            'tuition_fees': tuition_fees_cols
        }

    def _parse_esc_column_name(self, col_name: str) -> Dict[str, str]:
        """
        Parse ESC tuition column name to extract grade level and fee type.

        Expected column patterns:
        - Contains grade indicators: G7, G8, G9, G10
        - May contain fee type indicators: Tuition, Fees, etc.

        Args:
            col_name: Name of the tuition/fees column

        Returns:
            Dictionary with grade_level and fee_type
        """
        col_name_clean = str(col_name).strip()

        result = {
            'grade_level': None,
            'fee_type': None
        }

        # Extract grade level
        if 'G7' in col_name_clean or 'Grade 7' in col_name_clean:
            result['grade_level'] = 'G7'
        elif 'G8' in col_name_clean or 'Grade 8' in col_name_clean:
            result['grade_level'] = 'G8'
        elif 'G9' in col_name_clean or 'Grade 9' in col_name_clean:
            result['grade_level'] = 'G9'
        elif 'G10' in col_name_clean or 'Grade 10' in col_name_clean:
            result['grade_level'] = 'G10'

        # Extract fee type (default patterns - adjust based on actual data)
        col_lower = col_name_clean.lower()
        if 'tuition' in col_lower:
            result['fee_type'] = 'Tuition'
        elif 'fees' in col_lower or 'fee' in col_lower:
            result['fee_type'] = 'Fees'
        elif 'misc' in col_lower:
            result['fee_type'] = 'Miscellaneous'
        else:
            # If no specific type identified, use the column name
            # Remove grade information to get fee type
            fee_type = col_name_clean
            for grade in ['G7', 'G8', 'G9', 'G10', 'Grade 7', 'Grade 8', 'Grade 9', 'Grade 10']:
                fee_type = fee_type.replace(grade, '').strip()
            result['fee_type'] = fee_type if fee_type else 'Amount'

        return result

    def process_esc_wide_to_long(self) -> pd.DataFrame:
        """
        Transform ESC tuition data from wide to long format.

        Converts Grade 7-10 tuition data from wide format (one column per grade/fee)
        to long format with columns: school_id, grade_level, fee_type, amount

        Returns:
            Long format DataFrame with ESC tuition data
        """
        if self.raw_esc_data is None:
            raise ValueError("ESC data not loaded. Call load_data() first.")

        logger.info("Transforming ESC data from wide to long format")

        # Identify columns
        column_groups = self._identify_esc_columns(self.raw_esc_data)
        school_info_cols = column_groups['school_info']
        tuition_fees_cols = column_groups['tuition_fees']

        # Find School ID column (assumed to be in school info columns)
        # Common names: School ID, School_ID, SchoolID, etc.
        school_id_col = None
        for col in school_info_cols:
            if 'school' in col.lower() and 'id' in col.lower():
                school_id_col = col
                break

        if school_id_col is None:
            # If not found by name, assume first column is School ID
            school_id_col = school_info_cols[0]
            logger.warning(f"School ID column not found by name, using first column: {school_id_col}")

        logger.info(f"Using '{school_id_col}' as School ID column")

        # Prepare list to store long format records
        long_records = []

        for index, row in self.raw_esc_data.iterrows():
            school_id = row[school_id_col]

            # Skip rows with missing school ID
            if pd.isna(school_id):
                continue

            # Extract school info
            school_info = {col: row[col] for col in school_info_cols}

            # Extract tuition/fees for each column
            for col in tuition_fees_cols:
                amount = row[col]

                # Parse column name to get grade and fee type
                parsed_info = self._parse_esc_column_name(col)

                # Skip if grade level not identified
                if parsed_info['grade_level'] is None:
                    logger.warning(f"Could not identify grade level for column: {col}")
                    continue

                # Convert to numeric, handle non-numeric values
                try:
                    amount = pd.to_numeric(amount, errors='coerce')

                    # Include all records, even zero amounts (school might have zero fees)
                    if pd.notna(amount):
                        record = {
                            'school_id': str(school_id).strip(),
                            'grade_level': parsed_info['grade_level'],
                            'fee_type': parsed_info['fee_type'],
                            'amount': float(amount)
                        }
                        # Add all school info columns
                        for col_name, col_value in school_info.items():
                            if col_name != school_id_col:  # Avoid duplicating school_id
                                record[col_name] = col_value

                        long_records.append(record)
                except (ValueError, TypeError):
                    # Skip invalid amount values
                    continue

        # Create long format DataFrame
        self.processed_esc_data = pd.DataFrame(long_records)

        # Trim whitespaces from string columns
        self.processed_esc_data = self._trim_whitespaces(self.processed_esc_data)

        # Transform data types
        self.processed_esc_data['school_id'] = self.processed_esc_data['school_id'].astype('string')

        # Transform grade_level to categorical with proper ordering
        grade_order = ['G7', 'G8', 'G9', 'G10']
        self.processed_esc_data['grade_level'] = pd.Categorical(
            self.processed_esc_data['grade_level'],
            categories=grade_order,
            ordered=True
        )

        # Transform fee_type to categorical
        fee_types = self.processed_esc_data['fee_type'].unique().tolist()
        self.processed_esc_data['fee_type'] = pd.Categorical(
            self.processed_esc_data['fee_type'],
            categories=sorted(fee_types),
            ordered=False
        )

        logger.info(f"Created ESC long format data with {len(self.processed_esc_data)} records")
        logger.info(f"Grade levels: {self.processed_esc_data['grade_level'].unique().tolist()}")
        logger.info(f"Fee types: {self.processed_esc_data['fee_type'].unique().tolist()}")

        return self.processed_esc_data

    def _expand_concatenated_strands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Expand rows with concatenated strands into separate rows.

        Some rows contain multiple technical-vocational programs concatenated in a single
        Strand cell, marked by NC I, NC II, or NC III indicators. This method detects such rows
        and expands them into separate rows, one per program.

        Example:
            Input: "Housekeeping (NC II) Food and Beverage Services (NC II)"
            Output: Two rows with "Housekeeping (NC II)" and "Food and Beverage Services (NC II)"

        Args:
            df: DataFrame with Strand column

        Returns:
            DataFrame with expanded strand rows
        """
        if 'Strand' not in df.columns:
            logger.warning("No 'Strand' column found, skipping strand expansion")
            return df

        logger.info("Checking for concatenated strands to expand")

        # Pattern to match NC I, NC II, or NC III markers (with or without parentheses)
        nc_pattern = r'\(NC\s*I+\)|NC\s*I+'

        # Find rows with multiple NC markers
        strand_series = df['Strand'].astype(str)
        nc_matches = strand_series.apply(lambda x: len(re.findall(nc_pattern, x, re.IGNORECASE)))
        rows_to_expand = df[nc_matches > 1].copy()
        rows_single = df[nc_matches <= 1].copy()

        if len(rows_to_expand) == 0:
            logger.info("No concatenated strands found")
            return df

        logger.info(f"Found {len(rows_to_expand)} rows with concatenated strands")

        # Expand concatenated strands
        expanded_rows = []

        for idx, row in rows_to_expand.iterrows():
            strand_value = str(row['Strand'])

            # Find all NC marker positions
            nc_positions = [(m.start(), m.end()) for m in re.finditer(nc_pattern, strand_value, re.IGNORECASE)]

            if len(nc_positions) <= 1:
                # Should not happen based on filtering, but handle gracefully
                expanded_rows.append(row.to_dict())
                continue

            # Split strand by extracting text between NC markers
            individual_strands = []

            # Extract each program by finding the end of previous NC marker to start of current program
            for i in range(len(nc_positions)):
                if i == 0:
                    # First program: from start to end of first NC marker
                    program_start = 0
                else:
                    # Subsequent programs: from end of previous NC marker to end of current NC marker
                    program_start = nc_positions[i-1][1]

                program_end = nc_positions[i][1]
                program = strand_value[program_start:program_end].strip()

                if program:
                    individual_strands.append(program)

            # Create a new row for each strand
            for strand in individual_strands:
                new_row = row.to_dict()
                new_row['Strand'] = strand
                expanded_rows.append(new_row)

        # Combine single rows with expanded rows
        expanded_df = pd.DataFrame(expanded_rows)
        result_df = pd.concat([rows_single, expanded_df], ignore_index=True)

        # Trim whitespaces from Strand column
        result_df['Strand'] = result_df['Strand'].astype(str).str.strip()

        logger.info(f"Expanded {len(rows_to_expand)} rows into {len(expanded_rows)} rows")
        logger.info(f"Total rows after expansion: {len(result_df)} (was {len(df)})")

        return result_df

    def process_shsvp_data(self) -> pd.DataFrame:
        """
        Process SHSVP tuition data (already in long format).

        The SHSVP data is already in long format with Track and Strand as unique identifiers.
        This method performs basic cleaning, standardization, and expands concatenated strands.

        Returns:
            Processed SHSVP DataFrame in long format
        """
        if self.raw_shsvp_data is None:
            raise ValueError("SHSVP data not loaded. Call load_data() first.")

        logger.info("Processing SHSVP data (already in long format)")

        # Create a copy for processing
        self.processed_shsvp_data = self.raw_shsvp_data.copy()

        # Find School ID column
        school_id_col = None
        for col in self.processed_shsvp_data.columns:
            if 'school' in col.lower() and 'id' in col.lower():
                school_id_col = col
                break

        if school_id_col is None:
            # Try to find ID column among first 11 columns (A-K)
            for col in self.processed_shsvp_data.columns[:11]:
                if 'id' in col.lower():
                    school_id_col = col
                    break

        if school_id_col:
            logger.info(f"Using '{school_id_col}' as School ID column")
            # Standardize column name
            if school_id_col != 'school_id':
                self.processed_shsvp_data['school_id'] = self.processed_shsvp_data[school_id_col]
            self.processed_shsvp_data['school_id'] = self.processed_shsvp_data['school_id'].astype('string')

        # Process Track and Strand columns if they exist
        if 'Track' in self.processed_shsvp_data.columns:
            self.processed_shsvp_data['Track'] = pd.Categorical(
                self.processed_shsvp_data['Track']
            )

        if 'Strand' in self.processed_shsvp_data.columns:
            self.processed_shsvp_data['Strand'] = pd.Categorical(
                self.processed_shsvp_data['Strand']
            )

        # Remove rows with missing school ID
        if 'school_id' in self.processed_shsvp_data.columns:
            self.processed_shsvp_data = self.processed_shsvp_data[
                self.processed_shsvp_data['school_id'].notna()
            ]

        # Trim whitespaces
        self.processed_shsvp_data = self._trim_whitespaces(self.processed_shsvp_data)

        # Record counts before expansion
        rows_before_expansion = len(self.processed_shsvp_data)

        # Expand concatenated strands
        self.processed_shsvp_data = self._expand_concatenated_strands(self.processed_shsvp_data)

        # Record counts after expansion
        rows_after_expansion = len(self.processed_shsvp_data)
        rows_added = rows_after_expansion - rows_before_expansion

        logger.info(f"Processed SHSVP data with {len(self.processed_shsvp_data)} records")
        logger.info(f"Strand expansion added {rows_added} rows ({rows_before_expansion} -> {rows_after_expansion})")
        if 'Track' in self.processed_shsvp_data.columns:
            logger.info(f"Tracks: {self.processed_shsvp_data['Track'].unique().tolist()}")
        if 'Strand' in self.processed_shsvp_data.columns:
            logger.info(f"Strands after expansion: {self.processed_shsvp_data['Strand'].nunique()} unique")

        return self.processed_shsvp_data

    def process(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main processing pipeline: load and transform both ESC and SHSVP data.

        Returns:
            Tuple of (ESC long format DataFrame, SHSVP processed DataFrame)
        """
        logger.info("Starting subsidy tuition data processing")

        # Load raw data
        self.load_data()

        # Process ESC data (wide to long)
        esc_processed = self.process_esc_wide_to_long()

        # Process SHSVP data (already long format)
        shsvp_processed = self.process_shsvp_data()

        logger.info("Processing completed successfully")
        return esc_processed, shsvp_processed

    def get_esc_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the processed ESC data.

        Returns:
            Dictionary with ESC summary statistics
        """
        if self.processed_esc_data is None:
            raise ValueError("ESC data not processed. Call process() first.")

        summary = {
            'total_records': len(self.processed_esc_data),
            'unique_schools': self.processed_esc_data['school_id'].nunique(),
            'grade_levels': self.processed_esc_data['grade_level'].unique().tolist(),
            'fee_types': self.processed_esc_data['fee_type'].unique().tolist(),
            'amount_statistics': {
                'mean': self.processed_esc_data['amount'].mean(),
                'median': self.processed_esc_data['amount'].median(),
                'min': self.processed_esc_data['amount'].min(),
                'max': self.processed_esc_data['amount'].max(),
                'total': self.processed_esc_data['amount'].sum()
            },
            'amounts_by_grade': self.processed_esc_data.groupby('grade_level')['amount'].sum().to_dict(),
            'amounts_by_fee_type': self.processed_esc_data.groupby('fee_type')['amount'].sum().to_dict()
        }

        return summary

    def get_shsvp_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the processed SHSVP data.

        Returns:
            Dictionary with SHSVP summary statistics
        """
        if self.processed_shsvp_data is None:
            raise ValueError("SHSVP data not processed. Call process() first.")

        summary = {
            'total_records': len(self.processed_shsvp_data),
            'unique_schools': self.processed_shsvp_data['school_id'].nunique() if 'school_id' in self.processed_shsvp_data.columns else 0,
        }

        if 'Track' in self.processed_shsvp_data.columns:
            summary['unique_tracks'] = self.processed_shsvp_data['Track'].nunique()
            summary['tracks'] = self.processed_shsvp_data['Track'].unique().tolist()

        if 'Strand' in self.processed_shsvp_data.columns:
            summary['unique_strands'] = self.processed_shsvp_data['Strand'].nunique()
            summary['strands_after_expansion'] = self.processed_shsvp_data['Strand'].nunique()
            summary['strands'] = self.processed_shsvp_data['Strand'].unique().tolist()

        # Find amount/tuition column
        amount_cols = [col for col in self.processed_shsvp_data.columns
                      if any(keyword in col.lower() for keyword in ['amount', 'tuition', 'fee'])]

        if amount_cols:
            amount_col = amount_cols[0]
            summary['amount_column'] = amount_col
            summary['amount_statistics'] = {
                'mean': self.processed_shsvp_data[amount_col].mean(),
                'median': self.processed_shsvp_data[amount_col].median(),
                'min': self.processed_shsvp_data[amount_col].min(),
                'max': self.processed_shsvp_data[amount_col].max()
            }

        return summary

    def get_esc_processed_data(self) -> pd.DataFrame:
        """
        Get the processed ESC long format data.

        Returns:
            Processed ESC DataFrame in long format
        """
        if self.processed_esc_data is None:
            raise ValueError("ESC data not processed. Call process() first.")

        return self.processed_esc_data.copy()

    def get_shsvp_processed_data(self) -> pd.DataFrame:
        """
        Get the processed SHSVP data.

        Returns:
            Processed SHSVP DataFrame
        """
        if self.processed_shsvp_data is None:
            raise ValueError("SHSVP data not processed. Call process() first.")

        return self.processed_shsvp_data.copy()

    def export_processed(self, esc_output_path: str = 'output/esc_tuition_long_format.csv',
                        shsvp_output_path: str = 'output/shsvp_tuition_long_format.csv') -> None:
        """
        Export processed data to CSV files.

        Args:
            esc_output_path: Path for the ESC output CSV file
            shsvp_output_path: Path for the SHSVP output CSV file
        """
        if self.processed_esc_data is None or self.processed_shsvp_data is None:
            raise ValueError("Data not processed. Call process() first.")

        # Ensure output directory exists
        Path(esc_output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(shsvp_output_path).parent.mkdir(parents=True, exist_ok=True)

        self.processed_esc_data.to_csv(esc_output_path, index=False)
        logger.info(f"Exported {len(self.processed_esc_data)} ESC records to {esc_output_path}")

        self.processed_shsvp_data.to_csv(shsvp_output_path, index=False)
        logger.info(f"Exported {len(self.processed_shsvp_data)} SHSVP records to {shsvp_output_path}")


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = SubsidyTuitionProcessor()

    # Process both datasets
    esc_data, shsvp_data = processor.process()

    # Get ESC summary
    esc_summary = processor.get_esc_summary()
    print("\nESC Tuition Data Summary:")
    print(f"Total records: {esc_summary['total_records']:,}")
    print(f"Unique schools: {esc_summary['unique_schools']:,}")
    print(f"Grade levels: {esc_summary['grade_levels']}")
    print(f"Fee types: {esc_summary['fee_types']}")
    print(f"Amount range: {esc_summary['amount_statistics']['min']:,.2f} - {esc_summary['amount_statistics']['max']:,.2f}")

    # Get SHSVP summary
    shsvp_summary = processor.get_shsvp_summary()
    print("\nSHSVP Tuition Data Summary:")
    print(f"Total records: {shsvp_summary['total_records']:,}")
    print(f"Unique schools: {shsvp_summary['unique_schools']:,}")
    if 'tracks' in shsvp_summary:
        print(f"Tracks: {shsvp_summary['tracks']}")
    if 'strands' in shsvp_summary:
        print(f"Unique strands: {shsvp_summary['unique_strands']}")

    # Show sample data
    print("\nESC Sample Data:")
    print(esc_data.head(10))

    print("\nSHSVP Sample Data:")
    print(shsvp_data.head(10))

    # Export processed data
    processor.export_processed()