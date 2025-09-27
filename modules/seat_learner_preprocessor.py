"""
Seat-Learner Ratio Data Preprocessor

This module processes the SY 2023-2024 SEAT-LEARNER RATIO.xlsx file to extract
seat count data for public schools and transform it into long format for analysis.

The Excel file contains seat counts in columns T, U, V for Elementary, Junior High School,
and Senior High School respectively.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SeatLearnerProcessor:
    """
    Processes seat-learner ratio data from Excel to long format DataFrame.

    The processor reads the DATABASE sheet from the Excel file, extracts seat counts
    from columns T, U, V, and transforms the data into a long format suitable for
    analysis and merging with other datasets.
    """

    def __init__(self, file_path: Optional[str] = None):
        """
        Initialize the processor.

        Args:
            file_path: Path to the Excel file. If None, uses default path.
        """
        if file_path is None:
            file_path = "data/public/SY 2023-2024 SEAT-LEARNER RATIO.xlsx"

        self.file_path = Path(file_path)
        self.raw_data = None
        self.processed_data = None

    def load_data(self) -> pd.DataFrame:
        """
        Load the seat-learner ratio data from Excel file.

        Returns:
            Raw DataFrame with seat data
        """
        try:
            logger.info(f"Loading data from {self.file_path}")

            # Load DATABASE sheet with headers starting at row 7 (header=6)
            self.raw_data = pd.read_excel(
                self.file_path,
                sheet_name='DATABASE',
                header=6
            )

            # Clean column names (remove extra spaces)
            self.raw_data.columns = self.raw_data.columns.str.strip()

            # Trim whitespaces from string columns
            self.raw_data = self._trim_whitespaces(self.raw_data)

            logger.info(f"Loaded {len(self.raw_data)} records with {len(self.raw_data.columns)} columns")
            return self.raw_data

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

    def wide_to_long(self) -> pd.DataFrame:
        """
        Transform seat data from wide to long format.

        Extracts seat counts from columns T, U, V and creates a long format DataFrame
        with columns: school_id, education_level, seat_count

        Returns:
            Long format DataFrame with seat counts
        """
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        logger.info("Transforming seat data from wide to long format")

        # Get the SCHOOL ID column (column D)
        school_id_col = 'SCHOOL ID'

        # Verify the column exists
        if school_id_col not in self.raw_data.columns:
            raise ValueError(f"Column '{school_id_col}' not found in data. Available columns: {list(self.raw_data.columns)}")

        # Seat count columns are at indices 19, 20, 21 (T, U, V)
        seat_columns = {
            19: 'Elementary',           # Column T
            20: 'Junior High School',   # Column U
            21: 'Senior High School'    # Column V
        }

        # Prepare list to store long format records
        long_records = []

        for index, row in self.raw_data.iterrows():
            school_id = row[school_id_col]

            # Skip rows with missing school ID
            if pd.isna(school_id):
                continue

            # Extract seat counts for each education level
            for col_idx, education_level in seat_columns.items():
                if col_idx < len(self.raw_data.columns):
                    seat_count = row.iloc[col_idx]

                    # Convert to numeric, handle non-numeric values
                    try:
                        seat_count = pd.to_numeric(seat_count, errors='coerce')

                        # Only include valid, positive seat counts
                        if pd.notna(seat_count) and seat_count > 0:
                            long_records.append({
                                'school_id': str(school_id).strip(),
                                'education_level': education_level,
                                'seat_count': int(seat_count)
                            })
                    except (ValueError, TypeError):
                        # Skip invalid seat count values
                        continue

        # Create long format DataFrame
        self.processed_data = pd.DataFrame(long_records)

        # Trim whitespaces from string columns
        self.processed_data = self._trim_whitespaces(self.processed_data)

        # Transform school_id dtype to string
        self.processed_data['school_id'] = self.processed_data['school_id'].astype('string')

        # Transform education_level to categorical with proper ordering
        education_order = ['Elementary', 'Junior High School', 'Senior High School']
        self.processed_data['education_level'] = pd.Categorical(
            self.processed_data['education_level'],
            categories=education_order,
            ordered=True
        )

        logger.info(f"Created long format data with {len(self.processed_data)} records")
        logger.info(f"Education levels: {self.processed_data['education_level'].unique().tolist()}")

        return self.processed_data

    def process(self) -> pd.DataFrame:
        """
        Main processing pipeline: load data and transform to long format.

        Returns:
            Long format DataFrame with seat counts
        """
        logger.info("Starting seat-learner ratio data processing")

        # Load raw data
        self.load_data()

        # Transform to long format
        self.wide_to_long()

        logger.info("Processing completed successfully")
        return self.processed_data

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the processed data.

        Returns:
            Dictionary with summary statistics
        """
        if self.processed_data is None:
            raise ValueError("Data not processed. Call process() first.")

        summary = {
            'total_records': len(self.processed_data),
            'unique_schools': self.processed_data['school_id'].nunique(),
            'education_levels': self.processed_data['education_level'].unique().tolist(),
            'total_seats': self.processed_data['seat_count'].sum(),
            'seat_statistics': {
                'mean': self.processed_data['seat_count'].mean(),
                'median': self.processed_data['seat_count'].median(),
                'min': self.processed_data['seat_count'].min(),
                'max': self.processed_data['seat_count'].max()
            },
            'seats_by_level': self.processed_data.groupby('education_level')['seat_count'].sum().to_dict()
        }

        return summary

    def filter_by_education_level(self, levels: list) -> pd.DataFrame:
        """
        Filter data by education levels.

        Args:
            levels: List of education levels to include

        Returns:
            Filtered DataFrame
        """
        if self.processed_data is None:
            raise ValueError("Data not processed. Call process() first.")

        filtered_data = self.processed_data[
            self.processed_data['education_level'].isin(levels)
        ].copy()

        logger.info(f"Filtered to {len(filtered_data)} records for levels: {levels}")
        return filtered_data

    def export_processed(self, output_path: str) -> None:
        """
        Export processed data to CSV.

        Args:
            output_path: Path for the output CSV file
        """
        if self.processed_data is None:
            raise ValueError("Data not processed. Call process() first.")

        self.processed_data.to_csv(output_path, index=False)
        logger.info(f"Exported {len(self.processed_data)} records to {output_path}")

    def get_processed_data(self) -> pd.DataFrame:
        """
        Get the processed long format data.

        Returns:
            Processed DataFrame in long format
        """
        if self.processed_data is None:
            raise ValueError("Data not processed. Call process() first.")

        return self.processed_data.copy()


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = SeatLearnerProcessor()

    # Process data to long format
    long_data = processor.process()

    # Get summary
    summary = processor.get_summary()
    print("\nSeat-Learner Ratio Data Summary:")
    print(f"Total records: {summary['total_records']:,}")
    print(f"Unique schools: {summary['unique_schools']:,}")
    print(f"Total seats: {summary['total_seats']:,}")
    print(f"Education levels: {summary['education_levels']}")

    # Show sample data
    print("\nSample data:")
    print(long_data.head(10))

    # Export processed data
    processor.export_processed("output/seat_learner_long_format.csv")