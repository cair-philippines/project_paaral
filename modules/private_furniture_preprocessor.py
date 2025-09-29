"""
Private School Furniture Data Preprocessor

This module processes the priv_classroom_furniture.xlsx file to extract
furniture count data for private schools and transform it into long format for analysis.

The Excel file contains furniture counts in columns I to X with headers indicating
grade levels (Kinder, Gr1to6, JHS, SHS) and furniture types.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrivateFurnitureProcessor:
    """
    Processes private school furniture data from Excel to long format DataFrame.

    The processor reads the Excel file, extracts furniture counts from columns I-X,
    and transforms the data into a long format suitable for analysis and merging
    with other datasets.
    """

    def __init__(self, file_path: Optional[str] = None):
        """
        Initialize the processor.

        Args:
            file_path: Path to the Excel file. If None, uses default path.
        """
        if file_path is None:
            file_path = "data/private/priv_classroom_furniture.xlsx"

        self.file_path = Path(file_path)
        self.raw_data = None
        self.processed_data = None

    def load_data(self) -> pd.DataFrame:
        """
        Load the private furniture data from Excel file.

        Returns:
            Raw DataFrame with furniture data
        """
        try:
            logger.info(f"Loading data from {self.file_path}")

            # Load Excel with headers starting at row 10 (header=9)
            self.raw_data = pd.read_excel(
                self.file_path,
                header=9
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

    def _parse_furniture_column(self, column_name: str) -> Dict[str, str]:
        """
        Parse furniture column name to extract grade level and furniture type.

        Args:
            column_name: Name of the furniture column

        Returns:
            Dictionary with grade_level and furniture_type
        """
        # Clean the column name
        clean_name = str(column_name).strip()

        # Extract grade level patterns
        grade_patterns = {
            'Kinder': r'(?i)kinder',
            'Gr1to6': r'(?i)gr1to6|grade?\s*1\s*to\s*6|elementary',
            'JHS': r'(?i)jhs|junior\s*high',
            'SHS': r'(?i)shs|senior\s*high'
        }

        grade_level = None
        for grade, pattern in grade_patterns.items():
            if re.search(pattern, clean_name):
                grade_level = grade
                break

        # If no grade level found, try to infer from position or return None
        if grade_level is None:
            logger.warning(f"Could not determine grade level for column: {column_name}")
            return None

        # Extract furniture type (remove grade level part)
        furniture_type = clean_name
        for grade, pattern in grade_patterns.items():
            furniture_type = re.sub(pattern, '', furniture_type, flags=re.IGNORECASE)

        # Clean up furniture type
        furniture_type = re.sub(r'[^\w\s]', ' ', furniture_type).strip()
        furniture_type = re.sub(r'\s+', ' ', furniture_type)

        if not furniture_type:
            furniture_type = 'Unknown'

        return {
            'grade_level': grade_level,
            'furniture_type': furniture_type
        }

    def wide_to_long(self) -> pd.DataFrame:
        """
        Transform furniture data from wide to long format.

        Extracts furniture counts from columns I-X and creates a long format DataFrame
        with columns: school_id, raw_grade_level, furniture_type, furniture_count

        Returns:
            Long format DataFrame with furniture counts
        """
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        logger.info("Transforming furniture data from wide to long format")

        # Get the School ID column (column G, index 6)
        school_id_col = 'School ID'

        # Verify the column exists
        if school_id_col not in self.raw_data.columns:
            raise ValueError(f"Column '{school_id_col}' not found in data. Available columns: {list(self.raw_data.columns)}")

        # Furniture count columns are from I to X (indices 8-23)
        furniture_start_idx = 8
        furniture_end_idx = min(24, len(self.raw_data.columns))  # X is column 23, so up to 24

        # Prepare list to store long format records
        long_records = []

        for index, row in self.raw_data.iterrows():
            school_id = row[school_id_col]

            # Skip rows with missing school ID
            if pd.isna(school_id):
                continue

            # Extract furniture counts for each column
            for col_idx in range(furniture_start_idx, furniture_end_idx):
                if col_idx < len(self.raw_data.columns):
                    column_name = self.raw_data.columns[col_idx]
                    furniture_count = row.iloc[col_idx]

                    # Parse column name to get grade level and furniture type
                    parsed_info = self._parse_furniture_column(column_name)

                    if parsed_info is None:
                        continue

                    # Convert to numeric, handle non-numeric values
                    try:
                        furniture_count = pd.to_numeric(furniture_count, errors='coerce')

                        # Only include valid, non-negative furniture counts
                        if pd.notna(furniture_count) and furniture_count >= 0:
                            long_records.append({
                                'school_id': str(school_id).strip(),
                                'raw_grade_level': parsed_info['grade_level'],
                                'furniture_type': parsed_info['furniture_type'],
                                'furniture_count': int(furniture_count)
                            })
                    except (ValueError, TypeError):
                        # Skip invalid furniture count values
                        continue

        # Create long format DataFrame
        self.processed_data = pd.DataFrame(long_records)

        # Trim whitespaces from string columns
        self.processed_data = self._trim_whitespaces(self.processed_data)

        # Transform data types
        self.processed_data['school_id'] = self.processed_data['school_id'].astype('string')

        # Transform grade_level to categorical with proper ordering
        grade_order = ['Kinder', 'Gr1to6', 'JHS', 'SHS']
        self.processed_data['raw_grade_level'] = pd.Categorical(
            self.processed_data['raw_grade_level'],
            categories=grade_order,
            ordered=True
        )

        # Retain rows with furniture count > 0
        self.processed_data = self.processed_data[self.processed_data['furniture_count'] > 0]

        # Apply arbitrary rule of DepEd EMISD on "Desks"
        self.transform_furnitures_values()

        # Relabel raw_grade_level to match grade_level of other datasets
        self.transform_raw_grade_level()

        logger.info(f"Created long format data with {len(self.processed_data)} records")
        logger.info(f"Grade levels: {self.processed_data['grade_level'].unique().tolist()}")
        logger.info(f"Furniture types: {self.processed_data['furniture_type'].unique().tolist()}")

        return self.processed_data

    def transform_furnitures_values(self, parameters: Optional[dict] = None) -> pd.DataFrame:
        baseline_parameters = {
            'Desks': 2,
            'Sets of Chairs and Tables': 1,
            'Arm Chairs': 1,
            'Others': 1
        }
        parameters = parameters if parameters else baseline_parameters

        def _integrate_furniture_parameters(row):
            furniture_type = row['furniture_type']
            furniture_count = row['furniture_count']

            return parameters[furniture_type] * furniture_count
        
        self.processed_data['alt_furniture_counts'] = self.processed_data.apply(_integrate_furniture_parameters, axis=1)

        return self.processed_data

    def transform_raw_grade_level(self) -> pd.DataFrame:
        self.processed_data['grade_level'] = (
            self.processed_data['raw_grade_level']
            .replace(
                {
                    'Kinder':'Elementary',
                    'Gr1to6':'Elementary',
                    'JHS':'Junior High School',
                    'SHS':'Senior High School',
                }
            )
        )
        return self.processed_data
    
    def process(self) -> pd.DataFrame:
        """
        Main processing pipeline: load data and transform to long format.

        Returns:
            Long format DataFrame with furniture counts
        """
        logger.info("Starting private furniture data processing")

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
            'grade_levels': self.processed_data['raw_grade_level'].unique().tolist(),
            'furniture_types': self.processed_data['furniture_type'].unique().tolist(),
            'total_furniture_count': self.processed_data['furniture_count'].sum(),
            'furniture_statistics': {
                'mean': self.processed_data['furniture_count'].mean(),
                'median': self.processed_data['furniture_count'].median(),
                'min': self.processed_data['furniture_count'].min(),
                'max': self.processed_data['furniture_count'].max()
            },
            'furniture_by_grade': self.processed_data.groupby('raw_grade_level')['furniture_count'].sum().to_dict(),
            'furniture_by_type': self.processed_data.groupby('furniture_type')['furniture_count'].sum().to_dict()
        }

        return summary

    def filter_by_grade_level(self, levels: List[str]) -> pd.DataFrame:
        """
        Filter data by grade levels.

        Args:
            levels: List of grade levels to include

        Returns:
            Filtered DataFrame
        """
        if self.processed_data is None:
            raise ValueError("Data not processed. Call process() first.")

        filtered_data = self.processed_data[
            self.processed_data['raw_grade_level'].isin(levels)
        ].copy()

        logger.info(f"Filtered to {len(filtered_data)} records for grade levels: {levels}")
        return filtered_data

    def filter_by_furniture_type(self, furniture_types: List[str]) -> pd.DataFrame:
        """
        Filter data by furniture types.

        Args:
            furniture_types: List of furniture types to include

        Returns:
            Filtered DataFrame
        """
        if self.processed_data is None:
            raise ValueError("Data not processed. Call process() first.")

        filtered_data = self.processed_data[
            self.processed_data['furniture_type'].isin(furniture_types)
        ].copy()

        logger.info(f"Filtered to {len(filtered_data)} records for furniture types: {furniture_types}")
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
    processor = PrivateFurnitureProcessor()

    # Process data to long format
    long_data = processor.process()

    # Get summary
    summary = processor.get_summary()
    print("\nPrivate Furniture Data Summary:")
    print(f"Total records: {summary['total_records']:,}")
    print(f"Unique schools: {summary['unique_schools']:,}")
    print(f"Total furniture count: {summary['total_furniture_count']:,}")
    print(f"Grade levels: {summary['raw_grade_levels']}")
    print(f"Furniture types: {summary['furniture_types']}")

    # Show sample data
    print("\nSample data:")
    print(long_data.head(10))

    # Export processed data
    processor.export_processed("output/private_furniture_long_format.csv")