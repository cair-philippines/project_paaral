"""
Compact enrollment data preprocessor module.
Author: Data Processing System
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnrollmentDataProcessor:
    """Compact processor for enrollment data transformation from wide to long format."""

    def __init__(self, csv_path: Optional[str] = None):
        """Initialize processor with optional CSV path."""
        self.csv_path = csv_path or "data/public/Copy of SY 2023-2024 SCHOOL LEVEL DATA ON ENROLLMENT.csv"
        self.raw_data = None
        self.processed_data = None

    def load_data(self) -> pd.DataFrame:
        """Load and clean raw enrollment data."""
        try:
            # Read CSV, skip metadata rows (1-5), use row 6 as header
            self.raw_data = pd.read_csv(self.csv_path, skiprows=5)

            # Clean column names
            self.raw_data.columns = self.raw_data.columns.str.strip()

            # Remove any completely empty rows
            self.raw_data = self.raw_data.dropna(how='all')

            logger.info(f"Loaded {len(self.raw_data)} records with {len(self.raw_data.columns)} columns")
            return self.raw_data

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def wide_to_long(self) -> pd.DataFrame:
        """Transform wide format enrollment data to long format efficiently."""
        if self.raw_data is None:
            self.load_data()

        # School information columns (keep these as identifiers)
        school_info_cols = [
            'Region', 'Division', 'District', 'School ID', 'School Name',
            'Street Address', 'Province', 'Municipality', 'Legislative District',
            'Barangay', 'Sector', 'School Subclassification', 'School Type', 'Modified COC'
        ]

        # Filter existing school info columns
        existing_school_cols = [col for col in school_info_cols if col in self.raw_data.columns]

        # Get enrollment columns (exclude school info columns AND total columns to avoid double counting)
        enrollment_cols = [col for col in self.raw_data.columns
                          if col not in existing_school_cols
                          and not self._is_total_column(col)]

        # Melt the data
        long_data = pd.melt(
            self.raw_data,
            id_vars=existing_school_cols,
            value_vars=enrollment_cols,
            var_name='enrollment_category',
            value_name='enrollment_count'
        )

        # Parse enrollment category into components
        parsed_info = long_data['enrollment_category'].apply(self._parse_column_name)
        parsed_df = pd.DataFrame(parsed_info.tolist())

        # Combine with long data
        self.processed_data = pd.concat([
            long_data[existing_school_cols + ['enrollment_count']],
            parsed_df
        ], axis=1)

        # Clean data
        self.processed_data['enrollment_count'] = pd.to_numeric(
            self.processed_data['enrollment_count'], errors='coerce'
        ).fillna(0)

        # Remove rows with zero enrollment (optional - uncomment if needed)
        # self.processed_data = self.processed_data[self.processed_data['enrollment_count'] > 0]

        logger.info(f"Transformed to long format: {len(self.processed_data)} records")
        return self.processed_data

    def _parse_column_name(self, col_name: str) -> Dict[str, Any]:
        """Parse column name to extract grade, gender, and academic track."""
        col_name = str(col_name).strip()

        result = {
            'grade_level': None,
            'gender': None,
            'academic_track': None,
            'student_type': 'regular'
        }

        # Handle gender
        if 'Male' in col_name and 'Female' not in col_name:
            result['gender'] = 'Male'
        elif 'Female' in col_name:
            result['gender'] = 'Female'
        elif 'Total' in col_name or 'total' in col_name:
            result['gender'] = 'Total'

        # Handle grade levels (be more specific to avoid conflicts)
        if col_name.startswith('K ') or col_name == 'K Male' or col_name == 'K Female':
            result['grade_level'] = 'K'
        elif col_name.startswith('G1 ') or 'G1 Male' in col_name or 'G1 Female' in col_name:
            result['grade_level'] = 'G1'
        elif col_name.startswith('G2 ') or 'G2 Male' in col_name or 'G2 Female' in col_name:
            result['grade_level'] = 'G2'
        elif col_name.startswith('G3 ') or 'G3 Male' in col_name or 'G3 Female' in col_name:
            result['grade_level'] = 'G3'
        elif col_name.startswith('G4 ') or 'G4 Male' in col_name or 'G4 Female' in col_name:
            result['grade_level'] = 'G4'
        elif col_name.startswith('G5 ') or 'G5 Male' in col_name or 'G5 Female' in col_name:
            result['grade_level'] = 'G5'
        elif col_name.startswith('G6 ') or 'G6 Male' in col_name or 'G6 Female' in col_name:
            result['grade_level'] = 'G6'
        elif col_name.startswith('G7 ') or 'G7 Male' in col_name or 'G7 Female' in col_name:
            result['grade_level'] = 'G7'
        elif col_name.startswith('G8 ') or 'G8 Male' in col_name or 'G8 Female' in col_name:
            result['grade_level'] = 'G8'
        elif col_name.startswith('G9 ') or 'G9 Male' in col_name or 'G9 Female' in col_name:
            result['grade_level'] = 'G9'
        elif col_name.startswith('G10 ') or 'G10 Male' in col_name or 'G10 Female' in col_name:
            result['grade_level'] = 'G10'
        elif 'G11' in col_name and any(track in col_name for track in ['ABM', 'HUMSS', 'STEM', 'GAS', 'PBM', 'TVL', 'SPORTS', 'ARTS']):
            result['grade_level'] = 'G11'
        elif 'G12' in col_name and any(track in col_name for track in ['ABM', 'HUMSS', 'STEM', 'GAS', 'PBM', 'TVL', 'SPORTS', 'ARTS']):
            result['grade_level'] = 'G12'
        elif 'Elem NG' in col_name:
            result['grade_level'] = 'Elementary'
            result['student_type'] = 'SNEd'
        elif 'JHS NG' in col_name:
            result['grade_level'] = 'JHS'
            result['student_type'] = 'SNEd'

        # Handle academic tracks for SHS (be more specific)
        if 'G11' in col_name or 'G12' in col_name:
            if 'ACAD - ABM' in col_name:
                result['academic_track'] = 'ABM'
            elif 'ACAD - HUMSS' in col_name:
                result['academic_track'] = 'HUMSS'
            elif 'ACAD STEM' in col_name:
                result['academic_track'] = 'STEM'
            elif 'ACAD GAS' in col_name:
                result['academic_track'] = 'GAS'
            elif 'ACAD PBM' in col_name:
                result['academic_track'] = 'PBM'
            elif 'TVL' in col_name:
                result['academic_track'] = 'TVL'
            elif 'SPORTS' in col_name:
                result['academic_track'] = 'SPORTS'
            elif 'ARTS' in col_name:
                result['academic_track'] = 'ARTS & DESIGN'

        return result

    def _is_total_column(self, col_name: str) -> bool:
        """Check if column represents a total or aggregate (to avoid double counting)."""
        col_name = str(col_name).strip()
        col_lower = col_name.lower()

        # These are columns to exclude (totals and aggregates that sum other columns)
        # Be very specific to avoid excluding valid individual enrollment columns
        exclude_patterns = [
            # Exact matches for total columns
            'Ktotal', 'G1to6 Total', 'Kto6 Total', 'JHS Total',
            'G11ACAD Total', 'G11Total Total', 'G12ACAD Total', 'G12Total Total',
            'SHS Total', 'Kto12 Total',

            # Exact matches for aggregate columns (sums of other individual columns)
            'G1to6 Male', 'G1to6 Female',  # Sum of G1-G6 individual grades
            'Kto6 Male', 'Kto6 Female',    # Sum of K + G1-G6
            'JHS Male', 'JHS Female',      # Sum of G7-G10 individual grades
            'G11ACAD Male', 'G11ACAD Female',  # Sum of G11 academic tracks
            'G11Total Male', 'G11Total Female',  # Sum of all G11 tracks
            'G12ACAD Male', 'G12ACAD Female',  # Sum of G12 academic tracks
            'G12Total Male', 'G12Total Female',  # Sum of all G12 tracks
            'SHSTotal Male', 'SHSTotal Female',  # Sum of all SHS
            'Kto12 Male', 'Kto12 Female'   # Sum of all K-12
        ]

        return col_name in exclude_patterns

    def process(self) -> pd.DataFrame:
        """Main processing pipeline."""
        self.load_data()
        return self.wide_to_long()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of processed data."""
        if self.processed_data is None:
            logger.warning("No processed data available. Run process() first.")
            return {}

        return {
            'total_records': len(self.processed_data),
            'total_enrollment': self.processed_data['enrollment_count'].sum(),
            'unique_schools': self.processed_data['School ID'].nunique() if 'School ID' in self.processed_data else 0,
            'grade_levels': self.processed_data['grade_level'].value_counts().to_dict(),
            'academic_tracks': self.processed_data['academic_track'].value_counts().to_dict(),
            'gender_distribution': self.processed_data['gender'].value_counts().to_dict()
        }

    def filter_by_grade(self, grades: List[str]) -> pd.DataFrame:
        """Filter processed data by specific grade levels."""
        if self.processed_data is None:
            logger.warning("No processed data available. Run process() first.")
            return pd.DataFrame()

        return self.processed_data[self.processed_data['grade_level'].isin(grades)]

    def export_processed(self, output_path: str = 'output/enrollment_long_format.csv'):
        """Export processed data to CSV."""
        if self.processed_data is None:
            logger.warning("No processed data available. Run process() first.")
            return

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        self.processed_data.to_csv(output_path, index=False)
        logger.info(f"Exported processed data to {output_path}")


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = EnrollmentDataProcessor()

    # Process data
    long_data = processor.process()

    # Get summary
    summary = processor.get_summary()
    print("Data Summary:", summary)

    # Export processed data
    processor.export_processed()