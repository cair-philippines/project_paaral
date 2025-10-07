"""
Intelligent renaming of province road network files using matched PSGC data.

Format: {RR-PPP}_{Province_Name}.geojsonl
Example: 10-013_Bukidnon.geojsonl
"""

import re
from pathlib import Path
import geopandas as gpd
import pandas as pd

# Paths
# Since this script is in modules/, go up one level to get project root
PROJECT_ROOT = Path(__file__).parent.parent
NETWORKS_DIR = PROJECT_ROOT / "output" / "province_road_networks"
GEODATA_FILE = PROJECT_ROOT / "output" / "consolidated_geodata_matched.gpkg"

def load_province_mappings():
    """Load province PSGC mappings from matched geodata."""
    print("Loading matched PSGC data...")
    gdf = gpd.read_file(GEODATA_FILE)

    # Extract unique provinces
    provinces = gdf[['adm1_psgc', 'adm2_psgc', 'adm1_en', 'adm2_en']].drop_duplicates()
    provinces = provinces[provinces['adm2_psgc'].notna()].copy()

    # Extract region code (first 2 digits) and province code (3-digit portion)
    provinces['region_code'] = provinces['adm2_psgc'].str[:2]
    provinces['province_code'] = provinces['adm2_psgc'].str[2:5]
    provinces['psgc_prefix'] = provinces['region_code'] + '-' + provinces['province_code']

    print(f"Loaded {len(provinces)} provinces")

    # Also keep the full GeoDataFrame for historical PSGC lookups
    return provinces, gdf

def slugify_province_name(name):
    """Convert province name to slug format for matching."""
    if pd.isna(name):
        return ""
    slug = name.lower()
    # Remove "City of", "Province of", etc.
    slug = re.sub(r'\b(city of|province of)\b', '', slug)
    # Replace spaces and special chars with hyphens
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[\s_]+', '-', slug)
    slug = slug.strip('-')
    return slug

def clean_province_name_for_filename(name):
    """Clean province name for use in filename."""
    if pd.isna(name):
        return "Unknown"
    # Remove "City of", "Province of" prefixes
    cleaned = re.sub(r'^(City of|Province of)\s+', '', name, flags=re.IGNORECASE)
    # Replace spaces with underscores
    cleaned = cleaned.replace(' ', '_')
    # Remove special characters except underscores and hyphens
    cleaned = re.sub(r'[^\w-]', '', cleaned)
    return cleaned

def parse_existing_filename(filename):
    """Extract province identifier from existing filename."""
    stem = Path(filename).stem  # Remove .geojsonl

    # Pattern 1: Already PSGC format (e.g., "09-097")
    psgc_match = re.match(r'^(\d{2})-(\d{3})$', stem)
    if psgc_match:
        return {
            'type': 'psgc',
            'region_code': psgc_match.group(1),
            'province_code': psgc_match.group(2),
            'psgc_prefix': stem
        }

    # Pattern 2: Slug format (e.g., "region-x-northern-mindanao-bukidnon")
    # Extract the province name (last part after region prefix)
    parts = stem.split('-')

    # Special handling for regions
    region_prefixes = [
        'bangsamoro-autonomous-region-in-muslim-mindanao-barmm',
        'cordillera-administrative-region-car',
        'national-capital-region-ncr',
        'mimaropa-region',
        'region-i-ilocos-region',
        'region-ii-cagayan-valley',
        'region-iii-central-luzon',
        'region-iv-a-calabarzon',
        'region-iv-b-mimaropa',
        'region-v-bicol-region',
        'region-vi-western-visayas',
        'region-vii-central-visayas',
        'region-viii-eastern-visayas',
        'region-ix-zamboanga-peninsula',
        'region-x-northern-mindanao',
        'region-xi-davao-region',
        'region-xii-soccsksargen',
        'region-xiii-caraga',
    ]

    province_name = None
    for prefix in sorted(region_prefixes, key=len, reverse=True):
        if stem.startswith(prefix):
            province_name = stem[len(prefix):].lstrip('-')
            break

    if not province_name:
        # Fallback: assume last part is province
        province_name = parts[-1] if parts else stem

    return {
        'type': 'slug',
        'province_slug': province_name,
        'full_slug': stem
    }

def match_historical_psgc(old_psgc_prefix, full_gdf, provinces_df):
    """
    Match old/historical PSGC codes by finding barangays with that prefix.

    This handles cases where province codes changed (e.g., Basilan: 09-097 → 15-070)
    """
    # Look for barangays whose psgc_code starts with the old prefix
    # For "09-097", look for psgc_code starting with "09097"
    search_pattern = old_psgc_prefix.replace('-', '')  # "09-097" → "09097"

    # Find barangays with this old PSGC prefix
    matching_barangays = full_gdf[
        full_gdf['psgc_code'].astype(str).str.startswith(search_pattern, na=False)
    ]

    if len(matching_barangays) > 0:
        # Get the current province name from these barangays
        province_names = matching_barangays['adm2_en'].dropna().unique()
        if len(province_names) > 0:
            province_name = province_names[0]
            # Find this province in the provinces_df
            match = provinces_df[provinces_df['adm2_en'] == province_name]
            if len(match) > 0:
                return match.iloc[0]

    return None

def match_filename_to_psgc(parsed, provinces_df, full_gdf=None):
    """Match parsed filename to PSGC data."""

    if parsed['type'] == 'psgc':
        # Try 1: Direct match by PSGC prefix (handles current codes)
        match = provinces_df[provinces_df['psgc_prefix'] == parsed['psgc_prefix']]
        if len(match) > 0:
            return match.iloc[0]

        # Try 2: Historical PSGC lookup using barangay data
        # Handles province reorganizations (e.g., Basilan: 09-097 → 15-070)
        if full_gdf is not None:
            match = match_historical_psgc(parsed['psgc_prefix'], full_gdf, provinces_df)
            if match is not None:
                return match

        # Try 3: Match by province code only (last resort)
        province_code = parsed['province_code']
        match = provinces_df[provinces_df['province_code'] == province_code]
        if len(match) > 0:
            return match.iloc[0]

    elif parsed['type'] == 'slug':
        # Match by province name slug
        provinces_df['province_slug'] = provinces_df['adm2_en'].apply(slugify_province_name)

        # Try exact match
        match = provinces_df[provinces_df['province_slug'] == parsed['province_slug']]
        if len(match) > 0:
            return match.iloc[0]

        # Try partial match (contains)
        match = provinces_df[provinces_df['province_slug'].str.contains(parsed['province_slug'], na=False)]
        if len(match) > 0:
            return match.iloc[0]

    return None

def generate_rename_plan():
    """Generate complete rename plan for all files."""
    provinces, full_gdf = load_province_mappings()

    # Get all geojsonl files
    files = sorted(NETWORKS_DIR.glob("*.geojsonl"))
    print(f"\nFound {len(files)} files to process\n")

    rename_plan = []

    for old_path in files:
        old_name = old_path.name
        parsed = parse_existing_filename(old_name)

        # Match to PSGC data (with historical lookup support)
        matched_province = match_filename_to_psgc(parsed, provinces, full_gdf)

        if matched_province is not None:
            # Generate new filename
            psgc_prefix = matched_province['psgc_prefix']
            province_name = clean_province_name_for_filename(matched_province['adm2_en'])
            new_name = f"{psgc_prefix}_{province_name}.geojsonl"

            rename_plan.append({
                'old_name': old_name,
                'new_name': new_name,
                'psgc_code': matched_province['adm2_psgc'],
                'province': matched_province['adm2_en'],
                'region': matched_province['adm1_en'],
                'status': 'OK' if old_name != new_name else 'NO_CHANGE'
            })
        else:
            rename_plan.append({
                'old_name': old_name,
                'new_name': old_name,
                'psgc_code': 'UNKNOWN',
                'province': 'NOT_FOUND',
                'region': 'NOT_FOUND',
                'status': 'NO_MATCH'
            })

    return pd.DataFrame(rename_plan)

def display_rename_preview(plan_df):
    """Display rename plan in organized format."""
    print("="*130)
    print("PROVINCE ROAD NETWORK FILES - RENAME PREVIEW")
    print("="*130)
    print(f"\nTotal files: {len(plan_df)}")
    print(f"Files to rename: {(plan_df['status'] == 'OK').sum()}")
    print(f"No change needed: {(plan_df['status'] == 'NO_CHANGE').sum()}")
    print(f"No match found: {(plan_df['status'] == 'NO_MATCH').sum()}")
    print("\n" + "="*130)
    print(f"{'#':>3} {'Status':^6} {'Current Filename':<72} {'New Filename':<50}")
    print("="*130)

    # Display all files
    for idx, row in plan_df.iterrows():
        status_symbol = "✓" if row['status'] == 'OK' else ("=" if row['status'] == 'NO_CHANGE' else "✗")
        print(f"{idx+1:3d}. {status_symbol:^6} {row['old_name']:<72} → {row['new_name']:<50}")

    print("="*130)
    print("\nLegend:")
    print("  ✓ = Will be renamed")
    print("  = = Already in correct format (no change)")
    print("  ✗ = No PSGC match found")
    print("="*130)

def execute_renames(plan_df, dry_run=True):
    """Execute the rename operations."""
    if dry_run:
        print("\n[DRY RUN MODE - No files will be renamed]")
        return

    print("\n[EXECUTING RENAMES...]")
    renamed_count = 0

    for idx, row in plan_df.iterrows():
        if row['status'] == 'OK':
            old_path = NETWORKS_DIR / row['old_name']
            new_path = NETWORKS_DIR / row['new_name']

            if old_path.exists() and not new_path.exists():
                old_path.rename(new_path)
                print(f"✓ Renamed: {row['old_name']} → {row['new_name']}")
                renamed_count += 1
            else:
                print(f"✗ Skipped: {row['old_name']} (file issue)")

    print(f"\n✓ Renamed {renamed_count} files successfully")

if __name__ == "__main__":
    import sys

    # Generate rename plan
    plan = generate_rename_plan()

    # Display preview
    display_rename_preview(plan)

    # Save plan to CSV for reference
    plan_file = PROJECT_ROOT / "province_networks_rename_plan.csv"
    plan.to_csv(plan_file, index=False)
    print(f"\n✓ Rename plan saved to: {plan_file}")

    # Check for --execute flag
    if "--execute" in sys.argv:
        print("\n" + "="*80)
        print("EXECUTING RENAME...")
        print("="*80 + "\n")
        execute_renames(plan, dry_run=False)
        print("\n✓ DONE!")
    else:
        print("\nTo execute the renaming, run:")
        print("  python rename_province_networks.py --execute")
