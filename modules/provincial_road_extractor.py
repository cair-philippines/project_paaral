"""
Provincial Road Network Extractor

Extracts provincial road networks from OpenStreetMap PBF files using PyOsmium.
Uses adm2_pcode from consolidated geodata for reliable province identification.

Key Features:
- Memory-efficient streaming (processes 581MB PBF in ~3 minutes)
- Spatial indexing with Shapely STRtree for fast intersection queries
- LRU file handle cache to prevent "too many open files" errors
- Output: One .geojsonl file per province named {adm2_pcode}_{province_name}.geojsonl

Example Usage:
    from modules.provincial_road_extractor import ProvincialRoadExtractor

    extractor = ProvincialRoadExtractor(
        consolidated_geodata_path="output/consolidated_geodata_matched.gpkg",
        pbf_path="data/networks/philippines-251002.osm.pbf",
        output_dir="output/province_road_networks"
    )

    # Extract all provinces
    extractor.extract_all_provinces()

    # Or extract specific provinces
    extractor.extract_provinces(whitelist=["PH03014", "PH04021"])

Author: Claude Code
Date: 2025-10-07
"""

import os
import re
import json
import time
import logging
from pathlib import Path
from collections import OrderedDict
from typing import Optional, Set, Dict, List

import osmium
import geopandas as gpd
from shapely.geometry import LineString, mapping
from shapely.prepared import prep
from shapely.strtree import STRtree

# Setup logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# =============================================================================
# Helper Functions
# =============================================================================

def slugify(text: str) -> str:
    """
    Convert text to URL-friendly slug (lowercase, hyphens, no special chars).

    Args:
        text: Input text to slugify

    Returns:
        Slugified string

    Example:
        >>> slugify("City of Manila")
        'city-of-manila'
    """
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-{2,}", "-", text)
    return text or "unknown"


def write_feature_line(fh, geom, props: dict):
    """
    Write one GeoJSON Feature (LineString or MultiLineString) to a .geojsonl file.

    Args:
        fh: Open file handle
        geom: Shapely geometry (LineString or MultiLineString)
        props: Feature properties dictionary
    """
    feat = {
        "type": "Feature",
        "geometry": mapping(geom),
        "properties": props or {}
    }
    fh.write(json.dumps(feat, ensure_ascii=False) + "\n")


# =============================================================================
# LRU File Handle Cache
# =============================================================================

class LRUWriters:
    """
    LRU cache for open file handles to prevent "too many open files" errors.

    When writing to many provinces simultaneously, this class keeps only a small
    number of files open at once, automatically closing least recently used files.

    Args:
        out_dir: Output directory for .geojsonl files
        max_open: Maximum number of simultaneously open file handles
    """

    def __init__(self, out_dir: str, max_open: int = 16):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.max_open = max_open
        self._open = OrderedDict()  # filename -> file handle

    def get(self, filename: str):
        """
        Get file handle for given filename, opening it if needed.

        Args:
            filename: Name of file (without directory path)

        Returns:
            Open file handle
        """
        if filename in self._open:
            # Move to end (most recently used)
            fh = self._open.pop(filename)
            self._open[filename] = fh
            return fh

        # Need to open a new handle
        if len(self._open) >= self.max_open:
            # Close least recently used
            old_filename, old_fh = self._open.popitem(last=False)
            try:
                old_fh.flush()
                old_fh.close()
            except Exception as e:
                logger.warning(f"Error closing {old_filename}: {e}")

        path = os.path.join(self.out_dir, filename)
        fh = open(path, "a", buffering=1024*1024, encoding="utf-8")  # 1MB buffer
        self._open[filename] = fh
        return fh

    def close_all(self):
        """Close all open file handles."""
        for filename, fh in list(self._open.items()):
            try:
                fh.flush()
                fh.close()
            except Exception as e:
                logger.warning(f"Error closing {filename}: {e}")
        self._open.clear()


# =============================================================================
# Province Data Loader
# =============================================================================

def load_provinces(geodata_path: str, verbose: bool = True) -> Dict:
    """
    Load and aggregate barangay data to province level using adm2_pcode.

    This function:
    1. Reads barangay-level consolidated geodata
    2. Aggregates geometries by adm2_pcode (province pcode)
    3. Builds spatial index for fast intersection queries
    4. Creates filename mapping: adm2_pcode -> {pcode}_{name}.geojsonl

    Args:
        geodata_path: Path to consolidated_geodata file (.gpkg, .shp, etc.)
        verbose: If True, log INFO messages. If False, log only WARNING+

    Returns:
        Dictionary containing:
            - gdf: Province-level GeoDataFrame
            - geoms: List of province geometries
            - pcodes: List of adm2_pcode values
            - names: List of adm2_en province names
            - filenames: List of output filenames
            - prepared: List of prepared geometries for fast intersection
            - tree: STRtree spatial index
            - id_to_idx: Mapping from geometry id to index
            - wkb_to_idx: Mapping from geometry WKB to index
    """
    if verbose:
        logger.info(f"Loading consolidated geodata from {geodata_path}")

    # Read and ensure WGS84
    gdf = gpd.read_file(geodata_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    else:
        gdf = gdf.to_crs(4326)

    # Validate required columns
    required_cols = ['adm2_pcode', 'adm2_en', 'geometry']
    missing = [col for col in required_cols if col not in gdf.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Clean geometries before dissolve
    if verbose:
        logger.info("Cleaning geometries...")
    try:
        from shapely.validation import make_valid
        gdf["geometry"] = gdf.geometry.map(make_valid)
    except Exception:
        gdf["geometry"] = gdf.geometry.buffer(0)

    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()].copy()

    # Aggregate to province level by adm2_pcode
    if verbose:
        logger.info("Aggregating barangays to provinces by adm2_pcode...")

    # Get most common adm2_en for each adm2_pcode
    province_names = (
        gdf.groupby('adm2_pcode')['adm2_en']
        .apply(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
        .to_dict()
    )

    # Dissolve geometries by adm2_pcode
    dissolved = gdf[['adm2_pcode', 'geometry']].dissolve(
        by='adm2_pcode',
        as_index=False,
        aggfunc='first'
    )

    # Fix any invalids after dissolve
    try:
        from shapely.validation import make_valid
        dissolved["geometry"] = dissolved.geometry.map(make_valid)
    except Exception:
        dissolved["geometry"] = dissolved.geometry.buffer(0)

    dissolved = dissolved[~dissolved.geometry.is_empty & dissolved.geometry.notnull()].copy()

    # Add province names
    dissolved['adm2_en'] = dissolved['adm2_pcode'].map(province_names)

    # Generate filenames: {adm2_pcode}_{slugified_name}.geojsonl
    dissolved['filename'] = (
        dissolved['adm2_pcode'].astype(str) + '_' +
        dissolved['adm2_en'].apply(slugify) + '.geojsonl'
    )

    # Build spatial index structures
    geoms = list(dissolved.geometry.values)
    pcodes = list(dissolved['adm2_pcode'].astype(str))
    names = list(dissolved['adm2_en'].astype(str))
    filenames = list(dissolved['filename'].astype(str))

    prepared = [prep(g) for g in geoms]
    tree = STRtree(geoms)
    id_to_idx = {id(g): i for i, g in enumerate(geoms)}
    wkb_to_idx = {g.wkb: i for i, g in enumerate(geoms)}

    if verbose:
        logger.info(f"Loaded {len(dissolved)} provinces")
        logger.info(f"Sample filenames: {filenames[:3]}")

    return {
        "gdf": dissolved,
        "geoms": geoms,
        "pcodes": pcodes,
        "names": names,
        "filenames": filenames,
        "prepared": prepared,
        "tree": tree,
        "id_to_idx": id_to_idx,
        "wkb_to_idx": wkb_to_idx,
    }


# =============================================================================
# Osmium Streaming Handler
# =============================================================================

class DriveHandler(osmium.SimpleHandler):
    """
    Streaming OSM handler that writes driving roads to per-province .geojsonl files.

    Processes OSM ways one at a time, performs spatial intersection with province
    boundaries, and writes roads to appropriate province files.

    Args:
        prov: Province data dictionary from load_provinces()
        writers: LRUWriters instance for file handle management
        drive_highways: Set of highway tags considered driveable
        do_clip: If True, clip roads at province boundaries (slower, more RAM)
        whitelist: Optional set of adm2_pcode values to restrict processing
    """

    def __init__(self, prov: Dict, writers: LRUWriters,
                 drive_highways: Set[str], do_clip: bool = False,
                 whitelist: Optional[Set[str]] = None):
        super().__init__()

        # Province data
        self.geoms = prov["geoms"]
        self.prepared = prov["prepared"]
        self.tree = prov["tree"]
        self.pcodes = prov["pcodes"]
        self.filenames = prov["filenames"]
        self.id_to_idx = prov["id_to_idx"]
        self.wkb_to_idx = prov["wkb_to_idx"]

        # Configuration
        self.writers = writers
        self.drive_hw = set(drive_highways)
        self.do_clip = bool(do_clip)
        self.whitelist = set(whitelist) if whitelist else None

        # Statistics
        self.counts = {}  # pcode -> feature count

        # Capability detection for Shapely version compatibility
        self._has_query_items = hasattr(self.tree, "query_items")
        self._has_query_bulk = hasattr(self.tree, "query_bulk")

    def _idx_of(self, geom_obj):
        """Map STRtree-returned geometry object (or int) back to our index."""
        if isinstance(geom_obj, int):
            return geom_obj
        return self.id_to_idx.get(id(geom_obj), self.wkb_to_idx.get(geom_obj.wkb))

    def _candidate_indices(self, line: LineString) -> List[int]:
        """
        Find province indices that intersect the given LineString.
        Handles Shapely 1.x and 2.x API differences.

        Args:
            line: Road geometry

        Returns:
            List of province indices that intersect the road
        """
        # Fast path: Shapely 2.x with query_items -> indices directly
        if self._has_query_items:
            idxs = self.tree.query_items(line, predicate="intersects")
            return idxs.tolist() if hasattr(idxs, "tolist") else list(idxs)

        # Next best: query_bulk with single geometry
        if self._has_query_bulk:
            pairs = self.tree.query_bulk([line], predicate="intersects")
            cand = pairs[1]  # second row contains indices
            # De-duplicate while preserving order
            if hasattr(cand, "tolist"):
                out = []
                seen = set()
                for i in cand.tolist():
                    if i not in seen:
                        seen.add(i)
                        out.append(i)
                return out
            return list(dict.fromkeys(cand))

        # Fallback: query returns geometry objects; map to indices
        cands = self.tree.query(line)
        c_list = cands.tolist() if hasattr(cands, "tolist") else list(cands)
        out = []
        for g in c_list:
            idx = self._idx_of(g)
            if idx is None:
                continue
            # Precise intersection test (no predicate in fallback)
            if self.prepared[idx].intersects(line):
                out.append(idx)
        return out

    def way(self, w):
        """
        Process one OSM way. Called by osmium for each way in the PBF file.

        Args:
            w: osmium.osm.Way object
        """
        # Tag prefilter
        h = w.tags.get("highway")
        if not h or h not in self.drive_hw:
            return

        # Build geometry (requires locations=True in apply_file)
        if len(w.nodes) < 2:
            return
        try:
            coords = [(n.lon, n.lat) for n in w.nodes]
        except Exception:
            return
        if len(coords) < 2:
            return
        line = LineString(coords)

        # Find intersecting provinces
        idxs_list = self._candidate_indices(line)
        if len(idxs_list) == 0:
            return

        # Write to each intersecting province
        for idx in idxs_list:
            pcode = self.pcodes[idx]

            # Whitelist filter
            if self.whitelist and pcode not in self.whitelist:
                continue

            # Optional clipping
            geom_out = line if not self.do_clip else line.intersection(self.geoms[idx])
            if geom_out.is_empty:
                continue

            # Prepare feature properties
            props = {
                "osm_id": int(w.id),
                "highway": h,
                "name": w.tags.get("name"),
                "oneway": w.tags.get("oneway"),
                "maxspeed": w.tags.get("maxspeed"),
            }

            # Write to province file
            filename = self.filenames[idx]
            fh = self.writers.get(filename)
            write_feature_line(fh, geom_out, props)
            self.counts[pcode] = self.counts.get(pcode, 0) + 1


# =============================================================================
# Main Extractor Class
# =============================================================================

class ProvincialRoadExtractor:
    """
    Main interface for extracting provincial road networks from OSM PBF files.

    Args:
        consolidated_geodata_path: Path to consolidated geodata file
        pbf_path: Path to OSM PBF file
        output_dir: Directory for output .geojsonl files
        max_open_files: Maximum simultaneously open file handles
        do_clip: If True, clip roads at province boundaries
        verbose: If True, log INFO messages. If False, log only WARNING+

    Example:
        >>> extractor = ProvincialRoadExtractor(
        ...     consolidated_geodata_path="output/consolidated_geodata_matched.gpkg",
        ...     pbf_path="data/networks/philippines-251002.osm.pbf",
        ...     output_dir="output/province_road_networks",
        ...     verbose=True
        ... )
        >>> extractor.extract_all_provinces()
    """

    # Default highway tags for driving roads
    DRIVE_HIGHWAYS = {
        "motorway", "trunk", "primary", "secondary", "tertiary",
        "unclassified", "residential", "service", "road",
        "motorway_link", "trunk_link", "primary_link",
        "secondary_link", "tertiary_link"
    }

    def __init__(self, consolidated_geodata_path: str, pbf_path: str,
                 output_dir: str, max_open_files: int = 16, do_clip: bool = False,
                 verbose: bool = True):
        self.consolidated_geodata_path = consolidated_geodata_path
        self.pbf_path = pbf_path
        self.output_dir = output_dir
        self.max_open_files = max_open_files
        self.do_clip = do_clip
        self.verbose = verbose

        # Set logging level
        if verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)

        # Load province data
        self.prov = load_provinces(consolidated_geodata_path, verbose=verbose)

    def extract_provinces(self, whitelist: Optional[Set[str]] = None) -> Dict[str, int]:
        """
        Extract road networks for specified provinces (or all if whitelist is None).

        Args:
            whitelist: Optional set of adm2_pcode values to extract (e.g., {"PH03014", "PH04021"})
                      If None, extracts all provinces.

        Returns:
            Dictionary mapping adm2_pcode to number of roads extracted

        Example:
            >>> counts = extractor.extract_provinces(whitelist={"PH03014", "PH04021"})
            >>> print(counts)
            {'PH03014': 51760, 'PH04021': 81146}
        """
        if self.verbose:
            if whitelist:
                logger.info(f"Extracting {len(whitelist)} provinces: {sorted(whitelist)}")
            else:
                logger.info(f"Extracting all {len(self.prov['pcodes'])} provinces")

        start_time = time.time()

        # Initialize writer cache and handler
        writers = LRUWriters(self.output_dir, max_open=self.max_open_files)
        handler = DriveHandler(
            prov=self.prov,
            writers=writers,
            drive_highways=self.DRIVE_HIGHWAYS,
            do_clip=self.do_clip,
            whitelist=whitelist
        )

        # Process PBF file
        if self.verbose:
            logger.info(f"Processing PBF file: {self.pbf_path}")
        handler.apply_file(
            self.pbf_path,
            locations=True,
            idx="sparse_mmap_array"
        )

        # Cleanup
        writers.close_all()

        elapsed = time.time() - start_time
        if self.verbose:
            logger.info(f"Extraction complete in {elapsed/60:.2f} minutes")
            logger.info(f"Roads extracted per province:")
            for pcode in sorted(handler.counts.keys()):
                logger.info(f"  {pcode}: {handler.counts[pcode]:,} roads")

        return handler.counts

    def extract_all_provinces(self) -> Dict[str, int]:
        """
        Extract road networks for all provinces.

        Returns:
            Dictionary mapping adm2_pcode to number of roads extracted
        """
        return self.extract_provinces(whitelist=None)

    def get_province_list(self) -> List[Dict[str, str]]:
        """
        Get list of all provinces with their pcodes, names, and filenames.

        Returns:
            List of dictionaries with keys: pcode, name, filename
        """
        return [
            {
                "pcode": self.prov["pcodes"][i],
                "name": self.prov["names"][i],
                "filename": self.prov["filenames"][i]
            }
            for i in range(len(self.prov["pcodes"]))
        ]
