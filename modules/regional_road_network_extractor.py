"""
Regional Road Network Extractor Module

Extracts drive road networks from OpenStreetMap using OSMNx with province-level
querying to handle archipelagic regions reliably.

Author: Claude Code
Date: 2025-10-01

Examples
--------
Basic Setup
~~~~~~~~~~~
>>> from modules.psgc_consolidator import PSGCConsolidator
>>> from modules.regional_road_network_extractor import RegionalRoadNetworkExtractor
>>>
>>> # Load PSGC geographic data
>>> consolidator = PSGCConsolidator(
...     base_dir=r"C:\path\to\philippines-psgc-shapefiles\dist",
...     verbose=False
... )
>>> gdf = consolidator.load_complete_data()
>>>
>>> # Initialize extractor
>>> extractor = RegionalRoadNetworkExtractor(gdf, verbose=True)

List Available Regions and Provinces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
>>> # Get all regions
>>> regions = extractor.get_region_list()
>>> print(regions)
   region_code              region_name
0           01  Region I (Ilocos Region)
1           02      Region II (Cagayan Valley)
...

>>> # Get provinces in a specific region
>>> provinces = extractor.get_province_list('07')  # Central Visayas
>>> print(provinces)
   region_code province_code              region_name province_name
0           07          0722  Region VII (Central Visayas)         Cebu
1           07          0746  Region VII (Central Visayas)      Bohol
...

Query Region - Province Breakdown (RECOMMENDED)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
>>> # Query region using province-level breakdown (complete coverage)
>>> G, meta = extractor.query_region('07')  # Central Visayas
>>>
>>> # Print summary
>>> extractor.print_summary(meta)
============================================================
QUERY SUMMARY
============================================================
Region: Region VII (Central Visayas) (07)
Query Method: province_breakdown
Provinces: 4/4 successful

Network Statistics:
  Nodes: 45,231
  Edges: 98,456
...

Query Region - Direct Method (Faster)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
>>> # Query entire region directly (faster but may miss islands)
>>> G_direct, meta_direct = extractor.query_region(
...     '07',
...     use_province_breakdown=False  # Direct query
... )
>>>
>>> # Compare results
>>> print(f"Province breakdown: {meta['nodes']:,} nodes")
>>> print(f"Direct query: {meta_direct['nodes']:,} nodes")
Province breakdown: 45,231 nodes
Direct query: 42,100 nodes  # May be incomplete for archipelagic regions

Query Single Province
~~~~~~~~~~~~~~~~~~~~~
>>> # Query specific province
>>> G_cebu, meta_cebu = extractor.query_province('0722')  # Cebu
>>> # Or by name
>>> G_cebu, meta_cebu = extractor.query_province('Cebu')
>>>
>>> extractor.print_summary(meta_cebu)

Simple Network Plot
~~~~~~~~~~~~~~~~~~~
>>> # Plot network using OSMnx native methods
>>> fig, ax = extractor.plot_graph(
...     G,
...     figsize=(15, 15),
...     edge_color='red',
...     edge_linewidth=0.5,
...     save_path='region07_network.png',
...     show=True
... )

Plot Network with Boundary Overlay
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
>>> # Overlay road network on region shapefile
>>> fig, ax = extractor.plot_graph_with_boundary(
...     G,
...     region_or_province_code='07',  # Can use region or province code
...     figsize=(15, 15),
...     edge_color='red',
...     edge_linewidth=0.8,
...     boundary_facecolor='lightblue',
...     boundary_alpha=0.2,
...     save_path='region07_with_boundary.png',
...     show=True
... )

Advanced Query Options
~~~~~~~~~~~~~~~~~~~~~~
>>> # Query with custom parameters
>>> G, meta = extractor.query_province(
...     'Palawan',
...     network_type='drive',           # 'drive', 'walk', 'bike', 'all'
...     buffer_meters=100,              # Buffer to catch border roads
...     simplify_tolerance=0.0001,      # Simplify complex coastlines
...     decompose_islands=True          # Query each island separately
... )

Custom Plot Styling
~~~~~~~~~~~~~~~~~~~
>>> # Customize plot appearance
>>> fig, ax = extractor.plot_graph_with_boundary(
...     G,
...     region_or_province_code='07',
...     figsize=(20, 20),
...     edge_color='#FF6B6B',           # Custom color
...     edge_linewidth=1.0,
...     edge_alpha=0.9,
...     boundary_color='#2C3E50',
...     boundary_linewidth=2.0,
...     boundary_facecolor='#ECF0F1',
...     boundary_alpha=0.3,
...     dpi=300,                        # High resolution
...     save_path='custom_styled.png'
... )

Batch Processing Multiple Regions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
>>> # Process all regions
>>> all_regions = extractor.get_region_list()
>>>
>>> for _, row in all_regions.iterrows():
...     region_code = row['region_code']
...     region_name = row['region_name']
...
...     print(f"Processing {region_name}...")
...
...     # Query region
...     G, meta = extractor.query_region(region_code)
...
...     # Save plot
...     extractor.plot_graph_with_boundary(
...         G,
...         region_or_province_code=region_code,
...         save_path=f'output/region_{region_code}.png',
...         show=False
...     )
...
...     print(f"  → {meta['nodes']:,} nodes, {meta['edges']:,} edges")

Compare Query Methods
~~~~~~~~~~~~~~~~~~~~~
>>> # Compare province breakdown vs direct query
>>> print("Testing both methods on Central Visayas...")
>>>
>>> # Method 1: Province breakdown
>>> G1, meta1 = extractor.query_region('07', use_province_breakdown=True)
>>>
>>> # Method 2: Direct query
>>> G2, meta2 = extractor.query_region('07', use_province_breakdown=False)
>>>
>>> # Compare
>>> print(f"\\nProvince Breakdown:")
>>> print(f"  Nodes: {meta1['nodes']:,}")
>>> print(f"  Edges: {meta1['edges']:,}")
>>> print(f"  Query Method: {meta1['query_method']}")
>>>
>>> print(f"\\nDirect Query:")
>>> print(f"  Nodes: {meta2['nodes']:,}")
>>> print(f"  Edges: {meta2['edges']:,}")
>>> print(f"  Query Method: {meta2['query_method']}")
>>>
>>> print(f"\\nDifference: {abs(meta1['nodes'] - meta2['nodes']):,} nodes")

Cache Usage
~~~~~~~~~~~
>>> # First query - fetches from OSM
>>> G1, meta1 = extractor.query_province('Cebu')  # Queries OSM API
>>>
>>> # Second query - uses cache (instant)
>>> G2, meta2 = extractor.query_province('Cebu')  # Returns cached result
>>>
>>> # Clear cache if needed
>>> extractor.cache.clear()

Export Graph Data
~~~~~~~~~~~~~~~~~
>>> # Convert graph to GeoDataFrame for further analysis
>>> import osmnx as ox
>>> nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
>>>
>>> # Save to file
>>> edges_gdf.to_file('region07_roads.shp')
>>> edges_gdf.to_file('region07_roads.geojson', driver='GeoJSON')
>>>
>>> # Or save as GraphML
>>> ox.save_graphml(G, 'region07_graph.graphml')

Notes
-----
- **Province Breakdown**: Recommended for archipelagic regions (MIMAROPA, Central Visayas)
  to ensure complete coverage. Queries each province separately then merges.

- **Direct Query**: Faster but may miss islands in multi-polygon regions. Best for
  contiguous areas or when speed is prioritized over completeness.

- **Caching**: Results are automatically cached. Repeated queries return instantly.

- **Island Decomposition**: When `decompose_islands=True`, MultiPolygon geometries
  are split into individual islands for querying, improving completeness.

- **Visualization**: OSMnx native plotting is used for compatibility with ARM devices
  (e.g., Surface Pro 11) and provides clean, professional visualizations.
"""

import logging
import geopandas as gpd
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon, Polygon
from typing import Tuple, List, Dict, Optional, Union
import pandas as pd

# Try to import igraph for conversion
try:
    import igraph as ig
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False


class RegionalRoadNetworkExtractor:
    """
    Extract drive road networks for Philippine regions using province-level
    querying to ensure complete coverage in archipelagic areas.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame with Philippine geographic hierarchy (from psgc_consolidator)
    verbose : bool, default True
        Enable verbose logging (INFO level) or suppress (WARNING only)

    Attributes
    ----------
    gdf : GeoDataFrame
        Input geographic data
    logger : Logger
        Configured logger instance
    cache : dict
        Cache for province-level queries to avoid redundant API calls
    """

    def __init__(self, gdf: gpd.GeoDataFrame, verbose: bool = True):
        """Initialize the extractor with geographic data."""
        self.gdf = gdf
        self.verbose = verbose
        self.cache = {}

        # Configure logging
        self.logger = logging.getLogger(__name__)
        log_level = logging.INFO if verbose else logging.WARNING
        self.logger.setLevel(log_level)

        # Remove existing handlers to avoid duplicates
        self.logger.handlers = []

        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # Configure OSMNx settings
        ox.settings.log_console = verbose
        ox.settings.use_cache = True

        self.logger.info("Regional Road Network Extractor initialized")
        self.logger.info(f"Input GeoDataFrame: {len(gdf)} features")

    def _extract_region_code(self, psgc_code: str) -> str:
        """
        Extract region code (first 2 digits) from PSGC code.

        Parameters
        ----------
        psgc_code : str
            10-digit PSGC code

        Returns
        -------
        str
            2-digit region code
        """
        psgc_str = str(psgc_code).zfill(10)
        return psgc_str[:2]

    def _extract_province_code(self, psgc_code: str) -> str:
        """
        Extract province code (first 4 digits) from PSGC code.

        Parameters
        ----------
        psgc_code : str
            10-digit PSGC code

        Returns
        -------
        str
            4-digit province code
        """
        psgc_str = str(psgc_code).zfill(10)
        return psgc_str[:4]

    def get_region_list(self) -> pd.DataFrame:
        """
        Get list of available regions using first 2 digits of PSGC codes.

        Returns
        -------
        DataFrame
            Regions with codes and names
        """
        # Extract region codes from any PSGC column
        df = self.gdf.copy()
        df['region_code'] = df['adm1_psgc'].astype(str).str.zfill(10).str[:2]

        # Get unique regions
        regions = df[['region_code', 'adm1_en']].drop_duplicates()
        regions = regions.sort_values('region_code').reset_index(drop=True)
        regions.columns = ['region_code', 'region_name']

        self.logger.info(f"Available regions: {len(regions)}")
        return regions

    def get_province_list(self, region_name_or_code: Optional[str] = None) -> pd.DataFrame:
        """
        Get list of available provinces using first 4 digits of PSGC codes,
        optionally filtered by region (first 2 digits).

        Parameters
        ----------
        region_name_or_code : str, optional
            Region name or 2-digit region code to filter provinces

        Returns
        -------
        DataFrame
            Provinces with codes and names
        """
        # Extract codes
        df = self.gdf.copy()
        df['region_code'] = df['adm2_psgc'].astype(str).str.zfill(10).str[:2]
        df['province_code'] = df['adm2_psgc'].astype(str).str.zfill(10).str[:4]

        provinces = df[['region_code', 'province_code', 'adm1_en', 'adm2_en']].drop_duplicates()

        if region_name_or_code:
            # Filter by region code (2 digits) or region name
            region_code_input = str(region_name_or_code).zfill(2) if region_name_or_code.isdigit() and len(region_name_or_code) <= 2 else None

            mask = (
                (provinces['adm1_en'].str.contains(region_name_or_code, case=False, na=False)) |
                (provinces['region_code'] == region_code_input if region_code_input else False)
            )
            provinces = provinces[mask]

        provinces = provinces.sort_values('province_code').reset_index(drop=True)
        provinces.columns = ['region_code', 'province_code', 'region_name', 'province_name']

        self.logger.info(f"Available provinces: {len(provinces)}")
        return provinces

    def _decompose_multipolygon(self, geometry: Union[Polygon, MultiPolygon]) -> List[Polygon]:
        """
        Decompose MultiPolygon into individual Polygon components.

        Parameters
        ----------
        geometry : Polygon or MultiPolygon
            Input geometry

        Returns
        -------
        list of Polygon
            Individual polygon components
        """
        if isinstance(geometry, MultiPolygon):
            polygons = list(geometry.geoms)
            self.logger.debug(f"Decomposed MultiPolygon into {len(polygons)} polygons")
            return polygons
        else:
            return [geometry]

    def query_province(
        self,
        province_name_or_code: str,
        network_type: str = 'drive',
        buffer_meters: float = 100,
        simplify_tolerance: float = 0.0001,
        decompose_islands: bool = True
    ) -> Tuple[Optional[nx.MultiDiGraph], Dict]:
        """
        Query road network for a single province.

        Parameters
        ----------
        province_name_or_code : str
            Province name or PSGC code
        network_type : str, default 'drive'
            Network type ('drive', 'walk', 'bike', 'all', 'all_private')
        buffer_meters : float, default 100
            Buffer distance in meters to catch border roads
        simplify_tolerance : float, default 0.0001
            Simplification tolerance for complex coastlines (degrees)
        decompose_islands : bool, default True
            Break MultiPolygon into individual islands for querying

        Returns
        -------
        graph : MultiDiGraph or None
            Combined road network graph (None if query fails)
        metadata : dict
            Query metadata (province info, statistics, errors)
        """
        # Check cache first
        cache_key = f"{province_name_or_code}_{network_type}_{buffer_meters}_{simplify_tolerance}_{decompose_islands}"
        if cache_key in self.cache:
            self.logger.info(f"Using cached result for {province_name_or_code}")
            return self.cache[cache_key]

        # Find province in GeoDataFrame
        # Check if input is a 4-digit province code
        df = self.gdf.copy()
        df['province_code'] = df['adm2_psgc'].astype(str).str.zfill(10).str[:4]

        province_code_input = str(province_name_or_code).zfill(4) if province_name_or_code.isdigit() and len(province_name_or_code) <= 4 else None

        mask = (
            (df['adm2_en'].str.contains(province_name_or_code, case=False, na=False)) |
            (df['adm2_psgc'].astype(str) == str(province_name_or_code)) |
            (df['province_code'] == province_code_input if province_code_input else False)
        )

        province_data = df[mask]

        if len(province_data) == 0:
            self.logger.error(f"Province not found: {province_name_or_code}")
            return None, {'error': 'Province not found'}

        # Get province info
        province_name = province_data['adm2_en'].iloc[0]
        province_code = province_data['adm2_psgc'].iloc[0]
        region_name = province_data['adm1_en'].iloc[0]

        self.logger.info(f"Querying province: {province_name} ({province_code})")
        self.logger.info(f"Region: {region_name}")

        # Get province geometry (dissolve all barangays)
        province_geom = province_data.unary_union

        # Apply simplification if needed
        if simplify_tolerance > 0:
            province_geom = province_geom.simplify(simplify_tolerance)
            self.logger.debug(f"Simplified geometry with tolerance {simplify_tolerance}")

        # Apply buffer
        buffer_degrees = buffer_meters / 111320  # Approximate degrees at equator
        province_geom_buffered = province_geom.buffer(buffer_degrees)
        self.logger.debug(f"Applied {buffer_meters}m buffer")

        # Decompose into individual polygons if requested
        if decompose_islands:
            polygons = self._decompose_multipolygon(province_geom_buffered)
        else:
            polygons = [province_geom_buffered]

        self.logger.info(f"Querying {len(polygons)} polygon(s)")

        # Query each polygon
        graphs = []
        errors = []

        for i, polygon in enumerate(polygons, 1):
            try:
                self.logger.info(f"Querying polygon {i}/{len(polygons)}...")
                G = ox.graph_from_polygon(
                    polygon,
                    network_type=network_type,
                    simplify=True,
                    retain_all=True,  # Keep disconnected components
                    truncate_by_edge=True
                )

                if G is not None and len(G.nodes) > 0:
                    graphs.append(G)
                    self.logger.info(f"  → {len(G.nodes)} nodes, {len(G.edges)} edges")
                else:
                    self.logger.warning(f"  → Empty graph returned")

            except Exception as e:
                error_msg = f"Polygon {i} failed: {str(e)}"
                self.logger.warning(f"  → {error_msg}")
                errors.append(error_msg)

        # Merge graphs if multiple
        if len(graphs) == 0:
            self.logger.error(f"No graphs retrieved for {province_name}")
            metadata = {
                'province_name': province_name,
                'province_code': province_code,
                'region_name': region_name,
                'polygons_queried': len(polygons),
                'successful_queries': 0,
                'errors': errors,
                'nodes': 0,
                'edges': 0
            }
            result = (None, metadata)
            self.cache[cache_key] = result
            return result

        if len(graphs) == 1:
            combined_graph = graphs[0]
            self.logger.info(f"Single graph: {len(combined_graph.nodes)} nodes, {len(combined_graph.edges)} edges")
        else:
            self.logger.info(f"Merging {len(graphs)} graphs...")
            combined_graph = self._merge_graphs(graphs)
            self.logger.info(f"Merged graph: {len(combined_graph.nodes)} nodes, {len(combined_graph.edges)} edges")

        # Prepare metadata
        metadata = {
            'province_name': province_name,
            'province_code': province_code,
            'region_name': region_name,
            'polygons_queried': len(polygons),
            'successful_queries': len(graphs),
            'errors': errors if errors else None,
            'nodes': len(combined_graph.nodes),
            'edges': len(combined_graph.edges),
            'network_type': network_type,
            'buffer_meters': buffer_meters,
            'decompose_islands': decompose_islands
        }

        result = (combined_graph, metadata)
        self.cache[cache_key] = result
        return result

    def query_region(
        self,
        region_name_or_code: str,
        network_type: str = 'drive',
        buffer_meters: float = 100,
        simplify_tolerance: float = 0.0001,
        decompose_islands: bool = True,
        use_province_breakdown: bool = True
    ) -> Tuple[Optional[nx.MultiDiGraph], Dict]:
        """
        Query road network for a region.

        Parameters
        ----------
        region_name_or_code : str
            Region name or 2-digit PSGC code
        network_type : str, default 'drive'
            Network type ('drive', 'walk', 'bike', 'all', 'all_private')
        buffer_meters : float, default 100
            Buffer distance in meters to catch border roads
        simplify_tolerance : float, default 0.0001
            Simplification tolerance for complex coastlines (degrees)
        decompose_islands : bool, default True
            Break MultiPolygon into individual islands for querying
        use_province_breakdown : bool, default True
            If True, query each province separately then merge (RECOMMENDED for archipelagic regions).
            If False, query entire region shapefile directly (faster but may miss islands).

        Returns
        -------
        graph : MultiDiGraph or None
            Combined regional road network graph
        metadata : dict
            Query metadata (region info, province statistics, errors)
        """
        # Find region in GeoDataFrame
        # Check if input is a 2-digit region code
        df = self.gdf.copy()
        df['region_code'] = df['adm1_psgc'].astype(str).str.zfill(10).str[:2]

        region_code_input = str(region_name_or_code).zfill(2) if region_name_or_code.isdigit() and len(region_name_or_code) <= 2 else None

        mask = (
            (df['adm1_en'].str.contains(region_name_or_code, case=False, na=False)) |
            (df['adm1_psgc'].astype(str) == str(region_name_or_code)) |
            (df['region_code'] == region_code_input if region_code_input else False)
        )

        region_data = df[mask]

        if len(region_data) == 0:
            self.logger.error(f"Region not found: {region_name_or_code}")
            return None, {'error': 'Region not found'}

        # Get region info
        region_name = region_data['adm1_en'].iloc[0]
        region_code = region_data['region_code'].iloc[0]

        self.logger.info("="*60)
        self.logger.info(f"Querying region: {region_name} ({region_code})")
        self.logger.info("="*60)

        # Option 1: Query entire region directly (faster but may be incomplete for archipelagic regions)
        if not use_province_breakdown:
            self.logger.info("Using direct region query (entire shapefile)")

            # Get region geometry (dissolve all barangays)
            region_geom = region_data.unary_union

            # Apply simplification if needed
            if simplify_tolerance > 0:
                region_geom = region_geom.simplify(simplify_tolerance)
                self.logger.debug(f"Simplified geometry with tolerance {simplify_tolerance}")

            # Apply buffer
            buffer_degrees = buffer_meters / 111320
            region_geom_buffered = region_geom.buffer(buffer_degrees)
            self.logger.debug(f"Applied {buffer_meters}m buffer")

            # Decompose into individual polygons if requested
            if decompose_islands:
                polygons = self._decompose_multipolygon(region_geom_buffered)
            else:
                polygons = [region_geom_buffered]

            self.logger.info(f"Querying {len(polygons)} polygon(s)")

            # Query each polygon
            graphs = []
            errors = []

            for i, polygon in enumerate(polygons, 1):
                try:
                    self.logger.info(f"Querying polygon {i}/{len(polygons)}...")
                    G = ox.graph_from_polygon(
                        polygon,
                        network_type=network_type,
                        simplify=True,
                        retain_all=True,
                        truncate_by_edge=True
                    )

                    if G is not None and len(G.nodes) > 0:
                        graphs.append(G)
                        self.logger.info(f"  → {len(G.nodes)} nodes, {len(G.edges)} edges")
                    else:
                        self.logger.warning(f"  → Empty graph returned")

                except Exception as e:
                    error_msg = f"Polygon {i} failed: {str(e)}"
                    self.logger.warning(f"  → {error_msg}")
                    errors.append(error_msg)

            # Merge graphs if multiple
            if len(graphs) == 0:
                self.logger.error(f"No graphs retrieved for {region_name}")
                metadata = {
                    'region_name': region_name,
                    'region_code': region_code,
                    'query_method': 'direct',
                    'polygons_queried': len(polygons),
                    'successful_queries': 0,
                    'errors': errors,
                    'nodes': 0,
                    'edges': 0
                }
                return None, metadata

            if len(graphs) == 1:
                regional_graph = graphs[0]
            else:
                self.logger.info(f"Merging {len(graphs)} graphs...")
                regional_graph = self._merge_graphs(graphs)

            self.logger.info(f"Regional graph: {len(regional_graph.nodes)} nodes, {len(regional_graph.edges)} edges")

            # Prepare metadata
            metadata = {
                'region_name': region_name,
                'region_code': region_code,
                'query_method': 'direct',
                'polygons_queried': len(polygons),
                'successful_queries': len(graphs),
                'errors': errors if errors else None,
                'nodes': len(regional_graph.nodes),
                'edges': len(regional_graph.edges),
                'network_type': network_type,
                'buffer_meters': buffer_meters,
                'decompose_islands': decompose_islands
            }

            return regional_graph, metadata

        # Option 2: Query each province separately (RECOMMENDED for archipelagic regions)
        self.logger.info("Using province-level breakdown (recommended for archipelagic regions)")

        # Get list of provinces in region
        provinces = region_data[['adm2_psgc', 'adm2_en']].drop_duplicates()
        self.logger.info(f"Found {len(provinces)} provinces in region")

        # Query each province
        province_graphs = []
        province_metadata = []

        for idx, (_, row) in enumerate(provinces.iterrows(), 1):
            province_code = row['adm2_psgc']
            province_name = row['adm2_en']

            self.logger.info(f"\n[{idx}/{len(provinces)}] Processing: {province_name}")

            G, meta = self.query_province(
                province_code,
                network_type=network_type,
                buffer_meters=buffer_meters,
                simplify_tolerance=simplify_tolerance,
                decompose_islands=decompose_islands
            )

            if G is not None:
                province_graphs.append(G)

            province_metadata.append(meta)

        # Merge all province graphs
        if len(province_graphs) == 0:
            self.logger.error(f"No graphs retrieved for region {region_name}")
            metadata = {
                'region_name': region_name,
                'region_code': region_code,
                'provinces_total': len(provinces),
                'provinces_successful': 0,
                'province_details': province_metadata,
                'nodes': 0,
                'edges': 0
            }
            return None, metadata

        self.logger.info("\n" + "="*60)
        self.logger.info(f"Merging {len(province_graphs)} province graphs...")
        self.logger.info("="*60)

        regional_graph = self._merge_graphs(province_graphs)

        self.logger.info(f"Regional graph: {len(regional_graph.nodes)} nodes, {len(regional_graph.edges)} edges")

        # Prepare metadata
        metadata = {
            'region_name': region_name,
            'region_code': region_code,
            'query_method': 'province_breakdown',
            'provinces_total': len(provinces),
            'provinces_successful': len(province_graphs),
            'province_details': province_metadata,
            'nodes': len(regional_graph.nodes),
            'edges': len(regional_graph.edges),
            'network_type': network_type,
            'buffer_meters': buffer_meters,
            'decompose_islands': decompose_islands
        }

        return regional_graph, metadata

    def _merge_graphs(self, graphs: List[nx.MultiDiGraph]) -> nx.MultiDiGraph:
        """
        Merge multiple NetworkX graphs and deduplicate edges by osmid.

        Parameters
        ----------
        graphs : list of MultiDiGraph
            List of graphs to merge

        Returns
        -------
        MultiDiGraph
            Merged and deduplicated graph
        """
        if len(graphs) == 1:
            return graphs[0]

        # Compose all graphs
        combined = nx.compose_all(graphs)

        # Deduplicate edges by osmid
        edges_to_remove = []
        seen_osmids = set()

        for u, v, key, data in combined.edges(keys=True, data=True):
            osmid = data.get('osmid')

            if osmid is not None:
                # Handle both single osmid and lists of osmids
                osmid_tuple = tuple(osmid) if isinstance(osmid, list) else (osmid,)

                if osmid_tuple in seen_osmids:
                    edges_to_remove.append((u, v, key))
                else:
                    seen_osmids.add(osmid_tuple)

        combined.remove_edges_from(edges_to_remove)

        if edges_to_remove:
            self.logger.debug(f"Removed {len(edges_to_remove)} duplicate edges")

        return combined

    def to_igraph(self, G: nx.MultiDiGraph) -> 'ig.Graph':
        """
        Convert NetworkX graph to igraph for plotting.

        Parameters
        ----------
        G : MultiDiGraph
            NetworkX graph

        Returns
        -------
        igraph.Graph
            Converted graph for plotting

        Raises
        ------
        ImportError
            If igraph is not installed
        """
        if not IGRAPH_AVAILABLE:
            raise ImportError(
                "igraph is not installed. Install with: pip install igraph"
            )

        self.logger.info("Converting NetworkX graph to igraph...")

        # Get node and edge data
        nodes = list(G.nodes(data=True))
        edges = list(G.edges(data=True, keys=True))

        # Create node mapping
        node_to_idx = {node: idx for idx, (node, _) in enumerate(nodes)}

        # Create edge list with indices
        edge_list = [(node_to_idx[u], node_to_idx[v]) for u, v, _, _ in edges]

        # Create igraph
        g = ig.Graph(n=len(nodes), edges=edge_list, directed=True)

        # Add node attributes
        for attr in ['y', 'x', 'street_count']:
            values = [data.get(attr) for _, data in nodes]
            if any(v is not None for v in values):
                g.vs[attr] = values

        # Add edge attributes
        for attr in ['osmid', 'name', 'highway', 'length']:
            values = [data.get(attr) for _, _, _, data in edges]
            if any(v is not None for v in values):
                g.es[attr] = values

        self.logger.info(f"Converted to igraph: {g.vcount()} nodes, {g.ecount()} edges")

        return g

    def print_summary(self, metadata: Dict):
        """
        Print a summary of query results.

        Parameters
        ----------
        metadata : dict
            Metadata from query_region or query_province
        """
        print("\n" + "="*60)
        print("QUERY SUMMARY")
        print("="*60)

        if 'region_name' in metadata and 'provinces_total' in metadata:
            # Regional query - province breakdown
            print(f"Region: {metadata['region_name']} ({metadata['region_code']})")
            print(f"Query Method: {metadata.get('query_method', 'province_breakdown')}")
            print(f"Provinces: {metadata['provinces_successful']}/{metadata['provinces_total']} successful")
            print(f"\nNetwork Statistics:")
            print(f"  Nodes: {metadata['nodes']:,}")
            print(f"  Edges: {metadata['edges']:,}")

            print(f"\nProvince Details:")
            for detail in metadata['province_details']:
                status = "✓" if detail['nodes'] > 0 else "✗"
                print(f"  {status} {detail['province_name']}: {detail['nodes']:,} nodes, {detail['edges']:,} edges")
                if detail.get('errors'):
                    for error in detail['errors']:
                        print(f"      Error: {error}")
        elif 'region_name' in metadata and 'query_method' in metadata and metadata['query_method'] == 'direct':
            # Regional query - direct
            print(f"Region: {metadata['region_name']} ({metadata['region_code']})")
            print(f"Query Method: {metadata['query_method']}")
            print(f"\nQuery Statistics:")
            print(f"  Polygons queried: {metadata['polygons_queried']}")
            print(f"  Successful queries: {metadata['successful_queries']}")

            print(f"\nNetwork Statistics:")
            print(f"  Nodes: {metadata['nodes']:,}")
            print(f"  Edges: {metadata['edges']:,}")

            if metadata.get('errors'):
                print(f"\nErrors:")
                for error in metadata['errors']:
                    print(f"  - {error}")
        else:
            # Province query
            print(f"Province: {metadata['province_name']} ({metadata['province_code']})")
            print(f"Region: {metadata['region_name']}")
            print(f"\nQuery Statistics:")
            print(f"  Polygons queried: {metadata['polygons_queried']}")
            print(f"  Successful queries: {metadata['successful_queries']}")

            print(f"\nNetwork Statistics:")
            print(f"  Nodes: {metadata['nodes']:,}")
            print(f"  Edges: {metadata['edges']:,}")

            if metadata.get('errors'):
                print(f"\nErrors:")
                for error in metadata['errors']:
                    print(f"  - {error}")

        print("="*60 + "\n")

    def plot_graph(
        self,
        G: nx.MultiDiGraph,
        figsize: Tuple[int, int] = (15, 15),
        node_size: int = 0,
        edge_color: str = 'red',
        edge_linewidth: float = 0.5,
        edge_alpha: float = 0.7,
        bgcolor: str = 'white',
        save_path: Optional[str] = None,
        dpi: int = 300,
        show: bool = True
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot road network using OSMnx native methods.

        Parameters
        ----------
        G : MultiDiGraph
            NetworkX graph from OSMnx
        figsize : tuple, default (15, 15)
            Figure size (width, height) in inches
        node_size : int, default 0
            Size of nodes (0 to hide)
        edge_color : str, default 'red'
            Color of road edges
        edge_linewidth : float, default 0.5
            Width of road edges
        edge_alpha : float, default 0.7
            Transparency of edges (0-1)
        bgcolor : str, default 'white'
            Background color
        save_path : str, optional
            Path to save figure (e.g., 'output.png')
        dpi : int, default 300
            DPI for saved figure
        show : bool, default True
            Whether to display the plot

        Returns
        -------
        fig, ax : Figure, Axes
            Matplotlib figure and axes objects
        """
        self.logger.info(f"Plotting graph with {len(G.nodes)} nodes and {len(G.edges)} edges")

        fig, ax = ox.plot_graph(
            G,
            figsize=figsize,
            node_size=node_size,
            edge_color=edge_color,
            edge_linewidth=edge_linewidth,
            edge_alpha=edge_alpha,
            bgcolor=bgcolor,
            show=False,
            close=False
        )

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor=bgcolor)
            self.logger.info(f"Saved plot to {save_path}")

        if show:
            plt.show()

        return fig, ax

    def plot_graph_with_boundary(
        self,
        G: nx.MultiDiGraph,
        region_or_province_code: str,
        figsize: Tuple[int, int] = (15, 15),
        node_size: int = 0,
        edge_color: str = 'red',
        edge_linewidth: float = 0.8,
        edge_alpha: float = 0.8,
        boundary_color: str = 'black',
        boundary_linewidth: float = 1.5,
        boundary_facecolor: str = 'lightgray',
        boundary_alpha: float = 0.2,
        save_path: Optional[str] = None,
        dpi: int = 300,
        show: bool = True
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot road network overlaid on region/province boundary shapefile.

        Parameters
        ----------
        G : MultiDiGraph
            NetworkX graph from OSMnx
        region_or_province_code : str
            2-digit region code or 4-digit province code to get boundary
        figsize : tuple, default (15, 15)
            Figure size (width, height) in inches
        node_size : int, default 0
            Size of nodes (0 to hide)
        edge_color : str, default 'red'
            Color of road edges
        edge_linewidth : float, default 0.8
            Width of road edges
        edge_alpha : float, default 0.8
            Transparency of edges (0-1)
        boundary_color : str, default 'black'
            Color of boundary lines
        boundary_linewidth : float, default 1.5
            Width of boundary lines
        boundary_facecolor : str, default 'lightgray'
            Fill color for boundary polygons
        boundary_alpha : float, default 0.2
            Transparency of boundary fill (0-1)
        save_path : str, optional
            Path to save figure (e.g., 'output.png')
        dpi : int, default 300
            DPI for saved figure
        show : bool, default True
            Whether to display the plot

        Returns
        -------
        fig, ax : Figure, Axes
            Matplotlib figure and axes objects
        """
        self.logger.info(f"Plotting graph with boundary for code: {region_or_province_code}")

        # Determine if it's a region or province code
        code_length = len(str(region_or_province_code).zfill(4))

        if code_length == 2 or len(str(region_or_province_code)) <= 2:
            # Region code (2 digits)
            df = self.gdf.copy()
            df['region_code'] = df['adm1_psgc'].astype(str).str.zfill(10).str[:2]
            region_code = str(region_or_province_code).zfill(2)
            boundary_data = df[df['region_code'] == region_code]
            label = f"Region {region_code}"
        else:
            # Province code (4 digits)
            df = self.gdf.copy()
            df['province_code'] = df['adm2_psgc'].astype(str).str.zfill(10).str[:4]
            province_code = str(region_or_province_code).zfill(4)
            boundary_data = df[df['province_code'] == province_code]
            label = f"Province {province_code}"

        if len(boundary_data) == 0:
            self.logger.error(f"No boundary found for code: {region_or_province_code}")
            raise ValueError(f"No boundary found for code: {region_or_province_code}")

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot boundary first (background)
        boundary_data.plot(
            ax=ax,
            facecolor=boundary_facecolor,
            edgecolor=boundary_color,
            linewidth=boundary_linewidth,
            alpha=boundary_alpha
        )

        # Convert graph to GeoDataFrame for plotting
        edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)

        # Plot road network on top
        edges_gdf.plot(
            ax=ax,
            color=edge_color,
            linewidth=edge_linewidth,
            alpha=edge_alpha
        )

        ax.set_title(f'Road Network - {label}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_aspect('equal')

        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            self.logger.info(f"Saved plot to {save_path}")

        if show:
            plt.show()

        return fig, ax
