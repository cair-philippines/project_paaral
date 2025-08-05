"""Optimized network construction utilities for school routing analysis.

This module defines :class:`OptimizedSchoolNetworkBuilder`, the central
utility for constructing optimized road networks and computing routes
between schools. Both public **and private** schools inside the target
administrative boundary are treated as origin points. The builder
converts road data from NetworkX to ``igraph`` and uses spatial indexing
for fast queries on a single master graph so that routes for many
schools can be computed efficiently.
"""

import os
import re
import sys
import time
import pickle
import importlib
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt

import osmnx as ox
import networkx as nx
from shapely.geometry import Point, Polygon, LineString
from pyproj import Transformer

import igraph as iG
from igraph import Graph

# Add the project root to Python path
project_root = Path(__file__).parent.parent  # Go up from src/ to project root
sys.path.append(str(project_root))

from config.config import Config
cf = Config()

# Global builder reference for multiprocessing worker functions
_BUILDER = None


def _init_worker(builder):
    """Initializer that assigns the global builder for workers."""
    global _BUILDER
    _BUILDER = builder


def _process_school_worker(school_id, buffer_distance_m, max_distance_km):
    """Delegate school processing to the shared builder instance."""
    return _BUILDER._process_single_school_optimized(
        school_id, buffer_distance_m, max_distance_km
    )


class OptimizedSchoolNetworkBuilder:
    """Construct and query optimized road networks for school routing."""

    def __init__(
        self,
        road_network_graph,
        public_schools_gdf,
        private_schools_gdf,
        peripheral_schools_gdf,
        admin_boundary,
        num_processes=None,
    ):
        """Instantiate the builder with required datasets.

        Parameters
        ----------
        road_network_graph : networkx.DiGraph
            Base road network.
        public_schools_gdf : geopandas.GeoDataFrame
            GeoDataFrame of public schools in EPSG:4326 or convertible CRS.
        private_schools_gdf : geopandas.GeoDataFrame
            GeoDataFrame of private schools located **within** ``admin_boundary``.
            These schools act as origin schools together with the public schools.
        peripheral_schools_gdf : geopandas.GeoDataFrame
            GeoDataFrame of private schools in adjacent geographies to include
            as potential destinations when needed.
        admin_boundary : geopandas.GeoDataFrame
            Administrative boundary used to clip results.
        num_processes : int, optional
            Number of worker processes for parallel execution. ``None`` uses
            all available CPU cores.
        """
        self.G_road = road_network_graph  # NetworkX road network
        self.public_schools = public_schools_gdf
        self.private_schools = private_schools_gdf
        self.peripheral_schools = peripheral_schools_gdf
        self.admin_boundary = admin_boundary
        self.num_processes = num_processes
        self.G_road_3123 = None

        # Ensure all GDFs are in the same CRS
        target_crs = 3123
        if self.public_schools.crs != target_crs:
            self.public_schools = self.public_schools.to_crs(target_crs)
        if self.private_schools.crs != target_crs:
            self.private_schools = self.private_schools.to_crs(target_crs)
        if self.peripheral_schools.crs != target_crs:
            self.peripheral_schools = self.peripheral_schools.to_crs(
                target_crs
            )

        # Will be populated during setup
        self.master_igraph = None
        self.school_to_vertex_map = {}
        # track one or more schools that may snap to each network vertex
        self.vertex_to_school_map = {}
        self.spatial_indices = {}
        self.all_schools_combined = None
        # Lookup dictionaries for fast osmid-to-vertex translation
        self.osmid_to_vertex_3123 = {}
        self.osmid_to_vertex_4326 = {}

    def build_complete_network(
        self, buffer_distance_m=5000, max_distance_km=15
    ):
        """Execute the full network build workflow.

        Parameters
        ----------
        buffer_distance_m : int, optional
            Search radius around each origin school in meters.
        max_distance_km : int, optional
            Maximum road distance to consider for candidate schools.

        Returns
        -------
        dict
            Dictionary with consolidated distance matrices, routes and the
            generated master graphs.
        """
        print("🚀 Starting optimized network build...")

        # Step 1: One-time setup
        self._setup_master_infrastructure()

        # Step 2: Build spatial indices for fast lookups
        self._build_spatial_indices()

        # Step 3: Debug - test a single school first
        self._debug_single_school()

        # Step 4: Parallel processing with shared infrastructure
        results = self._parallel_process_schools(
            buffer_distance_m, max_distance_km
        )

        # Step 5: Consolidate results
        return self._consolidate_results(results)

    def _debug_single_school(self):
        """Print diagnostic information for the first public school."""
        print("\n🐛 DEBUG: Testing single school...")

        # Get first public school
        first_school_id = self.public_schools["school_id"].iloc[0]
        print(f"Testing school ID: {first_school_id}")

        # Check if it's in our mapping
        if first_school_id in self.school_to_vertex_map:
            vertex_idx = self.school_to_vertex_map[first_school_id]
            print(f"✅ School {first_school_id} mapped to vertex {vertex_idx}")
        else:
            print(f"❌ School {first_school_id} NOT found in vertex mapping")
            print(f"Available mappings: {len(self.school_to_vertex_map)}")
            return

        # Test finding nearby schools
        school_row = self.public_schools[
            self.public_schools["school_id"] == first_school_id
        ].iloc[0]
        nearby_schools = self._find_nearby_schools_fast(
            school_row.geometry, 5000
        )
        print(
            f"Found {len(nearby_schools)} nearby schools: {nearby_schools[:5]}..."
        )

        # Test distance calculation
        if nearby_schools:
            distance_result = self._calculate_distances_fast(
                first_school_id, nearby_schools[:5]
            )
            print(f"Distance matrix shape: {distance_result.shape}")
            print(f"Distance matrix:\n{distance_result.head()}")

        print("🐛 DEBUG: Single school test complete\n")

    def _setup_master_infrastructure(self):
        """Prepare the master graph and mapping dictionaries."""
        print("📊 Setting up master infrastructure...")

        # Combine all schools into master dataset
        self.all_schools_combined = self._combine_all_schools()
        print(f"Combined {len(self.all_schools_combined)} total schools")

        # Create master iGraph (3123 for calculations)
        self.master_igraph = self._create_master_igraph()
        self.osmid_to_vertex_3123 = {
            v["osmid"]: idx for idx, v in enumerate(self.master_igraph.vs)
        }

        # Create 4326 version for plotting
        self.master_igraph_4326 = self._create_master_igraph_4326()
        self.osmid_to_vertex_4326 = {
            v["osmid"]: idx for idx, v in enumerate(self.master_igraph_4326.vs)
        }

        # Create bidirectional school-to-vertex mapping
        self._create_school_vertex_mappings()

        print(
            f"✅ Master graph: {len(self.master_igraph.vs)} vertices, {len(self.master_igraph.es)} edges"
        )
        print(
            f"✅ Master graph 4326: {len(self.master_igraph_4326.vs)} vertices"
        )
        print(
            f"✅ Mapped {len(self.school_to_vertex_map)} schools to road network"
        )

    def _combine_all_schools(self):
        """Concatenate public, private and peripheral school tables.

        Returns
        -------
        pandas.DataFrame
            Combined dataframe with a ``school_type`` column.
        """
        all_schools_list = []

        # Add public schools
        public_std = self.public_schools.copy()
        public_std["school_type"] = "public"
        all_schools_list.append(public_std)

        # Add private schools
        private_std = self.private_schools.copy()
        private_std["school_type"] = "private"
        all_schools_list.append(private_std)

        # Add peripheral schools
        peripheral_std = self.peripheral_schools.copy()
        peripheral_std["school_type"] = "peripheral"
        all_schools_list.append(peripheral_std)

        # Combine all
        all_schools = pd.concat(all_schools_list, ignore_index=True)

        print(
            f"School counts: Public={len(public_std)}, Private={len(private_std)}, Peripheral={len(peripheral_std)}"
        )

        return all_schools

    def _create_master_igraph(self):
        """Create the projective iGraph used for calculations.

        Returns
        -------
        igraph.Graph
            Graph projected to EPSG:3123 with length attributes.
        """
        print("🔄 Converting NetworkX to iGraph...")

        # Project road network once and cache the result
        if self.G_road_3123 is None:
            self.G_road_3123 = ox.projection.project_graph(
                self.G_road, to_crs="epsg:3123"
            )
            print(
                f"Projected graph: {len(self.G_road_3123.nodes)} nodes, {len(self.G_road_3123.edges)} edges"
            )
        else:
            print("Using cached projected road graph")

        # Convert to iGraph using optimized method
        ig_graph = self._nx_to_igraph_optimized(self.G_road_3123)

        return ig_graph

    def _nx_to_igraph_optimized(self, G):
        """Internal helper to convert a projected NetworkX graph to ``igraph``.

        Parameters
        ----------
        G : networkx.Graph
            Projected road graph.

        Returns
        -------
        igraph.Graph
            Graph with length edge attribute and node metadata.
        """
        # Create node mapping
        nodes = list(G.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}

        # Create iGraph
        ig = iG.Graph(directed=G.is_directed())
        ig.add_vertices(len(nodes))

        # Add edges with length weights
        edges = []
        lengths = []

        for u, v, data in G.edges(data=True):
            edges.append((node_to_idx[u], node_to_idx[v]))
            lengths.append(data.get("length", 1.0))

        if edges:
            ig.add_edges(edges)
            ig.es["length"] = lengths

        # Store node attributes - CRITICAL: store both osmid and coordinates
        for i, node in enumerate(nodes):
            ig.vs[i]["osmid"] = node
            # Store coordinates from the projected graph
            if "x" in G.nodes[node] and "y" in G.nodes[node]:
                ig.vs[i]["x"] = G.nodes[node]["x"]
                ig.vs[i]["y"] = G.nodes[node]["y"]

        print(
            f"Created iGraph with {len(ig.vs)} vertices and {len(ig.es)} edges"
        )
        return ig

    def _create_school_vertex_mappings(self):
        """Map each school to the nearest road network vertex.

        The method uses precomputed dictionaries that translate OSM node IDs to
        igraph vertex indices for both the projected and WGS84 graphs. This
        avoids expensive linear searches when attaching school metadata.
        """
        print("🗺️  Mapping schools to road network vertices...")

        # Use cached projected graph for coordinate matching
        if self.G_road_3123 is None:
            self.G_road_3123 = ox.projection.project_graph(
                self.G_road, to_crs="epsg:3123"
            )

        G_proj = self.G_road_3123

        successful_mappings = 0
        failed_mappings = 0

        valid_mask = self.all_schools_combined.geometry.apply(
            lambda g: hasattr(g, "x") and hasattr(g, "y")
        )
        valid_schools = self.all_schools_combined[valid_mask]

        x_array = valid_schools.geometry.x.to_numpy()
        y_array = valid_schools.geometry.y.to_numpy()

        nearest_osmids = ox.distance.nearest_nodes(G_proj, x_array, y_array)

        for (idx, school), nearest_osmid in zip(
            valid_schools.iterrows(), nearest_osmids
        ):
            try:
                # Find corresponding vertex in iGraph using prebuilt mapping
                vertex_idx = self.osmid_to_vertex_3123.get(nearest_osmid)

                if vertex_idx is not None:
                    school_id = school["school_id"]
                    self.school_to_vertex_map[school_id] = vertex_idx
                    self.vertex_to_school_map.setdefault(
                        vertex_idx, []
                    ).append(school_id)

                    # attach school info to the vertex
                    v = self.master_igraph.vs[vertex_idx]
                    v["school_ids"] = self.vertex_to_school_map[vertex_idx]
                    prev_types = (
                        v["school_types"]
                        if "school_types" in v.attributes()
                        and v["school_types"] is not None
                        else []
                    )
                    prev_names = (
                        v["school_names"]
                        if "school_names" in v.attributes()
                        and v["school_names"] is not None
                        else []
                    )
                    prev_attrs = (
                        v["school_attrs"]
                        if "school_attrs" in v.attributes()
                        and v["school_attrs"] is not None
                        else []
                    )
                    v["school_types"] = prev_types + [school["school_type"]]
                    v["school_names"] = prev_names + [
                        school.get("school_name", "")
                    ]
                    v["school_attrs"] = prev_attrs + [
                        self._extract_school_attributes(school)
                    ]
                    v["is_school"] = True

                    # replicate attributes in WGS84 graph
                    v4326_idx = self.osmid_to_vertex_4326.get(nearest_osmid)
                    if v4326_idx is not None:
                        vtx = self.master_igraph_4326.vs[v4326_idx]
                        vtx["school_ids"] = self.vertex_to_school_map[
                            vertex_idx
                        ]
                        prev_t = (
                            vtx["school_types"]
                            if "school_types" in vtx.attributes()
                            and vtx["school_types"] is not None
                            else []
                        )
                        prev_n = (
                            vtx["school_names"]
                            if "school_names" in vtx.attributes()
                            and vtx["school_names"] is not None
                            else []
                        )
                        prev_a = (
                            vtx["school_attrs"]
                            if "school_attrs" in vtx.attributes()
                            and vtx["school_attrs"] is not None
                            else []
                        )
                        vtx["school_types"] = prev_t + [school["school_type"]]
                        vtx["school_names"] = prev_n + [
                            school.get("school_name", "")
                        ]
                        vtx["school_attrs"] = prev_a + [
                            self._extract_school_attributes(school)
                        ]
                        vtx["is_school"] = True

                    successful_mappings += 1
                else:
                    failed_mappings += 1
                    print(
                        f"Warning: Could not find vertex for osmid {nearest_osmid}"
                    )

            except Exception as e:
                failed_mappings += 1
                print(
                    f"Warning: Could not map school {school.get('school_id', 'unknown')}: {e}"
                )

        # Handle schools without valid coordinates
        invalid_schools = self.all_schools_combined[~valid_mask]
        for _, school in invalid_schools.iterrows():
            failed_mappings += 1
            print(
                f"Warning: School {school.get('school_id', 'unknown')} has invalid geometry"
            )

        print(
            f"Mapping results: {successful_mappings} successful, {failed_mappings} failed"
        )

    def _build_spatial_indices(self):
        """Create spatial indices for public, private and peripheral schools.

        Returns
        -------
        None
            The indices are stored in ``self.spatial_indices``.
        """
        from rtree import index

        print("🌍 Building spatial indices...")

        try:
            # Public schools index
            self.spatial_indices["public"] = index.Index()
            for idx, row in self.public_schools.iterrows():
                if hasattr(row.geometry, "bounds"):
                    # Use integer index for rtree
                    self.spatial_indices["public"].insert(
                        idx, row.geometry.bounds
                    )

            # Private schools index
            self.spatial_indices["private"] = index.Index()
            for idx, row in self.private_schools.iterrows():
                if hasattr(row.geometry, "bounds"):
                    self.spatial_indices["private"].insert(
                        idx, row.geometry.bounds
                    )

            # Peripheral schools index
            self.spatial_indices["peripheral"] = index.Index()
            for idx, row in self.peripheral_schools.iterrows():
                if hasattr(row.geometry, "bounds"):
                    self.spatial_indices["peripheral"].insert(
                        idx, row.geometry.bounds
                    )

            print(
                f"✅ Built spatial indices for {len(self.spatial_indices)} school types"
            )

        except Exception as e:
            print(f"Warning: Could not build spatial indices: {e}")
            print("Will use fallback geometric intersection")

    def _parallel_process_schools(self, buffer_distance_m, max_distance_km):
        """Process each origin school in parallel using ``multiprocessing``.

        Parameters
        ----------
        buffer_distance_m : int
            Buffer radius for nearby search.
        max_distance_km : int
            Maximum allowable route distance.

        Returns
        -------
        list
            List of tuples ``(school_id, distance_dict, routes_dict)``.
        """
        from multiprocessing import get_context, cpu_count

        # Get list of origin schools (public and private schools in target area)
        origin_schools = (
            self.public_schools["school_id"].tolist()
            + self.private_schools["school_id"].tolist()
        )

        print(f"🔄 Processing {len(origin_schools)} schools in parallel...")

        ctx = get_context("fork")
        num_workers = self.num_processes or cpu_count()
        with ctx.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(self,),
        ) as pool:
            results = pool.starmap(
                _process_school_worker,
                [
                    (sid, buffer_distance_m, max_distance_km)
                    for sid in origin_schools
                ],
            )

        return results

    def _process_single_school_optimized(
        self, school_id, buffer_distance_m, max_distance_km
    ):
        """Compute candidate routes for one origin school of any type.

        Parameters
        ----------
        school_id : str
            Identifier of the origin school.
        buffer_distance_m : int
            Search radius around the origin in meters.
        max_distance_km : int
            Maximum allowable route distance.

        Returns
        -------
        tuple
            ``(school_id, distance_matrix, routes)`` where ``distance_matrix``
            is a pandas DataFrame of distances and ``routes`` a dict keyed by
            destination IDs.
        """
        try:
            # Get school info from the appropriate dataset
            if school_id in self.public_schools["school_id"].values:
                school_row = self.public_schools[
                    self.public_schools["school_id"] == school_id
                ].iloc[0]
            else:
                school_row = self.private_schools[
                    self.private_schools["school_id"] == school_id
                ].iloc[0]

            # Find nearby schools using spatial indices
            nearby_schools = self._find_nearby_schools_fast(
                school_row.geometry, buffer_distance_m
            )

            # Calculate distance matrix using pre-built master graph
            distance_matrix = self._calculate_distances_fast(
                school_id, nearby_schools
            )

            # Filter by maximum distance
            if max_distance_km:
                distance_matrix = distance_matrix.where(
                    distance_matrix <= max_distance_km * 1000, np.nan
                )

            # Get candidates
            candidates = self._get_candidates_fast(distance_matrix)

            # Calculate routes
            routes = self._calculate_routes_fast(school_id, candidates)

            return school_id, distance_matrix.to_dict(), routes

        except Exception as e:
            print(f"Error processing school {school_id}: {e}")
            return school_id, {}, {}

    def _find_nearby_schools_fast(self, origin_geometry, buffer_distance_m):
        """Return school IDs within a buffer distance of an origin geometry.

        Parameters
        ----------
        origin_geometry : shapely.geometry.BaseGeometry
            Geometry of the origin school.
        buffer_distance_m : int
            Search radius in meters.

        Returns
        -------
        list
            IDs of schools intersecting the buffer.
        """
        buffer_geom = origin_geometry.buffer(buffer_distance_m)
        bounds = buffer_geom.bounds

        nearby_schools = []

        # Search in all school types
        school_gdfs = [
            ("public", self.public_schools),
            ("private", self.private_schools),
            ("peripheral", self.peripheral_schools),
        ]

        for school_type, gdf in school_gdfs:
            if school_type in self.spatial_indices:
                # Use spatial index
                candidates = list(
                    self.spatial_indices[school_type].intersection(bounds)
                )
                for idx in candidates:
                    if idx < len(gdf):
                        school_geom = gdf.iloc[idx].geometry
                        if buffer_geom.intersects(school_geom):
                            school_id = gdf.iloc[idx]["school_id"]
                            nearby_schools.append(school_id)
            else:
                # Fallback: direct geometric intersection
                for idx, row in gdf.iterrows():
                    if buffer_geom.intersects(row.geometry):
                        nearby_schools.append(row["school_id"])

        return nearby_schools

    def _calculate_distances_fast(
        self, origin_school_id, destination_school_ids
    ):
        """Compute pairwise road distances from one origin to many destinations.

        Parameters
        ----------
        origin_school_id : str
            ID of the origin school.
        destination_school_ids : list
            List of candidate destination IDs.

        Returns
        -------
        pandas.DataFrame
            DataFrame with a single row of distances (meters).
        """
        # Get origin vertex
        origin_vertex = self.school_to_vertex_map.get(origin_school_id)
        if origin_vertex is None:
            print(
                f"Warning: Origin school {origin_school_id} not mapped to vertex"
            )
            return pd.DataFrame()

        # Get destination vertices and remove duplicates
        valid_destinations = []
        dest_vertices = []
        seen_vertices = set()  # Track seen vertices to avoid duplicates

        for dest_id in destination_school_ids:
            dest_vertex = self.school_to_vertex_map.get(dest_id)
            if dest_vertex is not None and dest_vertex not in seen_vertices:
                valid_destinations.append(dest_id)
                dest_vertices.append(dest_vertex)
                seen_vertices.add(dest_vertex)  # Mark as seen
            elif dest_vertex in seen_vertices:
                # Skip duplicate vertex but still add the school ID for tracking
                # print(
                #     f"Skipping duplicate vertex {dest_vertex} for school {dest_id}"
                # )
                pass

        if not dest_vertices:
            # print(
            #     f"Warning: No valid destination vertices found for school {origin_school_id}"
            # )
            return pd.DataFrame()

        # print(
        #     f"Calculating distances from {origin_school_id} to {len(dest_vertices)} unique destinations"
        # )

        # Calculate distances using iGraph
        try:
            distances = self.master_igraph.distances(
                source=[origin_vertex],
                target=dest_vertices,  # Now guaranteed to have no duplicates
                weights="length",
                algorithm="dijkstra",
            )[
                0
            ]  # Get first (and only) row

            # Create distance matrix
            distance_matrix = pd.DataFrame(
                [distances],
                index=[origin_school_id],
                columns=valid_destinations,  # Only includes schools with unique vertices
            )

            # print(f"Distance calculation successful: {distance_matrix.shape}")
            return distance_matrix

        except Exception as e:
            # print(
            #     f"Distance calculation error for school {origin_school_id}: {e}"
            # )
            return pd.DataFrame()

    def _get_candidates_fast(self, distance_matrix):
        """Return destination IDs with finite distances from the origin.

        Parameters
        ----------
        distance_matrix : pandas.DataFrame
            Distance matrix produced by :meth:`_calculate_distances_fast`.

        Returns
        -------
        dict
            Mapping of origin ID to list of viable destination IDs.
        """
        if distance_matrix.empty:
            return {}

        origin_id = distance_matrix.index[0]
        valid_destinations = distance_matrix.columns[
            distance_matrix.iloc[0].notna()
        ].tolist()

        return {origin_id: valid_destinations}

    def _calculate_routes_fast(self, origin_school_id, candidates_dict):
        """Compute shortest paths from an origin to candidate schools.

        Parameters
        ----------
        origin_school_id : str
            ID of the origin school.
        candidates_dict : dict
            Mapping from origin ID to list of destination IDs.

        Returns
        -------
        dict
            Mapping of destination IDs to igraph vertex paths.
        """
        routes = {}

        origin_vertex = self.school_to_vertex_map.get(origin_school_id)
        if origin_vertex is None or origin_school_id not in candidates_dict:
            return routes

        candidates = candidates_dict[origin_school_id]

        # Calculate all routes at once
        dest_vertices = [
            self.school_to_vertex_map.get(dest_id) for dest_id in candidates
        ]
        dest_vertices = [v for v in dest_vertices if v is not None]

        if dest_vertices:
            try:
                paths = self.master_igraph.get_shortest_paths(
                    origin_vertex,
                    dest_vertices,
                    weights="length",
                    output="vpath",
                )

                # Map paths back to school IDs
                for i, dest_id in enumerate(candidates):
                    if (
                        i < len(paths)
                        and self.school_to_vertex_map.get(dest_id) is not None
                    ):
                        routes[dest_id] = paths[i]
                    else:
                        routes[dest_id] = []

            except Exception as e:
                print(
                    f"Route calculation error for school {origin_school_id}: {e}"
                )

        return {origin_school_id: routes}

    def _consolidate_results(self, results):
        """Combine distance matrices and route dictionaries from workers.

        Parameters
        ----------
        results : list
            Output list from :meth:`_parallel_process_schools`.

        Returns
        -------
        dict
            Dictionary containing consolidated distances, routes and graphs.
        """
        print("📋 Consolidating results...")

        all_distance_matrices = []
        all_routes = {}

        for school_id, distance_dict, routes_dict in results:
            if distance_dict:
                df = pd.DataFrame.from_dict(distance_dict)
                all_distance_matrices.append(df)

            if routes_dict:
                all_routes.update(routes_dict)

        # Combine all distance matrices
        if all_distance_matrices:
            consolidated_distances = pd.concat(
                all_distance_matrices, ignore_index=False
            )
        else:
            consolidated_distances = pd.DataFrame()

        print(
            f"✅ Consolidated {len(all_distance_matrices)} distance matrices"
        )
        print(f"✅ Consolidated {len(all_routes)} route sets")

        # Build separate graphs for road distances and beneficiary flows
        beneficiary_edges = self._load_beneficiary_edges()

        distance_graph, beneficiary_graph = self._build_distance_and_beneficiary_graphs(
            consolidated_distances, beneficiary_edges
        )

        # Remove any vertices that are not present in the configured school lists
        self._prune_invalid_vertices(distance_graph, beneficiary_graph)

        return {
            "distance_matrix": consolidated_distances,
            "routes": all_routes,
            "master_graph": self.master_igraph,  # 3123 for calculations
            "master_graph_4326": self.master_igraph_4326,  # 4326 for plotting
            "school_mappings": self.school_to_vertex_map,
            "distance_graph": distance_graph,
            "beneficiary_graph": beneficiary_graph,
        }

    def _standardize_school_data(self, gdf, school_type):
        """Ensure a GeoDataFrame has required columns and a type label.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Input school dataframe.
        school_type : str
            Label describing the school category.

        Returns
        -------
        geopandas.GeoDataFrame
            Standardized dataframe with ``school_type`` column.
        """
        standardized = gdf.copy()
        standardized["school_type"] = school_type

        # Ensure required columns exist
        required_cols = ["school_id", "school_name", "geometry"]
        for col in required_cols:
            if col not in standardized.columns:
                standardized[col] = f"Unknown_{col}"

        return standardized

    def _extract_school_attributes(self, row):
        """Return a dictionary of serializable attributes for a school row."""
        attrs = row.drop(labels="geometry").to_dict()
        if hasattr(row.geometry, "x"):
            attrs["x"] = row.geometry.x
            attrs["y"] = row.geometry.y
        return attrs

    def _create_master_igraph_4326(self):
        """Create a WGS84 (EPSG:4326) version of the master iGraph for plotting.

        Returns
        -------
        igraph.Graph
            Road network graph in geographic coordinates.
        """
        print("🔄 Creating WGS84 version of master iGraph...")

        # Use the ORIGINAL road network (which is likely in 4326)
        # If G_road is not in 4326, project it
        if hasattr(self.G_road, "graph") and "crs" in self.G_road.graph:
            if self.G_road.graph["crs"] != "epsg:4326":
                G_4326 = ox.projection.project_graph(
                    self.G_road, to_crs="epsg:4326"
                )
            else:
                G_4326 = self.G_road.copy()
        else:
            # Assume it's already in 4326 or project from 3123 back to 4326
            G_4326 = ox.projection.project_graph(
                self.G_road, to_crs="epsg:4326"
            )

        # Convert to iGraph using same method
        ig_graph_4326 = self._nx_to_igraph_optimized(G_4326)

        print(f"Created 4326 iGraph with {len(ig_graph_4326.vs)} vertices")
        return ig_graph_4326

    def _load_beneficiary_edges(
        self,
        esc_path=os.path.join(
            cf.get_path("public_data"),
            "SY 2024-2025 JHS ESC Tagged Learners in LIS (from their previous G6 Schools).xlsx",
        ),
        shs_vp_path=os.path.join(
            cf.get_path("public_data"),
            "SY 2024-2025 SHS QVR Tagged Learners in LIS (from their previous G10 Schools).xlsx",
        ),
    ):
        """Load ESC and SHS VP beneficiary information as edges.

        Each Excel file is expected to contain columns
        ``[origin_id, origin_name, dest_id, dest_name, count]``. Only the IDs and
        count are used. Missing files or columns are skipped with a warning.

        Parameters
        ----------
        esc_path : str, optional
            Path to the ESC workbook.
        shs_vp_path : str, optional
            Path to the SHS VP workbook.

        Returns
        -------
        list
            List of tuples ``(origin_id, dest_id, count)``.
        """

        edges = {}
        sheet_info = [
            (
                esc_path,
                "esc_beneficiaries",
                {
                    "School ID (Origin Grade 6)": "origin_id",
                    "School Name (Origin Grade 6)": "origin_school_name",
                    "School ID (Destination Grade 7)": "dest_id",
                    "School Name (Destination Grade 7)": "dest_school_name",
                    "Count of Learners": "count",
                },
                "Sheet1",
            ),
            (
                shs_vp_path,
                "shsvp_beneficiaries",
                {
                    "School ID (Origin Grade 10)": "origin_id",
                    "School Name (Origin Grade 10)": "origin_school_name",
                    "School ID (Destination Grade 11)": "dest_id",
                    "School Name (Destination Grade 11)": "dest_school_name",
                    "Count of Learners": "count",
                },
                "qvr_grantees_jhs",
            ),
        ]
        for path, label, rename_map, sheet in sheet_info:
            if not path or not os.path.exists(path):
                print(f"Warning: beneficiary file missing -> {path}")
                continue

            try:
                df = pd.read_excel(path, sheet_name=sheet)
            except Exception as e:  # pragma: no cover - runtime environment varies
                print(f"Warning: could not read {path}: {e}")
                continue

            if not set(rename_map).issubset(df.columns):
                print(
                    f"Warning: expected columns {set(rename_map)} not found in {path}"
                )
                continue

            df = df.rename(columns=rename_map)
            for _, row in df.iterrows():
                try:
                    o = row["origin_id"]
                    d = row["dest_id"]
                    c = int(row["count"])
                    key = (o, d)
                    if key not in edges:
                        edges[key] = {"esc_beneficiaries": None, "shsvp_beneficiaries": None}
                    edges[key][label] = c
                except Exception as e:  # pragma: no cover - data issues
                    print(f"Warning: skipping row in {path}: {e}")

        return [
            (o, d, attrs.get("esc_beneficiaries"), attrs.get("shsvp_beneficiaries"))
            for (o, d), attrs in edges.items()
        ]

    def _create_pruned_school_graph(self, distance_df, beneficiary_edges=None):
        """Create an igraph of schools combining distance and beneficiary edges."""

        if distance_df.empty and not beneficiary_edges:
            return iG.Graph()

        public_ids = set(self.public_schools["school_id"])
        private_ids = set(self.private_schools["school_id"])
        peripheral_ids = set(self.peripheral_schools["school_id"])

        edge_dict = {}
        nodes = set()

        # Edges from distance matrix
        if not distance_df.empty:
            for origin_id, row in distance_df.iterrows():
                if origin_id not in public_ids and origin_id not in private_ids:
                    continue
                for dest_id, dist in row.dropna().items():
                    if (
                        dest_id in public_ids
                        or dest_id in private_ids
                        or dest_id in peripheral_ids
                    ):
                        key = (origin_id, dest_id)
                        edge_dict[key] = {"length": float(dist), "esc_beneficiaries": None, "shsvp_beneficiaries": None}
                        nodes.add(origin_id)
                        nodes.add(dest_id)

        # Edges from ESC / SHS VP beneficiary counts
        if beneficiary_edges:
            for origin_id, dest_id, esc_count, shsvp_count in beneficiary_edges:
                key = (origin_id, dest_id)
                attrs = edge_dict.get(key, {"length": None, "esc_beneficiaries": None, "shsvp_beneficiaries": None})
                if esc_count is not None:
                    attrs["esc_beneficiaries"] = esc_count
                if shsvp_count is not None:
                    attrs["shsvp_beneficiaries"] = shsvp_count
                edge_dict[key] = attrs
                nodes.add(origin_id)
                nodes.add(dest_id)

        node_list = sorted(nodes)
        name_to_idx = {sid: i for i, sid in enumerate(node_list)}

        g = iG.Graph(directed=True)
        g.add_vertices(len(node_list))
        g.vs["school_id"] = node_list

        pub_gdf = self.public_schools.set_index("school_id")
        pri_gdf = self.private_schools.set_index("school_id")
        per_gdf = self.peripheral_schools.set_index("school_id")

        for idx, sid in enumerate(node_list):
            if sid in pub_gdf.index:
                row = pub_gdf.loc[sid]
                g.vs[idx]["school_type"] = "public"
            elif sid in pri_gdf.index:
                row = pri_gdf.loc[sid]
                g.vs[idx]["school_type"] = "private"
            elif sid in per_gdf.index:
                row = per_gdf.loc[sid]
                g.vs[idx]["school_type"] = "peripheral"
            else:
                row = None
                g.vs[idx]["school_type"] = "unknown"

            if row is not None:
                g.vs[idx]["school_name"] = row.get("school_name", "")
                g.vs[idx]["x"] = getattr(row.geometry, "x", None)
                g.vs[idx]["y"] = getattr(row.geometry, "y", None)
                g.vs[idx]["school_attrs"] = [self._extract_school_attributes(row)]
            else:
                g.vs[idx]["school_name"] = ""
                g.vs[idx]["x"] = None
                g.vs[idx]["y"] = None
                g.vs[idx]["school_attrs"] = [{}]
            g.vs[idx]["is_school"] = True

        if edge_dict:
            edge_pairs = []
            lengths = []
            esc_beneficiaries = []
            shsvp_beneficiaries = []
            for (o, d), attrs in edge_dict.items():
                edge_pairs.append((name_to_idx[o], name_to_idx[d]))
                lengths.append(attrs.get("length"))
                esc_beneficiaries.append(attrs.get("esc_beneficiaries"))
                shsvp_beneficiaries.append(attrs.get("shsvp_beneficiaries"))

            g.add_edges(edge_pairs)
            g.es["length"] = lengths
            g.es["esc_beneficiaries"] = esc_beneficiaries
            g.es["shsvp_beneficiaries"] = shsvp_beneficiaries

        return g

    def _split_pruned_graph(self, pruned_graph):
        """Return separate distance and beneficiary graphs from a pruned graph.

        Parameters
        ----------
        pruned_graph : igraph.Graph
            Graph produced by ``_create_pruned_school_graph``.

        Returns
        -------
        tuple(Graph, Graph)
            ``(distance_graph, beneficiary_graph)`` with the same vertices as
            ``pruned_graph``. ``distance_graph`` keeps only edges that have a
            ``length`` value, while ``beneficiary_graph`` keeps edges with
            ESC or SHS VP beneficiary counts.
        """

        if pruned_graph.vcount() == 0:
            return iG.Graph(), iG.Graph()

        distance_edges = [
            e.index for e in pruned_graph.es if e["length"] is not None
        ]
        beneficiary_edges = [
            e.index
            for e in pruned_graph.es
            if e["esc_beneficiaries"] is not None
            or e["shsvp_beneficiaries"] is not None
        ]

        distance_graph = pruned_graph.subgraph_edges(
            distance_edges, delete_vertices=False
        )
        beneficiary_graph = pruned_graph.subgraph_edges(
            beneficiary_edges, delete_vertices=False
        )

        return distance_graph, beneficiary_graph

    def _build_distance_and_beneficiary_graphs(self, distance_df, beneficiary_edges=None):
        """Construct separate graphs for road distances and beneficiary flows.

        Both graphs share the same vertex ordering so that indices can be used
        interchangeably between them.

        Parameters
        ----------
        distance_df : pandas.DataFrame
            Consolidated distance matrix where rows/columns are school IDs and
            values are road distances in meters.
        beneficiary_edges : list, optional
            List of tuples ``(origin_id, dest_id, esc_count, shsvp_count)`` as
            returned by :meth:`_load_beneficiary_edges`.

        Returns
        -------
        tuple(Graph, Graph)
            ``(distance_graph, beneficiary_graph)``
        """

        allowed_ids = (
            set(self.public_schools["school_id"]) |
            set(self.private_schools["school_id"]) |
            set(self.peripheral_schools["school_id"])
        )

        edge_data = {}
        node_set = set()

        if not distance_df.empty:
            for origin_id, row in distance_df.iterrows():
                if origin_id not in allowed_ids:
                    continue
                for dest_id, dist in row.dropna().items():
                    if dest_id in allowed_ids:
                        node_set.update([origin_id, dest_id])
                        edge_data.setdefault((origin_id, dest_id), {})["length"] = float(dist)

        if beneficiary_edges:
            for origin_id, dest_id, esc_count, shsvp_count in beneficiary_edges:
                if origin_id in allowed_ids and dest_id in allowed_ids:
                    node_set.update([origin_id, dest_id])
                    attrs = edge_data.setdefault((origin_id, dest_id), {})
                    if esc_count is not None:
                        attrs["esc_beneficiaries"] = esc_count
                    if shsvp_count is not None:
                        attrs["shsvp_beneficiaries"] = shsvp_count

        node_list = sorted(node_set)
        node_to_idx = {sid: i for i, sid in enumerate(node_list)}

        pub_gdf = self.public_schools.set_index("school_id")
        pri_gdf = self.private_schools.set_index("school_id")
        per_gdf = self.peripheral_schools.set_index("school_id")

        def _create_base_graph():
            g = iG.Graph(directed=True)
            g.add_vertices(len(node_list))
            g.vs["school_id"] = node_list
            for idx, sid in enumerate(node_list):
                if sid in pub_gdf.index:
                    row = pub_gdf.loc[sid]
                    g.vs[idx]["school_type"] = "public"
                elif sid in pri_gdf.index:
                    row = pri_gdf.loc[sid]
                    g.vs[idx]["school_type"] = "private"
                elif sid in per_gdf.index:
                    row = per_gdf.loc[sid]
                    g.vs[idx]["school_type"] = "peripheral"
                else:
                    row = None
                    g.vs[idx]["school_type"] = "unknown"

                if row is not None:
                    g.vs[idx]["school_name"] = row.get("school_name", "")
                    g.vs[idx]["x"] = getattr(row.geometry, "x", None)
                    g.vs[idx]["y"] = getattr(row.geometry, "y", None)
                    g.vs[idx]["school_attrs"] = [self._extract_school_attributes(row)]
                else:
                    g.vs[idx]["school_name"] = ""
                    g.vs[idx]["x"] = None
                    g.vs[idx]["y"] = None
                    g.vs[idx]["school_attrs"] = [{}]
                g.vs[idx]["is_school"] = True
            return g

        distance_graph = _create_base_graph()
        beneficiary_graph = _create_base_graph()

        distance_edges = []
        distance_lengths = []
        beneficiary_edges_list = []
        esc_vals = []
        shs_vals = []

        for (o, d), attrs in edge_data.items():
            idx_o = node_to_idx[o]
            idx_d = node_to_idx[d]
            if "length" in attrs:
                distance_edges.append((idx_o, idx_d))
                distance_lengths.append(attrs["length"])
            if "esc_beneficiaries" in attrs or "shsvp_beneficiaries" in attrs:
                beneficiary_edges_list.append((idx_o, idx_d))
                esc_vals.append(attrs.get("esc_beneficiaries"))
                shs_vals.append(attrs.get("shsvp_beneficiaries"))

        if distance_edges:
            distance_graph.add_edges(distance_edges)
            distance_graph.es["length"] = distance_lengths

        if beneficiary_edges_list:
            beneficiary_graph.add_edges(beneficiary_edges_list)
            beneficiary_graph.es["esc_beneficiaries"] = esc_vals
            beneficiary_graph.es["shsvp_beneficiaries"] = shs_vals

        return distance_graph, beneficiary_graph

    def _prune_invalid_vertices(self, distance_graph, beneficiary_graph):
        """Remove vertices not present in the configured school ID lists."""

        valid_ids = (
            set(self.public_schools["school_id"]) |
            set(self.private_schools["school_id"]) |
            set(self.peripheral_schools["school_id"])
        )

        to_delete = [v.index for v in distance_graph.vs if v["school_id"] not in valid_ids]

        if to_delete:
            distance_graph.delete_vertices(to_delete)
            beneficiary_graph.delete_vertices(to_delete)