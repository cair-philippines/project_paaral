import os
import re
import sys
import time
import pickle
import traceback
import importlib
import numpy as np
import pandas as pd

from shapely.geometry import Point, Polygon, LineString
from shapely.prepared import prep
import geopandas as gpd
import networkx as nx
import osmnx as ox

from rtree import index

class MapResources:
    def __init__(self, preloaded=True):
        self.gadm = self.load_psgc_shapefiles(preload=preloaded)
        # self.G_target = self.load_target_parent_network(preload=preloaded)

    def load_psgc_shapefiles(self, preload=True):
        if preload == True:
            fpath = '../datasets/processed/psgc_shapefiles.pkl'

            with open(fpath, 'rb') as file:
                df_gadm = pickle.load(file)

            return df_gadm

        else:
            gdf_ph = gpd.read_file('../datasets/philippines-psgc-maps/BgySubMuns')

            relevant_columns = [
                'psgc_code','name','geo_level','city_class','inc_class','urb_rur',
                'shape_sqkm','geometry'
            ]
            gdf = gdf_ph[relevant_columns].copy()
            
            gdf['psgc_code'] = gdf['psgc_code'].apply(self._add_leading_zero).astype('string')
            gdf = gdf.rename(
                columns={'psgc_code':'adm4_psgc', 'name':'adm4_psgc_name'}
            )
            gdf['adm4_psgc_name'] = gdf['adm4_psgc_name'].astype('string')
            
            # Remove rows without geometries
            mask = gdf['geometry'].notna()
            gdf = gdf.loc[mask]
            
            gdf['adm3_psgc'] = gdf['adm4_psgc'].apply(lambda x: x[:7] + '000')
            gdf['adm2_psgc'] = gdf['adm4_psgc'].apply(lambda x: x[:5] + '00000')
            gdf['adm1_psgc'] = gdf['adm4_psgc'].apply(lambda x: x[:2] + '00000000')
            
            # We write a function that matches PSGC's named labels with the shapefiles'
            # columns of numerical PSGC codes
            df_gadm = self.append_psgc_labels(gdf)

            return df_gadm
    
    def _add_leading_zero(self, x):
        str_x = str(x)
        if str_x not in ['NaN','nan']:
            if len(str_x) == 9:
                return '0' + str_x
            elif len(str_x) == 10:
                return str_x
    
    def append_psgc_labels(self, gdf_brgys):
        # Load the CSV files that accompanied the psgc shapefiles
        dir_csvs = '../datasets/philippines-psgc-maps/csv'
        files = os.listdir(dir_csvs)
        filepaths = [os.path.join(dir_csvs, file) for file in files]
        
        df_csvs = {}
        for i, path in enumerate(filepaths):
            tmp_df = pd.read_csv(path)
            columns = tmp_df.columns.tolist()
            psgc_cols = [col for col in columns if '_psgc' in col]
        
            for col in psgc_cols:
                tmp_df[col] = tmp_df[col].apply(self._add_leading_zero).astype('string')
        
            df_csvs[files[i][:7]] = tmp_df
        
        # Fix SubMuns within City of Manila without psgc codes for adm1 and adm3
        tmp_adm3 = df_csvs['PH_Adm3'].copy()
        tmp_adm4 = df_csvs['PH_Adm4'].copy()
        
        mask = tmp_adm4['adm1_psgc'].isna()
        manila_city_psgc3 = tmp_adm3[tmp_adm3['adm3_en'].astype('string').str.contains(r'manila', flags=re.IGNORECASE)]['adm3_psgc'].values[0]
        tmp_adm4.loc[mask, 'adm3_psgc'] = manila_city_psgc3
        tmp_adm4.loc[mask, 'adm1_psgc'] = '1300000000'
        
        mask = tmp_adm4['adm4_en'].isna()
        tmp_adm4.loc[mask, 'adm4_en'] = 'No Label'
        
        mask = tmp_adm4['adm2_psgc'].isna()
        tmp_adm4.loc[mask, 'adm2_psgc'] = '1380000000'
        
        df_csvs['PH_Adm4'] = tmp_adm4

        # Merge information from CSVs with dataframe with geometry to get location names
        mrg_adm1 = gdf_brgys.merge(
            df_csvs['PH_Adm1'][['adm1_psgc','adm1_en']],
            right_on='adm1_psgc', left_on='adm1_psgc',
            how='left'
        )
        mrg_adm2 = mrg_adm1.merge(
            df_csvs['PH_Adm2'][['adm2_psgc','adm2_en']],
            right_on='adm2_psgc', left_on='adm2_psgc',
            how='left'
        )
        mrg_adm3 = mrg_adm2.merge(
            df_csvs['PH_Adm3'][['adm3_psgc','adm3_en']],
            right_on='adm3_psgc', left_on='adm3_psgc',
            how='left'
        )
        
        final_mrg = (
            mrg_adm3
            .rename(
                columns={
                    'adm1_en':'adm1_psgc_name',
                    'adm2_en':'adm2_psgc_name',
                    'adm3_en':'adm3_psgc_name',
                }
            )
        )
        cols = final_mrg.columns.tolist()
        cols.remove('geometry')
        
        final_mrg = final_mrg[sorted(cols) + ['geometry']]
        
        return final_mrg


    def get_filepaths_of_regional_road_networks(self):
        dir_path = '../datasets/networks/regional_drive_graphs'
        pickle_files = os.listdir(dir_path)
        
        # We will use these as keys to our dictionary where keys are the region_names and the values are
        # the complete filepath of the pickle files in the regional_graphs directory for easy access later
        pattern = r'(.*?).pkl'
        keys = [re.findall(pattern, file, re.IGNORECASE)[0] for file in pickle_files]
        values = [os.path.join(dir_path, file) for file in pickle_files]
        
        regional_graph_dict = {k:v for k,v in zip(keys, values)}

        self.regional_graphs = regional_graph_dict

    def inspect_regional_road_network(self, filepath, figsize=(6,6)):
        start_time = time.time()
        pattern = r'/([^/]+)\.pkl$'
        region_name = re.findall(pattern, filepath)[0]
        
        print(f"Inspecting the preloaded OSMNx road network graph of {region_name}.")
        with open(filepath, 'rb') as file:
            G = pickle.load(file)

        # Include in the plotting of the graph road network the shape of the region
        mask = self.gadm['adm1_psgc_name'] == region_name
        gdf = self.gadm.loc[mask].copy()
        
        # We will dissolve by province so when we plot the shape of the region, we
        # have edges along the boundaries of provinces
        gdf_ds = gdf.dissolve(by='adm2_psgc_name')
        
        fig, ax = ox.plot_graph(
            G, figsize=figsize,
            node_size=.25, edge_linewidth=.25,
            bgcolor='none', show=False, close=False,
            node_zorder=2
        )

        gdf_ds.plot(
            ax=ax, facecolor='none', edgecolor='black',
            linewidth=.75, zorder=3
        )
        
        end_time = time.time() - start_time
        print(f"Time elapsed for inspection: {end_time:.2f} secs")

    def preprocess_public_school_coordinates_further(self, gdf_public):
        pub_tmp = gdf_public.copy()
        
        mask = (
            (pub_tmp['longitude'].notna())
            & (pub_tmp['latitude'].notna())
        )
        pub_tmp.loc[mask, 'geometry'] = pub_tmp.loc[mask].apply(
            lambda row: Point(row['longitude'], row['latitude']),
            axis=1
        )
        
        gpd_pub = gpd.GeoDataFrame(pub_tmp, geometry='geometry', crs=4326)
        
        return gpd_pub

    def preprocess_private_school_coordinates_further(self, gdf_private):
        gdf_priv = gdf_private.copy()
        gdf_priv[['longitude','latitude']] = gdf_priv[['longitude','latitude']].astype('string')
        
        validate_lon = gdf_priv.loc[:, 'longitude'].apply(
            lambda val: self.is_valid_decimal_coordinate(val, coordinate_type='longitude')
        )
        validate_lat = gdf_priv.loc[:, 'latitude'].apply(
            lambda val: self.is_valid_decimal_coordinate(val, coordinate_type='latitude')
        )
        
        df_val_lon = pd.DataFrame(
            validate_lon.values.tolist(),
            columns=['val_bool','val_lon','val_remarks'],
            index=gdf_priv.index
        )
        
        df_val_lat = pd.DataFrame(
            validate_lat.values.tolist(),
            columns=['val_bool','val_lat','val_remarks'],
            index=gdf_priv.index
        )
        
        # Append private school validity critera to original dataframe
        vdf_priv = gdf_priv.copy()
        vdf_priv.loc[:, 'longitude_is_valid'] = df_val_lon['val_bool']
        vdf_priv.loc[:, 'latitude_is_valid'] = df_val_lat['val_bool']

        vdf_priv.loc[:, 'longitude_valid'] = df_val_lon['val_lon']
        vdf_priv.loc[:, 'latitude_valid'] = df_val_lat['val_lat']
        
        mask_val = (
            # (mask_notna) &
            (vdf_priv['longitude_is_valid'] == True)
            & (vdf_priv['latitude_is_valid'] == True)
        )
        valid_priv = vdf_priv.loc[mask_val].copy()

        valid_priv['geometry'] = valid_priv.apply(
            lambda row: Point(row['longitude_valid'], row['latitude_valid']),
            axis=1
        )
        
        valid_priv = gpd.GeoDataFrame(valid_priv, geometry='geometry', crs=4326)
        
        return valid_priv

    def get_adjacent_geographies(self, gpd_gadm, target_psgc, max_depth=2):
        """
        Find a target geography and its adjacent geographies based on a PSGC code.
        
        Parameters:
        -----------
        gpd_gadm: GeoDataFrame
            GeoDataFrame containing the shapefiles of the Philippines with PSGC codes
        target_psgc: int or str
            The PSGC code of the target geography
        max_depth: int, default=2
            Maximum depth to crawl outward from the target geography
                
        Returns:
        --------
        dict
            A dictionary containing:
            - 'target_area': GeoDataFrame of the target geography
            - 'adjacent_areas': Dictionary of adjacent geographies by depth
            - 'search_area_complete': GeoDataFrame of all areas combined
            - 'target_shape': The unary_union shape of the target area
            - 'adjacent_shapes': Dictionary of unary_union shapes by depth
            - 'search_shape_complete': The unary_union shape of the entire search area
        """
        # Initialize results dictionary
        results = {
            'target_area': None,
            'adjacent_areas': {},
            'search_area_complete': None,
            'target_shape': None,
            'adjacent_shapes': {},
            'search_shape_complete': None
        }
        
        start_time = time.time()
        print(f"Finding adjacent geographies for PSGC {target_psgc} with max_depth={max_depth}")
        
        # Convert PSGC to string for consistent handling
        target_psgc = str(target_psgc)
        
        # Determine administrative level based on PSGC structure
        if target_psgc.endswith('0000000'):
            psgc_column = 'adm1_psgc'
            admin_level = 'region'
        elif target_psgc.endswith('00000') and not target_psgc.endswith('0000000'):
            psgc_column = 'adm2_psgc'
            admin_level = 'province'
        elif target_psgc.endswith('000') and not target_psgc.endswith('00000'):
            psgc_column = 'adm3_psgc'
            admin_level = 'city/municipality'
        else:
            psgc_column = 'adm4_psgc'
            admin_level = 'barangay'
        
        print(f"Detected administrative level: {admin_level}")
        
        # Create spatial index if it doesn't exist
        if not hasattr(gpd_gadm, 'sindex') or gpd_gadm.sindex is None:
            gpd_gadm = gpd_gadm.copy()
            gpd_gadm = gpd_gadm.reset_index(drop=True)
            gpd_gadm.create_sindex()
        
        # Convert target_psgc to multiple types for matching
        try:
            target_psgc_float = float(target_psgc)
            target_psgc_int = int(target_psgc)
        except ValueError:
            target_psgc_float = None
            target_psgc_int = None
        
        # Get the target geography
        mask_target = (
            (gpd_gadm[psgc_column] == target_psgc) | 
            (gpd_gadm[psgc_column] == target_psgc_float) | 
            (gpd_gadm[psgc_column] == target_psgc_int)
        )
        target_indices = gpd_gadm.index[mask_target].tolist()
        
        if not target_indices:
            print(f"Warning: No geography found with {psgc_column}={target_psgc}")
            # Try other columns as fallback
            for col in ['adm1_psgc', 'adm2_psgc', 'adm3_psgc', 'adm4_psgc']:
                if col == psgc_column:
                    continue
                    
                mask = (
                    (gpd_gadm[col] == target_psgc) | 
                    (gpd_gadm[col] == target_psgc_float) | 
                    (gpd_gadm[col] == target_psgc_int)
                )
                if any(mask):
                    target_indices = gpd_gadm.index[mask].tolist()
                    psgc_column = col
                    print(f"Found in {col}")
                    break
            
            if not target_indices:
                print(f"No geography found for PSGC {target_psgc} in any column")
                return results
        
        target_geography = gpd_gadm.iloc[target_indices].copy()
        target_shape = target_geography.unary_union
        
        # Store target area in results
        results['target_area'] = target_geography
        results['target_shape'] = target_shape
        
        # For recursive search, we'll use a specific approach based on admin level
        all_found_indices = set(target_indices)
        frontier_geographies = target_geography
        
        # Store all areas as we find them
        all_areas = [target_geography]
        
        # Define a buffer tolerance for finding adjacency
        buffer_tolerance = 0.0001
        
        # Find adjacent geographies at each depth
        for depth in range(1, max_depth + 1):
            print(f"Finding adjacent areas at depth {depth}...")
            
            # We'll collect new geographies discovered at this depth
            depth_indices = set()
            
            # For each geography in the frontier
            for _, frontier_geo in frontier_geographies.iterrows():
                # Use a small buffer to handle potential gaps between polygons
                buffered_geometry = frontier_geo.geometry.buffer(buffer_tolerance)
                
                # Get candidates using spatial index
                candidate_indices = list(gpd_gadm.sindex.intersection(buffered_geometry.bounds))
                
                # Filter out already found indices
                candidate_indices = [idx for idx in candidate_indices if idx not in all_found_indices]
                
                if not candidate_indices:
                    continue
                    
                candidates = gpd_gadm.iloc[candidate_indices]
                
                # Check for adjacency at the same admin level
                for idx, candidate in candidates.iterrows():
                    # Skip if we've already processed this or if it's not the same admin level
                    if idx in all_found_indices or pd.isna(candidate[psgc_column]):
                        continue
                        
                    # Check if it's actually adjacent
                    if (buffered_geometry.touches(candidate.geometry) or 
                        (buffered_geometry.intersects(candidate.geometry) and 
                         not frontier_geo.geometry.equals(candidate.geometry))):
                        depth_indices.add(idx)
            
            # If no new geographies found at this depth, break early
            if not depth_indices:
                print(f"No more adjacent areas found at depth {depth}")
                break
                
            # Create a GeoDataFrame for this level
            level_gdf = gpd_gadm.iloc[list(depth_indices)].copy()
            level_shape = level_gdf.unary_union
            print(f"Found {len(level_gdf)} adjacent areas at depth {depth}")
            
            # Store the adjacent areas for this depth
            results['adjacent_areas'][depth] = level_gdf
            results['adjacent_shapes'][depth] = level_shape
            all_areas.append(level_gdf)
            
            # Update tracking for next iteration
            all_found_indices.update(depth_indices)
            frontier_geographies = level_gdf
        
        # Combine all areas to get the complete search area
        if all_areas:
            search_area_complete = pd.concat(all_areas)
            search_shape_complete = search_area_complete.unary_union
            results['search_area_complete'] = search_area_complete
            results['search_shape_complete'] = search_shape_complete
        
        elapsed_time = time.time() - start_time
        print(f"Geography processing completed in {elapsed_time:.2f} seconds")
        
        return results

    def extract_schools_from_geographies(self, geography_results, gdf_public=None, gdf_private=None):
        """
        Extract school points from target and adjacent geographies.
        
        Parameters:
        -----------
        geography_results: dict
            Output from get_adjacent_geographies function containing geographic areas
        gdf_public: GeoDataFrame, optional
            GeoDataFrame containing public school points
        gdf_private: GeoDataFrame, optional
            GeoDataFrame containing private school points
                
        Returns:
        --------
        dict
            A dictionary containing:
            - 'public_schools_in_target': Public schools within the target area
            - 'public_schools_in_adjacent': Dictionary of public schools in adjacent areas by depth
            - 'private_schools_in_target': Private schools within the target area
            - 'private_schools_in_adjacent': Dictionary of private schools in adjacent areas by depth
            - 'private_schools_all': Private schools in the entire search area
        """
        start_time = time.time()
        print("Extracting schools from geographies...")
        
        # Initialize results dictionary
        results = {
            'public_schools_in_target': None,
            'public_schools_in_adjacent': {},
            'private_schools_in_target': None,
            'private_schools_in_adjacent': {},
            'private_schools_all': None
        }
        
        # Extract required shapes from geography_results
        target_shape = geography_results['target_shape']
        adjacent_shapes = geography_results['adjacent_shapes']
        search_shape_complete = geography_results['search_shape_complete']

        # Prepare shapes for faster spatial queries
        prep_start = time.time()
        prepared_target = prep(target_shape)
        prepared_adjacent = {
            depth: prep(shape) for depth, shape in adjacent_shapes.items()
        }
        prep_elapsed = time.time() - prep_start
        print(f"Prepared geometries in {prep_elapsed:.2f} seconds")
        
        if target_shape is None:
            print("Warning: No target shape found in geography_results")
            return results
        
        # Create indices for school data if provided
        if gdf_public is not None and (not hasattr(gdf_public, 'sindex') or gdf_public.sindex is None):
            gdf_public = gdf_public.copy()
            gdf_public.create_sindex()
            
        if gdf_private is not None and (not hasattr(gdf_private, 'sindex') or gdf_private.sindex is None):
            gdf_private = gdf_private.copy()
            gdf_private.create_sindex()
        
        # Extract public schools if provided
        if gdf_public is not None:
            print("Extracting public schools...")
            
            # 1. Public schools in the target area
            target_bounds = target_shape.bounds
            pub_candidates_idx = list(gdf_public.sindex.intersection(target_bounds))
            pub_candidates = gdf_public.iloc[pub_candidates_idx]
            
            # Use vectorized operation to find schools in target area
            query_start = time.time()
            in_target_mask = pub_candidates.geometry.apply(
                prepared_target.intersects
            )
            query_elapsed = time.time() - query_start
            print(
                f"Public target query completed in {query_elapsed:.2f} seconds"
            )
            public_schools_in_target = pub_candidates.loc[in_target_mask].copy()
            results['public_schools_in_target'] = public_schools_in_target
            print(f"Found {len(public_schools_in_target)} public schools in target area")
            
            # 2. Public schools in adjacent areas by depth
            for depth, adjacent_shape in adjacent_shapes.items():
                adjacent_bounds = adjacent_shape.bounds
                adjacent_candidates_idx = list(gdf_public.sindex.intersection(adjacent_bounds))
                adjacent_candidates = gdf_public.iloc[adjacent_candidates_idx]
                
                # Use vectorized operation to find schools in this adjacent area
                query_start = time.time()
                in_adjacent_mask = adjacent_candidates.geometry.apply(
                    prepared_adjacent[depth].contains
                )
                query_elapsed = time.time() - query_start
                print(
                    f"Public adjacent depth {depth} query completed in {query_elapsed:.2f} seconds"
                )
                public_schools_in_adjacent = adjacent_candidates.loc[in_adjacent_mask].copy()
                results['public_schools_in_adjacent'][depth] = public_schools_in_adjacent
                print(f"Found {len(public_schools_in_adjacent)} public schools in adjacent areas at depth {depth}")
        
        # Extract private schools if provided
        if gdf_private is not None and search_shape_complete is not None:
            print("Extracting private schools...")

            # Private schools in the entire search area
            search_bounds = search_shape_complete.bounds
            private_candidates_idx = list(
                gdf_private.sindex.intersection(search_bounds)
            )
            private_candidates = gdf_private.iloc[private_candidates_idx]

            # Use vectorized operation to find schools in the entire search area
            search_query_start = time.time()
            in_search_mask = private_candidates.geometry.intersects(
                search_shape_complete
            )
            search_query_elapsed = time.time() - search_query_start
            private_schools_all = private_candidates.loc[in_search_mask].copy()
            print(
                f"Private search area query completed in {search_query_elapsed:.2f} seconds"
            )
            results['private_schools_all'] = private_schools_all
            print(f"Found {len(private_schools_all)} private schools in the entire search area")

            # Private schools within the target area
            target_query_start = time.time()
            in_target_mask = private_schools_all.geometry.apply(
                prepared_target.intersects
            )
            target_query_elapsed = time.time() - target_query_start
            print(
                f"Private target query completed in {target_query_elapsed:.2f} seconds"
            )
            private_schools_in_target = private_schools_all.loc[in_target_mask].copy()
            results['private_schools_in_target'] = private_schools_in_target
            print(f"Found {len(private_schools_in_target)} private schools in target area")

            # Private schools in adjacent areas by depth
            for depth, adjacent_shape in adjacent_shapes.items():
                adjacent_query_start = time.time()
                in_adjacent_mask = private_schools_all.geometry.apply(
                    prepared_adjacent[depth].contains
                )
                adjacent_query_elapsed = time.time() - adjacent_query_start
                private_schools_in_adjacent = private_schools_all.loc[
                    in_adjacent_mask
                ].copy()
                print(
                    f"Private adjacent depth {depth} query completed in {adjacent_query_elapsed:.2f} seconds"
                )
                results['private_schools_in_adjacent'][depth] = private_schools_in_adjacent
                print(
                    f"Found {len(private_schools_in_adjacent)} private schools in adjacent areas at depth {depth}"
                )
        
        elapsed_time = time.time() - start_time
        print(
            f"School extraction completed in {elapsed_time:.2f} seconds (prepared geometries)"
        )
        
        return results

    def convert_nx_to_igraph(self, G):
        """Convert NetworkX graph to iGraph for faster processing."""
        print("Converting NetworkX graph to iGraph...")
        
        # Get edge list with attributes
        edges = []
        weights = []
        for u, v, data in G.edges(data=True):
            edges.append((u, v))
            weights.append(data.get('length', 1.0))
        
        # Create igraph from edge list
        unique_nodes = set()
        for u, v in edges:
            unique_nodes.add(u)
            unique_nodes.add(v)
        
        node_map = {node: i for i, node in enumerate(unique_nodes)}
        edges_remapped = [(node_map[u], node_map[v]) for u, v in edges]
        
        ig_graph = iG.Graph(directed=True)
        ig_graph.add_vertices(len(node_map))
        ig_graph.add_edges(edges_remapped)
        ig_graph.es['weight'] = weights
        
        # Store the mapping for later reference
        reverse_map = {i: node for node, i in node_map.items()}
        
        return ig_graph, node_map, reverse_map

    def generate_subgraph(self, geo_results):
        """Create a road network subgraph from the preloaded regional graphs.

        The method determines the regions intersecting ``search_area`` and loads
        their prebuilt road network graphs. Each regional graph is immediately
        truncated to the union of the search shapes (first by bounding box then
        by polygon) before all graphs are composed together.  This reduces the
        number of nodes and edges processed and speeds up subgraph generation.

        Parameters
        ----------
        geo_results : dict
            Dictionary produced by the map resource builder containing the
            ``search_area_complete`` GeoDataFrame.

        Returns
        -------
        networkx.MultiDiGraph
            Subgraph clipped to ``search_area``.
        """
        start_time = time.time()
        print("Generating subgraph from geo_results")
        
        search_area = geo_results['search_area_complete'] # a GeoDataFrame
        captured_regions = search_area['adm1_psgc_name'].unique()
    
        regional_network_graphs = []
        for region in captured_regions:
            reg_graph_filepath = self.regional_graphs.get(region)
    
            with open(reg_graph_filepath, 'rb') as file:
                G_region = pickle.load(file)
                regional_network_graphs.append(G_region)
    
        combined_regional_graphs = nx.compose_all(regional_network_graphs)
    
        search_shape = search_area.unary_union
        search_space_subgraph = ox.truncate.truncate_graph_polygon(
            combined_regional_graphs, search_shape, truncate_by_edge=True, retain_all=True,
        )
    
        time_elapsed = (time.time() - start_time) / 60
        print(f"Subgraph extracted. Time elapsed: {time_elapsed:.2f} minutes")
        
        return search_space_subgraph
        
    def is_valid_decimal_coordinate(self, value, coordinate_type='both'):
        """
        Check if a value is a valid decimal coordinate, handling various formats including commas.
        
        Args:
            value: The value to check (can be string, float, int, or tuple)
            coordinate_type: 'latitude', 'longitude', or 'both' (default)
        
        Returns:
            tuple: (is_valid, normalized_value, error_message)
        """
        # Handle different input types
        if isinstance(value, (list, tuple)):
            # Handle coordinate pairs (lat, lon)
            if len(value) == 2:
                if coordinate_type == 'latitude':
                    result = self.is_valid_decimal_coordinate(value[0], 'latitude')
                elif coordinate_type == 'longitude':
                    result = self.is_valid_decimal_coordinate(value[1], 'longitude')
                else:
                    lat_result = self.is_valid_decimal_coordinate(value[0], 'latitude')
                    lon_result = self.is_valid_decimal_coordinate(value[1], 'longitude')
                    if lat_result[0] and lon_result[0]:
                        return True, (lat_result[1], lon_result[1]), None
                    else:
                        errors = []
                        if not lat_result[0]:
                            errors.append(f"Latitude: {lat_result[2]}")
                        if not lon_result[0]:
                            errors.append(f"Longitude: {lon_result[2]}")
                        return False, value, "; ".join(errors)
                return result
            else:
                return False, value, "Coordinate pair must contain exactly 2 values"
        
        # Convert to string for parsing if needed
        if not isinstance(value, str):
            value = str(value)
        
        # Remove any whitespace
        value = value.strip()
        
        # Remove commas from numbers (could be thousands separators)
        value = value.replace(',', '')
    
        # Remove extra decimal points
        val_split = [char for char in value]
        # display(f'Debug: val_split: {val_split}')
        char_idxs = [i for i, char in enumerate(val_split) if char == '.']
        [val_split.pop(i) for i in char_idxs[1:]] # pop out excess periods
        value = ''.join(val_split)
        
        # Handle degrees, minutes, seconds (DMS) format and convert to decimal
        dms_pattern = r'(\d+)[°]\s*(\d+)[\']\s*(\d+(?:\.\d+)?)["\s]*([NSEW])?'
        dms_match = re.match(dms_pattern, value, re.IGNORECASE)
        
        if dms_match:
            degrees, minutes, seconds, direction = dms_match.groups()
            decimal = float(degrees) + float(minutes)/60 + float(seconds)/3600
            
            # Apply direction
            if direction:
                if direction.upper() in ['S', 'W']:
                    decimal = -decimal
            
            value = str(decimal)
        
        # Handle degrees and decimal minutes format
        dm_pattern = r'(\d+)[°]\s*(\d+(?:\.\d+)?)[\']\s*([NSEW])?'
        dm_match = re.match(dm_pattern, value, re.IGNORECASE)
        
        if dm_match:
            degrees, minutes, direction = dm_match.groups()
            decimal = float(degrees) + float(minutes)/60
            
            # Apply direction
            if direction:
                if direction.upper() in ['S', 'W']:
                    decimal = -decimal
            
            value = str(decimal)
        
        # Handle degree symbol alone
        degree_pattern = r'(\d+(?:\.\d+)?)[°]\s*([NSEW])?'
        degree_match = re.match(degree_pattern, value, re.IGNORECASE)
        
        if degree_match:
            decimal, direction = degree_match.groups()
            decimal = float(decimal)
            
            # Apply direction
            if direction:
                if direction.upper() in ['S', 'W']:
                    decimal = -decimal
            
            value = str(decimal)
        
        # Remove direction indicators if they're at the end
        value = re.sub(r'[NSEW]$', '', value, flags=re.IGNORECASE).strip()
        
        # Remove any non-numeric characters except decimal point and negative sign
        value = re.sub(r'[^\d\.\-]', '', value)
        
        try:
            # Convert to float
            coord = float(value)
            
            # Check for NaN or infinite values
            if not (-float('inf') < coord < float('inf')):
                return False, value, "Value is not a finite number"
            
            # Validate based on coordinate type
            if coordinate_type.lower() == 'latitude':
                if -90 <= coord <= 90:
                    return True, coord, None
                elif -180 <= coord <= 180:
                    return False, coord, f"Candidate for lon-lat switch."
                else:
                    coord_ = self._handle_excess_decimal_points(coord, 2)
                    return False, coord_, f"{coord} -> {coord_} Note: Longitude must be between -90 and 90 degrees"
            
            elif coordinate_type.lower() == 'longitude':
                if -180 <= coord <= 180:
                    return True, coord, None
                elif -90 <= coord <= 90:
                    return False, coord, f"Candidate for lon-lat switch."
                else:
                    coord_ = self._handle_excess_decimal_points(coord, 3)
                    return False, coord_, f"{coord} -> {coord_} Note: Longitude must be between -180 and 180 degrees"
            
            elif coordinate_type.lower() == 'both':
                # For 'both', check if it's a valid latitude OR longitude
                if -90 <= coord <= 90:
                    return True, coord, None
                elif -180 <= coord <= 180:
                    return True, coord, None
                else:
                    coord_ = self._handle_excess_decimal_points(coord)
                    return False, coord_, f"{coord} -> {coord_} Note: Value must be between -90 and 90 (latitude) or -180 and 180 (longitude)"
            
            else:
                return False, value, "coordinate_type must be 'latitude', 'longitude', or 'both'"
        
        except (ValueError, TypeError) as e:
            return False, value, f"Could not convert to decimal number: {str(e)}"

    def _handle_excess_decimal_points(self, coord, insert_index):
        coord_str = str(coord).replace('.','')
        coord_lst = list(coord_str)
        coord_lst.insert(insert_index, '.')
    
        coord_n = ''.join(coord_lst)
        coord_ = float(coord_n)
    
        return coord_