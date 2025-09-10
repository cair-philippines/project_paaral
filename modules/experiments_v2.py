"""Routines for running congestion and ESC slot experiments."""

import os
import copy
import math
from typing import Dict, List, Optional, Tuple

import igraph as ig

import numpy as np
import pandas as pd
import warnings

from config.config import Config
cf = Config()

VERBOSE = False


class Experiments:
    """Run baseline and redistribution experiments on a school network."""
    def __init__(
        self,
        graph,
        knowledge_base_class,
        geography={
            "level": "province",
            "name": "RIZAL",
        }
    ):
        """Construct an experiment runner.

        Parameters
        ----------
        graph : igraph.Graph
            Distance-based school network produced by the optimized builder.
        knowledge_base_class : KnowledgeBase
            Loaded datasets wrapper from :mod:`datasets`.
        geography : dict, optional
            Filter specifying which administrative unit to analyze. Defaults
            to ``{"level": "province", "name": "RIZAL"}``.
        """
        self.cwd = os.getcwd()
        self.graph = copy.deepcopy(graph)

        jhs_cocs = ['All Offering', 'ES and JHS', 'JHS with SHS', 'Purely JHS']
        nodes_jhs = [node for node in self.graph.vs if node['school_attrs'][0]['sector'] == "Public" and node['school_attrs'][0]['modified coc'] in jhs_cocs]
        
        self.geography = geography
        
        self.kb = knowledge_base_class
        self.public = knowledge_base_class.compile_public_datasets()
        self.private = knowledge_base_class.compile_private_datasets()

        self.run_experiments_setup()
    
    def run_experiments_setup(self):
        """Prepare commonly used experiment dataframes."""
        self.pdm = self.create_parent_distribution_matrix()
        self.esc_slots = self.extract_esc_slots_data()
        self.g6_enrollment = self.extract_grade_6_enrollment()
    
    def create_parent_distribution_matrix(self):
        """Return a dataframe of all directed edges between schools."""
        get_attrs = lambda node: node['school_attrs'][0]

        sources = [e.source for e in self.graph.es]
        targets = [e.target for e in self.graph.es]

        origins = self.graph.vs[sources]
        destinations = self.graph.vs[targets]

        data = {
            'origin_node_index': sources,
            'origin_school_id': [v['school_id'] for v in origins],
            'origin_sector': [get_attrs(v)['sector'] for v in origins],
            'origin_coc': [get_attrs(v)['modified coc'] for v in origins],
            'destination_node_index': targets,
            'destination_school_id': [v['school_id'] for v in destinations],
            'destination_sector': [get_attrs(v)['sector'] for v in destinations],
            'destination_coc': [get_attrs(v)['modified coc'] for v in destinations],
            'esc_participating': [get_attrs(v).get('esc_participating') for v in destinations],
            'road_distance': [e['length'] for e in self.graph.es]
        }

        return pd.DataFrame(data)

    def extract_esc_slots_data(self):
        """Compile ESC slot information for participating schools."""
        map_id_idx = self._get_esc_id_node_map()
        esc_ids_str = list(map_id_idx.keys())

        priv_dest = self._prepare_private_esc_subset(esc_ids_str)

        esc_slots_pvt = self._load_esc_slots_data_from_excel()

        esc_mrg = priv_dest.merge(
            esc_slots_pvt,
            left_on="esc_school_id", right_on="esc_school_id",
            how="left"
        )

        sum_cols = esc_mrg.loc[:, "slot_type_addon":].columns
        esc_mrg['total_esc_slots'] = esc_mrg.loc[:, sum_cols].sum(axis=1)
        esc_mrg['seats_minus_enrollment'] = esc_mrg['seats_jhs'] - esc_mrg['enrollment_jhs']
        esc_mrg = esc_mrg.drop(columns=sum_cols)
        esc_mrg['destination_node_index'] = esc_mrg['school_id'].map(map_id_idx)

        return esc_mrg

    def _get_esc_id_node_map(self) -> Dict[str, int]:
        """Return mapping of ESC school IDs to graph node indices."""
        mask = (
            (self.pdm['destination_sector'] == 'Private')
            & (self.pdm['esc_participating'] == 1)
        )
        ids = self.pdm.loc[mask, 'destination_school_id'].astype(str)
        idxs = self.pdm.loc[mask, 'destination_node_index']
        return dict(zip(ids, idxs))

    def _prepare_private_esc_subset(self, esc_ids: List[str]) -> pd.DataFrame:
        """Return private school records for ESC participants."""
        mask = self.private.index.isin(esc_ids)
        priv_dest = self.private.loc[mask, [
            'esc_school_id', 'modified coc', 'enrollment_jhs', 'seats_jhs', 'esc_(total)'
        ]].copy()
        priv_dest['esc_school_id'] = priv_dest['esc_school_id'].astype(int).astype('string')
        return priv_dest.reset_index()

    def _filter_nodes(
        self,
        sector: str,
        cocs: List[str],
        participating: Optional[int] = None,
        geography_filter: bool = True,
    ):
        """Return nodes filtered by sector, offering type and geography."""
        get_attrs = lambda node: node['school_attrs'][0]
        nodes = [
            n for n in self.graph.vs
            if get_attrs(n)['sector'] == sector
            and get_attrs(n)['modified coc'] in cocs
        ]
        if participating is not None:
            nodes = [n for n in nodes if get_attrs(n).get('esc_participating') == participating]
        if geography_filter:
            if isinstance(self.geography['name'], list):
                geography_name = self.geography['name']
            else:
                geography_name = [self.geography['name']]
            nodes = [n for n in nodes if get_attrs(n)[self.geography['level']] in geography_name]
        return nodes

    def _get_nearby_es_nodes(self, jhs_node, es_cocs, distance):
        """Return public elementary schools near a JHS node."""
        get_attrs = lambda node: node['school_attrs'][0]
        incident_edges = jhs_node.incident(mode='in')
        nearby_edges = [e for e in incident_edges if e['length'] <= distance]
        return [
            self.graph.vs[e.source]
            for e in nearby_edges
            if get_attrs(self.graph.vs[e.source])['sector'] == 'Public'
            and get_attrs(self.graph.vs[e.source])['modified coc'] in es_cocs
        ]

    def _transfer_enrollment(self, es_node, jhs_idx, enrollment_df, graph):
        """Move Grade 6 enrollment from ES node to a JHS vertex."""
        es_id = es_node['school_id']

        # Ensure the lookup uses the same type as the enrollment DataFrame index
        if len(enrollment_df.index) > 0:
            index_example = enrollment_df.index[0]
            try:
                es_id_lookup = type(index_example)(es_id)
            except (ValueError, TypeError):
                es_id_lookup = es_id
        else:
            es_id_lookup = es_id

        try:
            g6 = enrollment_df.loc[es_id_lookup, 'enr_grade_6']
        except KeyError:
            warnings.warn(
                f"School ID {es_id} not found in enrollment data", RuntimeWarning
            )
            return
        if g6 > 0 and not math.isnan(g6):
            enrollment_df.loc[es_id_lookup, 'enr_grade_6'] -= g6
            graph.vs[jhs_idx]['school_attrs'][0]['enrollment_jhs'] += g6

    def _compute_mean_congestion(self, graph, jhs_cocs):
        """Return mean congestion score for public JHS nodes in the geography."""
        get_attrs = lambda node: node['school_attrs'][0]
        jhs_nodes = self._filter_nodes('Public', jhs_cocs, geography_filter=True)
        jhs_nodes = [graph.vs[n.index] for n in jhs_nodes]
        scores = []
        for node in jhs_nodes:
            attrs = get_attrs(node)
            seats = attrs.get('seats_jhs')
            enrollment = attrs.get('enrollment_jhs')
            if seats and seats > 0 and enrollment is not None:
                scores.append(enrollment / seats)
        return float(np.nan) if not scores else float(np.mean(scores))
        
    def _load_esc_slots_data_from_excel(self):
        """Load ESC slot allocation sheets from the datasets folder."""
        if hasattr(self, "_esc_slots_cache"):
            return self._esc_slots_cache

        dir_path = cf.get_path("private_data") # "../datasets/private"
        filenames = os.listdir(dir_path)
    
        path_slots = [
            os.path.join(dir_path, fname) for fname in filenames 
            if 'Alphalist' in fname
        ]
        
        dfs_slots = []
        slot_types = ['addon','fixed','incentive']
        for i, path in enumerate(path_slots):
            df = pd.read_csv(path)
            df.columns = ['esc_school_id','school_name','esc_slots']
            df['esc_school_id'] = df['esc_school_id'].astype(int).astype('string')
            df['slot_type'] = slot_types[i]
            dfs_slots.append(df)
        
        esc_slots = pd.concat(dfs_slots)
        esc_slots_pvt = esc_slots.pivot_table(
            index='esc_school_id',
            columns='slot_type',
            values='esc_slots',
            aggfunc='sum'
        )
        esc_slots_pvt.columns = [
            'slot_type_addon','slot_type_fixed','slot_type_incentive',
        ]
        esc_slots_pvt = esc_slots_pvt.reset_index()

        self._esc_slots_cache = esc_slots_pvt
        return esc_slots_pvt

    def extract_grade_6_enrollment(self):
        """Return Grade 6 enrollment totals for elementary schools."""
        # Get the Grade 6 enrollment of our feeder ES schools and merge
        raw_enrollment = self.kb.enrollment.copy()
        pivot_enr = raw_enrollment.pivot_table(
            index='school_id',
            columns='grade_level',
            values='count_enrollment',
            aggfunc='sum'
        )
        g6_enr = pivot_enr[['Grade 6']]
        g6_enr.columns = ['enr_grade_6']
        g6_enr = g6_enr.reset_index()
        g6_enr['school_id'] = g6_enr['school_id'].astype(int)

        return g6_enr

    def _initialize_experiment(self, reuse_graph=None):
        """Collect common filters and base data for an experiment run."""
        get_attrs = lambda node: node['school_attrs'][0]
        
        # Dataframe where we will deduct G6 enrollment from
        dist_g6_enr = self.g6_enrollment.copy()
        dist_g6_enr = dist_g6_enr.set_index('school_id')

        # Relevant when running Redistribution experiment
        esc_slots = self.esc_slots.copy().set_index('school_id')
        
        # This graph will record the transition of G6 learners from congested Public JHS to Private ESC
        distribution_dg = copy.deepcopy(reuse_graph if reuse_graph else self.graph)
        
        # Used as filters
        es_cocs = ['All Offering', 'ES and JHS', 'Purely ES']
        jhs_cocs = ['All Offering', 'ES and JHS', 'JHS with SHS', 'Purely JHS']
        nearby_distance = 5_000

        public_jhs_nodes = self._filter_nodes('Public', jhs_cocs)
        public_jhs_nodes_idxs = [n.index for n in public_jhs_nodes]

        esc_jhs_nodes = self._filter_nodes('Private', jhs_cocs, participating=1, geography_filter=False)
        esc_jhs_node_idxs = [n.index for n in esc_jhs_nodes]
        
        current_jhs_congestion_scores = [(get_attrs(node)['enrollment_jhs'] / get_attrs(node)['seats_jhs']) for node in public_jhs_nodes]
        current_jhs_congestion_score = np.nanmean(current_jhs_congestion_scores)

        return {
            "g6_enrollment": dist_g6_enr,
            "distribution_graph": distribution_dg,
            "filters": {
                "es_cocs":es_cocs,
                "jhs_cocs":jhs_cocs,
                "distance":nearby_distance,
            },
            "public_jhs": {
                "nodes":public_jhs_nodes, "node_idxs":public_jhs_nodes_idxs,
            },
            "esc_jhs": {
                "nodes":esc_jhs_nodes, "node_idxs":esc_jhs_node_idxs,
            },
            "current_jhs_congestion": current_jhs_congestion_score,
            "esc_slots": esc_slots,
        }

    def run_baseline_experiment(self, exp_package):
        """Simulate moving Grade 6 cohorts to their nearest JHSs."""
        get_attrs = lambda node: node['school_attrs'][0]

        public_jhs_nodes = exp_package["public_jhs"]["nodes"]
        es_cocs = exp_package["filters"]["es_cocs"]
        jhs_cocs = exp_package["filters"]["jhs_cocs"]
        nearby_distance = exp_package["filters"]["distance"]
        dist_g6_enr = exp_package["g6_enrollment"]
        distribution_dg = exp_package["distribution_graph"]

        for jhs_node in public_jhs_nodes:
            es_nodes = self._get_nearby_es_nodes(jhs_node, es_cocs, nearby_distance)
            for es_node in es_nodes:
                # print(f"DEBUG: es_node enrollment: {get_attrs(es_node)}")
                self._transfer_enrollment(es_node, jhs_node.index, dist_g6_enr, distribution_dg)

        baseline_jhs_congestion_score = self._compute_mean_congestion(
            distribution_dg, jhs_cocs
        )
        return baseline_jhs_congestion_score

    def track_actual_jhs_enrollment(
        self, graph: ig.Graph, jhs_cocs: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, int]:
        """Calculate Grade 6 enrollment that entered junior high schools.

        This utility inspects the provided graph after an experiment run and
        compares it with the baseline network stored on initialization. It
        sums the increase in ``enrollment_jhs`` for all nodes classified as
        junior high schools and returns both per-school counts and the overall
        total.

        Parameters
        ----------
        graph : igraph.Graph
            Graph reflecting the state of the network after an experiment
            (typically the ``distribution_graph`` produced by
            :meth:`_initialize_experiment` and subsequently modified by
            :meth:`run_baseline_experiment`).
        jhs_cocs : list of str, optional
            Course-offering classifications that should be treated as junior
            high schools. If ``None``, a default list of JHS classifications is
            used.

        Returns
        -------
        pandas.DataFrame
            Table with columns ``school_id`` and ``added_grade_6`` listing the
            number of Grade 6 learners that entered each JHS. Schools with no
            additional learners are omitted.
        int
            Total number of Grade 6 learners that entered junior high schools
            in the provided graph.

        Examples
        --------
        >>> exp_pkg = exp._initialize_experiment()
        >>> exp.run_baseline_experiment(exp_pkg)
        >>> df, total = exp.track_actual_jhs_enrollment(exp_pkg['distribution_graph'])
        >>> total
        1234
        """

        get_attrs = lambda node: node['school_attrs'][0]
        if jhs_cocs is None:
            jhs_cocs = ['All Offering', 'ES and JHS', 'JHS with SHS', 'Purely JHS']

        records = []
        total_added = 0

        for node in self.graph.vs:
            attrs = get_attrs(node)
            if attrs['modified coc'] in jhs_cocs:
                idx = node.index
                base_enrollment = get_attrs(self.graph.vs[idx]).get('enrollment_jhs', 0) or 0
                new_enrollment = get_attrs(graph.vs[idx]).get('enrollment_jhs', 0) or 0
                diff = new_enrollment - base_enrollment
                if diff > 0:
                    records.append({
                        'school_id': graph.vs[idx]['school_id'],
                        'added_grade_6': int(diff)
                    })
                    total_added += int(diff)

        return pd.DataFrame(records), total_added

    def run_redistribution_experiment(self, exp_package):
        """Allocate ESC slots based on congestion pressure around schools."""
        get_attrs = lambda node: node['school_attrs'][0]
        
        # exp_package = self._initialize_experiment()
        public_jhs_nodes = exp_package["public_jhs"]["nodes"]
        public_jhs_nodes_idxs = exp_package["public_jhs"]["node_idxs"]
        esc_jhs_nodes = exp_package["esc_jhs"]["nodes"]
        es_cocs = exp_package["filters"]["es_cocs"]
        jhs_cocs = exp_package["filters"]["jhs_cocs"]
        nearby_distance = exp_package["filters"]["distance"]
        redist_g6_enr = exp_package["g6_enrollment"]
        distribution_dg = exp_package["distribution_graph"]
        redist_esc_slots = exp_package["esc_slots"]
        
        # Enhanced slot allocation with surplus redistribution
        redistribution_results = {}
        touched_es_school_ids = []

        itr = 0
        for esc_node in esc_jhs_nodes:
            esc_sch_name = get_attrs(esc_node)['school_name']
            esc_sch_id = esc_node['school_id']
        
            try:
                # FIX 1: Better error handling for missing ESC data
                if str(esc_sch_id) not in redist_esc_slots.index:
                    self._vprint(f"Warning: ESC school {esc_sch_id} not found in slots data")
                    continue
                
                esc_total_slots = redist_esc_slots.loc[str(esc_sch_id), 'total_esc_slots']
                
                # FIX 2: Skip schools with 0 or NaN slots
                if pd.isna(esc_total_slots) or esc_total_slots <= 0:
                    self._vprint(f"Skipping {esc_sch_name}: {esc_total_slots} slots")
                    continue
                
                self._vprint(f"\nESC school name: {esc_sch_name} ({esc_sch_id})")
                self._vprint(f"Total slots: {esc_total_slots}")
            
                incident_edges = esc_node.incident(mode='in')
                nearby_incident_edges = [edge for edge in incident_edges if edge['length'] <= nearby_distance]
                nearby_public_nodes = [self.graph.vs[edge.source] for edge in nearby_incident_edges]
                nearby_public_es_edges = [
                    edge for edge in nearby_incident_edges
                    if get_attrs(self.graph.vs[edge.source])['sector'] == 'Public'
                    and get_attrs(self.graph.vs[edge.source])['modified coc'] in es_cocs
                ]
                
                # Handle case when no nearby public ES schools exist
                if len(nearby_public_es_edges) == 0:
                    nearby_incident_edges = [edge for edge in incident_edges if edge['length'] <= nearby_distance + 2_000]
                    nearby_public_nodes = [self.graph.vs[edge.source] for edge in nearby_incident_edges]
                    nearby_public_es_edges = [
                        edge for edge in nearby_incident_edges
                        if get_attrs(self.graph.vs[edge.source])['sector'] == 'Public'
                        and get_attrs(self.graph.vs[edge.source])['modified coc'] in es_cocs
                    ]
                
                nearby_public_es_nodes = [
                    node for node in nearby_public_nodes
                    if get_attrs(node)['sector'] == 'Public'
                    and get_attrs(node)['modified coc'] in es_cocs
                ]
                # vprint(f"Count of nearby public ES: {len(nearby_public_es_nodes)}")
            
                # Calculate community congestion scores
                community_scores = {}
                for es_node in nearby_public_es_nodes:
                    es_node_idx = es_node.index
                    es_sch_id = es_node['school_id']
                    touched_es_school_ids.append(es_sch_id)
                    
                    incident_edges = es_node.incident(mode='out')
                    incident_nodes = [self.graph.vs[edge.target] for edge in incident_edges]
                    incident_public_jhs_nodes = [node for node in incident_nodes if node.index in public_jhs_nodes_idxs]
                
                    # FIX 3: Handle empty JHS lists
                    if len(incident_public_jhs_nodes) == 0:
                        self._vprint(f"Warning: No JHS connections for ES {es_sch_id}")
                        continue
                        
                    congestion_scores = [(get_attrs(node)['enrollment_jhs'] / get_attrs(node)['seats_jhs']) 
                                       for node in incident_public_jhs_nodes]
                    # Filter out NaN values
                    valid_scores = [score for score in congestion_scores if not pd.isna(score)]
                    
                    if len(valid_scores) == 0:
                        self._vprint(f"Warning: No valid congestion scores for ES {es_sch_id}")
                        continue
                        
                    community_jhs_congestion = np.mean(valid_scores)
                    community_scores[es_node_idx] = community_jhs_congestion

                # FIX 4: Skip if no valid community scores
                if len(community_scores) == 0:
                    self._vprint(f"No valid community scores for {esc_sch_name}")
                    continue
                
                # MULTI-ROUND ALLOCATION WITH SURPLUS REDISTRIBUTION
                remaining_slots = esc_total_slots
                allocation_round = 1
        
                # display(community_scores)
                while remaining_slots > 0:
                    self._vprint(f"Allocation Round {allocation_round}: {remaining_slots} slots remaining")
                    
                    # Filter schools that still have G6 enrollment available
                    available_schools = {}
                    for node_idx, score in community_scores.items():
                        es_sch_id = self.graph.vs[node_idx]['school_id']
                        current_g6_enr = redist_g6_enr.loc[es_sch_id, 'enr_grade_6']
                        if current_g6_enr > 0:
                            available_schools[node_idx] = score
                    
                    if not available_schools:
                        self._vprint(f"No schools with available G6 enrollment. {remaining_slots} slots unused.")
                        break
                    
                    # FIX 5: Better handling of remaining slots conversion
                    try:
                        remaining_slots_int = int(remaining_slots) if not pd.isna(remaining_slots) else 0
                        if remaining_slots_int <= 0:
                            break
                    except (ValueError, TypeError):
                        self._vprint(f"Error converting remaining_slots {remaining_slots} to int")
                        break
                    
                    # Distribute remaining slots among available schools
                    current_allocation = self._distribute_slots(available_schools, remaining_slots_int)
                    
                    # Apply allocation with enrollment constraints and track surplus
                    actual_allocations = {}
                    total_surplus = 0
                    
                    for node_idx, allocated_slots in current_allocation.items():
                        es_sch_id = self.graph.vs[node_idx]['school_id']
                        current_g6_enr = redist_g6_enr.loc[es_sch_id, 'enr_grade_6']
                        
                        # Cap allocation to available enrollment
                        actual_allocation = min(allocated_slots, current_g6_enr)
                        surplus = allocated_slots - actual_allocation
                        
                        if actual_allocation > 0:
                            # Update enrollment
                            redist_g6_enr.loc[es_sch_id, 'enr_grade_6'] -= actual_allocation
                            
                            # Track total allocations
                            if node_idx in redistribution_results:
                                redistribution_results[node_idx] += actual_allocation
                            else:
                                redistribution_results[node_idx] = actual_allocation
                            
                            actual_allocations[node_idx] = actual_allocation
                        
                        total_surplus += surplus
                    
                    self._vprint(f"Round {allocation_round} results: Allocated {sum(actual_allocations.values())}, Surplus: {total_surplus}")
                    
                    # Update remaining slots for next round
                    remaining_slots = int(total_surplus)
                    allocation_round += 1
                    
                    # Safety check to prevent infinite loops
                    if allocation_round > 10:  # Maximum 10 rounds
                        self._vprint(f"Maximum allocation rounds reached. {remaining_slots} slots remain unused.")
                        break
            
            except Exception as e:
                self._vprint(
                    f"\n* * * itr:{itr} {esc_node.index} {type(e).__name__}: {str(e)} * * *"
                )
            itr += 1
        
        return {
            "redistribution_results": redistribution_results,
            "redistributed_g6_enrollment": redist_g6_enr,
            "involved_es_school_ids": touched_es_school_ids,
        }
    
    def _distribute_slots(self, scores, total_slots, distribution_results=None):
        """
        Distribute slots proportionally based on node scores.
        
        Args:
            scores: Dictionary with node indices as keys and scores as values
            total_slots: Total number of slots to distribute (default: 100)
            distribution_results: Dictionary to accumulate results across multiple calls (default: None)
        
        Returns:
            Dictionary with node indices as keys and allocated slots as values
        """
        if distribution_results is None:
            distribution_results = {}
            
        # Calculate total score
        total_score = sum(scores.values())
        
        # Calculate proportional allocation (as floats)
        nodes = np.array(list(scores.keys()))
        score_array = np.array(list(scores.values()), dtype=float)
        proportional = (score_array / total_score) * total_slots
        allocated_slots = dict(zip(nodes, proportional.astype(int)))
        fractional_parts = proportional - np.floor(proportional)
        
        # Calculate remaining slots due to rounding down
        allocated_total = sum(allocated_slots.values())
        remaining_slots = int(total_slots - allocated_total)
        if remaining_slots > 0:
            order = np.argsort(fractional_parts)[::-1]
            for i in range(remaining_slots):
                node = nodes[order[i % len(nodes)]]
                allocated_slots[node] += 1
        
        # Update distribution_results with current allocation
        for node, slots in allocated_slots.items():
            if node in distribution_results:
                distribution_results[node] += slots
            else:
                distribution_results[node] = slots
        
        return distribution_results  # Return the accumulated results, not just current allocation

    def _vprint(self, *args, **kwargs):
        """Print only when ``VERBOSE`` is ``True``."""
        if VERBOSE:
            print(*args, **kwargs)

    def optimize_esc_slots(self):
        """
        Redistribute total ESC slots across ESC-participating schools based on
        congestion pressure from nearby public JHSs, without changing the total
        slot count.
        """
        get_attrs = lambda node: node['school_attrs'][0]
        esc_jhs_nodes = [
            node for node in self.graph.vs
            if get_attrs(node)['sector'] == 'Private'
            and get_attrs(node)['modified coc'] in ['All Offering', 'ES and JHS', 'JHS with SHS', 'Purely JHS']
            and get_attrs(node).get('esc_participating') == 1
        ]

        if type(self.geography['name']) != list:
            geography_name = [self.geography['name']]
        else:
            geography_name = self.geography['name']

        public_jhs_nodes = [
            node for node in self.graph.vs
            if get_attrs(node)['sector'] == 'Public'
            and get_attrs(node)['modified coc'] in ['All Offering', 'ES and JHS', 'JHS with SHS', 'Purely JHS']
            and get_attrs(node)[self.geography['level']] in geography_name
        ]
        public_jhs_node_idxs = [node.index for node in public_jhs_nodes]
    
        # Step 1: Calculate community congestion score for each ESC school
        esc_pressure_scores = {}
        for esc_node in esc_jhs_nodes:
            esc_id = str(esc_node['school_id'])
            
            # Find nearby public JHSs
            incident_edges = esc_node.incident(mode='in')
            nearby_edges = [edge for edge in incident_edges if edge['length'] <= 5000]
            nearby_sources = [self.graph.vs[edge.source] for edge in nearby_edges]
            nearby_public_jhs = [
                node for node in nearby_sources if node.index in public_jhs_node_idxs
            ]
            
            if not nearby_public_jhs:
                esc_pressure_scores[esc_id] = 0
                continue
            
            # Compute average congestion score
            congestion_scores = []
            for jhs_node in nearby_public_jhs:
                attrs = get_attrs(jhs_node)
                seats = attrs.get('seats_jhs', 1)
                enrollment = attrs.get('enrollment_jhs', 0)
                if seats and seats > 0:
                    congestion_scores.append(enrollment / seats)
            
            score = np.mean(congestion_scores) if congestion_scores else 0
            esc_pressure_scores[esc_id] = score
    
        # Step 2: Normalize and allocate slots proportionally to pressure scores
        total_slots = self.esc_slots['total_esc_slots'].sum()
        score_sum = sum(esc_pressure_scores.values())
    
        proposed_allocations = {}
        fractional_parts = {}
        for esc_id, score in esc_pressure_scores.items():
            if score_sum > 0:
                raw_allocation = (score / score_sum) * total_slots
            else:
                raw_allocation = total_slots / len(esc_pressure_scores)  # fallback: even dist.
    
            proposed_allocations[esc_id] = int(raw_allocation)
            fractional_parts[esc_id] = raw_allocation - int(raw_allocation)
    
        # Step 3: Distribute leftover slots based on fractional remainders
        allocated_so_far = sum(proposed_allocations.values())
        remaining = int(total_slots - allocated_so_far)
    
        sorted_by_fraction = sorted(fractional_parts.items(), key=lambda x: x[1], reverse=True)
        for i in range(remaining):
            esc_id = sorted_by_fraction[i % len(sorted_by_fraction)][0]
            proposed_allocations[esc_id] += 1
    
        # Step 4: Replace total_esc_slots with optimized values
        new_esc_df = self.esc_slots.copy()
        new_esc_df['optimized_esc_slots'] = new_esc_df['school_id'].map(proposed_allocations).fillna(0).astype(int)
        new_esc_df = new_esc_df.rename(columns={'total_esc_slots': 'original_esc_slots'})
        new_esc_df['total_esc_slots'] = new_esc_df['optimized_esc_slots']
        new_esc_df = new_esc_df.drop(columns=['optimized_esc_slots'])
    
        return new_esc_df

    def run_looped_esc_slot_optimization(self, max_iterations=5, tolerance=1e-5, verbose=True):
        """Iteratively optimize ESC slot allocation and track outcomes.

        Each iteration performs a redistribution experiment, measures the
        resulting congestion and updates the slot distribution until the
        congestion score converges or ``max_iterations`` is reached. The final
        return value now also includes the number of Grade 6 learners that
        entered junior high schools.
        """
        previous_score = None
        history: List[float] = []
        prev_slots_df = self.esc_slots.copy()

        # Initial setup
        current_esc_slots = self.esc_slots.copy()

        for i in range(max_iterations):
            if verbose:
                print(f"\n🔁 Iteration {i+1}/{max_iterations}")

            # Set up experiment package with current slot distribution
            experiment_package = self._initialize_experiment()
            experiment_package["esc_slots"] = current_esc_slots.set_index("school_id")

            # Run redistribution
            results = self.run_redistribution_experiment(experiment_package)
            experiment_package["g6_enrollment"] = results["redistributed_g6_enrollment"]

            # Evaluate resulting congestion
            congestion_score = self.run_baseline_experiment(experiment_package)
            history.append(congestion_score)

            if verbose:
                print(f"📉 Congestion Score: {congestion_score:.6f}")

            # Check convergence
            if previous_score is not None and abs(previous_score - congestion_score) < tolerance:
                if verbose:
                    print(f"✅ Converged: Δ = {abs(previous_score - congestion_score):.6f} < {tolerance}")
                break

            previous_score = congestion_score

            # Use current graph state to recompute ESC pressure and optimize slots
            self.esc_slots = current_esc_slots  # Update self.esc_slots so optimize_esc_slots uses latest distribution
            new_esc_slots = self.optimize_esc_slots()

            if prev_slots_df['total_esc_slots'].equals(new_esc_slots['total_esc_slots']):
                if verbose:
                    print("✅ Slot allocation converged. Stopping early.")
                current_esc_slots = new_esc_slots
                break

            prev_slots_df = new_esc_slots
            current_esc_slots = new_esc_slots

        if verbose:
            print("\n📊 Optimization History:")
            for i, score in enumerate(history):
                print(f"  Iter {i+1}: {score:.6f}")

        # Track actual Grade 6 movement into JHS for the final iteration
        jhs_enrollment_df, total_jhs_enrollment = self.track_actual_jhs_enrollment(
            experiment_package["distribution_graph"]
        )

        return {
            "final_congestion_score": history[-1],
            "iterations": len(history),
            "history": history,
            "optimized_esc_slots": current_esc_slots,
            "total_jhs_enrollment": total_jhs_enrollment,
            "jhs_enrollment_by_school": jhs_enrollment_df,
        }

    def summarize_esc_access(self, distance_threshold: int = 5000) -> pd.DataFrame:
        """Compute congestion around ESC-reachable elementary schools.

        Parameters
        ----------
        distance_threshold : int, optional
            Road distance in meters for considering two schools "nearby".

        Returns
        -------
        pandas.DataFrame
            Table containing one row per ESC school with the average congestion
            score of junior high schools surrounding reachable elementary
            schools. Columns include ``esc_school_id``, ``nearby_es_count``,
            ``avg_jhs_congestion`` and ``nearby_jhs_count_per_es`` which is a
            mapping of elementary school IDs to the number of nearby JHS.
        """

        get_attrs = lambda node: node['school_attrs'][0]

        jhs_cocs = ['All Offering', 'ES and JHS', 'JHS with SHS', 'Purely JHS']
        es_cocs = ['All Offering', 'ES and JHS', 'Purely ES']

        esc_nodes = [
            v for v in self.graph.vs
            if get_attrs(v)['sector'] == 'Private'
            and get_attrs(v)['modified coc'] in jhs_cocs
            and get_attrs(v).get('esc_participating') == 1
        ]

        records: List[Dict] = []

        for esc_node in esc_nodes:
            esc_id = esc_node['school_id']

            incident_edges = esc_node.incident(mode='in')
            es_edges = [
                e for e in incident_edges
                if self.graph.vs[e.source]['school_attrs'][0]['modified coc'] in es_cocs
                and e['length'] <= distance_threshold
            ]
            es_nodes = [self.graph.vs[e.source] for e in es_edges]
            num_es = len(es_nodes)

            all_jhs_scores: List[float] = []
            jhs_count_map: Dict[int, int] = {}

            for es_node in es_nodes:
                es_id = es_node['school_id']

                out_edges = es_node.incident(mode='out')
                jhs_edges = [
                    ed for ed in out_edges
                    if self.graph.vs[ed.target]['school_attrs'][0]['modified coc'] in jhs_cocs
                    and ed['length'] <= distance_threshold
                ]
                jhs_nodes = [self.graph.vs[ed.target] for ed in jhs_edges]
                jhs_count_map[es_id] = len(jhs_nodes)

                for jhs in jhs_nodes:
                    attrs = get_attrs(jhs)
                    seats = attrs.get('seats_jhs')
                    enrollment = attrs.get('enrollment_jhs')
                    if seats and seats > 0 and enrollment is not None:
                        all_jhs_scores.append(enrollment / seats)

            avg_score = float(np.nan) if len(all_jhs_scores) == 0 else float(np.mean(all_jhs_scores))

            records.append({
                'esc_school_id': esc_id,
                'nearby_es_count': num_es,
                'avg_jhs_congestion': avg_score,
                'nearby_jhs_count_per_es': jhs_count_map,
            })

        df = pd.DataFrame(records)
        self.es_jhs_congestion = df
        return df
