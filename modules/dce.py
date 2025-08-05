import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import igraph as ig
import re
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
from scipy.spatial.distance import cdist
from statsmodels.formula.api import logit
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper

import warnings
warnings.filterwarnings('ignore')

def load_esc_slots_data_from_excel(project_root):
    """Load ESC slots data from Excel files.
    
    Parameters
    ----------
    project_root : Path
        Path to the project root directory (project_paaral_2)
    
    Returns
    -------
    pandas.DataFrame
        Pivot table of ESC slots by type
    """
    dir_path = project_root / "datasets" / "private"
    
    # Check if directory exists
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    
    filenames = os.listdir(dir_path)
    print(f"Looking for 'Alphalist' files in: {dir_path}")
    
    path_slots = [
        dir_path / fname for fname in filenames 
        if 'Alphalist' in fname
    ]
    
    if not path_slots:
        print(f"Available files: {filenames}")
        raise FileNotFoundError("No files containing 'Alphalist' found in directory")
    
    print(f"Found {len(path_slots)} Alphalist files: {[p.name for p in path_slots]}")
    
    dfs_slots = []
    slot_types = ['addon','fixed','incentive']
    
    if len(path_slots) != len(slot_types):
        print(f"Warning: Expected {len(slot_types)} files, found {len(path_slots)}")
        print("Files found:", [p.name for p in path_slots])
    
    for i, path in enumerate(path_slots):
        print(f"  Loading: {path.name}")
        df = pd.read_csv(path)
        df.columns = ['esc_school_id','school_name','esc_slots']
        df['esc_school_id'] = df['esc_school_id'].astype(int).astype('string')
        
        # Use slot type if available, otherwise use index
        if i < len(slot_types):
            df['slot_type'] = slot_types[i]
        else:
            df['slot_type'] = f'slot_type_{i}'
            
        dfs_slots.append(df)
    
    esc_slots = pd.concat(dfs_slots)
    esc_slots_pvt = esc_slots.pivot_table(
        index='esc_school_id',
        columns='slot_type',
        values='esc_slots',
        aggfunc='sum'
    )
    
    # Create column names based on actual slot types found
    new_columns = [f'slot_type_{col}' for col in esc_slots_pvt.columns]
    esc_slots_pvt.columns = new_columns
    esc_slots_pvt = esc_slots_pvt.reset_index()
    
    print("✓ ESC slots data loaded successfully")
    return esc_slots_pvt


def extract_flows(graph: ig.Graph):
    """Extract origin-destination flows from a graph network.
    
    This function processes edges in an iGraph network to extract student flow 
    data between schools, focusing on ESC (Educational Service Contracting) 
    beneficiaries and their associated school characteristics.
    
    Parameters
    ----------
    graph : igraph.Graph
        An iGraph object containing school nodes and flow edges. Edges should 
        have 'esc_beneficiaries' attributes, and vertices should have 'school_attrs'
        containing school-level information including sector, coordinates, 
        enrollment, capacity, and program participation data.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing origin-destination flow records with the following columns:
        
        **Flow Identifiers:**
        - origin_school_id, destination_school_id : School ID strings
        - origin_vertex_index, destination_vertex_index : Graph vertex indices
        
        **Beneficiary Counts:**
        - esc_beneficiaries : Number of ESC program beneficiaries
        - total_beneficiaries : Total beneficiaries (currently same as esc_beneficiaries)
        
        **School Context:**
        - origin_school_name, destination_school_name : School names
        - origin_sector, destination_sector : School sector (Public/Private)
        - origin_x, origin_y, destination_x, destination_y : School coordinates
        
        **Program Information:**
        - esc_participating : ESC participation status (Private schools only)
        - esc_amount : ESC subsidy amount
        - esc_fees : Total ESC fees
        - topup_fees : Additional fees beyond ESC subsidy
        
        **Capacity and Enrollment:**
        - seats_jhs_origin, seats_jhs_dest : Junior high school seat capacity
        - enrollment_jhs_origin, enrollment_jhs_dest : Junior high school enrollment
        - enrollment_g6_origin : Grade 6 enrollment at origin school
        
        **Slot Allocation:**
        - slot_type_fixed, slot_type_addon, slot_type_incentive : ESC slot counts by type
        
        Only returns flows where esc_beneficiaries > 0.
    """
    flows_data = []
    for edge in graph.es:
        # Get edge attributes
        edge_attrs = edge.attributes()

        # Extract ESC beneficiaries
        esc_beneficiaries = edge_attrs.get('esc_beneficiaries', 0)

        # Total beneficiaries (learner count)
        total_beneficiaries = 0
        if esc_beneficiaries is not None:
            total_beneficiaries += esc_beneficiaries

        # Only include flows with actual beneficiaries
        if total_beneficiaries > 0:
            # Get source and target vertex indices
            source_idx = edge.source
            target_idx = edge.target

            # Get vertex attributes for context
            source_vertex = graph.vs[source_idx]
            target_vertex = graph.vs[target_idx]

            source_attrs = source_vertex.attributes()
            target_attrs = target_vertex.attributes()

            # Extract school_attrs for sector information
            source_school_attrs = source_attrs.get('school_attrs', [{}][0]
                                                   if source_attrs.get('school_attrs')
                                                   else {})
            target_school_attrs = target_attrs.get('school_attrs', [{}][0]
                                                   if target_attrs.get('school_attrs') else {})

            flow_data = {
                'origin_school_id': str(source_attrs.get('school_id',
                                                         source_idx)),
                'destination_school_id': str(target_attrs.get('school_id',
                                                              target_idx)),
                'origin_vertex_index': source_idx,
                'destination_vertex_index': target_idx,

                # Beneficiary counts
                'esc_beneficiaries': esc_beneficiaries if esc_beneficiaries is not None else 0,
                'total_beneficiaries': total_beneficiaries,

                # School context
                'origin_school_name': source_attrs.get('school_name',
                                                       f'School_{source_idx}'),
                'destination_school_name': target_attrs.get('school_name',
                                                            f'School_{target_idx}'),
                'origin_sector': source_school_attrs[0].get('sector', 'Unknown'),
                'destination_sector': target_school_attrs[0].get('sector', 'Unknown'),
                'origin_x': source_school_attrs[0].get('x', 0),
                'origin_y': source_school_attrs[0].get('y', 0),
                'destination_x':  target_school_attrs[0].get('x', 0),
                'destination_y': target_attrs.get('y', 0),
                'esc_participating': target_school_attrs[0].get('esc_participating', 'Unknown') if target_school_attrs[0].get('sector', 'Unknown') == 'Private' else 0.0,
                'esc_amount': target_school_attrs[0].get('esc_amount', np.nan),
                'esc_fees': target_school_attrs[0].get('esc_(total)', np.nan),
                'seats_jhs_dest': target_school_attrs[0].get('seats_jhs', np.nan),
                'seats_jhs_origin': source_school_attrs[0].get('seats_jhs', np.nan),
                'enrollment_jhs_dest': target_school_attrs[0].get('enrollment_jhs', np.nan),
                'enrollment_jhs_origin': source_school_attrs[0].get('enrollment_jhs', np.nan),
                'enrollment_g6_origin': source_school_attrs[0].get('enrollment_g6', np.nan),
                'topup_fees': target_school_attrs[0].get('topup_fees', np.nan),
                'slot_type_fixed': target_school_attrs[0].get('slot_type_fixed', np.nan),
                'slot_type_addon': target_school_attrs[0].get('slot_type_addon', np.nan),
                'slot_type_incentive': target_school_attrs[0].get('slot_type_incentive', np.nan),

                # # Individual context
                # 'municipality_income_per_household': source_school_attrs[0].get('municipality_income_per_household', np.nan),
            }
            
            flows_data.append(flow_data)
    df_esc_flows = pd.DataFrame(flows_data)
    df_esc_flows = df_esc_flows[df_esc_flows['esc_beneficiaries'] > 0].copy()
    return df_esc_flows


def calculate_distance_matrix(df_flows: pd.DataFrame):
    """Calculate Euclidean distances for origin-destination flow pairs.
    
    This function computes straight-line distances between origin and destination 
    schools for each flow record in the input DataFrame.
    
    Parameters
    ----------
    df_flows : pandas.DataFrame
        DataFrame containing flow records with coordinate information.
        Must include the following columns:
        - 'origin_x' : float, X-coordinate of origin school
        - 'origin_y' : float, Y-coordinate of origin school  
        - 'destination_x' : float, X-coordinate of destination school
        - 'destination_y' : float, Y-coordinate of destination school
    
    Returns
    -------
    pandas.DataFrame
        Copy of input DataFrame with additional column:
        - 'distance_meters' : float, Euclidean distance between origin and 
          destination coordinates in meters (assuming projected coordinates)
    """
    # Ensure we have the required coordinate columns
    required_cols = ['origin_x', 'origin_y', 'destination_x', 'destination_y']
    
    if not all(col in df_flows.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    print(f"Calculating distances for {len(df_flows)} flow records...")
    
    # Calculate Euclidean distance between origin and destination for each flow
    df_flows2 = df_flows.copy()
    df_flows2['distance_meters'] = np.sqrt(
        (df_flows2['destination_x'] - df_flows2['origin_x']) ** 2 + 
        (df_flows2['destination_y'] - df_flows2['origin_y']) ** 2
    )
    
    print(f"✓ Distance calculation complete")
    print(f"✓ Distance range: {df_flows2['distance_meters'].min():.1f}m to {df_flows2['distance_meters'].max():.1f}m")
    print(f"✓ Mean distance: {df_flows2['distance_meters'].mean():.1f}m")
    
    return df_flows2



def simulate_public_private_od(df, public_ratio=0.5, seed=42):
    """Simulate mixed-sector destination schools by reassigning private schools as public.
    
    This function randomly reassigns a specified proportion of private schools in the 
    origin-destination (OD) dataset as public schools. This allows for simulation of 
    scenarios where learners face a mixed choice set of public and private destination 
    schools. ESC-related attributes are updated accordingly to reflect the change in 
    sector designation.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input origin-destination DataFrame containing school flow records and attributes. 
        Must include the following columns:
        - 'destination_school_id' : Unique school identifiers
        - 'destination_sector' : Indicates school sector ('Public' or 'Private')
        - 'esc_amount', 'esc_fees', 'esc_participating' : ESC program fields
    
    public_ratio : float, optional
        Proportion of destination schools to convert to public (between 0 and 1). 
        Default is 0.5 (i.e., 50% of schools become public).
    
    seed : int, optional
        Random seed for reproducibility. Default is 42.
    
    Returns
    -------
    pandas.DataFrame
        A modified copy of the input DataFrame where a proportion of destination 
        schools have been converted to public. ESC-related attributes are set to:
        
        - esc_amount : 0 for public schools
        - esc_fees : 0 for public schools
        - esc_participating : 0.0 for public schools

        The rest of the records remain unchanged.
    """
    np.random.seed(seed)
    df_sim = df.copy()
    
    # Get unique destination schools
    unique_destinations = df_sim['destination_school_id'].unique()
    n_schools = len(unique_destinations)

    # Randomly seelct schools to convert to public
    n_public = int(n_schools * public_ratio)
    public_schools = np.random.choice(unique_destinations, size=n_public,
                                      replace=False)
    print(f"Converting {n_public} out of {n_schools} schools to public ({public_ratio:.1%})")

    # Convert selected schools to public
    df_sim.loc[df_sim['destination_school_id'].isin(public_schools), 'destination_sector'] = 'Public'

    # For public schools, set ESC-related variables appropriately
    public_mask = df_sim['destination_sector'] == 'Public'

    # Set ESC amounts to 0
    df_sim.loc[public_mask, 'esc_amount'] = 0
    df_sim.loc[public_mask, 'esc_fees'] = 0
    df_sim.loc[public_mask, 'esc_participating'] = 0.0

    return df_sim


def predict_choice_probability(results, esc_amount_k, topup_fees_k, distance_km,):
    """
    Predict the probability of choosing a private ESC school using estimated DCE coefficients.

    This function calculates the logit score and corresponding choice probability based on a 
    discrete choice model (DCE) fitted with logistic regression. It estimates the likelihood 
    that a student will choose a private school participating in the ESC program, given the 
    financial and geographic characteristics of the option.

    Parameters
    ----------
    results : glm
        The results of the DCE model
    
    esc_amount_k : float
        ESC subsidy amount offered by the private school, in thousands of pesos.

    topup_fees_k : float
        Additional top-up fees charged by the private school beyond the ESC subsidy, 
        in thousands of pesos.

    distance_km : float
        Distance from the student’s origin to the private school, in kilometers.

    Returns
    -------
    float
        The predicted probability (between 0 and 1) that a student chooses a private 
        ESC-participating school under the specified scenario.
    """
    logit_score = (results.params['Intercept'] +
                   results.params['esc_amount_k'] * esc_amount_k +
                   results.params['topup_fees_k'] * topup_fees_k +
                   results.params['distance_km'] * distance_km)
    
    return 1 / (1 + np.exp(-logit_score))


def calculate_scenario_metrics(data, scenario_name):
    """
    Calculate summary policy metrics for a given DCE-based scenario.

    This function computes key indicators that describe the reach, responsiveness, 
    and financial cost of a simulated ESC subsidy scenario. It estimates how many 
    students choose private schools under the scenario, how much the program costs, 
    and the average cost per student served, based on predicted choice probabilities 
    and Grade 6 enrollment.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing origin school-level data, with the following required columns:
        - 'predicted_prob' : Predicted probability of choosing private ESC school
        - 'enrollment_g6_origin' : Grade 6 enrollment (potential ESC-eligible students)
        - 'esc_amount_k' : ESC subsidy amount in thousands of pesos

    scenario_name : str
        Name or label for the scenario being evaluated.

    Returns
    -------
    dict
        Dictionary containing key policy metrics:
        - 'Scenario' : Scenario label
        - 'Total_number_of_origin_schools' : Number of origin schools considered
        - 'Total_potential_students' : Total Grade 6 enrollment across all origins
        - 'Avg_school_response_rate' : Average predicted probability across origin schools
        - 'Program_participation_rate' : Share of students choosing private ESC schools
        - 'Total_students_served' : Total number of students predicted to choose private schools
        - 'Total_program_cost' : Total cost of ESC subsidies under this scenario (in PHP)
        - 'Cost_per_student' : Average ESC cost per student served (in PHP)
    """
    total_areas = len(data)
    avg_origin_response_rate = data['predicted_prob'].mean() # How responsive schools are on average

    # Use Grade 6 enrollment as potential students
    total_potential_students = data['enrollment_g6_origin'].sum()
    
    # Calculate students choosing private under this scenario
    data['students_choosing_private'] = data['predicted_prob'] * data['enrollment_g6_origin']
    total_students_served = data['students_choosing_private'].sum()
    
    # Calculate total program cost
    # data['esc_cost_per_origin'] = data['esc_amount_k'] * data['predicted_prob'] * data['students_choosing_private'] * 1000 # Convert to pesos
    data['esc_cost_per_origin'] = data['esc_amount_k'] * data['students_choosing_private'] * 1000
    total_esc_cost = data['esc_cost_per_origin'].sum()

    # Calculate average cost per student served
    avg_cost_per_student = total_esc_cost / total_students_served if total_students_served > 0 else 0

    # Calculate market penetration (reworded as: program uptake)
    program_uptake = total_students_served / total_potential_students # What % of all students covered
    
    return {
        'Scenario': scenario_name,
        'Total_number_of_origin_schools': total_areas,
        'Total_potential_students': f'{total_potential_students:,.0f}', 
        'Avg_school_response_rate': f'{avg_origin_response_rate:.1%}', # Average across schools
        'Program_participation_rate': f'{program_uptake:.1%}', # System-wide coverage
        'Total_students_served': f'{total_students_served:.0f}',
        'Total_program_cost': f'₱{total_esc_cost:,.0f}',
        'Cost_per_student': f'₱{avg_cost_per_student:,.0f}'
    }







