# Project PAARAL (Phase 2)

## Overview
This repository contains code for analyzing spatial relationships between schools in the Philippines educational system. The project builds geographic network models to understand connections between public and private schools, with a particular focus on mapping path routes and distances between schools within cities and municipalities.

## Key Features
- **Spatial Data Processing**: Processes shapefiles and geodata from the Philippines to define administrative boundaries
- **School Data Management**: Loads and processes school location, enrollment, and seat capacity data
- **Network Building**: Constructs network graphs that model connections between schools based on road networks
- **Routing Analysis**: Calculates shortest paths between schools, particularly analyzing connections between public and private schools
- **Visualization**: Creates maps and network visualizations showing school relationships

## Core Components

### Datasets Module (`datasets.py`)
This module handles loading and preprocessing various school datasets:
- Public and private school coordinates
- Enrollment data
- Seat capacity information
- School shifting schedules
- GASTPE (Government Assistance to Students and Teachers in Private Education) data

### Map Resources (`map_resources.py`)
Manages geographic and administrative data:
- Loads and processes Philippines administrative boundary shapefiles
- Validates and processes school coordinates
- Identifies adjacent municipalities
- Prepares data for network analysis

### Network Building (`optimized_network_builder.py`)
Constructs networks of schools and roads using an optimized builder:
- Converts road data from NetworkX to iGraph for faster computations
- Builds spatial indices for quick lookup of nearby schools
- Calculates shortest paths and distance matrices
- Generates nodal routes and consolidates results for analysis
- See **docs/optimized_network_builder.md** for a detailed overview of this module

## Methodology
The project follows these general steps:
1. Load school data and administrative boundaries
2. Build a road network graph of the target region
3. For each public and private (or origin) school:
   - Define a buffer zone (catchment area)
   - Identify private and other public schools within the buffer
   - Calculate the shortest paths to those schools
   - Generate a distance matrix
   - Build a network representation of school connections

## Dependencies
- pandas/geopandas
- networkx
- osmnx
- igraph
- matplotlib
- shapely
- numpy
- pyproj

## Installation
Clone this repository and install required dependencies:

```bash
git clone [your-repository-url]
cd [repository-name]
pip install -r requirements.txt
```

## Data Requirements
This project requires several datasets that should be organized in a specific directory structure:
- Administrative boundary shapefiles
- School coordinates
- Enrollment records
- Road network data

## Usage
The main workflow is demonstrated in the Jupyter notebook. To replicate the analysis for a different city:
1. Load the required datasets
2. Define the target city or municipality
3. Extract schools within the target area
4. Build the network graph
5. Calculate routes between schools
6. Visualize the results

## Note
The full network calculation process can be computationally intensive, taking several hours for a complete city analysis as shown in the notebook.
