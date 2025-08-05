# Project PAARAL (Phase 2)

## High-Level Overview
Project PAARAL is a geospatial decision support system for analyzing school accessibility and planning educational subsidies in the Philippines. The repository revolves around two main workflows demonstrated in accompanying Jupyter notebooks.

### 1. Graph Generation to Deterministic Representation (`1.0-graph-generation-to-deterministic.ipynb`)
This notebook builds the foundation of the school network.
- **`modules/datasets.py`** – `KnowledgeBase` consolidates public and private school datasets such as coordinates, enrollment, seat capacity, shifting schedules, and GASTPE allocations.
- **`modules/map_resources.py`** – `MapResources` organizes administrative boundaries, validates school coordinates, and assembles regional road networks.
- **`modules/optimized_network_builder.py`** – `OptimizedSchoolNetworkBuilder` converts the road network into an iGraph structure, computes shortest paths between schools, and outputs deterministic routing results.

### 2. Discrete Choice Modeling (`2.0-discrete-choice-modeling.ipynb`)
This notebook explores how students might choose between public and private schools under different policy scenarios.
- **`modules/dce.py`** – Provides utilities to load ESC slot data, convert network flows to origin–destination pairs, compute distance matrices, simulate binary choice sets, estimate a binomial logit model, and evaluate subsidy policy scenarios.

## Repository Structure
```
project_paaral/
├── config/      # for dir & file paths
├── data/        # raw datasets (empty by default)
├── modules/     # core Python modules
├── output/      # generated networks and analysis results (empty by default)
├── 1.0-graph-generation-to-deterministic.ipynb
├── 2.0-discrete-choice-modeling.ipynb
└── requirements.txt
```

## Data and Outputs
The `data` and `output` directories are placeholders. Download the required files from Google Drive and place them in the corresponding folders:

- Data directory: [https://drive.google.com/drive/folders/1K9QuUvo47rbFZ4LHtN6PEzVuRoR91PL4?usp=sharing]
- Output directory: [https://drive.google.com/drive/folders/1d31485Ib4Pob-A2vqdIlRN64tIvmqyr4?usp=sharing]

## Getting Started
```
git clone <repository-url>
cd project_paaral
pip install -r requirements.txt
```
Run the notebooks in sequence to rebuild the knowledge base, generate the network, and perform discrete choice modeling.

---
