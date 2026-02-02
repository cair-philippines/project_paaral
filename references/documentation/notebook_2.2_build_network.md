# Notebook 2.2: Build School Distance Network

**File:** `notebooks/2.2b.-build-network-and-distance-matrix.ipynb`
**Last Updated:** 2026-01-26
**Status:** Implemented (OSRM-based approach)

---

## Purpose

Build a distance matrix representing road network distances (in meters) between public and private schools in NCR and Region IV-A (CALABARZON). The output is a school-to-school distance table computed using actual road network paths via OSRM (Open Source Routing Machine).

---

## Architecture Overview

### Previous Approach (Deprecated - Memory Issues)
- NetworkX graph from GeoDataFrame road segments
- scipy.sparse.csgraph.dijkstra() for shortest paths
- **Problem**: Kernel crashes due to memory exhaustion (~4-6GB for graph + sparse matrix)
- **Problem**: Directed graph bug (missing reverse edges for bidirectional roads)

### Current Approach (OSRM - Recommended)
- OSRM routing engine runs as separate Docker service
- Python sends HTTP requests to OSRM Table API
- OSRM handles all road network complexity (snapping, routing, one-way streets)
- **Benefits**: No memory issues, accurate road distances, simple Python code

---

## Data Dependencies

### Input Files

| File | Source Section | Description |
|------|----------------|-------------|
| `output/processed_project_bukas_school_information.parquet` | 1.0 | School metadata (61,442 schools) |
| `output/processed_public_school_coordinates.parquet` | 1.1 | Public school coordinates |
| `output/processed_private_school_coordinates.parquet` | 1.1 | Private school coordinates |

### Output Files

| File | Description |
|------|-------------|
| `output/school_distance_matrix_osrm.npy` | NumPy array (8,331 x 8,331) with distances in meters |
| `output/school_distance_matrix_index.json` | School ID list and multi-value ID-to-indices mapping |
| `output/checkpoints/distance_matrix_osrm.npy` | Checkpoint copy of distance matrix |
| `output/checkpoints/distance_matrix_school_ids.json` | Checkpoint copy of school IDs |

**Index JSON Structure:**
```json
{
  "school_ids": ["107900", "107901", ...],  // 8,331 entries (list, preserves matrix order)
  "school_id_to_indices": {                  // 8,307 unique IDs (dict of lists)
    "107900": [0],
    "401748": [6760, 6846],                  // Multi-location school
    ...
  }
}
```

**Notes:**
- We use numpy array format instead of DataFrame/parquet due to memory constraints.
- `school_id_to_indices` maps each school_id to a **list** of indices to handle integrated schools with multiple physical locations.

---

## OSRM Setup for Surface Pro 11 (ARM64)

### Critical: Windows Docker + ARM64 Compatibility

The Surface Pro 11 uses ARM64 architecture. Key considerations:
1. Official OSRM Docker images are AMD64 - use GitHub Container Registry ARM64 image
2. Windows bind mounts have mmap issues - use Docker named volumes
3. Platform must be explicitly specified as `linux/arm64`

### Step 1: Create Docker Named Volume

```bash
docker volume create osrm-data
```

### Step 2: Download Philippines OSM Data

```bash
cd C:\Users\<username>\Documents\Work\innovation-projects\osrm-data
wget https://download.geofabrik.de/asia/philippines-latest.osm.pbf
```

Or download manually from: https://download.geofabrik.de/asia/philippines.html

### Step 3: Copy PBF to Docker Volume

```bash
docker run --rm -v "${PWD}:/source" -v osrm-data:/data alpine cp /source/philippines-latest.osm.pbf /data/
```

### Step 4: Process OSM Data (ARM64 Image)

```bash
# Extract (~2 minutes)
docker run --platform linux/arm64 -t -v osrm-data:/data ghcr.io/project-osrm/osrm-backend:v6.0.0 osrm-extract -p /opt/car.lua /data/philippines-latest.osm.pbf

# Partition (~1 minute)
docker run --platform linux/arm64 -t -v osrm-data:/data ghcr.io/project-osrm/osrm-backend:v6.0.0 osrm-partition /data/philippines-latest.osrm

# Customize (~30 seconds)
docker run --platform linux/arm64 -t -v osrm-data:/data ghcr.io/project-osrm/osrm-backend:v6.0.0 osrm-customize /data/philippines-latest.osrm
```

### Step 5: Update docker-compose-arm.yml

```yaml
services:
  experiments-innovations:
    # ... existing Jupyter config ...
    depends_on:
      - osrm

  osrm:
    image: ghcr.io/project-osrm/osrm-backend:v6.0.0
    platform: linux/arm64
    container_name: osrm-routing
    ports:
      - "5000:5000"
    volumes:
      - osrm-data:/data
    command: osrm-routed --algorithm mld --max-table-size 10000 /data/philippines-latest.osrm
    restart: unless-stopped

# Top-level volumes section (same level as services)
volumes:
  jupyter_home:
    driver: local
  osrm-data:
    external: true
```

### Step 6: Start Services

```bash
docker-compose -f docker-compose-arm.yml up -d
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Platform mismatch warning | Use `ghcr.io/project-osrm/osrm-backend:v6.0.0` with `platform: linux/arm64` |
| mmap file error during extract | Use Docker named volume instead of bind mount |
| OSRM not responding | Check container logs: `docker logs osrm-routing` |
| "volumes additional properties not allowed" | Ensure `osrm-data:` is under top-level `volumes:` section, not inside service |

---

## Workflow Sections

### Section 1.0 - Load Data
Loads school information and coordinates. No changes from original.

### Section 2.1 - Prepare School Coordinates

```python
# Combine public and private school coordinates
df_coords = pd.concat([df_pub, df_priv], ignore_index=True)

# Keep only valid coordinates in regions of interest
df_coords = df_coords[df_coords['coord_valid'] == True].copy()
regions_of_interest = ['NCR', 'Region IV-A']
df_coords = df_coords[df_coords['old_region'].isin(regions_of_interest)].copy()

# Keep required columns - drop_duplicates on full row (school_id, lon, lat)
# This preserves multiple locations for integrated schools (same school_id, different coords)
df_coords = df_coords[['school_id', 'longitude', 'latitude']].drop_duplicates()
df_coords = df_coords.reset_index(drop=True)
```

**Output:** `df_coords` with 8,331 rows (8,307 unique school_ids, 24 schools have 2 locations each)

### Section 2.2 - Query OSRM for Distance Matrix

```python
import requests
import numpy as np
from tqdm import tqdm

OSRM_URL = "http://osrm:5000/table/v1/driving/"
MAX_COORDS_PER_REQUEST = 500

def get_osrm_distance_matrix(coords_list, batch_size=MAX_COORDS_PER_REQUEST):
    """
    Get distance matrix from OSRM Table API.
    Handles batching for large coordinate lists.
    Returns numpy array of distances in meters.
    """
    n = len(coords_list)

    if n <= batch_size:
        # Single request for small datasets
        coord_str = ";".join([f"{lon},{lat}" for lon, lat in coords_list])
        response = requests.get(f"{OSRM_URL}{coord_str}", params={"annotations": "distance"})
        data = response.json()
        distances = np.array(data["distances"], dtype=np.float64)
        distances[distances == None] = np.inf
        return distances

    # Batch processing for large datasets
    full_matrix = np.full((n, n), np.inf, dtype=np.float64)

    for i in tqdm(range(0, n, batch_size), desc="Processing batches"):
        batch_end = min(i + batch_size, n)
        batch_indices = list(range(i, batch_end))

        coord_str = ";".join([f"{lon},{lat}" for lon, lat in coords_list])
        sources = ";".join(map(str, batch_indices))

        response = requests.get(
            f"{OSRM_URL}{coord_str}",
            params={"annotations": "distance", "sources": sources}
        )

        if response.status_code == 200 and response.json()["code"] == "Ok":
            batch_distances = np.array(response.json()["distances"], dtype=np.float64)
            for j, src_idx in enumerate(batch_indices):
                full_matrix[src_idx, :] = batch_distances[j]

    return full_matrix
```

### Section 2.3 - Compute Distance Matrix

```python
coords_list = list(zip(df_coords['longitude'], df_coords['latitude']))
school_ids = df_coords['school_id'].tolist()

distance_matrix = get_osrm_distance_matrix(coords_list)
```

**Performance:** ~5-6 minutes per batch of 500 schools (17 batches for 8,331 schools)

### Section 2.3b - Checkpoint Distance Matrix

```python
# Save immediately after computation
np.save('output/checkpoints/distance_matrix_osrm.npy', distance_matrix)

with open('output/checkpoints/distance_matrix_school_ids.json', 'w') as f:
    json.dump(school_ids, f)
```

### Section 2.3c - Load Checkpoint (Recovery)

```python
# Use this if kernel crashed after 2.3
distance_matrix = np.load('output/checkpoints/distance_matrix_osrm.npy')

with open('output/checkpoints/distance_matrix_school_ids.json', 'r') as f:
    school_ids = json.load(f)
```

### Section 2.4 - Save Distance Matrix (Numpy Format)

```python
import json
from collections import defaultdict

# Save the numpy matrix
np.save('output/school_distance_matrix_osrm.npy', distance_matrix)

# Create multi-value mapping: school_id → list of indices
# This handles schools with multiple locations (ES vs JHS at different coordinates)
school_id_to_indices = defaultdict(list)
for i, sid in enumerate(school_ids):
    school_id_to_indices[sid].append(i)

school_id_to_indices = dict(school_id_to_indices)

# Save index data
with open('output/school_distance_matrix_index.json', 'w') as f:
    json.dump({
        'school_ids': school_ids,  # List preserving matrix order (8,331 entries)
        'school_id_to_indices': school_id_to_indices  # Dict of lists (8,307 unique IDs)
    }, f, indent=2)

# Summary
multi_location = sum(1 for v in school_id_to_indices.values() if len(v) > 1)
print(f"Saved distance matrix: {distance_matrix.shape}")
print(f"School IDs in matrix: {len(school_ids):,}")
print(f"Unique school IDs: {len(school_id_to_indices):,}")
print(f"Schools with multiple locations: {multi_location}")
```

**Key Decisions:**
1. We keep the numpy array format instead of converting to DataFrame. Converting 69M+ pairs to long-format DataFrame caused kernel crashes.
2. We use `school_id_to_indices` (dict of lists) instead of `school_id_to_idx` (dict of int) to handle integrated schools that have the same school_id but different coordinates for ES vs JHS buildings.

### Section 2.5 - Verify Output

```python
import numpy as np
import json

# Load and verify
distance_matrix = np.load('output/school_distance_matrix_osrm.npy')

with open('output/school_distance_matrix_index.json', 'r') as f:
    index_data = json.load(f)
    school_ids = index_data['school_ids']
    school_id_to_indices = index_data['school_id_to_indices']

# Index mapping stats
multi_location = sum(1 for v in school_id_to_indices.values() if len(v) > 1)

print("=== Distance Matrix Summary ===")
print(f"Matrix shape: {distance_matrix.shape}")
print(f"School IDs in matrix: {len(school_ids):,}")
print(f"Unique school IDs: {len(school_id_to_indices):,}")
print(f"Schools with multiple locations: {multi_location}")

# Count valid distances
valid_mask = (distance_matrix < np.inf) & (distance_matrix > 0)
valid_count = valid_mask.sum()
total_pairs = len(school_ids) * (len(school_ids) - 1)

print(f"\nValid distances: {valid_count:,} / {total_pairs:,} ({100*valid_count/total_pairs:.1f}%)")

# Distance statistics
valid_distances = distance_matrix[valid_mask]
print(f"\n=== Distance Statistics (meters) ===")
print(f"Min: {valid_distances.min():,.0f}")
print(f"Max: {valid_distances.max():,.0f}")
print(f"Mean: {valid_distances.mean():,.0f}")
print(f"Median: {np.median(valid_distances):,.0f}")
```

---

## Using the Distance Matrix

### Loading the Matrix

```python
import numpy as np
import json

distance_matrix = np.load('output/school_distance_matrix_osrm.npy')

with open('output/school_distance_matrix_index.json', 'r') as f:
    index_data = json.load(f)
    school_ids = index_data['school_ids']  # List (8,331 entries)
    school_id_to_indices = index_data['school_id_to_indices']  # Dict of lists (8,307 unique IDs)
```

### Helper Functions (Multi-Location Aware)

Schools with multiple locations (integrated schools offering ES and JHS at different buildings) have multiple indices in the matrix. The helper functions below handle this by computing distances for all location pairs.

```python
def get_distance(origin_id, dest_id, method='min'):
    """
    Get distance between two schools.

    Args:
        origin_id: Origin school ID
        dest_id: Destination school ID
        method: How to handle schools with multiple locations
                'min' (default) - shortest distance between any location pair
                'max' - longest distance
                'mean' - average distance

    Returns:
        Distance in meters, or np.inf if no route found
    """
    origin_indices = school_id_to_indices.get(str(origin_id), [])
    dest_indices = school_id_to_indices.get(str(dest_id), [])

    if not origin_indices or not dest_indices:
        return np.inf

    # Get all pairwise distances
    distances = []
    for i in origin_indices:
        for j in dest_indices:
            d = distance_matrix[i, j]
            if d < np.inf:
                distances.append(d)

    if not distances:
        return np.inf

    if method == 'min':
        return min(distances)
    elif method == 'max':
        return max(distances)
    else:  # mean
        return np.mean(distances)


def get_nearby_schools(school_id, max_distance_m=5000, method='min'):
    """
    Get all schools within X meters of a school.

    For schools with multiple locations, uses the specified method
    to determine the effective distance.
    """
    origin_indices = school_id_to_indices.get(str(school_id), [])

    if not origin_indices:
        return []

    nearby = {}
    for target_id, target_indices in school_id_to_indices.items():
        if target_id == str(school_id):
            continue

        # Get distances from all origin locations to all target locations
        distances = []
        for i in origin_indices:
            for j in target_indices:
                d = distance_matrix[i, j]
                if d < np.inf:
                    distances.append(d)

        if distances:
            if method == 'min':
                effective_dist = min(distances)
            elif method == 'max':
                effective_dist = max(distances)
            else:
                effective_dist = np.mean(distances)

            if 0 < effective_dist <= max_distance_m:
                nearby[target_id] = effective_dist

    # Sort by distance
    return sorted(nearby.items(), key=lambda x: x[1])
```

### Integration with Student Flow Data

```python
# Add distance column to flow DataFrame (multi-location aware)
def add_distance_to_flow(flow_df, distance_matrix, school_id_to_indices, method='min'):
    def lookup_distance(row):
        origin_indices = school_id_to_indices.get(str(row['school_id_origin']), [])
        dest_indices = school_id_to_indices.get(str(row['school_id_destination']), [])

        if not origin_indices or not dest_indices:
            return np.nan

        distances = []
        for i in origin_indices:
            for j in dest_indices:
                d = distance_matrix[i, j]
                if d < np.inf:
                    distances.append(d)

        if not distances:
            return np.nan

        if method == 'min':
            return min(distances)
        elif method == 'max':
            return max(distances)
        else:
            return np.mean(distances)

    flow_df['distance_m'] = flow_df.apply(lookup_distance, axis=1)
    return flow_df
```

---

## Key Statistics

| Metric | Value |
|--------|-------|
| Target regions | NCR, Region IV-A |
| Matrix positions | 8,331 |
| Unique school IDs | 8,307 |
| Schools with multiple locations | 24 |
| Distance pairs computed | ~69.4 million |
| Valid distances (not inf) | ~69.4 million |
| OSRM batch size | 500 coordinates |
| Total batches | 17 |
| Approximate runtime | ~1.5 hours |

---

## Technical Notes

### Why OSRM Instead of NetworkX + scipy

| Aspect | NetworkX + scipy | OSRM |
|--------|-----------------|------|
| Memory usage | ~4-6GB (graph + matrix) | Minimal (HTTP requests) |
| Kernel stability | Frequent crashes | Stable |
| Road network handling | Manual (graph building, snapping) | Automatic |
| One-way street handling | Manual (we had a bug) | Automatic |
| Code complexity | ~200 lines, 7 subsections | ~80 lines, 5 subsections |
| Accuracy | Had issues (see below) | Verified against Google Maps |

### Previous Distance Calculation Bug

The original NetworkX approach had a critical bug:
- `build_road_graph()` only added edges in digitization direction
- Roads were unidirectional even when they should be bidirectional
- Result: distances were 2-7km overestimated vs Google Maps

Example discrepancies found:
- San Nicolas ES → Bacoor NHS: Google 1.7km vs Matrix 9.3km
- St. Anthony Makati → Saint Francis: Google 0.95km vs Matrix 3.9km

OSRM resolves this by using OSM's built-in road directionality data.

**Verification (Notebook 2.3):** OSRM distances were validated against Google Maps and confirmed to be within acceptable tolerance.

### OSRM Table API Details

- **Endpoint:** `/table/v1/driving/{coordinates}`
- **Parameters:**
  - `annotations=distance` - return distances in meters
  - `sources=0;1;2;...` - specify which coordinates are origins
- **Response:** JSON with `distances` matrix (NxN or NxM)
- **Limits:** `max-table-size` configurable in OSRM startup command

### Coordinate Format

- OSRM expects: `longitude,latitude` (note order!)
- Separated by semicolons: `lon1,lat1;lon2,lat2;...`

### Schools with Multiple Locations (Integrated Schools)

Some schools have the same `school_id` but different coordinates. These are typically **integrated schools** where the elementary and high school buildings are at slightly different physical locations.

**Statistics:**
- Total matrix positions: 8,331
- Unique school IDs: 8,307
- Schools with multiple locations: 24 (48 rows total)

**Example:**
| school_id | longitude | latitude | Notes |
|-----------|-----------|----------|-------|
| 401748 | 120.98179 | 14.39703 | Location A (ES building) |
| 401748 | 120.981314 | 14.39699 | Location B (JHS building) |

**Solution:** The index mapping uses `school_id_to_indices` (dict of lists) instead of a simple dict. Each school_id maps to a **list** of matrix indices:
- Single-location school: `{'401234': [0]}`
- Multi-location school: `{'401748': [6760, 6846]}`

**Distance Lookup Strategy:**
When looking up distance between two schools where one or both have multiple locations:
- `method='min'` (default): Use shortest distance between any location pair
- `method='max'`: Use longest distance
- `method='mean'`: Use average of all pairwise distances

For student flow analysis, `method='min'` is recommended as it represents the most accessible route.

---

## Recovery from Kernel Crash

### If crash during Section 2.3 (OSRM query):
- Batches are processed sequentially
- Need to restart from beginning (no mid-batch checkpointing)
- Consider reducing batch size if memory issues

### If crash after Section 2.3b (checkpoint saved):
```python
# Load checkpoint and continue from 2.4
distance_matrix = np.load('output/checkpoints/distance_matrix_osrm.npy')
with open('output/checkpoints/distance_matrix_school_ids.json', 'r') as f:
    school_ids = json.load(f)

# Continue with Section 2.4 (save final output)
```

### Memory Crash During DataFrame Conversion
If kernel crashes when converting to DataFrame (original Section 2.4 approach), use the numpy-only approach documented above. The numpy array + index mapping is sufficient for all downstream use cases.

---

## Related Files

- **docker-compose-arm.yml:** OSRM service configuration
- **Notebook 2.1:** `notebooks/2.1.-build-student-flow-table.ipynb` - Builds student flow data
- **Notebook 2.3b:** `notebooks/2.3b.-verify-distance-matrix.ipynb` - Distance matrix verification
- **Documentation:** `references/documentation/notebook_2.3_visualize_network.md` - Verification workflow docs

---

## Changelog

| Date | Changes |
|------|---------|
| 2026-01-26 | Fixed school_id index mapping to handle integrated schools with multiple locations; Changed `school_id_to_idx` (dict of int) to `school_id_to_indices` (dict of lists); Updated helper functions to support `method` parameter for multi-location distance calculations |
| 2026-01-25 | Switched to numpy-only output (DataFrame conversion caused kernel crashes); Added helper functions for distance lookups |
| 2026-01-25 | Complete rewrite using OSRM approach; Added Surface Pro 11 ARM64 setup guide; Docker named volume solution for mmap issues |
| 2026-01-23 | Initial implementation with NetworkX + scipy (deprecated due to memory issues and directed graph bug) |
