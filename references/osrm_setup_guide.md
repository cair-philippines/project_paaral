# OSRM Setup Guide

This guide explains how to set up the Open Source Routing Machine (OSRM) backend server for computing road-network distances between schools. Notebook **2.2** uses OSRM to build the distance matrix required for congestion analysis.

---

## What is OSRM?

OSRM is a high-performance routing engine that computes shortest paths on road networks derived from OpenStreetMap (OSM) data. Unlike straight-line (haversine) distances, OSRM provides realistic travel distances that account for actual road layouts — critical for modeling how families choose schools based on commute distance.

---

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Docker** | Docker Engine 20.x+ | Docker Desktop (latest) |
| **Disk space** | 10 GB | 20 GB (for intermediate files) |
| **RAM** | 4 GB | 8 GB+ (extraction is memory-intensive) |
| **OS** | Linux, macOS, Windows (WSL2) | Linux or macOS |

Verify Docker is installed:

```bash
docker --version
```

---

## Step 1: Download OpenStreetMap Data

Download the Philippines OSM extract from [Geofabrik](https://download.geofabrik.de/asia/philippines.html):

```bash
# Create a directory for OSRM data
mkdir -p ~/osrm-data && cd ~/osrm-data

# Download the Philippines extract (~400 MB)
wget https://download.geofabrik.de/asia/philippines-latest.osm.pbf
```

Alternatively, for faster processing during development, you can download a regional extract (if available) or use [BBBike](https://extract.bbbike.org/) to create a custom bounding box for NCR and Region IV-A only.

---

## Step 2: Extract the Road Network

Run the OSRM extraction process using the official Docker image:

```bash
cd ~/osrm-data

# Extract road network for driving profile (~5-10 minutes)
docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-extract \
    -p /opt/car.lua \
    /data/philippines-latest.osm.pbf
```

This creates several intermediate files (`.osrm`, `.osrm.nodes`, etc.) in the same directory.

---

## Step 3: Partition and Customize

OSRM uses a multi-level Dijkstra (MLD) algorithm that requires partitioning:

```bash
# Partition the network (~2-5 minutes)
docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-partition \
    /data/philippines-latest.osrm

# Customize for the driving profile (~1-2 minutes)
docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-customize \
    /data/philippines-latest.osrm
```

---

## Step 4: Start the OSRM Server

Launch the routing server on port 5000:

```bash
docker run -t -p 5000:5000 -v "${PWD}:/data" osrm/osrm-backend osrm-routed \
    --algorithm mld \
    /data/philippines-latest.osrm
```

The server will start and display:
```
[info] running and waiting for requests
```

Keep this terminal open while running notebook 2.2.

---

## Step 5: Verify the Server

Test that OSRM is responding correctly:

**Using curl:**
```bash
# Request a route between two points in Metro Manila
curl "http://localhost:5000/route/v1/driving/121.0,14.6;121.05,14.65?overview=false"
```

**Using Python:**
```python
import requests

response = requests.get(
    "http://localhost:5000/route/v1/driving/121.0,14.6;121.05,14.65",
    params={"overview": "false"}
)
print(response.json())
```

A successful response will include `"code": "Ok"` and distance/duration values.

---

## How Notebook 2.2 Uses OSRM

Notebook 2.2 calls the OSRM **Table Service** to compute a distance matrix between all school coordinates:

```
GET http://localhost:5000/table/v1/driving/{coordinates}?annotations=distance
```

The notebook expects OSRM to be running at `http://localhost:5000`. If you need to use a different host or port, update the `OSRM_URL` variable in notebook 2.2.

**Output files:**
- `school_distance_matrix_osrm.npy` — NumPy array of pairwise distances (meters)
- `school_distance_matrix_index.json` — Mapping of school IDs to matrix indices

---

## Running OSRM in the Background

To run OSRM as a background service:

```bash
# Start in detached mode
docker run -d --name osrm-ph -p 5000:5000 -v "${PWD}:/data" osrm/osrm-backend osrm-routed \
    --algorithm mld \
    /data/philippines-latest.osrm

# Check status
docker ps

# View logs
docker logs osrm-ph

# Stop the server
docker stop osrm-ph

# Restart later
docker start osrm-ph
```

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `port 5000 already in use` | Another service on port 5000 | Use `-p 5001:5000` and update notebook URL |
| `Killed` during extraction | Out of memory | Increase Docker memory limit or use a smaller regional extract |
| `connection refused` on localhost | Docker networking issue (Windows/WSL2) | Try `host.docker.internal:5000` or check Docker Desktop settings |
| `NoSegment` errors in responses | Coordinates outside road network | Verify coordinates are within the Philippines and near roads |
| Slow table queries | Large coordinate sets | Batch requests (notebook 2.2 handles this automatically) |

---

## Quick Reference

```bash
# Full setup sequence (run from ~/osrm-data)
wget https://download.geofabrik.de/asia/philippines-latest.osm.pbf

docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-extract -p /opt/car.lua /data/philippines-latest.osm.pbf
docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-partition /data/philippines-latest.osrm
docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-customize /data/philippines-latest.osrm
docker run -t -p 5000:5000 -v "${PWD}:/data" osrm/osrm-backend osrm-routed --algorithm mld /data/philippines-latest.osrm
```

---

## Additional Resources

- [OSRM Project](http://project-osrm.org/)
- [OSRM API Documentation](http://project-osrm.org/docs/v5.24.0/api/)
- [Geofabrik Download Server](https://download.geofabrik.de/)
- [OSRM Docker Hub](https://hub.docker.com/r/osrm/osrm-backend)
