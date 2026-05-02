
## Technical Stack

### Current Phase: Mockup

**Framework:** React (Vite)
**Styling:** Tailwind CSS
**Icons:** Lucide-React
**Maps:** Carto-style SVG (simplified NCR + Region IV-A map)
**Charts:** Recharts (bar, line, Sankey)
**Deployment:** Vercel

**Critical Constraint:** No database queries, no BigQuery access. All data is synthetic JSON/GeoJSON.

### Future Phase: Development

**Maps:** Mapbox GL JS + deck.gl
**Backend:** Python API with OR solvers (PuLP, OR-Tools, Gurobi)
**Data Infrastructure:** GCP BigQuery (`ecair-data-repository.lis_dev`)
**Geospatial:** PostGIS for spatial queries
**Network Analysis:** NetworkX (when specific questions justify it)

## Coding Standards

### Field Naming Convention

**Database/Data Layer:** snake_case
- `school_id`, `slots_available`, `tuition_annual`, `distance_km`
- Mirrors BigQuery schema conventions

**Display Layer:** Human-readable via mapping
- Map via `constants/labels.js`: `{ tuition_annual: "Annual Tuition" }`
- Never mix naming conventions in the same file

**File exports from data pipeline:**
- `es_schools.geojson` (elementary schools)
- `jhs_schools.geojson` (junior high schools)
- `flows.json` (origin-destination pairs)

### School Type Taxonomy

```javascript
const SCHOOL_TYPES = {
  public_es: "Public Elementary",
  private_es: "Private Elementary", 
  public_jhs: "Public JHS",
  private_jhs: "Private JHS (No ESC)",
  private_jhs_esc: "Private JHS (With ESC)"
};
```

**Congestion metrics apply to:** All JHS types
**ESC slot metrics apply to:** `private_jhs` and `private_jhs_esc` only

### Region & Subsidy Taxonomy

Mockup scope is **NCR + Region IV-A only.**

```javascript
// Geographic region (two values only)
const REGIONS = {
  ncr: "National Capital Region (NCR)",
  iva: "Region IV-A (CALABARZON)",
};

// City type drives the ESC subsidy tier (three values only)
const CITY_TYPES = {
  ncr:   { label: "NCR",                   subsidy: 13000 },
  huc:   { label: "Highly Urbanized City", subsidy: 11000 },
  other: { label: "Other",                 subsidy: 9000  },
};
```

**How `city_type` is assigned:**
- `ncr` — all schools located in NCR (regardless of PSA HUC status)
- `huc` — schools in Lucena City only (sole Region IV-A HUC per 2025 PSA list)
- `other` — all remaining Region IV-A schools (Cavite, Laguna, Batangas, Rizal, Quezon municipalities)

**HUC classification source:** `public/reference/huc_list.png` (2025 PSA List of HUCs)

### Data Processing Rules

1. **No direct database access in mockup**
   - Load from static files in `public/`
   - Synthetic data generation scripts stay in `scripts/`

2. **GeoJSON structure**
   ```json
   {
     "type": "FeatureCollection",
     "features": [
       {
         "type": "Feature",
         "geometry": { "type": "Point", "coordinates": [lng, lat] },
         "properties": { "school_id": "JHS_001", "name": "...", ... }
       }
     ]
   }
   ```

3. **Flow data structure**
   ```json
   {
     "origin_school_id": "ES_001",
     "destination_school_id": "JHS_050", 
     "student_count": 47,
     "avg_distance_km": 3.2
   }
   ```
4. Never do a deep search of the files. Load the schema of the directory then infer from that.

5. Before running code implementations, **always** confirm with me first the objective of the code.