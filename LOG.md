# PAARAL DepEd Planning View — Build Log

> **Purpose:** Chronological record of everything built, moved, or decided.
> Future sessions: read this + CLAUDE.md + SKILL.md + SPEC.md + WORKFLOW.md to get full context.

---

## 2026-05-02

### Scope Refinement (pre-build)

**Changed:** Geographic scope narrowed from all Philippines to **NCR + Region IV-A only.**

**Docs updated:**
- `CLAUDE.md` — added Geographic Scope section with `city_type` subsidy table; updated ESC subsidy structure and synthetic data notes
- `SKILL.md` — updated map note to "NCR + Region IV-A map"; replaced `REGIONS` constant with separate `REGIONS` + `CITY_TYPES` constants; documented `city_type` values (`ncr`, `huc`, `other`) and HUC classification logic
- `SPEC.md` — updated Control Panel region labels and geographic heatmap scope

**Asset moved:** `src/huc_list.png` → `public/reference/huc_list.png`
> Reference: 2025 PSA List of Highly Urbanized Cities. Lucena City is the only HUC in Region IV-A.

**`city_type` values (final):**
| Value | Meaning | ESC Subsidy |
|---|---|---|
| `ncr` | All schools in NCR | ₱13,000 |
| `huc` | Lucena City (sole IVA HUC) | ₱11,000 |
| `other` | All other Region IV-A | ₱9,000 |

---

### Chunk 1: Project Scaffolding ✅

**New files created:**
| File | Purpose |
|---|---|
| `package.json` | React 18, Vite 6, Tailwind 3, Recharts 2, Lucide-React |
| `vite.config.js` | Vite + React plugin |
| `tailwind.config.js` | Tailwind content paths |
| `postcss.config.js` | PostCSS + Autoprefixer |
| `index.html` | App entry point |
| `src/main.jsx` | React root mount |
| `src/App.jsx` | Bare app shell ("PAARAL — DepEd Planning View") |
| `src/index.css` | Tailwind directives |
| `WORKFLOW.md` | 7-chunk build plan with decisions and rationale |

**Folders scaffolded (empty, `.gitkeep`):**
- `src/components/layout/`
- `src/components/controls/`
- `src/components/visualizations/`
- `src/components/shared/`
- `src/constants/`
- `src/engine/`
- `src/context/`
- `src/hooks/`
- `public/data/`

**Dependencies installed:** 170 packages, 0 vulnerabilities.

**Key decisions:**
- No router (single-page mockup)
- No state library (React context sufficient)
- `public/data/` for static JSON/GeoJSON — served directly, not bundled
- `src/engine/` isolated so production ILP swap touches one file

---

### Chunk 2: Synthetic Data Layer ✅

**New files created:**
| File | Purpose |
|---|---|
| `scripts/generate_data.js` | Node.js generation script — re-run to reshuffle data |
| `public/data/schools.geojson` | 100 schools as GeoJSON FeatureCollection |
| `public/data/students.json` | 500 student records with ranked preferences |
| `public/data/flows.json` | 24 origin-destination ES→JHS pairs |

**School breakdown:**
| Type | Count | Region split |
|---|---|---|
| `public_es` | 20 | 12 NCR, 8 IVA |
| `public_jhs` | 30 | 18 NCR, 12 IVA |
| `private_jhs` | 15 | 9 NCR, 6 IVA |
| `private_jhs_esc` | 35 | 21 NCR, 14 IVA |

**School properties by type:**
- All schools: `school_id`, `name`, `school_type`, `region`, `city`, `city_type`, `province`, `num_classrooms`, `coordinates`
- `public_jhs`: + `enrollment`, `congestion_ratio`
- `private_jhs` / `private_jhs_esc`: + `tuition_annual`
- `private_jhs_esc`: + `slots_total`, `slots_available`

**Student record fields:**
`student_id`, `origin_school_id`, `region`, `city_type`,
`rank1_school_id`, `rank2_school_id`, `rank3_school_id`,
`dist_rank1_km`, `dist_rank2_km`, `dist_rank3_km`

**Flow record fields:**
`origin_school_id`, `destination_school_id`, `student_count`, `avg_distance_km`

**Key decisions:**
- Road distance via **circuity factor**, not haversine:
  - NCR ×1.3 (dense grid)
  - Cavite / Laguna / Rizal ×1.4 (partial expressway access)
  - Batangas / Quezon ×1.6 (indirect provincial routes)
- Tuition ranges by `city_type`:
  - `ncr`: ₱50,000 – ₱150,000
  - `huc`: ₱35,000 – ₱100,000
  - `other`: ₱20,000 – ₱60,000
- Congestion ranges: NCR 40–55 students/classroom, IVA 30–50
- Student preference logic: sort by road distance + 5km region-crossing penalty (gravity model)
- `province` field stored on each school to support circuity lookup

---

### Chunk 3: Constants & Labels ✅

**New files created:**
| File | Purpose |
|---|---|
| `src/constants/labels.js` | `FIELD_LABELS` — every snake_case field → human-readable display string |
| `src/constants/taxonomy.js` | `SCHOOL_TYPES`, `REGIONS`, `CITY_TYPES` (with subsidy amounts), `getSubsidy(city_type)` |

**Exports summary:**
- `FIELD_LABELS` — covers all data layer fields + simulation output fields (for Chunks 4–6)
- `SCHOOL_TYPES` — 5 school type keys → display strings
- `REGIONS` — `ncr`, `iva` → display strings
- `CITY_TYPES` — `ncr`, `huc`, `other` → `{ label, subsidy }` objects
- `getSubsidy(city_type)` — returns ESC subsidy amount in PHP; `city_type` alone is sufficient (no region needed)

---

### Chunk 4: Mock Optimization Engine ✅

**New files created:**
| File | Purpose |
|---|---|
| `src/engine/optimizer.js` | Mock heuristics; exports `getBaseline()`, `runSimulation(params)`, `compareScenarios(scenarios)` |

**Params shape:** `{ threshold, rankTolerance, subsidies: { ncr, huc, other }, slotBudget: { ncr, iva } }`

**Output shape:** `{ students_affected, pct_students_affected, classrooms_saved, pct_at_rank1/2/3, avg_distance_increase, slots_used, slots_unused, budget_utilization, total_budget, enrollment_shift_pct, residual_congestion_pct, sankeyData, heatmapData, edgeCases }`

**Key decisions:**
- Baseline congestion = 43 students/classroom (weighted NCR + IVA average)
- `students_affected = min(demand, rankTolerance-capacity, totalSlots)` — three-way bottleneck
- Subsidy effect capped at ~3% per ₱1k increase (distance friction 4× dominates)
- `infeasible` flag: demand > tolerance-capacity
- `unusedSlots` flag: >50% of slot budget unfilled
- Sankey nodes: Public JHS (NCR/IVA) → ESC Private (NCR/IVA) × rank tier
- Heatmap: per-city congestion before/after, indexed to 30 cities in scope

---

### Chunk 5: Control Panel ✅

**New/updated files:**
| File | Purpose |
|---|---|
| `src/context/SimulationContext.jsx` | Global simulation state via useReducer |
| `src/hooks/useSimulation.js` | Public hook; exposes `preview()` for live estimates |
| `src/components/controls/SliderInput.jsx` | Reusable slider with live preview line |
| `src/components/controls/SubsidyPanel.jsx` | NCR / HUC / Other subsidy inputs |
| `src/components/controls/SlotBudgetPanel.jsx` | NCR / Region IV-A slot budget inputs |
| `src/components/controls/ControlPanel.jsx` | Full left-panel: sliders, advanced toggle, objective fn, action buttons |
| `src/App.jsx` | Updated: SimulationProvider wrapper + ControlPanel mount |
| `src/engine/optimizer.js` | Added `quickPreview()` export for drag-safe previews |

**Key decisions:**
- Subsidy panel labels: NCR / HUC / Other — generic, no city names
- Advanced options (subsidies + slots) behind progressive disclosure toggle
- Save Scenario shows inline name input after first successful run
- Build verified: 1584 modules, 0 errors

---

### Chunk 6: Output Visualizations ✅

**New/updated files:**
| File | Purpose |
|---|---|
| `src/components/visualizations/SummaryCards.jsx` | 4 metric cards: Students Affected, Classrooms Freed, Slot Utilization, At Rank 1. Delta vs. baseline shown after run. |
| `src/components/visualizations/FlowSankey.jsx` | Recharts Sankey: Public JHS → ESC Private by region + rank. Custom node labels. Empty states for pre-run and zero-flow. |
| `src/components/visualizations/ScenarioTable.jsx` | 12-row comparison table. Columns: Baseline + saved scenarios. Empty state prompts save. |
| `src/components/layout/AppShell.jsx` | Full page layout: header, inline alerts, cards, tabbed visualization panel, footer. |
| `src/App.jsx` | Updated: mounts AppShell directly. |

**Key decisions:**
- Summary cards fall back to baseline before first run ("Baseline" label, no delta)
- Sankey uses `key={JSON.stringify(sankeyData)}` to force remount on data change
- Inline edge case alerts rendered directly in AppShell (no separate component yet)
- Objective function shown in both ControlPanel and AppShell footer (transparency)
- Geographic heatmap deferred (optional per SPEC.md)
- Dev server verified: http://localhost:5173
- Build verified: 2207 modules, 0 errors

---

### Mid-chunk improvements (post-Chunk 6)

**Changes made:**

| File | Change |
|---|---|
| `src/engine/optimizer.js` | Subsidy deltas now moved before `students_affected` computation — subsidy boost feeds into effective demand so all charts respond to subsidy changes |
| `src/engine/optimizer.js` | Added `system_congestion_ratio` output: `(500 - students_affected) × 43 / 500` — effective avg students/classroom system-wide |
| `src/components/visualizations/SummaryCards.jsx` | Added 5th card: System Congestion — color-coded red >43, amber 35–43, green ≤35 |
| `src/components/visualizations/MapPanel.jsx` | New SVG dot map: loads `/data/schools.geojson`, projects lat/lng to SVG, colors by congestion vs. threshold, hover tooltip with school detail. Region labels for orientation. |
| `src/components/layout/AppShell.jsx` | Added "School Map" tab; renders MapPanel |
| `src/context/SimulationContext.jsx` | Reset now clears scenarios (was only clearing params + results) |
| `src/components/controls/ControlPanel.jsx` | Reset button upgraded from tiny text link to full outlined button labeled "Reset to Default" |

**Key decisions:**
- Subsidy boost formula: ₱1k NCR increase → ~9 more students drawn into ESC (3% × 500 × 60%)
- Map dots sized by school type: public JHS r=6, private ESC r=5, elementary r=3.5
- Map threshold-reactive: dot color updates live as slider moves (reads `params.threshold` from context)
- Map disclaimer note: positions are approximate (jittered from city centroids)

---

### FlowMap v2 — Navigation + Animated Flows ✅

**Inspired by:** flowmap.blue (pan/zoom, animated arcs, SunsetDark color scale, flow-weighted circles)

**Changes to `src/components/visualizations/FlowMap.jsx`:**

| Feature | Detail |
|---|---|
| Pan | Mouse drag; `dragOrig` ref tracks anchor; pan state updated on `mousemove` |
| Zoom | Scroll wheel (cursor-centered), +/− buttons (map-center zoom), ⌖ reset button |
| Zoom indicator | Live `zoom%` badge bottom-left |
| Non-passive wheel | Registered via `useEffect` with `{ passive: false }` — React's synthetic `onWheel` is passive and blocks `preventDefault` |
| Stale closure fix | `panRef`/`zoomRef` updated each render; wheel handler reads refs, not state closures |
| Animated arcs | CSS `@keyframes dash-flow` (dashoffset 13→0) inside SVG `<defs><style>`; `stroke-dasharray: 8 5`; animation speed varies with flow weight (1.1s–1.8s) |
| Color scale | SunsetDark ramp (7 stops, linear interpolation): warm yellow `#f3e79b` → deep purple `#5c53a5`; mapped to normalized flow weight |
| Flow-weighted circles | Schools with ESC flow get scaled radius: `baseR + √(flow/maxTotal) × 3.5` + soft halo ring |
| Hover tooltip | Top-left; shows school detail + up to 3 connected flows with `→`/`←` direction, city name, student count |
| Drag-safe hover | `onMouseEnter` checks `dragOrig.current` before setting hover state — prevents tooltip during drag |

**Arc animation direction:** `dashoffset: 13→0` makes dashes slide forward (origin→destination). Period = 13 (8+5 dasharray), so loop is seamless.

**Legend:** replaced dot swatch for arcs with a 40×10px SunsetDark gradient rect.

---

### FlowMap + MapPanel Overhaul ✅

**Replaced:** `FlowSankey.jsx` (Recharts Sankey) → `FlowMap.jsx` (geographic SVG arc map)

**Changed:** `MapPanel.jsx` upgraded from plain dot grid to Carto Light styled map

**Renamed tab:** "Flow Visualization" → "Flow Map" in AppShell

---

**New file: `src/components/visualizations/FlowMap.jsx`**

| Feature | Detail |
|---|---|
| Canvas | 700 × 530 px SVG |
| Bounds | lng 120.70–122.30, lat 13.60–14.80 (full NCR + Region IV-A) |
| Palette | Carto Light: land `#f0f3f4`, water `#d1dce5` |
| Water bodies | Manila Bay, Laguna de Bay, Taal Lake — closed SVG polygons |
| Highways | NLEX, SLEX, EDSA, C5 — border + white-fill stroke technique |
| Labels | Province labels (NCR, Cavite, Laguna, Rizal, Batangas, Quezon) + italic water labels |
| Flow arcs | Quadratic bezier curves (origin ES → destination ESC); thickness/opacity scale with `budget_utilization` |
| Reactivity | Arcs respond live to subsidy, slot budget, rank tolerance via `preview()` on drag, `results` after run |
| Hover | Tooltip on school dot; connected arcs highlight orange with glow filter |

**Arc weight formula:**
- `flowScale = 0.25 + (budgetUtil / 100) × 0.75` — minimum 25% at default params, full weight at 100% utilization
- Per-arc: `strokeWidth = 0.5 + (count/maxCount × flowScale) × 4.0`

**Updated file: `src/components/visualizations/MapPanel.jsx`**
- Same Carto Light palette, water bodies, highways, province labels as FlowMap
- Bounds expanded from (120.78–122.25, 13.65–14.80) → (120.70–122.30, 13.60–14.80)
- Canvas: 620 × 470 px (slightly smaller; no arc layer)
- School dot + hover tooltip functionality unchanged

**Key decisions:**
- Arc control point lifts upward (−y in SVG = north on map) by 25% of arc distance — clean visual separation, direction-independent
- FlowSankey.jsx retained as dead code (not imported anywhere); safe to delete later
- Both map components are self-contained (no shared geo utility file — duplication is ~30 lines of coordinate constants, acceptable for mockup)

---

---

## 2026-05-04

### Full Dashboard Redesign ✅

**Decision:** Scrapped the optimization simulation sandbox (Chunks 1–6 output) in favor of a simpler, flat descriptive dashboard. The previous mockup was doing too much — flows, predictions, scenario comparison — before stakeholders have even confirmed the basic congestion picture is right. The new view answers a simpler prior question: *what is the current state of congestion, and what does Rank-1 ESC demand look like against slot availability?*

**Deleted:**
- `src/context/` — SimulationContext, useReducer state
- `src/hooks/` — useSimulation
- `src/engine/` — optimizer.js, heuristics
- `src/components/layout/` — AppShell
- `src/components/controls/` — all sliders, panels
- `src/components/visualizations/` — FlowMap, MapPanel, SummaryCards, ScenarioTable, FlowSankey
- `src/components/shared/` — empty

**New files:**
| File | Purpose |
|---|---|
| `src/App.jsx` | Loads `schools.geojson` + `students.json`; passes to Dashboard |
| `src/components/Dashboard.jsx` | Full-width layout; holds all filter/toggle state via useState |
| `src/components/SummaryStrip.jsx` | 4 stat cards: total JHS, congested, public JHS congested, ESC oversubscribed |
| `src/components/FilterBar.jsx` | Type tabs (All / Public JHS / Private No-ESC / Private ESC); congestion definition toggle + ratio slider; cascading region → province → city dropdowns |
| `src/components/GeographicBreakdown.jsx` | Collapsible region → province → municipality rows with congestion badges |
| `src/components/SchoolTable.jsx` | Sortable, paginated table; inline ESC columns (Slots, Rank-1 Demand, Overflow) for private ESC rows |

**Key design decisions:**
- No React context or state library — plain `useState` in Dashboard is sufficient
- Congestion has two definitions: (a) enrollees > seats, (b) ratio > user-set threshold; toggle in FilterBar
- ESC overflow = `max(0, rank1_demand - slots_total)` — computed from `students.json` rank-1 counts
- Private schools can be congested too — enrollment ranges allow exceeding seats (~23% private non-ESC, ~13% private ESC)
- Geographic breakdown filter is independent of the school table filter (breakdown always shows all JHS; table responds to all active filters)

---

### Real School Data Integration ✅

**Source:** `https://github.com/cair-philippines/project-school-coordinates`

**Files used:**
- `data/gold/public_school_coordinates.csv` — 48,254 schools; filtered to NCR + Region IV-A, active, with coordinates
- `data/gold/private_school_coordinates.csv` — 12,167 schools; same filter

**CSVs are cached at `/tmp/` on first run; re-downloaded if missing.**

**School counts (real):**
| Type | Count | Source |
|---|---|---|
| `public_es` | 3,160 | public CSV, offers_es=True, offers_jhs=False |
| `public_jhs` | 1,086 | public CSV, offers_jhs=True |
| `private_jhs` | 561 | private CSV, offers_jhs=True, esc_participating=0 |
| `private_jhs_esc` | 952 | private CSV, offers_jhs=True, esc_participating=1 |

**Real fields used:** `school_id`, `school_name`, `latitude`, `longitude`, `region`, `province`, `municipality`, `barangay`

**Still synthetic:** `num_classrooms`, `seats`, `enrollment`, `congestion_ratio`, `tuition_annual`, `slots_total`, `slots_available`

**Normalization applied:**
- Province for NCR: "NCR   SECOND DISTRICT" → "NCR Second District"
- Province for Region IV-A: "CAVITE" → "Cavite" (title case)
- City: "CITY OF MANDALUYONG" → "Mandaluyong" (stripped prefix, title case)
- `city_type`: NCR → `ncr`; Lucena City → `huc`; all other IVA → `other`

**Students:** 2,000 synthetic students (up from 500) across 300 sampled ES origin schools → distributed to 952 real ESC destinations via gravity model (road distance + 5km cross-region penalty). Scaled up to produce meaningful Rank-1 demand clusters.

---

### Header Update ✅

Replicated header style from `paaral-mockup/src/App.jsx`:
- ECAIR logo + DepEd logo row (assets copied from `paaral-mockup/public/assets/`)
- "PAARAL" in SF Pro Display / system-ui, bold, 2xl/3xl
- "Planning View" badge (blue, uppercase, tracking-widest) in place of "BETA"
- Subtitle: "Platform for Analyzing Access and Resource Allocation in Learning"

---

### Number Formatting ✅

Applied `.toLocaleString()` to all numeric outputs across:
- `SummaryStrip.jsx` — all card values and sub-text counts
- `GeographicBreakdown.jsx` — badge counts and school totals
- `SchoolTable.jsx` — slots_total, rank1_demand, overflow, school count label, pagination range
