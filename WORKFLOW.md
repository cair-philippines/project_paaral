# PAARAL DepEd Planning View — Build Workflow

## Overview

Scope: NCR + Region IV-A only.

> **Note:** Chunks 1–6 below document the first iteration of this mockup (a simulation sandbox with optimization heuristics). That approach was **redesigned on 2026-05-04** in favor of a simpler descriptive dashboard. The redesigned dashboard is documented in the **Current State** section at the bottom. Chunks 1–6 are kept for historical reference only.

---

## [SUPERSEDED] Chunk 1: Project Scaffolding ✅

Vite + React skeleton; folder structure; package.json; no router; no state library; `public/data/` for static JSON.

---

## [SUPERSEDED] Chunk 2: Synthetic Data Layer ✅

100 synthetic schools (NCR + IVA); 500 student records; 24 origin-destination flows. All data fully synthetic (names, coordinates, enrollment).

---

## [SUPERSEDED] Chunk 3: Constants & Labels ✅

`src/constants/labels.js` (FIELD_LABELS), `src/constants/taxonomy.js` (SCHOOL_TYPES, REGIONS, CITY_TYPES, getSubsidy). These files still exist and are still used.

---

## [SUPERSEDED] Chunk 4: Mock Optimization Engine ✅

`src/engine/optimizer.js` — heuristic simulation of ILP results. **Deleted in redesign.**

---

## [SUPERSEDED] Chunk 5: Control Panel ✅

Left-side policy lever UI: congestion threshold slider, rank tolerance, subsidy inputs, slot budget. `SimulationContext`, `useSimulation`, `ControlPanel`, `SliderInput`, `SubsidyPanel`, `SlotBudgetPanel`. **Deleted in redesign.**

---

## [SUPERSEDED] Chunk 6: Output Visualizations ✅

`SummaryCards`, `FlowMap` (animated SVG arcs), `MapPanel` (Carto Light dot map), `ScenarioTable`, `AppShell`. **Deleted in redesign.**

---

## Current State: Descriptive Dashboard ✅

**Redesigned 2026-05-04.** The view was rebuilt as a flat, full-width descriptive dashboard — no simulation engine, no flow maps, no scenario comparison. The goal is to let DepEd see the current congestion state before layering on prescriptive analytics.

### What it shows

1. **Summary strip** — 4 stat cards: total JHS in scope, schools with congestion, congested public JHS, ESC schools oversubscribed (Rank-1 demand > slots)
2. **Filter bar** — school type tabs (All / Public JHS / Private No-ESC / Private ESC); congestion definition toggle (Enrollees > Seats OR Classroom-to-Learner Ratio with threshold slider); cascading region → province → city dropdowns
3. **Geographic breakdown** — collapsible region → province → municipality rows; congestion badge at each level; barangay placeholder
4. **School table** — sortable, paginated (20/page); congested rows tinted red; inline ESC columns (ESC Slots, Rank-1 Demand, Overflow) on private ESC rows only

### Component map

```
src/
  App.jsx                          ← Fetches schools.geojson + students.json
  components/
    Dashboard.jsx                  ← All filter state (useState); derives annotated + filtered school lists
    SummaryStrip.jsx               ← 4 stat cards
    FilterBar.jsx                  ← Type tabs, congestion toggle + slider, geo dropdowns
    GeographicBreakdown.jsx        ← Collapsible region → province → municipality
    SchoolTable.jsx                ← Sortable/paginated table with inline ESC columns
  constants/
    labels.js                      ← Field display strings
    taxonomy.js                    ← SCHOOL_TYPES, REGIONS, CITY_TYPES, getSubsidy()
public/
  data/
    schools.geojson                ← 5,759 schools (real names/coords from cair-philippines GitHub)
    students.json                  ← 2,000 synthetic students with ranked ESC preferences
    flows.json                     ← 500+ origin-destination pairs
  ecair-logo.png
  deped-logo.png
  reference/
    huc_list.png
scripts/
  generate_data.js                 ← Downloads real CSVs from GitHub; adds synthetic numerics; outputs public/data/
```

### Data

Real school identities from `github.com/cair-philippines/project-school-coordinates`:
- 1,086 public JHS + 561 private (no ESC) + 952 private ESC = **2,599 JHS schools**
- Real: school_id, name, lat/lng, region, province, municipality, barangay
- Synthetic: num_classrooms, seats, enrollment, congestion_ratio, tuition_annual, slots_total, slots_available

### Congestion logic

```javascript
// Two modes, toggled in FilterBar
is_congested = (mode === 'seats')
  ? school.enrollment > school.seats
  : school.congestion_ratio > ratioThreshold
```

### ESC overflow logic

```javascript
// Computed in Dashboard from students.json
rank1_demand  = count of students whose rank1_school_id === school.school_id
overflow      = Math.max(0, rank1_demand - slots_total)
```

---

## Next Steps

The descriptive dashboard gives stakeholders the "current state" view. The next layer — when stakeholder feedback confirms the workflow — is to add:

1. **Rank-1 feasibility simulation** — honor ESC Rank-1 preferences subject to slot caps; show which public JHS schools would be decongested as a result
2. **Scenario toggle** — Baseline vs. "If Rank-1 preferences are honored" side-by-side
3. **Vercel deployment** — `vercel.json` config + production deploy
