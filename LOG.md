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

## Pending

| Chunk | Status |
|---|---|
| 7 — Edge Cases + Deployment | Not started |
