# PAARAL DepEd Planning View — Build Workflow

## Overview

Seven-chunk build plan for the UI/UX mockup. Each chunk is independently confirmable.
Scope: NCR + Region IV-A only.

---

## Chunk 1: Project Scaffolding ✅

**Goal:** Empty folder → working skeleton that renders in the browser.

**Decisions made:**
- Vite + React, Tailwind CSS, Lucide-React, Recharts (per SKILL.md)
- No router: single-page mockup, no URL routing needed
- No state library: React context is sufficient for mockup scope
- `public/data/` for static JSON/GeoJSON (served directly, not bundled)
- `engine/` isolated so production ILP swap touches one file only

**Folder structure established:**
```
src/
  components/
    layout/         ← Header, Sidebar, main shell
    controls/       ← Sliders, toggles (Control Panel)
    visualizations/ ← Cards, Table, Sankey
    shared/         ← Reusable UI primitives
  constants/        ← labels.js, taxonomy.js
  engine/           ← optimizer.js (mock heuristics)
  context/          ← SimulationContext.jsx
  hooks/            ← useSimulation.js
public/
  data/             ← students.json, schools.json, flows.json (Chunk 2)
  reference/        ← huc_list.png and other reference assets
```

**Artifacts:** package.json, vite.config.js, tailwind.config.js, postcss.config.js,
index.html, src/main.jsx, src/App.jsx, src/index.css, folder placeholders, node_modules.

---

## Chunk 2: Synthetic Data Layer ✅

**Goal:** Populate `public/data/` with realistic-looking static JSON/GeoJSON.

**Files produced:**
- `public/data/schools.geojson` — 100 schools (60 NCR, 40 IVA); types: public_es ×20, public_jhs ×30, private_jhs ×15, private_jhs_esc ×35
- `public/data/students.json` — 500 student records with ranked ESC preferences + road distances
- `public/data/flows.json` — 24 origin-destination pairs (power-law concentration)
- `scripts/generate_data.js` — generation script; re-run to reshuffle synthetic data

**Key decisions:**
- Road distance via circuity factor (NCR ×1.3, near-NCR IVA ×1.4, provincial IVA ×1.6) — not haversine
- Tuition by city_type: NCR ₱50k–₱150k, HUC ₱35k–₱100k, Other ₱20k–₱60k
- `province` field stored on each school to support circuity factor lookup and future routing
- Student preferences ranked by road distance + 5km region-crossing penalty (gravity model)

---

## Chunk 3: Constants & Labels ✅

**Goal:** Centralize all display strings, taxonomy values, and subsidy constants.

**Files produced:**
- `src/constants/labels.js` — FIELD_LABELS map: every snake_case field → human-readable string (covers data fields + simulation output fields)
- `src/constants/taxonomy.js` — SCHOOL_TYPES, REGIONS, CITY_TYPES (with subsidy amounts), getSubsidy(city_type) helper

---

## Chunk 4: Mock Optimization Engine ✅

**Goal:** Heuristic functions that take policy lever inputs and return output metrics.

**Files produced:**
- `src/engine/optimizer.js` — exports: `getBaseline()`, `runSimulation(params)`, `compareScenarios(scenarios)`

**Key heuristics:**
- `demandForReassignment = (43 - threshold) / 43 × 500` — students needing to move
- `capacityForReassignment = 500 × (rankTolerance / 100)` — students allowed to move
- `students_affected = min(demand, capacity, totalSlots)` — actual movers
- Rank split: Rank 2 capped at 70%, Rank 3 capped at 25% of affected students
- Subsidy effect: ₱1k NCR increase → ~3% more enrollment (distance friction dominates)
- Edge cases: infeasible when demand > capacity; unusedSlots when >50% of budget unfilled

---

## Chunk 5: Control Panel ✅

**Goal:** The left-side policy lever UI — sliders and inputs that feed the engine.

**Files produced:**
- `src/context/SimulationContext.jsx` — global state via useReducer; actions: updateParam, updateSubsidy, updateSlotBudget, runSim, saveScenario, resetParams
- `src/hooks/useSimulation.js` — public hook for components; adds `preview()` for live slider estimates
- `src/components/controls/SliderInput.jsx` — reusable slider with live preview line
- `src/components/controls/SubsidyPanel.jsx` — NCR / HUC / Other subsidy inputs (labels are generic tier names, no city references)
- `src/components/controls/SlotBudgetPanel.jsx` — NCR / Region IV-A slot budget inputs with running total
- `src/components/controls/ControlPanel.jsx` — full panel: basic levers, advanced toggle, objective function display, Run + Save + Reset buttons
- `src/App.jsx` *(updated)* — wrapped with SimulationProvider; placeholder for Chunk 6 output area

**Key decisions:**
- `quickPreview()` added to optimizer.js — lightweight, safe to call on every drag
- Advanced options (subsidies + slots) behind toggle — progressive disclosure per SPEC.md
- Save Scenario shows inline name input after first run
- Subsidy tier labels: NCR / HUC / Other — no city names, intentionally generic

---

## Chunk 6: Output Visualizations ✅

**Goal:** The right-side output area — cards, comparison table, Sankey diagram, full layout.

**Files produced:**
- `src/components/visualizations/SummaryCards.jsx` — 4 metric cards with delta vs. baseline; fallback to baseline before first run
- `src/components/visualizations/FlowSankey.jsx` — Recharts Sankey; empty states for pre-run and zero-flow scenarios; custom node renderer with labels
- `src/components/visualizations/ScenarioTable.jsx` — 12-row metric table; columns = Baseline + saved scenarios; empty state prompts user to save
- `src/components/layout/AppShell.jsx` — full layout: header, inline edge case alerts, SummaryCards, tabbed Flow/Scenarios panel, footer transparency note
- `src/App.jsx` *(updated)* — mounts AppShell only; SimulationProvider wraps all

**Key decisions:**
- Summary cards always visible; show "Baseline" label before first run, delta badge after
- Inline alerts (infeasible / unused slots) appear above cards after run — no separate AlertBanner component needed until Chunk 7
- Sankey remounts on data change via `key={JSON.stringify(sankeyData)}` to avoid stale render
- Objective function shown twice: in ControlPanel (interactive) and AppShell footer (always visible reminder)
- Geographic heatmap deferred — marked optional in SPEC.md
- Build verified: 2207 modules, 0 errors

---

## Chunk 7: Edge Cases + Deployment

**Goal:** Infeasible scenario messaging, unused slot warnings, Vercel deploy config.

**Files to produce:**
- `src/components/shared/AlertBanner.jsx`
- `src/engine/edgeCases.js`
- `vercel.json`

**Status:** Pending confirmation.
