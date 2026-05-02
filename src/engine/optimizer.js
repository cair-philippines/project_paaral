// Mock optimization engine — heuristics only.
// Production replacement: real ILP solver (PuLP / OR-Tools) via Python API.

const TOTAL_STUDENTS = 500
const BASELINE_CONGESTION = 43   // weighted avg students/classroom across NCR + IVA
const BASELINE_THRESHOLD = 43
const BASELINE_SUBSIDIES = { ncr: 13000, huc: 11000, other: 9000 }
const BASELINE_SLOT_BUDGET = { ncr: 3000, iva: 2000 }
const REGION_SPLIT = { ncr: 0.60, iva: 0.40 }  // 60% NCR, 40% IVA students
const IVA_HUC_SHARE = 0.08                       // ~8% of IVA students near Lucena

// --- Core heuristic ---

function simulate(params) {
  const { threshold, rankTolerance, subsidies, slotBudget } = params
  const totalSlots = slotBudget.ncr + slotBudget.iva

  // Subsidy deltas — computed early so they feed into effective demand
  const subsidyDeltaNCR   = (subsidies.ncr   - BASELINE_SUBSIDIES.ncr)   / 1000
  const subsidyDeltaHUC   = (subsidies.huc   - BASELINE_SUBSIDIES.huc)   / 1000
  const subsidyDeltaOther = (subsidies.other - BASELINE_SUBSIDIES.other) / 1000

  // Higher subsidies reduce the cost barrier, drawing more students into ESC.
  // Distance friction dominates (4×), so the boost is modest (~3% per ₱1k NCR).
  const subsidyBoost = Math.max(0, Math.round(
    subsidyDeltaNCR   * 0.03 * TOTAL_STUDENTS * REGION_SPLIT.ncr +
    subsidyDeltaHUC   * 0.02 * TOTAL_STUDENTS * REGION_SPLIT.iva * IVA_HUC_SHARE +
    subsidyDeltaOther * 0.02 * TOTAL_STUDENTS * REGION_SPLIT.iva * (1 - IVA_HUC_SHARE)
  ))

  // Students who must move to hit the congestion target
  const demandForReassignment = Math.max(
    0,
    Math.round(((BASELINE_CONGESTION - threshold) / BASELINE_CONGESTION) * TOTAL_STUDENTS)
  )

  // Effective demand includes students newly drawn in by higher subsidies
  const effectiveDemand = demandForReassignment + subsidyBoost

  // Students allowed to move given rank tolerance
  const capacityForReassignment = Math.round(TOTAL_STUDENTS * (rankTolerance / 100))

  // Actual movers: bottlenecked by effective demand, tolerance, and slot budget
  const students_affected = Math.min(effectiveDemand, capacityForReassignment, totalSlots)
  const pct_students_affected = Math.round((students_affected / TOTAL_STUDENTS) * 100)

  // Classrooms freed (40 students per classroom standard)
  const classrooms_saved = Math.round(students_affected / 40)

  // Rank distribution among moved students
  const movedToRank2 = Math.round(students_affected * Math.min(rankTolerance / 30, 0.70))
  const movedToRank3 = Math.round(students_affected * Math.min(rankTolerance / 60, 0.25))
  const movedToRank1 = students_affected - movedToRank2 - movedToRank3

  // % across all 500 students
  const pct_at_rank1 = Math.round(((TOTAL_STUDENTS - movedToRank2 - movedToRank3) / TOTAL_STUDENTS) * 100)
  const pct_at_rank2 = Math.round((movedToRank2 / TOTAL_STUDENTS) * 100)
  const pct_at_rank3 = Math.round((movedToRank3 / TOTAL_STUDENTS) * 100)

  // Average road distance increase for moved students
  // Rank 2 choice is ~2.5km further; Rank 3 is ~5km further (from generated data avg)
  const avg_distance_increase =
    students_affected > 0
      ? Math.round(((movedToRank2 * 2.5 + movedToRank3 * 5.0) / students_affected) * 10) / 10
      : 0

  // Slot utilization
  const slots_used = students_affected
  const slots_unused = totalSlots - slots_used
  const budget_utilization = Math.round((slots_used / totalSlots) * 100)

  // Enrollment shift % — summary of subsidy effect for display
  const enrollment_shift_pct = Math.round(
    subsidyDeltaNCR   * 3 * REGION_SPLIT.ncr +
    subsidyDeltaHUC   * 2 * REGION_SPLIT.iva * IVA_HUC_SHARE +
    subsidyDeltaOther * 2 * REGION_SPLIT.iva * (1 - IVA_HUC_SHARE)
  )

  // System-level effective congestion ratio after reassignment
  const system_congestion_ratio = Math.round(
    ((TOTAL_STUDENTS - students_affected) * BASELINE_CONGESTION / TOTAL_STUDENTS) * 10
  ) / 10

  // Total ESC budget (subsidy × students assigned, split by region)
  const ncrStudents   = Math.round(students_affected * REGION_SPLIT.ncr)
  const ivaStudents   = students_affected - ncrStudents
  const hucStudents   = Math.round(ivaStudents * IVA_HUC_SHARE)
  const otherStudents = ivaStudents - hucStudents
  const total_budget  =
    ncrStudents   * subsidies.ncr +
    hucStudents   * subsidies.huc +
    otherStudents * subsidies.other

  // Residual congestion: schools still above threshold after reassignment
  const residual_congestion_pct = Math.max(
    0,
    Math.round(((demandForReassignment - students_affected) / TOTAL_STUDENTS) * 100)
  )

  return {
    students_affected,
    pct_students_affected,
    classrooms_saved,
    pct_at_rank1,
    pct_at_rank2,
    pct_at_rank3,
    movedToRank1,
    movedToRank2,
    movedToRank3,
    avg_distance_increase,
    slots_used,
    slots_unused,
    budget_utilization,
    total_budget,
    enrollment_shift_pct,
    residual_congestion_pct,
    system_congestion_ratio,
    // internals exposed for edge case detection
    _demand: demandForReassignment,
    _capacity: capacityForReassignment,
    _totalSlots: totalSlots,
  }
}

// --- Sankey data builder ---
// Nodes: congested public JHS regions → ESC private schools regions
// Values derived from simulation results

function buildSankeyData(results) {
  const { movedToRank1, movedToRank2, movedToRank3 } = results
  const ncrMoved = Math.round(results.students_affected * REGION_SPLIT.ncr)
  const ivaMoved = results.students_affected - ncrMoved

  return {
    nodes: [
      { name: "Public JHS — NCR" },
      { name: "Public JHS — Region IV-A" },
      { name: "ESC Private — NCR (Rank 1)" },
      { name: "ESC Private — NCR (Rank 2/3)" },
      { name: "ESC Private — Region IV-A (Rank 1)" },
      { name: "ESC Private — Region IV-A (Rank 2/3)" },
    ],
    links: [
      { source: 0, target: 2, value: Math.round(ncrMoved * (results.pct_at_rank1 / 100)) },
      { source: 0, target: 3, value: Math.round(ncrMoved * ((results.pct_at_rank2 + results.pct_at_rank3) / 100)) },
      { source: 1, target: 4, value: Math.round(ivaMoved * (results.pct_at_rank1 / 100)) },
      { source: 1, target: 5, value: Math.round(ivaMoved * ((results.pct_at_rank2 + results.pct_at_rank3) / 100)) },
    ].filter(l => l.value > 0),
  }
}

// --- Heatmap data builder ---
// Per-city congestion relief estimate

const CITY_BASELINE_CONGESTION = {
  // NCR cities (higher congestion)
  "Manila": 52, "Quezon City": 50, "Caloocan": 48, "Valenzuela": 47,
  "Navotas": 46, "Malabon": 45, "Marikina": 44, "Mandaluyong": 44,
  "Pasig": 43, "Pasay": 43, "Makati": 41, "San Juan": 40,
  "Parañaque": 40, "Las Piñas": 39, "Taguig": 39, "Muntinlupa": 38,
  // IVA cities (lower congestion)
  "Bacoor": 42, "Dasmariñas": 41, "Imus": 40, "General Trias": 38,
  "Antipolo": 42, "Cainta": 40,
  "Calamba": 38, "Santa Rosa": 37, "Biñan": 36, "San Pedro": 35,
  "Batangas City": 36, "Lipa City": 35, "Tanauan": 33,
  "Lucena City": 34,
}

function buildHeatmapData(results, threshold) {
  return Object.entries(CITY_BASELINE_CONGESTION).map(([city, baseline]) => {
    const relief = Math.max(0, Math.min(
      baseline - threshold,
      baseline * (results.pct_students_affected / 100) * 1.2
    ))
    return {
      city,
      region: baseline >= 38 && city !== "Lucena City" && !["Calamba","Santa Rosa","Biñan","San Pedro","Batangas City","Lipa City","Tanauan","Bacoor","Dasmariñas","Imus","General Trias","Antipolo","Cainta"].includes(city) ? "ncr" : "iva",
      congestion_before: baseline,
      congestion_after: Math.round((baseline - relief) * 10) / 10,
      relief: Math.round(relief * 10) / 10,
    }
  })
}

// --- Edge case detection ---

function detectEdgeCases(results) {
  const { _demand, _capacity, _totalSlots, slots_unused } = results

  const infeasible = _demand > _capacity && _demand > 0
  const slotShortfall = _demand > _totalSlots && _demand > 0

  const unusedSlotThreshold = _totalSlots * 0.5
  const unusedSlots = slots_unused > unusedSlotThreshold

  return {
    infeasible,
    infeasibleReason: infeasible
      ? `Cannot hit target: ${_demand} students must move but rank tolerance only allows ${_capacity}.`
      : slotShortfall
        ? `Cannot hit target: ${_demand} students must move but slot budget only covers ${_totalSlots}.`
        : null,
    unusedSlots,
    unusedSlotCount: slots_unused,
    unusedSlotReasons: unusedSlots
      ? ["Distance too far for remaining students", "Tuition gap after subsidy", "No ESC school in catchment area"]
      : [],
  }
}

// --- Public API ---

// Lightweight preview — only core metrics, no Sankey/heatmap rebuild.
// Safe to call on every slider drag.
export function quickPreview(params) {
  const r = simulate(params)
  return {
    students_affected: r.students_affected,
    classrooms_saved: r.classrooms_saved,
    pct_at_rank1: r.pct_at_rank1,
    budget_utilization: r.budget_utilization,
  }
}

export function getBaseline() {
  const params = {
    threshold: BASELINE_THRESHOLD,
    rankTolerance: 0,
    subsidies: { ...BASELINE_SUBSIDIES },
    slotBudget: { ...BASELINE_SLOT_BUDGET },
  }
  const results = simulate(params)
  return {
    ...results,
    sankeyData: buildSankeyData(results),
    heatmapData: buildHeatmapData(results, BASELINE_THRESHOLD),
    edgeCases: detectEdgeCases(results),
    params,
  }
}

export function runSimulation(params) {
  const results = simulate(params)
  return {
    ...results,
    sankeyData: buildSankeyData(results),
    heatmapData: buildHeatmapData(results, params.threshold),
    edgeCases: detectEdgeCases(results),
    params,
  }
}

export function compareScenarios(scenarios) {
  // scenarios: [{ label: "Scenario A", params: {...} }, ...]
  return scenarios.map(({ label, params }) => {
    const results = runSimulation(params)
    return {
      label,
      params,
      students_affected: results.students_affected,
      pct_students_affected: results.pct_students_affected,
      classrooms_saved: results.classrooms_saved,
      pct_at_rank1: results.pct_at_rank1,
      pct_at_rank2: results.pct_at_rank2,
      pct_at_rank3: results.pct_at_rank3,
      avg_distance_increase: results.avg_distance_increase,
      budget_utilization: results.budget_utilization,
      total_budget: results.total_budget,
      enrollment_shift_pct: results.enrollment_shift_pct,
      residual_congestion_pct: results.residual_congestion_pct,
      edgeCases: results.edgeCases,
    }
  })
}
