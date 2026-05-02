import { useSimulationContext } from '../context/SimulationContext'
import { quickPreview } from '../engine/optimizer'

export function useSimulation() {
  const ctx = useSimulationContext()

  // Returns lightweight preview metrics for a given param set without
  // triggering a full Sankey/heatmap rebuild. Safe on every slider drag.
  const preview = (overrides = {}) =>
    quickPreview({ ...ctx.params, ...overrides })

  return { ...ctx, preview }
}
