import { useSimulation } from '../../hooks/useSimulation'

const REGIONS = [
  { key: 'ncr', label: 'NCR' },
  { key: 'iva', label: 'Region IV-A' },
]

export function SlotBudgetPanel() {
  const { params, updateSlotBudget } = useSimulation()
  const total = params.slotBudget.ncr + params.slotBudget.iva

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Slot Budget</p>
        <span className="text-xs text-slate-400">{total.toLocaleString()} total</span>
      </div>
      {REGIONS.map(({ key, label }) => (
        <div key={key} className="flex items-center justify-between gap-3">
          <label className="text-sm text-slate-600 shrink-0">{label}</label>
          <input
            type="number"
            step={100}
            min={0}
            value={params.slotBudget[key]}
            onChange={e => updateSlotBudget(key, Number(e.target.value))}
            className="w-24 text-sm border border-slate-200 rounded px-2 py-1 text-right focus:outline-none focus:ring-1 focus:ring-blue-400"
          />
        </div>
      ))}
    </div>
  )
}
