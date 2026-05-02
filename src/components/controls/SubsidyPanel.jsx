import { useSimulation } from '../../hooks/useSimulation'

const TIERS = [
  { key: 'ncr',   label: 'NCR'   },
  { key: 'huc',   label: 'HUC'   },
  { key: 'other', label: 'Other' },
]

export function SubsidyPanel() {
  const { params, updateSubsidy } = useSimulation()

  return (
    <div className="space-y-3">
      <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">ESC Subsidy Amount</p>
      {TIERS.map(({ key, label }) => (
        <div key={key} className="flex items-center justify-between gap-3">
          <label className="text-sm text-slate-600 w-10 shrink-0">{label}</label>
          <div className="flex items-center gap-1 ml-auto">
            <span className="text-sm text-slate-400">₱</span>
            <input
              type="number"
              step={1000}
              min={0}
              value={params.subsidies[key]}
              onChange={e => updateSubsidy(key, Number(e.target.value))}
              className="w-28 text-sm border border-slate-200 rounded px-2 py-1 text-right focus:outline-none focus:ring-1 focus:ring-blue-400"
            />
          </div>
        </div>
      ))}
    </div>
  )
}
