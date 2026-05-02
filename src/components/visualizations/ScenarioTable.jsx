import { useSimulation } from '../../hooks/useSimulation'

const fmt = {
  pct:    v => `${v}%`,
  count:  v => v.toLocaleString(),
  php:    v => `₱${v.toLocaleString()}`,
  km:     v => `${v} km`,
}

const ROWS = [
  { label: 'Students affected',       key: 'students_affected',       format: fmt.count },
  { label: '% of cohort',             key: 'pct_students_affected',   format: fmt.pct   },
  { label: 'Classrooms freed',        key: 'classrooms_saved',        format: fmt.count },
  { label: 'At Rank 1',               key: 'pct_at_rank1',            format: fmt.pct   },
  { label: 'At Rank 2',               key: 'pct_at_rank2',            format: fmt.pct   },
  { label: 'At Rank 3',               key: 'pct_at_rank3',            format: fmt.pct   },
  { label: 'Avg. distance increase',  key: 'avg_distance_increase',   format: fmt.km    },
  { label: 'Slots used',              key: 'slots_used',              format: fmt.count },
  { label: 'Slots unused',            key: 'slots_unused',            format: fmt.count },
  { label: 'Slot utilization',        key: 'budget_utilization',      format: fmt.pct   },
  { label: 'Total ESC budget',        key: 'total_budget',            format: fmt.php   },
  { label: 'Residual congestion',     key: 'residual_congestion_pct', format: fmt.pct   },
]

function Cell({ value }) {
  return (
    <td className="px-4 py-2.5 text-sm text-slate-700 text-right tabular-nums">
      {value}
    </td>
  )
}

export function ScenarioTable() {
  const { baseline, scenarios } = useSimulation()

  if (!baseline) return null

  if (!scenarios.length) {
    return (
      <div className="flex items-center justify-center h-48 text-slate-400 text-sm">
        No scenarios saved yet — run a simulation and click "Save as Scenario"
      </div>
    )
  }

  const columns = [
    { label: 'Baseline', results: baseline },
    ...scenarios.map(s => ({ label: s.label, results: s.results })),
  ]

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left border-collapse">
        <thead>
          <tr className="border-b border-slate-200">
            <th className="px-4 py-2.5 text-xs font-semibold text-slate-500 uppercase tracking-wide w-48">
              Metric
            </th>
            {columns.map(({ label }) => (
              <th
                key={label}
                className="px-4 py-2.5 text-xs font-semibold text-slate-700 uppercase tracking-wide text-right"
              >
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {ROWS.map(({ label, key, format }, i) => (
            <tr
              key={key}
              className={i % 2 === 0 ? 'bg-white' : 'bg-slate-50'}
            >
              <td className="px-4 py-2.5 text-sm text-slate-500">{label}</td>
              {columns.map(({ label: colLabel, results }) => (
                <Cell key={colLabel} value={format(results[key])} />
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
