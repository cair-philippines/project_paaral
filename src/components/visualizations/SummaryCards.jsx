import { Users, LayoutGrid, Layers, Star, Activity } from 'lucide-react'
import { useSimulation } from '../../hooks/useSimulation'

function congestionColor(ratio) {
  if (ratio <= 35) return { bg: 'bg-green-50',  border: 'border-green-100',  text: 'text-green-700',  icon: 'text-green-400'  }
  if (ratio <= 43) return { bg: 'bg-amber-50',  border: 'border-amber-100',  text: 'text-amber-700',  icon: 'text-amber-400'  }
  return               { bg: 'bg-red-50',    border: 'border-red-100',    text: 'text-red-700',    icon: 'text-red-400'    }
}

const STATIC_CARDS = [
  {
    key:    'students_affected',
    label:  'Students Affected',
    icon:   Users,
    value:  r => r.students_affected.toLocaleString(),
    sub:    r => `${r.pct_students_affected}% of cohort`,
    delta:  (r, b) => r.students_affected - b.students_affected,
    color:  () => ({ bg: 'bg-blue-50',   border: 'border-blue-100',   text: 'text-blue-700',   icon: 'text-blue-400'   }),
  },
  {
    key:    'classrooms_saved',
    label:  'Classrooms Freed',
    icon:   LayoutGrid,
    value:  r => r.classrooms_saved.toLocaleString(),
    sub:    r => `${r.students_affected.toLocaleString()} students redistributed`,
    delta:  (r, b) => r.classrooms_saved - b.classrooms_saved,
    color:  () => ({ bg: 'bg-green-50',  border: 'border-green-100',  text: 'text-green-700',  icon: 'text-green-400'  }),
  },
  {
    key:    'budget_utilization',
    label:  'Slot Utilization',
    icon:   Layers,
    value:  r => `${r.budget_utilization}%`,
    sub:    r => `${r.slots_used.toLocaleString()} of ${(r.slots_used + r.slots_unused).toLocaleString()} slots`,
    delta:  (r, b) => r.budget_utilization - b.budget_utilization,
    color:  () => ({ bg: 'bg-violet-50', border: 'border-violet-100', text: 'text-violet-700', icon: 'text-violet-400' }),
  },
  {
    key:    'pct_at_rank1',
    label:  'At Rank 1 Choice',
    icon:   Star,
    value:  r => `${r.pct_at_rank1}%`,
    sub:    r => `${r.pct_at_rank2}% Rank 2 · ${r.pct_at_rank3}% Rank 3`,
    delta:  (r, b) => r.pct_at_rank1 - b.pct_at_rank1,
    color:  () => ({ bg: 'bg-indigo-50', border: 'border-indigo-100', text: 'text-indigo-700', icon: 'text-indigo-400' }),
  },
  {
    key:    'system_congestion_ratio',
    label:  'System Congestion',
    icon:   Activity,
    value:  r => `${r.system_congestion_ratio}:1`,
    sub:    r => `avg students per classroom`,
    delta:  (r, b) => Math.round((r.system_congestion_ratio - b.system_congestion_ratio) * 10) / 10,
    color:  (r) => congestionColor(r.system_congestion_ratio),
  },
]

function DeltaBadge({ value }) {
  if (value === 0 || value == null) return null
  const sign = value > 0 ? '+' : ''
  return (
    <span className="text-xs text-slate-500 mt-1">
      {sign}{value} vs. baseline
    </span>
  )
}

export function SummaryCards() {
  const { baseline, results } = useSimulation()
  const display = results ?? baseline
  const isBaseline = !results

  if (!display) return null

  return (
    <div className="grid grid-cols-2 xl:grid-cols-5 gap-4">
      {STATIC_CARDS.map(({ key, label, icon: Icon, value, sub, delta, color }) => {
        const c = color(display)
        return (
          <div
            key={key}
            className={`rounded-xl border p-4 flex flex-col gap-1 ${c.bg} ${c.border}`}
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-500 uppercase tracking-wide">{label}</span>
              <Icon size={15} className={c.icon} />
            </div>

            <p className={`text-3xl font-bold mt-1 ${c.text}`}>{value(display)}</p>
            <p className="text-xs text-slate-500">{sub(display)}</p>

            {isBaseline
              ? <span className="text-xs text-slate-400 mt-1">Baseline</span>
              : <DeltaBadge value={delta(display, baseline)} />
            }
          </div>
        )
      })}
    </div>
  )
}
