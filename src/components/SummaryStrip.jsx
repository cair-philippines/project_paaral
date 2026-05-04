const LABEL = {
  public_jhs:      'Public JHS',
  private_jhs:     'Private (No ESC)',
  private_jhs_esc: 'Private (ESC)',
}

function StatCard({ label, value, sub, accent }) {
  const colors = {
    red:    'bg-red-50 border-red-200 text-red-700',
    amber:  'bg-amber-50 border-amber-200 text-amber-700',
    blue:   'bg-blue-50 border-blue-200 text-blue-700',
    slate:  'bg-white border-slate-200 text-slate-700',
  }
  return (
    <div className={`rounded-lg border px-5 py-4 ${colors[accent] ?? colors.slate}`}>
      <div className="text-2xl font-bold tabular-nums">{value}</div>
      <div className="text-sm font-medium mt-0.5">{label}</div>
      {sub && <div className="text-xs opacity-70 mt-1">{sub}</div>}
    </div>
  )
}

export default function SummaryStrip({ schools, congestionMode, ratioThreshold }) {
  const total      = schools.length
  const congested  = schools.filter(s => s.is_congested).length
  const pct        = total ? Math.round((congested / total) * 100) : 0

  const escSchools    = schools.filter(s => s.school_type === 'private_jhs_esc')
  const oversubscribed = escSchools.filter(s => s.overflow > 0).length

  const byType = Object.entries(LABEL).map(([type, name]) => {
    const subset = schools.filter(s => s.school_type === type)
    const c = subset.filter(s => s.is_congested).length
    return `${name}: ${c.toLocaleString()}/${subset.length.toLocaleString()}`
  }).join('  ·  ')

  const pubJhsCongested = schools.filter(s => s.school_type === 'public_jhs' && s.is_congested).length
  const pubJhsTotal     = schools.filter(s => s.school_type === 'public_jhs').length

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      <StatCard
        label="Total JHS in scope"
        value={total.toLocaleString()}
        sub="Public + Private (NCR + IVA)"
        accent="slate"
      />
      <StatCard
        label="Schools with congestion"
        value={congested.toLocaleString()}
        sub={`${pct}% of total · ${byType}`}
        accent={pct >= 50 ? 'red' : pct >= 25 ? 'amber' : 'slate'}
      />
      <StatCard
        label="Congested public JHS"
        value={pubJhsCongested.toLocaleString()}
        sub={`of ${pubJhsTotal.toLocaleString()} public JHS`}
        accent="red"
      />
      <StatCard
        label="ESC schools oversubscribed"
        value={oversubscribed.toLocaleString()}
        sub={`of ${escSchools.length.toLocaleString()} ESC schools (Rank-1 demand > slots)`}
        accent={oversubscribed > 0 ? 'amber' : 'slate'}
      />
    </div>
  )
}
