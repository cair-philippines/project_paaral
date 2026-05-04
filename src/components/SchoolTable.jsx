import { useMemo, useState } from 'react'
import { ChevronUp, ChevronDown, ChevronsUpDown } from 'lucide-react'

const TYPE_LABEL = {
  public_jhs:      'Public JHS',
  private_jhs:     'Private (No ESC)',
  private_jhs_esc: 'Private (ESC)',
}

const REGION_LABEL = { ncr: 'NCR', iva: 'IVA' }

function SortIcon({ col, sortCol, sortDir }) {
  if (sortCol !== col) return <ChevronsUpDown size={13} className="text-slate-300" />
  return sortDir === 'asc'
    ? <ChevronUp size={13} className="text-blue-500" />
    : <ChevronDown size={13} className="text-blue-500" />
}

function Th({ col, label, sortCol, sortDir, onSort, className = '' }) {
  return (
    <th
      onClick={() => onSort(col)}
      className={`text-left py-2.5 px-3 text-xs font-medium text-slate-500 cursor-pointer select-none whitespace-nowrap hover:text-slate-700 ${className}`}
    >
      <span className="inline-flex items-center gap-1">
        {label}
        <SortIcon col={col} sortCol={sortCol} sortDir={sortDir} />
      </span>
    </th>
  )
}

function CongestionBadge({ isCongested }) {
  return isCongested
    ? <span className="inline-block px-2 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-700">Congested</span>
    : <span className="inline-block px-2 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-700">OK</span>
}

function OverflowCell({ school }) {
  if (school.school_type !== 'private_jhs_esc') {
    return <td className="py-2.5 px-3 text-slate-300 text-xs" colSpan={3}>—</td>
  }
  const overflow = school.overflow ?? 0
  return (
    <>
      <td className="py-2.5 px-3 text-sm tabular-nums text-slate-600">{school.slots_total?.toLocaleString() ?? '—'}</td>
      <td className="py-2.5 px-3 text-sm tabular-nums text-slate-600">{school.rank1_demand.toLocaleString()}</td>
      <td className="py-2.5 px-3 text-sm tabular-nums">
        {overflow > 0
          ? <span className="font-semibold text-amber-700">+{overflow.toLocaleString()} unplaced</span>
          : <span className="text-green-700">—</span>
        }
      </td>
    </>
  )
}

export default function SchoolTable({ schools }) {
  const [sortCol, setSortCol] = useState('enrollment')
  const [sortDir, setSortDir] = useState('desc')
  const [page, setPage]       = useState(0)
  const PAGE_SIZE = 20

  const handleSort = col => {
    if (col === sortCol) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortCol(col); setSortDir('desc') }
    setPage(0)
  }

  const sorted = useMemo(() => [...schools].sort((a, b) => {
    let av = a[sortCol] ?? -Infinity
    let bv = b[sortCol] ?? -Infinity
    if (typeof av === 'string') av = av.toLowerCase()
    if (typeof bv === 'string') bv = bv.toLowerCase()
    if (av < bv) return sortDir === 'asc' ? -1 : 1
    if (av > bv) return sortDir === 'asc' ? 1 : -1
    return 0
  }), [schools, sortCol, sortDir])

  const totalPages = Math.ceil(sorted.length / PAGE_SIZE)
  const paged      = sorted.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)

  const showEscColumns = schools.some(s => s.school_type === 'private_jhs_esc')

  const thProps = { sortCol, sortDir, onSort: handleSort }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-600 uppercase tracking-wide">
          School-level detail
        </h2>
        <span className="text-xs text-slate-400">{schools.length.toLocaleString()} schools</span>
      </div>

      <div className="bg-white border border-slate-200 rounded-lg overflow-x-auto">
        <table className="w-full">
          <thead className="bg-slate-50 border-b border-slate-200">
            <tr>
              <Th col="name"             label="School"           {...thProps} className="pl-4 min-w-[220px]" />
              <Th col="school_type"      label="Type"             {...thProps} />
              <Th col="region"           label="Region"           {...thProps} />
              <Th col="province"         label="Province"         {...thProps} />
              <Th col="city"             label="City / Mun."      {...thProps} />
              <Th col="enrollment"       label="Enrollees"        {...thProps} />
              <Th col="seats"            label="Seats"            {...thProps} />
              <Th col="congestion_ratio" label="Ratio"            {...thProps} />
              <Th col="is_congested"     label="Status"           {...thProps} />
              {showEscColumns && <>
                <th className="py-2.5 px-3 text-xs font-medium text-slate-500 whitespace-nowrap bg-amber-50 border-l border-amber-100">ESC Slots</th>
                <th className="py-2.5 px-3 text-xs font-medium text-slate-500 whitespace-nowrap bg-amber-50">Rank-1 Demand</th>
                <th className="py-2.5 px-3 text-xs font-medium text-slate-500 whitespace-nowrap bg-amber-50">Overflow</th>
              </>}
            </tr>
          </thead>
          <tbody>
            {paged.map(school => (
              <tr
                key={school.school_id}
                className={`border-t border-slate-100 hover:bg-slate-50 transition-colors ${
                  school.is_congested ? 'bg-red-50/30' : ''
                }`}
              >
                <td className="py-2.5 px-3 pl-4">
                  <div className="text-sm font-medium text-slate-800">{school.name}</div>
                  <div className="text-xs text-slate-400">{school.school_id}</div>
                </td>
                <td className="py-2.5 px-3 text-xs text-slate-500 whitespace-nowrap">{TYPE_LABEL[school.school_type]}</td>
                <td className="py-2.5 px-3 text-xs text-slate-500">{REGION_LABEL[school.region] ?? school.region}</td>
                <td className="py-2.5 px-3 text-sm text-slate-600">{school.province}</td>
                <td className="py-2.5 px-3 text-sm text-slate-600">{school.city}</td>
                <td className="py-2.5 px-3 text-sm tabular-nums text-slate-700">{school.enrollment?.toLocaleString() ?? '—'}</td>
                <td className="py-2.5 px-3 text-sm tabular-nums text-slate-500">{school.seats?.toLocaleString() ?? '—'}</td>
                <td className="py-2.5 px-3 text-sm tabular-nums text-slate-600">{school.congestion_ratio ?? '—'}</td>
                <td className="py-2.5 px-3"><CongestionBadge isCongested={school.is_congested} /></td>
                {showEscColumns && <OverflowCell school={school} />}
              </tr>
            ))}
            {paged.length === 0 && (
              <tr>
                <td colSpan={showEscColumns ? 12 : 9} className="py-12 text-center text-slate-400 text-sm">
                  No schools match the current filters.
                </td>
              </tr>
            )}
          </tbody>
        </table>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between border-t border-slate-100 px-4 py-3">
            <span className="text-xs text-slate-400">
              {(page * PAGE_SIZE + 1).toLocaleString()}–{Math.min((page + 1) * PAGE_SIZE, sorted.length).toLocaleString()} of {sorted.length.toLocaleString()}
            </span>
            <div className="flex gap-1">
              <button
                disabled={page === 0}
                onClick={() => setPage(p => p - 1)}
                className="px-3 py-1 text-xs rounded border border-slate-200 disabled:opacity-30 hover:bg-slate-50"
              >
                Prev
              </button>
              <button
                disabled={page >= totalPages - 1}
                onClick={() => setPage(p => p + 1)}
                className="px-3 py-1 text-xs rounded border border-slate-200 disabled:opacity-30 hover:bg-slate-50"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
