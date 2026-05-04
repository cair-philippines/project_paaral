import { useMemo, useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'

const REGION_LABELS = { ncr: 'NCR', iva: 'Region IV-A (CALABARZON)' }
const JHS_TYPES     = ['public_jhs', 'private_jhs', 'private_jhs_esc']

function congestionBadge(count, total) {
  const pct = total ? Math.round((count / total) * 100) : 0
  const color = pct >= 60 ? 'bg-red-100 text-red-700'
              : pct >= 30 ? 'bg-amber-100 text-amber-700'
              : 'bg-green-100 text-green-700'
  return (
    <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${color}`}>
      {count.toLocaleString()}/{total.toLocaleString()} congested ({pct}%)
    </span>
  )
}

function MunicipalityRow({ city, schools }) {
  const total     = schools.length
  const congested = schools.filter(s => s.is_congested).length
  return (
    <tr className="border-t border-slate-100 text-sm">
      <td className="py-2 pl-12 pr-4 text-slate-600">{city}</td>
      <td className="py-2 px-4 text-slate-500 tabular-nums">{total.toLocaleString()}</td>
      <td className="py-2 px-4">{congestionBadge(congested, total)}</td>
      <td className="py-2 px-4 text-slate-400 text-xs">Barangay data not available in mockup</td>
    </tr>
  )
}

function ProvinceSection({ province, schools, isOpen, onToggle }) {
  const total     = schools.length
  const congested = schools.filter(s => s.is_congested).length
  const cities    = [...new Set(schools.map(s => s.city))].sort()

  return (
    <>
      <tr
        className="border-t border-slate-200 bg-slate-50 cursor-pointer hover:bg-slate-100 transition-colors"
        onClick={onToggle}
      >
        <td className="py-2 pl-8 pr-4 font-medium text-slate-700 flex items-center gap-1.5">
          {isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          {province}
        </td>
        <td className="py-2 px-4 text-slate-600 tabular-nums text-sm">{total.toLocaleString()}</td>
        <td className="py-2 px-4">{congestionBadge(congested, total)}</td>
        <td className="py-2 px-4" />
      </tr>
      {isOpen && cities.map(city => (
        <MunicipalityRow
          key={city}
          city={city}
          schools={schools.filter(s => s.city === city)}
        />
      ))}
    </>
  )
}

function RegionSection({ region, schools, schoolTypeFilter }) {
  const [open, setOpen] = useState(true)
  const [openProvinces, setOpenProvinces] = useState({})

  const displayed = schoolTypeFilter === 'all'
    ? schools
    : schools.filter(s => s.school_type === schoolTypeFilter)

  const provinces = [...new Set(displayed.map(s => s.province))].sort()
  const total     = displayed.length
  const congested = displayed.filter(s => s.is_congested).length

  const toggleProvince = p => setOpenProvinces(prev => ({ ...prev, [p]: !prev[p] }))

  return (
    <div className="bg-white border border-slate-200 rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-2 px-5 py-3 text-left hover:bg-slate-50 transition-colors"
      >
        {open ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
        <span className="font-semibold text-slate-800 text-sm">{REGION_LABELS[region] ?? region}</span>
        <span className="ml-3">{congestionBadge(congested, total)}</span>
        <span className="ml-auto text-xs text-slate-400">{total.toLocaleString()} schools</span>
      </button>

      {open && (
        <table className="w-full text-sm">
          <thead>
            <tr className="border-t border-slate-200 bg-slate-50">
              <th className="text-left py-2 pl-8 pr-4 text-xs font-medium text-slate-500 w-1/3">Province / City</th>
              <th className="text-left py-2 px-4 text-xs font-medium text-slate-500 w-20">Schools</th>
              <th className="text-left py-2 px-4 text-xs font-medium text-slate-500">Congestion</th>
              <th className="text-left py-2 px-4 text-xs font-medium text-slate-500">Barangay</th>
            </tr>
          </thead>
          <tbody>
            {provinces.map(prov => (
              <ProvinceSection
                key={prov}
                province={prov}
                schools={displayed.filter(s => s.province === prov)}
                isOpen={!!openProvinces[prov]}
                onToggle={() => toggleProvince(prov)}
              />
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

export default function GeographicBreakdown({ schools, congestionMode, ratioThreshold, schoolTypeFilter }) {
  const jhsOnly = useMemo(() => schools.filter(s => JHS_TYPES.includes(s.school_type)), [schools])
  const regions = useMemo(() => [...new Set(jhsOnly.map(s => s.region))].sort(), [jhsOnly])

  return (
    <div className="space-y-3">
      <h2 className="text-sm font-semibold text-slate-600 uppercase tracking-wide">
        Geographic breakdown
      </h2>
      {regions.map(region => (
        <RegionSection
          key={region}
          region={region}
          schools={jhsOnly.filter(s => s.region === region)}
          schoolTypeFilter={schoolTypeFilter}
        />
      ))}
    </div>
  )
}
