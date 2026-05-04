import { useMemo, useState } from 'react'
import SummaryStrip from './SummaryStrip'
import FilterBar from './FilterBar'
import GeographicBreakdown from './GeographicBreakdown'
import SchoolTable from './SchoolTable'

const JHS_TYPES = ['public_jhs', 'private_jhs', 'private_jhs_esc']

export default function Dashboard({ schools, students }) {
  // Filter state
  const [schoolTypeFilter, setSchoolTypeFilter] = useState('all')
  const [congestionMode, setCongestionMode]     = useState('seats')   // 'seats' | 'ratio'
  const [ratioThreshold, setRatioThreshold]     = useState(40)
  const [regionFilter, setRegionFilter]         = useState('all')
  const [provinceFilter, setProvinceFilter]     = useState('all')
  const [cityFilter, setCityFilter]             = useState('all')

  // Rank-1 demand counts per ESC school (derived once from students data)
  const rank1Demand = useMemo(() => {
    const counts = {}
    for (const s of students) {
      if (s.rank1_school_id) counts[s.rank1_school_id] = (counts[s.rank1_school_id] || 0) + 1
    }
    return counts
  }, [students])

  // Annotate each school with derived fields
  const annotated = useMemo(() => schools
    .filter(s => JHS_TYPES.includes(s.school_type))
    .map(s => {
      const rank1_demand  = rank1Demand[s.school_id] ?? 0
      const overflow      = s.school_type === 'private_jhs_esc'
        ? Math.max(0, rank1_demand - (s.slots_total ?? 0))
        : null
      const is_congested  = congestionMode === 'seats'
        ? s.enrollment > s.seats
        : s.congestion_ratio > ratioThreshold
      return { ...s, rank1_demand, overflow, is_congested }
    }), [schools, rank1Demand, congestionMode, ratioThreshold])

  // Cascade geography options from annotated JHS list
  const geoOptions = useMemo(() => {
    const regions   = [...new Set(annotated.map(s => s.region))].sort()
    const provinces = [...new Set(
      annotated
        .filter(s => regionFilter === 'all' || s.region === regionFilter)
        .map(s => s.province)
    )].sort()
    const cities = [...new Set(
      annotated
        .filter(s => regionFilter === 'all' || s.region === regionFilter)
        .filter(s => provinceFilter === 'all' || s.province === provinceFilter)
        .map(s => s.city)
    )].sort()
    return { regions, provinces, cities }
  }, [annotated, regionFilter, provinceFilter])

  // Reset lower-level filters when upper level changes
  const handleRegionChange = v => { setRegionFilter(v); setProvinceFilter('all'); setCityFilter('all') }
  const handleProvinceChange = v => { setProvinceFilter(v); setCityFilter('all') }

  // Apply all active filters
  const filtered = useMemo(() => annotated.filter(s => {
    if (schoolTypeFilter !== 'all' && s.school_type !== schoolTypeFilter) return false
    if (regionFilter   !== 'all' && s.region   !== regionFilter)   return false
    if (provinceFilter !== 'all' && s.province !== provinceFilter) return false
    if (cityFilter     !== 'all' && s.city     !== cityFilter)     return false
    return true
  }), [annotated, schoolTypeFilter, regionFilter, provinceFilter, cityFilter])

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800">
      {/* Header */}
      <header className="border-b border-[#e2e4e9] bg-white px-4 py-4 sm:px-6 sm:py-5 z-10">
        <div className="max-w-screen-2xl mx-auto">
          <div className="flex items-center gap-4 mb-4">
            <img src="/ecair-logo.png" alt="ECAIR" className="h-5 object-contain" />
            <img src="/deped-logo.png" alt="DepEd" className="h-8 object-contain" />
          </div>
          <div className="flex items-center gap-2">
            <h1 className="font-['SF_Pro_Display',-apple-system,BlinkMacSystemFont,'Segoe_UI',system-ui,sans-serif] text-2xl font-bold tracking-normal text-[#1a1d23] sm:text-3xl">
              PAARAL
            </h1>
            <span className="text-[10px] font-bold uppercase tracking-widest text-[#1a4b8c] bg-[#1a4b8c]/10 px-1.5 py-0.5 rounded">Planning View</span>
          </div>
          <p className="mt-1 text-[13px] leading-5 text-[#6b7280]">
            Platform for Analyzing Access and Resource Allocation in Learning
          </p>
        </div>
      </header>

      <main className="max-w-screen-2xl mx-auto px-6 py-6 space-y-5">
        <SummaryStrip
          schools={annotated}
          congestionMode={congestionMode}
          ratioThreshold={ratioThreshold}
        />

        <FilterBar
          schoolTypeFilter={schoolTypeFilter}  onSchoolTypeChange={setSchoolTypeFilter}
          congestionMode={congestionMode}       onCongestionModeChange={setCongestionMode}
          ratioThreshold={ratioThreshold}       onRatioThresholdChange={setRatioThreshold}
          regionFilter={regionFilter}           onRegionChange={handleRegionChange}
          provinceFilter={provinceFilter}       onProvinceChange={handleProvinceChange}
          cityFilter={cityFilter}               onCityChange={setCityFilter}
          geoOptions={geoOptions}
        />

        <GeographicBreakdown
          schools={annotated}
          congestionMode={congestionMode}
          ratioThreshold={ratioThreshold}
          schoolTypeFilter={schoolTypeFilter}
        />

        <SchoolTable
          schools={filtered}
          congestionMode={congestionMode}
          ratioThreshold={ratioThreshold}
        />
      </main>

      <footer className="max-w-screen-2xl mx-auto px-6 py-4 border-t border-slate-100 text-xs text-slate-400">
        Data is synthetic. Congestion defined as {congestionMode === 'seats'
          ? 'enrollees exceeding physical seats'
          : `classroom-to-learner ratio exceeding ${ratioThreshold}:1`
        }. ESC overflow = Rank-1 applicants minus available slots.
      </footer>
    </div>
  )
}
