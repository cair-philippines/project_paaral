const TYPE_TABS = [
  { value: 'all',             label: 'All JHS' },
  { value: 'public_jhs',      label: 'Public JHS' },
  { value: 'private_jhs',     label: 'Private (No ESC)' },
  { value: 'private_jhs_esc', label: 'Private (ESC)' },
]

const REGION_LABELS = { ncr: 'NCR', iva: 'Region IV-A (CALABARZON)' }

function Select({ label, value, onChange, options, placeholder }) {
  return (
    <div className="flex flex-col gap-1 min-w-[160px]">
      <label className="text-xs text-slate-500 font-medium">{label}</label>
      <select
        value={value}
        onChange={e => onChange(e.target.value)}
        className="text-sm border border-slate-200 rounded-md px-3 py-1.5 bg-white text-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-200"
      >
        <option value="all">{placeholder ?? 'All'}</option>
        {options.map(o => (
          <option key={o.value ?? o} value={o.value ?? o}>{o.label ?? o}</option>
        ))}
      </select>
    </div>
  )
}

export default function FilterBar({
  schoolTypeFilter, onSchoolTypeChange,
  congestionMode,   onCongestionModeChange,
  ratioThreshold,   onRatioThresholdChange,
  regionFilter,     onRegionChange,
  provinceFilter,   onProvinceChange,
  cityFilter,       onCityChange,
  geoOptions,
}) {
  return (
    <div className="bg-white border border-slate-200 rounded-lg px-5 py-4 space-y-4">
      {/* School type tabs */}
      <div className="flex gap-1 flex-wrap">
        {TYPE_TABS.map(tab => (
          <button
            key={tab.value}
            onClick={() => onSchoolTypeChange(tab.value)}
            className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
              schoolTypeFilter === tab.value
                ? 'bg-blue-600 text-white'
                : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Bottom row: congestion definition + geography */}
      <div className="flex flex-wrap gap-4 items-end">
        {/* Congestion definition toggle */}
        <div className="flex flex-col gap-1">
          <span className="text-xs text-slate-500 font-medium">Congestion definition</span>
          <div className="flex rounded-md border border-slate-200 overflow-hidden text-sm">
            <button
              onClick={() => onCongestionModeChange('seats')}
              className={`px-3 py-1.5 transition-colors ${
                congestionMode === 'seats'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white text-slate-600 hover:bg-slate-50'
              }`}
            >
              Enrollees &gt; Seats
            </button>
            <button
              onClick={() => onCongestionModeChange('ratio')}
              className={`px-3 py-1.5 border-l border-slate-200 transition-colors ${
                congestionMode === 'ratio'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white text-slate-600 hover:bg-slate-50'
              }`}
            >
              Classroom-to-Learner Ratio
            </button>
          </div>
        </div>

        {/* Ratio slider — only visible in ratio mode */}
        {congestionMode === 'ratio' && (
          <div className="flex flex-col gap-1 min-w-[200px]">
            <span className="text-xs text-slate-500 font-medium">
              Threshold: <strong className="text-slate-700">{ratioThreshold}:1</strong> students/classroom
            </span>
            <input
              type="range"
              min={25} max={60} step={1}
              value={ratioThreshold}
              onChange={e => onRatioThresholdChange(Number(e.target.value))}
              className="accent-blue-600"
            />
          </div>
        )}

        {/* Geography cascade */}
        <Select
          label="Region"
          value={regionFilter}
          onChange={onRegionChange}
          options={geoOptions.regions.map(r => ({ value: r, label: REGION_LABELS[r] ?? r }))}
          placeholder="All regions"
        />
        <Select
          label="Province"
          value={provinceFilter}
          onChange={onProvinceChange}
          options={geoOptions.provinces}
          placeholder="All provinces"
        />
        <Select
          label="City / Municipality"
          value={cityFilter}
          onChange={onCityChange}
          options={geoOptions.cities}
          placeholder="All cities"
        />
      </div>
    </div>
  )
}
