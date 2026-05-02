import { useState, useEffect } from 'react'
import { useSimulation } from '../../hooks/useSimulation'

// Geographic bounds for NCR + Region IV-A
const LNG_MIN = 120.78, LNG_MAX = 122.25
const LAT_MIN = 13.65,  LAT_MAX = 14.80

const SVG_W   = 620
const SVG_H   = 460
const PAD     = 36

function project(lng, lat) {
  const x = PAD + ((lng - LNG_MIN) / (LNG_MAX - LNG_MIN)) * (SVG_W - PAD * 2)
  const y = PAD + ((LAT_MAX - lat) / (LAT_MAX - LAT_MIN)) * (SVG_H - PAD * 2)
  return [Math.round(x * 10) / 10, Math.round(y * 10) / 10]
}

function dotColor(school, threshold) {
  switch (school.school_type) {
    case 'public_jhs':
      return school.congestion_ratio > threshold ? '#ef4444' : '#22c55e'
    case 'private_jhs_esc':
      return '#3b82f6'
    case 'public_es':
      return '#94a3b8'
    default:
      return '#cbd5e1'
  }
}

function dotRadius(school) {
  if (school.school_type === 'public_jhs')      return 6
  if (school.school_type === 'private_jhs_esc') return 5
  return 3.5
}

const LEGEND = [
  { color: '#ef4444', label: 'Public JHS — Congested' },
  { color: '#22c55e', label: 'Public JHS — Within threshold' },
  { color: '#3b82f6', label: 'Private JHS (ESC)' },
  { color: '#94a3b8', label: 'Public Elementary' },
]

// Approximate region label positions (for geographic orientation)
const REGION_LABELS = [
  { label: 'NCR',          lng: 121.02, lat: 14.62 },
  { label: 'Cavite',       lng: 120.90, lat: 14.32 },
  { label: 'Laguna',       lng: 121.20, lat: 14.27 },
  { label: 'Rizal',        lng: 121.18, lat: 14.59 },
  { label: 'Batangas',     lng: 121.05, lat: 13.88 },
  { label: 'Quezon',       lng: 121.80, lat: 14.00 },
]

export function MapPanel() {
  const [schools, setSchools] = useState([])
  const [loading, setLoading] = useState(true)
  const [hovered, setHovered]  = useState(null)
  const { params } = useSimulation()

  useEffect(() => {
    fetch('/data/schools.geojson')
      .then(r => r.json())
      .then(data => {
        setSchools(
          data.features.map(f => ({
            ...f.properties,
            lng: f.geometry.coordinates[0],
            lat: f.geometry.coordinates[1],
          }))
        )
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64 text-slate-400 text-sm">
        Loading map…
      </div>
    )
  }

  // Sort so JHS dots render on top of ES dots
  const sorted = [...schools].sort((a, b) => {
    const order = { public_es: 0, private_jhs: 1, private_jhs_esc: 2, public_jhs: 3 }
    return (order[a.school_type] ?? 0) - (order[b.school_type] ?? 0)
  })

  return (
    <div className="space-y-3">
      {/* Legend */}
      <div className="flex flex-wrap gap-x-5 gap-y-1.5">
        {LEGEND.map(({ color, label }) => (
          <span key={label} className="flex items-center gap-1.5 text-xs text-slate-500">
            <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: color }} />
            {label}
          </span>
        ))}
      </div>

      {/* Map */}
      <div className="relative overflow-x-auto rounded-lg border border-slate-200 bg-slate-50">
        <svg
          width={SVG_W}
          height={SVG_H}
          className="block"
          style={{ minWidth: SVG_W }}
        >
          {/* Region label overlays for orientation */}
          {REGION_LABELS.map(({ label, lng, lat }) => {
            const [x, y] = project(lng, lat)
            return (
              <text
                key={label}
                x={x} y={y}
                fontSize={10}
                fill="#cbd5e1"
                textAnchor="middle"
                fontWeight="600"
                letterSpacing="0.05em"
              >
                {label}
              </text>
            )
          })}

          {/* School dots */}
          {sorted.map(school => {
            const [x, y] = project(school.lng, school.lat)
            const fill   = dotColor(school, params.threshold)
            const r      = dotRadius(school)
            return (
              <circle
                key={school.school_id}
                cx={x}
                cy={y}
                r={r}
                fill={fill}
                fillOpacity={0.85}
                stroke="white"
                strokeWidth={1.2}
                className="cursor-pointer transition-opacity hover:opacity-100"
                style={{ opacity: hovered?.school_id === school.school_id ? 1 : 0.85 }}
                onMouseEnter={() => setHovered(school)}
                onMouseLeave={() => setHovered(null)}
              />
            )
          })}
        </svg>

        {/* Hover tooltip */}
        {hovered && (
          <div className="absolute top-3 right-3 bg-white border border-slate-200 rounded-lg p-3 shadow text-xs space-y-0.5 max-w-52 pointer-events-none">
            <p className="font-semibold text-slate-800 leading-snug">{hovered.name}</p>
            <p className="text-slate-500">{hovered.city}{hovered.province !== hovered.city ? `, ${hovered.province}` : ''}</p>
            <p className="text-slate-400 capitalize">{hovered.school_type.replace(/_/g, ' ')}</p>
            {hovered.congestion_ratio != null && (
              <p className={`font-semibold mt-1 ${hovered.congestion_ratio > params.threshold ? 'text-red-600' : 'text-green-600'}`}>
                {hovered.congestion_ratio}:1 congestion ratio
              </p>
            )}
            {hovered.slots_total != null && (
              <p className="text-slate-500">
                {hovered.slots_available} / {hovered.slots_total} ESC slots available
              </p>
            )}
            {hovered.tuition_annual != null && (
              <p className="text-slate-500">
                ₱{hovered.tuition_annual.toLocaleString()} / year
              </p>
            )}
          </div>
        )}
      </div>

      <p className="text-xs text-slate-400">
        Dot positions are approximate — jittered from city centroids for synthetic data.
        Red/green status updates with the congestion threshold slider.
      </p>
    </div>
  )
}
