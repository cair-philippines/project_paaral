/**
 * Generates public/data/ files using real school coordinates from:
 * https://github.com/cair-philippines/project-school-coordinates
 *
 * Real fields:  school_id, name, lat/lng, region, province, municipality, barangay
 * Synthetic:    num_classrooms, seats, enrollment, congestion_ratio, slots, tuition
 *
 * Run: node scripts/generate_data.js
 * Requires: /tmp/public_schools.csv and /tmp/private_schools.csv
 *   → curl -s https://raw.githubusercontent.com/cair-philippines/project-school-coordinates/main/data/gold/public_school_coordinates.csv -o /tmp/public_schools.csv
 *   → curl -s https://raw.githubusercontent.com/cair-philippines/project-school-coordinates/main/data/gold/private_school_coordinates.csv -o /tmp/private_schools.csv
 */

import { writeFileSync, existsSync } from 'fs'
import { readFileSync }              from 'fs'
import { join, dirname }            from 'path'
import { fileURLToPath }            from 'url'
import { execSync }                 from 'child_process'

const __dirname = dirname(fileURLToPath(import.meta.url))
const OUT       = join(__dirname, '../public/data')

// --- Fetch CSVs if not cached ---

const PUB_CSV = '/tmp/public_schools.csv'
const PRI_CSV = '/tmp/private_schools.csv'
const BASE    = 'https://raw.githubusercontent.com/cair-philippines/project-school-coordinates/main/data/gold'

if (!existsSync(PUB_CSV)) {
  console.log('Downloading public_school_coordinates.csv …')
  execSync(`curl -s "${BASE}/public_school_coordinates.csv" -o ${PUB_CSV}`)
}
if (!existsSync(PRI_CSV)) {
  console.log('Downloading private_school_coordinates.csv …')
  execSync(`curl -s "${BASE}/private_school_coordinates.csv" -o ${PRI_CSV}`)
}

// --- CSV parser (handles quoted fields with commas) ---

function parseCSV(path) {
  const lines   = readFileSync(path, 'utf8').split('\n')
  const headers = lines[0].split(',').map(h => h.trim())
  const rows    = []
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim()
    if (!line) continue
    const cols = []
    let cur = '', inQ = false
    for (const ch of line) {
      if (ch === '"') { inQ = !inQ }
      else if (ch === ',' && !inQ) { cols.push(cur); cur = '' }
      else cur += ch
    }
    cols.push(cur)
    const row = {}
    headers.forEach((h, idx) => { row[h] = (cols[idx] ?? '').trim() })
    rows.push(row)
  }
  return rows
}

// --- Utilities ---

const rand    = (min, max) => Math.random() * (max - min) + min
const randInt = (min, max) => Math.floor(rand(min, max + 1))
const round1  = v          => Math.round(v * 10) / 10
const pick    = arr        => arr[Math.floor(Math.random() * arr.length)]
const jitter  = (v, d)     => v + rand(-d, d)

function haversineKm([lng1, lat1], [lng2, lat2]) {
  const R    = 6371
  const dLat = (lat2 - lat1) * Math.PI / 180
  const dLng = (lng2 - lng1) * Math.PI / 180
  const a    =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
    Math.sin(dLng / 2) ** 2
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
}

const CIRCUITY = { ncr: 1.3, Cavite: 1.4, Laguna: 1.4, Rizal: 1.4, Batangas: 1.6, Quezon: 1.6 }
const roadDistKm = (a, b, pA, pB) =>
  haversineKm(a, b) * Math.max(CIRCUITY[pA] ?? 1.4, CIRCUITY[pB] ?? 1.4)

// --- Province normalization ---
// NCR uses "NCR   FIRST DISTRICT" etc. Collapse to just the district label.

function normalizeProvince(province, region) {
  if (region === 'NCR') {
    const m = province.match(/(FIRST|SECOND|THIRD|FOURTH)/i)
    return m ? `NCR ${m[1].charAt(0) + m[1].slice(1).toLowerCase()} District` : 'NCR'
  }
  return toTitleCase(province)
}

function toTitleCase(s) {
  return s.toLowerCase().replace(/\b\w/g, c => c.toUpperCase())
}

// Normalize city name: "CITY OF MANDALUYONG" → "Mandaluyong"
function normalizeCity(city) {
  return toTitleCase(
    city
      .replace(/^CITY OF /i, '')
      .replace(/ CITY$/i, '')       // keep suffix only for disambiguation if needed
      .trim()
  )
}

// city_type: NCR → 'ncr', Lucena City → 'huc', all other IVA → 'other'
function cityType(region, rawCity) {
  if (region === 'NCR') return 'ncr'
  const city = rawCity.toUpperCase()
  if (city.includes('LUCENA')) return 'huc'
  return 'other'
}

const TUITION = { ncr: [50000, 150000], huc: [35000, 100000], other: [20000, 60000] }
const makeTuition = ct => Math.round(rand(...TUITION[ct]) / 1000) * 1000

const SEATS_PER_CLASSROOM = 40
const TARGET_REGIONS = ['NCR', 'Region IV-A']

// --- Load & filter raw data ---

const pubRaw = parseCSV(PUB_CSV)
const priRaw = parseCSV(PRI_CSV)

const hasCoords = s => s.latitude && s.longitude && s.coord_status !== 'no_coords'
const inScope   = s => TARGET_REGIONS.includes(s.region) && s.enrollment_status === 'active' && hasCoords(s)

const pubScope = pubRaw.filter(inScope)
const priScope = priRaw.filter(inScope)

// --- Build school objects from real data ---

function baseProps(raw, schoolType) {
  const region   = raw.region === 'NCR' ? 'ncr' : 'iva'
  const province = normalizeProvince(raw.province, raw.region)
  const city     = normalizeCity(raw.municipality)
  const ct       = cityType(raw.region, raw.municipality)
  const coords   = [parseFloat(raw.longitude), parseFloat(raw.latitude)]
  return {
    school_id:   raw.school_id,
    name:        toTitleCase(raw.school_name),
    school_type: schoolType,
    region,
    city,
    city_type:   ct,
    province,
    barangay:    toTitleCase(raw.barangay),
    coordinates: coords,
  }
}

// Public ES (origin schools) — all public ES-only schools in scope
const publicES = pubScope
  .filter(s => s.offers_es === 'True' && s.offers_jhs === 'False')
  .map(s => {
    const base          = baseProps(s, 'public_es')
    const num_classrooms = randInt(8, 20)
    const seats          = num_classrooms * SEATS_PER_CLASSROOM
    const enrollment     = Math.round(seats * rand(0.75, 1.05))
    return {
      ...base,
      num_classrooms,
      seats,
      enrollment,
      congestion_ratio: round1(enrollment / num_classrooms),
    }
  })

// Public JHS — all public JHS schools in scope
const publicJHS = pubScope
  .filter(s => s.offers_jhs === 'True')
  .map(s => {
    const base           = baseProps(s, 'public_jhs')
    const isNcr          = base.region === 'ncr'
    const num_classrooms = randInt(15, 40)
    const seats          = num_classrooms * SEATS_PER_CLASSROOM
    const congestion_ratio = round1(rand(isNcr ? 40 : 30, isNcr ? 55 : 50))
    const enrollment       = Math.round(congestion_ratio * num_classrooms)
    return {
      ...base,
      num_classrooms,
      seats,
      enrollment,
      congestion_ratio,
    }
  })

// Private JHS no ESC — mostly below capacity but ~25% can be over
const privateJHS = priScope
  .filter(s => s.offers_jhs === 'True' && s.esc_participating === '0')
  .map(s => {
    const base           = baseProps(s, 'private_jhs')
    const num_classrooms = randInt(8, 25)
    const seats          = num_classrooms * SEATS_PER_CLASSROOM
    const enrollment     = Math.round(seats * rand(0.65, 1.10))
    return {
      ...base,
      num_classrooms,
      seats,
      enrollment,
      congestion_ratio: round1(enrollment / num_classrooms),
      tuition_annual:   makeTuition(base.city_type),
    }
  })

// Private JHS ESC — slots ≤ seats; total enrollment can exceed seats
const privateESC = priScope
  .filter(s => s.offers_jhs === 'True' && s.esc_participating === '1')
  .map(s => {
    const base              = baseProps(s, 'private_jhs_esc')
    const num_classrooms    = randInt(10, 30)
    const seats             = num_classrooms * SEATS_PER_CLASSROOM
    const base_enrollment   = Math.round(seats * rand(0.65, 1.00))
    const slots_total       = Math.min(randInt(20, 100), seats)
    const slots_available   = Math.round(slots_total * rand(0.20, 0.65))
    const enrollment        = base_enrollment + (slots_total - slots_available)
    return {
      ...base,
      num_classrooms,
      seats,
      enrollment,
      congestion_ratio: round1(enrollment / num_classrooms),
      tuition_annual:   makeTuition(base.city_type),
      slots_total,
      slots_available,
    }
  })

const allSchools = [...publicES, ...publicJHS, ...privateJHS, ...privateESC]

// --- Student generation ---
// Use a random sample of ES schools as origins to keep generation fast.
// 2,000 students distributed across origins — enough for meaningful ESC demand clusters.

const ES_ORIGIN_SAMPLE = 300
const STUDENT_COUNT    = 2000

function sampleES(schools, n) {
  const shuffled = [...schools].sort(() => Math.random() - 0.5)
  return shuffled.slice(0, n)
}

function generateStudents(esOrigins, escSchools) {
  const students = []
  for (let i = 1; i <= STUDENT_COUNT; i++) {
    const origin = pick(esOrigins)
    const scored = escSchools.map(s => {
      const road          = roadDistKm(origin.coordinates, s.coordinates, origin.province, s.province)
      const regionPenalty = s.region === origin.region ? 0 : 5
      return { school_id: s.school_id, road_dist: road, score: road + regionPenalty + rand(0, 1.5) }
    })
    scored.sort((a, b) => a.score - b.score)
    students.push({
      student_id:        `STU_${String(i).padStart(4, '0')}`,
      origin_school_id:  origin.school_id,
      region:            origin.region,
      city_type:         origin.city_type,
      rank1_school_id:   scored[0]?.school_id ?? null,
      rank2_school_id:   scored[1]?.school_id ?? null,
      rank3_school_id:   scored[2]?.school_id ?? null,
      dist_rank1_km:     scored[0] ? round1(scored[0].road_dist) : null,
      dist_rank2_km:     scored[1] ? round1(scored[1].road_dist) : null,
      dist_rank3_km:     scored[2] ? round1(scored[2].road_dist) : null,
    })
  }
  return students
}

// --- Flow generation ---

function generateFlows(students) {
  const counts = {}, dists = {}
  for (const s of students) {
    if (!s.rank1_school_id) continue
    const key = `${s.origin_school_id}||${s.rank1_school_id}`
    counts[key] = (counts[key] || 0) + 1
    if (!dists[key]) dists[key] = []
    if (s.dist_rank1_km != null) dists[key].push(s.dist_rank1_km)
  }
  return Object.entries(counts)
    .map(([key, student_count]) => {
      const [origin_school_id, destination_school_id] = key.split('||')
      const d = dists[key] || []
      const avg_distance_km = d.length ? round1(d.reduce((a, b) => a + b, 0) / d.length) : 0
      return { origin_school_id, destination_school_id, student_count, avg_distance_km }
    })
    .sort((a, b) => b.student_count - a.student_count)
}

// --- GeoJSON ---

function toGeoJSON(schools) {
  return {
    type: 'FeatureCollection',
    features: schools.map(({ coordinates, ...props }) => ({
      type:     'Feature',
      geometry: { type: 'Point', coordinates },
      properties: props,
    })),
  }
}

// --- Run ---

console.log('\nSampling ES origins and generating students …')
const esOrigins = sampleES(publicES, Math.min(ES_ORIGIN_SAMPLE, publicES.length))
const students  = generateStudents(esOrigins, privateESC)
const flows     = generateFlows(students)

writeFileSync(join(OUT, 'schools.geojson'), JSON.stringify(toGeoJSON(allSchools), null, 2))
writeFileSync(join(OUT, 'students.json'),   JSON.stringify(students, null, 2))
writeFileSync(join(OUT, 'flows.json'),      JSON.stringify(flows, null, 2))

const count = (arr, key) => arr.reduce((a, s) => ({ ...a, [s[key]]: (a[s[key]] || 0) + 1 }), {})

console.log('\nGenerated → public/data/\n')
console.log(`schools.geojson  ${allSchools.length} schools`)
console.log('  by type:     ', count(allSchools, 'school_type'))
console.log('  by region:   ', count(allSchools, 'region'))
console.log('  by city_type:', count(allSchools, 'city_type'))
console.log(`\nstudents.json    ${students.length} students`)
console.log(`flows.json       ${flows.length} origin-destination pairs`)

// Verify slot constraint
const violations = privateESC.filter(s => s.slots_total > s.seats).length
console.log(`\nESC slot constraint violations: ${violations}`)
