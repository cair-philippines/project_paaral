import { writeFileSync } from 'fs'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const OUT = join(__dirname, '../public/data')

// --- Utilities ---

const rand = (min, max) => Math.random() * (max - min) + min
const randInt = (min, max) => Math.floor(rand(min, max + 1))
const pick = arr => arr[randInt(0, arr.length - 1)]
const jitter = (val, delta) => val + rand(-delta, delta)
const round1 = v => Math.round(v * 10) / 10

function haversineKm([lng1, lat1], [lng2, lat2]) {
  const R = 6371
  const dLat = (lat2 - lat1) * Math.PI / 180
  const dLng = (lng2 - lng1) * Math.PI / 180
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLng / 2) ** 2
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
}

// Road distance via circuity factor.
// NCR dense grid ×1.3; near-NCR IVA provinces with partial expressway access ×1.4;
// IVA provincial routes (Batangas, Quezon) more indirect ×1.6.
const CIRCUITY = {
  ncr: 1.3, Cavite: 1.4, Laguna: 1.4, Rizal: 1.4, Batangas: 1.6, Quezon: 1.6,
}

function roadDistKm(coordA, coordB, provA, provB) {
  const straight = haversineKm(coordA, coordB)
  return straight * Math.max(CIRCUITY[provA] ?? 1.4, CIRCUITY[provB] ?? 1.4)
}

// Tuition ranges by city_type (annual, PHP)
const TUITION = {
  ncr:   [50000,  150000],
  huc:   [35000,  100000],
  other: [20000,   60000],
}

// --- City definitions ---

const NCR_CITIES = [
  { city: 'Manila',      city_type: 'ncr', province: 'ncr', coords: [120.9842, 14.5995] },
  { city: 'Quezon City', city_type: 'ncr', province: 'ncr', coords: [121.0437, 14.6760] },
  { city: 'Caloocan',    city_type: 'ncr', province: 'ncr', coords: [120.9667, 14.6500] },
  { city: 'Las Piñas',   city_type: 'ncr', province: 'ncr', coords: [120.9833, 14.4500] },
  { city: 'Makati',      city_type: 'ncr', province: 'ncr', coords: [121.0244, 14.5547] },
  { city: 'Malabon',     city_type: 'ncr', province: 'ncr', coords: [120.9566, 14.6625] },
  { city: 'Mandaluyong', city_type: 'ncr', province: 'ncr', coords: [121.0359, 14.5794] },
  { city: 'Marikina',    city_type: 'ncr', province: 'ncr', coords: [121.1024, 14.6507] },
  { city: 'Muntinlupa',  city_type: 'ncr', province: 'ncr', coords: [121.0415, 14.4081] },
  { city: 'Navotas',     city_type: 'ncr', province: 'ncr', coords: [120.9500, 14.6667] },
  { city: 'Parañaque',   city_type: 'ncr', province: 'ncr', coords: [121.0198, 14.4793] },
  { city: 'Pasay',       city_type: 'ncr', province: 'ncr', coords: [120.9930, 14.5378] },
  { city: 'Pasig',       city_type: 'ncr', province: 'ncr', coords: [121.0851, 14.5764] },
  { city: 'San Juan',    city_type: 'ncr', province: 'ncr', coords: [121.0355, 14.6019] },
  { city: 'Taguig',      city_type: 'ncr', province: 'ncr', coords: [121.0792, 14.5243] },
  { city: 'Valenzuela',  city_type: 'ncr', province: 'ncr', coords: [120.9830, 14.7011] },
]

const IVA_CITIES = [
  { city: 'Lucena City',   city_type: 'huc',   province: 'Quezon',   coords: [121.6170, 13.9302] },
  { city: 'Bacoor',        city_type: 'other', province: 'Cavite',   coords: [120.9576, 14.4624] },
  { city: 'Dasmariñas',    city_type: 'other', province: 'Cavite',   coords: [120.9367, 14.3294] },
  { city: 'General Trias', city_type: 'other', province: 'Cavite',   coords: [120.8815, 14.3874] },
  { city: 'Imus',          city_type: 'other', province: 'Cavite',   coords: [120.9367, 14.4297] },
  { city: 'Calamba',       city_type: 'other', province: 'Laguna',   coords: [121.1653, 14.2117] },
  { city: 'Santa Rosa',    city_type: 'other', province: 'Laguna',   coords: [121.1117, 14.3121] },
  { city: 'Biñan',         city_type: 'other', province: 'Laguna',   coords: [121.0761, 14.3414] },
  { city: 'San Pedro',     city_type: 'other', province: 'Laguna',   coords: [121.0474, 14.3588] },
  { city: 'Antipolo',      city_type: 'other', province: 'Rizal',    coords: [121.1768, 14.5883] },
  { city: 'Cainta',        city_type: 'other', province: 'Rizal',    coords: [121.1225, 14.5744] },
  { city: 'Batangas City', city_type: 'other', province: 'Batangas', coords: [121.0583, 13.7565] },
  { city: 'Lipa City',     city_type: 'other', province: 'Batangas', coords: [121.1631, 13.9411] },
  { city: 'Tanauan',       city_type: 'other', province: 'Batangas', coords: [121.0006, 14.0862] },
]

// --- Name pools ---

const PUBLIC_ES_NAMES = [
  'Bagumbayan Elementary School', 'Barangay Holy Spirit Elementary School',
  'Commonwealth Elementary School', 'Dona Aurora Elementary School',
  'Fairview Elementary School', 'Kamuning Elementary School',
  'Lagro Elementary School', 'Malaya Elementary School',
  'Novaliches Elementary School', 'Pinyahan Elementary School',
  'Project 7 Elementary School', 'Rosario Elementary School',
  'San Andres Elementary School', 'Ugong Elementary School',
  'Urdaneta Elementary School', 'Vasra Elementary School',
  'West Avenue Elementary School', 'Lucena Central Elementary School',
  'Calamba Elementary School', 'Batangas West Elementary School',
]

const PUBLIC_JHS_NAMES = [
  'Bagong Silang National High School', 'Bignay National High School',
  'Commonwealth High School', 'Culiat High School',
  'Dona Remedios Trinidad High School', 'Fairview High School',
  'Gen. T. de Leon National High School', 'Kabayanan National High School',
  'Krus na Ligas High School', 'Maligaya High School',
  'Novaliches High School', 'Pasig City Science High School',
  'Payatas National High School', 'Rosario High School',
  'San Andres National High School', 'Tandang Sora High School',
  'Ugong National High School', 'Valenzuela City School of Mathematics and Science',
  'Batangas National High School', 'Lucena National High School',
  'Calamba National High School', 'Dasmariñas National High School',
  'Antipolo National High School', 'Santa Rosa Science and Technology High School',
  'General Trias National High School', 'Imus National High School',
  'San Pedro National High School', 'Biñan National High School',
  'Cainta National High School', 'Lipa City National High School',
]

const PRIVATE_NAMES = [
  'Saint Joseph Academy', 'Holy Cross School',
  'Our Lady of Lourdes School', 'San Beda College Preparatory School',
  'Colegio de San Lorenzo', 'Saint Theresa College',
  'Holy Family School', 'Saint Francis of Assisi School',
  'Good Shepherd School', 'Immaculate Conception Academy',
  'Our Lady of Peace School', 'Saint Gabriel School',
  'Holy Redeemer School', 'Saint Michael Academy',
  'Our Lady of Fatima School', 'Saint Anthony School',
  'Colegio de la Inmaculada Concepcion', 'Saint Paul College',
  'Holy Rosary Academy', 'Saint John Bosco School',
  "Saint Mary's Academy", 'Assumption College',
  'Blessed Trinity School', "Saint Peter's College",
  'Dominican School', 'Saint Vincent de Paul School',
  'Notre Dame School', 'Sacred Heart School',
  "Saint Mark's School", 'Colegio de Santa Ana',
  "Saint Luke's Academy", 'Holy Spirit School',
  'Montessori de Sta. Rosa', 'Batangas Academy',
  'Colegio de Lucena',
]

// --- School generation ---

function generateSchools() {
  const schools = []
  let esIdx = 0, pubJhsIdx = 0, priJhsIdx = 0, escJhsIdx = 0
  let privateNameCursor = 0
  const nextPrivateName = () => PRIVATE_NAMES[privateNameCursor++ % PRIVATE_NAMES.length]

  const makeTuition = ct => Math.round(rand(...TUITION[ct]) / 1000) * 1000

  // public_es: 12 NCR, 8 IVA
  for (let i = 0; i < 20; i++) {
    const isNcr = i < 12
    const c = pick(isNcr ? NCR_CITIES : IVA_CITIES)
    schools.push({
      school_id: `ES_${String(++esIdx).padStart(3, '0')}`,
      name: PUBLIC_ES_NAMES[i % PUBLIC_ES_NAMES.length],
      school_type: 'public_es',
      region: isNcr ? 'ncr' : 'iva',
      city: c.city, city_type: c.city_type, province: c.province,
      num_classrooms: randInt(8, 20),
      coordinates: [jitter(c.coords[0], 0.02), jitter(c.coords[1], 0.02)],
    })
  }

  // public_jhs: 18 NCR, 12 IVA
  for (let i = 0; i < 30; i++) {
    const isNcr = i < 18
    const c = pick(isNcr ? NCR_CITIES : IVA_CITIES)
    const num_classrooms = randInt(15, 40)
    const congestion_ratio = round1(rand(isNcr ? 40 : 30, isNcr ? 55 : 50))
    schools.push({
      school_id: `JHS_PUB_${String(++pubJhsIdx).padStart(3, '0')}`,
      name: PUBLIC_JHS_NAMES[i % PUBLIC_JHS_NAMES.length],
      school_type: 'public_jhs',
      region: isNcr ? 'ncr' : 'iva',
      city: c.city, city_type: c.city_type, province: c.province,
      num_classrooms,
      enrollment: Math.round(congestion_ratio * num_classrooms),
      congestion_ratio,
      coordinates: [jitter(c.coords[0], 0.02), jitter(c.coords[1], 0.02)],
    })
  }

  // private_jhs (no ESC): 9 NCR, 6 IVA
  for (let i = 0; i < 15; i++) {
    const isNcr = i < 9
    const c = pick(isNcr ? NCR_CITIES : IVA_CITIES)
    schools.push({
      school_id: `JHS_PRI_${String(++priJhsIdx).padStart(3, '0')}`,
      name: nextPrivateName(),
      school_type: 'private_jhs',
      region: isNcr ? 'ncr' : 'iva',
      city: c.city, city_type: c.city_type, province: c.province,
      num_classrooms: randInt(8, 25),
      tuition_annual: makeTuition(c.city_type),
      coordinates: [jitter(c.coords[0], 0.02), jitter(c.coords[1], 0.02)],
    })
  }

  // private_jhs_esc: 21 NCR, 14 IVA
  for (let i = 0; i < 35; i++) {
    const isNcr = i < 21
    const c = pick(isNcr ? NCR_CITIES : IVA_CITIES)
    const slots_total = randInt(20, 100)
    schools.push({
      school_id: `JHS_ESC_${String(++escJhsIdx).padStart(3, '0')}`,
      name: nextPrivateName(),
      school_type: 'private_jhs_esc',
      region: isNcr ? 'ncr' : 'iva',
      city: c.city, city_type: c.city_type, province: c.province,
      num_classrooms: randInt(10, 30),
      tuition_annual: makeTuition(c.city_type),
      slots_total,
      slots_available: Math.round(slots_total * rand(0.5, 0.9)),
      coordinates: [jitter(c.coords[0], 0.02), jitter(c.coords[1], 0.02)],
    })
  }

  return schools
}

// --- Student generation ---

function generateStudents(schools) {
  const esSchools  = schools.filter(s => s.school_type === 'public_es')
  const escSchools = schools.filter(s => s.school_type === 'private_jhs_esc')
  const students   = []

  for (let i = 1; i <= 500; i++) {
    const origin = pick(esSchools)

    const scored = escSchools.map(s => {
      const road = roadDistKm(origin.coordinates, s.coordinates, origin.province, s.province)
      const regionPenalty = s.region === origin.region ? 0 : 5
      return { school_id: s.school_id, road_dist: road, score: road + regionPenalty + rand(0, 1.5) }
    })
    scored.sort((a, b) => a.score - b.score)

    students.push({
      student_id: `STU_${String(i).padStart(4, '0')}`,
      origin_school_id: origin.school_id,
      region: origin.region,
      city_type: origin.city_type,
      rank1_school_id: scored[0]?.school_id ?? null,
      rank2_school_id: scored[1]?.school_id ?? null,
      rank3_school_id: scored[2]?.school_id ?? null,
      dist_rank1_km: scored[0] ? round1(scored[0].road_dist) : null,
      dist_rank2_km: scored[1] ? round1(scored[1].road_dist) : null,
      dist_rank3_km: scored[2] ? round1(scored[2].road_dist) : null,
    })
  }

  return students
}

// --- Flow generation ---

function generateFlows(students) {
  const counts = {}
  const dists  = {}

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

// --- GeoJSON conversion ---

function toGeoJSON(schools) {
  return {
    type: 'FeatureCollection',
    features: schools.map(({ coordinates, ...properties }) => ({
      type: 'Feature',
      geometry: { type: 'Point', coordinates },
      properties,
    })),
  }
}

// --- Run ---

const schools  = generateSchools()
const students = generateStudents(schools)
const flows    = generateFlows(students)

writeFileSync(join(OUT, 'schools.geojson'), JSON.stringify(toGeoJSON(schools), null, 2))
writeFileSync(join(OUT, 'students.json'),   JSON.stringify(students, null, 2))
writeFileSync(join(OUT, 'flows.json'),      JSON.stringify(flows, null, 2))

const count = (arr, key) => arr.reduce((a, s) => ({ ...a, [s[key]]: (a[s[key]] || 0) + 1 }), {})

console.log('\nGenerated → public/data/\n')
console.log(`schools.geojson  ${schools.length} schools`)
console.log('  by type:     ', count(schools, 'school_type'))
console.log('  by region:   ', count(schools, 'region'))
console.log('  by city_type:', count(schools, 'city_type'))
console.log(`\nstudents.json    ${students.length} students`)
console.log(`flows.json       ${flows.length} origin-destination pairs`)
