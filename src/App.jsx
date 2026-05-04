import { useEffect, useState } from 'react'
import Dashboard from './components/Dashboard'

export default function App() {
  const [schools, setSchools]   = useState([])
  const [students, setStudents] = useState([])
  const [loading, setLoading]   = useState(true)

  useEffect(() => {
    Promise.all([
      fetch('/data/schools.geojson').then(r => r.json()),
      fetch('/data/students.json').then(r => r.json()),
    ]).then(([geo, studs]) => {
      setSchools(geo.features.map(f => ({ ...f.properties, coordinates: f.geometry.coordinates })))
      setStudents(studs)
      setLoading(false)
    })
  }, [])

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center text-slate-400 text-sm">
        Loading school data…
      </div>
    )
  }

  return <Dashboard schools={schools} students={students} />
}
