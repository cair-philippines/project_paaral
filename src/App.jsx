import { SimulationProvider } from './context/SimulationContext'
import { AppShell } from './components/layout/AppShell'

function App() {
  return (
    <SimulationProvider>
      <AppShell />
    </SimulationProvider>
  )
}

export default App
