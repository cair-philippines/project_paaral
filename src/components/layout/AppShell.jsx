import { useState } from 'react'
import { AlertTriangle, Info } from 'lucide-react'
import { useSimulation } from '../../hooks/useSimulation'
import { ControlPanel } from '../controls/ControlPanel'
import { SummaryCards } from '../visualizations/SummaryCards'
import { FlowSankey } from '../visualizations/FlowSankey'
import { ScenarioTable } from '../visualizations/ScenarioTable'
import { MapPanel } from '../visualizations/MapPanel'

const TABS = [
  { key: 'flow',      label: 'Flow Visualization' },
  { key: 'map',       label: 'School Map'         },
  { key: 'scenarios', label: 'Scenarios'           },
]

function InlineAlert({ variant, children }) {
  const styles = {
    warning: { wrapper: 'bg-amber-50 border-amber-200 text-amber-800', icon: <AlertTriangle size={15} className="shrink-0 mt-0.5" /> },
    info:    { wrapper: 'bg-blue-50  border-blue-200  text-blue-800',  icon: <Info          size={15} className="shrink-0 mt-0.5" /> },
  }
  const { wrapper, icon } = styles[variant]
  return (
    <div className={`flex gap-2 border rounded-lg px-4 py-3 text-sm ${wrapper}`}>
      {icon}
      <span>{children}</span>
    </div>
  )
}

export function AppShell() {
  const [activeTab, setActiveTab] = useState('flow')
  const { results, hasRun } = useSimulation()

  return (
    <div className="flex min-h-screen bg-slate-50">
      <ControlPanel />

      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">

        {/* Header */}
        <header className="bg-white border-b border-slate-200 px-6 py-3 flex items-baseline gap-3 shrink-0">
          <h1 className="text-base font-bold text-slate-800">PAARAL</h1>
          <span className="text-slate-300">|</span>
          <span className="text-sm text-slate-500">DepEd Planning View — ESC Slot Optimization</span>
          <span className="ml-auto text-xs text-slate-400 bg-slate-100 px-2 py-0.5 rounded">Mockup</span>
        </header>

        {/* Main content */}
        <main className="flex-1 overflow-y-auto p-6 space-y-5">

          {/* Edge case alerts — shown after first run */}
          {hasRun && results?.edgeCases?.infeasible && (
            <InlineAlert variant="warning">
              <strong>Infeasible scenario: </strong>
              {results.edgeCases.infeasibleReason}
              {' '}Options: increase rank tolerance, lower slot budget threshold, or relax congestion target.
            </InlineAlert>
          )}
          {hasRun && results?.edgeCases?.unusedSlots && !results?.edgeCases?.infeasible && (
            <InlineAlert variant="info">
              <strong>{results.edgeCases.unusedSlotCount.toLocaleString()} slots unused. </strong>
              Possible reasons: {results.edgeCases.unusedSlotReasons.join('; ')}.
            </InlineAlert>
          )}

          {/* Summary cards */}
          <SummaryCards />

          {/* Tabbed visualizations */}
          <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">

            {/* Tab bar */}
            <div className="flex border-b border-slate-200 px-1 pt-1">
              {TABS.map(({ key, label }) => (
                <button
                  key={key}
                  onClick={() => setActiveTab(key)}
                  className={`px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${
                    activeTab === key
                      ? 'border-blue-600 text-blue-700'
                      : 'border-transparent text-slate-500 hover:text-slate-700'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>

            {/* Tab content */}
            <div className="p-5">
              {activeTab === 'flow'      && <FlowSankey />}
              {activeTab === 'map'       && <MapPanel />}
              {activeTab === 'scenarios' && <ScenarioTable />}
            </div>
          </div>

          {/* Objective function — always visible at bottom for transparency */}
          <div className="text-xs text-slate-400 text-center pb-2">
            Objective: <span className="font-mono">min Σ(rank_perturbation) − α·Σ(decongestion)</span>
            {' · '}Mockup — results are illustrative, not research-validated
          </div>

        </main>
      </div>
    </div>
  )
}
