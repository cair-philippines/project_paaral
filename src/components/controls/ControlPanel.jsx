import { useState } from 'react'
import { ChevronDown, ChevronUp, Play, RotateCcw, Bookmark } from 'lucide-react'
import { useSimulation } from '../../hooks/useSimulation'
import { SliderInput } from './SliderInput'
import { SubsidyPanel } from './SubsidyPanel'
import { SlotBudgetPanel } from './SlotBudgetPanel'

export function ControlPanel() {
  const { params, hasRun, updateParam, runSim, saveScenario, resetParams, preview } = useSimulation()
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [scenarioLabel, setScenarioLabel] = useState('')
  const [showSaveInput, setShowSaveInput] = useState(false)

  // Live preview — recomputes on every param change without full engine run
  const livePreview = preview()

  const handleSave = () => {
    const label = scenarioLabel.trim() || `Scenario ${Date.now()}`
    saveScenario(label)
    setScenarioLabel('')
    setShowSaveInput(false)
  }

  return (
    <aside className="w-80 shrink-0 bg-white border-r border-slate-200 h-screen overflow-y-auto flex flex-col">
      {/* Header */}
      <div className="px-5 py-4 border-b border-slate-100">
        <h2 className="text-sm font-semibold text-slate-800 uppercase tracking-wide">Policy Levers</h2>
        <p className="text-xs text-slate-400 mt-0.5">Adjust parameters and run simulation</p>
      </div>

      <div className="flex-1 px-5 py-5 space-y-6">

        {/* Basic levers */}
        <SliderInput
          label="Congestion Threshold"
          value={params.threshold}
          min={25}
          max={50}
          step={1}
          onChange={v => updateParam('threshold', v)}
          formatValue={v => `${v}:1`}
          preview={
            livePreview.students_affected > 0
              ? `~${livePreview.students_affected} students affected · ~${livePreview.classrooms_saved} classrooms freed`
              : 'No reassignment needed at this threshold'
          }
        />

        <SliderInput
          label="Rank Tolerance"
          value={params.rankTolerance}
          min={0}
          max={30}
          step={1}
          onChange={v => updateParam('rankTolerance', v)}
          formatValue={v => `${v}%`}
          preview={
            params.rankTolerance === 0
              ? 'All students stay at Rank 1 choice'
              : `Up to ${params.rankTolerance}% of students may be assigned to Rank 2 or 3`
          }
        />

        {/* Advanced options toggle */}
        <div>
          <button
            onClick={() => setShowAdvanced(v => !v)}
            className="flex items-center gap-1.5 text-xs font-medium text-slate-500 hover:text-slate-700 transition-colors"
          >
            {showAdvanced ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            Advanced options
          </button>

          {showAdvanced && (
            <div className="mt-4 space-y-5 pl-1">
              <SubsidyPanel />
              <div className="border-t border-slate-100 pt-4">
                <SlotBudgetPanel />
              </div>
            </div>
          )}
        </div>

        {/* Objective function — transparency */}
        <div className="bg-slate-50 rounded-lg p-3 space-y-1">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Objective Function</p>
          <p className="text-xs font-mono text-slate-600 leading-relaxed">
            min Σ(rank_perturbation) − α·Σ(decongestion)
          </p>
          <p className="text-xs text-slate-400">
            α balances preference disruption against congestion relief
          </p>
        </div>

        {/* Live preview strip */}
        {livePreview.students_affected > 0 && (
          <div className="grid grid-cols-2 gap-2">
            {[
              { label: 'Est. affected', value: livePreview.students_affected },
              { label: 'Classrooms freed', value: livePreview.classrooms_saved },
              { label: 'At Rank 1', value: `${livePreview.pct_at_rank1}%` },
              { label: 'Slots used', value: `${livePreview.budget_utilization}%` },
            ].map(({ label, value }) => (
              <div key={label} className="bg-blue-50 rounded p-2 text-center">
                <p className="text-lg font-bold text-blue-700">{value}</p>
                <p className="text-xs text-blue-500">{label}</p>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Action buttons */}
      <div className="px-5 py-4 border-t border-slate-100 space-y-2">
        <button
          onClick={runSim}
          className="w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold py-2.5 rounded-lg transition-colors"
        >
          <Play size={14} />
          Run Simulation
        </button>

        {hasRun && (
          <>
            {showSaveInput ? (
              <div className="flex gap-2">
                <input
                  autoFocus
                  value={scenarioLabel}
                  onChange={e => setScenarioLabel(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && handleSave()}
                  placeholder="Scenario name…"
                  className="flex-1 text-sm border border-slate-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-400"
                />
                <button
                  onClick={handleSave}
                  className="text-sm bg-slate-700 text-white px-3 py-1.5 rounded hover:bg-slate-800 transition-colors"
                >
                  Save
                </button>
              </div>
            ) : (
              <button
                onClick={() => setShowSaveInput(true)}
                className="w-full flex items-center justify-center gap-2 border border-slate-200 text-slate-600 text-sm py-2 rounded-lg hover:bg-slate-50 transition-colors"
              >
                <Bookmark size={14} />
                Save as Scenario
              </button>
            )}
          </>
        )}

        <button
          onClick={resetParams}
          className="w-full flex items-center justify-center gap-2 border border-slate-200 bg-white text-slate-600 text-sm py-2 rounded-lg hover:bg-slate-50 hover:border-slate-300 transition-colors"
        >
          <RotateCcw size={14} />
          Reset to Default
        </button>
      </div>
    </aside>
  )
}
