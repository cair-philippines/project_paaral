import { createContext, useContext, useReducer, useEffect } from 'react'
import { getBaseline, runSimulation } from '../engine/optimizer'

const SimulationContext = createContext(null)

export const INITIAL_PARAMS = {
  threshold:     43,
  rankTolerance: 0,
  subsidies:     { ncr: 13000, huc: 11000, other: 9000 },
  slotBudget:    { ncr: 3000,  iva: 2000 },
}

function reducer(state, action) {
  switch (action.type) {
    case 'SET_BASELINE':
      return { ...state, baseline: action.payload }
    case 'UPDATE_PARAM':
      return { ...state, params: { ...state.params, [action.key]: action.value } }
    case 'UPDATE_SUBSIDY':
      return { ...state, params: { ...state.params, subsidies: { ...state.params.subsidies, [action.key]: action.value } } }
    case 'UPDATE_SLOT_BUDGET':
      return { ...state, params: { ...state.params, slotBudget: { ...state.params.slotBudget, [action.key]: action.value } } }
    case 'RUN':
      return { ...state, results: runSimulation(state.params), hasRun: true }
    case 'SAVE_SCENARIO':
      return {
        ...state,
        scenarios: [...state.scenarios, { label: action.label, results: state.results, params: { ...state.params } }],
      }
    case 'RESET':
      return { ...state, params: INITIAL_PARAMS, results: null, hasRun: false, scenarios: [] }
    default:
      return state
  }
}

export function SimulationProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, {
    params:    INITIAL_PARAMS,
    baseline:  null,
    results:   null,
    scenarios: [],
    hasRun:    false,
  })

  useEffect(() => {
    dispatch({ type: 'SET_BASELINE', payload: getBaseline() })
  }, [])

  const ctx = {
    ...state,
    updateParam:     (key, value) => dispatch({ type: 'UPDATE_PARAM',     key, value }),
    updateSubsidy:   (key, value) => dispatch({ type: 'UPDATE_SUBSIDY',   key, value }),
    updateSlotBudget:(key, value) => dispatch({ type: 'UPDATE_SLOT_BUDGET', key, value }),
    runSim:          ()           => dispatch({ type: 'RUN' }),
    saveScenario:    (label)      => dispatch({ type: 'SAVE_SCENARIO', label }),
    resetParams:     ()           => dispatch({ type: 'RESET' }),
  }

  return <SimulationContext.Provider value={ctx}>{children}</SimulationContext.Provider>
}

export const useSimulationContext = () => useContext(SimulationContext)
