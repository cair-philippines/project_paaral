import { Sankey, Tooltip } from 'recharts'
import { useSimulation } from '../../hooks/useSimulation'

function EmptyState({ message }) {
  return (
    <div className="flex items-center justify-center h-64 text-slate-400 text-sm">
      {message}
    </div>
  )
}

// Custom node renderer for labelled rectangles
function SankeyNode({ x, y, width, height, index, payload }) {
  return (
    <g>
      <rect x={x} y={y} width={width} height={height} fill="#3b82f6" rx={3} />
      <text
        x={x + width + 8}
        y={y + height / 2}
        dy="0.35em"
        fontSize={11}
        fill="#475569"
        textAnchor="start"
      >
        {payload.name}
      </text>
    </g>
  )
}

export function FlowSankey() {
  const { results, hasRun } = useSimulation()

  if (!hasRun || !results) {
    return <EmptyState message="Run the simulation to see student flow diagram" />
  }

  const { sankeyData } = results

  if (!sankeyData?.links?.length) {
    return <EmptyState message="No student flows at current settings — try lowering the threshold or increasing rank tolerance" />
  }

  return (
    <div className="overflow-x-auto">
      <Sankey
        key={JSON.stringify(sankeyData)}
        width={700}
        height={300}
        data={sankeyData}
        nodePadding={50}
        nodeWidth={12}
        margin={{ top: 20, right: 220, bottom: 20, left: 20 }}
        node={<SankeyNode />}
        link={{ stroke: '#93c5fd', opacity: 0.5 }}
      >
        <Tooltip
          formatter={(value) => [`${value} students`, 'Flow']}
        />
      </Sankey>
    </div>
  )
}
