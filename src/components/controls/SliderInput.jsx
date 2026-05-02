export function SliderInput({ label, value, min, max, step = 1, onChange, formatValue, preview }) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium text-slate-700">{label}</label>
        <span className="text-sm font-semibold text-blue-700">{formatValue(value)}</span>
      </div>

      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="w-full h-2 rounded-lg appearance-none cursor-pointer accent-blue-600 bg-slate-200"
      />

      <div className="flex justify-between text-xs text-slate-400">
        <span>{formatValue(min)}</span>
        <span>{formatValue(max)}</span>
      </div>

      {preview && (
        <p className="text-xs text-slate-500 italic">{preview}</p>
      )}
    </div>
  )
}
