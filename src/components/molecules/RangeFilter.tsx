import Slider from "@mui/material/Slider";
import FilterGroup from "@/components/molecules/FilterGroup";

interface RangeFilterProps {
  label: string;
  value: [number, number];
  bounds: [number, number];
  format: (value: number) => string;
  onChange: (value: [number, number]) => void;
}

/** One min/max slider filter group — shared by the fee, subsidy, and net
 * fee filters in the browse sidebar. */
export default function RangeFilter({
  label,
  value,
  bounds,
  format,
  onChange,
}: RangeFilterProps) {
  return (
    <FilterGroup label={label}>
      <div className="px-1">
        <Slider
          size="small"
          value={value}
          min={bounds[0]}
          max={bounds[1]}
          disabled={bounds[0] === bounds[1]}
          onChange={(_, next) => onChange(next as [number, number])}
          valueLabelDisplay="off"
        />
        <div className="flex justify-between text-xs text-slate-500">
          <span>{format(value[0])}</span>
          <span>{format(value[1])}</span>
        </div>
      </div>
    </FilterGroup>
  );
}
