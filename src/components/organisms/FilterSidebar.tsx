import TextField from "@mui/material/TextField";
import InputAdornment from "@mui/material/InputAdornment";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Checkbox from "@mui/material/Checkbox";
import FormControlLabel from "@mui/material/FormControlLabel";
import Switch from "@mui/material/Switch";
import { Search } from "lucide-react";
import FilterGroup from "@/components/molecules/FilterGroup";
import RangeFilter from "@/components/molecules/RangeFilter";
import { pesos, titleCase } from "@/lib/schools";
import type {
  SchoolFilters,
  SchoolTypeFilter,
} from "@/hooks/useSchoolFilters";

interface FilterSidebarProps {
  filters: SchoolFilters;
  feeBounds: [number, number];
  subsidyBounds: [number, number];
  netFeeBounds: [number, number];
  barangayOptions: string[];
  resultCount: number;
  onSearchChange: (value: string) => void;
  onSchoolTypeToggle: (type: SchoolTypeFilter) => void;
  onEscOnlyChange: (value: boolean) => void;
  onBarangayChange: (value: string | null) => void;
  onFeeRangeChange: (value: [number, number]) => void;
  onSubsidyRangeChange: (value: [number, number]) => void;
  onNetFeeRangeChange: (value: [number, number]) => void;
  onReset: () => void;
}

const SCHOOL_TYPES: { value: SchoolTypeFilter; label: string }[] = [
  { value: "public", label: "Public" },
  { value: "private", label: "Private" },
];

/** Filter sidebar for the school-search/browse page — scoped to Quezon
 * City only for now, so the municipality field is fixed rather than a
 * live filter. Everything else (barangay, type, ESC participation, fee/
 * subsidy/net-fee ranges) filters the real BigQuery-sourced dataset. */
export default function FilterSidebar({
  filters,
  feeBounds,
  subsidyBounds,
  netFeeBounds,
  barangayOptions,
  resultCount,
  onSearchChange,
  onSchoolTypeToggle,
  onEscOnlyChange,
  onBarangayChange,
  onFeeRangeChange,
  onSubsidyRangeChange,
  onNetFeeRangeChange,
  onReset,
}: FilterSidebarProps) {
  return (
    <aside className="flex h-full w-full flex-col gap-6 overflow-y-auto bg-white p-6">
      <div className="flex items-center justify-between">
        <p className="font-semibold text-primary">Filters</p>
        <button
          type="button"
          onClick={onReset}
          className="text-xs font-semibold text-primary hover:underline"
        >
          Reset
        </button>
      </div>

      <TextField
        size="small"
        placeholder="Search school name"
        value={filters.search}
        onChange={(e) => onSearchChange(e.target.value)}
        slotProps={{
          input: {
            startAdornment: (
              <InputAdornment position="start">
                <Search size={16} />
              </InputAdornment>
            ),
          },
        }}
      />

      <FilterGroup label="Location">
        <Select size="small" value="Quezon City" disabled>
          <MenuItem value="Quezon City">Quezon City</MenuItem>
        </Select>
        <Select
          size="small"
          displayEmpty
          value={filters.barangay ?? ""}
          onChange={(e) => onBarangayChange(e.target.value || null)}
        >
          <MenuItem value="">All barangays</MenuItem>
          {barangayOptions.map((barangay) => (
            <MenuItem key={barangay} value={barangay}>
              {titleCase(barangay)}
            </MenuItem>
          ))}
        </Select>
      </FilterGroup>

      <FilterGroup label="School Type">
        <div className="flex flex-col">
          {SCHOOL_TYPES.map((type) => (
            <FormControlLabel
              key={type.value}
              control={
                <Checkbox
                  size="small"
                  checked={filters.schoolTypes.has(type.value)}
                  onChange={() => onSchoolTypeToggle(type.value)}
                />
              }
              label={type.label}
            />
          ))}
        </div>
      </FilterGroup>

      <FilterGroup label="ESC Participation">
        <FormControlLabel
          control={
            <Switch
              size="small"
              checked={filters.escOnly}
              onChange={(e) => onEscOnlyChange(e.target.checked)}
            />
          }
          label="ESC-participating schools only"
        />
      </FilterGroup>

      <RangeFilter
        label="Total Fees (per year)"
        value={filters.feeRange}
        bounds={feeBounds}
        format={pesos}
        onChange={onFeeRangeChange}
      />

      <RangeFilter
        label="Subsidy Amount"
        value={filters.subsidyRange}
        bounds={subsidyBounds}
        format={pesos}
        onChange={onSubsidyRangeChange}
      />

      <RangeFilter
        label="Net Fees (after subsidy)"
        value={filters.netFeeRange}
        bounds={netFeeBounds}
        format={pesos}
        onChange={onNetFeeRangeChange}
      />

      <p className="text-xs text-slate-500">{resultCount} schools found</p>
    </aside>
  );
}
