"use client";

import { useState } from "react";
import TextField from "@mui/material/TextField";
import InputAdornment from "@mui/material/InputAdornment";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import { BadgeCheck, ChevronDown, SlidersHorizontal, Search } from "lucide-react";
import RangeFilter from "@/components/molecules/RangeFilter";
import { pesos, titleCase } from "@/lib/schools";
import type {
  SchoolFilters,
  SchoolTypeFilter,
} from "@/hooks/useSchoolFilters";

interface BrowseFilterBarProps {
  filters: SchoolFilters;
  feeBounds: [number, number];
  subsidyBounds: [number, number];
  netFeeBounds: [number, number];
  barangayOptions: string[];
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

/**
 * Horizontal filter bar for `/browse` — restructured (2026-09-08) from a
 * collapsible left sidebar into a single card in the normal page flow,
 * matching the "schoolpath-portal" reference's browse filter bar shape:
 * one scannable row of the filters most people actually use (search,
 * barangay, school type, ESC participation), with the three fee/subsidy/
 * net-fee sliders tucked behind a "More filters" disclosure so the primary
 * row doesn't get crowded — PAARAL has three range filters where the
 * reference only has one, so folding them away (rather than shipping them
 * all inline) is this app's own adaptation, not a straight port.
 */
export default function BrowseFilterBar({
  filters,
  feeBounds,
  subsidyBounds,
  netFeeBounds,
  barangayOptions,
  onSearchChange,
  onSchoolTypeToggle,
  onEscOnlyChange,
  onBarangayChange,
  onFeeRangeChange,
  onSubsidyRangeChange,
  onNetFeeRangeChange,
  onReset,
}: BrowseFilterBarProps) {
  const [moreOpen, setMoreOpen] = useState(false);

  return (
    <div className="rounded-2xl border border-slate-100 bg-white p-4 shadow-sm md:p-5">
      <div className="grid gap-3 lg:grid-cols-[1.4fr_.85fr_auto_auto]">
        <TextField
          size="small"
          placeholder="Search school name"
          value={filters.search}
          onChange={(e) => onSearchChange(e.target.value)}
          slotProps={{
            input: {
              startAdornment: (
                <InputAdornment position="start">
                  <Search size={16} className="text-slate-400" />
                </InputAdornment>
              ),
            },
          }}
        />

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

        <div className="flex items-center gap-1.5">
          {SCHOOL_TYPES.map((type) => {
            const active = filters.schoolTypes.has(type.value);
            return (
              <button
                key={type.value}
                type="button"
                onClick={() => onSchoolTypeToggle(type.value)}
                aria-pressed={active}
                className={`flex h-10 items-center justify-center rounded-xl border px-3.5 text-sm font-bold transition ${
                  active
                    ? "border-primary bg-primary/5 text-primary"
                    : "border-slate-200 bg-white text-slate-600 hover:border-slate-300"
                }`}
              >
                {type.label}
              </button>
            );
          })}
        </div>

        <button
          type="button"
          onClick={() => onEscOnlyChange(!filters.escOnly)}
          aria-pressed={filters.escOnly}
          className={`flex h-10 items-center justify-center gap-2 whitespace-nowrap rounded-xl border px-3.5 text-sm font-bold transition ${
            filters.escOnly
              ? "border-primary bg-primary/5 text-primary"
              : "border-slate-200 bg-white text-slate-600 hover:border-slate-300"
          }`}
        >
          <BadgeCheck size={17} /> ESC-participating only
        </button>
      </div>

      <div className="mt-3 flex flex-wrap items-center justify-between gap-2 border-t border-slate-100 pt-3">
        <button
          type="button"
          onClick={() => setMoreOpen((open) => !open)}
          aria-expanded={moreOpen}
          className="flex h-9 items-center gap-1.5 rounded-lg px-2 text-xs font-bold text-slate-600 hover:bg-slate-50"
        >
          <SlidersHorizontal size={14} />
          More filters (fees, subsidy, slots)
          <ChevronDown
            size={14}
            className={`transition-transform ${moreOpen ? "rotate-180" : ""}`}
          />
        </button>
        <button
          type="button"
          onClick={onReset}
          className="text-xs font-semibold text-primary hover:underline"
        >
          Reset filters
        </button>
      </div>

      {moreOpen && (
        <div className="mt-3 grid gap-4 border-t border-slate-100 pt-4 sm:grid-cols-3">
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
        </div>
      )}
    </div>
  );
}
