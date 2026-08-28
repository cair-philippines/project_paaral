"use client";

import { useEffect, useMemo, useState } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import CircularProgress from "@mui/material/CircularProgress";
import SiteHeader from "@/components/organisms/SiteHeader";
import FilterSidebar from "@/components/organisms/FilterSidebar";
import SchoolMap from "@/components/organisms/SchoolMap";
import SchoolResultCard from "@/components/organisms/SchoolResultCard";
import ViewToggle, { type BrowseViewMode } from "@/components/molecules/ViewToggle";
import { useSchoolFilters } from "@/hooks/useSchoolFilters";
import { getBarangayOptions, fetchSchools } from "@/lib/schools";
import type { School } from "@/types/school";

const PANEL_WIDTH = 320;

/** Live school data (Chunk 16, step 7) — fetched once on mount from
 * `paaral-student-api` (Option A: fetch everything, filter client-side
 * via `useSchoolFilters`, exactly as when this read the bundled
 * `qc-schools.json`). See `docs/post-pilot-scaling.md` for why this
 * approach, and what changes if it stops being a good fit. */
export default function BrowsePage() {
  const [schools, setSchools] = useState<School[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchSchools()
      .then((data) => {
        if (!cancelled) setSchools(data);
      })
      .catch(() => {
        if (!cancelled) {
          setLoadError(
            "Couldn't load the school list. Please check your connection and try again."
          );
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const barangayOptions = useMemo(
    () => getBarangayOptions(schools ?? []),
    [schools]
  );

  const {
    filters,
    feeBounds,
    subsidyBounds,
    netFeeBounds,
    filteredSchools,
    setSearch,
    toggleSchoolType,
    setEscOnly,
    setBarangay,
    setFeeRange,
    setSubsidyRange,
    setNetFeeRange,
    resetFilters,
  } = useSchoolFilters(schools ?? []);

  const [viewMode, setViewMode] = useState<BrowseViewMode>("map");
  const [selectedSchool, setSelectedSchool] = useState<School | null>(null);
  const [filtersOpen, setFiltersOpen] = useState(true);

  if (loadError) {
    return (
      <div className="flex h-screen flex-col overflow-hidden">
        <SiteHeader />
        <div className="flex flex-1 items-center justify-center bg-background p-6">
          <p className="max-w-sm rounded border border-red-200 bg-red-50 p-4 text-center text-sm text-red-700">
            {loadError}
          </p>
        </div>
      </div>
    );
  }

  if (schools === null) {
    return (
      <div className="flex h-screen flex-col overflow-hidden">
        <SiteHeader />
        <div className="flex flex-1 items-center justify-center bg-background">
          <CircularProgress />
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen flex-col overflow-hidden">
      <SiteHeader />
      <div className="relative flex-1 overflow-hidden bg-background">
        <div className="absolute inset-0">
          {viewMode === "map" && (
            <SchoolMap
              schools={filteredSchools}
              selectedSchoolId={selectedSchool?.school_id ?? null}
              onSelectSchool={setSelectedSchool}
            />
          )}

          {viewMode === "list" && (
            <div className="h-full overflow-y-auto p-6 pt-20">
              <div className="mx-auto flex max-w-2xl flex-col gap-3">
                {filteredSchools.map((school) => (
                  <SchoolResultCard
                    key={school.school_id}
                    school={school}
                    variant="list"
                    selected={school.school_id === selectedSchool?.school_id}
                    onSelect={setSelectedSchool}
                  />
                ))}
              </div>
            </div>
          )}

          {viewMode === "card" && (
            <div className="h-full overflow-y-auto p-6 pt-20">
              <div className="mx-auto grid max-w-5xl grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {filteredSchools.map((school) => (
                  <SchoolResultCard
                    key={school.school_id}
                    school={school}
                    variant="card"
                    selected={school.school_id === selectedSchool?.school_id}
                    onSelect={setSelectedSchool}
                  />
                ))}
              </div>
            </div>
          )}
        </div>

        <div
          className="absolute inset-y-0 left-0 z-20 overflow-hidden shadow-2xl transition-[width] duration-300"
          style={{ width: filtersOpen ? PANEL_WIDTH : 0 }}
        >
          <FilterSidebar
            filters={filters}
            feeBounds={feeBounds}
            subsidyBounds={subsidyBounds}
            netFeeBounds={netFeeBounds}
            barangayOptions={barangayOptions}
            resultCount={filteredSchools.length}
            onSearchChange={setSearch}
            onSchoolTypeToggle={toggleSchoolType}
            onEscOnlyChange={setEscOnly}
            onBarangayChange={setBarangay}
            onFeeRangeChange={setFeeRange}
            onSubsidyRangeChange={setSubsidyRange}
            onNetFeeRangeChange={setNetFeeRange}
            onReset={resetFilters}
          />
        </div>

        <button
          type="button"
          onClick={() => setFiltersOpen((open) => !open)}
          className="absolute top-4 z-30 flex h-9 w-9 items-center justify-center rounded-full bg-white shadow-md transition-[left] duration-300 hover:bg-slate-50"
          style={{ left: (filtersOpen ? PANEL_WIDTH : 0) + 12 }}
          aria-label={filtersOpen ? "Hide filters" : "Show filters"}
        >
          {filtersOpen ? (
            <ChevronLeft size={16} className="text-slate-600" />
          ) : (
            <ChevronRight size={16} className="text-slate-600" />
          )}
        </button>

        <div className="absolute top-4 right-4 z-20 flex items-center gap-3 rounded-full bg-white px-4 py-2 shadow-md">
          <p className="text-sm text-slate-500">
            {filteredSchools.length} schools in Quezon City
          </p>
          <ViewToggle value={viewMode} onChange={setViewMode} />
        </div>
      </div>
    </div>
  );
}
