"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Search } from "lucide-react";
import CircularProgress from "@mui/material/CircularProgress";
import SiteHeader from "@/components/organisms/SiteHeader";
import BrowseHero from "@/components/organisms/BrowseHero";
import BrowseFilterBar from "@/components/organisms/BrowseFilterBar";
import BrowseCallToAction from "@/components/organisms/BrowseCallToAction";
import SchoolMap from "@/components/organisms/SchoolMap";
import SchoolResultCard from "@/components/organisms/SchoolResultCard";
import ViewToggle, { type BrowseViewMode } from "@/components/molecules/ViewToggle";
import BrowsePagination from "@/components/molecules/BrowsePagination";
import { useApplication } from "@/components/templates/ApplicationStateProvider";
import { useSchoolFilters } from "@/hooks/useSchoolFilters";
import { getBarangayOptions, fetchSchools } from "@/lib/schools";
import type { School } from "@/types/school";

const PAGE_SIZE = 24;

/** Live school data (Chunk 16, step 7) — fetched once on mount from
 * `paaral-student-api` (Option A: fetch everything, filter client-side
 * via `useSchoolFilters`, exactly as when this read the bundled
 * `qc-schools.json`). See `docs/post-pilot-scaling.md` for why this
 * approach, and what changes if it stops being a good fit.
 *
 * Page structure rebuilt 2026-09-08, adapting the "schoolpath-portal"
 * reference's browse experience (its guest `BrowseSchools` page plus its
 * logged-in `SchoolFinder`, embedded in `FamilyDashboard`) — a normal
 * scrolling page, not the full-screen immersive map this page used from
 * 2026-08-21 (the Chile vitrina benchmark). PAARAL's own tokens/typography
 * throughout; only the page's shape is borrowed. */
export default function BrowsePage() {
  const { account, wishlist } = useApplication();
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
  const resultsHeaderRef = useRef<HTMLDivElement>(null);

  const [page, setPage] = useState(1);
  const totalPages = Math.max(1, Math.ceil(filteredSchools.length / PAGE_SIZE));
  const pagedSchools = filteredSchools.slice(
    (page - 1) * PAGE_SIZE,
    page * PAGE_SIZE
  );

  // Any actual filter change should return to page 1 — a stale page number
  // from a broader search (e.g. sitting on page 12) would otherwise show an
  // empty results panel until the person also thought to reset the page.
  // Adjusted during render (React's own recommended pattern for resetting
  // state when a derived value changes), not in an effect, since an effect
  // would need a second render just to apply the reset.
  const filterSignature = JSON.stringify({
    search: filters.search,
    types: Array.from(filters.schoolTypes).sort(),
    escOnly: filters.escOnly,
    barangay: filters.barangay,
    fee: filters.feeRange,
    subsidy: filters.subsidyRange,
    netFee: filters.netFeeRange,
  });
  const [prevFilterSignature, setPrevFilterSignature] =
    useState(filterSignature);
  if (filterSignature !== prevFilterSignature) {
    setPrevFilterSignature(filterSignature);
    setPage(1);
  }

  const goToPage = (next: number) => {
    setPage(next);
    resultsHeaderRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  };

  // Clicking a map marker must make the matching card visible in the list
  // beside it — the list is paginated but the map shows every filtered
  // school's pin (a spatial overview loses its point if it only shows one
  // page's worth), so a clicked school outside the current page previously
  // just... didn't appear. Jump to whichever page actually contains it and
  // scroll that specific card into view once it's rendered. Kept separate
  // from plain list/card selection (which just calls `setSelectedSchool`
  // directly) — re-centering the page around a card someone just clicked,
  // one they can already see, would be a jarring, unasked-for jump.
  //
  // A ref, not state, holds the pending target: the effect below needs to
  // clear it once handled, and clearing it via setState from inside an
  // effect triggers an avoidable extra render (and this repo's own
  // react-hooks/set-state-in-effect lint rule) for something that isn't
  // ever rendered anyway.
  const pendingScrollToId = useRef<string | null>(null);

  const selectSchoolFromMap = (school: School | null) => {
    setSelectedSchool(school);
    if (!school) return;
    const index = filteredSchools.findIndex(
      (s) => s.school_id === school.school_id
    );
    if (index === -1) return;
    const targetPage = Math.floor(index / PAGE_SIZE) + 1;
    pendingScrollToId.current = school.school_id;
    if (targetPage !== page) setPage(targetPage);
  };

  useEffect(() => {
    if (!pendingScrollToId.current) return;
    document
      .getElementById(`browse-school-card-${pendingScrollToId.current}`)
      ?.scrollIntoView({ behavior: "instant", block: "center" });
    pendingScrollToId.current = null;
  }, [selectedSchool, page]);

  if (loadError) {
    return (
      <div className="min-h-screen bg-background">
        <SiteHeader />
        <div className="flex items-center justify-center p-16">
          <p className="max-w-sm rounded border border-red-200 bg-red-50 p-4 text-center text-sm text-red-700">
            {loadError}
          </p>
        </div>
      </div>
    );
  }

  if (schools === null) {
    return (
      <div className="min-h-screen bg-background">
        <SiteHeader />
        <div className="flex items-center justify-center p-16">
          <CircularProgress />
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <SiteHeader />
      <BrowseHero loggedIn={Boolean(account)} wishlistCount={wishlist.length} />

      <main>
        <section className="mx-auto max-w-6xl px-6 py-8 md:px-12 md:py-10">
          <BrowseFilterBar
            filters={filters}
            feeBounds={feeBounds}
            subsidyBounds={subsidyBounds}
            netFeeBounds={netFeeBounds}
            barangayOptions={barangayOptions}
            onSearchChange={setSearch}
            onSchoolTypeToggle={toggleSchoolType}
            onEscOnlyChange={setEscOnly}
            onBarangayChange={setBarangay}
            onFeeRangeChange={setFeeRange}
            onSubsidyRangeChange={setSubsidyRange}
            onNetFeeRangeChange={setNetFeeRange}
            onReset={resetFilters}
          />

          <div
            ref={resultsHeaderRef}
            className="mt-6 flex flex-wrap items-center justify-between gap-3 scroll-mt-24"
          >
            <p className="text-sm font-semibold text-slate-600">
              {filteredSchools.length === 0 ? (
                "0 schools shown"
              ) : (
                <>
                  Showing{" "}
                  <span className="font-bold text-primary">
                    {(page - 1) * PAGE_SIZE + 1}–
                    {Math.min(page * PAGE_SIZE, filteredSchools.length)}
                  </span>{" "}
                  of{" "}
                  <span className="font-bold text-primary">
                    {filteredSchools.length}
                  </span>{" "}
                  schools
                </>
              )}
            </p>
            <ViewToggle value={viewMode} onChange={setViewMode} />
          </div>

          <div className="mt-5">
            {viewMode === "map" && (
              <div className="grid gap-5 lg:grid-cols-[1fr_1fr] lg:items-start">
                <div className="h-[420px] overflow-hidden rounded-2xl border border-slate-100 shadow-sm lg:sticky lg:top-24 lg:h-[560px]">
                  <SchoolMap
                    schools={filteredSchools}
                    selectedSchoolId={selectedSchool?.school_id ?? null}
                    onSelectSchool={selectSchoolFromMap}
                  />
                </div>
                <div className="flex flex-col gap-3">
                  {pagedSchools.map((school) => (
                    <SchoolResultCard
                      key={school.school_id}
                      id={`browse-school-card-${school.school_id}`}
                      school={school}
                      variant="list"
                      selected={school.school_id === selectedSchool?.school_id}
                      onSelect={setSelectedSchool}
                    />
                  ))}
                  {filteredSchools.length === 0 && <EmptyResults />}
                </div>
              </div>
            )}

            {viewMode === "list" && (
              <div className="mx-auto flex max-w-2xl flex-col gap-3">
                {pagedSchools.map((school) => (
                  <SchoolResultCard
                    key={school.school_id}
                    school={school}
                    variant="list"
                    selected={school.school_id === selectedSchool?.school_id}
                    onSelect={setSelectedSchool}
                  />
                ))}
                {filteredSchools.length === 0 && <EmptyResults />}
              </div>
            )}

            {viewMode === "card" && (
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {pagedSchools.map((school) => (
                  <SchoolResultCard
                    key={school.school_id}
                    school={school}
                    variant="card"
                    selected={school.school_id === selectedSchool?.school_id}
                    onSelect={setSelectedSchool}
                  />
                ))}
                {filteredSchools.length === 0 && <EmptyResults />}
              </div>
            )}
          </div>

          <BrowsePagination
            page={page}
            totalPages={totalPages}
            onChange={goToPage}
          />
        </section>
      </main>

      <BrowseCallToAction wishlistCount={wishlist.length} />
    </div>
  );
}

function EmptyResults() {
  return (
    <div className="col-span-full rounded-2xl border border-dashed border-slate-300 bg-white p-12 text-center">
      <Search className="mx-auto h-6 w-6 text-slate-400" />
      <p className="mt-3 font-bold text-primary">
        No schools match those filters.
      </p>
      <p className="mt-1 text-sm text-slate-500">
        Try a broader search or reset one of the filters above.
      </p>
    </div>
  );
}
