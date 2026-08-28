# Post-Pilot Scaling Notes

> **Purpose:** a running list of simplifications made deliberately for the
> Quezon City pilot's scope and size, paired with what each one would need
> to become before a wider (multi-city or national) rollout. Not a plan to
> execute now — a checklist to revisit when Paula decides to expand scope
> beyond Quezon City. Add to this as new pilot-only shortcuts get made;
> don't let them go untracked the way the old synthetic-data assumptions
> did before the 2026-08-20 stack rebuild.
>
> Scope: the school search/browse subsystem specifically (frontend +
> `paaral-student-api`). Platform-wide items that already have their own
> tracker are pointed to at the bottom, not duplicated here.

---

## 1. Client-side filtering, not server-side

**Current (Chunk 16, step 7 decision):** the frontend fetches the full
school list once from `GET /api/v1/schools` with no query params, then
filters it in the browser via `useSchoolFilters` — instant, no network
call per keystroke or slider drag. Chosen over having every filter
change hit the backend, because at 606 rows (~470KB) shipping the whole
list is free and the UX is identical to today's static-JSON version.

**What changes for wider rollout:** the backend endpoint already accepts
every filter as a query param (built in step 6) specifically so this
migration doesn't require backend work later — only the frontend side
needs to change. `useSchoolFilters` goes from a synchronous `useMemo`
filter to something that fetches on filter change; free-text search
needs debouncing (wait for a pause in typing before firing a request,
not one request per letter); the UI needs loading states for the gap
between "filter changed" and "results arrived."

**When to revisit:** no hard row-count threshold is set yet — worth
picking one when a second city is actually added, rather than waiting
until the page visibly feels slow.

---

## 2. Geographic scope hardcoded to Quezon City

**Current:** `paaral-student-api/app/db/sync_schools.py`'s BigQuery query
has `WHERE deped_municipality = 'QUEZON CITY' AND deped_region = 'NCR'`.
`FilterSidebar.tsx`'s municipality control is a disabled dropdown fixed
to "Quezon City" (already commented in code as a future-scale
placeholder, not an oversight).

**What changes:** parameterize or drop the sync script's `WHERE` clause;
turn the municipality/region/province/barangay filters into real,
cascading selects (region → province → municipality → barangay), similar
to the Chile SAE benchmark's admission portal that this app's browse UI
is structurally modeled on; decide how often the sync re-runs at the
new, larger scope (currently a manually-run one-off script, not
scheduled).

---

## 3. No pagination on `GET /api/v1/schools`

**Current:** the endpoint returns the entire filtered result set in one
response — reasonable at 606 rows, currently unbounded.

**What changes:** add limit/offset or cursor-based pagination once
result sets can plausibly get large even after filtering; the frontend's
list/card views would need paging or infinite scroll instead of
rendering every result at once.

---

## 4. No database indexes beyond the primary key

**Current:** the `school` table has no index besides `school_id` (the
PK) — confirmed while building step 6. A `municipality`/`barangay`
filter or the `ILIKE` name search is a full table scan today; invisible
at 606 rows, not at national scale.

**What changes:** add indexes matching the real filter/search
predicates once row count and query volume justify it — likely a btree
index on `municipality`, `barangay`, `school_type`, and
`is_esc_participating`, plus a trigram (`pg_trgm`) index to keep the
`ILIKE` name search fast.

---

## 5. Static detail pages, pre-rendered at build time

**Current:** `/schools/[school_id]` uses `generateStaticParams()` to
pre-render every school's page at build time (currently reading
`qc-schools.json`; about to switch to a build-time API call as part of
step 7). This is only practical because there are 606 schools.

**What changes:** pre-rendering tens of thousands of pages at build time
would make builds slow and put the live BigQuery/Postgres data behind a
build-and-redeploy cycle for freshness. The already-considered fallback,
from the 2026-08-20 stack-rebuild decision, is switching from
`output: 'export'` to ISR (Incremental Static Regeneration) — documented
there as a config change plus a deploy-target switch (Firebase Hosting →
Firebase App Hosting/Cloud Run), not a rewrite.

---

## Related, tracked elsewhere — not duplicated here

- **Cloud SQL production tier sizing** (currently `db-f1-micro`, a dev
  tier) — `WORKFLOW.md`, Chunk 13.
- **CD/deploy pipeline** (no Workload Identity Federation, no chosen
  compute target yet) — `WORKFLOW.md`, Chunk 13.
- **Distance/commute-to-school fields** (must be computed per learner at
  request time, not stored on `school` — depends on real learner
  addresses from DepEd LIS) — `.claude/rules/memory-decisions.md`,
  2026-08-19 entry.
- **BigQuery dataset completeness gaps** (no grade level, shift,
  religious affiliation, photos, contact info, quality indicators; slot
  data still sparse) — `.claude/rules/memory-decisions.md`, 2026-08-20
  entry.
