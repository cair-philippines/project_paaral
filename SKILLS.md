# PAARAL Student View — Technical Skills & Standards

## Technical Stack

### Current Phase: Mockup

| Tool | Version | Purpose |
|---|---|---|
| React | 19.x | UI framework |
| Vite | 8.x | Build tool and dev server |
| Tailwind CSS | 4.x | Utility-first styling |
| Lucide React | 1.x | Icon set |
| Vercel | — | Deployment |

**No router.** Single-page mockup — all state lives in `PAARALStudentMockup()`.

**No state library.** `useState` and `useMemo` are sufficient for mockup scope.

**No backend.** All data is static and inline. Submissions are local state only.

### Future Phase: Production

| Tool | Purpose |
|---|---|
| DepEd LIS API | Real LRN verification |
| BEIS / school registry | Live school data |
| Backend API | Application submission, draft persistence |
| Auth layer | LRN-based session management |
| Database | Survey responses, submission records |

---

## File Structure

```
src/
  App.jsx       ← Entire app: data, helpers, components, layout
  index.css     ← Tailwind directives + custom scrollbar
  main.jsx      ← React root mount

public/
  assets/       ← ecair-logo.png, deped-logo.png

.claude/
  memory/       ← Persistent memory for Claude Code sessions
```

**Single-file architecture** is intentional for mockup phase. When moving to production, break into:
- `src/components/portal/` — tab panels (Verify, Eligibility, Choices, Survey)
- `src/components/map/` — PhilippinesMap, SchoolInfoCard
- `src/components/filters/` — left sidebar filter sections
- `src/data/` — schools, typeMeta, constants
- `src/hooks/` — useLRN, useEligibility, useSurvey

---

## Coding Standards

### Naming Conventions

**School types** (values used in `school.type`):
```
public          Public JHS
private_esc     Private JHS with ESC subsidy
private_no_esc  Private JHS without ESC
```

**Portal tab IDs:**
```
verify       LRN verification
eligibility  ESC eligibility questionnaire
choices      School wishlist
survey       Pilot feedback + submit
```

**Eligibility step IDs:**
```
schoolType   How did you complete Grade 6?
seg          Social Equity Group membership (public school path only)
income       Monthly household income
employment   Parent/guardian employment status
result       Category determination + document list
```

**Eligibility answer values:**
```
schoolType:  'public' | 'private' | 'als'
segs:        ['4ps', 'gidca', 'ip', 'pwd', 'special', 'cbms', 'none']
income:      'poor' | 'low' | 'lower_middle' | 'middle' | 'above'
employment:  'local' | 'abroad' | 'business' | 'unemployed'
```

**ESC categories:** `'A' | 'B' | 'C' | 'D' | null` (null = ineligible or incomplete)

### School Data Schema

Each school object in the `schools` array:

```js
{
  id:                   "SCH001",
  name:                 "St. Mary's Academy of Taguig",
  type:                 "private_esc",       // see school types above
  sector:               "sectarian",         // 'sectarian' | 'non_sectarian' | null
  region:               "NCR",
  province:             "Metro Manila",
  municipality:         "Taguig City",
  barangay:             "Bagumbayan",
  postal_code:          "1630",
  lat:                  14.5176,
  lng:                  121.0509,
  tuition:              45000,               // annual, PHP
  esc_subsidy:          13000,               // 0 if not ESC partner
  net_cost:             32000,               // tuition - esc_subsidy
  slots_total:          40,
  slots_available:      12,
  distance_km:          3.2,
  commute_minutes:      15,
  esc_rating:           4,                   // 1–5, 0 if N/A
  religious_affiliation: "Sectarian",
  admission_category:   "ESC Partner",
}
```

### ESC Subsidy Tiers

Per E-GASTPE 2026 and the `city_type` of the school:

| Area | Subsidy |
|---|---|
| NCR | ₱13,000 |
| Region IV-A HUC (Lucena City) | ₱11,000 |
| Region IV-A other | ₱9,000 |

### Income Classification (PIDS)

Source: PIDS table provided 2026-05-04.

| Label | Monthly Income |
|---|---|
| Poor | < ₱10,957 |
| Low income (but not poor) | ₱9,520 – ₱21,194 |
| Lower middle class | ₱21,194 – ₱43,828 |
| Middle class | ₱43,828 – ₱76,669 |
| Upper middle income | ₱76,669 – ₱131,484 |
| High income (but not rich) | ₱131,484 – ₱219,140 |
| Rich | ≥ ₱219,140 |

ESC eligibility (Categories B/C/D) covers Poor through Middle class. Upper middle income and above → not eligible. Category A SEG criteria override income.

### Key Helper Functions

**`getDocList(category, answers)`** — module-level pure function. Takes ESC category ('A'–'D') and eligAnswers object, returns ordered array of document strings to prepare.

**`pesos(value)`** — formats PHP currency. Returns `"Free"` for 0.

**`slotTone(school)`** — returns Tailwind bg color class based on slot availability ratio.

---

## Python Standards

All Python files in this project must pass `ruff check` with zero errors.

**Linter:** `ruff` — configured in `pyproject.toml` (line-length = 79).
**Run:** `ruff check docs/` and `ruff format docs/` before considering Python work done.

**Rules enforced:** E/W (PEP8), F (pyflakes), D (pydocstyle, NumPy convention), I (isort).

### Docstring format (NumPy style)

Every public function and module requires a docstring:

```python
def example(arg1, arg2):
    """Return a one-line summary ending with a period.

    Parameters
    ----------
    arg1 : dict
        Description of arg1.
    arg2 : str
        Description of arg2.

    Returns
    -------
    str
        Description of the return value.
    """
```

**Module docstrings:** Summary on the first line, blank line before body.

```python
"""Do the thing this module does.

Usage: python3 docs/example.py
"""
```

### Other rules
- One import per line — never `import a, b`
- Imports sorted: stdlib → third-party → local (isort enforces this)
- Maximum line length: 79 characters

---

## Design System

**Primary color:** `#1a4b8c` (DepEd blue)
**Background:** `#f8f9fa` (left sidebar), `#ffffff` (right sidebar), `#e9f0f5` (map)
**Border:** `#e2e4e9`

**Font:** SF Pro Text / -apple-system stack (body); SF Pro Display (headings)

**Label pattern:** `text-[10px] font-bold uppercase tracking-widest text-slate-500` — used for all section labels and tab identifiers throughout the portal.

**Consistent component patterns:**
- Option cards: `p-3 rounded-xl border border-slate-200 hover:border-[#1a4b8c] hover:bg-blue-50 transition group`
- Confirmation banners (green): `text-xs text-green-700 bg-green-50 p-2 rounded border border-green-200`
- Warning banners (amber): `bg-amber-50 text-amber-800 border border-amber-200`
- Error banners (red): `text-xs text-red-700 bg-red-50 p-2 rounded border border-red-200`

---

## Map

**Component:** `PhilippinesMap` — SVG-based, Carto Light styled.
**Bounds:** lng 120.52–121.25, lat 14.15–15.22
**Canvas:** 750 × 1000px SVG
**Palette:** Land `#f0f3f4`, Water `#d1dce5`

School dots are colored by `typeMeta[school.type].dot`. Selected school is highlighted. User GPS location is simulated to NCR coordinates for demo bounds.
