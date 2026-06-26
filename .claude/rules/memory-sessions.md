# Session Summaries

## 2026-05-04 — Chunks 1–6 (Full Mockup Build)

Built the complete student-facing portal mockup:
1. **Scaffolding** — Vite + React 19, Tailwind CSS v4, Lucide React. 50 synthetic schools, SVG Carto Light map (NCR + Region III + IV-A bounds).
2. **LRN Verification** — Hardcoded valid LRN set, 800ms simulated delay, success/error states, draft detection.
3. **Tabbed Portal** — Four tabs (Verify, Eligibility, Choices, Survey). Submit in Survey tab only.
4. **ESC Eligibility Questionnaire** — 4-step branching flow (schoolType → seg → income → employment → result). Category A–D determination. `getDocList()` pure function for document checklists.
5. **Pilot Survey** — 3 required questions gate the Submit button. Pre-submit checklist shows gate status.
6. **UI Refinements** — Logo layout (bare, above title), BETA badge inline, collapsible right panel (40px strip ↔ 340px).

Core docs created: CLAUDE.md, SKILLS.md, WORKFLOW.md, LOG.md

## 2026-05-26 — ERD Design

Designed production database schema (7 tables). Key decisions:
- `APPLICATION.lrn` as PK+FK (no surrogate — one-to-one with LEARNER)
- `WISHLIST` composite PK enforces no duplicate schools per application
- `ASSESSMENT_SEG` junction table replaces JSON array for SEG multi-select
- `DOCUMENT_UPLOAD` is the only table with a surrogate UUID PK (true one-to-many)
- Settled coordinate types: `DECIMAL(9,6)`/`DECIMAL(10,7)`, `FLOAT` for distance

## 2026-06-09 — CLAUDE.md Restructure + Chunk 7

Recalibrated project memory system:
- CLAUDE.md → slim routing file (under 150 lines)
- Knowledge split into `.claude/rules/memory-*.md` (profile, preferences, decisions, sessions)
- Added Stop hook in `.claude/settings.json` to catch session learnings

Implemented Chunk 7 — modal-based account creation:
- LRN verification + ESC eligibility questionnaire moved from right panel tabs into a modal overlay
- Students browse freely as guests; account creation triggered by CTA or "Add to My Choices"
- `account` state (null | { lrn, category, answers }) is the new auth gate for canSubmit
- Right panel: guest state (school mini card + CTA) vs authenticated state (2-tab: Choices + Survey)
- `pendingSchool` auto-added to wishlist on account creation
- Pending: Chunks 8–12 (School View, DepEd View, Vercel, a11y, mobile)

## 2026-06-10 — Full layout redesign (hero + browse + drawer)

Replaced the old 3-panel sidebar layout with a new multi-screen flow:
- **Hero view** — full-screen gradient, Baskervville (Fry's Baskerville) title "PROJECT PAARAL", 3 CTAs
- **Browse view** — gradient navbar + Map/School Profile tabs
  - Map tab: collapsible filter sidebar (38px strip), full PhilippinesMap, collapsible floating results panel (38px strip). School cards expand in-place when selected (Airbnb style) with stats + "See full school details →" + "Add to My Choices"
  - School Profile tab: sticky sub-nav with scroll-to-section (direct `scrollTo` on container, not `scrollIntoView`). Active section tracked via scroll event listener on container (replaced IntersectionObserver). Three sections: Overview (gallery placeholder + about), School Characteristics (grid), Fee Information (boxes + ESC note)
  - Application Drawer: slides from right, toggled by "My Account" button in navbar (authenticated only). Has Choices / Documents / Survey tabs with all existing gate logic
- Auth modal: preserved exactly
- `appView` state: `'hero' | 'browse'`; `browseTab` state: `'map' | 'profile'`
- Navbar: authenticated = "My Account" button only (no LRN display, no badge count)
- Map click → results panel auto-opens + selected card scrolls into view (`selectedCardRef` + `resultsScrollRef`)
- Font: Baskervville loaded via Google Fonts (Fry's Baskerville revival, weight 400)

## 2026-06-26 (continued) — Architecture Candidates 3 & 4

**Candidate 3 — `nextEligStep` pure function**
Added `nextEligStep(step, answers)` alongside `eligGo`. It owns the full eligibility transition graph:
- `schoolType → seg` (public) or `income` (private/als)
- `seg → result` (any non-none SEG) or `income`
- `income → result` (above) or `employment`
- `employment → result` always

Removed `next` hardcoded on each option object in `schoolType` and `income` steps. SEG Continue button no longer has inline conditional. All four step handlers now call `eligGo(nextEligStep(...), patch)`.

**Candidate 4 — `WishlistButton` component**
Extracted single `WishlistButton({ school, isInList, onAdd, variant = 'full' })` above `ResultCard`. Handles `isFull` logic internally. Two variants:
- `compact` — icon-only pill (used in `ResultCard`)
- `full` (default) — full-width button with text (used in `SchoolInfoCard` and profile view fee section)

Removed `isFull` local variable from both `ResultCard` and `SchoolInfoCard`. Build verified clean.

## 2026-06-26 — v3 Full Rewrite (ESC State Machine)

Complete rewrite of `src/App.jsx` (1778 lines). Key changes from v2:

**Auth:** Switched from LRN entry to DepEd email (`100000000001@deped.gov.ph`) as ICTS SSO stand-in. Login modal shows learner info card (from `LEARNER_RECORD`) before account creation.

**Eligibility:** Moved from modal to full-screen `appView: 'eligibility'`. Same branching questionnaire (schoolType → seg → income → employment → result). On complete, sets `applicationState: 'browsing'` and navigates to browse.

**State machine:** 9 states stored in localStorage — `eligibility`, `browsing`, `submitted`, `pre_approved`, `rejected`, `docs_pending`, `docs_submitted`, `granted`, `non_esc`. Single test account persists state across logout/login.

**Drawer restructure:**
- Pre-submission: Choices | Documents | Survey tabs
- Post-submission: Status | Documents | Choices (read-only) tabs
- Status tab has per-state UI + demo advance controls (dashed-border "Demo Controls" box)
- Rejection path: two buttons — "Continue Non-ESC" → `non_esc`, "Browse Again" → `browsing` (resets wishlist)
- Granted state: ICTS portal external link

**Browse nav:** Map | School Profile (main tabs). My Choices moved into drawer. "My Account" button opens drawer.

**Compile verified:** Vite transpiles cleanly, dev server responds at localhost:5173.

## 2026-06-11 — v2 UI polish (student view complete)

Final polish pass on the browse view:
- **My Choices tab** added to main browse tab bar (alongside Map and School Profile). Guest state shows account CTA; authenticated state shows draggable wishlist, public JHS warning, pre-submit checklist, and Submit button. Tab label shows count when schools are added.
- **School Options legend** moved from `bottom-5 right-5` to `bottom-5 left-5` on the map — was occluded by the floating results panel (z-20 vs z-10).
- **Location filters** changed from 2-column grid to single-column flex (one row per: Region, Province, Municipality, Barangay).
- Reverted a stats-row simplification (user preferred the 3 mini-cards over inline text).

**Status: Student view v2 is done.** Pending work is Chunks 8–12 (School View, DepEd View, Vercel deploy, a11y, mobile).

## 2026-06-09 (continued) — Modal refinements + Landing page wireframe

Additional work in same session:
- Added `lrnConfirmed` state to modal — LRN verification now shows an intermediate "LIS learner info" card (name, current school, grade, municipality, division) before advancing to ESC eligibility; `LEARNER_RECORDS` mock data added for LRNs 100000000001 and 100000000002
- Added Documents tab (3rd tab) to authenticated right panel: shows ESC doc checklist with upload controls, "Simulate all uploads (demo)" helper, gates canSubmit via `docsReady` computed from `uploadedDocs` Set vs `requiredDocs` array
- Restyled auth modal to match wireframe palette: gradient backdrop `#1a0e3e → #2b1260 → #661843 → #7c1c30`, 22px radius card, pill inputs/buttons, 5-dot progress bar driven by `eligHistory.length`, catMeta colors updated to wireframe purple/blue/teal/amber per category
- Created `docs/paaral-landing.html` — 8-screen wireframe for the landing page, patterned after Redfin (PDF: `docs/student-view-landing.pdf`). Screens: Hero (default), Hero "Browse" hover, Browse split view (search panel + map + amber callout), School Detail × 3 tabs (Overview with gallery / School Details / Fee Information + Facilities), Hero "Submit" hover, Create Account modal
