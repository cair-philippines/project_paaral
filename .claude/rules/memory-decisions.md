# Past Decisions

## Mockup Architecture (2026-05-04)
- **What's mocked:** LRN verification (hardcoded set), school data (50 synthetic schools), draft choices (hardcoded per LRN), submission (local state only), survey responses (not persisted)
- **LRN check:** `VALID_LRNS = new Set(["100000000001", "100000000002"])` — deliberate, not a prefix trick. Represents registry identity check only, not an ESC eligibility gate.
- **Single-page app:** No router. All state in `PAARALStudentMockup()`.

## App Layout (SUPERSEDED 2026-06-10 — see Browse UX below)
- ~~Old: 3-panel sidebar layout (filters + map + right portal panel)~~
- Now: `appView` state drives hero vs browse; no persistent sidebar panels

## Browse UX — Hero + Map + Profile + Drawer (2026-06-10, SUPERSEDED by v3 2026-06-26)
- See v3 architecture below for current state.

## v3 Architecture — ESC Application State Machine (2026-06-26)

**Auth:** DepEd email (`100000000001@deped.gov.ph`) as ICTS SSO stand-in. LRN no longer used directly for login.

**Single test account:** One account progresses through all 9 states. State is persisted in `localStorage` under key `paaral_v3_account`. Survives logout/login.

**`appView`:** `'hero' | 'eligibility' | 'browse'`
- `eligibility` is a new full-screen view (replaces the modal questionnaire)
- On login, account created with `applicationState: 'eligibility'` → `appView` set to `'eligibility'`
- On return login, restores to `browse` (skips eligibility if already past it)

**9 application states (`applicationState`):**
```
eligibility → browsing → submitted → pre_approved → docs_pending → docs_submitted → granted
                                   → rejected → non_esc | browsing (reset)
```

**Drawer tabs:**
- Pre-submission (browsing): Choices | Documents | Survey
- Post-submission: Status | Documents | Choices (read-only)

**`isPostSubmission`:** `!['eligibility', 'browsing'].includes(applicationState)`

**Demo advance controls:** Embedded in Status tab, per-state. Reject path has two real action buttons (Continue Non-ESC / Browse Again). Granted state shows ICTS external link.

**`canSubmit` gates:** wishlist ≥ 1 + hasPublicAlternative + docsReady + surveyComplete.

**PAARAL vs ICTS scope:** PAARAL owns ESC eligibility docs and ESC lifecycle. ICTS owns school admission and enrollment. External link to ICTS on `granted` state.

**Module-level constants:** `STORAGE_KEY`, `TEST_EMAIL`, `TEST_LRN`, `LEARNER_RECORD`, `catMeta`

## Tab Structure (SUPERSEDED 2026-06-09 — see Portal UX below)
- ~~Four tabs: Verify → Eligibility → Choices → Survey~~
- Now: 2 tabs in authenticated state only (My Choices + Survey)

## Portal UX — Modal Account Creation (2026-06-09)
- Students browse schools freely without an account (map, filters, school details always accessible)
- Account creation triggered by "Create Account to Apply" CTA or "Add to My Choices" when guest
- Modal: Step 1 LRN verify → optional draft detection → Step 2 ESC eligibility → result → "Create My Account"
- `account` state (null | { lrn, category, answers }) is the auth gate — replaces `lrnVerified === 'valid'`
- `pendingSchool` queues a school to auto-add to wishlist after account creation
- Right panel guest state: selected school mini card + CTAs, or welcome CTA if nothing selected
- Right panel authenticated state: LRN badge + category, 2 tabs (My Choices + Survey)
- Survey tab still gates submission with 3 required questions
- Modal chosen over separate page or right-panel inline after benchmarking (Airbnb, Zillow, LinkedIn as modal examples)

## Draft Choices (2026-05-04)
- LRN `100000000002` returns 3 pre-saved choices: Bagumbayan NHS, St. Mary's Academy of Taguig, Pasig Grace Christian School
- Draft state is editable after loading (not locked)
- "Load Draft" populates wishlist AND navigates to My Choices tab

## Income Brackets (2026-05-04)
Source: PIDS table provided by Paula. Use these exact values — do not substitute.

| Tier | Monthly Income |
|------|----------------|
| Poor | < ₱10,957 |
| Low income | ₱10,957 – ₱21,194 |
| Lower middle class | ₱21,194 – ₱43,828 |
| Middle class | ₱43,828 – ₱76,669 |
| Upper middle income+ | > ₱76,669 → ineligible for ESC B/C/D |

Category A (SEG) eligibility is independent of income.

## ESC Category Logic (2026-05-04)
| Path | Category |
|------|----------|
| public + any SEG | A |
| public + no SEG + eligible income | B |
| als + eligible income | C |
| private + eligible income | D |
| any + above income (no SEG) | null (ineligible) |

## Production ERD (2026-05-26)
Seven tables finalized:
- `LEARNER` (PK: `lrn` string)
- `APPLICATION` (PK+FK: `lrn` → LEARNER — one-to-one, identifying relationship)
- `WISHLIST` (composite PK: `application_id` + `school_id`; `rank` column for preference order; no surrogate key)
- `ESC_ELIGIBILITY_ASSESSMENT` (PK+FK: `lrn` → LEARNER — optional, ESC applicants only; zero-or-one)
- `ASSESSMENT_SEG` (composite PK: `lrn` + `seg` — junction table for multi-select SEGs; `none` excluded, absence of rows implies none)
- `SURVEY_RESPONSE` (PK+FK: `lrn` → APPLICATION — one-to-one; no surrogate key)
- `DOCUMENT_UPLOAD` (surrogate PK: uuid — only table with surrogate, true one-to-many; FK: `lrn` → APPLICATION)

`seg` enum: `fourps | gidca | ip | pwd | special | cbms`
Coord types: `DECIMAL(9,6)` lat, `DECIMAL(10,7)` lon. `distance_km`: `FLOAT`.
