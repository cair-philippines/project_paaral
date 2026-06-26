# PAARAL Student View — Build Log

> **Purpose:** Chronological record of everything built, decided, or changed.
> Future sessions: read this + CLAUDE.md + SKILLS.md + WORKFLOW.md to get full context.

---

## 2026-05-04

### Session Start

**Context established:**
- Project is the student-facing view of PAARAL — the enrollment portal for Grade 6 → Grade 7 transition
- Companion to `../deped-planning-view` (DepEd prescriptive analytics tool)
- Initial commit already had: 50 synthetic schools, SVG map (Carto Light), left sidebar filters, basic right sidebar with wishlist and submit button

---

### LRN Verification — Refinement

**Original behavior:** `lrn.startsWith("1") ? "valid" : "invalid"` — too crude for demos.

**Decision:** LRN verification = pure registry identity check. Confirmed LRN is in the DepEd Learner Information System (LIS). Not an ESC eligibility gate.

**Changes made:**
- Replaced prefix trick with hardcoded `VALID_LRNS = new Set(["100000000001", "100000000002"])`
- Success message updated to: "LRN confirmed in the DepEd Learner Information System (LIS)"
- Demo LRNs: valid fresh = `100000000001`, valid with draft = `100000000002`, invalid = `200000000001`

---

### Draft Choices for Returning Users

**Feature:** LRN `100000000002` returns with 3 pre-saved school choices. Student sees a summary card "You already have choices on record" with Load Draft / Start fresh actions.

**Draft schools for `100000000002`:**
1. Bagumbayan National High School (public)
2. St. Mary's Academy of Taguig (private ESC)
3. Pasig Grace Christian School (private ESC)

**Decision made:** Draft state = editable (not locked). Student can modify after loading.

**"Load Draft" behavior:** Populates wishlist AND navigates to My Choices tab.

---

### UX Architecture Discussion — Portal Organization

**Problem:** Portal was getting crowded with LRN, wishlist, and upcoming eligibility + survey features.

**Options considered:**
| Option | Description |
|---|---|
| A | Hard gate — verify LRN before accessing system |
| B | Current inline approach — verify before submit |
| C | Soft gate — verify when adding first school to wishlist |

**Decision: Option B.** Keep verification inline. Option C is a future extension.

**Survey placement decision:**
- Rejected: survey after submission gated behind "confirmation reference number" — students don't need that number for anything
- Adopted: survey as a step in the submission flow. Required questions unlock the Submit button. Transparent framing: "help us improve PAARAL before you submit."

**Final tab structure:**
1. Verify — LRN check
2. Eligibility — ESC category questionnaire
3. My Choices — wishlist
4. Survey — pilot feedback + Submit

---

### Tabbed Portal Structure

**Implemented:**
- `portalTab` state: `'verify' | 'eligibility' | 'choices' | 'survey'`
- Tab bar in portal header (underline indicator, DepEd blue active state)
- Submit button moved exclusively to Survey tab
- `canSubmit` updated: requires wishlist + public JHS + LRN verified + survey complete

---

### ESC Eligibility Questionnaire

**Policy source:** E-GASTPE 2026 Revised Guidelines, Article III (Sections 7–13), read directly from PDF.

**Implemented:** Branching 4-step questionnaire (schoolType → seg → income → employment → result).

**Categories per Article III Section 8 (Table 1):**
| Category | Criteria |
|---|---|
| A | Grade 6 grad, Social Equity Group (4Ps, GIDCA, IP, PWD, special needs, poor/near-poor per CBMS) — highest priority |
| B | Grade 6 grad from public school, poor to middle class income |
| C | ALS A&E Test or PEPT passer, poor to middle class income |
| D | Grade 6 grad from private school, poor to middle class income |

**Income bracket source issue:** Initial implementation used figures from training data (attributed to PIDS but unverified). User challenged the source.

**Resolution:** User provided authoritative PIDS income classification table (2026-05-04). Updated income step to use exact PIDS labels and thresholds:
- Poor: < ₱10,957
- Low income: ₱10,957 – ₱21,194
- Lower middle class: ₱21,194 – ₱43,828
- Middle class: ₱43,828 – ₱76,669
- Above (ineligible for B/C/D): > ₱76,669

**Navigation:** Back button at every step. "Start over" resets questionnaire. No enforced sequence between portal tabs.

**Result step:** Shows category badge (color-coded A–D), tailored document checklist, and a note that the ESC School Committee makes the final determination.

**`getDocList(category, answers)`** — module-level pure function. Called in result step render.

---

### Pilot Survey

**3 required questions:**
1. Ease of finding schools (1–5 scale)
2. Did information help decision? (Yes / Somewhat / No)
3. Biggest concern about private school enrollment (Cost / Distance / Quality / Slot availability)

**1 optional question:** Open text suggestions.

**`surveyComplete`** — `Boolean(ease && helpful && concern)` — drives submit gate.

**Pre-submit checklist** shown above Submit button: visual indicator of which conditions are green.

---

### UI Refinements

**Left sidebar header:**
- Removed "Executive Demo" badge
- Logos (ECAIR + DepEd) moved above PAARAL title, displayed bare (no rounded rectangle, border, or shadow)
- ECAIR logo: `h-5`; DepEd logo: `h-8`
- "BETA" label added inline right of "PAARAL", `items-center` aligned (vertically centered to title), DepEd blue pill style

**Right panel collapse:**
- `portalCollapsed` state added
- Collapsed width: `w-10` (40px strip)
- Expanded width: `w-[340px] xl:w-[380px]`
- Toggle button floats on left edge (`-left-3.5`, `top-6`), uses `ChevronLeft` / `ChevronRight`
- Collapsed state shows vertical "Student Portal" label (`writing-mode: vertical-rl`)
- Smooth transition: `transition-all duration-300`

---

### Core Docs Created

| File | Purpose |
|---|---|
| `CLAUDE.md` | Project context, what's real vs mocked, policy references, demo credentials |
| `SKILLS.md` | Tech stack, file structure, naming conventions, data schema, design system |
| `WORKFLOW.md` | Feature chunks with decisions and status |
| `LOG.md` | This file — chronological record |

---

## 2026-05-26

### ERD Design Session

Designed the entity-relationship diagram for the student view production database.

---

### Learner-to-Application Relationship

**Decision:** One-to-one, scoped to the G6→G7 transition only. A learner makes this transition exactly once — no enrollment cycle dimension needed.

**Implementation:** `APPLICATION.lrn` serves as both PK and FK pointing to `LEARNER.lrn`. No surrogate key on `APPLICATION`.

---

### Key Concepts Established

- **Surrogate key** — artificial identifier with no business meaning (e.g. UUID, auto-increment). Used when the natural key is composite, unstable, or needs to be opaque.
- **Natural key** — identifier with real-world meaning (e.g. `lrn`). Preferred when stable and unique.
- **Enum** — column restricted to a fixed set of values. Used for `school_type`, `income_bracket`, `category`, `seg`, etc.
- **Identifying relationship (UID bar)** — FK is also part of the child's PK. Child cannot exist or be identified without its parent.
- **Non-identifying relationship** — child has its own independent PK; FK is just a reference.

---

### Table Decisions

**LEARNER**
- PK: `lrn` (string)
- FK: `grade6_school_id` → SCHOOL

**APPLICATION**
- PK, FK: `lrn` (string) → LEARNER.lrn
- No surrogate key — one-to-one with LEARNER

**WISHLIST**
- Composite PK: `(application_id, school_id)` — no surrogate key
- `application_id` FK → APPLICATION.lrn (names differ, same value; declared explicitly in SQL)
- `school_id` FK → SCHOOL.id
- UID bars on both sides (identifying relationship)
- Added `rank` column for order of preference
- Composite PK enforces business rule: same school cannot appear twice in one application

**ESC_ELIGIBILITY_ASSESSMENT**
- PK, FK: `lrn` → LEARNER.lrn
- Optional relationship — only learners intending private school take this (zero or one per learner)
- Columns: `esc_intent` (boolean), `school_type` (enum), `income_bracket` (enum), `employment_status` (enum), `category` (enum: A/B/C/D), `assessed_at` (timestamp)
- `segs` column removed — replaced by `ASSESSMENT_SEG` junction table

**ASSESSMENT_SEG**
- Composite PK: `(lrn, seg)`
- FK: `lrn` → ESC_ELIGIBILITY_ASSESSMENT.lrn
- `seg` enum: `fourps | gidca | ip | pwd | special | cbms` — `none` excluded (absence of rows implies none)
- Junction table replaces JSON array for multi-select SEG memberships
- Composite PK enforces: same learner cannot have the same SEG listed twice

**SURVEY_RESPONSE**
- PK, FK: `lrn` → APPLICATION.lrn — no surrogate key (one-to-one with APPLICATION)
- Columns: `ease` (integer 1–5), `helpful` (enum), `concern` (enum), `suggestions` (text, nullable), `submitted_at` (timestamp)

**DOCUMENT_UPLOAD**
- Surrogate PK: `id` (uuid) — needed because one application has many uploads (one-to-many)
- FK: `lrn` → APPLICATION.lrn (plain FK, no UID bar)
- Columns: `document_type` (enum), `file_url` (string), `uploaded_at` (timestamp)

---

### Data Type Decisions

| Column | Type | Reason |
|---|---|---|
| `latitude` | `DECIMAL(9,6)` | exact storage, ~0.1m precision |
| `longitude` | `DECIMAL(10,7)` | one extra digit for 180° range |
| `distance_km` | `FLOAT` | computed value, approximate is fine |

---

### Relationship Summary

```
LEARNER ──|────|── APPLICATION ──|────< DOCUMENT_UPLOAD
                       |
                       |────|< WISHLIST >|────── SCHOOL
                       |
                       ○|────|── SURVEY_RESPONSE
                       |
LEARNER ──○|────|── ESC_ELIGIBILITY_ASSESSMENT ──|────|< ASSESSMENT_SEG
```

- `LEARNER → APPLICATION`: one-to-one (mandatory both sides)
- `APPLICATION → WISHLIST`: one-to-many, identifying (UID bar)
- `WISHLIST → SCHOOL`: many-to-one
- `APPLICATION → SURVEY_RESPONSE`: one-to-zero-or-one
- `APPLICATION → DOCUMENT_UPLOAD`: one-to-many, non-identifying
- `LEARNER → ESC_ELIGIBILITY_ASSESSMENT`: one-to-zero-or-one (optional — ESC applicants only)
- `ESC_ELIGIBILITY_ASSESSMENT → ASSESSMENT_SEG`: one-to-many, identifying (UID bar)

---

## 2026-06-09

### Student View Restructure — Modal Account Creation (Chunk 7)

**Previous structure:** 4-tab right panel (Verify | Eligibility | My Choices | Survey). LRN verification and ESC eligibility lived in tabs.

**New structure:** Students browse freely as guests. Account creation (LRN + ESC eligibility) is a modal overlay triggered by "Create Account to Apply" or "Add to My Choices".

**State changes:**
- Added `account` (null | { lrn, category, answers }) — replaces `lrnVerified === 'valid'` as the auth gate
- Added `showAuthModal` (boolean), `pendingSchool` (null | school)
- `portalTab` default changed from `'verify'` to `'choices'`
- `canSubmit` gates on `account !== null`

**Modal flow:**
- Step 1 of 2: LRN entry + verification (800ms delay preserved)
- Draft detection: LRN `100000000002` shows draft summary; Load Draft / Start fresh
- Step 2 of 2: Full ESC eligibility questionnaire (same branching logic, same steps)
- Result: category badge + doc list + "Create My Account" CTA
- Backdrop click or X closes without creating account; state preserved across open/close

**Right panel — guest state:** selected school mini card + CTAs, or welcome message if nothing selected.
**Right panel — authenticated:** 2 tabs only (My Choices + Survey); LRN badge + ESC category in header.

**`addToWishlist`:** When `account === null`, saves `pendingSchool` and opens modal instead of adding directly. `pendingSchool` auto-added to wishlist on account creation.
