# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary user: a **parent or guardian** managing their Grade 6 child's ESC (Educational Service Contracting) subsidy application and Grade 7 school choice on the child's behalf, using the child's LRN (Learner Reference Number). This is a broad, mandatory-government-service audience — not opt-in, not assumed tech-savvy, often a first-time user of a formal digital enrollment process, frequently on modest devices or intermittent connections. The product should still work for an older learner operating the account directly, but tone, reading level, and copy are written for an adult managing it on a child's behalf, not a peer-to-peer student app.

## Product Purpose

PAARAL Student View is the public-facing enrollment portal for the Philippine Department of Education's Educational Service Contracting (ESC) program, helping families navigate a child's Grade 6 → Grade 7 transition. It lets a family verify a learner's identity against DepEd's Learner Information System (LIS), determine ESC subsidy eligibility (Category A–D), browse and rank ESC-participating public and private junior high schools with real fee/slot data, submit up to 3 simultaneous ESC applications, track status, upload required documents, and redeem an awarded subsidy at one school. Success means a family understands its ESC eligibility, finds real schools with real data, and completes a subsidy application without needing to understand DepEd's internal bureaucracy.

**Current status: a working pilot with a real backend, not a UI-only mockup.** Login, school search/filtering, wishlist management, the pilot survey, document uploads (to Cloud Storage), and eligibility persistence are all live against a real Postgres database (Cloud SQL) and real Quezon City school data synced from BigQuery. It is still pre-launch/internal (a shared dev environment, two seeded demo accounts, no production auth layer), and the real external integrations — DepEd LIS, BEIS, and ICTS SSO — are not yet wired up; those stay mocked/local until DepEd's own systems are ready to integrate against.

## Positioning

PAARAL is functionally an intermediary matching system — the team's own internal benchmark is Varbi (a recruitment/ATS platform), reframed as matching families to schools instead of candidates to jobs — not a system of record for either side. Its real point of view is a specific policy stance, not just a UX preference: PAARAL implements a **"Portable Eligibility" / decoupled model** (from the team's own E-GASTPE policy paper) where ESC subsidy eligibility and school admission are two independent, parallel tracks that only converge at redemption. A family can pursue a subsidy and shop for schools in either order, or simultaneously — unlike DepEd's actual current enrollment-first sequencing, which the team's own research argues traps families rejected late in the process with no slots left elsewhere. Concretely, this shows up as a capped slate of up to 3 simultaneous private-school ESC applications, explicit backfill on rejection, and an explicit "redeem" action — no neighboring tool applies decoupled, parallel-track ESC pursuit this way.

## Operating Context

- Government service, built under the ECAIR research umbrella for DepEd (Department of Education, Philippines). The DepEd and ECAIR logos are real, required branding, not placeholders.
- A family's meaningful use starts by verifying a learner's LRN against the DepEd Learner Information System — treated as a hard gate for account creation, though guest browsing works without one.
- School data (name, location, fees, ESC subsidy amount, slot availability) is real, synced periodically from a DepEd/ECAIR BigQuery dataset into the product's own Postgres database — currently scoped to Quezon City only as the pilot city, with a stated intent to widen to more regions later.
- Companion platforms exist or are planned under the same PAARAL umbrella: School View (the schools' side, not yet started) and DepEd View (`../deped-planning-view`, prescriptive analytics for DepEd staff — slot allocation, congestion simulation). Student View does not share a backend with either; each platform gets its own infrastructure.
- Users may be on modest phones and slower or intermittent connections — a stated, explicit design constraint.
- Real legal/regulatory grounding: RA No. 8545 (E-GASTPE Act) and the DepEd Order s. 2026 E-GASTPE Revised Guidelines (Article III, Sections 7–13) define the actual ESC eligibility categories, document requirements, and subsidy tiers this product implements — not invented policy.

## Capabilities and Constraints

**Confirmed working, verified live against real infrastructure (not just compiled):**
- LRN verification against a real (seeded, demo-scale) Postgres `Learner` table, standing in for the real DepEd LIS integration.
- Real Quezon City school search/browse (map, list, card views) with filters (name, school type, ESC participation, barangay, fee/subsidy/net-fee ranges) against 606 real synced schools.
- ESC eligibility questionnaire (Category A–D determination, branching on how the learner completed Grade 6, Social Equity Group membership, PIDS household income bracket, and employment status), persisted server-side.
- Ranked wishlist with drag-to-reorder (mouse/touch/keyboard); a capped-parallel ESC application model — up to 3 ranked private schools pursued simultaneously, explicit backfill on rejection, and an explicit "redeem" action that accepts one offer and withdraws the rest.
- A 2-section pilot survey (general usability + ESC-specific) gating final submission.
- Real document uploads (ID, income proof, etc.) to Cloud Storage — staged locally, uploaded only on an explicit "Submit Documents" action, and the final application submission is gated on backend-confirmed uploads, not a locally-picked file.
- Login state hydration: a returning family's saved wishlist, eligibility result, survey answers, and uploaded documents are restored on login rather than rebuilt blank.

**Known, explicit constraints (not gaps to silently design around):**
- Real LRN verification against the live DepEd LIS, real BEIS data, and real ICTS SSO are deliberately not yet integrated ("infra first, integrate later") — deferred until those DepEd systems are ready. The current LRN check is a seeded demo table, not the live registry.
- Pilot scope is Quezon City only (real data); the originally-planned wider NCR + Region III + Region IV-A footprint is a deliberate later scale-up, not yet built.
- Real, currently un-fillable school-dataset gaps: no grade-level/curriculum-offered field, no shift/jornada, no religious affiliation, no photos, no descriptions, no contact info, no quality/program indicators. The product must not fabricate placeholder content for these — it shows honest "not available" states instead.
- Distance/commute-to-school is **not** a static per-school field (a deliberate correction from an earlier mockup simplification) — it depends on a family's real address, unavailable until real LIS integration. Do not reintroduce a fixed per-school distance figure.
- No production authentication/session layer yet — the demo login is an LRN + a DepEd-email-shaped identifier standing in for real ICTS SSO.
- Redis/caching is explicitly declined for now (no concrete workload yet) — a deliberate choice, not a gap.
- Terminology: **ESC** = Educational Service Contracting; **LRN** = Learner Reference Number; **LIS** = Learner Information System; **SEG** = Social Equity Group (4Ps, GIDCA, IP, PWD, special needs, poor/near-poor per CBMS); **Categories A–D** are the four ESC eligibility tiers; **redemption** is the act of accepting one ESC offer and finalizing enrollment at that school.

## Brand Commitments

- Product name is **"PAARAL"** (a real word, not a placeholder — "for school" in Filipino). "Student View" is the actual, deliberate name for this platform within the broader PAARAL system, not a generic "frontend" label.
- Official partner logos required and already in place: DepEd logo and ECAIR logo (`public/assets/`), real assets, not placeholders.
- Approved final color palette (`src/app/globals.css`): `--text` #01010d / #e8e9fd (dark), `--background` #fcfcfd / #020203 (dark), `--primary` #19266b / #94a1e6 (dark), `--secondary` #fcca81 / #7c4c03 (dark, amber/gold), `--accent` #b23836 / #c9514f (dark), plus six defined gradient tokens. Dark-mode values are defined but not yet toggled on anywhere — light-only for now, a known/tracked deferral, not an oversight.
- Typography: Geist Sans / Geist Mono (Next.js's `next/font/google` defaults).
- Visual/IA benchmarks explicitly named by the user (not Claude's own judgment calls): Chile's Sistema de Admisión Escolar (structural/IA baseline for the functional search/browse/detail app), ISB.be (a later, landing/marketing-page-only motion layer — never on the functional app surface), Dallas ISD and International School of Brussels (general visual references for a future full visual-design pass).
- Core, load-bearing design commitment stated explicitly by the user, applying to every design decision: **PAARAL is a mandatory government service for a broad, often non-tech-savvy public, not an opt-in product for a tech-savvy audience.** When a choice trades polish for legibility/predictability, legibility wins, and the tradeoff gets flagged rather than silently decided.

## Evidence on Hand

- Real school dataset: 606 real Quezon City school records (name, location, type, ESC participation, fees, subsidy amount, net fees, slot totals/unutilized) synced from a real DepEd/ECAIR BigQuery table — not synthetic. The wider nationwide table (60,421 rows) has similar real fields but several (grade level, shift, photos, quality indicators) are genuinely absent — do not fabricate them.
- Real legal/policy source documents: RA 8545 (E-GASTPE Act) and DepEd's E-GASTPE 2026 Revised Guidelines PDF, read directly — not recalled from training data.
- Real, user-provided PIDS income classification table (exact monthly income bracket figures) — must never be substituted with generic/training-data figures.
- Two seeded demo accounts for internal testing (`100000000001` fresh, `100000000002` with a 3-school draft wishlist) plus one deliberately-invalid demo LRN (`200000000001`) for testing the not-found path.
- No real user research, testimonials, or usage data yet (pre-launch/internal pilot) — do not fabricate any.

## Product Principles

1. **Decoupled, parallel-track pursuit, not sequential gatekeeping.** ESC eligibility and school admission are independent; the product actively works against "apply to one school, wait, then try another" bottlenecks — this is the product's real policy point of view, not incidental UX.
2. **Government-mandated, not opt-in — legibility beats polish.** A broad, often non-tech-savvy public must use this without prior digital-service experience; plain-language glosses for domain jargon, conventional interaction patterns only, generous touch targets, and WCAG 2.1 AA as a floor are defaults, not enhancements.
3. **Real data or honest absence, never invented content.** Fee, slot, eligibility, and legal/policy figures come from real, sourced data; where the dataset is genuinely missing something, the product says so rather than filling the gap with a plausible-looking placeholder.
4. **The server is authoritative.** Every meaningful action (wishlist changes, document uploads, submission) confirms against the real backend before the UI reflects it — a family should never see a false "saved" state.
5. **One task, one obvious way back.** Every screen has a clear forward action and an obvious way back, matching a first-time applicant's need for reassurance over a power user's need for speed.

## Accessibility & Inclusion

WCAG 2.1 AA is a stated floor, not a target: sufficient contrast, mobile/touch-first design (44px+ tap targets), plain-language error messages and jargon glosses, only conventional/learnable interaction patterns (explicitly no hover-reveal, gesture-only, or scroll-jacking interactions on the functional app surface), an obvious forward/back on every screen, one task per screen, and low-bandwidth-conscious asset choices (real photography deliberately deferred/limited — a 23.8MB hero SVG was already caught and replaced with a lightweight CSS gradient for exactly this reason).
