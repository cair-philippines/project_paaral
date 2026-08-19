# Past Decisions

## Prototype pivots to decoupled ESC-only state machine; `logout()` full-reset is mockup-only (2026-08-19)

**Supersedes the 2026-07-13 admission-first rework entry below for `src/App.jsx`.** Paula scrapped the admission-first sequencing (confirmed with DepEd stakeholders that session) in favor of the same decoupled "Portable Eligibility" model from the policy paper (`docs/ESC-policy-note-revised.md`, 2026-07-31 entry) — but applied to the prototype's own state machine, not just the policy paper. Her framing: "we have to scrap and do a decoupled process for both enrollment and ESC application. but PAARAL should handle first and foremost the ESC application" — and explicitly, admission/enrollment tracking is **out of scope for now**: "don't assume the admission status. students can enroll before or after being accepted as an ESC beneficiary."

**What changed in `src/App.jsx`:** the half-finished July 13 admission-first diff (`conditionally_accepted`/`admission_rejected`/etc., never committed) was discarded via `git checkout`, and the state machine was rebuilt from the clean committed base:
- Account-level: `eligibility → submitted → granted | non_esc | eligibility(reapply)`, plus `not_eligible → non_esc`. No admission-dependent states anywhere — `granted` is a true terminal outcome (ESC certificate secured) regardless of enrollment timing.
- Per-school `escStatuses` (private only): `submitted → granted | rejected | docs_pending → docs_submitted → granted | rejected`. Both `granted` and `rejected` are terminal at the school level.
- **Public schools carry no ESC status at all** — they're purely the `hasPublicAlternative` guaranteed-placement checkbox, never entered into the ESC pursuit. Rank-restriction and "apply to next choice" walk `privateChoices` only (a public school ranked in between is skipped, not resolved) — a real simplification the old admission-era model couldn't make.
- Kept from the already-validated 2026-07-07 design (independent of the admission question): `not_eligible` path, split general/ESC survey sections, per-school Choices-tab status badges.
- Dropped: the "Continue at ICTS Portal" external link on the granted screen — replaced with a one-line note that enrollment is a separate, independent process. No admission mechanics anywhere.
- **Explicitly not rebuilt this pass** (were in the discarded diff, ruled out of scope): in-app Messages/timeline tab, localStorage session-persistence across page reload, wishlist drag-reorder.

Verified end-to-end via a JS-driven browser session (no test framework in this repo): single-private-school submit→reject→continue-without-subsidy naming the right school; multi-rank submit→reject→next-choice-prompt (correctly skipping an intervening public rank)→apply→grant naming the right school throughout; `not_eligible`→enroll-without-ESC naming the right school. Zero console errors.

**Separate fix, same session — `logout()` now also resets `eligStep`/`eligHistory`/`eligAnswers`.** Previously those questionnaire-local states survived logout (only `account`/`appView`/`drawerOpen` were cleared), so a fresh demo login could resume mid-questionnaire or on a stale result screen instead of starting clean. **Paula's explicit caveat: this full reset-on-logout is mockup-only behavior.** In production, account state (including any in-progress eligibility answers) is server-persisted and must survive logout — don't carry this reset pattern into the production migration (`docs/gcp-production-migration.md`) without re-deriving it against real session/auth semantics.

## ESC policy note (docs/ESC-policy-note-revised.md): proposed model changed from strict "Eligibility-First" to decoupled "Portable Eligibility" (2026-07-31)

**Scope: this is Paula's independent DepEd policy-reform paper (real-world ESC program critique), not the PAARAL Student View prototype's own state machine.** Do not confuse with the 2026-07-13 entry below — that one governs `src/App.jsx`'s actual sequencing (admission-before-ESC, per DepEd stakeholder direction) and is unaffected by this entry.

Working through the §5 "Current vs. Proposed Process" logic, Paula redirected the proposed model away from the original draft's strict sequential fix (certificate must be issued before any school application, mirroring the SHS Voucher Program) toward a fully **decoupled, parallel-track model**:

- **ESC eligibility and school admission are independent processes.** Neither is gated on the other; a family may apply for the ESC certificate first, apply to a school first, or pursue both at once.
- **Redemption is the single convergence point.** A family finalizes subsidized enrollment by presenting its (school-agnostic, learner-held) certificate at whichever school has admitted it. Before redemption, a certificate and an admission offer are both inert with respect to each other.
- **A rejection on either track never voids the other** — this is what actually closes the paper's own §3.4 gap ("a denial is terminal"), which the original strict-order fix only partially addressed.
- **Slot claim timing: only at redemption**, confirmed explicitly over the alternative (provisional hold at admission). Admission offers and certificate issuance never decrement a school's ESC slot count.
- **Parallel admission shopping is allowed** — a family may pursue admission at multiple schools simultaneously (as in ordinary non-ESC admissions), confirmed explicitly over one-school-at-a-time.
- **Named consequence, stated deliberately rather than left implicit:** because slots are claimed only at redemption and admission shopping is parallel, a popular school can see more certified-and-admitted learners arrive than it has remaining slots (a "redemption race condition"). The rewrite states this plainly as a first-come-at-redemption allocation property, not a flaw to hide.
- **Renamed throughout:** "Eligibility-First" → **"Portable Eligibility"** (or "Decoupled Track"), since "Eligibility-First" implied a strict order that no longer describes the model. This propagated to the paper's own title (`docs/ESC-policy-note-revised.md` H1), §5's diagram (both the inline ASCII art and the Mermaid appendix — redesigned from two linear tracks into two parallel lanes converging on a "REDEMPTION" node), §6's comparison-table column header, and §7's TWG mandate/roles/transition-plan language.
- **Explicitly deferred by Paula, not yet touched:** a sample operational calendar (e.g., "school starts June, final ESC-enrolled list must exist by then") — she said to focus on the sequencing logic first, not the timeline. Revisit only if she raises it again.
- Both `docs/ESC-policy-note-revised.md` and its HTML rendering `docs/ESC-policy-note.html` (published as a Claude.ai Artifact) were updated in lockstep — keep them in sync on any future edits to this section.

## Stakeholder answer inverts module order: school admission comes BEFORE ESC application, not after (2026-07-13)

**DepEd stakeholder clarification (Dir. TCR, Louie CD, Marlon Custodio), received in response to a pilot-scope question Paula sent via the shared live doc's new "Process Flow Diagrams" tab:**

> Q: Between actual school application and ESC application, which comes first? That is, will ESC applications be held only after a student has been admitted/accepted to a school?
> A: The school application comes first. Students must first apply and be conditionally accepted by a school before proceeding with or finalizing their ESC application, subject to eligibility and requirements.

**This inverts the entire current state machine and requires a major rework of both the process flow diagram and the prototype (`src/App.jsx`).** Everything built so far (Module 1 rework — per-private-school ESC status, 2026-07-07 entries below) assumed ESC subsidy resolution happens first, then a separate not-yet-built Module 2 (General Admissions) happens after. That assumption is now confirmed wrong: general school admission (conditional acceptance) is the *first* gate; ESC eligibility/subsidy only proceeds or finalizes once a school has conditionally accepted the student.

**Not yet designed / open as of 2026-07-13 — do not guess, ask Paula:**
- Whether the ESC eligibility questionnaire (category A–D self-assessment) can still happen any time, independent of school admission, or whether it's also gated behind conditional acceptance — the stakeholder wording ("proceeding with or finalizing") suggests these may be two different gates, not one.
- What "conditionally accepted" resolves to afterward — does it become a distinct "fully accepted" state once ESC resolves (granted or rejected-but-enrolling-non-esc), or is conditional acceptance simply superseded by the existing `esc_granted`/`esc_rejected`/`enrolled_esc`/`enrolled_non_esc` terminal states?
- Whether public schools (currently instant/guaranteed placement, no ESC involved) also need an explicit conditional-acceptance step now that admission is a first-class prior stage, or keep their current always-accepted shortcut.
- Whether the existing rank/wishlist behavior (only rank 1 active, explicit opt-in to next rank on rejection — see Module 1 rework entry below) carries over unchanged onto the new admission-first sequence, or needs its own redesign.

**Per repeated lesson in this file (see "process note" entries below): do not implement a state-machine rework from partial signal — confirm the above with Paula first.**

**Resolved 2026-07-13 — flow diagram rebuilt and confirmed, saved as `docs/student-view-diagram-v4.html`.** Answers to the open questions above, worked out via iterative Mermaid diagram review with Paula:
- **The ESC-first track is removed entirely.** There is no longer a way to apply for the ESC subsidy independent of school admission — admission via ICTS always happens first, full stop. This also collapses what was previously a two-page diagram (`student-view-diagram-v3.pdf`, which modeled "either order" with a "Pre-approved (slot reserved)" vs "None (want to apply now)" split) into a single linear flow.
- **New gate, confirmed separate from the existing ESC document checklist:** after a school's admission result comes back "Conditionally Accepted," the student must upload proof of conditional acceptance before an ESC application can even start for that school. This is its own gate, not bundled into `getDocList`'s existing per-category document list (see the 2026-07-13 conversation earlier this session — Paula's call, not yet implemented in code).
- **Corrected from the old v3 PDF's logic:** if a student misses the attendance-confirmation deadline (the second of the two conditions needed to actually fill an ESC slot — the other being school admission), this is now a dead end that lapses the conditional acceptance entirely and returns the student to browsing/next choice. The old v3 PDF had this "fall through to non-ESC enrollment at the same school" instead — that only made sense in the old either-order model where admission was independently secured before ESC ran. Since admission is now only ever *conditional* until this deadline is met, missing it means neither ESC nor regular admission was secured at that school.
- **Simplified during review:** the "confirm attendance by deadline" question and the "pending documents?" check are each a single shared step in the diagram, used identically whether the student is on the ESC-approved track or the proceed-as-non-ESC-after-ESC-rejection track — no need to duplicate the same mechanical question per track. The ESC vs. non-ESC distinction is carried forward as already-decided status from earlier in the flow, not re-asked.

**Not yet done:** the prototype (`src/App.jsx`) state machine has not been reworked to match this — this session's diagram work was explicitly step one, prototype rework is next. Do not start that from this diagram alone without confirming the per-school `escStatus` state names/transitions with Paula first (same "ask before implementing" lesson).

**Paula's correction: "opting out of ESC entirely just means they will enroll to that school without a subsidy. so, whatever happens, there will always be a school id."** There is no such thing as a schoolless `enrolled_non_esc` — every path to it names a specific school:
1. **Public auto-admit** (rank order reaches a public school) — always had a school, unchanged.
2. **`esc_rejected` → "Continue Enrollment at `<school>` (No Subsidy)"** — the student enrolls at the *same* school that rejected the subsidy, just paying full price. New button, replaces the old schoolless "Continue Non-ESC Enrollment."
3. **`not_eligible` → "Enroll Without ESC"** — previously had no school reference at all (a real gap, fixed same-day); now ties to `wishlist[0]` (their top choice) with its own message.

**Important distinction surfaced while implementing this — `esc_rejected` vs `admission_rejected` are not interchangeable:**
- `esc_rejected` = the ESC committee denied the *subsidy*, before any admission decision. The student can still enroll at that school full-price → "Continue Enrollment (No Subsidy)" is offered.
- `admission_rejected` = the school already refused the *admission* itself, after the subsidy was granted. There is no fallback enrollment at that school — only "apply to next rank" (if available) or "stop and choose different schools" (if ranks are exhausted). Offering a non-ESC enrollment option here would be offering to enroll somewhere that already said no.
- `ESC_SCHOOL_TRANSITIONS.esc_rejected` now includes `enrolled_non_esc`; `admission_rejected`'s transition list stays empty (terminal, no fallback state — only reachable via the account-level "next rank" or "stop" actions, not a school-status transition).

**Simplification side-effects:**
- `ACCOUNT_MESSAGES` (the schoolless account-level message map) is now dead code and was removed — every message-worthy transition builds its own school-specific message at the call site. `advance()` simplified accordingly (dropped its auto-message lookup).
- **ERD impact: `MESSAGE.school_id` can become `NOT NULL`** — every row always has one now. (Was previously modeled as nullable "for the schoolless opt-out case," which no longer exists.)

**Bug caught during verification, same day:** the top-level `enrolled_non_esc` Status-tab card was re-deriving its description text from `schoolStatusConfigs.enrolled_non_esc.desc(name)` (the generic "general admission" wording) regardless of which of the 3 paths above produced it — so a student who explicitly continued without subsidy after an ESC rejection saw the wrong copy ("placed through general admission," implying automatic/guaranteed, when they'd actually made an explicit choice). Fixed by having that card display `messages[messages.length - 1]?.text` — the actual stored message, which each path already computes correctly at the source — instead of re-deriving it in the render layer. General lesson: when the same terminal state is reachable via multiple paths with different correct wording, render the stored message rather than re-deriving text from state, or the two are guaranteed to drift apart.

Verified end-to-end with Playwright (5/5 checks): esc_rejected shows the continue-without-subsidy button and produces the right message; admission_rejected does not show that button. Screenshots confirmed the copy fix visually.

## Enrollment status added on top of ESC status — esc_granted is now intermediate, not final (2026-07-07)

**Extends the per-school ESC status work above.** `granted` used to be a per-school terminal value; it's now an *intermediate* one — ESC subsidy secured, but the school's actual admission decision is a separate, later step that can still fail. Full per-school escStatus lifecycle for a private school:
```
submitted → docs_pending → docs_submitted → esc_granted → enrolled_esc | admission_rejected
                                          ↘ esc_rejected
```
Both `esc_rejected` and `admission_rejected` lead to the same "apply to your next choice?" prompt (`REJECTED_STATES = new Set(['esc_rejected', 'admission_rejected'])`) — a subsidy that was granted but then failed admission is just as much a reason to move to the next rank as never getting the subsidy at all.

**Naming, deliberately explicit per Paula's correction:** `granted`/`rejected` were renamed to `esc_granted`/`esc_rejected` specifically so they're never confused with the later admission-stage outcome (`admission_rejected`, `enrolled_esc`). Account-level final states also renamed: `granted` → `enrolled_esc`, `non_esc` → `enrolled_non_esc`. Public schools skip the whole cycle and resolve straight to `enrolled_non_esc` (still "always accepted," per the earlier ranking-order fix) — they never pass through `esc_granted` at all, since there's no subsidy or separate admission decision to simulate for them.

**New demo controls:** "Simulate: Admission Approved" / "Simulate: Admission Rejected," shown only while a school's status is `esc_granted` (reuses the exact same `schoolStatusConfigs[status].demo` rendering already built for `submitted`/`docs_submitted` — no new UI pattern needed).

**New "Not Selected" tag in the Choices tab:** once ANY school reaches `esc_granted` or a final `enrolled_*` state (`isLockedIn`), every other *not-yet-engaged* school in the wishlist shows "Not Selected" instead of a blank badge — since the pursuit only ever ends in one placement, everything else is now moot. Schools that were actually tried and failed keep their specific rejection label (not relabeled to "Not Selected") — only the never-reached ones get it.

**Simplification side-effect:** since `enrolled_esc` and `enrolled_non_esc` are now distinct, unambiguous values (no more overloaded `granted` meaning two different things depending on school type), the `labelFor(state, schoolType)` disambiguation helper from the previous fix became unnecessary and was removed — `STATE_LABELS[state]` alone is now always correct. Worth remembering if this comes up again: more specific state names are often simpler than a generic name plus a disambiguation function.

Verified end-to-end with Playwright (13/13 checks): rank-1 → esc_granted → admission_rejected → explicit opt-in to rank-2 → esc_granted → enrolled_esc (final), confirming the intermediate card, the new demo controls, the "Not Selected" tag on the never-reached rank-3, and all badge/timeline labels. Screenshots confirmed visually.

**ERD impact (not yet written to actual `.sql`):** `WISHLIST.esc_status` enum grows to 8 values: `submitted, docs_pending, docs_submitted, esc_granted, esc_rejected, admission_rejected, enrolled_esc, enrolled_non_esc`. `APPLICATION.status` enum stays at 5 values, just renamed: `eligibility, not_eligible, submitted, enrolled_esc, enrolled_non_esc`.

## Scope Correction — PAARAL owns ESC-track enrollment end-to-end, via two modules (2026-07-07)

**Stakeholder context, learned 2026-07-07: the stakeholder's long-term vision is for PAARAL to become a full enrollment management system for ALL graduating Grade 6 students (ESC-interested or not), since PAARAL will be significantly ahead of ICTS in production readiness.** Pilot scale is unchanged (~300,000 students). **However, Paula's direction: stay focused on ESC for now.** The concrete, in-scope change from this broader vision is narrower than "handle everyone" — it's specifically:

**PAARAL now owns the full journey of ESC-track students — application AND actual school enrollment — not just ESC eligibility/subsidy determination with a handoff to ICTS for enrollment.** Two technical modules, conceptually:
- **Module 1 — ESC Application** (existing, mostly built): the eligibility questionnaire, category determination (A-D), and subsidy approval. State machine: `eligibility → submitted → docs_pending → docs_submitted → granted/rejected`.
- **Module 2 — General School Admissions** (not yet built): the actual admission/enrollment process at a specific school. Entered by students who've cleared Module 1 — **confirmed 2026-07-07: both `granted` (with subsidy) and `rejected` → "Continue Non-ESC Enrollment" (without subsidy) feed into this same Module 2**, not two separate flows.

**What's explicitly still out of scope:** students who are `not_eligible` from the start (never entered Module 1 at all — determined ineligible at the eligibility questionnaire, not rejected after submitting) are NOT part of this change. Their "browse without ESC" pathway is unaffected for now — this is the stakeholder's broader universal-enrollment vision, not today's scope. Don't conflate the two: "rejected-then-non-ESC" (was in Module 1, now goes to Module 2) is different from "never-eligible" (never entered Module 1, stays as-is).

**Concrete things in the current mockup now confirmed stale (found 2026-07-07, not yet fixed):**
- `granted` state currently treats enrollment as complete via an external handoff — `schoolStatusConfigs.granted.desc` and the "Continue at ICTS Portal" button (literal link to `https://icts.deped.gov.ph`) in `src/App.jsx`. Under the correction, `granted` should transition into Module 2 (built in PAARAL), not point external. (Names updated 2026-07-07 per the per-school rework below — was `STATE_MESSAGES.granted`/`statusConfigs.granted.desc` before that rework.)
- `non_esc` (reached via rejected → "Continue Non-ESC Enrollment") is currently a one-click terminal state with no real process. Under the correction, it also needs to feed into Module 2 (same admissions flow, no subsidy attached).

**Not yet designed:** Module 2's own state machine/ERD, and whether `granted`/`non_esc` become entry states into it or get renamed. Treat this as an open design discussion for a follow-up session — don't unilaterally design Module 2 from this note alone.

### Module 1 rework — per-private-school ESC status, implemented 2026-07-07

**Confirms and operationalizes the retracted answer above: per-school status tracking is real, and now built.** Paula's direction: ESC results are per-*private* JHS choice, not one blanket status for the whole application — and the public-school safety-net choice in the wishlist gets no ESC status at all (that's Module 2's territory, not simulated yet). Resolution model, after walking through the tradeoffs (serial-with-auto-advance vs. fully-parallel vs. a hybrid): **rank restricts submission, and advancing to the next rank is never automatic.**

- **Account-level `applicationState` shrank to 5 states:** `eligibility → not_eligible → submitted → granted | non_esc`. It stays `submitted` for the entire time the student is working through their ranked private schools, however many ranks that takes.
- **New per-school lifecycle, on each private `WISHLIST` entry:** absent (not yet applied) → `submitted` → `docs_pending` → `docs_submitted` → `granted` | `rejected`. Only one private school is ever active at a time, always starting from rank 1.
- **On "Submit Application":** all ranked preferences are recorded as before, but only rank 1 (the first *private* school in wishlist order — public entries are skipped for this purpose) actually gets applied to.
- **On rejection, the UI explicitly asks** — "Would you like to apply to your next choice, `<school>`?" — with a real button, never auto-advancing. The student can instead choose "Continue Non-ESC Enrollment" or "Stop and Choose Different Private Schools" at any point.
- **Messages and the Application Timeline are now per-school** — e.g. "Your ESC application to St. Mary's Academy of Taguig has been received..." — via `ESC_SCHOOL_MESSAGES` (text) and `STATE_LABELS` (short timeline labels), both taking the school's name.
- **Choices tab** shows a small status badge per school once engaged (public schools included as of the bugfix below); not-yet-engaged schools show nothing.
- Implementation detail for future reference: `account.escStatuses` is a plain `{ [schoolId]: status }` map (absence = not yet applied). `activeChoice`, `lastEngagedChoice`, `nextChoice`, `grantedChoice` are all derived in the main component from `wishlist` order + `escStatuses` — none of it is stored separately, so wishlist reordering before any engagement just changes who's "rank 1" for free.
- Verified with a Playwright driver end-to-end (rank-1 submit → reject → explicit opt-in to rank-2 → grant), 17/17 checks passed, screenshots confirmed visually.
- **ERD impact (not yet written to actual `.sql`):** `WISHLIST` needs an `esc_status` column (nullable enum, private schools only); `MESSAGE` needs a `school_id` FK. `APPLICATION.status` enum shrinks to the 5 account-level states. See Chunk 14 progress in `memory-sessions.md`.

**Bugfix, same day, caught by Paula from a screenshot:** the initial implementation computed `nextChoice`/`lastEngagedChoice` from `privateChoices` (private-only, filtered) instead of the full ranked `wishlist` — so after a rank-1 private rejection, the "apply to your next choice?" prompt silently skipped over a public rank-2 and jumped straight to a private rank-3. **Fix — "always follow the ranked choices; if the next in line is public, simulate general admissions, assumed always accepted":**
- `nextChoice`/`lastEngagedChoice`/`grantedChoice` now derive from the full `wishlist` (public included), not `privateChoices`. Only `activeChoice` (a school currently *pending* review) stays private-only, since public never sits in a pending state — it resolves instantly.
- New `buildRankOutcome(school)` branches on `school.type`: private → normal `submitted` (starts the review cycle); public → resolves straight to `granted` with distinct messaging ("placed through general admission, no ESC subsidy applies"), no review cycle, no yes/no ambiguity about the outcome (though the explicit "would you like to apply to your next choice?" prompt still appears before committing — only the *result* is instant, not the opt-in step).
- `granted` now means two different things depending on `grantedChoice.type` — an ESC subsidy (private) or a guaranteed general admission (public, Module 2 simulated as always-accepted). `schoolStatusConfigs.granted.title`/`.desc` became functions of the school object (not just a name string) to branch on this. New `labelFor(state, schoolType)` helper (next to `STATE_LABELS`) does the same for the Choices-tab badge, the Application Timeline, and the top-level `renderStateBadge` — all three needed the same public/private distinction fixed.
- Verified by reproducing the exact reported scenario (rank order: private → public → private) with Playwright, 11/11 checks, screenshots confirmed visually — including the top-level badge, which needed a second pass after the first fix left it still reading "ESC Granted" for the general-admission case.

## Domain Framing — Varbi as Benchmark (2026-07-07)

**Student View's core role is functionally analogous to Varbi (recruitment system): an intermediary tool that matches applicants to institutions, not a system of record for either side.** PAARAL-specific additions (map-based school discovery, ESC eligibility determination, ranked wishlist, document checklist/verification) sit on top of that same core pattern — they don't change what kind of system this fundamentally is.

**Why this matters:** the ESC application state machine (`eligibility`/`not_eligible` → `submitted` → `docs_pending` → `docs_submitted` → `granted`/`rejected`, see v3 Architecture below) is structurally the same shape as a recruitment ATS pipeline (applied → screening → interview → offer/reject). When a future feature decision is ambiguous, ask "what would a mature recruitment/enrollment intermediary do here" (status visibility, document handling, applicant communication, pipeline stages) rather than inventing a bespoke pattern from scratch.

**How to apply:** use this as a benchmark/reference point when evaluating UX or feature parity questions on the application lifecycle, status tracking, or document handling — not as a mandate to copy Varbi's UI. Domain-specific features (map, ESC eligibility, PIDS income logic) remain PAARAL's own and aren't benchmarked against Varbi.

### In-app system messages — implemented in the mockup (2026-07-07); email notifications explicitly deferred

Following the Varbi benchmark, an in-app "inbox" (Varbi's "you have a message" pattern) was added to the prototype — a new **Messages** drawer tab, a red unread-count badge on both the tab and the "My Account" button, and an auto-generated message on every committee-driven state transition (`submitted`, `docs_pending`, `docs_submitted`, `granted`, `rejected`, `non_esc` — not `eligibility`/`not_eligible`, since those are user-driven, not something a committee "messages" about).

**Explicitly deferred: email notifications.** Paula's direction: don't implement email during dev. Varbi's pattern sends a short generic "ping" email ("You have a message from X, click here to read") separate from the in-app content — cost scales with number of notification *events*, not message content, so it's a small line item once implemented, likely free-tier at pilot scale and modest (rough order of magnitude: tens–low hundreds of $/month) at the ~300k-student full-scale target. The real open decision isn't the dollar cost, it's **delivery mechanism**: a third-party transactional email provider (SendGrid/Mailgun) vs. relaying through DepEd's own mail infra for `@deped.gov.ph` addresses — the latter needs DepEd IT coordination, similar in kind to the LIS/BEIS/ICTS "infra first, integrate later" deferrals already made. Revisit alongside Chunk 20/21.

**Implementation (mockup, `src/App.jsx`):**
- `STATE_MESSAGES` — module-level map of `applicationState` → message text, next to `VALID_TRANSITIONS`
- `advance()` appends a `{ id, text, createdAt, read: false }` entry to `account.messages` whenever the target state has an entry in `STATE_MESSAGES`
- `unreadCount` / `markMessagesRead()` derived in the main component; messages marked read when the Messages tab is opened
- **Production implication:** this needs a `MESSAGE`/`NOTIFICATION` table in the real schema (not yet in the 7-table ERD) — flagged in `docs/gcp-production-migration.md`. The mockup's `STATE_MESSAGES` map (auto-generate text from a transition) is a reasonable pattern to carry into the backend rather than having a human author each message by hand.

## Mockup Architecture (2026-05-04)
- **What's mocked:** LRN verification (hardcoded set), school data (50 synthetic schools), draft choices (hardcoded per LRN), submission (local state only), survey responses (not persisted)
- **LRN check:** `VALID_LRNS = new Set(["100000000001", "100000000002"])` — deliberate, not a prefix trick. Represents registry identity check only, not an ESC eligibility gate.
- **Single-page app:** No router. All state in `PAARALStudentMockup()`.

## App Layout (SUPERSEDED 2026-06-10 — see Browse UX below)
- ~~Old: 3-panel sidebar layout (filters + map + right portal panel)~~
- Now: `appView` state drives hero vs browse; no persistent sidebar panels

## Browse UX — Hero + Map + Profile + Drawer (2026-06-10, SUPERSEDED by v3 2026-06-26)
- See v3 architecture below for current state.

## v3 Architecture — ESC Application State Machine (2026-06-26, revised 2026-07-07)

**Auth:** DepEd email (`100000000001@deped.gov.ph`) as ICTS SSO stand-in. LRN no longer used directly for login.

**Single test account:** One account progresses through all states. State is persisted in `localStorage` under key `paaral_v3_account`. Note: `account` React state is NOT restored from `localStorage` on a hard page reload (no restore-on-mount effect exists) — only survives logout/login within a continuous session, not a browser refresh. This is a pre-existing gap, not something fixed in the 2026-07-07 revision; flag if it matters for a future session.

**`appView`:** `'hero' | 'eligibility' | 'browse'`
- `eligibility` is a full-screen view (replaces the modal questionnaire)
- On login, account created with `applicationState: 'eligibility'` → `appView` set to `'eligibility'`

**Revised 2026-07-07: removed the `browsing` state; added `not_eligible`.** Previously `browsing` was overloaded — used both for ESC-eligible applicants building their wishlist AND for ineligible learners browsing without ESC intent, with the pre-submission/post-submission split otherwise undocumented in code. Now:
- **`eligibility`** covers the whole pre-submission span for an ESC-track learner: still answering the questionnaire, and (once a category is assigned) browsing/building the wishlist. No separate stored state for "done with questionnaire, now browsing."
- **`not_eligible`** is a distinct state for learners whose questionnaire result was ineligible (no category). They can still browse and build a wishlist, then enroll directly as non-ESC.
- The state badge (`renderStateBadge`) disambiguates the two sub-phases of `eligibility` at the UI layer only (label reads "Completing Eligibility" vs "Browsing Schools" based on `Boolean(account.category)`) — this is not a separate stored value.

**8 application states (`applicationState`), current `VALID_TRANSITIONS`:**
```
eligibility    → submitted
not_eligible   → non_esc
submitted      → granted | rejected | docs_pending
docs_pending   → docs_submitted
docs_submitted → granted | rejected
rejected       → non_esc | eligibility   (reapply: resets wishlistIds to force a different school choice)
non_esc        → (terminal)
granted        → (terminal)
```
Confirmed against the actual `VALID_TRANSITIONS` object in `src/App.jsx` — `submitted`'s three-way branch (granted/rejected/docs_pending) was verified correct as-is; only the pre-submission side (`browsing`→removed, `not_eligible`→added) changed in the 2026-07-07 revision. (Earlier memory here incorrectly described a 9-state machine with a `pre_approved` state between `submitted` and `docs_pending` — that state never existed in code; corrected 2026-07-07.)

**Drawer tabs:**
- Pre-submission (`eligibility` or `not_eligible`): Choices | Documents | Survey
- Post-submission: Status | Documents | Choices (read-only)

**`isPostSubmission`:** `POST_SUBMISSION_STATES.has(applicationState)` — the set `{submitted, rejected, docs_pending, docs_submitted, granted, non_esc}`. `eligibility` and `not_eligible` are the only non-post-submission states.

**Survey — revised 2026-07-07 into two sections, so it's inclusive of non-ESC applicants:**
- **"Using PAARAL"** (all users, both tracks): ease-of-finding-schools (1-5) + did-the-info-help-you-decide (Yes/Somewhat/No). Drives `generalSurveyComplete`.
- **"About Your ESC Application"** (ESC-track only, hidden entirely when `applicationState === 'not_eligible'`): biggest-concern-enrolling-through-ESC (Cost/Distance/School quality/Slot availability). Drives `escSurveyComplete`.

**Two submit paths, replacing the old single `canSubmit`/`handleSubmit`:**
- `canSubmitEsc` / `handleSubmitEsc` — gates: `applicationState === 'eligibility'` + wishlist ≥ 1 + hasPublicAlternative + docsReady + `generalSurveyComplete` + `escSurveyComplete`. Advances to `submitted`. Checklist label: "Submission Checklist."
- `canEnrollNonEsc` / `handleEnrollNonEsc` — gates: `applicationState === 'not_eligible'` + wishlist ≥ 1 + `generalSurveyComplete` only (no public-JHS gate, no ESC-docs gate — not applicable to a non-ESC enrollment). Advances directly to `non_esc`. Checklist label: "Enrollment Checklist." Button reads "Enroll Without ESC," not "Submit Application."

**Demo advance controls:** Embedded in Status tab, per-state. Reject path has two real action buttons: "Continue Non-ESC Enrollment →" (`advance('non_esc')`) and "Apply Again — Choose a Different School →" (`advance('eligibility', { wishlistIds: [] })` — explicitly clears the wishlist so re-application forces a different school choice, matching the button's copy). Granted state shows ICTS external link.

**Verified 2026-07-07** via a Playwright-driven browser session (real login → questionnaire → wishlist → survey → submit/enroll → status transitions) for both the ESC-eligible and not-eligible paths — all 31 behavioral assertions passed, including badge labels, checklist gating, section visibility, and button text.

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

## GCP Production Migration Architecture (2026-07-06)

Full roadmap: `docs/gcp-production-migration.md`; tracked in `WORKFLOW.md` Chunks 13+.

- **Backend:** Node.js + Express + TypeScript. Same language as the frontend; ICTS will eventually API into student view for ESC applications, so this needs a documented OpenAPI contract, not just an internal API.
- **Structured data:** Cloud SQL for PostgreSQL — matches the finalized 7-table ERD exactly (composite PKs, junction tables, enums). No Firestore — the "documents to store" are files (ID scans, income certs), not NoSQL documents.
- **Files:** Cloud Storage, referenced by `DOCUMENT_UPLOAD.file_url` — that column was already designed for this.
- **Migration strategy:** Strangle-fig — replace one mocked concern at a time behind the current UI (LRN verify first, then schools, then wishlist/survey, then documents, then eligibility), not a parallel-build-then-cutover.
- **External integrations (DepEd LIS, BEIS, ICTS SSO):** Infra first, integrate later — confirmed by the 2026-06-02 ICTS alignment deck (`docs/`), which shows ICTS had not yet committed to a tech stack, schema, or whether their portal consumes PAARAL, as of that meeting.
- **Frontend hosting:** Stays on Vercel; only the backend/DB move to GCP for now.
- **Backend repo — DECIDED 2026-07-06, renamed 2026-07-06:** new sibling repo `paaral-student-api` (renamed from `paaral-api`) at `/Users/paulamartinez/paaral-student-api` (local only, not yet on GitHub). Empty — README + .gitignore committed, no code. Chunk 14 (Express/TypeScript skeleton + raw SQL schema) is next.
- **GCP project — REVISED 2026-07-06: reuse `ecair-paaral-project`, not a dedicated new project.** Originally planned a fresh `ecair-paaral-student-dev` project to isolate backend infra from `ecair-paaral-project`'s unrelated ad hoc resources (Cloud Run, Maps APIs, BigQuery, storage buckets). Blocked in practice: Paula's account can create projects (one was created and later deleted) but lacks `billing.user` permission on the SEAMEO billing account (`010812-0A04B3-791EAF`), so new projects can't be billing-linked without someone else's help. Rather than block on that, backend resources go into the existing `ecair-paaral-project` (already billing-enabled), separated from the ad hoc resources and from each other by **naming convention**, not project boundaries: prefix all Student View backend resources with `paaral-student-` (e.g. Cloud SQL instance `paaral-student-dev-db`, bucket `paaral-student-dev-uploads`). This is a real trade-off — no project-level IAM/quota isolation between backend infra and ad hoc resources — acceptable for now given the dev-only, pre-launch stage; revisit a dedicated project once `billing.user` access is sorted (see IAM request list below).
- **Orphaned artifact:** `ecair-paaral-student-dev` project exists (undeleted after an earlier accidental deletion) but is unused — billing was never linked. Not actively cleaned up; safe to ignore or delete later.
- **Permission still worth requesting later:** `roles/billing.user` on billing account `010812-0A04B3-791EAF`, from whoever administers it — would unblock creating dedicated per-platform projects (e.g. reviving the `ecair-paaral-student-dev` / `-staging` / `-prod` split) without needing this workaround.
- **Dev Cloud SQL instance created via free trial (2026-07-06) — deadline 2026-08-05.** `paaral-student-dev-db` in `ecair-paaral-project`, created 2026-07-06T16:16Z using Google's 30-day Cloud SQL free trial: `db-perf-optimized-N-8` (N2, 8 vCPU / 64GB RAM), Enterprise Plus edition, 100GB PD_SSD, ZONAL. This is much bigger than the originally-planned `db-f1-micro` dev tier — free for 30 days, but **stops serving requests on 2026-08-05** if not upgraded to paid before then (data is preserved for a further 90-day grace period after that — no silent billing risk, but the instance goes read-only/inaccessible). **Decision made 2026-07-07: option (b) — migrate to a new right-sized instance, not upgrade-in-place.** Target: `db-f1-micro`, `PD_HDD` storage, small disk (10-20GB) — cheapest viable dev tier. **Key constraint:** `db-f1-micro` (shared-core) requires **Enterprise** edition; the current trial instance is **Enterprise Plus**, and Cloud SQL does not support downgrading an existing instance's edition in place. So this is not a tier resize — it requires creating a new Enterprise-edition instance.

**Revised again 2026-07-07: plan is to load sample/seed data into the trial instance before the deadline, then migrate whatever exists.** Paula intends to use `paaral-student-dev-db` for Chunk 14/15 sample data in the meantime, so the "let it lapse, nothing to lose" reasoning above no longer applies once that data exists — export/import (or `pg_dump`/`pg_restore`) will be needed at cutover. Plan: use the trial instance normally up to 2026-08-05, then export whatever's accumulated (schema + sample data) and restore it into the new Enterprise-edition `db-f1-micro` + `PD_HDD` instance, then delete the old trial instance. Not yet executed — revisit exact export/import commands closer to the deadline once real schema/data exists to migrate.
- **Database access layer — DECIDED 2026-07-06: raw SQL via `pg` (node-postgres), not Prisma/an ORM.** Prisma was proposed in the original roadmap draft but never confirmed — reconsidered in favor of raw SQL to match the project's no-unnecessary-abstraction preference (see `.claude/rules/memory-preferences.md`) and because a solo/small-team pilot at modest scale doesn't need migration/type-safety tooling on top of SQL. This is reversible: Prisma can introspect an existing Postgres DB later (`prisma db pull`) and be adopted per-route/module without a rewrite, since the ORM only affects how the app talks to the DB, not the schema or data itself.
- **Platform architecture — CORRECTED 2026-07-06: PAARAL is three fully separate platforms, not one backend serving three frontends.** Student View, School View, and DepEd View each get their own infrastructure and database. `paaral-student-api` serves Student View only (plus ICTS later) — it does not serve School View or DepEd View. School View (not started) will get its own backend, `paaral-school-api`, following the same naming convention. DepEd View (`paaral-deped-view`) does not share infra with either — it gets read access to both Student View's and School View's databases to generate analytics (slot allocation, congestion simulation); the cross-platform read mechanism (replica, export, warehouse) is undesigned, deferred until School View exists. This corrects earlier framing in `docs/gcp-production-migration.md` that assumed a shared backend.

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
