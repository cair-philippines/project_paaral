# PAARAL Student View

> **Routing file.** Under 150 lines. Details live in `.claude/rules/` and `docs/`.

## What This Is

Public-facing enrollment portal for Grade 6 → Grade 7 learners applying to the DepEd Educational Service Contracting (ESC) program.

**Current phase: MOCKUP** — UI/UX pilot demo, no real backend.

**Companion repos:**
- `../deped-planning-view` — DepEd prescriptive analytics (slot allocation, congestion simulation)
- School View — coming soon

---

## Auto-Update Memory (MANDATORY)

**Update memory files AS YOU GO, not at the end.** When you learn something new, update immediately.

| Trigger | Action |
|---------|--------|
| User shares a fact about themselves | → Update `.claude/rules/memory-profile.md` |
| User states a preference | → Update `.claude/rules/memory-preferences.md` |
| A decision is made | → Update `.claude/rules/memory-decisions.md` with date |
| Completing substantive work | → Add to `.claude/rules/memory-sessions.md` |

**Skip:** Quick factual questions, trivial tasks with no new info.

**DO NOT ASK. Just update the files when you learn something.**

---

## Context Files

@.claude/rules/memory-profile.md
@.claude/rules/memory-preferences.md
@.claude/rules/memory-decisions.md
@.claude/rules/memory-sessions.md

| What | Where |
|------|-------|
| Tech stack + coding standards | @SKILLS.md |
| **Python standards (PEP8, NumPy docstrings, ruff)** | @SKILLS.md § Python Standards |
| Build plan + chunk status | @WORKFLOW.md |
| Chronological build log | @LOG.md |
| Architecture docs, ERD, meeting notes | `docs/` |

---

## Demo Credentials

| LRN | Behavior |
|-----|----------|
| `100000000001` | Valid, fresh (no draft) |
| `100000000002` | Valid, with 3 draft choices pre-loaded |
| `200000000001` | Invalid (not found in LIS) |

---

## Key Submission Gates

All four required for `canSubmit`:
1. LRN verified against DepEd LIS
2. At least one school in wishlist
3. At least one public JHS in wishlist (guarantees placement)
4. All three required survey questions answered

---

## Legal / Policy Basis

- **RA No. 8545** — E-GASTPE Act (amended from RA 6728)
- **E-GASTPE 2026 Revised Guidelines** — DepEd Order s. 2026 (Article III, Sections 7–13)
- **PIDS Income Classification** — brackets provided by Paula 2026-05-04; use those exact figures

---

## Agent skills

### Issue tracker

GitHub Issues on `cair-philippines/project_paaral` (branch: `mockups/student-view`). External PRs are not a triage surface. See `docs/agents/issue-tracker.md`.

### Triage labels

Default label vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout. No `CONTEXT.md` yet — domain knowledge currently lives in `CLAUDE.md` and `.claude/rules/`. See `docs/agents/domain.md`.
