# Domain Docs

How the engineering skills should consume this repo's domain documentation.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root — not yet created. Until it exists, domain knowledge lives in:
  - `CLAUDE.md` (project overview, policy references, key gates)
  - `.claude/rules/memory-decisions.md` (past decisions with rationale)
  - `.claude/rules/memory-sessions.md` (chronological build log)
- **`docs/adr/`** — not yet created. Architectural decisions are currently recorded in `.claude/rules/memory-decisions.md`.

If these files don't exist, **proceed silently**. Don't flag their absence upfront.

## File structure (single-context)

```
/
├── CONTEXT.md          ← create when domain modeling is needed
├── docs/adr/           ← create when architectural decisions need formal records
└── src/
```

## Use the glossary's vocabulary

When naming domain concepts (issue titles, refactor proposals, test names), use terms as defined in `CONTEXT.md` or as used in `CLAUDE.md`. Key terms for this project: LRN, LIS, ESC, Category A/B/C/D, SEG, canSubmit, applicationState, wishlist, PAARAL.

## Flag ADR conflicts

If your output contradicts an existing decision in `.claude/rules/memory-decisions.md` or a future ADR, surface it explicitly rather than silently overriding.
