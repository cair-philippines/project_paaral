# Work Preferences

## Code Style (JavaScript / React)
- **Single-file architecture** for mockup phase — all components inline in `App.jsx` by design, not an oversight
- No router, no state library — `useState`/`useMemo` only until production phase
- No unnecessary abstractions — three similar lines beats a premature helper
- No comments unless the WHY is non-obvious; never write what the code already says

## Code Style (Python)
- **PEP8** enforced via `ruff` (`pyproject.toml` in project root)
- **NumPy docstring convention** for all functions and modules — no other docstring style
- One import per line — never `import a, b, c`
- Line length: 79 characters (strict PEP8)
- Run `ruff check docs/` and `ruff format docs/` before considering Python work done

## Documentation
- CLAUDE.md is a routing file, under 150 lines — no knowledge dumps inline
- Split detail into `.claude/rules/memory-*.md` files
- Update memory files in-session as things happen, not at the end

## Policy / Data Integrity
- Never substitute income brackets or legal thresholds from training data — always use figures Paula has explicitly provided
- If policy is ambiguous (eligibility edge cases, category boundaries), ask Paula before implementing

## Collaboration
- Paula provides policy/scope direction; Claude executes
- Don't redesign scope or suggest architectural pivots unless asked
- When Paula says something is intentional (e.g. single-file), don't flag it as a concern
