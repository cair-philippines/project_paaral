---
name: PAARAL Student View
description: The public enrollment portal for DepEd's Educational Service Contracting (ESC) program.
colors:
  ledger-navy: "#19266b"
  warm-gold: "#fcca81"
  brick-red: "#b23836"
  ink-black: "#020315"
  paper-white: "#fcfcfd"
  neutral-surface: "#f8fafc"
  neutral-border-subtle: "#f1f5f9"
  neutral-border: "#e2e8f0"
  neutral-border-strong: "#cbd5e1"
  neutral-muted: "#94a3b8"
  neutral-caption: "#64748b"
  neutral-body: "#475569"
  neutral-emphasis: "#1e293b"
  status-info: "#1d4ed8"
  status-success: "#15803d"
  status-warning: "#b45309"
  status-danger: "#b91c1c"
  status-offer: "#7e22ce"
typography:
  display:
    fontFamily: "var(--font-geist-sans), Arial, Helvetica, sans-serif"
    fontSize: "clamp(3rem, 3rem + 4vw, 6rem)"
    fontWeight: 800
    lineHeight: 0.9
    letterSpacing: "-0.025em"
  headline:
    fontFamily: "var(--font-geist-sans), Arial, Helvetica, sans-serif"
    fontSize: "clamp(1.5rem, 1.5rem + 1vw, 1.875rem)"
    fontWeight: 700
    lineHeight: 1.25
  title:
    fontFamily: "var(--font-geist-sans), Arial, Helvetica, sans-serif"
    fontSize: "1.25rem"
    fontWeight: 700
    lineHeight: 1.4
  body:
    fontFamily: "var(--font-geist-sans), Arial, Helvetica, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 400
    lineHeight: 1.6
  label:
    fontFamily: "var(--font-geist-sans), Arial, Helvetica, sans-serif"
    fontSize: "0.625rem"
    fontWeight: 700
    lineHeight: 1.4
    letterSpacing: "0.1em"
rounded:
  sm: "8px"
  md: "12px"
  lg: "16px"
  full: "9999px"
spacing:
  xs: "8px"
  sm: "12px"
  md: "16px"
  lg: "24px"
  xl: "48px"
components:
  button-primary:
    backgroundColor: "{colors.ledger-navy}"
    textColor: "{colors.paper-white}"
    rounded: "{rounded.sm}"
    padding: "8px 22px"
  button-primary-hover:
    backgroundColor: "{colors.ledger-navy}"
  button-decisive:
    backgroundColor: "{colors.brick-red}"
    textColor: "{colors.paper-white}"
    rounded: "{rounded.sm}"
    padding: "6px 12px"
  button-decisive-hover:
    backgroundColor: "{colors.brick-red}"
  card-elevated:
    backgroundColor: "{colors.paper-white}"
    rounded: "{rounded.lg}"
    padding: "16px"
  badge-label:
    backgroundColor: "{colors.ledger-navy}"
    textColor: "{colors.paper-white}"
    rounded: "{rounded.full}"
    padding: "2px 8px"
---

# Design System: PAARAL Student View

## Overview

**Creative North Star: "The Civic Ledger"**

PAARAL Student View looks like a well-run government service window, not a startup product: a plain, trustworthy record of one family's application, held together by a numbered sequence of small, legible entries rather than a single sprawling form. Its personality sits deliberately between two things at once — **restrained and official** (a deep navy authority color, sparse use of any color that isn't neutral, real DepEd/ECAIR institutional branding in the header) and **approachable and reassuring** (plain-language labels instead of bureaucratic jargon, warm gold used as a quiet reassuring accent rather than an alarm color, generous white space and rounded — never sharp — corners). Neither reads as the whole system; the two together are the point. This is not a marketing surface performing trustworthiness — it is an actual civic record, and it looks like one: honest about what it doesn't yet know ("Not available" rather than an invented value), authoritative about what it does.

The system is built almost entirely from two disciplined moves, repeated everywhere: a **numbered entry** (a small circular index badge, an uppercase eyebrow, a heading — the "path marker" pattern that structures the account page, the eligibility questionnaire's progress, and the landing page's guided-process section) and a **flat, bordered card** (never a shadow alone, never a border alone — always both, together, at a consistent `rounded-2xl`). Almost the entire palette is neutral (near-black ink, near-white paper, a wide ladder of slate grays for secondary text and hairline borders); the three brand colors are used sparingly and each has exactly one job. This restraint is a considered choice for a mandatory, non-opt-in government service used by a broad and often non-tech-savvy public — see the Colors and Do's & Don'ts sections below for the rules that keep it that way.

**Key Characteristics:**
- A numbered, path-marker structure for any multi-step or multi-section flow, instead of tabs or wizards that hide content behind a click.
- Flat cards with a hairline border *and* a soft shadow together — never either alone — at a consistently rounded `16px` (`rounded-2xl`).
- A near-monochrome neutral base (ink on paper, a slate gray ladder) with three brand colors, each reserved for a single, specific job rather than freely mixed.
- A dense, compact type scale (`text-sm`/`text-xs` dominate body copy) that reads as an official record more than a marketing page — except for a single, deliberate departure: the landing-page hero, the one place the system allows itself to be expressive.
- Plain-language uppercase "kicker" labels (10px, bold, wide letter-spacing) used everywhere a piece of content needs a quiet category label — "DepEd ICTS Sign-In," "Your Eligibility," "Apply before."

## Colors

The palette is deliberately narrow: one authority color, one warm accent used with restraint, one color reserved for the single most decisive action on a screen, and a wide, quiet neutral scale that does almost all of the actual work.

### Primary
- **Deep Civic Navy** (`#19266b`): the system's authority color. Headings, links, ordinary interactive text, the "kicker" numbered badges, and — on dark surfaces (the header, the landing hero) — the dominant background via a gradient into Brick Red. Used far more than any other brand color; this is the visual signature of the whole product.

### Secondary
- **Warm Gold** (`#fcca81`): used sparingly and only decoratively — a dashed callout border, a decorative icon on a dark surface (the hero's deadline panel), a numbered badge's alternate color. **Never used as body text on a light background** — at 1.47:1 contrast on Paper White it fails WCAG outright, confirmed and deliberately worked around during the palette rollout (see Do's & Don'ts).

### Tertiary
- **Brick Red** (`#b23836`): reserved for exactly one job — the single most decisive action on a plain, light-background surface ("View School Details" on a school card; the header's primary CTA when the header itself isn't already on a gradient). Never used for two competing actions on the same screen, and never applied on top of the Primary→Accent gradient itself (it visually disappears there — see the Named Rule below).

### Neutral
- **Ink Black** (`#020315`): default body/heading text color role (in practice, headings mostly render in Deep Civic Navy instead; Ink Black is the MUI-level default text color).
- **Paper White** (`#fcfcfd`): the page background, and the default surface for every elevated card.
- **Surface Gray** (`#f8fafc`, Tailwind `slate-50`): panel and page-section backgrounds that need to sit visually behind a Paper White card (the eligibility page's page background, empty-state boxes).
- **Border Subtle** (`#f1f5f9`, `slate-100`): the hairline border on the system's most common elevated cards (school result cards, fact tiles).
- **Border** (`#e2e8f0`, `slate-200`): the default border for panels, dialogs, and option cards — the more commonly reached-for of the two border tones.
- **Border Strong** (`#cbd5e1`, `slate-300`): a firmer border for a card that needs slightly more definition, and for the honest "no photo available" placeholder box.
- **Muted Text** (`#94a3b8`, `slate-400`): the lightest text tone — meta values under a label, placeholder-ish text.
- **Caption Text** (`#64748b`, `slate-500`): the standard secondary-text and uppercase-label color; the single most reused text tone in the system.
- **Body Text** (`#475569`, `slate-600`): standard paragraph copy.
- **Emphasis Text** (`#1e293b`, `slate-800`): body text that needs to read a shade more confident than standard copy (an option card's selected label, a found-learner's name).

### Semantic status colors (not brand colors — application/eligibility state only)
- **Status Info** (`#1d4ed8` on `#eff6ff`/`#bfdbfe`): "submitted," "under review" — also Category B's badge color.
- **Status Success** (`#15803d` on `#f0fdf4`/`#bbf7d0`): "redeemed," confirmed states.
- **Status Warning** (`#b45309` on `#fffbeb`/`#fde68a`): "additional document requested" — also Category D's badge color.
- **Status Danger** (`#b91c1c` on `#fef2f2`/`#fecaca`): "not approved" / rejected states.
- **Status Offer** (`#7e22ce` on `#faf5ff`/`#e9d5ff`): "subsidy offered, not yet redeemed" — a deliberately distinct fourth hue so an *offer awaiting a decision* never reads as either a plain success or a plain in-progress state.

These always appear as a matched trio — a `50`-level background, a `200`-level border, and the `700`-level text/icon color above — never the text color alone against Paper White.

### Named Rules
**The One Job Rule.** Brick Red exists to mark exactly one decisive action per screen. If a screen already has a decisive Brick Red button, no other element on that screen gets Brick Red — reach for Deep Civic Navy or a plain border instead.

**The Gradient-Proof Fallback Rule.** On any surface that itself uses the Primary→Accent gradient (the header, the landing hero), Brick Red is never used for a button — it visually disappears at the gradient's red end. Use white-on-Navy instead (`background: var(--background)`, `color: var(--primary)`), confirmed the hard way twice in this project (the hero CTA, then again when the header adopted the same gradient and its CTA had to be fixed the same way).

**The Trio Rule.** A semantic status color never appears as text alone on Paper White — it always comes with its matching light background and border (see Semantic status colors above). A bare colored word with no surrounding card reads as a broken link, not a status.

## Typography

**Display Font:** Geist Sans (with Arial, Helvetica, sans-serif fallback)
**Body Font:** Geist Sans (same family — no separate body typeface)
**Label/Mono Font:** Geist Mono, used only for literal data (an LRN, a demo email address), never for prose

**Character:** One typeface family carries the entire system, differentiated purely by size, weight, and case rather than a second typeface — appropriate for a civic record where a second display face would read as branding rather than information. The one deliberate exception is the landing-page hero, where the same family is pushed to an unusually large, heavy, tight-leading display size — the system's single moment of expressive typography, everywhere else stays compact.

### Hierarchy
- **Display** (800, `clamp(3rem, 3rem + 4vw, 6rem)`, line-height 0.9): the landing-page hero headline only ("PROJECT PAARAL") — its lead word bumped further, to 900/`font-black`, as the one intentional weight jump in the whole system.
- **Headline** (700, `clamp(1.5rem, 1.5rem + 1vw, 1.875rem)`, line-height 1.25): section titles on the landing page ("About ESC," "About PAARAL").
- **Title** (700, 20px/`1.25rem`, line-height 1.4): the most common heading size in the system — numbered section titles on the account page, the eligibility questionnaire's per-step heading, a modal's title.
- **Body** (400, 14px/`0.875rem`, line-height 1.6): the dominant text size everywhere — descriptions, form helper text, card copy. A smaller `text-xs` (12px) step is used one level down for meta/caption text (fact-tile values, timestamps).
- **Label** (700, 10px/`0.625rem`, letter-spacing 0.1em, uppercase): the recurring "kicker" — an uppercase category tag above a heading ("DepEd ICTS Sign-In," "Your Eligibility," "Apply before"). Always Caption Text (`slate-500`) in color, never a brand color.

### Named Rules
**The One Typeface Rule.** Geist Sans (and Geist Mono for literal data) is the only typeface in the system. A new surface does not introduce a second display face — hierarchy comes from size, weight, and the display/headline/title/body/label scale above, not from typeface variety.

## Layout

The system has no persistent multi-panel chrome outside of `/browse` (a collapsible filter sidebar plus a full-bleed map/list/card result area, mirroring the same collapsible-panel mechanics on both sides of that page). Everywhere else, content flows as a single centered column (`max-w-lg` for the eligibility questionnaire's card, `max-w-2xl` for account content, `max-w-5xl`/`max-w-6xl` for landing-page sections) inside generous horizontal page padding (`px-6` on mobile, `px-12` at `md`).

Density is compact and civic, not airy and marketing-led: card internal padding is typically `16px` (`p-4`), section padding `24px` (`p-6`), and the recurring vertical rhythm between stacked elements is small (`gap-3`/`gap-2`, 8–12px) except between major page sections, which get generously more room (`py-14`, `mt-8`). Multi-step or multi-section content (the account page, the eligibility questionnaire, the landing page's guided process) always renders as a numbered, top-to-bottom sequence — never tabs, never an accordion that hides a step behind a click.

Responsive behavior is mobile-first and touch-conscious: the header collapses into a hamburger sheet under `sm`, hero buttons stack to full-width under `sm`, and every interactive target (buttons, the mobile menu toggle, wishlist drag handles) is held to a `44px` (`min-h-11`/`h-11`) minimum regardless of its visual size — a floor stated explicitly in code, not incidental.

## Elevation & Depth

The system is flat, with one deliberate exception: elevated cards get both a hairline border *and* a soft `shadow-sm` at rest, together, never either alone — border alone reads flat and untrustworthy for a civic record, shadow alone can be too subtle to read as a card boundary on a low-quality or low-brightness screen, which matters directly for this product's stated broad, often-lower-bandwidth-device audience. A card's hover state (where hover exists at all — result cards, prep-step cards) deepens this into a slightly larger `shadow-md`/`shadow-lg` plus a small `-translate-y` lift, never a color change.

### Shadow Vocabulary
- **Resting card** (`box-shadow` per Tailwind's `shadow-sm`): the default state for every elevated card — school result cards, fact tiles, prep-step cards, the eligibility questionnaire's main panel.
- **Hover-lifted card** (Tailwind's `shadow-md`/`shadow-lg`, paired with `hover:-translate-y-0.5` to `-translate-y-1`): the response to hover/interaction on a card that itself does something on click (a result card, a "Before You Start" step) — never applied to a purely informational card.
- **Selected-state ring** (`ring-4 ring-primary/10`, no shadow change): a school result card that is the current map selection uses a soft Navy ring instead of a deeper shadow, so "selected" and "hovered" stay visually distinct.

### Named Rules
**The Border-and-Shadow-Together Rule.** An elevated card always carries both a light border and `shadow-sm` at rest. Neither substitutes for the other; this was a deliberate correction made during the palette rollout after a border-only card (`FactTile`) was found reading as flat.

## Shapes

Corners are rounded everywhere and the radius scales with a component's weight, never sharp (no `rounded-none` used deliberately anywhere in the system):
- **8px** (`rounded-lg`): small buttons, tags, and compact controls.
- **12px** (`rounded-xl`): option cards, dialogs, and mid-weight panels (the eligibility questionnaire's choice buttons, the login modal's confirmation box).
- **16px** (`rounded-2xl`): the system's dominant "elevated card" radius — school result cards, fact tiles, prep-step cards, the eligibility questionnaire's main panel. This is the radius to reach for by default for any new card.
- **Full/pill** (`9999px`): numbered circular badges, status/type pill labels, and — distinctively — the landing hero's three action buttons, which are the one place in the system buttons go fully pill-shaped rather than the standard `8px`.

A secondary, less common shape move is the **dashed border** (`border-2 border-dashed`), currently used in exactly two places for two different reasons — a gold dashed border around an informational aside on the landing page (the guided-process callouts), and a neutral gray dashed border around the honest "no photo available" empty state on a school detail page. These are task-local choices, not yet generalized into a system-wide "dashed = X" rule — treat any future dashed-border use as its own decision, not an automatic reach.

## Components

Buttons, cards, and inputs read as **quietly confident**: a step more considered than a bare government form (the fully pill-shaped hero buttons, the consistent `rounded-2xl`+border+shadow card treatment, the deliberate one-decisive-action-per-screen color rule), while staying well inside "plain and legible" — no gradients on interactive elements beyond the two intentional full-bleed background surfaces, no novel control shapes, nothing a first-time, possibly anxious applicant has to puzzle out.

### Buttons
- **Shape:** `8px` radius (`rounded-lg`) by default; the landing hero's three action buttons are the one deliberate exception, fully pill-shaped (`999px`) with a `2px` semi-transparent white border visible at rest — not hover-only, because a border that only appears on hover is invisible to a touch user who has no hover state at all.
- **Primary (MUI `variant="contained"`):** Deep Civic Navy fill, white text — the default for standard form actions (login "Continue," submit buttons).
- **Decisive (Brick Red fill):** reserved for the single most important action on a plain-background surface — see The One Job Rule above.
- **Outlined (on dark/gradient surfaces):** white text, a `40%`-opacity white border that solidifies to full white on hover — the header/hero's "Log In"/"My Account" treatment.
- **Hover / Focus:** MUI's default ripple/opacity shift for standard buttons; hero buttons get a `scale(1.04)` on hover and `scale(0.97–0.98)` on tap (via Framer Motion) plus a translucent white fill and soft shadow.
- **Ghost / Text:** plain text-weight links in Deep Civic Navy, underline on hover — used for secondary actions ("← Use a different email," "Browse More Schools").

### Cards / Containers
- **Corner Style:** `16px` (`rounded-2xl`) for the dominant elevated-card type; `12px` (`rounded-xl`) for a lighter panel (option cards, dialogs).
- **Background:** Paper White at rest; an eligibility option card gets a faint `bg-primary/5` wash on hover and when selected.
- **Shadow Strategy:** see Elevation & Depth — border and `shadow-sm` together, always.
- **Border:** `Border Subtle` (`slate-100`) on the most common cards (result cards, fact tiles); `Border` (`slate-200`) on option cards and panels.
- **Internal Padding:** `16px` (`p-4`) standard; `20–24px` (`p-5`/`p-6`) for a card meant to read as more important (category cards, the eligibility panel itself).

### Inputs / Fields
- **Style:** plain MUI `TextField`/`Select`/`Checkbox`/`Switch` at their library defaults — no custom radius, border, or focus-ring override anywhere in the codebase. This is a deliberate restraint, not an oversight: form controls stay maximally conventional for an audience that must never have to learn a novel input pattern.
- **Focus:** MUI's default outlined-field focus treatment (a filled Navy-colored border/label, via the theme's `primary.main`).
- **Error:** MUI's default error state (the theme's `error.main`, which happens to share Brick Red's hex value — a coincidence of the palette, not a deliberate dual-use; the app's own hand-built error/rejection banners use plain Tailwind red, not this token — see Do's & Don'ts).

### Navigation
- **Style:** a sticky, backdrop-blurred header. Guest state uses the Primary→Accent gradient; a logged-in state switches to a solid `Deep Civic Navy` bar at 95% opacity — a visible, deliberate signal that the account context has changed. Nav links and the account control are plain white text/outlined buttons; a hamburger sheet (bordered white square icon button) appears under `sm` and expands into a full-width stacked link list.

### Badges / Pills
- **Type badge** (school card): a soft-tinted pill (`rounded-full`, 10px uppercase bold text) — Navy-on-Navy/10 for public, green-on-green/10 for ESC-participating private, amber-on-amber/10 for non-ESC private.
- **Status badge** (application/eligibility state): the four-color semantic system above, always as a bordered, tinted card (never a bare pill) since these carry real consequence for the family reading them, not just a category label.
- **Numbered index badge:** a solid-fill circle (Navy, Gold, or Brick Red), white number, `64px` in the landing page's guided-process section, `36px` (`h-9 w-9`) and Navy-tinted-on-white (`bg-primary/10`) in the account page's numbered sections — the same badge shape, deliberately shrunk and lightened for a denser, more civic-record context.

### Numbered Path Marker (signature component)
The system's single most distinctive recurring pattern: a small circular index badge, an uppercase eyebrow label, and a bold title, with an optional single right-aligned action link — used to structure the account page's Status/Choices/Documents/Survey content, and echoed (at a larger scale) by the eligibility questionnaire's step badges and the landing page's 3-step guided process. Content following this header always scroll-reveals into view as a unit. Any new multi-part flow in this system should default to this pattern before reaching for tabs.

## Do's and Don'ts

### Do:
- **Do** pair a border and `shadow-sm` on every elevated card — never one without the other.
- **Do** use `rounded-2xl` (16px) as the default radius for any new card; drop to `rounded-xl` (12px) only for a lighter panel like a dialog or option button.
- **Do** keep Brick Red to exactly one use per screen — the single most decisive action on a plain, light background.
- **Do** fall back to white-text-on-Navy-background for a primary action placed on the Primary→Accent gradient (the header, the hero) — never Brick Red there.
- **Do** pair any semantic status color with its matching light background and border, never as bare colored text.
- **Do** use the 10px uppercase "kicker" label (Caption Text, wide letter-spacing) above a heading whenever content needs a quiet category tag.
- **Do** default to the numbered path-marker pattern for any new multi-step or multi-section flow, not tabs.
- **Do** hold every interactive target to a 44px minimum hit area, regardless of its visual size.

### Don't:
- **Don't** use Warm Gold as text on a light (Paper White) background — it fails contrast outright (1.47:1) and was deliberately worked around, not overlooked.
- **Don't** introduce a second display typeface. Hierarchy comes from size/weight/case within Geist Sans, not from typeface variety.
- **Don't** use hover-only affordances (a border, a button, an icon that only appears on `:hover`) anywhere on the functional app surface — this system's users include touch-only devices with no hover state, and first-time applicants who need a control to already look interactive.
- **Don't** style a form control (`TextField`, `Select`, `Checkbox`, `Switch`) beyond MUI's defaults. Conventional, learnable controls are a stated product requirement, not a placeholder waiting for polish.
- **Don't** assume MUI's `error` palette color and the Brick Red accent token are interchangeable just because they share a hex value — the app's actual error/rejection banners are built from plain Tailwind red (`red-50`/`red-200`/`red-700`), not the accent CSS variable.
- **Don't** apply scroll-jacking, gesture-only, or otherwise novel interaction patterns anywhere on the functional app surface — motion here is limited to a short (220ms) upward scroll-reveal and small hover lifts/scales, always `prefers-reduced-motion`-aware except the landing hero's one-time load-in sequence (a deliberate, narrowly scoped exception).
