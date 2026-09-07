"use client";

import Image from "next/image";
import Link from "next/link";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import { motion } from "framer-motion";
import { User } from "lucide-react";
import DeadlinePanel from "@/components/molecules/DeadlinePanel";
import { useApplication } from "@/components/templates/ApplicationStateProvider";

/**
 * Structural template (2026-08-24, Step 2 of the recolor pass) — the hero
 * background uses the primary→accent gradient token decoratively (tasteful,
 * not garish: both colors are dark/mid-tone, so white text stays readable
 * no matter where in the gradient it falls). Superseded below for the
 * background specifically (2026-09-07).
 *
 * Institutional branding chip (added 2026-09-07, closing a question left
 * open since 2026-08-24 when SiteHeader — and its DepEd/ECAIR logos — was
 * removed from this page): a landing page for a mandatory government
 * service showing zero official branding before asking a family to trust
 * it undercuts the "restrained and official" half of the Civic Ledger
 * identity. Both logos are full-color (navy/red/gold) on a transparent
 * background, so placing them directly on the gradient would repeat the
 * exact "Brick Red disappears on its own gradient" problem already solved
 * for buttons — same fix applied here: a small Paper White chip, not the
 * full navbar (deliberately minimal, per the critique's suggested fix).
 *
 * Background + glassmorphism buttons + typography (2026-09-07): matches
 * the confirmed comp at impeccable.style artifact 8b9e9df0 (this session's
 * hero typography/glass-exploration thread — NOT the earlier, unrelated
 * "Blue Ballot Hero" flat-joined-bar comp briefly and incorrectly built
 * here in between). Background is the real docs/hero.svg asset — that
 * source file is 23.8MB (4 base64-embedded PNGs behind a duotone filter,
 * too heavy to ship as-is), re-rendered once and exported as
 * public/assets/hero-bg.jpg (2400x1350, ~37KB). Two scrims keep genuinely-
 * translucent glass buttons legible regardless of where the photo's own
 * bright/dark regions land: a bottom-anchored one (fixed px distance from
 * the bottom edge, not a percentage — min-h-svh plus vertically-centered
 * content means the button row's distance from either edge shifts with
 * viewport height) for the action-button row, plus a small top-right
 * corner scrim for the Log In/My Account pill specifically, which sits
 * right where the source image's own bright glow tends to land, outside
 * the bottom scrim's reach — found by real contrast sampling that dropped
 * to 4.06:1 (below the 4.5:1 floor) before that scrim was added.
 *
 * Buttons: real backdrop-filter blur, layered inset highlights (top/bottom
 * + a diffuse inner glow) AND top/left edge-light streak pseudo-elements —
 * adapted from Paula's own reference `.glass-card` CSS (both the shadow
 * stack and the streaks, not just the shadow — a real gap in an earlier
 * pass here, corrected after "you did not follow the template we
 * created"), scaled down from that reference's literal 16px-blur/8px-
 * spread/0.8-opacity values (tuned for a 360px-tall card) to fit our much
 * shorter pill buttons. Layered a real SVG feTurbulence/feDisplacementMap
 * filter (progressive enhancement via @supports — plain blur is the
 * universal fallback, notably for Safari) for genuine refraction, not
 * just a translucent tint.
 *
 * Typography is a deliberate, surface-scoped exception to DESIGN.md's "One
 * Typeface Rule": Public Sans ExtraBold (headline), Public Sans Light
 * (the acronym-expansion caption), Lato Light (body paragraph) — Geist
 * stays the sitewide default everywhere else.
 */
// Ported from Paula's reference `.glass-card` template (a 240x360 settings
// panel): the layered inset box-shadow (top/bottom highlight + diffuse inner
// glow) AND the top/left edge-light streak pseudo-elements — both pieces,
// not just the shadow. Adapted per element below since our hero has 4
// separate pill buttons, not one card: glow magnitude scaled to each
// button's actual (much shorter) height rather than copied at the
// template's literal 16px-blur/8px-spread/0.8-opacity values, which were
// tuned for a 360px-tall box and would flood a 37-64px pill with white.
const glassBase = {
  position: "relative" as const,
  overflow: "hidden" as const,
  isolation: "isolate" as const,
  background: "rgba(255,255,255,0.16)",
  backdropFilter: "blur(16px) saturate(180%)",
  WebkitBackdropFilter: "blur(16px) saturate(180%)",
  textShadow: "0 1px 4px rgba(4,7,26,0.5)",
  "@supports (backdrop-filter: url(#glass-distortion))": {
    backdropFilter: "blur(9px) url(#glass-distortion) saturate(180%)",
  },
  "&::before": {
    content: '""',
    position: "absolute",
    top: 0,
    left: "10%",
    right: "10%",
    height: "1px",
    background:
      "linear-gradient(90deg, transparent, rgba(255,255,255,0.85), transparent)",
    pointerEvents: "none",
  },
  "&::after": {
    content: '""',
    position: "absolute",
    top: 0,
    left: 0,
    width: "1px",
    height: "100%",
    background:
      "linear-gradient(180deg, rgba(255,255,255,0.8), transparent 55%, rgba(255,255,255,0.25))",
    pointerEvents: "none",
  },
};

// The action row is ONE joined glass bar with the three actions as internal
// segments (a divider between them, not a gap) — "the white bar crossing
// the seam" from the original Civic Ballot comp: a single continuous band
// that visually cuts across the photo's own diagonal blue→red transition,
// not three separate floating pills. An earlier pass here split this into
// three individual pill buttons with gaps when porting the comp into React
// — a real deviation from the confirmed design, not a deliberate choice.
const actionBarSx = {
  ...glassBase,
  display: "flex",
  flexDirection: { xs: "column", sm: "row" },
  borderRadius: "20px",
  border: "1px solid rgba(255,255,255,0.4)",
  boxShadow:
    "0 10px 28px rgba(20,10,50,0.28), inset 0 1px 0 rgba(255,255,255,0.55), inset 0 -1px 0 rgba(255,255,255,0.12), inset 0 0 16px 4px rgba(255,255,255,0.35)",
};

const actionBarLinkSx = {
  flex: 1,
  minHeight: 64,
  borderRadius: 0,
  color: "white",
  fontWeight: 700,
  letterSpacing: "0.05em",
  textTransform: "uppercase" as const,
  fontSize: "0.78rem",
  px: { xs: 2, sm: 3.5 },
  // MUI's own button base styles set `border: 0` with enough specificity to
  // beat a parent `divide-x` utility, so the seam between segments has to
  // live on each button directly — flips to a bottom border when the bar
  // stacks vertically on mobile.
  borderRight: { xs: "none", sm: "1px solid rgba(255,255,255,0.28)" },
  borderBottom: { xs: "1px solid rgba(255,255,255,0.28)", sm: "none" },
  transition: "background-color 170ms ease",
  "&:hover": { bgcolor: "rgba(255,255,255,0.12)" },
};
const actionBarLinkLastSx = {
  ...actionBarLinkSx,
  borderRight: "none",
  borderBottom: "none",
};

const loginButtonSx = {
  ...glassBase,
  border: "1px solid rgba(255,255,255,0.4)",
  color: "white",
  borderRadius: "100px",
  boxShadow:
    "0 4px 16px rgba(20,10,50,0.2), inset 0 1px 0 rgba(255,255,255,0.5), inset 0 -1px 0 rgba(255,255,255,0.1), inset 0 0 6px 1px rgba(255,255,255,0.22)",
  "&:hover": {
    borderColor: "white",
    bgcolor: "rgba(255,255,255,0.26)",
  },
};
// `behavior: "smooth"` here (rather than relying on the global scroll-smooth
// CSS via `behavior: "auto"`) is deliberate — calling scrollIntoView with an
// implicit/CSS-driven smooth behavior synchronously inside a click handler
// was found to silently no-op in testing. Explicit "instant" is the only
// behavior confirmed reliable in that exact context.
function scrollToSection(id: string) {
  document.getElementById(id)?.scrollIntoView({ behavior: "instant", block: "start" });
}

// Load-in sequence for the hero — headline, subtitle, and buttons fade up
// one after another. Not scroll-triggered (it's above the fold, always
// visible on load); deliberately not reduced-motion-aware, per Paula's
// explicit call to skip that for now on this specific animation.
const fadeUp = (delay: number) => ({
  initial: { opacity: 0, y: 16 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.4, delay, ease: "easeOut" as const },
});

export default function HeroSection() {
  const { account, openLoginModal } = useApplication();

  return (
    <>
      {/* Hidden SVG filter powering the buttons' glass refraction — see
          glassBase's @supports block above. Browsers that don't understand
          `backdrop-filter: url(#...)` (Safari, older browsers) simply never
          reference this and fall back to the plain blur. */}
      <svg aria-hidden="true" style={{ position: "absolute", width: 0, height: 0 }}>
        <filter id="glass-distortion">
          <feTurbulence type="fractalNoise" baseFrequency="0.008" numOctaves="2" seed="8" result="noise" />
          <feDisplacementMap in="SourceGraphic" in2="noise" scale="28" xChannelSelector="R" yChannelSelector="G" />
        </filter>
      </svg>
      <section
        className="relative flex min-h-svh flex-col items-center justify-center bg-[#1d2c6b] bg-cover bg-center px-6 py-10 text-center md:px-12 md:py-16"
        style={{
          // Two scrims: a bottom-anchored one (fixed px distance from the
          // bottom edge, not a percentage) for the action-button row, plus
          // a small top-right corner scrim for the Log In/My Account pill
          // specifically. See the block comment above for why both exist.
          backgroundImage:
            "radial-gradient(400px 300px at 100% 0%, rgba(5,8,20,0.55), rgba(5,8,20,0) 75%), linear-gradient(180deg, rgba(5,8,20,0) 0%, rgba(5,8,20,0) calc(100% - 480px), rgba(5,8,20,0.32) calc(100% - 200px), rgba(5,8,20,0.46) 100%), url(/assets/hero-bg.jpg)",
        }}
      >
      <div className="absolute left-6 top-6 flex items-center gap-0 md:left-12 md:top-8">
        <Image
          src="/assets/deped-logo.svg"
          alt="Department of Education"
          width={28}
          height={16}
          className="h-8 w-auto md:h-12"
        />
        <Image
          src="/assets/ecair-logo.svg"
          alt="ECAIR"
          width={17}
          height={6}
          className="h-8.5 w-auto md:h-13.5"
        />
      </div>
      <div className="absolute right-6 top-6 md:right-12 md:top-8">
        {account ? (
          <Button
            component={Link}
            href="/account"
            variant="outlined"
            startIcon={<User className="h-4 w-4" />}
            sx={loginButtonSx}
          >
            My Account
          </Button>
        ) : (
          <Button onClick={openLoginModal} variant="outlined" sx={loginButtonSx}>
            Log In
          </Button>
        )}
      </div>
      <motion.h1
        {...fadeUp(0.1)}
        className="font-[family-name:var(--font-public-sans)] mt-4 text-5xl font-extrabold leading-[0.92] tracking-tight text-white sm:text-6xl md:text-7xl lg:text-8xl"
      >
        PROJECT PAARAL
      </motion.h1>
      <motion.p
        {...fadeUp(0.15)}
        className="font-[family-name:var(--font-public-sans)] mx-auto -mt-3 whitespace-nowrap text-sm font-light uppercase tracking-[0.14em] text-[#fcca81] max-sm:max-w-[30ch] max-sm:whitespace-normal sm:-mt-4 sm:text-base md:-mt-6 lg:-mt-8"
      >
        Platform for Analyzing Access and Resource Allocation in Learning
      </motion.p>
      <motion.p
        {...fadeUp(0.2)}
        className="font-[family-name:var(--font-lato)] mx-auto mt-5 max-w-[64ch] text-base font-light leading-7 text-white/90 sm:text-lg sm:leading-7"
      >
        PAARAL is a platform built for the Educational Service
        Contracting (ESC) program of the Department of Education. It
        helps families of Grade 6 learners find ESC-participating
        schools, apply for a subsidy, and see where school slots remain
        insufficient.
      </motion.p>
      <motion.div {...fadeUp(0.3)} className="mt-8 flex w-full flex-col items-center gap-3">
        <motion.div
          whileHover={{ scale: 1.015 }}
          whileTap={{ scale: 0.99 }}
          className="w-full max-w-3xl"
        >
          <Box sx={actionBarSx}>
            <Button component={Link} href="/browse" disableElevation sx={actionBarLinkSx}>
              Browse Schools
            </Button>
            <Button
              onClick={() => scrollToSection("about-esc")}
              disableElevation
              sx={actionBarLinkSx}
            >
              Know More About ESC
            </Button>
            <Button
              onClick={() => scrollToSection("about-paaral")}
              disableElevation
              sx={actionBarLinkLastSx}
            >
              Know More About PAARAL
            </Button>
          </Box>
        </motion.div>
        <DeadlinePanel />
      </motion.div>
      </section>
    </>
  );
}
