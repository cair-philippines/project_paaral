"use client";

import { useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { useRouter } from "next/navigation";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import { motion, type Variants } from "framer-motion";
import { User, Globe, Menu, X } from "lucide-react";
import DeadlinePanel from "@/components/molecules/DeadlinePanel";
import { useApplication } from "@/components/templates/ApplicationStateProvider";

/**
 * Structural template (2026-08-24, Step 2 of the recolor pass) — the hero
 * background uses the primary→accent gradient token decoratively (tasteful,
 * not garish: both colors are dark/mid-tone, so white text stays readable
 * no matter where in the gradient it falls). Superseded below for the
 * background specifically (2026-09-07).
 *
 * Background + glassmorphism buttons + typography (2026-09-07): matches
 * the confirmed comp at impeccable.style artifact 8b9e9df0 (this session's
 * hero typography/glass-exploration thread). Background is the real
 * docs/hero.svg asset — that source file is 23.8MB (4 base64-embedded PNGs
 * behind a duotone filter, too heavy to ship as-is), re-rendered once and
 * exported as public/assets/hero-bg.jpg (2400x1350, ~37KB).
 *
 * Buttons: real backdrop-filter blur, layered inset highlights (top/bottom
 * + a diffuse inner glow) AND top/left edge-light streak pseudo-elements —
 * adapted from Paula's own reference `.glass-card` CSS, scaled down from
 * that reference's literal 16px-blur/8px-spread/0.8-opacity values (tuned
 * for a 360px-tall card) to fit our much shorter pill buttons. Layered a
 * real SVG feTurbulence/feDisplacementMap filter (progressive enhancement
 * via @supports — plain blur is the universal fallback, notably Safari)
 * for genuine refraction, not just a translucent tint.
 *
 * Typography is a deliberate, surface-scoped exception to DESIGN.md's "One
 * Typeface Rule": Public Sans ExtraBold (headline), Public Sans Light
 * (the acronym-expansion caption), Lato Light (body paragraph) — Geist
 * stays the sitewide default everywhere else. Unchanged by the nav rebuild
 * below — the headline/caption/body block is deliberately out of scope for
 * that pass, per direct instruction.
 *
 * Nav bar rebuilt (2026-09-08), replacing the 2026-09-08-earlier small
 * centered "logo + Log In" pill with a full-width header — adapted from an
 * approved standalone comp ("Bukas ng Paaral") explored via `/impeccable`
 * against a real Hero27-shaped component reference. Real destinations only:
 * Home / About PAARAL / About ESC reuse this page's existing anchors and
 * `scrollToSection`; **FAQs is included per direct instruction but is a
 * known no-op — there is no FAQ page/section anywhere in the app yet.**
 * The EN/FIL language toggle is also a deliberate stub, per direct
 * instruction: it flips its own displayed label only and does not
 * translate any page content yet — real localization is a separate,
 * later effort. The old 3-segment joined action bar (Browse Schools /
 * Know More About ESC / Know More About PAARAL) is retired: the latter two
 * moved into the nav above, leaving Browse Schools as the hero's one
 * primary CTA, restyled as its own glass pill rather than a bar segment —
 * this reopens the 2026-09-07 "don't revisit the 3-button hierarchy
 * without a new explicit reason" decision, per this session's direct,
 * explicit instruction to build the approved comp into production.
 */
// Ported from Paula's reference `.glass-card` template (a 240x360 settings
// panel): the layered inset box-shadow (top/bottom highlight + diffuse inner
// glow) AND the top/left edge-light streak pseudo-elements — both pieces,
// not just the shadow. Adapted per element below since our hero has several
// separate pill elements, not one card: glow magnitude scaled to each
// element's actual height rather than copied at the template's literal
// 16px-blur/8px-spread/0.8-opacity values, which were tuned for a 360px-tall
// box and would flood a much shorter pill with white.
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

// Both the Log In/My Account control and the language-toggle stub share this
// treatment — scaled down from glassBase's literal values (see note above):
// at pill-button height, the full-strength border/glow reads as one solid
// bright ring rather than glass, so the fill is thinner, the blur shallower,
// and the border softer, letting more of the actual photo tone show through.
const pillButtonSx = {
  ...glassBase,
  background: "rgba(255,255,255,0.10)",
  backdropFilter: "blur(10px) saturate(170%)",
  WebkitBackdropFilter: "blur(10px) saturate(170%)",
  "@supports (backdrop-filter: url(#glass-distortion))": {
    backdropFilter: "blur(7px) url(#glass-distortion) saturate(170%)",
  },
  border: "1px solid rgba(255,255,255,0.2)",
  color: "white",
  borderRadius: "100px",
  whiteSpace: "nowrap" as const,
  flexShrink: 0,
  boxShadow:
    "0 3px 10px rgba(20,10,50,0.16), inset 0 1px 0 rgba(255,255,255,0.28), inset 0 -1px 0 rgba(255,255,255,0.06), inset 0 0 4px 1px rgba(255,255,255,0.12)",
  "&:hover": {
    borderColor: "rgba(255,255,255,0.85)",
    bgcolor: "rgba(255,255,255,0.2)",
  },
};

// The hero's one primary CTA (2026-09-08) — "Know More About ESC/PAARAL"
// moved into the persistent nav above, retiring the old 3-segment joined
// action bar along with them. Browse Schools gets its own, slightly
// stronger glass treatment (closer to glassBase's full values) since it's
// the single most important action on the page, not a small secondary
// control like the pills above.
const ctaButtonSx = {
  ...glassBase,
  color: "white",
  fontWeight: 700,
  letterSpacing: "0.02em",
  fontSize: "1rem",
  borderRadius: "999px",
  border: "1px solid rgba(255,255,255,0.4)",
  px: 5,
  py: 2,
  boxShadow:
    "0 8px 22px rgba(20,10,50,0.24), inset 0 1px 0 rgba(255,255,255,0.5), inset 0 -1px 0 rgba(255,255,255,0.1), inset 0 0 10px 2px rgba(255,255,255,0.25)",
  "&:hover": {
    bgcolor: "rgba(255,255,255,0.26)",
    borderColor: "rgba(255,255,255,0.85)",
  },
};

// Mobile nav dropdown — solid Paper White, not glass: it holds dark text
// (Deep Civic Navy on the active row) for legibility, which a translucent
// fill would undermine. Matches the approved comp's own mobile menu, which
// used the same solid-card treatment for the same reason.
const mobileMenuCardSx = {
  background: "var(--background)",
  borderRadius: "20px",
  boxShadow: "0 18px 40px rgba(9,13,40,0.35)",
  padding: "8px",
};

type NavId = "home" | "browse-schools" | "about-paaral" | "about-esc" | "faqs";
// Kept in lockstep with SiteHeader.tsx's own nav — same items, same
// order, same labels (2026-09-08 parity pass), since the two are the
// only navbars in the app and users move between them constantly.
const NAV_ITEMS: { id: NavId; label: string }[] = [
  { id: "home", label: "Home" },
  { id: "browse-schools", label: "Browse Schools" },
  { id: "about-paaral", label: "About PAARAL" },
  { id: "about-esc", label: "About ESC" },
  { id: "faqs", label: "FAQs" },
];

// `behavior: "smooth"` here (rather than relying on the global scroll-smooth
// CSS via `behavior: "auto"`) is deliberate — calling scrollIntoView with an
// implicit/CSS-driven smooth behavior synchronously inside a click handler
// was found to silently no-op in testing. Explicit "instant" is the only
// behavior confirmed reliable in that exact context.
function scrollToSection(id: string) {
  document.getElementById(id)?.scrollIntoView({ behavior: "instant", block: "start" });
}

// Load-in sequence for the hero — independent spring physics with a
// blur-to-focus resolve on each element, not one identical fade for all of
// them. Not scroll-triggered (it's above the fold, always visible on load);
// deliberately not reduced-motion-aware, matching the standing,
// already-confirmed exception for this specific animation.
const navVariants: Variants = {
  hidden: { opacity: 0, y: -20, filter: "blur(8px)", scale: 0.97 },
  visible: {
    opacity: 1,
    y: 0,
    filter: "blur(0px)",
    scale: 1,
    transition: { type: "spring", damping: 24, stiffness: 120, duration: 0.6 },
  },
};
const titleVariants: Variants = {
  hidden: { opacity: 0, y: 32, filter: "blur(10px)" },
  visible: {
    opacity: 1,
    y: 0,
    filter: "blur(0px)",
    transition: { type: "spring", damping: 28, stiffness: 80, mass: 1.4, delay: 0.3 },
  },
};
const captionVariants: Variants = {
  hidden: { opacity: 0, y: 14, filter: "blur(4px)" },
  visible: {
    opacity: 1,
    y: 0,
    filter: "blur(0px)",
    transition: { type: "spring", damping: 22, stiffness: 110, delay: 0.5 },
  },
};
const bodyVariants: Variants = {
  hidden: { opacity: 0, y: 14, filter: "blur(4px)" },
  visible: {
    opacity: 1,
    y: 0,
    filter: "blur(0px)",
    transition: { type: "spring", damping: 22, stiffness: 110, delay: 0.6 },
  },
};
const ctaGroupVariants: Variants = {
  hidden: { opacity: 0, scale: 0.94, y: 10 },
  visible: {
    opacity: 1,
    scale: 1,
    y: 0,
    transition: { type: "spring", damping: 20, stiffness: 140, delay: 0.8 },
  },
};

export default function HeroSection() {
  const { account, openLoginModal } = useApplication();
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<NavId>("home");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  // EN/FIL toggle stub (2026-09-08, direct instruction): flips only its own
  // displayed label, does not translate any hero content yet. Real
  // localization needs to cover the whole site, not one section — a
  // separate, later effort.
  const [lang, setLang] = useState<"en" | "fil">("en");

  function handleNavClick(id: NavId) {
    setActiveTab(id);
    setMobileMenuOpen(false);
    if (id === "about-paaral" || id === "about-esc") {
      scrollToSection(id);
    } else if (id === "browse-schools") {
      router.push("/browse");
    }
    // "home" and "faqs" only move the active pill for now — home is already
    // this page, and faqs has no real destination yet (see block comment
    // above).
  }

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
        className="relative flex min-h-svh flex-col bg-[#1d2c6b] bg-cover bg-center text-center"
        style={{
          // One combined scrim: a top band protecting the full-width nav row
          // (logos through the mobile menu button — widened from a small
          // centered spot now that the nav spans the whole hero, not one
          // centered pill), transparent through the middle, and a bottom
          // band (fixed px distance from the bottom edge, not a percentage —
          // min-h-svh plus vertically-centered content means the CTA row's
          // distance from either edge shifts with viewport height) for the
          // Browse Schools button and deadline panel.
          backgroundImage:
            "linear-gradient(180deg, rgba(5,8,20,0.5) 0%, rgba(5,8,20,0.3) 100px, rgba(5,8,20,0) 220px, rgba(5,8,20,0) calc(100% - 480px), rgba(5,8,20,0.32) calc(100% - 200px), rgba(5,8,20,0.46) 100%), url(/assets/hero-bg.jpg)",
        }}
      >
      {/* Pinned flush to the very top of the hero, in its own compact strip —
          not sharing the section's own generous content padding. That
          padding was tuned for the centered headline block; reusing it for
          the nav read as the nav floating somewhere inside the hero's
          content rhythm rather than sitting above it as a persistent bar,
          and no amount of margin on the headline fixed that structurally —
          the two needed to stop being centered together as one group. */}
      <motion.header
        variants={navVariants}
        initial="hidden"
        animate="visible"
        className="relative z-20 flex w-full items-center justify-between gap-3 px-6 pt-5 pb-4 md:px-12 md:pt-6 md:pb-5"
      >
        <div className="flex items-center gap-0">
          <Image
            src="/assets/deped-logo.svg"
            alt="Department of Education"
            width={28}
            height={16}
            className="h-6 w-auto sm:h-8 md:h-12"
          />
          <Image
            src="/assets/ecair-logo.svg"
            alt="ECAIR"
            width={17}
            height={6}
            className="h-6.5 w-auto sm:h-8.5 md:h-13.5"
          />
        </div>

        <nav className="hidden items-center gap-1 md:flex" aria-label="Primary">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              onClick={() => handleNavClick(item.id)}
              className="relative appearance-none border-0 bg-transparent rounded-full px-4 py-2 text-sm font-medium transition-colors lg:px-5"
              style={{ color: activeTab === item.id ? "var(--primary)" : "white" }}
            >
              {activeTab === item.id && (
                <motion.span
                  layoutId="heroNavActivePill"
                  className="absolute inset-0 -z-10 rounded-full bg-white shadow-sm"
                  transition={{ type: "spring", stiffness: 400, damping: 30 }}
                />
              )}
              <span className="relative">{item.label}</span>
            </button>
          ))}
        </nav>

        <div className="flex items-center gap-2 sm:gap-3">
          <Button
            onClick={() => setLang((l) => (l === "en" ? "fil" : "en"))}
            variant="outlined"
            startIcon={<Globe className="h-3.5 w-3.5" />}
            sx={pillButtonSx}
            className="!hidden sm:!inline-flex"
          >
            {lang === "en" ? "FIL" : "EN"}
          </Button>
          {account ? (
            <Button
              component={Link}
              href="/account"
              variant="outlined"
              startIcon={<User className="h-4 w-4" />}
              sx={pillButtonSx}
            >
              My Account
            </Button>
          ) : (
            <Button onClick={openLoginModal} variant="outlined" sx={pillButtonSx}>
              Log In
            </Button>
          )}
          <button
            onClick={() => setMobileMenuOpen((open) => !open)}
            aria-label={mobileMenuOpen ? "Close menu" : "Open menu"}
            aria-expanded={mobileMenuOpen}
            className="appearance-none flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full border border-white/30 bg-transparent text-white transition-colors hover:bg-white/10 md:hidden"
          >
            {mobileMenuOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
          </button>
        </div>
      </motion.header>

      {mobileMenuOpen && (
        <motion.div
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative z-20 mx-6 -mt-1 max-w-xs self-center md:hidden"
        >
          <Box sx={mobileMenuCardSx} className="flex flex-col gap-1">
            {NAV_ITEMS.map((item) => (
              <button
                key={item.id}
                onClick={() => handleNavClick(item.id)}
                className="appearance-none border-0 rounded-2xl px-4 py-3 text-left text-base font-semibold transition-colors"
                style={{
                  backgroundColor: activeTab === item.id ? "rgba(25,38,107,0.08)" : "transparent",
                  color: activeTab === item.id ? "var(--primary)" : "var(--text)",
                }}
              >
                {item.label}
              </button>
            ))}
            <button
              onClick={() => setLang((l) => (l === "en" ? "fil" : "en"))}
              className="appearance-none border-0 bg-transparent mt-1 flex items-center gap-2 rounded-2xl px-4 py-3 text-left text-base font-semibold text-[var(--text)] transition-colors hover:bg-black/5"
            >
              <Globe className="h-4 w-4" />
              {lang === "en" ? "Switch to Filipino" : "Switch to English"}
            </button>
          </Box>
        </motion.div>
      )}

      {/* The centered hero content — headline through the CTA — now lives in
          its own flex-1 region below the pinned header, filling whatever
          vertical space the header didn't take rather than centering
          together with it as one group (see the header's own comment). */}
      <div className="flex flex-1 flex-col items-center justify-center px-6 py-8 md:px-12 md:py-16">
      <motion.h1
        variants={titleVariants}
        initial="hidden"
        animate="visible"
        className="font-[family-name:var(--font-public-sans)] text-5xl font-extrabold leading-[0.92] tracking-tight text-white sm:text-6xl md:text-7xl lg:text-8xl"
      >
        PROJECT PAARAL
      </motion.h1>
      <motion.p
        variants={captionVariants}
        initial="hidden"
        animate="visible"
        className="font-[family-name:var(--font-public-sans)] mx-auto -mt-3 whitespace-nowrap text-sm font-light uppercase tracking-[0.14em] text-[#fcca81] max-sm:max-w-[30ch] max-sm:whitespace-normal sm:-mt-4 sm:text-base md:-mt-6 lg:-mt-8"
      >
        Platform for Analyzing Access and Resource Allocation in Learning
      </motion.p>
      <motion.p
        variants={bodyVariants}
        initial="hidden"
        animate="visible"
        className="font-[family-name:var(--font-lato)] mx-auto mt-5 max-w-[64ch] text-base font-light leading-7 text-white/90 sm:text-lg sm:leading-7"
      >
        PAARAL is a platform built for the Educational Service
        Contracting (ESC) program of the Department of Education. It
        helps families of Grade 6 learners find ESC-participating
        schools, apply for a subsidy, and see where school slots remain
        insufficient.
      </motion.p>
      <motion.div
        variants={ctaGroupVariants}
        initial="hidden"
        animate="visible"
        className="mt-6 flex w-full flex-col items-center gap-3 md:mt-8"
      >
        <motion.div whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }}>
          <Button component={Link} href="/browse" disableElevation sx={ctaButtonSx}>
            Browse Schools
          </Button>
        </motion.div>
        <DeadlinePanel />
      </motion.div>
      </div>
      </section>
    </>
  );
}
