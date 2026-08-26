"use client";

import Link from "next/link";
import Button from "@mui/material/Button";
import { User } from "lucide-react";
import TricolorRule from "@/components/atoms/TricolorRule";
import DeadlinePanel from "@/components/molecules/DeadlinePanel";
import { useApplication } from "@/components/templates/ApplicationStateProvider";

/**
 * Structural template (2026-08-24, Step 2 of the recolor pass) — the hero
 * background now uses the primary→accent gradient token decoratively
 * (tasteful, not garish: both colors are dark/mid-tone, so white text stays
 * readable no matter where in the gradient it falls). The CTA button is
 * white-on-primary rather than accent, deliberately — an accent-colored
 * button risked visually blending into the accent end of the same gradient,
 * and a guaranteed-contrast light button is safer for a first-time,
 * possibly low-vision user than a button that could nearly disappear.
 */
// `behavior: "smooth"` here (rather than relying on the global scroll-smooth
// CSS via `behavior: "auto"`) is deliberate — calling scrollIntoView with an
// implicit/CSS-driven smooth behavior synchronously inside a click handler
// was found to silently no-op in testing. Explicit "instant" is the only
// behavior confirmed reliable in that exact context.
function scrollToSection(id: string) {
  document.getElementById(id)?.scrollIntoView({ behavior: "instant", block: "start" });
}

export default function HeroSection() {
  const { account, openLoginModal } = useApplication();

  return (
    <section className="relative flex min-h-svh flex-col items-center justify-center bg-[image:var(--linearPrimaryAccent)] px-6 py-16 text-center md:px-12 md:py-24">
      <div className="absolute right-6 top-6 md:right-12 md:top-8">
        {account ? (
          <Button
            component={Link}
            href="/account"
            variant="outlined"
            startIcon={<User className="h-4 w-4" />}
            sx={{
              borderColor: "rgba(255,255,255,0.4)",
              color: "white",
              "&:hover": { borderColor: "white" },
            }}
          >
            My Account
          </Button>
        ) : (
          <Button
            onClick={openLoginModal}
            variant="outlined"
            sx={{
              borderColor: "rgba(255,255,255,0.4)",
              color: "white",
              "&:hover": { borderColor: "white" },
            }}
          >
            Log In
          </Button>
        )}
      </div>
      <TricolorRule className="mx-auto mb-8" />
      <h1 className="text-4xl font-bold tracking-tight text-white md:text-6xl">
        Find your ESC school.
        <br />
        Apply with confidence.
      </h1>
      <p className="mx-auto mt-4 max-w-xl text-white/80">
        PAARAL is a platform built for the Educational Service Contracting
        (ESC) program of the Department of Education. It helps Grade 6
        learners find ESC-participating schools and apply for a subsidy,
        and shows where school slots remain insufficient.
      </p>
      <div className="mt-8 flex flex-col items-center gap-6">
        <div className="flex flex-col items-center gap-3 sm:flex-row sm:gap-4">
          <Button
            component={Link}
            href="/browse"
            variant="contained"
            size="large"
            sx={{
              bgcolor: "var(--background)",
              color: "var(--primary)",
              fontWeight: 700,
              "&:hover": { bgcolor: "var(--background)", opacity: 0.9 },
            }}
          >
            Browse Schools
          </Button>
          <Button
            onClick={() => scrollToSection("about-esc")}
            variant="outlined"
            size="large"
            sx={{
              borderColor: "rgba(255,255,255,0.4)",
              color: "white",
              "&:hover": { borderColor: "white" },
            }}
          >
            Know More About ESC
          </Button>
          <Button
            onClick={() => scrollToSection("about-paaral")}
            variant="outlined"
            size="large"
            sx={{
              borderColor: "rgba(255,255,255,0.4)",
              color: "white",
              "&:hover": { borderColor: "white" },
            }}
          >
            Know More About PAARAL
          </Button>
        </div>
        <DeadlinePanel />
      </div>
    </section>
  );
}
