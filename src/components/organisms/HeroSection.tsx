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
export default function HeroSection() {
  const { account, openLoginModal } = useApplication();

  return (
    <section className="relative bg-[image:var(--linearPrimaryAccent)] px-6 py-16 text-center md:px-12 md:py-24">
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
        Check your Educational Service Contracting (ESC) eligibility and
        browse participating schools — no fixed order required. Apply for
        ESC and school admission independently, at your own pace.
      </p>
      <div className="mt-8 flex flex-col items-center gap-6">
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
        <DeadlinePanel />
      </div>
    </section>
  );
}
