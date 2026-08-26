"use client";

import JourneyStep from "@/components/molecules/JourneyStep";
import type { ApplicationState, EscCategory } from "@/types/application";

interface AccountJourneyStripProps {
  category: EscCategory;
  applicationState: ApplicationState;
  isPostSubmission: boolean;
  wishlistCount: number;
}

/**
 * The 3-step "your journey so far" strip below the hero band, on its own
 * tinted background — adapted from the SchoolPath reference's journey
 * strip onto PAARAL's actual pipeline: eligibility check, then building a
 * ranked wishlist, then the submitted ESC application itself. Exactly one
 * step is ever "active" (the phase the learner is currently in).
 */
export default function AccountJourneyStrip({
  category,
  applicationState,
  isPostSubmission,
  wishlistCount,
}: AccountJourneyStripProps) {
  const step1Active =
    !category && applicationState === "eligibility" && !isPostSubmission;
  const step2Active = !step1Active && !isPostSubmission;
  const step3Active = isPostSubmission;

  const step1Detail = category
    ? `Category ${category} determined`
    : applicationState === "not_eligible"
      ? "Not eligible for the ESC subsidy"
      : "Not yet completed";

  const step2Detail =
    wishlistCount > 0
      ? `${wishlistCount} school${wishlistCount === 1 ? "" : "s"} saved`
      : "No schools saved yet";

  const step3Detail =
    applicationState === "granted"
      ? "ESC certificate granted"
      : applicationState === "non_esc"
        ? "Enrolling without a subsidy"
        : applicationState === "submitted"
          ? "Under review"
          : "Not yet submitted";

  return (
    <section className="border-b border-slate-200 bg-slate-50">
      <div className="mx-auto grid max-w-6xl px-6 md:grid-cols-3 md:px-12">
        <JourneyStep
          number="01"
          title="Your Eligibility"
          detail={step1Detail}
          active={step1Active}
        />
        <JourneyStep
          number="02"
          title="Your Choices"
          detail={step2Detail}
          active={step2Active}
        />
        <JourneyStep
          number="03"
          title="Your Application"
          detail={step3Detail}
          active={step3Active}
        />
      </div>
    </section>
  );
}
