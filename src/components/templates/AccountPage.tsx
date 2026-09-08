"use client";

import Button from "@mui/material/Button";
import { User } from "lucide-react";
import SiteHeader from "@/components/organisms/SiteHeader";
import AccountHero from "@/components/organisms/AccountHero";
import AccountJourneyStrip from "@/components/organisms/AccountJourneyStrip";
import AccountSidebar from "@/components/organisms/AccountSidebar";
import ApplicationPanel from "@/components/organisms/ApplicationPanel";
import StudentRecordSection from "@/components/organisms/StudentRecordSection";
import FamilyRecordSection from "@/components/organisms/FamilyRecordSection";
import { useApplication } from "@/components/templates/ApplicationStateProvider";

/**
 * The single, standalone "My Account" page — a page-specific hero band
 * (name + LRN "record matched" card), a 3-step journey strip, and a
 * two-column workspace (numbered application sections on the left, a
 * sticky account-summary sidebar on the right).
 *
 * Restructured 2026-08-24, replicating the layout (not the palette or
 * content) of a manus.ai "SchoolPath" reference site Paula pointed to —
 * see `.claude/rules/memory-decisions.md` and `LOG.md`'s 2026-08-24 entry
 * for the section-by-section mapping from SchoolPath's content onto
 * PAARAL's own data model. `useApplicationState` itself is untouched;
 * this is a presentation/layout change only.
 *
 * `StudentRecordSection`/`FamilyRecordSection` added (2026-09-08) — the
 * SchoolPath reference's own "official student record" and "family
 * contacts" sections, which this project had left out until now since
 * neither has real backing data (see each component's own doc comment
 * for exactly which fields are a schema gap vs. a plumbing gap). Numbered
 * 01/02, with `ApplicationPanel`'s own sections continuing the same
 * sequence via `startIndex={2}` rather than restarting at 01.
 */
export default function AccountPage() {
  const {
    account,
    logout,
    isPostSubmission,
    wishlist,
    redeemedChoice,
    allEscApplicationsUnsuccessful,
    openLoginModal,
  } = useApplication();

  return (
    <div className="min-h-screen bg-background">
      <SiteHeader />

      {!account ? (
        <div className="flex flex-col items-center px-4 py-24 text-center">
          <User className="mb-4 h-10 w-10 text-slate-300" />
          <p className="mb-4 text-sm text-slate-500">
            Log in to see your account.
          </p>
          <Button variant="contained" sx={{ minHeight: 48 }} onClick={openLoginModal}>
            Log In
          </Button>
        </div>
      ) : (
        <>
          <AccountHero name={account.name} lrn={account.lrn} />
          <AccountJourneyStrip
            category={account.category}
            isEligible={account.isEligible}
            isPostSubmission={isPostSubmission}
            wishlistCount={wishlist.length}
            hasRedeemed={redeemedChoice !== null}
            allEscApplicationsUnsuccessful={allEscApplicationsUnsuccessful}
          />
          <div className="mx-auto grid max-w-6xl gap-8 px-6 py-10 md:px-12 lg:grid-cols-[minmax(0,1fr)_320px] lg:py-14">
            <div className="space-y-10">
              <StudentRecordSection account={account} />
              <FamilyRecordSection />
              <ApplicationPanel startIndex={2} />
            </div>
            <AccountSidebar account={account} onLogout={logout} />
          </div>
        </>
      )}
    </div>
  );
}
