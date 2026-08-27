"use client";

import Button from "@mui/material/Button";
import { User } from "lucide-react";
import SiteHeader from "@/components/organisms/SiteHeader";
import AccountHero from "@/components/organisms/AccountHero";
import AccountJourneyStrip from "@/components/organisms/AccountJourneyStrip";
import AccountSidebar from "@/components/organisms/AccountSidebar";
import ApplicationPanel from "@/components/organisms/ApplicationPanel";
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
 */
export default function AccountPage() {
  const { account, logout, applicationState, isPostSubmission, wishlist, openLoginModal } =
    useApplication();

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
            applicationState={applicationState}
            isPostSubmission={isPostSubmission}
            wishlistCount={wishlist.length}
          />
          <div className="mx-auto grid max-w-6xl gap-8 px-6 py-10 md:px-12 lg:grid-cols-[minmax(0,1fr)_320px] lg:py-14">
            <ApplicationPanel />
            <AccountSidebar account={account} onLogout={logout} />
          </div>
        </>
      )}
    </div>
  );
}
