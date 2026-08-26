"use client";

import { createContext, useContext, useState } from "react";
import { useApplicationState } from "@/hooks/useApplicationState";
import { getQcSchools } from "@/lib/schools";
import LoginModal from "@/components/organisms/LoginModal";

// Loaded once at module scope — same static QC dataset every page already
// reads from (getQcSchools()), not re-fetched per route.
const schools = getQcSchools();

type ApplicationContextValue = ReturnType<typeof useApplicationState> & {
  loginModalOpen: boolean;
  openLoginModal: () => void;
  closeLoginModal: () => void;
};

const ApplicationContext = createContext<ApplicationContextValue | null>(
  null
);

/**
 * Mounted once at the root layout so account/wishlist/eligibility state
 * (and the login modal, which needs to overlay any page) survive
 * client-side navigation between routes — without this, each route calling
 * useApplicationState() separately would reset to a fresh, logged-out state
 * on every navigation.
 *
 * Note: there is no longer a global application drawer mounted here.
 * Identity, Status, Choices, Documents, and Survey all live together on the
 * single standalone `/account` page (`AccountPage.tsx`) — an earlier pass
 * put Status/Choices/Documents/Survey inline in a floating panel on
 * `/browse` instead, but Paula reviewed and rejected that (2026-08-24
 * correction, see memory-decisions.md): she wants a fully separate page,
 * not anything docked/overlaid on the map view. Only the login modal still
 * needs to overlay any page, since a guest can trigger it from several
 * different places.
 */
export function ApplicationStateProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const applicationState = useApplicationState(schools);
  const [loginModalOpen, setLoginModalOpen] = useState(false);

  const value: ApplicationContextValue = {
    ...applicationState,
    loginModalOpen,
    openLoginModal: () => setLoginModalOpen(true),
    closeLoginModal: () => setLoginModalOpen(false),
  };

  return (
    <ApplicationContext.Provider value={value}>
      {children}
      <LoginModal
        open={loginModalOpen}
        onClose={() => setLoginModalOpen(false)}
      />
    </ApplicationContext.Provider>
  );
}

export function useApplication() {
  const ctx = useContext(ApplicationContext);
  if (!ctx) {
    throw new Error(
      "useApplication must be used within an ApplicationStateProvider"
    );
  }
  return ctx;
}
