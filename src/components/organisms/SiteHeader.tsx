"use client";

import { useState } from "react";
import Image from "next/image";
import Link from "next/link";
import Button from "@mui/material/Button";
import { LogOut, Menu, X } from "lucide-react";
import { useApplication } from "@/components/templates/ApplicationStateProvider";

/**
 * Sticky top navbar (reworked 2026-09-07, per the "schoolpath-portal" Manus
 * reference's `PortalHeader.tsx`) — adopts its structure only, not its
 * colors: sticky + backdrop-blur, logo+wordmark on the left, real nav
 * links, an account control on the right, and a hamburger sheet on small
 * screens.
 *
 * Background pinned to the Primary→Accent gradient always (2026-09-08) —
 * originally this switched to a solid Navy bar once logged in (a
 * deliberate "your account state changed" signal, matching the reference's
 * own ivory-vs-navy split), but that's reverted for now per direct
 * instruction; revisit if a login-state-aware header treatment is wanted
 * again later.
 *
 * Not rendered on the landing page — HeroSection carries its own nav there.
 *
 * Nav parity pass (2026-09-08): this header's nav must show the exact same
 * items, in the same order, as the landing hero's own nav — Home / Browse
 * Schools / About PAARAL / About ESC / FAQs, kept in lockstep with
 * `HeroSection.tsx`'s `NAV_ITEMS`. "Browse Schools" is a real, always-
 * present item now (not swapped out on `/browse` — a self-link there is
 * an accepted tradeoff for the two navbars staying identical). "About
 * PAARAL"/"About ESC" are plain anchor links to `/`'s real sections
 * (`/#about-paaral`, `/#about-esc`); "FAQs" is a known no-op, matching the
 * hero's own copy — no FAQ page or section exists anywhere in the app yet.
 *
 * Typography/layout/logo-size unified with the hero's nav too (`/frontend-
 * design` pass, same date): `text-sm font-medium` pill-shaped nav links
 * (was an unconditional bold flat link), the hero's exact responsive logo
 * classes, and the hero's own header padding rhythm (`px-6 pt-5 pb-4
 * md:px-12 md:pt-6 md:pb-5`, intrinsic height) in place of a flat `h-[76px]`
 * box — deliberately NOT the hero's glassmorphism/blur/active-pill-slide
 * treatment, which stays a reserved, surface-scoped exception for the
 * hero/landing page (an Operate-mode page like `/browse` keeps this
 * header's plainer, flatter visual language; only the type/spacing/logo
 * sizing needed to match).
 *
 * Guest right-side "Browse Schools" button removed (2026-09-08) — it
 * duplicated the nav's own "Browse Schools" item once nav parity landed
 * above. Guests now just see "Log In" there; the nav is the one place
 * "Browse Schools" lives. Same fix applied to the logged-in right-side
 * "My Account" button — the nav already has one (conditionally, only
 * when logged in); the right side now just shows "Log Out". */
const NAV_ITEMS: { href: string; label: string }[] = [
  { href: "/", label: "Home" },
  { href: "/browse", label: "Browse Schools" },
  { href: "/#about-paaral", label: "About PAARAL" },
  { href: "/#about-esc", label: "About ESC" },
];

const navLinkSx = {
  color: "white",
  fontSize: "0.875rem",
  fontWeight: 500,
  textTransform: "none" as const,
  borderRadius: "999px",
  px: 2.5,
  py: 1,
  "&:hover": { bgcolor: "rgba(255,255,255,0.1)" },
};

export default function SiteHeader() {
  const { account, openLoginModal, logout } = useApplication();
  const [menuOpen, setMenuOpen] = useState(false);
  const loggedIn = Boolean(account);

  const closeMenu = () => setMenuOpen(false);

  return (
    <header className="sticky top-0 z-50 bg-[image:var(--linearPrimaryAccent)] backdrop-blur-md">
      <div className="flex items-center justify-between gap-3 px-6 pt-5 pb-4 md:px-12 md:pt-6 md:pb-5">
        <Link href="/" className="flex min-w-0 items-center gap-0">
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
        </Link>

        <nav
          className="hidden items-center gap-1 md:flex"
          aria-label="Primary navigation"
        >
          {NAV_ITEMS.map((item) => (
            <Button key={item.label} component={Link} href={item.href} sx={navLinkSx}>
              {item.label}
            </Button>
          ))}
          {/* FAQs: known no-op, matching the landing hero's own nav — no
              FAQ page/section exists anywhere in the app yet. */}
          <Button sx={navLinkSx}>FAQs</Button>
          {loggedIn && (
            <Button component={Link} href="/account" sx={navLinkSx}>
              My Account
            </Button>
          )}
        </nav>

        <div className="hidden items-center gap-3 sm:flex">
          {loggedIn ? (
            <Button
              onClick={logout}
              startIcon={<LogOut className="h-4 w-4" />}
              sx={navLinkSx}
            >
              Log Out
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

        <button
          onClick={() => setMenuOpen((open) => !open)}
          aria-label={menuOpen ? "Close menu" : "Open menu"}
          aria-expanded={menuOpen}
          className="grid h-11 w-11 shrink-0 place-items-center rounded-xl border border-white/30 text-white sm:hidden"
        >
          {menuOpen ? <X size={20} /> : <Menu size={22} />}
        </button>
      </div>

      {menuOpen && (
        <div className="flex flex-col gap-1 border-t border-white/15 px-6 py-3 sm:hidden">
          {NAV_ITEMS.map((item) => (
            <Link
              key={item.label}
              href={item.href}
              onClick={closeMenu}
              className="flex h-11 items-center rounded-lg px-3 text-sm font-semibold text-white hover:bg-white/10"
            >
              {item.label}
            </Link>
          ))}
          {/* FAQs: known no-op, same as the landing hero's own nav — see
              block comment above. */}
          <button className="flex h-11 items-center rounded-lg px-3 text-left text-sm font-semibold text-white hover:bg-white/10">
            FAQs
          </button>
          {loggedIn ? (
            <>
              <Link
                href="/account"
                onClick={closeMenu}
                className="flex h-11 items-center rounded-lg px-3 text-sm font-semibold text-white hover:bg-white/10"
              >
                My Account
              </Link>
              <button
                onClick={() => {
                  closeMenu();
                  logout();
                }}
                className="flex h-11 items-center gap-2 rounded-lg px-3 text-left text-sm font-semibold text-white hover:bg-white/10"
              >
                <LogOut size={17} /> Log Out
              </button>
            </>
          ) : (
            <button
              onClick={() => {
                closeMenu();
                openLoginModal();
              }}
              className="flex h-11 items-center rounded-lg px-3 text-left text-sm font-semibold text-white hover:bg-white/10"
            >
              Log In
            </button>
          )}
        </div>
      )}
    </header>
  );
}
