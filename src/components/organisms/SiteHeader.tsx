"use client";

import { useState } from "react";
import Image from "next/image";
import Link from "next/link";
import Button from "@mui/material/Button";
import { LogOut, Menu, User, X } from "lucide-react";
import { useApplication } from "@/components/templates/ApplicationStateProvider";

/**
 * Sticky top navbar (reworked 2026-09-07, per the "schoolpath-portal" Manus
 * reference's `PortalHeader.tsx`) — adopts its structure only, not its
 * colors: sticky + backdrop-blur, a background that visibly differs by
 * login state (guest keeps the existing primary→accent gradient; logged-in
 * switches to a solid dark `primary` bar, echoing the reference's
 * ivory-vs-navy split with PAARAL's own tokens), logo+wordmark on the left,
 * real nav links (only ones with an actual destination — no fabricated
 * "How it works"/"Help" pages), an account control on the right, and a
 * hamburger sheet on small screens (previously the header had no dedicated
 * mobile treatment at all).
 *
 * Not rendered on the landing page — HeroSection carries its own floating
 * Log In control there (2026-08-24 decision).
 */
const navLinkSx = {
  color: "white",
  fontWeight: 700,
  textTransform: "none" as const,
  "&:hover": { bgcolor: "rgba(255,255,255,0.1)" },
};

export default function SiteHeader() {
  const { account, openLoginModal, logout } = useApplication();
  const [menuOpen, setMenuOpen] = useState(false);
  const loggedIn = Boolean(account);

  const closeMenu = () => setMenuOpen(false);

  return (
    <header
      className={`sticky top-0 z-50 backdrop-blur-md ${
        loggedIn ? "bg-primary/95" : "bg-[image:var(--linearPrimaryAccent)]"
      }`}
    >
      <div className="flex h-[76px] items-center justify-between gap-4 px-6 md:px-12">
        <Link href="/" className="flex min-w-0 items-center gap-4">
          <Image
            src="/assets/deped-logo.png"
            alt="DepEd"
            width={32}
            height={32}
            className="h-8 w-auto"
          />
          <Image
            src="/assets/ecair-logo.png"
            alt="ECAIR"
            width={20}
            height={20}
            className="h-5 w-auto"
          />
          <span className="text-xl font-bold tracking-tight text-white">
            PAARAL
          </span>
        </Link>

        <nav
          className="hidden items-center gap-1 md:flex"
          aria-label="Primary navigation"
        >
          <Button component={Link} href="/browse" sx={navLinkSx}>
            Browse Schools
          </Button>
          {loggedIn && (
            <Button component={Link} href="/account" sx={navLinkSx}>
              My Account
            </Button>
          )}
        </nav>

        <div className="hidden items-center gap-3 sm:flex">
          {loggedIn ? (
            <>
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
              <Button
                onClick={logout}
                startIcon={<LogOut className="h-4 w-4" />}
                sx={navLinkSx}
              >
                Log Out
              </Button>
            </>
          ) : (
            <>
              <Button
                component={Link}
                href="/browse"
                variant="contained"
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
            </>
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
          <Link
            href="/browse"
            onClick={closeMenu}
            className="flex h-11 items-center rounded-lg px-3 text-sm font-semibold text-white hover:bg-white/10"
          >
            Browse Schools
          </Link>
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
