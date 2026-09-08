"use client";

import Link from "next/link";
import { ArrowRight, BadgeCheck, Compass } from "lucide-react";

interface BrowseHeroProps {
  loggedIn: boolean;
  wishlistCount: number;
}

/**
 * Intro band for `/browse` — adapted from the "schoolpath-portal" reference's
 * browse-page hero and its logged-in `SchoolFinder` intro.
 *
 * `/impeccable bolder` pass (2026-09-08): the plain-white/navy register this
 * band originally shipped with (matching `AccountHero`) read flat next to
 * the rest of the app's own "arrival" surfaces — the header already goes
 * fully solid Deep Civic Navy in its logged-in state, and the landing
 * hero/login modal both commit hard to color. Reused that exact
 * already-shipped "solid Navy band, white text" treatment here rather than
 * inventing a new one (no gradient, no pill buttons — those stay reserved
 * for the app's genuine marketing/arrival moments, not this Operate-mode
 * task page) — the status card becomes a real white card floating on that
 * navy, echoing the same "white card on a saturated surface" move the
 * login modal now uses.
 *
 * The right-side card is the one place this page visibly differs by login
 * state: a guest sees the existing "browse first, sign in later" reassurance;
 * a logged-in learner sees their real saved-schools count instead, since
 * that's the actual, more useful fact once an account exists.
 */
export default function BrowseHero({
  loggedIn,
  wishlistCount,
}: BrowseHeroProps) {
  return (
    <section className="bg-primary py-10 md:py-14">
      <div className="mx-auto max-w-6xl px-6 md:px-12">
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px] lg:items-end">
          <div>
            <p className="text-[10px] font-bold uppercase tracking-widest text-white/70">
              Find a School · Quezon City Pilot
            </p>
            <h1 className="mt-2 text-4xl font-extrabold tracking-tight text-white sm:text-5xl">
              Compare ESC-participating schools.
            </h1>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-white/80">
              Search and filter public and private junior high schools in
              Quezon City by fees, ESC subsidy, and available slots. Add
              schools to your ranked list as you go — you can save first and
              apply later.
            </p>
          </div>

          {loggedIn ? (
            <div className="rounded-2xl border border-slate-100 bg-white p-4 shadow-lg">
              <div className="flex items-center gap-3">
                <div className="grid h-11 w-11 shrink-0 place-items-center rounded-full bg-primary text-white">
                  <BadgeCheck className="h-5 w-5" />
                </div>
                <div className="min-w-0">
                  <p className="text-sm font-bold text-primary">
                    {wishlistCount === 0
                      ? "No schools saved yet"
                      : `${wishlistCount} school${wishlistCount === 1 ? "" : "s"} saved`}
                  </p>
                  <p className="mt-0.5 text-xs text-slate-500">
                    {wishlistCount === 0
                      ? "Use the heart button on any school to save it."
                      : "Rank and submit them from My Account."}
                  </p>
                </div>
              </div>
              {wishlistCount > 0 && (
                <Link
                  href="/account"
                  className="mt-3 inline-flex min-h-9 items-center gap-1.5 text-sm font-bold text-primary hover:underline"
                >
                  Go to My Account <ArrowRight className="h-4 w-4" />
                </Link>
              )}
            </div>
          ) : (
            <div className="rounded-2xl border border-slate-100 bg-white p-4 shadow-lg">
              <div className="flex items-center gap-3">
                <div className="grid h-11 w-11 shrink-0 place-items-center rounded-full bg-primary text-white">
                  <Compass className="h-5 w-5" />
                </div>
                <div className="min-w-0">
                  <p className="text-sm font-bold text-primary">
                    Browse without signing in.
                  </p>
                  <p className="mt-0.5 text-xs leading-5 text-slate-500">
                    Signing in is only needed to save schools and apply for
                    ESC support.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
