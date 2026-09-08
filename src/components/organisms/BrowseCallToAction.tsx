"use client";

import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { useApplication } from "@/components/templates/ApplicationStateProvider";

/**
 * Closing banner for `/browse` — adapted from the "schoolpath-portal"
 * reference's "Ready to keep your choices?" band, but with two real
 * variants instead of one, since the point of putting this in `/browse`
 * (not just the guest-only reference page) was to give the logged-in
 * browsing experience its own payoff too: a guest is nudged to sign in,
 * a learner with schools already saved is nudged toward `/account` to
 * rank and submit — the actual next step for each, in plain language.
 */
export default function BrowseCallToAction({
  wishlistCount,
}: {
  wishlistCount: number;
}) {
  const { account, openLoginModal } = useApplication();

  if (!account) {
    return (
      <section className="border-t border-slate-200 bg-slate-50 py-10">
        <div className="mx-auto flex max-w-6xl flex-col justify-between gap-5 px-6 md:flex-row md:items-center md:px-12">
          <div>
            <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500">
              Ready to keep your choices?
            </p>
            <h2 className="mt-2 text-2xl font-bold tracking-tight text-primary">
              Sign in to save and rank schools.
            </h2>
            <p className="mt-2 max-w-xl text-sm leading-6 text-slate-600">
              Your saved list stays separate from your ESC application until
              you decide to submit it.
            </p>
          </div>
          <button
            type="button"
            onClick={openLoginModal}
            className="inline-flex min-h-12 shrink-0 items-center justify-center gap-2 rounded-xl bg-primary px-5 text-sm font-bold text-white transition hover:opacity-90"
          >
            Sign in to save choices <ArrowRight className="h-4 w-4" />
          </button>
        </div>
      </section>
    );
  }

  return (
    <section className="border-t border-slate-200 bg-slate-50 py-10">
      <div className="mx-auto flex max-w-6xl flex-col justify-between gap-5 px-6 md:flex-row md:items-center md:px-12">
        <div>
          <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500">
            Your saved schools
          </p>
          <h2 className="mt-2 text-2xl font-bold tracking-tight text-primary">
            {wishlistCount === 0
              ? "Save a few schools to get started."
              : `You have ${wishlistCount} school${wishlistCount === 1 ? "" : "s"} saved so far.`}
          </h2>
          <p className="mt-2 max-w-xl text-sm leading-6 text-slate-600">
            {wishlistCount === 0
              ? "Use the heart button on any school card to add it to your ranked list."
              : "Rank your list in order of preference and submit when you're ready."}
          </p>
        </div>
        <Link
          href="/account"
          className="inline-flex min-h-12 shrink-0 items-center justify-center gap-2 rounded-xl bg-primary px-5 text-sm font-bold text-white transition hover:opacity-90"
        >
          Go to My Account <ArrowRight className="h-4 w-4" />
        </Link>
      </div>
    </section>
  );
}
