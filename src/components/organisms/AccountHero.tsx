"use client";

import Link from "next/link";
import { ArrowLeft, BadgeCheck } from "lucide-react";
import ScrollReveal from "@/components/atoms/ScrollReveal";

function initialsFor(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return "?";
  const first = parts[0][0] ?? "";
  const last = parts[parts.length - 1][0] ?? "";
  return `${first}${last}`.toUpperCase();
}

/** Page-specific intro band for `/account` — distinct from `SiteHeader`'s
 * normal nav, which still renders above this. Adapted from the SchoolPath
 * reference's hero: a back link, a kicker label, the page heading, a short
 * description, and a "record matched" card naming the real PAARAL fields
 * (learner name + LRN) instead of SchoolPath's own mock student ID. */
export default function AccountHero({
  name,
  lrn,
}: {
  name: string;
  lrn: string;
}) {
  return (
    <section className="border-b border-slate-200 bg-white py-8 md:py-10">
      <div className="mx-auto max-w-6xl px-6 md:px-12">
        <Link
          href="/browse"
          className="inline-flex min-h-11 items-center gap-2 rounded-xl text-sm font-bold text-primary hover:underline"
        >
          <ArrowLeft className="h-4 w-4" /> Back to Browse
        </Link>

        <div className="mt-4 grid gap-5 lg:grid-cols-[minmax(0,1fr)_340px] lg:items-end">
          <div>
            <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500">
              Student Account
            </p>
            <h1 className="mt-2 text-3xl font-bold tracking-tight text-primary sm:text-4xl">
              {name}&apos;s Account
            </h1>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-600">
              This page shows your ESC subsidy application and your ranked
              school choices, all in one place.
            </p>
          </div>

          <ScrollReveal delay={0.05}>
            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4 shadow-sm">
              <div className="flex items-center gap-3">
                <div className="grid h-11 w-11 shrink-0 place-items-center rounded-full bg-primary text-sm font-bold text-white">
                  {initialsFor(name)}
                </div>
                <div className="min-w-0">
                  <p className="text-sm font-bold text-primary">
                    Student record matched
                  </p>
                  <p className="mt-0.5 truncate font-mono text-xs text-slate-500">
                    LRN {lrn}
                  </p>
                </div>
                <BadgeCheck
                  className="ml-auto h-6 w-6 shrink-0 text-primary"
                  aria-label="Verified"
                />
              </div>
            </div>
          </ScrollReveal>
        </div>
      </div>
    </section>
  );
}
