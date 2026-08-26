"use client";

import { CircleHelp, LogOut } from "lucide-react";
import ScrollReveal from "@/components/atoms/ScrollReveal";
import { catMeta } from "@/lib/eligibility";
import { getStateBadge } from "@/lib/applicationState";
import type { Account } from "@/types/application";

function SidebarFact({
  label,
  value,
  tw,
  mono,
}: {
  label: string;
  value: string;
  tw?: string;
  mono?: boolean;
}) {
  return (
    <div>
      <p className="text-[10px] font-bold uppercase tracking-widest text-slate-400">
        {label}
      </p>
      {tw ? (
        <span
          className={`mt-1 inline-block rounded border px-2 py-0.5 text-xs font-bold uppercase ${tw}`}
        >
          {value}
        </span>
      ) : (
        <p
          className={`mt-1 text-sm font-semibold text-slate-800 ${mono ? "font-mono" : ""}`}
        >
          {value}
        </p>
      )}
    </div>
  );
}

/** Sticky right-column sidebar for `/account` — an account-summary card
 * (key facts, same data already shown in the hero/journey strip, just
 * scannable in one place) and a plain-language "what happens next" card.
 * No fabricated contact info (no phone number exists elsewhere in this
 * app) — unlike the SchoolPath reference's "call a guide" card. */
export default function AccountSidebar({
  account,
  onLogout,
}: {
  account: Account;
  onLogout: () => void;
}) {
  const badge = getStateBadge(account.applicationState);

  return (
    <aside className="space-y-4 lg:sticky lg:top-24 lg:self-start">
      <ScrollReveal>
        <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500">
            Account Summary
          </p>
          <div className="mt-4 space-y-4 border-t border-slate-100 pt-4">
            <SidebarFact
              label="Learner Reference Number (LRN)"
              value={account.lrn}
              mono
            />
            <SidebarFact
              label="ESC Category"
              value={
                account.category
                  ? `Cat. ${account.category}`
                  : "Not yet determined"
              }
              tw={account.category ? catMeta[account.category].tw : undefined}
            />
            <SidebarFact
              label="Application Status"
              value={badge ? badge.label : "Not yet submitted"}
              tw={badge?.tw}
            />
          </div>
          <button
            type="button"
            onClick={onLogout}
            className="mt-5 flex min-h-11 w-full items-center justify-center gap-1.5 rounded-xl border border-slate-200 text-xs font-semibold text-slate-500 hover:bg-slate-50 hover:text-slate-700"
          >
            <LogOut className="h-3.5 w-3.5" /> Log Out
          </button>
        </section>
      </ScrollReveal>

      <ScrollReveal delay={0.05}>
        <section className="rounded-2xl border border-slate-200 bg-slate-50 p-5">
          <div className="grid h-10 w-10 place-items-center rounded-xl bg-primary text-white">
            <CircleHelp className="h-5 w-5" />
          </div>
          <h2 className="mt-4 text-lg font-bold text-primary">
            What happens next?
          </h2>
          <p className="mt-2 text-sm leading-6 text-slate-600">
            Every update to your ESC application appears on this page —
            there&apos;s nothing else you need to check. If an application
            is not approved, we&apos;ll always tell you exactly what you can
            do next.
          </p>
        </section>
      </ScrollReveal>
    </aside>
  );
}
