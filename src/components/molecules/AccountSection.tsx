"use client";

import type { ReactNode } from "react";
import Link from "next/link";
import ScrollReveal from "@/components/atoms/ScrollReveal";

interface AccountSectionAction {
  label: string;
  href?: string;
  onClick?: () => void;
}

interface AccountSectionProps {
  number: string; // "01", "02", ...
  eyebrow: string;
  title: string;
  action?: AccountSectionAction;
  children: ReactNode;
}

/** One numbered section in the account page's left column (the "path
 * marker" pattern from the SchoolPath reference — a small numbered badge,
 * an uppercase eyebrow, a heading, an optional single action, then
 * content). Replaces the old MUI `Tabs` shell in `ApplicationPanel` — all
 * sections are visible at once, in page order, rather than switched
 * behind tabs. Scroll-reveals as the user scrolls to it. */
export default function AccountSection({
  number,
  eyebrow,
  title,
  action,
  children,
}: AccountSectionProps) {
  return (
    <ScrollReveal>
      <section>
        <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
          <div className="flex items-start gap-3">
            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-primary/10 text-sm font-bold text-primary">
              {number}
            </span>
            <div>
              <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500">
                {eyebrow}
              </p>
              <h2 className="mt-1 text-xl font-bold text-primary">{title}</h2>
            </div>
          </div>
          {action &&
            (action.href ? (
              <Link
                href={action.href}
                className="inline-flex min-h-11 items-center gap-1.5 rounded-xl px-2 text-sm font-semibold text-primary hover:underline"
              >
                {action.label}
              </Link>
            ) : (
              <button
                type="button"
                onClick={action.onClick}
                className="inline-flex min-h-11 items-center gap-1.5 rounded-xl px-2 text-sm font-semibold text-primary hover:underline"
              >
                {action.label}
              </button>
            ))}
        </div>
        {children}
      </section>
    </ScrollReveal>
  );
}
