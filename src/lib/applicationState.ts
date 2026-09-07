import type { EscSchoolStatus } from "@/types/application";

// ── ESC APPLICATION STATE MACHINE ─────────────────────────────────────
// Real process, confirmed directly rather than assumed: a student submits
// ESC applications to up to MAX_ESC_APPLICATIONS schools in one action, but
// those get reviewed one at a time, in rank order — a lower-ranked choice is
// never looked at until the higher-ranked one is fully resolved. A student
// can never hold two live offers at once, so there's no "redeem one,
// withdraw the rest" step anymore — 'granted' leads to a single
// 'redeemed'/'declined' choice by the family, and rejecting/declining
// promotes the next 'queued' school (lowest rank) to 'submitted' — a hook
// action, not a server-side transition, matching every other state-machine
// decision in this app.
export const ESC_SCHOOL_TRANSITIONS: Record<EscSchoolStatus, EscSchoolStatus[]> = {
  queued: ["submitted"],
  submitted: ["granted", "rejected", "docs_pending"],
  docs_pending: ["docs_submitted"],
  docs_submitted: ["granted", "rejected"],
  granted: ["redeemed", "declined"],
  rejected: [],
  redeemed: [],
  declined: [],
};

// A resolution at one school (rejected, or the family declining an offer)
// is what triggers promoting the next queued school to submitted.
export const ADVANCE_TRIGGERING_STATES = new Set<EscSchoolStatus>([
  "rejected",
  "declined",
]);

export const TERMINAL_UNSUCCESSFUL_STATES = new Set<EscSchoolStatus>([
  "rejected",
  "declined",
]);

// Ranked preference list: 3–5 schools of any type (public, private-ESC,
// private-non-ESC) — must include at least one ESC-participating school,
// since that's the only kind PAARAL can actually submit an application to.
export const MIN_WISHLIST_SIZE = 3;
export const MAX_WISHLIST_SIZE = 5;

// Max ESC-participating schools selectable for submission, from within the
// ranked wishlist — an explicit student choice, not auto-derived from rank.
export const MAX_ESC_APPLICATIONS = 3;

export interface SchoolStatusMeta {
  title: string;
  desc: (schoolName: string) => string;
  color: string;
}

/** Per-school ESC status display metadata — the data half of the old
 * `schoolStatusConfigs` (icon/demo-button markup is a UI concern, added
 * when the status UI itself is built). */
export const SCHOOL_STATUS_META: Record<EscSchoolStatus, SchoolStatusMeta> = {
  queued: {
    title: "Waiting for Your Turn",
    desc: (name) =>
      `${name} will review your application once your higher-ranked choice has been decided.`,
    color: "bg-slate-50 border-slate-200",
  },
  submitted: {
    title: "ESC Application Submitted",
    desc: (name) =>
      `Your ESC application to ${name} has been received. You will be notified once it has been reviewed.`,
    color: "bg-blue-50 border-blue-200",
  },
  rejected: {
    title: "ESC Application Not Approved",
    desc: (name) => `Your ESC application to ${name} was not approved this cycle.`,
    color: "bg-red-50 border-red-200",
  },
  docs_pending: {
    title: "Additional Document Requested",
    desc: (name) =>
      `${name}'s ESC School Committee has requested an additional document. Please check the Documents tab.`,
    color: "bg-amber-50 border-amber-200",
  },
  docs_submitted: {
    title: "Additional Document Under Review",
    desc: (name) =>
      `Your documents for ${name} have been submitted and are being reviewed by the ESC School Committee.`,
    color: "bg-blue-50 border-blue-200",
  },
  granted: {
    title: "ESC Subsidy Offered",
    desc: (name) =>
      `${name} has offered you an ESC subsidy. Redeem it to accept, or decline if you no longer wish to enroll there.`,
    color: "bg-purple-50 border-purple-200",
  },
  redeemed: {
    title: "ESC Certificate Redeemed",
    desc: (name) => `Your ESC subsidy for ${name} has been confirmed.`,
    color: "bg-green-50 border-green-200",
  },
  declined: {
    title: "Offer Declined",
    desc: (name) => `You declined the ESC subsidy offered by ${name}.`,
    color: "bg-slate-50 border-slate-200",
  },
};

export interface StateBadgeMeta {
  tw: string;
  label: string;
}

/** Account-level status badge, derived from `isEligible` + the set of ESC
 * application statuses (there's no stored account-level status anymore —
 * see `app/db/models/application.py`'s docstring for why). No badge at
 * all before an eligibility result exists, or while eligible but not yet
 * submitted — matches the original "no badge until there's something to
 * report" rule. */
export function getAccountStatusBadge(account: {
  isEligible: boolean | null;
  escApplications: Record<string, { status: EscSchoolStatus }>;
}): StateBadgeMeta | null {
  if (account.isEligible === null) return null;
  if (account.isEligible === false) {
    return {
      tw: "bg-slate-100 text-slate-600 border-slate-300",
      label: "Not Eligible for ESC",
    };
  }

  const statuses = Object.values(account.escApplications).map((e) => e.status);
  if (statuses.length === 0) return null;
  if (statuses.includes("redeemed")) {
    return {
      tw: "bg-purple-50 text-purple-700 border-purple-200",
      label: "ESC Certificate Redeemed",
    };
  }
  if (statuses.every((s) => TERMINAL_UNSUCCESSFUL_STATES.has(s))) {
    return {
      tw: "bg-slate-100 text-slate-600 border-slate-300",
      label: "ESC Applications Unsuccessful",
    };
  }
  return {
    tw: "bg-blue-100 text-blue-800 border-blue-300",
    label: "ESC Application In Progress",
  };
}
