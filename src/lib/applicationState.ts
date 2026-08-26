import type { ApplicationState, EscSchoolStatus } from "@/types/application";

// ── APPLICATION STATE MACHINE ─────────────────────────────────────────
// Decoupled model: PAARAL tracks the ESC application track only. School
// admission/enrollment is an independent, unmodeled track — 'granted' means
// the ESC certificate is secured, full stop, regardless of enrollment timing.
export const POST_SUBMISSION_STATES = new Set<ApplicationState>([
  "submitted",
  "granted",
  "non_esc",
]);

export const VALID_TRANSITIONS: Record<ApplicationState, ApplicationState[]> = {
  eligibility: ["submitted"],
  not_eligible: ["non_esc"],
  submitted: ["granted", "non_esc", "eligibility"], // 'eligibility' = stop, choose different schools
  granted: [],
  non_esc: [],
};

// Per-school ESC status — private schools only. Public schools are never
// entered into the ESC pursuit; they're just the hasPublicAlternative
// guaranteed-placement checkbox. Up to ESC_SLATE_CAP schools can be in a
// non-terminal (slate) status at once — 'granted' is an offer, not a win,
// until explicitly redeemed via redeemChoice(). Redeeming one school moves
// every other still-open slate school to 'withdrawn'. 'rejected' is a
// school's own "no"; 'withdrawn' is the student pulling out after redeeming
// elsewhere — the two are never conflated in the UI.
export const ESC_SCHOOL_TRANSITIONS: Record<EscSchoolStatus, EscSchoolStatus[]> = {
  submitted: ["granted", "rejected", "docs_pending", "withdrawn"],
  docs_pending: ["docs_submitted", "withdrawn"],
  docs_submitted: ["granted", "rejected", "withdrawn"],
  granted: ["redeemed", "withdrawn"],
  rejected: [],
  redeemed: [],
  withdrawn: [],
};

// Statuses that count toward the ESC_SLATE_CAP — "currently in flight" at a
// school, including an unredeemed grant (still occupies a slate slot until
// the student acts on it).
export const ESC_SLATE_STATUSES = new Set<EscSchoolStatus>([
  "submitted",
  "docs_pending",
  "docs_submitted",
  "granted",
]);

export const ESC_SLATE_CAP = 3;

export const REJECTED_STATES = new Set<EscSchoolStatus>(["rejected"]);

export interface SchoolStatusMeta {
  title: string;
  desc: (schoolName: string) => string;
  color: string;
}

/** Per-school ESC status display metadata — the data half of the old
 * `schoolStatusConfigs` (icon/demo-button markup is a UI concern, added
 * when the status UI itself is built). */
export const SCHOOL_STATUS_META: Record<EscSchoolStatus, SchoolStatusMeta> = {
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
      `${name} has offered you an ESC subsidy. Redeem it to accept — this will withdraw your other active applications.`,
    color: "bg-purple-50 border-purple-200",
  },
  redeemed: {
    title: "ESC Certificate Redeemed",
    desc: (name) => `Your ESC subsidy for ${name} has been confirmed.`,
    color: "bg-green-50 border-green-200",
  },
  withdrawn: {
    title: "Application Withdrawn",
    desc: (name) =>
      `Your ESC application to ${name} was withdrawn after you redeemed a subsidy elsewhere.`,
    color: "bg-slate-50 border-slate-200",
  },
};

export interface StateBadgeMeta {
  tw: string;
  label: string;
}

/** Account-level status badge — data half of the old `renderStateBadge`.
 * Pre-submission states (eligibility/not_eligible) show no badge at all,
 * matching the original — there's no application yet. */
export function getStateBadge(state: ApplicationState): StateBadgeMeta | null {
  if (!POST_SUBMISSION_STATES.has(state)) return null;
  const map: Record<string, StateBadgeMeta> = {
    submitted: {
      tw: "bg-blue-100 text-blue-800 border-blue-300",
      label: "ESC Application In Progress",
    },
    granted: {
      tw: "bg-purple-50 text-purple-700 border-purple-200",
      label: "ESC Certificate Granted",
    },
    non_esc: {
      tw: "bg-slate-100 text-slate-600 border-slate-300",
      label: "Non-ESC Pathway",
    },
  };
  return map[state] ?? { tw: "bg-slate-100 text-slate-500 border-slate-200", label: state };
}
