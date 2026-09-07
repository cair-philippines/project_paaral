export type EscCategory = "A" | "B" | "C" | "D" | null;

/** How the learner completed Grade 6 — distinct from `School.school_type`
 * (the school's own public/private classification). */
export type Grade6Pathway = "public" | "private" | "als";

export type Seg = "4ps" | "gidca" | "ip" | "pwd" | "special" | "cbms" | "none";

export type IncomeBracket =
  | "poor"
  | "low"
  | "lower_middle"
  | "middle"
  | "above";

export type EmploymentStatus =
  | "local"
  | "abroad"
  | "business"
  | "unemployed";

export interface EligAnswers {
  escIntent: boolean;
  schoolType: Grade6Pathway | null;
  segs: Seg[];
  income: IncomeBracket | null;
  employment: EmploymentStatus | null;
}

export type EligStep =
  | "schoolType"
  | "seg"
  | "income"
  | "employment"
  | "result";

export interface EligHistoryEntry {
  step: EligStep;
  answers: EligAnswers;
}

/** Per-school ESC application lifecycle — only for schools explicitly
 * selected for submission (see `MAX_ESC_APPLICATIONS`), a separate,
 * smaller set than the ranked `wishlistIds`. Real process, confirmed
 * directly: a student submits to up to `MAX_ESC_APPLICATIONS` schools in
 * one action, but those get reviewed one at a time, in rank order — a
 * lower-ranked choice is never looked at until the higher-ranked one
 * fully resolves. `queued` means "submitted, but a higher-ranked choice
 * hasn't resolved yet" — distinct from `submitted` ("actively pending a
 * verdict"). A student can never hold two live offers at once, so
 * there's no "redeem one, withdraw the rest" step anymore — `granted`
 * leads to a single `redeemed`/`declined` choice by the family. */
export type EscSchoolStatus =
  | "queued"
  | "submitted"
  | "docs_pending"
  | "docs_submitted"
  | "granted"
  | "rejected"
  | "redeemed"
  | "declined";

export interface EscApplicationEntry {
  status: EscSchoolStatus;
  submittedAt: string;
  resolvedAt: string | null;
}

export interface SurveyAnswers {
  ease: number | null;
  helpful: string | null;
  concern: string | null;
  suggestions: string;
}

export interface Account {
  email: string;
  lrn: string;
  name: string;
  category: EscCategory;
  eligAnswers: EligAnswers | null;
  /** Mirrors the backend's `Application.is_eligible`: `null` until the
   * eligibility assessment is completed, `true`/`false` after —
   * disambiguates "not yet assessed" from "assessed, ineligible" more
   * directly than inferring it from `eligAnswers`/`category` alone. */
  isEligible: boolean | null;
  /** The full ranked preference list (3–5 schools, any type — public,
   * private-ESC, private-non-ESC). Separate from `escApplications`,
   * which schools were actually submitted to. */
  wishlistIds: string[];
  /** The up-to-`MAX_ESC_APPLICATIONS` ESC schools actually applied to,
   * keyed by school_id — a subset of `wishlistIds`, explicitly chosen,
   * not auto-derived from rank order. */
  escApplications: Record<string, EscApplicationEntry>;
  surveyAnswers: SurveyAnswers;
  uploadedDocs: string[];
}
