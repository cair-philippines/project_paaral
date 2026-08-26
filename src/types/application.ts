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

/** Per-private-school ESC lifecycle. Public schools never enter this —
 * they're only the `hasPublicAlternative` guaranteed-placement checkbox.
 * Up to `ESC_SLATE_CAP` private schools can be in a non-terminal status at
 * once (the "slate") — 'granted' is an offer, not a win, until the student
 * redeems it. Redeeming one school withdraws every other slate school. */
export type EscSchoolStatus =
  | "submitted"
  | "docs_pending"
  | "docs_submitted"
  | "granted"
  | "rejected"
  | "redeemed"
  | "withdrawn";

/** Account-level ESC application state. Decoupled model — this tracks the
 * ESC application track only; school admission/enrollment is a separate,
 * unmodeled track. 'granted' means the ESC certificate is secured, full
 * stop, regardless of enrollment timing. */
export type ApplicationState =
  | "eligibility"
  | "not_eligible"
  | "submitted"
  | "granted"
  | "non_esc";

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
  applicationState: ApplicationState;
  wishlistIds: string[];
  escStatuses: Record<string, EscSchoolStatus>;
  surveyAnswers: SurveyAnswers;
  uploadedDocs: string[];
  nonEscSchoolId?: string;
}
