import type {
  ApplicationState,
  EligAnswers,
  EscCategory,
  EscSchoolStatus,
  Grade6Pathway,
  Seg,
  SurveyAnswers,
} from "@/types/application";
import { apiGet, apiPatch, apiPut } from "@/lib/api";

/** One wishlist entry as the API expects/returns it - camelCase,
 * matching `paaral-student-api`'s `CamelModel` convention. */
export interface ApiWishlistEntry {
  schoolId: string;
  rank?: number;
  escStatus: EscSchoolStatus | null;
}

/** Replace a learner's entire wishlist, in rank order.
 *
 * Covers every wishlist/per-school-ESC-status mutation
 * `useApplicationState` makes (add, remove, reorder, redeem,
 * backfill, submit, reject) - they all produce a new ordered
 * snapshot, which this replaces wholesale (Chunk 17). */
export function replaceWishlist(
  lrn: string,
  wishlistIds: string[],
  escStatuses: Record<string, EscSchoolStatus>
): Promise<ApiWishlistEntry[]> {
  return apiPut<ApiWishlistEntry[]>(`/api/v1/applications/${lrn}/wishlist`, {
    schools: wishlistIds.map((schoolId) => ({
      schoolId,
      escStatus: escStatuses[schoolId] ?? null,
    })),
  });
}

export function getWishlist(lrn: string): Promise<ApiWishlistEntry[]> {
  return apiGet<ApiWishlistEntry[]>(`/api/v1/applications/${lrn}/wishlist`);
}

interface ApplicationStatusResponse {
  lrn: string;
  status: ApplicationState;
  nonEscSchoolId: string | null;
}

/** Update a learner's account-level application status. */
export function updateApplicationStatus(
  lrn: string,
  status: ApplicationState,
  nonEscSchoolId: string | null = null
): Promise<ApplicationStatusResponse> {
  return apiPatch<ApplicationStatusResponse>(`/api/v1/applications/${lrn}`, {
    status,
    nonEscSchoolId,
  });
}

// The frontend's survey UI stores the display label itself as the
// answer value (e.g. "School quality"), not the backend's lowercase
// enum value ("quality") - translated here, at the one API boundary,
// rather than changing the UI's long-established value convention.
const HELPFUL_TO_ENUM: Record<string, string> = {
  Yes: "yes",
  Somewhat: "somewhat",
  No: "no",
};

const CONCERN_TO_ENUM: Record<string, string> = {
  Cost: "cost",
  Distance: "distance",
  "School quality": "quality",
  "Slot availability": "slot_availability",
};

// Reverse of the two maps above - used when hydrating a saved survey
// response (Chunk 22) back into the display labels the survey UI
// actually renders/compares against.
const ENUM_TO_HELPFUL: Record<string, string> = Object.fromEntries(
  Object.entries(HELPFUL_TO_ENUM).map(([label, value]) => [value, label])
);

const ENUM_TO_CONCERN: Record<string, string> = Object.fromEntries(
  Object.entries(CONCERN_TO_ENUM).map(([label, value]) => [value, label])
);

interface SurveyResponseOut {
  ease: number;
  helpful: string;
  concern: string | null;
  suggestions: string | null;
}

/** Submit (create or replace) a learner's survey response.
 *
 * `answers.concern` is null for a `not_eligible` learner - the
 * ESC-specific question doesn't apply to that track. */
export function submitSurvey(
  lrn: string,
  answers: SurveyAnswers
): Promise<SurveyResponseOut> {
  return apiPut<SurveyResponseOut>(`/api/v1/applications/${lrn}/survey`, {
    ease: answers.ease,
    helpful: answers.helpful ? HELPFUL_TO_ENUM[answers.helpful] : null,
    concern: answers.concern ? CONCERN_TO_ENUM[answers.concern] : null,
    suggestions: answers.suggestions || null,
  });
}

interface EligibilityAssessmentResponse {
  escIntent: boolean;
  schoolType: string;
  segs: string[];
  income: string | null;
  employment: string | null;
  category: EscCategory;
  assessedAt: string;
}

/** Submit (create or replace) a learner's eligibility questionnaire
 * result (Chunk 19).
 *
 * `answers.segs` may include the frontend-only `"none"` sentinel (its
 * "none of these apply" option) - filtered out here, since the
 * backend's `Seg` enum has no such member (absence of rows already
 * means "no SEG"). `answers.income`/`answers.employment` are commonly
 * null - the questionnaire skips both for a Category A (SEG) result,
 * and skips just `employment` for the "above income, no SEG"
 * ineligible outcome. */
export function submitEligibilityAssessment(
  lrn: string,
  answers: EligAnswers,
  category: EscCategory
): Promise<EligibilityAssessmentResponse> {
  return apiPut<EligibilityAssessmentResponse>(
    `/api/v1/applications/${lrn}/eligibility`,
    {
      escIntent: answers.escIntent,
      schoolType: answers.schoolType,
      segs: answers.segs.filter((s) => s !== "none"),
      income: answers.income,
      employment: answers.employment,
      category,
    }
  );
}

interface ApiDocumentUpload {
  documentType: string;
  fileUrl: string;
  uploadedAt: string;
}

interface ApiApplicationState {
  lrn: string;
  status: ApplicationState;
  nonEscSchoolId: string | null;
  wishlist: ApiWishlistEntry[];
  eligibility: EligibilityAssessmentResponse | null;
  survey: SurveyResponseOut | null;
  documents: ApiDocumentUpload[];
}

/** Everything a login needs to restore a returning student's account -
 * the frontend-shaped equivalent of `ApiApplicationState`, after
 * translating enum values back to the display labels/field shapes
 * the rest of the app already expects. */
export interface HydratedAccountState {
  applicationState: ApplicationState;
  nonEscSchoolId?: string;
  wishlistIds: string[];
  escStatuses: Record<string, EscSchoolStatus>;
  category: EscCategory;
  eligAnswers: EligAnswers | null;
  surveyAnswers: SurveyAnswers;
  uploadedDocs: string[];
}

/** Fetch and translate a learner's complete saved application state
 * (Chunk 22) - called on login instead of always starting from empty
 * defaults, since wishlist/eligibility/survey/documents all already
 * persist correctly but were never fetched back. A brand-new account
 * with nothing saved yet comes back with the same empty/default shape
 * `createAccount()` used to hardcode - not an error case. */
export async function getApplicationState(
  lrn: string
): Promise<HydratedAccountState> {
  const state = await apiGet<ApiApplicationState>(`/api/v1/applications/${lrn}`);

  const escStatuses: Record<string, EscSchoolStatus> = {};
  for (const entry of state.wishlist) {
    if (entry.escStatus) escStatuses[entry.schoolId] = entry.escStatus;
  }

  const eligAnswers: EligAnswers | null = state.eligibility
    ? {
        escIntent: state.eligibility.escIntent,
        schoolType: state.eligibility.schoolType as Grade6Pathway,
        segs: state.eligibility.segs as Seg[],
        income: state.eligibility.income as EligAnswers["income"],
        employment: state.eligibility.employment as EligAnswers["employment"],
      }
    : null;

  return {
    applicationState: state.status,
    nonEscSchoolId: state.nonEscSchoolId ?? undefined,
    wishlistIds: state.wishlist.map((entry) => entry.schoolId),
    escStatuses,
    category: state.eligibility?.category ?? null,
    eligAnswers,
    surveyAnswers: state.survey
      ? {
          ease: state.survey.ease,
          helpful: ENUM_TO_HELPFUL[state.survey.helpful] ?? null,
          concern: state.survey.concern
            ? (ENUM_TO_CONCERN[state.survey.concern] ?? null)
            : null,
          suggestions: state.survey.suggestions ?? "",
        }
      : { ease: null, helpful: null, concern: null, suggestions: "" },
    uploadedDocs: state.documents.map((doc) => doc.documentType),
  };
}
