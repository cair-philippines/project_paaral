import type {
  ApplicationState,
  EscSchoolStatus,
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
