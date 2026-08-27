import { useState } from "react";
import type { School } from "@/types/school";
import type {
  Account,
  EligAnswers,
  EligHistoryEntry,
  EligStep,
  EscSchoolStatus,
  SurveyAnswers,
} from "@/types/application";
import { computeCategory, getDocList } from "@/lib/eligibility";
import {
  ESC_SCHOOL_TRANSITIONS,
  ESC_SLATE_CAP,
  ESC_SLATE_STATUSES,
  POST_SUBMISSION_STATES,
  VALID_TRANSITIONS,
} from "@/lib/applicationState";
import { apiPost } from "@/lib/api";
import {
  APPLICATION_STORAGE_KEY,
  LEARNER_RECORD,
  TEST_EMAIL,
  TEST_EMAIL_WITH_DRAFT,
  TEST_LRN,
} from "@/lib/constants";

const DEFAULT_ELIG_ANSWERS: EligAnswers = {
  escIntent: true,
  schoolType: null,
  segs: [],
  income: null,
  employment: null,
};

const DEFAULT_SURVEY_ANSWERS: SurveyAnswers = {
  ease: null,
  helpful: null,
  concern: null,
  suggestions: "",
};

export interface LoginLookupResult {
  ok: boolean;
  error?: string;
  lrn?: string;
  hasDraft?: boolean;
}

/** Real LRN verification, via paaral-student-api's
 * `/auth/verify-login-email` endpoint (Chunk 15) — replaces the old
 * hardcoded two-email mock. Same three outcomes as before (fresh
 * learner, learner with a draft wishlist, not-found), just backed by
 * a real Postgres lookup now. Only two demo LRNs actually exist in
 * the seeded dev database, so the demo hint is still added here on
 * the frontend rather than baked into the backend's (production-shaped)
 * error message. A network/server failure is reported as its own
 * distinct error rather than silently looking like "not found". */
export async function verifyLoginEmail(
  email: string
): Promise<LoginLookupResult> {
  let result: LoginLookupResult;
  try {
    result = await apiPost<LoginLookupResult>(
      "/api/v1/auth/verify-login-email",
      { email }
    );
  } catch {
    return {
      ok: false,
      error:
        "Couldn't reach the PAARAL server. Check your connection and try again.",
    };
  }

  if (!result.ok) {
    return {
      ok: false,
      error: `${result.error} Try ${TEST_EMAIL} or ${TEST_EMAIL_WITH_DRAFT} for this demo.`,
    };
  }
  return result;
}

/** Ported from src/App.jsx's v3 decoupled ESC application state machine —
 * unchanged in shape/logic, per SKILLS.md. UI (login modal, application
 * panel, questionnaire screens) is intentionally not part of this hook; it only
 * manages account/eligibility/wishlist/ESC-status state and the actions
 * that transition between them. */
export function useApplicationState(schools: School[]) {
  const [account, setAccountState] = useState<Account | null>(null);

  const updateAccount = (patch: Partial<Account>) => {
    setAccountState((prev) => {
      if (!prev) return prev;
      const next = { ...prev, ...patch };
      localStorage.setItem(APPLICATION_STORAGE_KEY, JSON.stringify(next));
      return next;
    });
  };

  // `wishlistIds` lets the login modal preload the LRN 100000000002 demo
  // draft ("Load Draft") — passing none (or "Start Fresh") creates a normal
  // empty-wishlist account for either demo LRN. Draft state is editable,
  // not locked, per CLAUDE.md.
  const createAccount = (lrn: string = TEST_LRN, wishlistIds: string[] = []) => {
    const newAccount: Account = {
      email: `${lrn}@deped.gov.ph`,
      lrn,
      name: `${LEARNER_RECORD.firstName} ${LEARNER_RECORD.mi}. ${LEARNER_RECORD.lastName}`,
      category: null,
      eligAnswers: null,
      applicationState: "eligibility",
      wishlistIds,
      escStatuses: {},
      surveyAnswers: DEFAULT_SURVEY_ANSWERS,
      uploadedDocs: [],
    };
    localStorage.setItem(APPLICATION_STORAGE_KEY, JSON.stringify(newAccount));
    setAccountState(newAccount);
  };

  // Mockup-only: logout wipes local account + questionnaire state so each
  // demo login starts clean. In production, account state is
  // server-persisted and survives logout — this reset should not carry
  // over (see memory-decisions.md).
  const logout = () => {
    localStorage.removeItem(APPLICATION_STORAGE_KEY);
    setAccountState(null);
    setEligStep("schoolType");
    setEligHistory([]);
    setEligAnswers(DEFAULT_ELIG_ANSWERS);
  };

  const applicationState = account?.applicationState ?? "eligibility";
  const isPostSubmission = POST_SUBMISSION_STATES.has(applicationState);

  const advance = (
    toState: Account["applicationState"],
    extra: Partial<Account> = {}
  ) => {
    const valid = VALID_TRANSITIONS[applicationState] ?? [];
    if (!valid.includes(toState)) return;
    updateAccount({ applicationState: toState, ...extra });
  };

  // ── ELIGIBILITY QUESTIONNAIRE ─────────────────────────────────
  const [eligStep, setEligStep] = useState<EligStep>("schoolType");
  const [eligHistory, setEligHistory] = useState<EligHistoryEntry[]>([]);
  const [eligAnswers, setEligAnswers] = useState<EligAnswers>(
    DEFAULT_ELIG_ANSWERS
  );

  // UI-convenience only, same reasoning as eligRestart below — the old
  // mockup could set eligAnswers directly (single-file, raw setter in
  // scope). This lets the SEG multi-select step patch answers in place
  // (toggling a checkbox) without pushing a history entry or changing
  // step, which eligGo always does — history should only grow on an
  // actual "Continue" between questions.
  const patchEligAnswers = (patch: Partial<EligAnswers>) => {
    setEligAnswers((a) => ({ ...a, ...patch }));
  };

  // UI-convenience only — resets the local questionnaire progress (not
  // account.applicationState) so "Start over" on the ineligible result can
  // return to the first question. Not part of the original ported logic;
  // added while building the questionnaire screen since the old mockup
  // had this same inline reset.
  const eligRestart = () => {
    setEligStep("schoolType");
    setEligHistory([]);
    setEligAnswers(DEFAULT_ELIG_ANSWERS);
  };

  const eligBack = () => {
    if (eligHistory.length === 0) return;
    const prev = eligHistory[eligHistory.length - 1];
    setEligStep(prev.step);
    setEligAnswers(prev.answers);
    setEligHistory((h) => h.slice(0, -1));
  };

  const eligGo = (step: EligStep, patch: Partial<EligAnswers> = {}) => {
    setEligHistory((h) => [...h, { step: eligStep, answers: { ...eligAnswers } }]);
    setEligAnswers((a) => ({ ...a, ...patch }));
    setEligStep(step);
  };

  const completeEligibility = () => {
    if (!account) return;
    const category = computeCategory(eligAnswers);
    updateAccount({
      category,
      eligAnswers,
      applicationState: category ? "eligibility" : "not_eligible",
    });
  };

  // ── WISHLIST ───────────────────────────────────────────────────
  const wishlist: School[] = account
    ? account.wishlistIds
        .map((id) => schools.find((s) => s.school_id === id))
        .filter((s): s is School => Boolean(s))
    : [];

  const isInWishlist = (schoolId: string) =>
    (account?.wishlistIds ?? []).includes(schoolId);

  const addToWishlist = (schoolId: string) => {
    if (!account || isPostSubmission) return;
    if (account.wishlistIds.includes(schoolId)) return;
    updateAccount({ wishlistIds: [...account.wishlistIds, schoolId] });
  };

  const removeFromWishlist = (schoolId: string) => {
    if (!account || isPostSubmission) return;
    updateAccount({
      wishlistIds: account.wishlistIds.filter((id) => id !== schoolId),
    });
  };

  // Drag-and-drop reordering (dnd-kit, touch + mouse). Only allowed
  // pre-submission — same gate as removeFromWishlist/addToWishlist, since
  // rank determines ESC application order (rank 1 gets applied to first).
  const reorderWishlist = (fromIndex: number, toIndex: number) => {
    if (!account || isPostSubmission) return;
    if (
      fromIndex === toIndex ||
      fromIndex < 0 ||
      toIndex < 0 ||
      fromIndex >= account.wishlistIds.length ||
      toIndex >= account.wishlistIds.length
    )
      return;
    const next = [...account.wishlistIds];
    const [moved] = next.splice(fromIndex, 1);
    next.splice(toIndex, 0, moved);
    updateAccount({ wishlistIds: next });
  };

  const hasPublicAlternative = wishlist.some((s) => s.school_type === "public");

  // ── PER-SCHOOL ESC STATUS (private schools only) ────────────────
  // Parallel/capped model: up to ESC_SLATE_CAP private schools can be "in
  // the slate" (a non-terminal status) at once. 'granted' is an offer, not
  // a win — redeemChoice() is the explicit convergence point that picks one
  // and withdraws the rest. See memory-decisions.md for the full design.
  const escStatuses = account?.escStatuses ?? {};
  const privateChoices = wishlist.filter((s) => s.school_type !== "public");
  const hasPrivateChoice = privateChoices.length > 0;
  const slateChoices = privateChoices.filter((s) =>
    ESC_SLATE_STATUSES.has(escStatuses[s.school_id])
  );
  const pendingGrants = privateChoices.filter(
    (s) => escStatuses[s.school_id] === "granted"
  );
  const redeemedChoice =
    privateChoices.find((s) => escStatuses[s.school_id] === "redeemed") ?? null;
  const rejectedChoices = privateChoices.filter(
    (s) => escStatuses[s.school_id] === "rejected"
  );
  // The next unengaged rank, only surfaced while there's slate room left.
  const firstUnengaged =
    privateChoices.find((s) => !escStatuses[s.school_id]) ?? null;
  const backfillCandidate =
    applicationState === "submitted" && slateChoices.length < ESC_SLATE_CAP
      ? firstUnengaged
      : null;
  // Every slate school resolved (rejected/withdrawn), nothing granted or
  // redeemed, and no one left to backfill with — the private track is dead.
  const isSlateExhausted =
    applicationState === "submitted" &&
    slateChoices.length === 0 &&
    !backfillCandidate;

  // Advance one specific PRIVATE school's ESC status. Only reaching
  // 'redeemed' ends the account-level pursuit — 'granted' is just an offer.
  const advanceSchool = (schoolId: string, toState: EscSchoolStatus) => {
    const current = escStatuses[schoolId];
    const valid = ESC_SCHOOL_TRANSITIONS[current] ?? [];
    if (!valid.includes(toState)) return;
    const nextEscStatuses = { ...escStatuses, [schoolId]: toState };
    if (toState === "redeemed") {
      advance("granted", { escStatuses: nextEscStatuses });
    } else {
      updateAccount({ escStatuses: nextEscStatuses });
    }
  };

  // Accept one school's ESC offer. This is the redemption/convergence
  // point: the chosen school is redeemed, and every other school still
  // occupying a slate slot (pending review or a competing offer) is
  // withdrawn — not rejected, since the school never said no.
  const redeemChoice = (schoolId: string) => {
    if (escStatuses[schoolId] !== "granted") return;
    const nextEscStatuses = { ...escStatuses };
    for (const choice of privateChoices) {
      const id = choice.school_id;
      if (id === schoolId) {
        nextEscStatuses[id] = "redeemed";
      } else if (ESC_SLATE_STATUSES.has(nextEscStatuses[id])) {
        nextEscStatuses[id] = "withdrawn";
      }
    }
    advance("granted", { escStatuses: nextEscStatuses });
  };

  // A slate slot opened up (a rejection) and there's room — explicit
  // opt-in, never automatic, matching the rest of this app's advance-choice
  // pattern.
  const backfillSlate = () => {
    if (!backfillCandidate) return;
    updateAccount({
      escStatuses: { ...escStatuses, [backfillCandidate.school_id]: "submitted" },
    });
  };

  const continueWithoutSubsidy = (schoolId: string) => {
    advance("non_esc", { nonEscSchoolId: schoolId });
  };

  const applyAgainDifferentSchool = () => {
    advance("eligibility", { wishlistIds: [], escStatuses: {} });
  };

  // ── DOCUMENTS ────────────────────────────────────────────────────
  const uploadedDocs = account?.uploadedDocs ?? [];
  const requiredDocs = account?.category
    ? getDocList(account.category, account.eligAnswers ?? DEFAULT_ELIG_ANSWERS)
    : [];
  const docsReady =
    requiredDocs.length > 0 && requiredDocs.every((d) => uploadedDocs.includes(d));

  const uploadDoc = (doc: string) => {
    if (!account || uploadedDocs.includes(doc)) return;
    updateAccount({ uploadedDocs: [...uploadedDocs, doc] });
  };

  const simulateAllUploads = () => {
    updateAccount({ uploadedDocs: requiredDocs });
  };

  // ── SURVEY ───────────────────────────────────────────────────────
  const [surveyAnswers, setSurveyAnswers] = useState<SurveyAnswers>(
    DEFAULT_SURVEY_ANSWERS
  );
  const generalSurveyComplete = Boolean(surveyAnswers.ease && surveyAnswers.helpful);
  const escSurveyComplete = Boolean(surveyAnswers.concern);

  // ── SUBMIT / ENROLL ───────────────────────────────────────────────
  const canSubmitEsc =
    applicationState === "eligibility" &&
    hasPrivateChoice &&
    hasPublicAlternative &&
    docsReady &&
    generalSurveyComplete &&
    escSurveyComplete;

  const canEnrollNonEsc =
    applicationState === "not_eligible" &&
    wishlist.length > 0 &&
    generalSurveyComplete;

  const handleSubmitEsc = () => {
    if (!canSubmitEsc) return;
    const initialSlate = privateChoices.slice(0, ESC_SLATE_CAP);
    const nextEscStatuses = { ...escStatuses };
    for (const s of initialSlate) nextEscStatuses[s.school_id] = "submitted";
    updateAccount({
      escStatuses: nextEscStatuses,
      surveyAnswers,
      uploadedDocs,
    });
    advance("submitted");
  };

  const handleEnrollNonEsc = () => {
    if (!canEnrollNonEsc) return;
    const school = wishlist[0];
    advance("non_esc", { nonEscSchoolId: school.school_id, surveyAnswers });
  };

  return {
    account,
    createAccount,
    logout,
    updateAccount,

    applicationState,
    isPostSubmission,
    advance,

    eligStep,
    eligHistory,
    eligAnswers,
    eligBack,
    eligGo,
    eligRestart,
    patchEligAnswers,
    completeEligibility,

    wishlist,
    isInWishlist,
    addToWishlist,
    removeFromWishlist,
    reorderWishlist,
    hasPublicAlternative,

    escStatuses,
    privateChoices,
    hasPrivateChoice,
    slateChoices,
    pendingGrants,
    redeemedChoice,
    rejectedChoices,
    backfillCandidate,
    isSlateExhausted,
    advanceSchool,
    redeemChoice,
    backfillSlate,
    continueWithoutSubsidy,
    applyAgainDifferentSchool,

    uploadedDocs,
    requiredDocs,
    docsReady,
    uploadDoc,
    simulateAllUploads,

    surveyAnswers,
    setSurveyAnswers,
    generalSurveyComplete,
    escSurveyComplete,

    canSubmitEsc,
    canEnrollNonEsc,
    handleSubmitEsc,
    handleEnrollNonEsc,
  };
}
