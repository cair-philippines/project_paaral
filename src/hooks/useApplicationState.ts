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
  POST_SUBMISSION_STATES,
  VALID_TRANSITIONS,
} from "@/lib/applicationState";
import {
  APPLICATION_STORAGE_KEY,
  LEARNER_RECORD,
  TEST_EMAIL,
  TEST_EMAIL_WITH_DRAFT,
  TEST_LRN,
  TEST_LRN_WITH_DRAFT,
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

/** Demo stand-in for a DepEd LIS lookup by email. Three outcomes, matching
 * CLAUDE.md's "Demo Credentials" table: a fresh valid learner, a valid
 * learner with a pending draft wishlist, or not-found (anything else). */
export function validateLoginEmail(email: string): LoginLookupResult {
  const normalized = email.trim().toLowerCase();
  if (normalized === TEST_EMAIL.toLowerCase()) {
    return { ok: true, lrn: TEST_LRN, hasDraft: false };
  }
  if (normalized === TEST_EMAIL_WITH_DRAFT.toLowerCase()) {
    return { ok: true, lrn: TEST_LRN_WITH_DRAFT, hasDraft: true };
  }
  return {
    ok: false,
    error: `Email not found in the DepEd Learner Information System. Try ${TEST_EMAIL} or ${TEST_EMAIL_WITH_DRAFT} for this demo.`,
  };
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
  const escStatuses = account?.escStatuses ?? {};
  const privateChoices = wishlist.filter((s) => s.school_type !== "public");
  const hasPrivateChoice = privateChoices.length > 0;
  const activeChoice =
    privateChoices.find((s) =>
      ["submitted", "docs_pending", "docs_submitted"].includes(
        escStatuses[s.school_id]
      )
    ) ?? null;
  const lastEngagedIndex = privateChoices.reduce(
    (last, s, i) => (escStatuses[s.school_id] ? i : last),
    -1
  );
  const lastEngagedChoice =
    lastEngagedIndex >= 0 ? privateChoices[lastEngagedIndex] : null;
  const nextChoice = privateChoices[lastEngagedIndex + 1] ?? null;
  const grantedChoice =
    privateChoices.find((s) => escStatuses[s.school_id] === "granted") ?? null;

  // Advance one specific PRIVATE school's ESC status. Only reaching
  // 'granted' ends the account-level pursuit — 'rejected' re-opens the
  // next-rank prompt.
  const advanceSchool = (schoolId: string, toState: EscSchoolStatus) => {
    const current = escStatuses[schoolId];
    const valid = ESC_SCHOOL_TRANSITIONS[current] ?? [];
    if (!valid.includes(toState)) return;
    const nextEscStatuses = { ...escStatuses, [schoolId]: toState };
    if (toState === "granted") {
      advance("granted", { escStatuses: nextEscStatuses });
    } else {
      updateAccount({ escStatuses: nextEscStatuses });
    }
  };

  // After a rejection, apply to the next-ranked PRIVATE choice — explicit
  // opt-in, never automatic.
  const applyToNextRank = () => {
    if (!nextChoice) return;
    updateAccount({
      escStatuses: { ...escStatuses, [nextChoice.school_id]: "submitted" },
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
    const rank1 = privateChoices[0];
    updateAccount({
      escStatuses: { ...escStatuses, [rank1.school_id]: "submitted" },
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
    activeChoice,
    lastEngagedChoice,
    nextChoice,
    grantedChoice,
    advanceSchool,
    applyToNextRank,
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
