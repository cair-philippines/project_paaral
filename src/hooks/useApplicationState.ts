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
  getApplicationState,
  replaceWishlist,
  submitEligibilityAssessment,
  submitSurvey,
  updateApplicationStatus,
} from "@/lib/application";
import { deleteDocument, uploadDocument } from "@/lib/documents";
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

  // ── BACKEND SYNC (Chunk 17) ─────────────────────────────────────
  // Every mutating action now waits for the backend to confirm the
  // write before the screen updates (chosen over an instant-update-
  // then-sync-in-background approach, since a silent background-sync
  // failure would leave a student believing something saved when it
  // didn't - see WORKFLOW.md's Chunk 17 entry). `isSyncing` disables
  // actionable buttons while a request is in flight; `syncError` is a
  // plain-language message the UI can show on failure.
  const [isSyncing, setIsSyncing] = useState(false);
  const [syncError, setSyncError] = useState<string | null>(null);

  const withSync = async (
    fn: () => Promise<void>,
    errorMessage: string
  ): Promise<boolean> => {
    setIsSyncing(true);
    setSyncError(null);
    try {
      await fn();
      return true;
    } catch {
      setSyncError(errorMessage);
      return false;
    } finally {
      setIsSyncing(false);
    }
  };

  // Persists a wishlist/per-school-status snapshot with no
  // account-level status change - used by add/remove/reorder/
  // backfill and non-redeeming per-school advances.
  const persistWishlist = (
    wishlistIds: string[],
    statuses: Record<string, EscSchoolStatus>
  ): Promise<boolean> =>
    withSync(async () => {
      if (!account) return;
      await replaceWishlist(account.lrn, wishlistIds, statuses);
      updateAccount({ wishlistIds, escStatuses: statuses });
    }, "Couldn't save your changes. Check your connection and try again.");

  // Persists an account-level status change, optionally bundled with
  // a wishlist/status snapshot in the same call (redemption and
  // "apply again" both change the wishlist and the status together) -
  // mirrors the original `advance(toState, extra)` shape exactly.
  const persistStatus = (
    toState: Account["applicationState"],
    extra: Partial<Account> = {}
  ): Promise<boolean> =>
    withSync(async () => {
      if (!account) return;
      if (extra.wishlistIds !== undefined || extra.escStatuses !== undefined) {
        await replaceWishlist(
          account.lrn,
          extra.wishlistIds ?? account.wishlistIds,
          extra.escStatuses ?? account.escStatuses
        );
      }
      await updateApplicationStatus(
        account.lrn,
        toState,
        extra.nonEscSchoolId ?? null
      );
      updateAccount({ applicationState: toState, ...extra });
    }, "Couldn't save your application status. Check your connection and try again.");

  const persistSurvey = (answers: SurveyAnswers): Promise<boolean> =>
    withSync(async () => {
      if (!account) return;
      await submitSurvey(account.lrn, answers);
    }, "Couldn't save your survey answers. Check your connection and try again.");

  // `wishlistIds` still lets the login modal preload the LRN 100000000002
  // demo draft ("Load Draft") - but only as a fallback now (Chunk 22): the
  // real saved wishlist, fetched below, wins whenever one actually exists,
  // since a returning account's real data is always more meaningful than a
  // hardcoded demo shape it happens to match anyway. Draft state is
  // editable, not locked, per CLAUDE.md.
  const createAccount = async (
    lrn: string = TEST_LRN,
    wishlistIds: string[] = []
  ) => {
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

    // Chunk 22: restore whatever this LRN actually has saved server-side
    // (wishlist, eligibility result, survey answers, confirmed document
    // uploads) instead of always starting from the blank shape above. A
    // brand-new account - or a fetch failure, e.g. a dropped connection
    // mid-login - just falls back to the blank shell rather than blocking
    // login entirely on this one request.
    let hydrated = newAccount;
    try {
      const saved = await getApplicationState(lrn);
      hydrated = {
        ...newAccount,
        applicationState: saved.applicationState,
        nonEscSchoolId: saved.nonEscSchoolId,
        wishlistIds:
          saved.wishlistIds.length > 0 ? saved.wishlistIds : wishlistIds,
        escStatuses: saved.escStatuses,
        category: saved.category,
        eligAnswers: saved.eligAnswers,
        uploadedDocs: saved.uploadedDocs,
      };
      setSurveyAnswers(saved.surveyAnswers);
    } catch {
      setSurveyAnswers(DEFAULT_SURVEY_ANSWERS);
    }

    localStorage.setItem(APPLICATION_STORAGE_KEY, JSON.stringify(hydrated));
    setAccountState(hydrated);
    return hydrated;
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

  const advance = async (
    toState: Account["applicationState"],
    extra: Partial<Account> = {}
  ) => {
    const valid = VALID_TRANSITIONS[applicationState] ?? [];
    if (!valid.includes(toState)) return;
    await persistStatus(toState, extra);
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

  const completeEligibility = async () => {
    if (!account) return;
    const category = computeCategory(eligAnswers);

    // Chunk 19: persist the assessment itself before flipping the
    // account-level status, same "every write confirms before the next
    // one starts" ordering used elsewhere (e.g. wishlist+survey before
    // the submitted status in handleSubmitEsc) - a failed save here
    // must not leave the account showing a category/state it never
    // actually recorded on the backend.
    const eligOk = await withSync(async () => {
      await submitEligibilityAssessment(account.lrn, eligAnswers, category);
    }, "Couldn't save your eligibility answers. Check your connection and try again.");
    if (!eligOk) return;

    // Bypasses `advance()`'s transition-validity guard on purpose,
    // same as the original: resolving the initial ambiguous
    // "eligibility" state into one of its two sub-branches isn't a
    // transition in the state-machine sense.
    await persistStatus(category ? "eligibility" : "not_eligible", {
      category,
      eligAnswers,
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

  const addToWishlist = async (schoolId: string) => {
    if (!account || isPostSubmission) return;
    if (account.wishlistIds.includes(schoolId)) return;
    await persistWishlist(
      [...account.wishlistIds, schoolId],
      account.escStatuses
    );
  };

  const removeFromWishlist = async (schoolId: string) => {
    if (!account || isPostSubmission) return;
    await persistWishlist(
      account.wishlistIds.filter((id) => id !== schoolId),
      account.escStatuses
    );
  };

  // Drag-and-drop reordering (dnd-kit, touch + mouse). Only allowed
  // pre-submission — same gate as removeFromWishlist/addToWishlist, since
  // rank determines ESC application order (rank 1 gets applied to first).
  const reorderWishlist = async (fromIndex: number, toIndex: number) => {
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
    await persistWishlist(next, account.escStatuses);
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
  const advanceSchool = async (schoolId: string, toState: EscSchoolStatus) => {
    if (!account) return;
    const current = escStatuses[schoolId];
    const valid = ESC_SCHOOL_TRANSITIONS[current] ?? [];
    if (!valid.includes(toState)) return;
    const nextEscStatuses = { ...escStatuses, [schoolId]: toState };
    if (toState === "redeemed") {
      await advance("granted", { escStatuses: nextEscStatuses });
    } else {
      await persistWishlist(account.wishlistIds, nextEscStatuses);
    }
  };

  // Accept one school's ESC offer. This is the redemption/convergence
  // point: the chosen school is redeemed, and every other school still
  // occupying a slate slot (pending review or a competing offer) is
  // withdrawn — not rejected, since the school never said no.
  const redeemChoice = async (schoolId: string) => {
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
    await advance("granted", { escStatuses: nextEscStatuses });
  };

  // A slate slot opened up (a rejection) and there's room — explicit
  // opt-in, never automatic, matching the rest of this app's advance-choice
  // pattern.
  const backfillSlate = async () => {
    if (!backfillCandidate || !account) return;
    await persistWishlist(account.wishlistIds, {
      ...escStatuses,
      [backfillCandidate.school_id]: "submitted",
    });
  };

  const continueWithoutSubsidy = async (schoolId: string) => {
    await advance("non_esc", { nonEscSchoolId: schoolId });
  };

  const applyAgainDifferentSchool = async () => {
    await advance("eligibility", { wishlistIds: [], escStatuses: {} });
  };

  // ── DOCUMENTS ────────────────────────────────────────────────────
  const uploadedDocs = account?.uploadedDocs ?? [];
  const requiredDocs = account?.category
    ? getDocList(account.category, account.eligAnswers ?? DEFAULT_ELIG_ANSWERS)
    : [];
  // True only once every required document is CONFIRMED on GCS (i.e. in
  // `uploadedDocs`) - a merely-staged file doesn't count. This is what
  // `canSubmitEsc` gates on, so an application can never be submitted on
  // documents that haven't actually reached storage yet (Paula's explicit
  // direction, 2026-09-02).
  const docsReady =
    requiredDocs.length > 0 && requiredDocs.every((d) => uploadedDocs.includes(d));

  // Files chosen but not yet sent to the backend - local-only (`File`
  // objects can't go in `localStorage`), lost on a hard refresh. An
  // accepted tradeoff: uploading only happens when the student explicitly
  // clicks "Submit Documents," not the moment a file is chosen, so a
  // half-finished application never leaves documents sitting in Cloud
  // Storage. Same staging model applies both before the first submission
  // and for a later "additional document requested" round - there's no
  // separate immediate-upload path for either case.
  const [stagedDocs, setStagedDocs] = useState<Record<string, File>>({});
  const [docUploadProgress, setDocUploadProgress] = useState<{
    completed: number;
    total: number;
  } | null>(null);

  const stageDoc = (doc: string, file: File) => {
    setStagedDocs((prev) => ({ ...prev, [doc]: file }));
  };

  // Clears one document, whichever state it's in - a merely-staged file
  // is removed locally with no network call; an already-confirmed upload
  // needs a real DELETE so it's actually removed from GCS too.
  const removeDoc = (doc: string): Promise<boolean> => {
    if (doc in stagedDocs) {
      setStagedDocs((prev) => {
        const next = { ...prev };
        delete next[doc];
        return next;
      });
      return Promise.resolve(true);
    }
    return withSync(async () => {
      if (!account) return;
      await deleteDocument(account.lrn, doc);
      updateAccount({ uploadedDocs: uploadedDocs.filter((d) => d !== doc) });
    }, "Couldn't remove your file. Check your connection and try again.");
  };

  // The "Submit Documents" action - uploads every currently-staged file
  // to GCS one at a time (not in parallel: several large files competing
  // for one slow connection is worse than uploading them in sequence,
  // and it keeps failure attribution to one specific document instead of
  // several at once). Tracks success as it goes, both so `docUploadProgress`
  // can drive a real progress bar (Paula's call, over a per-request
  // timeout - a timeout would abort a slow-but-working upload instead of
  // letting it finish) and so a retry after a failure never re-uploads a
  // document that already made it through.
  const submitDocuments = async (): Promise<boolean> => {
    if (!account) return false;
    const entries = Object.entries(stagedDocs);
    if (entries.length === 0) return true;

    setIsSyncing(true);
    setSyncError(null);
    setDocUploadProgress({ completed: 0, total: entries.length });

    let nextUploaded = uploadedDocs;
    for (const [doc, file] of entries) {
      try {
        await uploadDocument(account.lrn, doc, file);
      } catch {
        setSyncError(
          `Couldn't upload "${doc}." Check your connection and try again.`
        );
        setIsSyncing(false);
        setDocUploadProgress(null);
        return false;
      }
      nextUploaded = nextUploaded.includes(doc)
        ? nextUploaded
        : [...nextUploaded, doc];
      updateAccount({ uploadedDocs: nextUploaded });
      setStagedDocs((prev) => {
        const next = { ...prev };
        delete next[doc];
        return next;
      });
      setDocUploadProgress((prev) =>
        prev ? { ...prev, completed: prev.completed + 1 } : null
      );
    }

    setIsSyncing(false);
    setDocUploadProgress(null);
    return true;
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

  const handleSubmitEsc = async () => {
    if (!canSubmitEsc || !account) return;
    const initialSlate = privateChoices.slice(0, ESC_SLATE_CAP);
    const nextEscStatuses = { ...escStatuses };
    for (const s of initialSlate) nextEscStatuses[s.school_id] = "submitted";

    const wishlistOk = await persistWishlist(
      account.wishlistIds,
      nextEscStatuses
    );
    if (!wishlistOk) return;
    const surveyOk = await persistSurvey(surveyAnswers);
    if (!surveyOk) return;
    await advance("submitted");
  };

  const handleEnrollNonEsc = async () => {
    if (!canEnrollNonEsc) return;
    const school = wishlist[0];
    const surveyOk = await persistSurvey(surveyAnswers);
    if (!surveyOk) return;
    await advance("non_esc", { nonEscSchoolId: school.school_id, surveyAnswers });
  };

  return {
    account,
    createAccount,
    logout,
    updateAccount,

    isSyncing,
    syncError,

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
    stagedDocs,
    docUploadProgress,
    stageDoc,
    removeDoc,
    submitDocuments,

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
