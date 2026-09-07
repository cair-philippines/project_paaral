import { useState } from "react";
import type { School } from "@/types/school";
import type {
  Account,
  EligAnswers,
  EligHistoryEntry,
  EligStep,
  EscApplicationEntry,
  EscSchoolStatus,
  SurveyAnswers,
} from "@/types/application";
import { computeCategory, getDocList } from "@/lib/eligibility";
import {
  ADVANCE_TRIGGERING_STATES,
  ESC_SCHOOL_TRANSITIONS,
  MAX_ESC_APPLICATIONS,
  MAX_WISHLIST_SIZE,
  MIN_WISHLIST_SIZE,
  TERMINAL_UNSUCCESSFUL_STATES,
} from "@/lib/applicationState";
import { apiPost } from "@/lib/api";
import {
  createApplication,
  getApplicationState,
  replaceWishlist,
  submitEligibilityAssessment,
  submitEscApplications,
  submitSurvey,
  updateEscApplicationStatus,
  type HydratedAccountState,
} from "@/lib/application";
import { deleteDocument, uploadDocument } from "@/lib/documents";
import {
  APPLICATION_STORAGE_KEY,
  LEARNER_RECORD,
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
 * a real Postgres lookup now. The not-found error is shown as-is —
 * the login modal already lists the demo accounts below the form, so
 * repeating them in the error message would be redundant. A network/
 * server failure is reported as its own distinct error rather than
 * silently looking like "not found". */
export async function verifyLoginEmail(
  email: string
): Promise<LoginLookupResult> {
  try {
    return await apiPost<LoginLookupResult>(
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
}

/** Ported from src/App.jsx's v3 decoupled ESC application state machine,
 * then reworked for the ranked-preferences/ESC-application split: a
 * separate ranked preference list (`wishlistIds`, 3–5 schools of any
 * type) from an explicit ESC application submission (`escApplications`,
 * up to `MAX_ESC_APPLICATIONS` ESC-participating schools). Real process,
 * confirmed directly: schools review one at a time, in rank order — a
 * lower-ranked choice is never looked at until the higher-ranked one
 * resolves, so a student can never hold two live offers at once. UI
 * (login modal, application panel, questionnaire screens) is
 * intentionally not part of this hook; it only manages account/
 * eligibility/wishlist/ESC-application state and the actions that
 * transition between them. */
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

  // Patches one school's `escApplications` entry, reading the merge base
  // from the functional updater's `prev` rather than the outer `account`
  // closure — required because `advanceSchool` can make two sequential
  // network calls in one action (a rejection/decline, then promoting the
  // next queued school), and a merge based on a stale closure snapshot
  // would silently undo the first call's change when the second commits.
  const applyEscApplicationUpdate = (
    schoolId: string,
    entry: EscApplicationEntry
  ) => {
    setAccountState((prev) => {
      if (!prev) return prev;
      const next = {
        ...prev,
        escApplications: { ...prev.escApplications, [schoolId]: entry },
      };
      localStorage.setItem(APPLICATION_STORAGE_KEY, JSON.stringify(next));
      return next;
    });
  };

  // ── BACKEND SYNC (Chunk 17, extended for the ESC-application split) ──
  // Every mutating action waits for the backend to confirm the write
  // before the screen updates (chosen over an instant-update-then-sync-
  // in-background approach, since a silent background-sync failure would
  // leave a student believing something saved when it didn't). `isSyncing`
  // disables actionable buttons while a request is in flight; `syncError`
  // is a plain-language message the UI can show on failure.
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

  // Persists the ranked preference list only — never touches ESC
  // application submission, which is a separate, explicit step.
  const persistWishlist = (wishlistIds: string[]): Promise<boolean> =>
    withSync(async () => {
      if (!account) return;
      await replaceWishlist(account.lrn, wishlistIds);
      updateAccount({ wishlistIds });
    }, "Couldn't save your changes. Check your connection and try again.");

  const persistSurvey = (answers: SurveyAnswers): Promise<boolean> =>
    withSync(async () => {
      if (!account) return;
      await submitSurvey(account.lrn, answers);
    }, "Couldn't save your survey answers. Check your connection and try again.");

  // Updates one school's ESC application status - the shared primitive
  // behind advanceSchool/redeemChoice/declineOffer. No transition-
  // validity check here; callers already validated before calling this.
  const persistEscStatus = (
    schoolId: string,
    status: EscSchoolStatus
  ): Promise<boolean> =>
    withSync(async () => {
      if (!account) return;
      const updated = await updateEscApplicationStatus(
        account.lrn,
        schoolId,
        status
      );
      applyEscApplicationUpdate(schoolId, {
        status: updated.status,
        submittedAt: updated.submittedAt,
        resolvedAt: updated.resolvedAt,
      });
    }, "Couldn't update this application. Check your connection and try again.");

  // Turns a fetched/created HydratedAccountState into the actual
  // Account object and commits it to local/global state - shared by
  // both the existing-account login path and the new-account
  // creation path below, since both ultimately produce the same
  // shape from the same translated response.
  const applyHydratedAccount = (
    lrn: string,
    saved: HydratedAccountState
  ): Account => {
    const hydrated: Account = {
      email: `${lrn}@deped.gov.ph`,
      lrn,
      name: `${LEARNER_RECORD.firstName} ${LEARNER_RECORD.mi}. ${LEARNER_RECORD.lastName}`,
      category: saved.category,
      eligAnswers: saved.eligAnswers,
      isEligible: saved.isEligible,
      wishlistIds: saved.wishlistIds,
      escApplications: saved.escApplications,
      surveyAnswers: saved.surveyAnswers,
      uploadedDocs: saved.uploadedDocs,
    };
    setSurveyAnswers(saved.surveyAnswers);
    localStorage.setItem(APPLICATION_STORAGE_KEY, JSON.stringify(hydrated));
    setAccountState(hydrated);
    setSelectedEscSchoolIds([]);
    return hydrated;
  };

  // Hydrates an existing account on login (Chunk 22) - restores
  // whatever this LRN actually has saved server-side (wishlist,
  // eligibility result, survey answers, confirmed document uploads,
  // ESC applications). Lets any failure propagate, including a 404
  // (`ApiError`) - a genuinely new learner has no Application row
  // yet, and the login modal needs to tell that apart from a real
  // network/server failure to decide whether to show the account-
  // creation screen at all, so it's no longer silently swallowed
  // into a blank fallback here.
  const createAccount = async (lrn: string = TEST_LRN) => {
    const saved = await getApplicationState(lrn);
    return applyHydratedAccount(lrn, saved);
  };

  // Explicit account creation - the "Create My Account & Continue"
  // button. Creates the Application row server-side if it doesn't
  // exist yet (idempotent either way), then hydrates exactly like a
  // normal login.
  const createNewAccount = async (lrn: string) => {
    const saved = await createApplication(lrn);
    return applyHydratedAccount(lrn, saved);
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
    setSelectedEscSchoolIds([]);
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
  // account state) so "Start over" on the ineligible result can return to
  // the first question. Not part of the original ported logic; added
  // while building the questionnaire screen since the old mockup had this
  // same inline reset.
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

  // Chunk 19, simplified for the is_eligible split: the backend's
  // eligibility-submit endpoint now sets `Application.is_eligible` itself
  // as part of the same write, so there's no separate account-level
  // status flip to persist afterward — one network call, then update
  // local state to match.
  const completeEligibility = async (): Promise<boolean> => {
    if (!account) return false;
    const category = computeCategory(eligAnswers);
    return withSync(async () => {
      await submitEligibilityAssessment(account.lrn, eligAnswers, category);
      updateAccount({
        isEligible: category !== null,
        category,
        eligAnswers,
      });
    }, "Couldn't save your eligibility answers. Check your connection and try again.");
  };

  // ── WISHLIST (ranked preferences, 3–5 schools, any type) ──────────
  const wishlist: School[] = account
    ? account.wishlistIds
        .map((id) => schools.find((s) => s.school_id === id))
        .filter((s): s is School => Boolean(s))
    : [];

  const escApplications = account?.escApplications ?? {};
  // Locked once any ESC application exists — matches the composite
  // foreign key from EscApplication onto Wishlist server-side (a
  // wishlist row referenced by a submitted ESC application can't be
  // removed without violating it), so this gate also keeps every
  // wishlist mutation on the safe side of that constraint.
  const isPostSubmission = Object.keys(escApplications).length > 0;

  const isInWishlist = (schoolId: string) =>
    (account?.wishlistIds ?? []).includes(schoolId);

  const addToWishlist = async (schoolId: string) => {
    if (!account || isPostSubmission) return;
    if (account.wishlistIds.includes(schoolId)) return;
    if (account.wishlistIds.length >= MAX_WISHLIST_SIZE) return;
    await persistWishlist([...account.wishlistIds, schoolId]);
  };

  const removeFromWishlist = async (schoolId: string) => {
    if (!account || isPostSubmission) return;
    setSelectedEscSchoolIds((prev) => prev.filter((id) => id !== schoolId));
    await persistWishlist(account.wishlistIds.filter((id) => id !== schoolId));
  };

  // Drag-and-drop reordering (dnd-kit, touch + mouse). Only allowed
  // pre-submission — same gate as removeFromWishlist/addToWishlist.
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
    await persistWishlist(next);
  };

  // 3–5 schools of any type, with at least one ESC-participating school
  // (otherwise there's nothing to ever submit an ESC application to).
  // Replaces the old hasPublicAlternative guaranteed-placement gate,
  // dropped since PAARAL only processes ESC applications now.
  const canFinalizeWishlist =
    wishlist.length >= MIN_WISHLIST_SIZE &&
    wishlist.length <= MAX_WISHLIST_SIZE &&
    wishlist.some((s) => s.is_esc_participating);

  // ── ESC APPLICATION SELECTION (pre-submission) ────────────────────
  // Which ESC-participating wishlist schools the student has picked to
  // actually submit to — an explicit choice, capped at
  // MAX_ESC_APPLICATIONS, not auto-derived from rank order. Local/
  // transient until handleSubmitEsc actually submits it.
  const escParticipatingWishlistSchools = wishlist.filter(
    (s) => s.is_esc_participating
  );
  const [selectedEscSchoolIds, setSelectedEscSchoolIds] = useState<string[]>(
    []
  );

  const toggleEscSelection = (schoolId: string) => {
    if (isPostSubmission) return;
    setSelectedEscSchoolIds((prev) => {
      if (prev.includes(schoolId)) return prev.filter((id) => id !== schoolId);
      if (prev.length >= MAX_ESC_APPLICATIONS) return prev;
      return [...prev, schoolId];
    });
  };

  // ── ESC APPLICATIONS (post-submission, sequential by rank) ────────
  // Real process, confirmed directly: schools review one at a time, in
  // rank order — a lower-ranked choice is never looked at until the
  // higher-ranked one resolves. So there's only ever one school that's
  // actively "current" (submitted/docs_pending/docs_submitted/granted)
  // at a time, by construction, not just by convention.
  const escApplicationSchools = escParticipatingWishlistSchools.filter((s) =>
    Boolean(escApplications[s.school_id])
  );
  const currentEscApplication =
    escApplicationSchools.find((s) => {
      const status = escApplications[s.school_id]?.status;
      return (
        status === "submitted" ||
        status === "docs_pending" ||
        status === "docs_submitted" ||
        status === "granted"
      );
    }) ?? null;
  const queuedEscApplications = escApplicationSchools.filter(
    (s) => escApplications[s.school_id]?.status === "queued"
  );
  const resolvedEscApplications = escApplicationSchools.filter((s) =>
    ["redeemed", "rejected", "declined"].includes(
      escApplications[s.school_id]?.status ?? ""
    )
  );
  const redeemedChoice =
    escApplicationSchools.find(
      (s) => escApplications[s.school_id]?.status === "redeemed"
    ) ?? null;
  // Every submitted ESC application ended unsuccessfully (rejected, or
  // the family declined an offer), none redeemed — the ESC track is
  // over. Drives the plain informational "enroll without a subsidy"
  // message, alongside `isEligible === false` for the never-eligible case.
  const allEscApplicationsUnsuccessful =
    escApplicationSchools.length > 0 &&
    escApplicationSchools.every((s) =>
      TERMINAL_UNSUCCESSFUL_STATES.has(escApplications[s.school_id]?.status)
    );
  const hasDocsPending =
    currentEscApplication !== null &&
    escApplications[currentEscApplication.school_id]?.status === "docs_pending";

  // Advance one school's ESC application status - demo-driven for now,
  // standing in for what would eventually arrive from School View. A
  // resolution that isn't a success (rejected, or the family declining
  // an offer) also promotes the next queued school (lowest rank) to
  // submitted — this stands in for DepEd's own round-based processing
  // moving on, not a student action, matching the real process.
  const advanceSchool = async (schoolId: string, toState: EscSchoolStatus) => {
    if (!account) return;
    const current = escApplications[schoolId]?.status;
    if (!current) return;
    const valid = ESC_SCHOOL_TRANSITIONS[current] ?? [];
    if (!valid.includes(toState)) return;

    const ok = await persistEscStatus(schoolId, toState);
    if (!ok) return;

    if (ADVANCE_TRIGGERING_STATES.has(toState)) {
      const nextQueued = escParticipatingWishlistSchools.find(
        (s) =>
          s.school_id !== schoolId &&
          escApplications[s.school_id]?.status === "queued"
      );
      if (nextQueued) {
        await persistEscStatus(nextQueued.school_id, "submitted");
      }
    }
  };

  // Accept the single live offer. No competing offer to withdraw
  // anymore - a student can never hold two at once.
  const redeemChoice = async (schoolId: string) => {
    if (escApplications[schoolId]?.status !== "granted") return;
    await advanceSchool(schoolId, "redeemed");
  };

  // Decline the single live offer - promotes the next queued school,
  // same as a rejection (see advanceSchool).
  const declineOffer = async (schoolId: string) => {
    if (escApplications[schoolId]?.status !== "granted") return;
    await advanceSchool(schoolId, "declined");
  };

  // ── DOCUMENTS ────────────────────────────────────────────────────
  const uploadedDocs = account?.uploadedDocs ?? [];
  const requiredDocs = account?.category
    ? getDocList(account.category, account.eligAnswers ?? DEFAULT_ELIG_ANSWERS)
    : [];
  // True only once every required document is CONFIRMED on GCS (i.e. in
  // `uploadedDocs`) - a merely-staged file doesn't count. This is what
  // `canSubmitEsc` gates on, so an application can never be submitted on
  // documents that haven't actually reached storage yet.
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
  // can drive a real progress bar and so a retry after a failure never
  // re-uploads a document that already made it through.
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
  // The ESC-specific concern question only makes sense for a student
  // actually pursuing the ESC track - hidden for a never-eligible one,
  // same as the original two-section survey design.
  const showEscSurveySection = account?.isEligible === true;

  // ── SUBMIT ───────────────────────────────────────────────────────
  const canSubmitEsc =
    account?.isEligible === true &&
    !isPostSubmission &&
    canFinalizeWishlist &&
    selectedEscSchoolIds.length >= 1 &&
    docsReady &&
    generalSurveyComplete &&
    escSurveyComplete;

  const handleSubmitEsc = async () => {
    if (!canSubmitEsc || !account) return;
    const ok = await withSync(async () => {
      const entries = await submitEscApplications(
        account.lrn,
        selectedEscSchoolIds
      );
      const nextEscApplications: Record<string, EscApplicationEntry> = {};
      for (const entry of entries) {
        nextEscApplications[entry.schoolId] = {
          status: entry.status,
          submittedAt: entry.submittedAt,
          resolvedAt: entry.resolvedAt,
        };
      }
      updateAccount({ escApplications: nextEscApplications });
    }, "Couldn't submit your ESC applications. Check your connection and try again.");
    if (!ok) return;
    await persistSurvey(surveyAnswers);
  };

  // For an ineligible student, or one whose ESC applications are all
  // unsuccessful, PAARAL shows a plain informational message only - no
  // school selection, no application, no tracking (out of scope: PAARAL
  // only processes direct ESC applications). Their general survey
  // feedback is still worth capturing, though, so a lightweight
  // feedback-only submit stays available.
  const showEnrollWithoutSubsidyMessage =
    account?.isEligible === false || allEscApplicationsUnsuccessful;

  const canSubmitGeneralFeedback = generalSurveyComplete;

  const submitGeneralFeedback = async () => {
    if (!canSubmitGeneralFeedback) return;
    await persistSurvey(surveyAnswers);
  };

  return {
    account,
    createAccount,
    createNewAccount,
    logout,
    updateAccount,

    isSyncing,
    syncError,

    isPostSubmission,

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
    canFinalizeWishlist,

    escApplications,
    escParticipatingWishlistSchools,
    selectedEscSchoolIds,
    toggleEscSelection,

    escApplicationSchools,
    currentEscApplication,
    queuedEscApplications,
    resolvedEscApplications,
    redeemedChoice,
    allEscApplicationsUnsuccessful,
    hasDocsPending,
    advanceSchool,
    redeemChoice,
    declineOffer,

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
    showEscSurveySection,

    canSubmitEsc,
    handleSubmitEsc,
    showEnrollWithoutSubsidyMessage,
    canSubmitGeneralFeedback,
    submitGeneralFeedback,
  };
}
