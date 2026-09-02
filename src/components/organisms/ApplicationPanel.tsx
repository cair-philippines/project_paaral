"use client";

import { useRef, useState, type ReactNode } from "react";
import Button from "@mui/material/Button";
import ToggleButtonGroup from "@mui/material/ToggleButtonGroup";
import ToggleButton from "@mui/material/ToggleButton";
import {
  X,
  Clock3,
  AlertCircle,
  FileCheck,
  Award,
  Info,
  Heart,
  Check,
  GripVertical,
  Upload,
} from "lucide-react";
import {
  DndContext,
  KeyboardSensor,
  PointerSensor,
  TouchSensor,
  closestCenter,
  useSensor,
  useSensors,
  type DragEndEvent,
} from "@dnd-kit/core";
import {
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import AccountSection from "@/components/molecules/AccountSection";
import { useApplication } from "@/components/templates/ApplicationStateProvider";
import { SCHOOL_STATUS_META } from "@/lib/applicationState";
import {
  ALLOWED_DOCUMENT_TYPES,
  MAX_DOCUMENT_SIZE_BYTES,
} from "@/lib/documents";
import type { EscSchoolStatus } from "@/types/application";
import type { School } from "@/types/school";

type PanelTab = "status" | "documents" | "choices" | "survey";

const STATUS_ICON: Record<EscSchoolStatus, React.ReactNode> = {
  submitted: <Clock3 className="h-6 w-6 shrink-0 text-blue-500" />,
  rejected: <AlertCircle className="h-6 w-6 shrink-0 text-red-500" />,
  docs_pending: <FileCheck className="h-6 w-6 shrink-0 text-amber-500" />,
  docs_submitted: <FileCheck className="h-6 w-6 shrink-0 text-blue-500" />,
  granted: <Award className="h-6 w-6 shrink-0 text-purple-500" />,
  redeemed: <Check className="h-6 w-6 shrink-0 text-green-500" />,
  withdrawn: <X className="h-6 w-6 shrink-0 text-slate-400" />,
};

// UI-only demo controls (deliberately not part of the ported hook/lib —
// see memory-sessions.md, SCHOOL_STATUS_META dropped these on purpose).
const DEMO_TRANSITIONS: Partial<
  Record<
    EscSchoolStatus,
    { label: string; next: EscSchoolStatus; className: string }[]
  >
> = {
  submitted: [
    {
      label: "Demo: School Committee Grants Subsidy",
      next: "granted",
      className: "bg-green-600 hover:bg-green-700",
    },
    {
      label: "Demo: School Committee Rejects Application",
      next: "rejected",
      className: "bg-red-600 hover:bg-red-700",
    },
    {
      label: "Demo: Additional Document Requested",
      next: "docs_pending",
      className: "bg-slate-700 hover:bg-slate-800",
    },
  ],
  docs_submitted: [
    {
      label: "Demo: School Committee Grants Subsidy",
      next: "granted",
      className: "bg-green-600 hover:bg-green-700",
    },
    {
      label: "Demo: School Committee Rejects Application",
      next: "rejected",
      className: "bg-red-600 hover:bg-red-700",
    },
  ],
};

const SECTION_LABEL = "text-[10px] font-bold uppercase tracking-widest text-slate-500";

/**
 * The learner's application workspace — Status/Choices/Documents/Survey —
 * embedded directly inside the single, standalone `/account` page
 * (`AccountPage.tsx`), below the hero band and journey strip.
 *
 * Restructured 2026-08-24 (manus.ai "SchoolPath" reference, per Paula):
 * previously an MUI `Tabs` shell switching between four panels; now each
 * panel is its own always-visible numbered `AccountSection` in page order,
 * matching the reference's "numbered path sections" layout. Section order
 * and content are otherwise unchanged — `tabList` still decides which
 * sections exist and in what order (pre- vs. post-submission), it just no
 * longer drives a tab switch. No business logic changed; every gate,
 * demo control, and the dnd-kit drag-reorder all behave exactly as before.
 */
export default function ApplicationPanel() {
  const app = useApplication();
  const {
    account,
    applicationState,
    isPostSubmission,
    wishlist,
    escStatuses,
    removeFromWishlist,
    reorderWishlist,
    hasPublicAlternative,
    hasPrivateChoice,
    privateChoices,
    redeemedChoice,
    rejectedChoices,
    backfillCandidate,
    isSlateExhausted,
    advanceSchool,
    redeemChoice,
    backfillSlate,
    continueWithoutSubsidy,
    applyAgainDifferentSchool,
    requiredDocs,
    uploadedDocs,
    stagedDocs,
    docsReady,
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
    isSyncing,
    syncError,
  } = app;

  // Which rejected school is picked for "Continue Enrollment (No Subsidy)"
  // once the private-school slate is fully exhausted — only meaningful when
  // more than one school ended in 'rejected', otherwise it's the only one.
  const [nonEscPickId, setNonEscPickId] = useState<string | null>(null);

  if (!account) return null;

  const docsPendingChoices = privateChoices.filter(
    (s) => escStatuses[s.school_id] === "docs_pending"
  );
  const selectedNonEscChoice =
    rejectedChoices.find((s) => s.school_id === nonEscPickId) ??
    rejectedChoices[0] ??
    null;

  const tabList: PanelTab[] = isPostSubmission
    ? ["status", "documents", "choices"]
    : ["choices", "documents", "survey"];

  const choicesTitle = isPostSubmission
    ? "The schools you ranked, in order"
    : wishlist.length > 0
      ? `${wishlist.length} school${wishlist.length === 1 ? "" : "s"} in your ranked list`
      : "Add schools to build your ranked list";

  const documentsTitle = isPostSubmission
    ? "Documents on file"
    : "Prepare your required documents";

  const surveyTitle =
    applicationState === "not_eligible"
      ? "Tell us about your experience"
      : "A few quick questions, then submit";

  const sectionFor: Record<
    PanelTab,
    { eyebrow: string; title: string; content: ReactNode }
  > = {
    status: {
      eyebrow: "Your ESC Application",
      title: "Track your subsidy application",
      content: (
        <div className="space-y-4">
          {applicationState === "granted" &&
            (() => {
              const cfg = SCHOOL_STATUS_META.redeemed;
              const name = redeemedChoice?.school_name || "your chosen school";
              return (
                <div className={`rounded-xl border p-4 ${cfg.color}`}>
                  <div className="flex items-start gap-3">
                    {STATUS_ICON.redeemed}
                    <div>
                      <p className="text-sm font-bold text-slate-800">
                        {cfg.title}
                      </p>
                      <p className="mt-1 text-xs leading-relaxed text-slate-600">
                        {cfg.desc(name)}
                      </p>
                      <p className="mt-2 text-xs leading-relaxed text-slate-500">
                        Enrolling at {name} is a separate, independent
                        step — you can enroll before or after this
                        approval.
                      </p>
                    </div>
                  </div>
                </div>
              );
            })()}

          {applicationState === "non_esc" &&
            (() => {
              const school = wishlist.find(
                (s) => s.school_id === account.nonEscSchoolId
              );
              const name = school?.school_name || "your chosen school";
              return (
                <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4 shadow-sm">
                  <div className="flex items-start gap-3">
                    <Info className="h-6 w-6 shrink-0 text-slate-400" />
                    <div>
                      <p className="text-sm font-bold text-slate-800">
                        Enrolling Without ESC
                      </p>
                      <p className="mt-1 text-xs leading-relaxed text-slate-600">
                        You&apos;re proceeding with enrollment at {name}{" "}
                        without the ESC fee subsidy.
                      </p>
                    </div>
                  </div>
                </div>
              );
            })()}

          {applicationState === "submitted" && (
            <>
              {privateChoices
                .filter((school) => Boolean(escStatuses[school.school_id]))
                .map((school) => {
                  const status = escStatuses[school.school_id];
                  const cfg = SCHOOL_STATUS_META[status];
                  const demo = DEMO_TRANSITIONS[status] ?? [];
                  return (
                    <div
                      key={school.school_id}
                      className={`rounded-xl border p-4 ${cfg.color}`}
                    >
                      <div className="flex items-start gap-3">
                        {STATUS_ICON[status]}
                        <div className="min-w-0 flex-1">
                          <p className="text-xs font-semibold text-slate-500">
                            {school.school_name}
                          </p>
                          <p className="text-sm font-bold text-slate-800">
                            {cfg.title}
                          </p>
                          <p className="mt-1 text-xs leading-relaxed text-slate-600">
                            {cfg.desc(school.school_name)}
                          </p>
                        </div>
                      </div>
                      {status === "granted" && (
                        <Button
                          fullWidth
                          variant="contained"
                          sx={{ minHeight: 48, mt: 3 }}
                          disabled={isSyncing}
                          onClick={() => redeemChoice(school.school_id)}
                        >
                          {isSyncing ? "Saving…" : "Redeem This Offer"}
                        </Button>
                      )}
                      {demo.length > 0 && (
                        <div className="mt-3 rounded-xl border border-dashed border-slate-300 p-4">
                          <p className={`${SECTION_LABEL} mb-3 text-slate-400`}>
                            Demo Controls
                          </p>
                          <div className="space-y-2">
                            {demo.map((d) => (
                              <button
                                key={d.next}
                                type="button"
                                disabled={isSyncing}
                                onClick={() =>
                                  advanceSchool(school.school_id, d.next)
                                }
                                className={`w-full rounded-lg py-2.5 text-xs font-bold uppercase tracking-wide text-white disabled:opacity-50 ${d.className}`}
                              >
                                {d.label}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}

              {backfillCandidate && (
                <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
                  <p className="mb-3 text-sm text-slate-700">
                    One of your schools said no. Would you like to add your
                    next choice,{" "}
                    <span className="font-semibold">
                      {backfillCandidate.school_name}
                    </span>
                    , to your active applications?
                  </p>
                  <Button
                    fullWidth
                    variant="contained"
                    sx={{ minHeight: 48 }}
                    disabled={isSyncing}
                    onClick={backfillSlate}
                  >
                    {isSyncing
                      ? "Saving…"
                      : `Yes, Add ${backfillCandidate.school_name}`}
                  </Button>
                </div>
              )}

              {isSlateExhausted && (
                <div className="space-y-3 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
                  <p className="text-sm text-slate-700">
                    None of your private school choices worked out this
                    time. What would you like to do next?
                  </p>

                  {rejectedChoices.length > 1 && (
                    <div>
                      <p className="mb-2 text-xs font-semibold text-slate-500">
                        Which school would you like to enroll at without a
                        subsidy?
                      </p>
                      <ToggleButtonGroup
                        fullWidth
                        exclusive
                        orientation="vertical"
                        value={selectedNonEscChoice?.school_id ?? null}
                        onChange={(_, value) =>
                          value !== null && setNonEscPickId(value)
                        }
                      >
                        {rejectedChoices.map((s) => (
                          <ToggleButton
                            key={s.school_id}
                            value={s.school_id}
                            sx={{ minHeight: 44, justifyContent: "flex-start" }}
                          >
                            {s.school_name}
                          </ToggleButton>
                        ))}
                      </ToggleButtonGroup>
                    </div>
                  )}

                  <div className="space-y-2">
                    {selectedNonEscChoice && (
                      <Button
                        fullWidth
                        variant="contained"
                        color="inherit"
                        sx={{
                          minHeight: 48,
                          bgcolor: "#1e293b",
                          color: "white",
                          "&:hover": { bgcolor: "#0f172a" },
                        }}
                        disabled={isSyncing}
                        onClick={() =>
                          continueWithoutSubsidy(selectedNonEscChoice.school_id)
                        }
                      >
                        {isSyncing
                          ? "Saving…"
                          : `Continue Enrollment at ${selectedNonEscChoice.school_name} (No Subsidy)`}
                      </Button>
                    )}
                    <Button
                      fullWidth
                      variant="outlined"
                      sx={{ minHeight: 48 }}
                      disabled={isSyncing}
                      onClick={applyAgainDifferentSchool}
                    >
                      Stop and Choose Different Private Schools
                    </Button>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      ),
    },
    choices: {
      eyebrow: "Your School Choices",
      title: choicesTitle,
      content: (
        <div className="space-y-3">
          {isPostSubmission && (
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <p className="text-xs text-slate-500">
                Your school choices have been submitted and are now
                read-only.
              </p>
            </div>
          )}
          {wishlist.length === 0 && (
            <div className="py-12 text-center">
              <Heart className="mx-auto mb-3 h-8 w-8 text-slate-200" />
              <p className="text-sm text-slate-400">No schools added yet.</p>
              <p className="mt-2 text-xs text-slate-400">
                Use the map, list, or card view to find schools and add
                them here.
              </p>
            </div>
          )}
          {wishlist.length > 0 &&
            (!isPostSubmission ? (
              <SortableWishlist
                wishlist={wishlist}
                escStatuses={escStatuses}
                onRemove={removeFromWishlist}
                onReorder={reorderWishlist}
              />
            ) : (
              wishlist.map((school, i) => (
                <WishlistRowContent
                  key={school.school_id}
                  rank={i + 1}
                  school={school}
                  meta={
                    escStatuses[school.school_id]
                      ? SCHOOL_STATUS_META[escStatuses[school.school_id]]
                      : null
                  }
                />
              ))
            ))}
          {!isPostSubmission && !hasPublicAlternative && wishlist.length > 0 && (
            <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs text-amber-700">
              Add at least one public school to guarantee a placement,
              even if your ESC application isn&apos;t approved.
            </div>
          )}
        </div>
      ),
    },
    documents: {
      eyebrow: "Your Documents",
      title: documentsTitle,
      content: (
        <DocumentsTab
          applicationState={applicationState}
          isPostSubmission={isPostSubmission}
          category={account.category}
          requiredDocs={requiredDocs}
          uploadedDocs={uploadedDocs}
          stagedDocs={stagedDocs}
          docsReady={docsReady}
          docUploadProgress={docUploadProgress}
          stageDoc={stageDoc}
          removeDoc={removeDoc}
          submitDocuments={submitDocuments}
          isSyncing={isSyncing}
          docsPendingChoices={docsPendingChoices}
          advanceSchool={advanceSchool}
        />
      ),
    },
    survey: {
      eyebrow: "Before You Submit",
      title: surveyTitle,
      content: (
        <div className="space-y-6">
          <p className="text-xs text-slate-400">
            {applicationState === "not_eligible"
              ? "2 quick questions before you can enroll."
              : "3 quick questions before you can submit."}
          </p>

          <div className="space-y-5">
            <p className="text-[10px] font-bold uppercase tracking-widest text-slate-400">
              Using PAARAL
            </p>
            <div>
              <p className="mb-2 text-sm font-semibold text-slate-800">
                1. How easy was it to find schools?
              </p>
              <p className="mb-3 text-xs text-slate-400">
                1 = very hard, 5 = very easy
              </p>
              <ToggleButtonGroup
                fullWidth
                exclusive
                value={surveyAnswers.ease}
                onChange={(_, value) =>
                  value !== null &&
                  setSurveyAnswers((a) => ({ ...a, ease: value }))
                }
              >
                {[1, 2, 3, 4, 5].map((n) => (
                  <ToggleButton key={n} value={n} sx={{ minHeight: 44 }}>
                    {n}
                  </ToggleButton>
                ))}
              </ToggleButtonGroup>
            </div>

            <div>
              <p className="mb-3 text-sm font-semibold text-slate-800">
                2. Did this information help you decide where to apply?
              </p>
              <ToggleButtonGroup
                fullWidth
                exclusive
                orientation="vertical"
                value={surveyAnswers.helpful}
                onChange={(_, value) =>
                  value !== null &&
                  setSurveyAnswers((a) => ({ ...a, helpful: value }))
                }
              >
                {["Yes", "Somewhat", "No"].map((opt) => (
                  <ToggleButton
                    key={opt}
                    value={opt}
                    sx={{ minHeight: 44, justifyContent: "flex-start" }}
                  >
                    {opt}
                  </ToggleButton>
                ))}
              </ToggleButtonGroup>
            </div>
          </div>

          {applicationState !== "not_eligible" && (
            <div className="space-y-3">
              <p className="text-[10px] font-bold uppercase tracking-widest text-slate-400">
                About Your ESC Application
              </p>
              <p className="text-sm font-semibold text-slate-800">
                3. What is your biggest worry about getting a subsidized
                private-school spot?
              </p>
              <ToggleButtonGroup
                fullWidth
                exclusive
                orientation="vertical"
                value={surveyAnswers.concern}
                onChange={(_, value) =>
                  value !== null &&
                  setSurveyAnswers((a) => ({ ...a, concern: value }))
                }
              >
                {["Cost", "Distance", "School quality", "Slot availability"].map(
                  (opt) => (
                    <ToggleButton
                      key={opt}
                      value={opt}
                      sx={{ minHeight: 44, justifyContent: "flex-start" }}
                    >
                      {opt}
                    </ToggleButton>
                  )
                )}
              </ToggleButtonGroup>
            </div>
          )}

          <div className="space-y-2 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
            <p className={`${SECTION_LABEL} mb-2`}>
              {applicationState === "not_eligible"
                ? "Enrollment Checklist"
                : "Submission Checklist"}
            </p>
            {(applicationState === "not_eligible"
              ? [
                  {
                    done: wishlist.length > 0,
                    label: "At least one school added",
                  },
                  { done: generalSurveyComplete, label: "Survey complete" },
                ]
              : [
                  {
                    done: hasPrivateChoice,
                    label: "At least one private school added",
                  },
                  {
                    done: hasPublicAlternative,
                    label: "Public school included (guaranteed fallback)",
                  },
                  { done: docsReady, label: "Documents submitted" },
                  {
                    done: generalSurveyComplete && escSurveyComplete,
                    label: "Survey complete",
                  },
                ]
            ).map(({ done, label }) => (
              <div
                key={label}
                className={`flex items-center gap-2 text-xs ${done ? "text-green-700" : "text-slate-400"}`}
              >
                <span
                  className={`flex h-4 w-4 shrink-0 items-center justify-center rounded-full ${done ? "bg-green-500" : "bg-slate-200"}`}
                >
                  {done && (
                    <Check className="h-2.5 w-2.5 text-white" strokeWidth={3} />
                  )}
                </span>
                {label}
              </div>
            ))}
          </div>

          {applicationState === "not_eligible" ? (
            <Button
              fullWidth
              variant="contained"
              sx={{ minHeight: 48 }}
              disabled={!canEnrollNonEsc || isSyncing}
              onClick={handleEnrollNonEsc}
            >
              {isSyncing ? "Saving…" : "Enroll Without ESC"}
            </Button>
          ) : (
            <Button
              fullWidth
              variant="contained"
              sx={{ minHeight: 48 }}
              disabled={!canSubmitEsc || isSyncing}
              onClick={handleSubmitEsc}
            >
              {isSyncing ? "Saving…" : "Submit Application"}
            </Button>
          )}
        </div>
      ),
    },
  };

  return (
    <div className="space-y-10">
      {syncError && (
        <p className="rounded border border-red-200 bg-red-50 p-3 text-sm text-red-700">
          {syncError}
        </p>
      )}
      {tabList.map((t, i) => {
        const section = sectionFor[t];
        const number = String(i + 1).padStart(2, "0");
        const action =
          t === "choices" && !isPostSubmission
            ? { label: "Browse More Schools", href: "/browse" }
            : undefined;
        return (
          <AccountSection
            key={t}
            number={number}
            eyebrow={section.eyebrow}
            title={section.title}
            action={action}
          >
            {section.content}
          </AccountSection>
        );
      })}
    </div>
  );
}

// Minimal shape so the row components below don't need to import the full
// SchoolStatusMeta type just for `.title`.
type SchoolStatusMetaLike = { title: string };

// ── Read-only wishlist row — post-submission only, no drag handle and no
// remove button (choices are locked once submitted). ─────────────────────
function WishlistRowContent({
  rank,
  school,
  meta,
}: {
  rank: number;
  school: School;
  meta: SchoolStatusMetaLike | null;
}) {
  return (
    <div className="flex items-start gap-3 rounded-2xl border border-slate-200 bg-white p-3 shadow-sm">
      <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-bold text-primary">
        {rank}
      </span>
      <div className="min-w-0 flex-1">
        <p className="text-sm font-semibold leading-snug text-primary">
          {school.school_name}
        </p>
        <p className="mt-0.5 text-xs text-slate-400">
          {school.school_type === "public"
            ? "Public"
            : school.is_esc_participating
              ? "Private, ESC-participating"
              : "Private, no ESC"}
        </p>
        {meta && (
          <span className="mt-1.5 inline-block rounded border px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wide text-slate-500">
            {meta.title}
          </span>
        )}
      </div>
    </div>
  );
}

// ── Draggable wishlist, pre-submission only. dnd-kit's PointerSensor
// handles mouse; TouchSensor (with a small activation delay so a tap-to-
// scroll gesture isn't mistaken for a drag start) handles touch. ─────────
function SortableWishlist({
  wishlist,
  escStatuses,
  onRemove,
  onReorder,
}: {
  wishlist: School[];
  escStatuses: Record<string, EscSchoolStatus>;
  onRemove: (schoolId: string) => void;
  onReorder: (fromIndex: number, toIndex: number) => void;
}) {
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 4 } }),
    useSensor(TouchSensor, {
      activationConstraint: { delay: 150, tolerance: 8 },
    }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );

  const ids = wishlist.map((s) => s.school_id);

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    if (!over || active.id === over.id) return;
    const fromIndex = ids.indexOf(String(active.id));
    const toIndex = ids.indexOf(String(over.id));
    if (fromIndex === -1 || toIndex === -1) return;
    onReorder(fromIndex, toIndex);
  };

  return (
    <DndContext
      sensors={sensors}
      collisionDetection={closestCenter}
      onDragEnd={handleDragEnd}
    >
      <SortableContext items={ids} strategy={verticalListSortingStrategy}>
        <div className="space-y-3">
          {wishlist.map((school, i) => (
            <SortableWishlistRow
              key={school.school_id}
              rank={i + 1}
              school={school}
              meta={
                escStatuses[school.school_id]
                  ? SCHOOL_STATUS_META[escStatuses[school.school_id]]
                  : null
              }
              onRemove={onRemove}
            />
          ))}
        </div>
      </SortableContext>
    </DndContext>
  );
}

function SortableWishlistRow({
  rank,
  school,
  meta,
  onRemove,
}: {
  rank: number;
  school: School;
  meta: SchoolStatusMetaLike | null;
  onRemove: (schoolId: string) => void;
}) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } =
    useSortable({ id: school.school_id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
  };

  return (
    <div
      ref={setNodeRef}
      style={style}
      className={`flex items-start gap-2 rounded-2xl border border-slate-200 bg-white p-3 ${isDragging ? "z-10 shadow-lg" : "shadow-sm"}`}
    >
      <button
        type="button"
        {...attributes}
        {...listeners}
        aria-label={`Drag to reorder ${school.school_name}`}
        className="flex h-11 w-11 shrink-0 -m-1 cursor-grab touch-none items-center justify-center text-slate-300 hover:text-slate-500 active:cursor-grabbing"
      >
        <GripVertical className="h-5 w-5" />
      </button>
      <span className="mt-1 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-bold text-primary">
        {rank}
      </span>
      <div className="min-w-0 flex-1">
        <p className="text-sm font-semibold leading-snug text-primary">
          {school.school_name}
        </p>
        <p className="mt-0.5 text-xs text-slate-400">
          {school.school_type === "public"
            ? "Public"
            : school.is_esc_participating
              ? "Private, ESC-participating"
              : "Private, no ESC"}
        </p>
        {meta && (
          <span className="mt-1.5 inline-block rounded border px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wide text-slate-500">
            {meta.title}
          </span>
        )}
      </div>
      <button
        type="button"
        onClick={() => onRemove(school.school_id)}
        aria-label={`Remove ${school.school_name}`}
        className="flex h-11 w-11 shrink-0 -m-1 items-center justify-center text-slate-300 hover:text-red-400"
      >
        <X className="h-4 w-4" />
      </button>
    </div>
  );
}

// ── Documents tab content, split out for readability (used once, kept in
// this file rather than a new atomic-design component since it's not
// reused). No longer takes a `setTab` prop — there are no tabs to switch
// between anymore, every section is always visible on the page. ─────────
/** One required-document "bin," in one of three states: empty (no
 * file picked), staged (a file is chosen locally but hasn't been sent
 * anywhere yet - the student hasn't clicked "Submit Documents"), or
 * confirmed (actually on GCS). Only "confirmed" counts toward
 * `docsReady`. Manages its own file input ref and client-side
 * validation error; staging a file is purely local (no network call),
 * so there's no "uploading" state here for that action - only
 * `isPending`, used for the network-backed Remove of an
 * already-confirmed document. */
function DocumentRow({
  doc,
  uploaded,
  stagedFile,
  isSyncing,
  isPending,
  onStage,
  onRemove,
}: {
  doc: string;
  uploaded: boolean;
  stagedFile: File | null;
  isSyncing: boolean;
  isPending: boolean;
  onStage: (doc: string, file: File) => void;
  onRemove: (doc: string) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);
  const filled = uploaded || Boolean(stagedFile);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = ""; // lets the same file be chosen again later
    if (!file) return;
    if (!ALLOWED_DOCUMENT_TYPES.includes(file.type)) {
      setError("Please choose a JPG, PNG, or PDF file.");
      return;
    }
    if (file.size > MAX_DOCUMENT_SIZE_BYTES) {
      setError("This file is too large. Please choose one under 10MB.");
      return;
    }
    setError(null);
    onStage(doc, file);
  };

  return (
    <div
      className={`flex items-start gap-3 rounded-xl border p-3 ${uploaded ? "border-green-200 bg-green-50" : stagedFile ? "border-primary/40 bg-primary/5" : "border-slate-200 bg-white"}`}
    >
      <div
        className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full ${uploaded ? "bg-green-500" : stagedFile ? "border-2 border-primary" : "border-2 border-slate-300"}`}
      >
        {uploaded && <Check className="h-3 w-3 text-white" strokeWidth={3} />}
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-xs leading-snug text-slate-700">{doc}</p>
        {stagedFile && !uploaded && (
          <p className="mt-0.5 truncate text-[10px] text-primary">
            Ready to submit: {stagedFile.name}
          </p>
        )}
        <input
          ref={inputRef}
          type="file"
          accept={ALLOWED_DOCUMENT_TYPES.join(",")}
          onChange={handleFileChange}
          className="hidden"
        />
        {isPending ? (
          <p className="mt-1.5 text-[10px] font-bold uppercase tracking-wide text-slate-400">
            Removing…
          </p>
        ) : (
          <div className="mt-1.5 flex items-center gap-3">
            <button
              type="button"
              disabled={isSyncing}
              onClick={() => inputRef.current?.click()}
              className="flex items-center gap-1 text-[10px] font-bold uppercase tracking-wide text-primary hover:underline disabled:opacity-50"
            >
              {filled ? (
                "Replace File"
              ) : (
                <>
                  <Upload className="h-3 w-3" /> Choose File (JPG, PNG, or
                  PDF, up to 10MB)
                </>
              )}
            </button>
            {filled && (
              <button
                type="button"
                disabled={isSyncing}
                onClick={() => onRemove(doc)}
                className="text-[10px] font-bold uppercase tracking-wide text-red-600 hover:underline disabled:opacity-50"
              >
                Remove
              </button>
            )}
          </div>
        )}
        {error && <p className="mt-1 text-[10px] text-red-600">{error}</p>}
      </div>
    </div>
  );
}

function DocumentsTab({
  applicationState,
  isPostSubmission,
  category,
  requiredDocs,
  uploadedDocs,
  stagedDocs,
  docsReady,
  docUploadProgress,
  stageDoc,
  removeDoc,
  submitDocuments,
  isSyncing,
  docsPendingChoices,
  advanceSchool,
}: {
  applicationState: string;
  isPostSubmission: boolean;
  category: string | null;
  requiredDocs: string[];
  uploadedDocs: string[];
  stagedDocs: Record<string, File>;
  docsReady: boolean;
  docUploadProgress: { completed: number; total: number } | null;
  stageDoc: (doc: string, file: File) => void;
  removeDoc: (doc: string) => Promise<boolean>;
  submitDocuments: () => Promise<boolean>;
  isSyncing: boolean;
  docsPendingChoices: School[];
  advanceSchool: (schoolId: string, toState: EscSchoolStatus) => void;
}) {
  const hasDocsPending = docsPendingChoices.length > 0;
  const [pendingDoc, setPendingDoc] = useState<string | null>(null);
  const stagedCount = Object.keys(stagedDocs).length;

  const handleRemove = async (doc: string) => {
    setPendingDoc(doc);
    await removeDoc(doc);
    setPendingDoc(null);
  };

  if (isPostSubmission && !hasDocsPending) {
    return (
      <div className="space-y-4">
        <div className="rounded-xl border border-blue-200 bg-blue-50 p-4">
          <div className="flex items-start gap-3">
            <FileCheck className="mt-0.5 h-5 w-5 shrink-0 text-blue-500" />
            <div>
              <p className="text-sm font-semibold text-blue-800">
                Documents Submitted
              </p>
              <p className="mt-1 text-xs leading-relaxed text-blue-600">
                The school committee is reviewing your documents. You&apos;ll
                be notified here if anything else is needed.
              </p>
            </div>
          </div>
        </div>
        {uploadedDocs.length > 0 && (
          <>
            <p className={SECTION_LABEL}>Submitted Documents</p>
            <div className="space-y-2">
              {uploadedDocs.map((doc) => (
                <div
                  key={doc}
                  className="flex items-start gap-3 rounded-xl border border-green-200 bg-green-50 p-3"
                >
                  <div className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-green-500">
                    <Check className="h-3 w-3 text-white" strokeWidth={3} />
                  </div>
                  <p className="text-xs leading-snug text-slate-700">{doc}</p>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    );
  }

  return (
    <div>
      {applicationState === "not_eligible" ? (
        <p className="text-sm text-slate-500">
          You&apos;re not eligible for the ESC fee subsidy, so no ESC
          documents are required. You can still enroll directly at a school.
        </p>
      ) : !category ? (
        <p className="text-sm text-slate-500">
          Complete your eligibility check first to see which documents you
          need.
        </p>
      ) : (
        <>
          {hasDocsPending && (
            <div className="mb-4 rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs leading-relaxed text-amber-800">
              {docsPendingChoices.length === 1
                ? `${docsPendingChoices[0].school_name}'s ESC School Committee has requested an additional document. Please add it below.`
                : "Some of your schools' ESC Committees have requested additional documents. Please add them below."}
            </div>
          )}
          <p className={`${SECTION_LABEL} mb-4`}>
            Required Documents — Category {category}
          </p>
          <div className="space-y-3">
            {requiredDocs.map((doc) => (
              <DocumentRow
                key={doc}
                doc={doc}
                uploaded={uploadedDocs.includes(doc)}
                stagedFile={stagedDocs[doc] ?? null}
                isSyncing={isSyncing}
                isPending={pendingDoc === doc}
                onStage={stageDoc}
                onRemove={handleRemove}
              />
            ))}
          </div>
          {docUploadProgress ? (
            <div className="mt-4">
              <div className="mb-1 flex items-center justify-between text-[10px] font-bold uppercase tracking-wide text-slate-500">
                <span>
                  Uploading document {docUploadProgress.completed + 1} of{" "}
                  {docUploadProgress.total}
                </span>
                <span>
                  {Math.round(
                    (docUploadProgress.completed / docUploadProgress.total) *
                      100
                  )}
                  %
                </span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-slate-200">
                <div
                  className="h-full rounded-full bg-primary transition-all duration-300"
                  style={{
                    width: `${(docUploadProgress.completed / docUploadProgress.total) * 100}%`,
                  }}
                />
              </div>
            </div>
          ) : stagedCount > 0 ? (
            <Button
              fullWidth
              variant="contained"
              sx={{ minHeight: 48, mt: 2 }}
              disabled={isSyncing}
              onClick={submitDocuments}
            >
              Submit Documents ({stagedCount})
            </Button>
          ) : (
            docsReady && (
              <div className="mt-4 rounded-lg border border-green-200 bg-green-50 p-3 text-center text-xs font-medium text-green-700">
                All documents submitted
              </div>
            )
          )}
          {hasDocsPending && docsReady && (
            <div className="mt-2 space-y-2">
              {docsPendingChoices.map((school) => (
                <Button
                  key={school.school_id}
                  fullWidth
                  variant="contained"
                  sx={{ minHeight: 48 }}
                  onClick={() =>
                    advanceSchool(school.school_id, "docs_submitted")
                  }
                >
                  Submit Additional Document to {school.school_name}
                </Button>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
