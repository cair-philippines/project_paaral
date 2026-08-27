"use client";

import { useRouter } from "next/navigation";
import Button from "@mui/material/Button";
import { Check, FileCheck } from "lucide-react";
import SiteHeader from "@/components/organisms/SiteHeader";
import { useApplication } from "@/components/templates/ApplicationStateProvider";
import { catMeta, computeCategory, getDocList, nextEligStep } from "@/lib/eligibility";
import type {
  EmploymentStatus,
  Grade6Pathway,
  IncomeBracket,
  Seg,
} from "@/types/application";

const STEP_ORDER = ["schoolType", "seg", "income", "employment", "result"] as const;

const OPTION_CARD =
  "w-full rounded-xl border border-slate-200 bg-white p-4 text-left shadow-sm transition hover:border-primary hover:bg-primary/5";

export default function EligibilityPage() {
  const router = useRouter();
  const {
    account,
    eligStep,
    eligHistory,
    eligAnswers,
    eligBack,
    eligGo,
    eligRestart,
    patchEligAnswers,
    completeEligibility,
    openLoginModal,
  } = useApplication();

  if (!account) {
    return (
      <div className="flex min-h-screen flex-col bg-background">
        <SiteHeader />
        <div className="flex flex-1 flex-col items-center justify-center gap-4 p-8 text-center">
          <p className="text-sm text-slate-500">
            Please log in to start your ESC eligibility check.
          </p>
          <Button variant="contained" sx={{ minHeight: 48 }} onClick={openLoginModal}>
            Log In
          </Button>
        </div>
      </div>
    );
  }

  const catResult = eligStep === "result" ? computeCategory(eligAnswers) : null;
  const docList = catResult ? getDocList(catResult, eligAnswers) : [];

  return (
    <div className="flex min-h-screen flex-col bg-slate-50">
      <SiteHeader />
      <div className="flex flex-1 items-start justify-center px-4 py-12">
        <div className="w-full max-w-lg rounded-2xl border border-slate-200 bg-white p-8 shadow-sm">
          <div className="mb-6 flex gap-2">
            {STEP_ORDER.map((s) => (
              <div
                key={s}
                className={`h-1.5 flex-1 rounded-full transition-colors ${
                  s === eligStep
                    ? "bg-primary"
                    : eligHistory.some((h) => h.step === s)
                      ? "bg-primary/40"
                      : "bg-slate-200"
                }`}
              />
            ))}
          </div>

          <p className="mb-1 text-[10px] font-bold uppercase tracking-widest text-slate-500">
            ESC Eligibility Check
          </p>

          {eligStep === "schoolType" && (
            <>
              <h1 className="mb-2 text-xl font-bold text-primary">
                How did you complete Grade 6?
              </h1>
              <p className="mb-6 text-sm text-slate-500">
                This is the first step in figuring out whether your family
                qualifies for a school-fee subsidy called ESC (Educational
                Service Contracting).
              </p>
              <div className="space-y-3">
                {(
                  [
                    {
                      v: "public",
                      label: "Public elementary school",
                      sub: "Any school run by DepEd",
                    },
                    {
                      v: "private",
                      label: "Private elementary school",
                      sub: "A non-DepEd-operated school",
                    },
                    {
                      v: "als",
                      label: "Alternative Learning System (ALS) or PEPT",
                      sub: "Passed the ALS Accreditation & Equivalency Test or the Philippine Educational Placement Test",
                    },
                  ] satisfies { v: Grade6Pathway; label: string; sub: string }[]
                ).map((opt) => (
                  <button
                    key={opt.v}
                    type="button"
                    onClick={() =>
                      eligGo(
                        nextEligStep("schoolType", {
                          ...eligAnswers,
                          schoolType: opt.v,
                        }),
                        { schoolType: opt.v }
                      )
                    }
                    className={`${OPTION_CARD} group`}
                  >
                    <p className="text-sm font-semibold text-slate-800 group-hover:text-primary">
                      {opt.label}
                    </p>
                    <p className="mt-0.5 text-xs text-slate-400">{opt.sub}</p>
                  </button>
                ))}
              </div>
            </>
          )}

          {eligStep === "seg" && (
            <>
              <h1 className="mb-2 text-xl font-bold text-primary">
                Do you belong to a Social Equity Group?
              </h1>
              <p className="mb-5 text-sm text-slate-500">
                These are groups DepEd gives the highest ESC priority to.
                Select all that apply to your family.
              </p>
              <div className="mb-6 space-y-2">
                {(
                  [
                    { v: "4ps", label: "4Ps (Pantawid Pamilyang Pilipino Program)" },
                    {
                      v: "gidca",
                      label:
                        "Geographically Isolated and Disadvantaged Community (GIDCA)",
                    },
                    { v: "ip", label: "Indigenous People (IP)" },
                    { v: "pwd", label: "Person with Disability (PWD)" },
                    { v: "special", label: "Child with Special Needs" },
                    {
                      v: "cbms",
                      label: "Identified as poor or near-poor (CBMS survey)",
                    },
                    { v: "none", label: "None of the above" },
                  ] satisfies { v: Seg; label: string }[]
                ).map((opt) => {
                  const checked = eligAnswers.segs.includes(opt.v);
                  return (
                    <button
                      key={opt.v}
                      type="button"
                      onClick={() => {
                        let segs = eligAnswers.segs;
                        if (opt.v === "none") {
                          segs = checked ? [] : ["none"];
                        } else {
                          segs = checked
                            ? segs.filter((s) => s !== opt.v)
                            : [...segs.filter((s) => s !== "none"), opt.v];
                        }
                        patchEligAnswers({ segs });
                      }}
                      className={`flex w-full items-center gap-3 rounded-xl border p-3 text-left text-sm shadow-sm transition ${
                        checked
                          ? "border-primary bg-primary/5 font-medium text-primary"
                          : "border-slate-200 bg-white text-slate-700 hover:border-slate-300"
                      }`}
                    >
                      <span
                        className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-md border ${
                          checked ? "border-primary bg-primary" : "border-slate-300"
                        }`}
                      >
                        {checked && (
                          <Check className="h-3.5 w-3.5 text-white" strokeWidth={3} />
                        )}
                      </span>
                      {opt.label}
                    </button>
                  );
                })}
              </div>
              <div className="flex gap-2">
                <Button variant="outlined" sx={{ minHeight: 48 }} onClick={eligBack}>
                  Back
                </Button>
                <Button
                  fullWidth
                  variant="contained"
                  sx={{ minHeight: 48 }}
                  disabled={eligAnswers.segs.length === 0}
                  onClick={() => eligGo(nextEligStep("seg", eligAnswers))}
                >
                  Continue
                </Button>
              </div>
            </>
          )}

          {eligStep === "income" && (
            <>
              <h1 className="mb-2 text-xl font-bold text-primary">
                Monthly household income
              </h1>
              <p className="mb-6 text-sm text-slate-500">
                Add up the income of everyone in your household per month.
                This uses the PIDS income classification.
              </p>
              <div className="mb-6 space-y-3">
                {(
                  [
                    { v: "poor", label: "Poor", sub: "Less than ₱10,957/month" },
                    { v: "low", label: "Low income", sub: "₱10,957 – ₱21,194/month" },
                    {
                      v: "lower_middle",
                      label: "Lower middle class",
                      sub: "₱21,194 – ₱43,828/month",
                    },
                    {
                      v: "middle",
                      label: "Middle class",
                      sub: "₱43,828 – ₱76,669/month",
                    },
                    {
                      v: "above",
                      label: "Upper middle income or above",
                      sub: "More than ₱76,669/month — not eligible for an ESC subsidy",
                    },
                  ] satisfies { v: IncomeBracket; label: string; sub: string }[]
                ).map((opt) => (
                  <button
                    key={opt.v}
                    type="button"
                    onClick={() =>
                      eligGo(
                        nextEligStep("income", { ...eligAnswers, income: opt.v }),
                        { income: opt.v }
                      )
                    }
                    className={`${OPTION_CARD} group`}
                  >
                    <p className="text-sm font-semibold text-slate-800 group-hover:text-primary">
                      {opt.label}
                    </p>
                    <p className="mt-0.5 text-xs text-slate-400">{opt.sub}</p>
                  </button>
                ))}
              </div>
              <Button variant="outlined" sx={{ minHeight: 48 }} onClick={eligBack}>
                Back
              </Button>
            </>
          )}

          {eligStep === "employment" && (
            <>
              <h1 className="mb-2 text-xl font-bold text-primary">
                Parent or guardian&apos;s employment
              </h1>
              <p className="mb-6 text-sm text-slate-500">
                This tells us which income document to ask for later.
              </p>
              <div className="mb-6 space-y-3">
                {(
                  [
                    {
                      v: "local",
                      label: "Employed in the Philippines",
                      sub: "Salaried employee",
                    },
                    {
                      v: "abroad",
                      label: "Overseas Filipino Worker (OFW)",
                      sub: "Working abroad",
                    },
                    {
                      v: "business",
                      label: "Self-employed or business owner",
                      sub: "Entrepreneur, freelancer, or sole proprietor",
                    },
                    {
                      v: "unemployed",
                      label: "Unemployed or informal livelihood",
                      sub: "No formal employer or fixed income",
                    },
                  ] satisfies { v: EmploymentStatus; label: string; sub: string }[]
                ).map((opt) => (
                  <button
                    key={opt.v}
                    type="button"
                    onClick={() =>
                      eligGo(nextEligStep("employment", eligAnswers), {
                        employment: opt.v,
                      })
                    }
                    className={`${OPTION_CARD} group`}
                  >
                    <p className="text-sm font-semibold text-slate-800 group-hover:text-primary">
                      {opt.label}
                    </p>
                    <p className="mt-0.5 text-xs text-slate-400">{opt.sub}</p>
                  </button>
                ))}
              </div>
              <Button variant="outlined" sx={{ minHeight: 48 }} onClick={eligBack}>
                Back
              </Button>
            </>
          )}

          {eligStep === "result" && (
            <>
              <h1 className="mb-4 text-xl font-bold text-primary">
                Your ESC Eligibility Result
              </h1>
              {catResult ? (
                <>
                  <div className={`mb-5 rounded-xl border p-4 ${catMeta[catResult].tw}`}>
                    <p className="text-sm font-bold">{catMeta[catResult].label}</p>
                    <p className="mt-1 text-xs opacity-80">
                      {catMeta[catResult].desc}
                    </p>
                  </div>
                  <div className="mb-5">
                    <p className="mb-3 text-[10px] font-bold uppercase tracking-widest text-slate-500">
                      Documents you&apos;ll need to prepare
                    </p>
                    <ul className="space-y-2">
                      {docList.map((doc) => (
                        <li
                          key={doc}
                          className="flex items-start gap-2 text-sm text-slate-700"
                        >
                          <FileCheck className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
                          {doc}
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div className="mb-5 rounded-lg border border-amber-200 bg-amber-50 p-3">
                    <p className="text-xs text-amber-700">
                      This is your own self-assessment. The ESC School
                      Committee makes the final decision once you apply.
                    </p>
                  </div>
                  <Button
                    fullWidth
                    variant="contained"
                    size="large"
                    sx={{ minHeight: 48 }}
                    onClick={() => {
                      completeEligibility();
                      router.push("/browse");
                    }}
                  >
                    Continue to Browse Schools
                  </Button>
                </>
              ) : (
                <>
                  <div className="mb-5 rounded-xl border border-red-200 bg-red-50 p-4">
                    <p className="text-sm font-bold text-red-700">
                      Not eligible for an ESC subsidy
                    </p>
                    <p className="mt-1 text-xs text-red-600">
                      Households above the middle-class income threshold
                      don&apos;t qualify for ESC categories B, C, or D. You
                      can still enroll at any school at full cost.
                    </p>
                  </div>
                  <Button
                    fullWidth
                    variant="outlined"
                    sx={{ minHeight: 48, mb: 1.5 }}
                    onClick={eligRestart}
                  >
                    Start Over
                  </Button>
                  <Button
                    fullWidth
                    variant="contained"
                    sx={{ minHeight: 48 }}
                    onClick={() => {
                      completeEligibility();
                      router.push("/browse");
                    }}
                  >
                    Browse Schools Without ESC
                  </Button>
                </>
              )}
              <Button
                sx={{ mt: 1.5, color: "text.secondary" }}
                size="small"
                onClick={eligBack}
              >
                ← Back
              </Button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
