import type {
  EligAnswers,
  EligStep,
  EmploymentStatus,
  EscCategory,
} from "@/types/application";

export const catMeta: Record<
  "A" | "B" | "C" | "D",
  { label: string; tw: string; desc: string }
> = {
  A: {
    label: "Category A — Social Equity Group",
    tw: "bg-[#f5f3ff] border-[#c4b5fd] text-[#5b21b6]",
    desc: "Highest priority. Applies to learners from equity-protected groups regardless of income.",
  },
  B: {
    label: "Category B — Public School Graduate",
    tw: "bg-[#eff6ff] border-[#bfdbfe] text-[#1d4ed8]",
    desc: "Public school graduate with poor to middle-class household income.",
  },
  C: {
    label: "Category C — ALS / PEPT Passer",
    tw: "bg-[#f0fdfa] border-[#99f6e4] text-[#0f766e]",
    desc: "ALS A&E Test or PEPT passer with eligible income.",
  },
  D: {
    label: "Category D — Private School Graduate",
    tw: "bg-[#fffbeb] border-[#fde68a] text-[#b45309]",
    desc: "Private school graduate with poor to middle-class household income.",
  },
};

/** Category A–D determination — see CLAUDE.md "ESC Category Logic". */
export function computeCategory(answers: EligAnswers): EscCategory {
  const { schoolType, segs, income } = answers;
  const hasSegs = segs.some((s) => s !== "none");
  if (schoolType === "public" && hasSegs) return "A";
  if (income === "above") return null;
  if (schoolType === "public") return "B";
  if (schoolType === "als") return "C";
  if (schoolType === "private") return "D";
  return null;
}

/** Eligibility questionnaire's branching transition graph. */
export function nextEligStep(
  step: EligStep,
  answers: EligAnswers
): EligStep {
  if (step === "schoolType") {
    return answers.schoolType === "public" ? "seg" : "income";
  }
  if (step === "seg") {
    return answers.segs.some((s) => s !== "none") ? "result" : "income";
  }
  if (step === "income") {
    return answers.income === "above" ? "result" : "employment";
  }
  return "result";
}

const INCOME_PROOF_DOC: Record<EmploymentStatus, string> = {
  local: "Income Tax Return (ITR), Certificate of Employment, or recent Payslip",
  abroad: "Certificate of Employment, Employment Contract, or recent Payslip",
  business: "Notarized Affidavit (business income)",
  unemployed: "Certificate of Tax Exemption or Barangay Certificate of Indigency",
};

const SEG_DOC: Record<string, string> = {
  "4ps": "Copy of 4Ps ID (DSWD)",
  gidca: "Barangay Certification of GIDCA residency",
  ip: "Certificate of Indigenous People Membership — NCIP",
  pwd: "PWD ID issued by LGU",
  special: "Medical or psychological assessment",
  cbms: "CBMS poverty assessment document",
};

/** Tailored document checklist for a determined category. */
export function getDocList(
  category: EscCategory,
  answers: EligAnswers
): string[] {
  const base = [
    "Valid ID (National ID, Birth Certificate, or Passport)",
    "Accomplished ESC Application Form (Annex D)",
  ];
  const affidavit = `Affidavit of Family's Financial Capacity (Annex F) — ${
    (answers.employment && INCOME_PROOF_DOC[answers.employment]) ||
    "income proof document"
  }`;

  if (category === "A") {
    const extra = answers.segs
      .filter((s) => s !== "none" && SEG_DOC[s])
      .map((s) => SEG_DOC[s]);
    return [...base, "SF9 — Learner's Progress Report Card", ...extra];
  }
  if (category === "B") {
    return [...base, "SF9 — Learner's Progress Report Card", affidavit];
  }
  if (category === "C") {
    return [...base, "Certificate of Rating from BEA (ALS A&E / PEPT)", affidavit];
  }
  if (category === "D") {
    return [...base, "SF9 — Learner's Progress Report Card", affidavit];
  }
  return [];
}
