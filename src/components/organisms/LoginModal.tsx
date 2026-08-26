"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Dialog from "@mui/material/Dialog";
import DialogTitle from "@mui/material/DialogTitle";
import DialogContent from "@mui/material/DialogContent";
import IconButton from "@mui/material/IconButton";
import TextField from "@mui/material/TextField";
import Button from "@mui/material/Button";
import CircularProgress from "@mui/material/CircularProgress";
import { X, CheckCircle2 } from "lucide-react";
import { useApplication } from "@/components/templates/ApplicationStateProvider";
import {
  validateLoginEmail,
  type LoginLookupResult,
} from "@/hooks/useApplicationState";
import { getSchoolById, getTypeBadge, titleCase } from "@/lib/schools";
import {
  DRAFT_WISHLIST_SCHOOL_IDS,
  LEARNER_RECORD,
  TEST_EMAIL,
  TEST_EMAIL_WITH_DRAFT,
} from "@/lib/constants";

interface LoginModalProps {
  open: boolean;
  onClose: () => void;
}

/** DepEd ICTS sign-in stand-in — a plain email lookup against the demo
 * DepEd Learner Information System (LIS), not a real password/SSO flow.
 * Two steps: enter email → confirm the learner record found (with an extra
 * "you have saved choices" branch for the draft-account demo LRN). */
export default function LoginModal({ open, onClose }: LoginModalProps) {
  const router = useRouter();
  const { createAccount } = useApplication();

  const [email, setEmail] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [lookup, setLookup] = useState<LoginLookupResult | null>(null);

  const reset = () => {
    setEmail("");
    setError("");
    setLoading(false);
    setLookup(null);
  };

  const handleClose = () => {
    reset();
    onClose();
  };

  const handleContinue = () => {
    setError("");
    const result = validateLoginEmail(email);
    if (!result.ok) {
      setError(result.error ?? "Email not found.");
      return;
    }
    setLoading(true);
    // Simulated LIS lookup delay, matching the original mockup's convention
    // for making the "we checked a real registry" step feel real.
    setTimeout(() => {
      setLoading(false);
      setLookup(result);
    }, 800);
  };

  const handleUseDifferentEmail = () => {
    setLookup(null);
    setError("");
  };

  const startApplication = (wishlistIds: string[]) => {
    if (!lookup?.lrn) return;
    createAccount(lookup.lrn, wishlistIds);
    reset();
    onClose();
    router.push("/eligibility");
  };

  const draftSchools = DRAFT_WISHLIST_SCHOOL_IDS.map((id) =>
    getSchoolById(id)
  ).filter((s) => s !== undefined);

  return (
    <Dialog
      open={open}
      onClose={handleClose}
      fullWidth
      maxWidth="xs"
      aria-labelledby="login-modal-title"
    >
      <DialogTitle
        id="login-modal-title"
        sx={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", pr: 2 }}
      >
        <span>
          <span className="mb-1 block text-[10px] font-bold uppercase tracking-widest text-slate-500">
            DepEd ICTS Sign-In
          </span>
          <span className="block text-lg font-bold text-primary">
            {lookup ? "Learner Found" : "Log In to PAARAL"}
          </span>
        </span>
        <IconButton onClick={handleClose} size="small" aria-label="Close">
          <X className="h-4 w-4" />
        </IconButton>
      </DialogTitle>

      <DialogContent className="!pt-1">
        {!lookup ? (
          <div className="pb-2">
            <p className="mb-4 text-sm text-slate-600">
              Enter your DepEd email address. This is usually your Learner
              Reference Number (LRN) followed by{" "}
              <span className="font-mono text-slate-800">@deped.gov.ph</span>.
            </p>
            <TextField
              fullWidth
              type="email"
              label="DepEd email address"
              placeholder="e.g. 100000000001@deped.gov.ph"
              value={email}
              onChange={(e) => {
                setEmail(e.target.value);
                setError("");
              }}
              onKeyDown={(e) => e.key === "Enter" && handleContinue()}
              error={Boolean(error)}
              helperText={error || " "}
              autoFocus
            />
            <Button
              fullWidth
              variant="contained"
              onClick={handleContinue}
              disabled={loading || !email}
              sx={{ minHeight: 48, mt: 1 }}
            >
              {loading ? (
                <CircularProgress size={20} color="inherit" />
              ) : (
                "Continue"
              )}
            </Button>
            <p className="mt-4 text-center text-xs text-slate-400">
              Demo accounts: <br />
              <span className="font-mono">{TEST_EMAIL}</span> (new
              application)
              <br />
              <span className="font-mono">{TEST_EMAIL_WITH_DRAFT}</span>{" "}
              (has saved choices)
            </p>
          </div>
        ) : (
          <div className="pb-2">
            <div className="mb-5 rounded-xl border border-green-200 bg-green-50 p-4">
              <div className="mb-2 flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 shrink-0 text-green-600" />
                <p className="text-[10px] font-bold uppercase tracking-widest text-green-700">
                  Found in the Learner Information System (LIS)
                </p>
              </div>
              <p className="text-base font-bold text-slate-800">
                {LEARNER_RECORD.firstName} {LEARNER_RECORD.mi}.{" "}
                {LEARNER_RECORD.lastName}
              </p>
              <p className="mt-1 text-xs text-slate-500">
                {LEARNER_RECORD.school} &middot; {LEARNER_RECORD.grade}
              </p>
              <p className="text-xs text-slate-500">
                {LEARNER_RECORD.municipality} &middot;{" "}
                {LEARNER_RECORD.division}
              </p>
              <p className="mt-1 font-mono text-xs text-slate-400">
                LRN: {lookup.lrn}
              </p>
            </div>

            {lookup.hasDraft ? (
              <>
                <p className="mb-3 text-sm text-slate-600">
                  You already started a list of schools last time. You can
                  pick up where you left off, or start over.
                </p>
                <ul className="mb-4 space-y-2">
                  {draftSchools.map((school) => {
                    const badge = getTypeBadge(school!);
                    return (
                      <li
                        key={school!.school_id}
                        className="rounded-lg border border-slate-200 p-3"
                      >
                        <p className="text-sm font-semibold leading-snug text-primary">
                          {school!.school_name}
                        </p>
                        <div className="mt-1 flex items-center gap-2">
                          <p className="text-xs text-slate-500">
                            {titleCase(school!.deped_barangay)}, Quezon City
                          </p>
                          <span
                            className={`rounded-full px-1.5 py-0.5 text-[9px] font-bold uppercase ${badge.className}`}
                          >
                            {badge.label}
                          </span>
                        </div>
                      </li>
                    );
                  })}
                </ul>
                <Button
                  fullWidth
                  variant="contained"
                  sx={{ minHeight: 48 }}
                  onClick={() => startApplication(DRAFT_WISHLIST_SCHOOL_IDS)}
                >
                  Continue With My Saved Choices
                </Button>
                <Button
                  fullWidth
                  variant="text"
                  sx={{ minHeight: 44, mt: 1 }}
                  onClick={() => startApplication([])}
                >
                  Start Fresh Instead
                </Button>
              </>
            ) : (
              <>
                <p className="mb-4 text-sm text-slate-600">
                  Creating your PAARAL account starts your ESC eligibility
                  check — a few short questions to see if your family
                  qualifies for a school-fee subsidy.
                </p>
                <Button
                  fullWidth
                  variant="contained"
                  sx={{ minHeight: 48 }}
                  onClick={() => startApplication([])}
                >
                  Create My Account &amp; Continue
                </Button>
              </>
            )}
            <Button
              fullWidth
              variant="text"
              size="small"
              sx={{ mt: 1, color: "text.secondary" }}
              onClick={handleUseDifferentEmail}
            >
              ← Use a different email
            </Button>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
