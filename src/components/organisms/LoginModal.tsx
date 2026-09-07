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
  verifyLoginEmail,
  type LoginLookupResult,
} from "@/hooks/useApplicationState";
import { ApiError } from "@/lib/api";
import {
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
 *
 * A returning student (an Application row already exists for this
 * LRN, whatever it has or hasn't got saved yet) goes straight into
 * the app with no extra screen, per Paula's direct instruction
 * (2026-09-07) — real hydration (Chunk 22) restores their actual
 * saved state, so there's no "load your draft or start fresh?"
 * question left to ask. Only a genuinely new learner — no Application
 * row yet, confirmed by a 404 from the hydration fetch — sees the
 * one-step "create my account" confirmation, which now calls a real
 * backend endpoint (`createNewAccount`) that creates that row before
 * starting the eligibility check. */
export default function LoginModal({ open, onClose }: LoginModalProps) {
  const router = useRouter();
  const { createAccount, createNewAccount } = useApplication();

  const [email, setEmail] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState("");
  const [lookup, setLookup] = useState<LoginLookupResult | null>(null);

  const reset = () => {
    setEmail("");
    setError("");
    setLoading(false);
    setCreating(false);
    setCreateError("");
    setLookup(null);
  };

  const handleClose = () => {
    reset();
    onClose();
  };

  const handleContinue = async () => {
    setError("");
    setLoading(true);
    const result = await verifyLoginEmail(email);
    if (!result.ok) {
      setLoading(false);
      setError(result.error ?? "Email not found.");
      return;
    }

    // Hydrate immediately - this also decides new vs. existing, so a
    // returning student is only ever asked to click "Continue" once,
    // not verified then asked a second, now-meaningless question. A
    // 404 here means this LRN has no Application row yet (a genuinely
    // new learner) - that's not an error, it's what shows the
    // "Create My Account" screen below. Anything else is a real
    // failure (network drop, server down).
    try {
      const account = await createAccount(result.lrn!);
      setLoading(false);
      reset();
      onClose();
      router.push(account.isEligible !== null ? "/browse" : "/eligibility");
    } catch (err) {
      setLoading(false);
      if (err instanceof ApiError && err.status === 404) {
        setLookup(result);
      } else {
        setError(
          "Couldn't reach the PAARAL server. Check your connection and try again."
        );
      }
    }
  };

  const handleUseDifferentEmail = () => {
    setLookup(null);
    setError("");
  };

  // Only reachable for a genuinely new account (no Application row
  // yet) - actually creates it server-side now, then navigates.
  const handleCreateAccount = async () => {
    if (!lookup?.lrn) return;
    setCreateError("");
    setCreating(true);
    try {
      await createNewAccount(lookup.lrn);
      setCreating(false);
      reset();
      onClose();
      router.push("/eligibility");
    } catch {
      setCreating(false);
      setCreateError(
        "Couldn't create your account. Check your connection and try again."
      );
    }
  };

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

            <p className="mb-4 text-sm text-slate-600">
              Creating your PAARAL account starts your ESC eligibility
              check — a few short questions to see if your family
              qualifies for a school-fee subsidy.
            </p>
            {createError && (
              <p className="mb-3 rounded border border-red-200 bg-red-50 p-2 text-xs text-red-700">
                {createError}
              </p>
            )}
            <Button
              fullWidth
              variant="contained"
              sx={{ minHeight: 48 }}
              disabled={creating}
              onClick={handleCreateAccount}
            >
              {creating ? (
                <CircularProgress size={20} color="inherit" />
              ) : (
                "Create My Account & Continue"
              )}
            </Button>
            <Button
              fullWidth
              variant="text"
              size="small"
              sx={{ mt: 1, color: "text.secondary" }}
              disabled={creating}
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
