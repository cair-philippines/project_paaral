"use client";

import { useState } from "react";
import { Heart, Check } from "lucide-react";
import Dialog from "@mui/material/Dialog";
import DialogTitle from "@mui/material/DialogTitle";
import DialogContent from "@mui/material/DialogContent";
import DialogActions from "@mui/material/DialogActions";
import Button from "@mui/material/Button";
import { useApplication } from "@/components/templates/ApplicationStateProvider";
import { MAX_WISHLIST_SIZE } from "@/lib/applicationState";
import type { School } from "@/types/school";

interface AddToWishlistButtonProps {
  school: School;
  variant?: "compact" | "full";
}

/** Add/remove a school from the learner's wishlist. Requires an account —
 * clicking while logged out opens the login modal instead, rather than
 * silently doing nothing or erroring. Hidden entirely once the application
 * has been submitted, since the wishlist becomes read-only at that point.
 *
 * Once the ranked list is at MAX_WISHLIST_SIZE, "add" stays tappable
 * (never uses the native `disabled` attribute) so a dialog can explain
 * why nothing happened - a plain `disabled` button would silently do
 * nothing on tap, which is exactly the confusing gap this exists to
 * avoid, especially on touch devices where there's no hover to reveal a
 * tooltip. The button is still styled muted so the limit is visible at a
 * glance too, not just on the one time someone taps it. A real `Dialog`
 * (centered, no auto-dismiss timer - the student closes it explicitly)
 * rather than a corner Snackbar, per Paula's direct instruction
 * (2026-09-07): a limit worth interrupting for deserves a screen you
 * have to actually read and dismiss, not a toast that can disappear
 * before it's noticed. `Dialog` already portals to `document.body`
 * itself, so this doesn't need the manual portal workaround the old
 * Snackbar required to escape a transformed Mapbox-popup ancestor. */
export default function AddToWishlistButton({
  school,
  variant = "full",
}: AddToWishlistButtonProps) {
  const {
    account,
    isPostSubmission,
    wishlist,
    isInWishlist,
    addToWishlist,
    removeFromWishlist,
    openLoginModal,
  } = useApplication();
  const [showLimitMessage, setShowLimitMessage] = useState(false);

  if (account && isPostSubmission) return null;

  const inList = isInWishlist(school.school_id);
  const atLimit = !inList && wishlist.length >= MAX_WISHLIST_SIZE;

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    if (!account) {
      openLoginModal();
      return;
    }
    if (inList) {
      removeFromWishlist(school.school_id);
      return;
    }
    if (atLimit) {
      setShowLimitMessage(true);
      return;
    }
    addToWishlist(school.school_id);
  };

  const label = inList
    ? "Added to My Choices"
    : atLimit
      ? "Your ranked list is full"
      : "Add to My Choices";

  const limitDialog = (
    <Dialog
      open={showLimitMessage}
      onClose={() => setShowLimitMessage(false)}
      fullWidth
      maxWidth="xs"
    >
      <DialogTitle>Your ranked list is full</DialogTitle>
      <DialogContent>
        <p className="text-sm text-slate-600">
          {`You already have ${MAX_WISHLIST_SIZE} schools in your ranked list, the most allowed. Remove one to add a different school.`}
        </p>
      </DialogContent>
      <DialogActions sx={{ px: 3, pb: 3 }}>
        <Button
          fullWidth
          variant="contained"
          sx={{ minHeight: 44 }}
          onClick={() => setShowLimitMessage(false)}
        >
          Got It
        </Button>
      </DialogActions>
    </Dialog>
  );

  if (variant === "compact") {
    return (
      <>
        <button
          type="button"
          onClick={handleClick}
          aria-label={label}
          aria-pressed={inList}
          className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-full border transition ${
            inList
              ? "border-primary bg-primary text-white"
              : atLimit
                ? "border-slate-300 bg-slate-100 text-slate-500"
                : "border-slate-200 bg-white text-slate-400 hover:border-primary hover:text-primary"
          }`}
        >
          {inList ? (
            <Check className="h-4 w-4" strokeWidth={3} />
          ) : (
            <Heart className="h-4 w-4" />
          )}
        </button>
        {limitDialog}
      </>
    );
  }

  return (
    <>
      <button
        type="button"
        onClick={handleClick}
        aria-pressed={inList}
        className={`flex h-11 w-full items-center justify-center gap-2 rounded-lg text-sm font-semibold transition ${
          inList
            ? "border border-primary bg-primary/5 text-primary"
            : atLimit
              ? "border border-slate-300 bg-slate-100 text-slate-500"
              : "bg-primary text-white hover:opacity-90"
        }`}
      >
        {inList ? (
          <Check className="h-4 w-4" strokeWidth={3} />
        ) : (
          <Heart className="h-4 w-4" />
        )}
        {label}
      </button>
      {limitDialog}
    </>
  );
}
