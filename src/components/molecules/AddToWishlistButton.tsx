"use client";

import { Heart, Check } from "lucide-react";
import { useApplication } from "@/components/templates/ApplicationStateProvider";
import type { School } from "@/types/school";

interface AddToWishlistButtonProps {
  school: School;
  variant?: "compact" | "full";
}

/** Add/remove a school from the learner's wishlist. Requires an account —
 * clicking while logged out opens the login modal instead, rather than
 * silently doing nothing or erroring. Hidden entirely once the application
 * has been submitted, since the wishlist becomes read-only at that point. */
export default function AddToWishlistButton({
  school,
  variant = "full",
}: AddToWishlistButtonProps) {
  const {
    account,
    isPostSubmission,
    isInWishlist,
    addToWishlist,
    removeFromWishlist,
    openLoginModal,
  } = useApplication();

  if (account && isPostSubmission) return null;

  const inList = isInWishlist(school.school_id);

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    if (!account) {
      openLoginModal();
      return;
    }
    if (inList) {
      removeFromWishlist(school.school_id);
    } else {
      addToWishlist(school.school_id);
    }
  };

  const label = inList ? "Added to My Choices" : "Add to My Choices";

  if (variant === "compact") {
    return (
      <button
        type="button"
        onClick={handleClick}
        aria-label={label}
        aria-pressed={inList}
        className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-full border transition ${
          inList
            ? "border-primary bg-primary text-white"
            : "border-slate-200 bg-white text-slate-400 hover:border-primary hover:text-primary"
        }`}
      >
        {inList ? (
          <Check className="h-4 w-4" strokeWidth={3} />
        ) : (
          <Heart className="h-4 w-4" />
        )}
      </button>
    );
  }

  return (
    <button
      type="button"
      onClick={handleClick}
      aria-pressed={inList}
      className={`flex h-11 w-full items-center justify-center gap-2 rounded-lg text-sm font-semibold transition ${
        inList
          ? "border border-primary bg-primary/5 text-primary"
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
  );
}
