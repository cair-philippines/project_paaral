/**
 * Thin gradient divider line — a small decorative flourish under the hero
 * headline. Retired the old Philippine-flag three-stripe treatment
 * (2026-08-24 recolor pass) in favor of a single bar using the new
 * palette's primary→secondary gradient token, kept distinct from the
 * hero's own primary→accent background gradient so the two don't blend
 * into a flat wash of one color.
 */
export default function TricolorRule({ className = "" }: { className?: string }) {
  return (
    <div
      className={`h-[3px] w-full max-w-xs rounded-full ${className}`}
      style={{ backgroundImage: "var(--linearPrimarySecondary)" }}
    />
  );
}
