/**
 * Compact floating note over the top of the immersive `/browse` map (added
 * 2026-09-07, adapted from the "schoolpath-portal" reference's top-of-page
 * "Planning note" card). `/browse` is a deliberately full-screen, no-scroll
 * surface (2026-08-24 decision) — this stays a small overlay rather than
 * the reference's full scrolling hero band, so the map/list/card view
 * underneath is never pushed down or interrupted.
 *
 * Right-anchored (not centered) so it never sits under the collapsible
 * filter panel on the left at narrower widths.
 */
export default function BrowseIntroNote() {
  return (
    <div className="pointer-events-none absolute right-4 top-16 z-10 hidden max-w-xs sm:block">
      <div className="pointer-events-auto rounded-2xl border border-black/5 bg-white/95 px-4 py-3 text-right shadow-md backdrop-blur">
        <p className="text-sm font-semibold text-primary">
          Browse without signing in.
        </p>
        <p className="mt-1 text-xs leading-5 text-slate-600">
          Sign in only when you&apos;re ready to save schools and apply for
          ESC support.
        </p>
      </div>
    </div>
  );
}
