import { ChevronLeft, ChevronRight } from "lucide-react";

interface BrowsePaginationProps {
  page: number;
  totalPages: number;
  onChange: (page: number) => void;
}

/** Prev/Next pager for the browse results — plain "Page X of Y" rather than
 * numbered page buttons, since the real dataset runs to ~25 pages at the
 * default page size and a numbered strip that long would need its own
 * ellipsis/windowing logic for no real benefit to this audience: nobody
 * is jumping straight to page 19 of a school directory, they're paging
 * forward from wherever their filters left them. Native `disabled` at the
 * ends is the right call here (unlike the wishlist-limit button elsewhere
 * in this app) since there's nothing to explain — the boundary is obvious
 * from the "Page 1 of 25" label right next to it. */
export default function BrowsePagination({
  page,
  totalPages,
  onChange,
}: BrowsePaginationProps) {
  if (totalPages <= 1) return null;

  return (
    <nav
      aria-label="School results pages"
      className="mt-6 flex items-center justify-center gap-4"
    >
      <button
        type="button"
        onClick={() => onChange(page - 1)}
        disabled={page <= 1}
        className="flex h-11 items-center gap-1.5 rounded-xl border border-slate-200 px-4 text-sm font-bold text-primary transition hover:border-slate-300 disabled:cursor-not-allowed disabled:border-slate-100 disabled:text-slate-300"
      >
        <ChevronLeft className="h-4 w-4" /> Previous
      </button>

      <p className="text-sm font-semibold text-slate-600">
        Page <span className="font-bold text-primary">{page}</span> of{" "}
        {totalPages}
      </p>

      <button
        type="button"
        onClick={() => onChange(page + 1)}
        disabled={page >= totalPages}
        className="flex h-11 items-center gap-1.5 rounded-xl border border-slate-200 px-4 text-sm font-bold text-primary transition hover:border-slate-300 disabled:cursor-not-allowed disabled:border-slate-100 disabled:text-slate-300"
      >
        Next <ChevronRight className="h-4 w-4" />
      </button>
    </nav>
  );
}
