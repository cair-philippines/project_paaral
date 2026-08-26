import Link from "next/link";
import type { School } from "@/types/school";
import AddToWishlistButton from "@/components/molecules/AddToWishlistButton";
import {
  getSlotAvailability,
  getTypeBadge,
  netFeesLabel,
  pesos,
  titleCase,
} from "@/lib/schools";

const SLOT_TONE_CLASS: Record<string, string> = {
  green: "bg-[#16a34a]",
  amber: "bg-[#f59e0b]",
  red: "bg-[#dc2626]",
  unknown: "bg-slate-200",
};

interface SchoolResultCardProps {
  school: School;
  selected?: boolean;
  variant?: "card" | "list";
  onSelect: (school: School) => void;
}

/** One school in the browse results — shared between the card grid and
 * the list view so both render from the same filtered data. A plain
 * flex-wrap row (not a CSS grid with equal-fraction columns) keeps the
 * facts from ever stretching past the card's own width.
 *
 * Structural template (2026-08-24, Step 2 of the recolor pass) —
 * `rounded-2xl` to match the app's other elevated cards (was `rounded-xl`,
 * a small inconsistency); a light border plus `shadow-sm` at rest, since
 * shadow alone can be too subtle to read as a card boundary for a
 * low-vision or low-quality-screen user, and border alone reads flatter —
 * together they're the more legible choice, at the cost of a very slightly
 * busier look; and an accent-colored "View School Details" button — accent
 * is reserved for the single most decisive action on a light-background
 * surface, distinct from the plain `primary` used for ordinary links/
 * interactive elements throughout the rest of the app. */
export default function SchoolResultCard({
  school,
  selected = false,
  variant = "card",
  onSelect,
}: SchoolResultCardProps) {
  const badge = getTypeBadge(school);
  const slots = getSlotAvailability(school);

  const facts = [
    { label: "Subsidy Amount", value: pesos(school.esc_subsidy_amount) },
    { label: "Total Fees", value: pesos(school.esc_total_fees) },
    { label: "Net Fees", value: netFeesLabel(school.esc_net_fees) },
    {
      label: "Slots Available",
      value: slots.available === null ? "Not available" : slots.available,
    },
  ];

  return (
    <div
      onClick={() => onSelect(school)}
      className={[
        "flex w-full cursor-pointer flex-col gap-3 rounded-2xl border bg-white p-4 shadow-sm transition hover:-translate-y-0.5 hover:shadow-md",
        selected ? "border-primary ring-4 ring-primary/10" : "border-slate-100",
      ].join(" ")}
    >
      <div className="flex min-w-0 items-start justify-between gap-2">
        <div className="min-w-0">
          <h3 className="line-clamp-2 text-sm font-semibold leading-snug text-primary">
            {school.school_name}
          </h3>
          <p className="mt-1 truncate text-xs text-slate-500">
            {titleCase(school.deped_barangay)}, Quezon City
          </p>
          <span
            className={`mt-2 inline-block rounded-full px-2 py-0.5 text-[10px] font-bold uppercase ${badge.className}`}
          >
            {badge.label}
          </span>
        </div>
        <AddToWishlistButton school={school} variant="compact" />
      </div>

      <div className="flex flex-wrap gap-x-6 gap-y-2">
        {facts.map((fact) => (
          <div key={fact.label} className="min-w-[110px]">
            <p className="text-[11px] uppercase tracking-[0.08em] text-slate-400">
              {fact.label}
            </p>
            <p className="mt-1 text-sm font-semibold text-primary">
              {fact.value}
            </p>
          </div>
        ))}
      </div>

      {slots.total !== null && slots.available !== null && variant === "card" && (
        <div className="h-1.5 overflow-hidden rounded-full bg-slate-100">
          <div
            className={`h-full rounded-full ${SLOT_TONE_CLASS[slots.tone]}`}
            style={{
              width: `${Math.min((slots.available / slots.total) * 100, 100)}%`,
            }}
          />
        </div>
      )}

      <Link
        href={`/schools/${school.school_id}`}
        onClick={(e) => e.stopPropagation()}
        className="self-start rounded-lg bg-accent px-3 py-1.5 text-center text-xs font-semibold text-white hover:opacity-90"
      >
        View School Details
      </Link>
    </div>
  );
}
