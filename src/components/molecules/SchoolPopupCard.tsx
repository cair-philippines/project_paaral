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

interface SchoolPopupCardProps {
  school: School;
}

/** Content shown in the map popup when a school marker is clicked. */
export default function SchoolPopupCard({ school }: SchoolPopupCardProps) {
  const slots = getSlotAvailability(school);
  const badge = getTypeBadge(school);

  return (
    <div className="flex w-64 flex-col gap-2 p-1">
      <div className="flex items-start justify-between gap-2">
        <h3 className="text-sm font-semibold leading-snug text-primary">
          {school.school_name}
        </h3>
        <AddToWishlistButton school={school} variant="compact" />
      </div>
      <p className="text-xs text-slate-500">
        {titleCase(school.barangay)}, {titleCase(school.municipality)}
      </p>
      <span
        className={`inline-block w-fit rounded-full px-2 py-0.5 text-[10px] font-bold uppercase ${badge.className}`}
      >
        {badge.label}
      </span>

      <div className="mt-1 grid grid-cols-2 gap-2 text-xs">
        <div>
          <p className="text-slate-400">Subsidy Amount</p>
          <p className="font-semibold text-primary">
            {pesos(school.esc_subsidy_amount)}
          </p>
        </div>
        <div>
          <p className="text-slate-400">Total Fees</p>
          <p className="font-semibold text-primary">
            {pesos(school.esc_total_fees)}
          </p>
        </div>
        <div>
          <p className="text-slate-400">Net Fees</p>
          <p className="font-semibold text-primary">
            {netFeesLabel(school.esc_net_fees)}
          </p>
        </div>
        <div>
          <p className="text-slate-400">Subsidy Slots</p>
          <p className="font-semibold text-primary">
            {slots.total === null ? "Not available" : slots.total}
          </p>
        </div>
      </div>

      <Link
        href={`/schools/${school.school_id}`}
        className="mt-2 rounded-lg bg-primary px-3 py-1.5 text-center text-xs font-semibold text-white hover:opacity-90"
      >
        View more info
      </Link>
    </div>
  );
}
