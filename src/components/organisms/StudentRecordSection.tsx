import type { ReactNode } from "react";
import { CalendarDays, GraduationCap, MapPin, UserRound } from "lucide-react";
import AccountSection from "@/components/molecules/AccountSection";
import type { Account } from "@/types/application";

interface DataTileProps {
  icon: ReactNode;
  label: string;
  value: string;
  note: string;
  known: boolean;
}

function DataTile({ icon, label, value, note, known }: DataTileProps) {
  return (
    <div className="rounded-2xl border border-slate-100 bg-white p-4 shadow-sm">
      <div className="flex items-center gap-2 text-sm font-bold text-primary">
        {icon} {label}
      </div>
      <p
        className={
          known
            ? "mt-3 text-lg font-bold text-primary"
            : "mt-3 text-base font-semibold italic text-slate-400"
        }
      >
        {value}
      </p>
      <p className="mt-1 text-xs leading-5 text-slate-500">{note}</p>
    </div>
  );
}

/** "Official Student Record" — adapted from the SchoolPath reference's own
 * first account section (`docs/schoolpath-portal/client/src/pages/
 * Account.tsx`), per direct request (2026-09-08). Only "Student Name" is
 * a real, already-known field; date of birth, current school, and home
 * area have no backing data anywhere in the app yet, shown as honest
 * "Not yet collected" placeholders rather than invented values — matching
 * the project's "real data or honest absence" rule elsewhere (e.g. a
 * school's missing photos/quality indicators).
 *
 * Worth flagging for whoever wires these up: "Current School" is NOT a
 * schema gap — `Learner.grade6_school_name` already exists in
 * `paaral-student-api` — it's a plumbing gap (that field isn't part of
 * the frontend's `Account` type or the login-hydration response yet).
 * Date of birth and a home-area/address field are genuine schema gaps —
 * no such columns exist on `Learner` at all. */
export default function StudentRecordSection({
  account,
}: {
  account: Account;
}) {
  return (
    <AccountSection
      number="01"
      eyebrow="Official Student Record"
      title="The details we use for this transition"
    >
      <div className="grid gap-3 sm:grid-cols-2">
        <DataTile
          icon={<UserRound className="h-4 w-4" />}
          label="Student Name"
          value={account.name}
          note="From your DepEd record."
          known
        />
        <DataTile
          icon={<CalendarDays className="h-4 w-4" />}
          label="Date of Birth"
          value="Not yet collected"
          note="Will come from the DepEd Learner Information System once connected."
          known={false}
        />
        <DataTile
          icon={<GraduationCap className="h-4 w-4" />}
          label="Current School"
          value="Not yet collected"
          note="Your Grade 6 school will show here once connected to your account."
          known={false}
        />
        <DataTile
          icon={<MapPin className="h-4 w-4" />}
          label="Home Area"
          value="Not yet collected"
          note="Used to show nearby schools once we have your address on file."
          known={false}
        />
      </div>
    </AccountSection>
  );
}
