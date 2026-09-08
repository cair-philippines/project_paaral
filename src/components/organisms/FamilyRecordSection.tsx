import { Users } from "lucide-react";
import AccountSection from "@/components/molecules/AccountSection";

/** "Family Contacts" — adapted from the SchoolPath reference's guardian-
 * contact section, per direct request (2026-09-08). Unlike
 * `StudentRecordSection`, this isn't a record with some fields missing —
 * PAARAL's data model has no guardian/parent concept at all yet (no
 * table, no columns, a genuine schema gap, not a plumbing one). Shown as
 * a single honest empty state rather than fake-populated contact cards,
 * since there's no real "how many guardians, what fields" shape to guess
 * at yet — that's a schema decision for whoever designs it. */
export default function FamilyRecordSection() {
  return (
    <AccountSection
      number="02"
      eyebrow="Family Contacts"
      title="Who we contact about your child's application"
    >
      <div className="rounded-2xl border border-dashed border-slate-300 bg-white p-8 text-center">
        <Users className="mx-auto h-6 w-6 text-slate-400" />
        <p className="mt-3 font-bold text-primary">
          No guardian contacts on file yet.
        </p>
        <p className="mx-auto mt-1 max-w-sm text-sm text-slate-500">
          Once available, you&apos;ll be able to see and manage who we
          contact about your child&apos;s ESC application here.
        </p>
      </div>
    </AccountSection>
  );
}
