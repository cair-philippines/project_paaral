import { ESC_APPLICATION_DEADLINE_LABEL } from "@/lib/constants";

/**
 * The hero's "apply before [date]" pill — placeholder date, see
 * lib/constants.ts. Restyled 2026-09-07 to match the confirmed "Blue
 * Ballot Hero" comp exactly: a single-line white pill with a red dot,
 * replacing the earlier vertical calendar-icon card.
 */
export default function DeadlinePanel() {
  return (
    <div className="flex w-fit items-center gap-2 rounded-full bg-white px-4 py-2 shadow-[0_4px_14px_rgba(0,0,0,0.16)]">
      <span className="h-2 w-2 flex-shrink-0 rounded-full bg-accent" />
      <span className="text-xs text-[#5b3a13]">
        Apply before <strong className="font-extrabold text-accent">{ESC_APPLICATION_DEADLINE_LABEL}</strong>
      </span>
    </div>
  );
}
