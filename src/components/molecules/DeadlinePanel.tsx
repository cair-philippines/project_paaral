import { CalendarClock } from "lucide-react";
import { ESC_APPLICATION_DEADLINE_LABEL } from "@/lib/constants";

/** The hero's "apply before [date]" panel — placeholder date, see lib/constants.ts. */
export default function DeadlinePanel() {
  return (
    <div className="flex items-center gap-3 rounded-2xl border border-white/20 bg-white/10 px-5 py-4">
      <CalendarClock className="text-ph-gold" size={28} />
      <div>
        <p className="text-xs uppercase tracking-widest text-white/70">
          Apply before
        </p>
        <p className="text-lg font-bold text-white">
          {ESC_APPLICATION_DEADLINE_LABEL}
        </p>
      </div>
    </div>
  );
}
