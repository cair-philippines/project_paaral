interface JourneyStepProps {
  number: string;
  title: string;
  detail: string;
  active?: boolean;
}

/** One item in the account page's 3-step "your journey so far" strip. The
 * active step gets a colored top border + accent-colored number, the
 * SchoolPath reference's "currently here" treatment — adapted to this
 * app's own navy/accent palette instead of copying its teal. */
export default function JourneyStep({
  number,
  title,
  detail,
  active = false,
}: JourneyStepProps) {
  return (
    <div
      className={`flex items-center gap-3 border-t-4 px-1 py-5 md:px-4 ${
        active ? "border-accent" : "border-transparent"
      }`}
    >
      <span
        className={`text-sm font-bold ${active ? "text-accent" : "text-slate-400"}`}
      >
        {number}
      </span>
      <div className="min-w-0">
        <p
          className={`text-sm font-bold ${active ? "text-primary" : "text-slate-600"}`}
        >
          {title}
        </p>
        <p className="text-xs text-slate-500">{detail}</p>
      </div>
    </div>
  );
}
