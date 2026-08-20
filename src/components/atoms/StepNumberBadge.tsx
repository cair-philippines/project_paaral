const COLORS = {
  blue: "bg-ph-blue",
  gold: "bg-ph-gold text-navy",
  red: "bg-ph-red",
} as const;

interface StepNumberBadgeProps {
  number: number;
  color?: keyof typeof COLORS;
}

/** Large circular numbered badge — used in the 3-step guided process section. */
export default function StepNumberBadge({
  number,
  color = "blue",
}: StepNumberBadgeProps) {
  return (
    <div
      className={`flex h-16 w-16 shrink-0 items-center justify-center rounded-full text-2xl font-bold text-white ${COLORS[color]}`}
    >
      {number}
    </div>
  );
}
