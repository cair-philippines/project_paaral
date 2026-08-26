const COLORS = {
  primary: "bg-primary",
  secondary: "bg-secondary text-primary",
  accent: "bg-accent",
} as const;

interface StepNumberBadgeProps {
  number: number;
  color?: keyof typeof COLORS;
}

/** Large circular numbered badge — used in the 3-step guided process section. */
export default function StepNumberBadge({
  number,
  color = "primary",
}: StepNumberBadgeProps) {
  return (
    <div
      className={`flex h-16 w-16 shrink-0 items-center justify-center rounded-full text-2xl font-bold text-white ${COLORS[color]}`}
    >
      {number}
    </div>
  );
}
