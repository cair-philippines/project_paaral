import type { LucideIcon } from "lucide-react";

const BADGE_COLORS = {
  primary: "bg-primary",
  secondary: "bg-secondary text-primary",
  accent: "bg-accent",
} as const;

interface PrepStepCardProps {
  icon: LucideIcon;
  title: string;
  description: string;
  color?: keyof typeof BADGE_COLORS;
}

/** One card in the "Before You Start" row — icon badge + short instruction. */
export default function PrepStepCard({
  icon: Icon,
  title,
  description,
  color = "primary",
}: PrepStepCardProps) {
  return (
    <div className="flex flex-col items-center gap-3 rounded-2xl bg-white p-6 text-center shadow-sm">
      <div
        className={`flex h-14 w-14 items-center justify-center rounded-full text-white ${BADGE_COLORS[color]}`}
      >
        <Icon size={26} />
      </div>
      <p className="font-semibold text-primary">{title}</p>
      <p className="text-sm text-slate-500">{description}</p>
    </div>
  );
}
