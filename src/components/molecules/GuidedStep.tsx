import StepNumberBadge from "@/components/atoms/StepNumberBadge";

interface GuidedStepProps {
  number: number;
  title: string;
  description: string;
  color?: "blue" | "gold" | "red";
}

/** One entry in the numbered 3-step guided-process section, dashed callout box. */
export default function GuidedStep({
  number,
  title,
  description,
  color = "blue",
}: GuidedStepProps) {
  return (
    <div className="flex items-start gap-5">
      <StepNumberBadge number={number} color={color} />
      <div className="flex-1 rounded-xl border-2 border-dashed border-ph-gold/60 p-4">
        <p className="font-semibold text-navy">{title}</p>
        <p className="mt-1 text-sm text-slate-600">{description}</p>
      </div>
    </div>
  );
}
