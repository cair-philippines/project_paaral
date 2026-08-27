import StepNumberBadge from "@/components/atoms/StepNumberBadge";

interface GuidedStepProps {
  number: number;
  title: string;
  description: string;
  color?: "primary" | "secondary" | "accent";
}

/** One entry in the numbered 3-step guided-process section, dashed callout box. */
export default function GuidedStep({
  number,
  title,
  description,
  color = "primary",
}: GuidedStepProps) {
  return (
    <div className="flex items-start gap-5">
      <StepNumberBadge number={number} color={color} />
      <div className="flex-1 rounded-xl border-2 border-dashed border-secondary/60 p-4 transition-all duration-200 hover:-translate-y-0.5 hover:border-primary/60 hover:shadow-md">
        <p className="font-semibold text-primary">{title}</p>
        <p className="mt-1 text-sm text-slate-600">{description}</p>
      </div>
    </div>
  );
}
