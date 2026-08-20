import { IdCard, ClipboardCheck, Search, ListChecks } from "lucide-react";
import PrepStepCard from "@/components/molecules/PrepStepCard";

const STEPS = [
  {
    icon: IdCard,
    title: "Verify your LRN",
    description: "Confirm your identity against the DepEd LIS registry.",
    color: "blue" as const,
  },
  {
    icon: ClipboardCheck,
    title: "Check your ESC eligibility",
    description: "A short self-assessment places you in a category (A–D).",
    color: "gold" as const,
  },
  {
    icon: Search,
    title: "Browse & discover schools",
    description: "Search by location, sector, and ESC participation.",
    color: "navy" as const,
  },
  {
    icon: ListChecks,
    title: "Build your wishlist and apply",
    description: "Rank as many schools as you want, then apply.",
    color: "blue" as const,
  },
];

export default function BeforeYouStartSection() {
  return (
    <section className="bg-slate-50 px-6 py-14 md:px-12">
      <h2 className="mb-8 text-center text-2xl font-bold text-navy">
        Before You Start
      </h2>
      <div className="mx-auto grid max-w-5xl gap-6 md:grid-cols-4">
        {STEPS.map((step) => (
          <PrepStepCard key={step.title} {...step} />
        ))}
      </div>
    </section>
  );
}
