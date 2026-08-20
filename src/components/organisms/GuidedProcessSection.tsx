import GuidedStep from "@/components/molecules/GuidedStep";

const STEPS = [
  {
    title: "Learn before you apply",
    description:
      "Use school search to research participating schools that could be a good fit — location, sector, ESC participation, and available ESC slots.",
    color: "blue" as const,
  },
  {
    title: "Rank your preferences",
    description:
      "Add schools to your wishlist in the order you'd prefer them — there's no limit on how many you can add.",
    color: "gold" as const,
  },
  {
    title: "Understand your outcomes",
    description:
      "ESC eligibility and school admission are decided independently. A denial on one track never closes the other — you finalize enrollment by redeeming your certificate at whichever school admits you.",
    color: "red" as const,
  },
];

export default function GuidedProcessSection() {
  return (
    <section className="mx-auto max-w-3xl px-6 py-14 md:px-12">
      <h2 className="mb-8 text-center text-2xl font-bold text-navy">
        How It Works
      </h2>
      <div className="flex flex-col gap-6">
        {STEPS.map((step, i) => (
          <GuidedStep key={step.title} number={i + 1} {...step} />
        ))}
      </div>
    </section>
  );
}
