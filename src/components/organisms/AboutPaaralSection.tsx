import ScrollReveal from "@/components/atoms/ScrollReveal";

/** Second paragraph reuses the already-approved decoupled-model language
 * from ReassuranceSection/GuidedProcessSection rather than restating it
 * differently. Linked from HeroSection's "Know More About PAARAL" button
 * via #about-paaral. */
export default function AboutPaaralSection() {
  return (
    <section id="about-paaral" className="bg-slate-50 px-6 py-14 md:px-12">
      <ScrollReveal>
        <h2 className="mb-6 text-center text-2xl font-bold text-primary">
          About PAARAL
        </h2>
        <div className="mx-auto max-w-2xl space-y-4 text-center text-slate-600">
          <p>
            PAARAL is a platform built for the Educational Service Contracting
            (ESC) program of the Department of Education. It helps Grade 6
            learners find ESC-participating schools and apply for a subsidy,
            and shows where school slots remain insufficient.
          </p>
          <p>
            ESC eligibility and school admission are two independent tracks —
            apply for one first, the other first, or both at once. Neither is
            gated on the other, and you finalize enrollment by presenting your
            certificate at whichever school admits you.
          </p>
        </div>
      </ScrollReveal>
    </section>
  );
}
