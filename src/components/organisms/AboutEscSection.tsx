import { catMeta } from "@/lib/eligibility";
import ScrollReveal from "@/components/atoms/ScrollReveal";

const CATEGORIES = ["A", "B", "C", "D"] as const;

/** Content pulled from already-established facts elsewhere in this app —
 * `catMeta` (src/lib/eligibility.ts) and CLAUDE.md's legal basis — not
 * invented for this section. Linked from HeroSection's "Know More About
 * ESC" button via #about-esc. */
export default function AboutEscSection() {
  return (
    <section id="about-esc" className="bg-white px-6 py-14 md:px-12">
      <ScrollReveal>
        <h2 className="mb-6 text-center text-2xl font-bold text-primary">
          About ESC
        </h2>
        <p className="mx-auto max-w-2xl text-center text-slate-600">
          Educational Service Contracting (ESC) is a Department of Education
          program that helps qualified Grade 6 graduates attend a private
          school through a government subsidy, when public schools don&apos;t
          have room for them. It&apos;s established under Republic Act No.
          8545 (the E-GASTPE Act) and DepEd&apos;s 2026 Revised Guidelines.
        </p>
      </ScrollReveal>

      <div className="mx-auto mt-8 grid max-w-5xl gap-4 md:grid-cols-2">
        {CATEGORIES.map((cat, i) => (
          <ScrollReveal key={cat} delay={i * 0.06}>
            <div
              className={`rounded-2xl border p-5 transition-all duration-200 hover:-translate-y-0.5 hover:shadow-md ${catMeta[cat].tw}`}
            >
              <p className="text-sm font-bold">{catMeta[cat].label}</p>
              <p className="mt-1 text-sm opacity-80">{catMeta[cat].desc}</p>
            </div>
          </ScrollReveal>
        ))}
      </div>

      <p className="mx-auto mt-8 max-w-2xl text-center text-sm text-slate-500">
        This is a self-assessment — the ESC School Committee makes the final
        decision once you apply.
      </p>
    </section>
  );
}
