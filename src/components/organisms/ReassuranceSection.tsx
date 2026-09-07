import ScrollReveal from "@/components/atoms/ScrollReveal";

export default function ReassuranceSection() {
  return (
    <section className="mx-auto max-w-3xl px-6 py-12 text-center md:px-12">
      <ScrollReveal>
        <p className="text-slate-600">
          There&apos;s no set order between the two tracks — a family may apply
          for the ESC certificate first, apply to a school first, or pursue both
          at once. A decision on one track never affects the other; you finalize
          subsidized enrollment by presenting your certificate once a school has
          admitted you.
        </p>
        <div className="mt-6 rounded-xl border-2 border-dashed border-secondary/60 bg-white p-5 text-primary">
          <p className="font-semibold">
            You can rank as many schools as you&apos;d like.
          </p>
          <p className="mt-1 text-sm text-slate-500">
            Up to 3 are pursued for an ESC subsidy at once — if one turns you
            down, the next on your list steps in, so a single rejection never
            ends your chances.
          </p>
        </div>
      </ScrollReveal>
    </section>
  );
}
