import Button from "@mui/material/Button";
import TricolorRule from "@/components/atoms/TricolorRule";
import DeadlinePanel from "@/components/molecules/DeadlinePanel";

export default function HeroSection() {
  return (
    <section className="bg-navy px-6 py-16 text-center md:px-12 md:py-24">
      <TricolorRule className="mx-auto mb-8" />
      <h1 className="text-4xl font-bold tracking-tight text-white md:text-6xl">
        Find your ESC school.
        <br />
        Apply with confidence.
      </h1>
      <p className="mx-auto mt-4 max-w-xl text-white/80">
        Check your Educational Service Contracting (ESC) eligibility and
        browse participating schools — no fixed order required. Apply for
        ESC and school admission independently, at your own pace.
      </p>
      <div className="mt-8 flex flex-col items-center gap-6">
        <Button
          variant="contained"
          size="large"
          sx={{ bgcolor: "var(--color-ph-gold)", color: "var(--color-navy)" }}
        >
          Browse Schools
        </Button>
        <DeadlinePanel />
      </div>
    </section>
  );
}
