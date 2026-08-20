import Image from "next/image";
import Button from "@mui/material/Button";

/** Minimal top header — logo lockup + a single primary CTA. Nav links are
 * deliberately not fabricated yet; add them once real destination pages exist. */
export default function SiteHeader() {
  return (
    <header className="flex items-center justify-between bg-navy px-6 py-4 md:px-12">
      <div className="flex items-center gap-4">
        <Image
          src="/assets/deped-logo.png"
          alt="DepEd"
          width={32}
          height={32}
          className="h-8 w-auto"
        />
        <Image
          src="/assets/ecair-logo.png"
          alt="ECAIR"
          width={20}
          height={20}
          className="h-5 w-auto"
        />
        <span className="text-xl font-bold tracking-tight text-white">
          PAARAL
        </span>
      </div>
      <Button
        variant="contained"
        sx={{ bgcolor: "var(--color-ph-gold)", color: "var(--color-navy)" }}
      >
        Browse Schools
      </Button>
    </header>
  );
}
