import type { ReactNode } from "react";

interface FactTileProps {
  label: string;
  value: ReactNode;
}

/** One label/value pair in a fact-sheet grid. */
export default function FactTile({ label, value }: FactTileProps) {
  return (
    <div className="rounded-xl bg-white p-4 shadow-sm">
      <p className="text-[10px] font-bold uppercase tracking-widest text-slate-400">
        {label}
      </p>
      <p className="mt-1 text-base font-semibold text-primary">{value}</p>
    </div>
  );
}
