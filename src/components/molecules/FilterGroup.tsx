import type { ReactNode } from "react";

interface FilterGroupProps {
  label: string;
  children: ReactNode;
}

/** Labeled wrapper around one filter control in the browse sidebar. */
export default function FilterGroup({ label, children }: FilterGroupProps) {
  return (
    <div className="flex flex-col gap-2">
      <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500">
        {label}
      </p>
      {children}
    </div>
  );
}
