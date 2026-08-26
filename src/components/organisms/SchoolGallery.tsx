import { ImageOff } from "lucide-react";

/** Placeholder gallery section. The real dataset has no photo data yet
 * (a known gap, per SKILLS.md) — shown honestly instead of faked. */
export default function SchoolGallery() {
  return (
    <div className="flex flex-col items-center justify-center gap-3 rounded-2xl border border-dashed border-slate-300 bg-white py-16 text-center">
      <ImageOff className="h-8 w-8 text-slate-300" />
      <p className="text-sm text-slate-500">
        Photos are not yet available for this school.
      </p>
    </div>
  );
}
