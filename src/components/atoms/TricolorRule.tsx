/**
 * Thin three-segment divider line in Philippine flag colors — the PAARAL
 * equivalent of the Chile SAE benchmark's tricolor section dividers.
 */
export default function TricolorRule({ className = "" }: { className?: string }) {
  return (
    <div className={`flex h-[3px] w-full max-w-xs gap-1 ${className}`}>
      <span className="flex-1 rounded-full bg-ph-blue" />
      <span className="flex-1 rounded-full bg-ph-gold" />
      <span className="flex-1 rounded-full bg-ph-red" />
    </div>
  );
}
