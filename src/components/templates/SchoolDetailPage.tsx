import Link from "next/link";
import { ChevronLeft } from "lucide-react";
import SiteHeader from "@/components/organisms/SiteHeader";
import SchoolFactSheet from "@/components/organisms/SchoolFactSheet";
import SchoolGallery from "@/components/organisms/SchoolGallery";
import AddToWishlistButton from "@/components/molecules/AddToWishlistButton";
import { getTypeBadge, titleCase } from "@/lib/schools";
import type { School } from "@/types/school";

interface SchoolDetailPageProps {
  school: School;
}

export default function SchoolDetailPage({ school }: SchoolDetailPageProps) {
  const badge = getTypeBadge(school);

  return (
    <div className="flex flex-1 flex-col">
      <SiteHeader />
      <div className="mx-auto flex w-full max-w-4xl flex-1 flex-col gap-6 bg-background p-6">
        <Link
          href="/browse"
          className="flex items-center gap-1 text-sm font-semibold text-primary hover:underline"
        >
          <ChevronLeft size={16} />
          Back to Browse
        </Link>

        <div>
          <h1 className="text-2xl font-bold text-primary">
            {school.school_name}
          </h1>
          <p className="mt-1 text-sm text-slate-500">
            {titleCase(school.barangay)},{" "}
            {titleCase(school.municipality)}
          </p>
          <span
            className={`mt-3 inline-block rounded-full px-2 py-0.5 text-[10px] font-bold uppercase ${badge.className}`}
          >
            {badge.label}
          </span>
          <div className="mt-4 max-w-xs">
            <AddToWishlistButton school={school} />
          </div>
        </div>

        <SchoolGallery />
        <SchoolFactSheet school={school} />
      </div>
    </div>
  );
}
