import { notFound } from "next/navigation";
import SchoolDetailPage from "@/components/templates/SchoolDetailPage";
import { getQcSchools, getSchoolById } from "@/lib/schools";

export function generateStaticParams() {
  return getQcSchools().map((school) => ({ school_id: school.school_id }));
}

interface SchoolPageProps {
  params: Promise<{ school_id: string }>;
}

export default async function SchoolPage({ params }: SchoolPageProps) {
  const { school_id } = await params;
  const school = getSchoolById(school_id);

  if (!school) {
    notFound();
  }

  return <SchoolDetailPage school={school} />;
}
